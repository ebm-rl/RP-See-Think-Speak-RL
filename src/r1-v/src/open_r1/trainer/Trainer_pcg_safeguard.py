# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import random

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

from qwen_vl_utils import process_vision_info

import copy
import re
import numpy as np

from contextlib import nullcontext

ANSWER_PATTERN = r"<answer>\s*(.*?)\s*</answer>"
VT_COMBINED_RE = re.compile(r"(<vision>.*?</vision>)[^<]*(<think>.*?</think>)", re.DOTALL)

# Use a persistent subprocess to run sentence-transformers encode,
# completely isolated from DeepSpeed ZeRO-3 hooks.

import sys
import struct
import pickle
import subprocess
import tempfile

_SENT_SPLIT_RE = re.compile(
    r'(?<=[.!?])\s+'
    r'|(?<=[.!?])(?=[A-Z])'
    r'|\n+'
)
_ABBREV_RE = re.compile(r'\b(?:Mr|Mrs|Ms|Dr|Prof|St|Jr|Sr|vs|etc|e\.g|i\.e)\.\s', re.IGNORECASE)


def _split_sentences(text: str) -> list:
    if not text or not text.strip():
        return []
    placeholder = "ABBREV_PLACEHOLDER"
    protected = _ABBREV_RE.sub(lambda m: m.group().replace(". ", f".{placeholder}"), text)
    parts = _SENT_SPLIT_RE.split(protected)
    sentences = [s.replace(f".{placeholder}", ". ").strip() for s in parts if s and s.strip()]
    return sentences


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

_PERSISTENT_WORKER = None

def _ensure_persistent_worker():
    """Start a persistent background subprocess for encoding."""
    global _PERSISTENT_WORKER
    if _PERSISTENT_WORKER is not None and _PERSISTENT_WORKER.poll() is None:
        return _PERSISTENT_WORKER

    worker_code = r'''
import sys, pickle, struct
def main():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    sys.stdout.buffer.write(b"READY\n")
    sys.stdout.buffer.flush()
    while True:
        header = sys.stdin.buffer.read(4)
        if len(header) < 4:
            break
        data_len = struct.unpack("!I", header)[0]
        if data_len == 0:
            break
        data = b""
        while len(data) < data_len:
            chunk = sys.stdin.buffer.read(data_len - len(data))
            if not chunk:
                break
            data += chunk
        texts = pickle.loads(data)
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        result_bytes = pickle.dumps(embeddings, protocol=pickle.HIGHEST_PROTOCOL)
        sys.stdout.buffer.write(struct.pack("!I", len(result_bytes)))
        sys.stdout.buffer.write(result_bytes)
        sys.stdout.buffer.flush()
if __name__ == "__main__":
    main()
'''
    worker_path = os.path.join(tempfile.gettempdir(), "_pcg_persistent_worker.py")
    with open(worker_path, "w") as f:
        f.write(worker_code)

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.Popen(
        [sys.executable, worker_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    ready_line = proc.stdout.readline()
    if b"READY" not in ready_line:
        stderr_out = proc.stderr.read().decode()[:500]
        raise RuntimeError(f"PCG worker failed to start: {stderr_out}")
    
    print("[PCG] Persistent encode worker started successfully.")
    _PERSISTENT_WORKER = proc
    return proc


def _encode_via_worker(texts: list) -> np.ndarray:
    """Encode texts via the persistent subprocess, fully isolated from DeepSpeed."""
    proc = _ensure_persistent_worker()
    data = pickle.dumps(texts, protocol=pickle.HIGHEST_PROTOCOL)
    proc.stdin.write(struct.pack("!I", len(data)))
    proc.stdin.write(data)
    proc.stdin.flush()

    header = proc.stdout.read(4)
    if len(header) < 4:
        stderr_out = proc.stderr.read().decode()[:500] if proc.stderr else "unknown"
        raise RuntimeError(f"PCG worker died: {stderr_out}")

    result_len = struct.unpack("!I", header)[0]
    result_data = b""
    while len(result_data) < result_len:
        chunk = proc.stdout.read(result_len - len(result_data))
        if not chunk:
            break
        result_data += chunk

    return pickle.loads(result_data)


def shutdown_pcg_worker():
    """Shutdown the persistent worker (call at end of training)."""
    global _PERSISTENT_WORKER
    if _PERSISTENT_WORKER is not None:
        try:
            _PERSISTENT_WORKER.stdin.write(struct.pack("!I", 0))
            _PERSISTENT_WORKER.stdin.flush()
            _PERSISTENT_WORKER.wait(timeout=5)
        except:
            _PERSISTENT_WORKER.kill()
        _PERSISTENT_WORKER = None

def filter_think_sentences(think_text: str, answer_text: str, tau: float = 0.55):
    sentences = _split_sentences(think_text)
    if not sentences:
        return think_text, 0, 0, 0.0

    all_texts = sentences + [answer_text]
    embeddings = _encode_via_worker(all_texts)

    answer_emb = embeddings[-1]
    sentence_embs = embeddings[:-1]

    kept, removed, max_sim = [], 0, 0.0
    for sent, emb in zip(sentences, sentence_embs):
        sim = _cosine_similarity(emb, answer_emb)
        max_sim = max(max_sim, sim)
        if sim > tau:
            removed += 1
        else:
            kept.append(sent)

    filtered_text = " ".join(kept) if kept else ""
    return filtered_text, removed, len(sentences), max_sim


def filter_vt_block(vt_block: str, answer_text: str, tau: float = 0.55):
    vision_match = re.search(r"(<vision>.*?</vision>)", vt_block, re.DOTALL)
    think_match = re.search(r"<think>(.*?)</think>", vt_block, re.DOTALL)

    if not think_match:
        return vt_block, 0, 0, 0.0

    think_content = think_match.group(1).strip()
    filtered_think, num_removed, num_total, max_sim = filter_think_sentences(
        think_content, answer_text, tau
    )

    vision_part = vision_match.group(1) if vision_match else ""
    if filtered_think:
        result = vision_part + "\n<think>\n" + filtered_think + "\n</think>"
    else:
        result = vision_part

    return result, num_removed, num_total, max_sim

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb
    

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

from contextlib import contextmanager


class Qwen2VLNoPCGleakGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """
    
    PCG_SIMILARITY_TAU = 0.2
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        script_args = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
            
        final_pad_token_id = None 
        
        if processing_class is not None:
            if hasattr(processing_class, "pad_token_id"):
                final_pad_token_id = processing_class.pad_token_id
            elif hasattr(processing_class, "tokenizer") and processing_class.tokenizer is not None:
                final_pad_token_id = processing_class.tokenizer.pad_token_id

        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
                # model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                # self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id or True:
                processing_class = AutoProcessor.from_pretrained(model_id)
                if hasattr(processing_class, "pad_token_id"):
                    final_pad_token_id = processing_class.pad_token_id
                elif hasattr(processing_class.tokenizer, "pad_token_id"):
                    final_pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                # pad_token_id = processing_class.pad_token_id
                final_pad_token_id = processing_class.pad_token_id

        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features
        
        if final_pad_token_id is None and processing_class is not None:
             if hasattr(processing_class, "eos_token_id"):
                final_pad_token_id = processing_class.eos_token_id
        
        if final_pad_token_id is None:
            print("!!! WARNING: pad_token_id could not be found. Forcing to 0.")
            final_pad_token_id = 0 
            
        print(f"DEBUG: Final pad_token_id used for init: {final_pad_token_id}")

        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.temporal = script_args.temporal
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=self.num_generations,
            # pad_token_id=pad_token_id,
            pad_token_id=final_pad_token_id,
        )
        self.shuffled_num_generations = self.num_generations // 2
        self.shuffled_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=self.shuffled_num_generations,
            # pad_token_id=pad_token_id,
            pad_token_id=final_pad_token_id,
        )
        
        self.dummy_generation_config = GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=1,
            # pad_token_id=pad_token_id,
            pad_token_id=final_pad_token_id,
        )
        self.len_control = script_args.len_control
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    # def _log_comprehensive_metrics(self, inputs, completion_texts, gt_answer, rewards_tensor, problem_ids):
    def _log_comprehensive_metrics(self, inputs, completion_texts, gt_answer, rewards_tensor, problem_ids, pcg_filter_info=None):
        """
        Comprehensive record of Vision (CLIP), Think (PCG), Answer (BERT), and their corresponding textual content
        """
        import torch.distributed as dist
        import re
        import os
        from datetime import datetime
        
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
            
        log_dir = "./src/open_r1/debug_log/RP_NEW_w_1-1-0.8-0.8-no-PCG-LEAK"
        os.makedirs(log_dir, exist_ok=True)
        log_path = f"{log_dir}/comprehensive_log_rank_{rank}.txt"
        
        vision_pattern = re.compile(r"<vision>(.*?)</vision>", re.DOTALL)
        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        
        num_gens = self.num_generations
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*20} Batch Log at {current_time} {'='*20}\n")
            
            for i, content in enumerate(completion_texts):
                input_idx = i // num_gens
                
                if problem_ids and input_idx < len(problem_ids):
                    pid = problem_ids[input_idx]
                else:
                    pid = "Unknown"

                acc_score = rewards_tensor[i, 0].item()
                fmt_score = rewards_tensor[i, 1].item()
                clip_score = rewards_tensor[i, 2].item()
                pcg_score = rewards_tensor[i, 3].item()

                v_match = vision_pattern.search(content)
                t_match = think_pattern.search(content)
                a_match = answer_pattern.search(content)
                
                vision_text = v_match.group(1).strip() if v_match else "[No Vision]"
                think_text = t_match.group(1).strip() if t_match else "[No Think]"
                answer_text = a_match.group(1).strip() if a_match else "[No Answer]"

                f.write(f"Sample Global_Idx: {i} | Problem_ID: {pid}\n")
                f.write(f"Ref GT Answer: {gt_answer}\n")
                f.write(f"Predictions:\n")
                f.write(f"   [Vision] ({clip_score:.4f}): {vision_text}\n")
                f.write(f"   [Think]  ({pcg_score:.4f}): {think_text}\n")
                f.write(f"   [Answer] ({acc_score:.4f}): {answer_text}\n")
                f.write(f"   [Format] ({fmt_score:.1f})\n")
                
                # PCG filtering details for this completion
                if pcg_filter_info and i in pcg_filter_info:
                    fvt, f_removed, f_total, f_max_sim = pcg_filter_info[i]
                    f.write(f"   [PCG Filter] tau={self.PCG_SIMILARITY_TAU}, "
                            f"removed={f_removed}/{f_total} sents, "
                            f"max_sim={f_max_sim:.4f}\n")
                    if f_removed > 0:
                        # Also log what the original think looked like vs filtered
                        orig_vt = self._extract_vt_block_clean(content)
                        f.write(f"   [PCG Original Think]: {orig_vt}\n")
                        f.write(f"   [PCG Filtered Think]: {fvt}\n")
                f.write("-" * 50 + "\n")


    def _get_per_token_logps(self, model, input_ids, **kwargs):
        logits = model(input_ids, **kwargs).logits
        logits = logits[:, :-1, :] 
        input_ids = input_ids[:, 1:]  

        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    def remove_none_from_data(self, data):
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs
    
    def _extract_answer(self, text: str) -> str:
        m = re.search(ANSWER_PATTERN, text, re.DOTALL)
        return m.group(1).strip() if m else ""

    def _compute_log_likelihood(self, model, input_ids, attention_mask, labels, visual_inputs):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        kwargs.update(visual_inputs)
        
        with torch.no_grad():
            outputs = model(**kwargs)
            nll = outputs.loss 
        return -nll.item()

    def _extract_vt_block_clean(self, text: str) -> str:
        """
        PCG Context Extraction Strategy:
        1. Must match the structure: Vision block + [non-tag gap] + Think block
        2. Extract the Vision block and Think block separately
        3. Ignore any nonsense in the gap and concatenate these two blocks directly (inserting a line break between them)
        """
        match = VT_COMBINED_RE.search(text)
        
        if match:
            vision_part = match.group(1) # <vision>...content...</vision>
            think_part = match.group(2)  # <think>...content...</think>
            
            # Concatenate, filtering out any potentially present filler phrases like “Okay then...”
            return vision_part + "\n" + think_part
        
        return ""
    
    def _extract_vt_block_filtered(self, text: str, answer_text: str, tau: float = None):
        """
        Extract vision+think block, then filter think sentences that are 
        too similar to answer_text (sim > tau).
        
        Returns:
            (filtered_vt_block, num_removed, num_total, max_sim)
        """
        if tau is None:
            tau = self.PCG_SIMILARITY_TAU
            
        vt_block = self._extract_vt_block_clean(text)
        
        if not vt_block or not answer_text:
            return vt_block, 0, 0, 0.0
        
        return filter_vt_block(vt_block, answer_text, tau)    
    

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

                
        
        input_copy = copy.deepcopy(inputs[0]['prompt'])    
        input_copy = self.remove_none_from_data(input_copy)
        
        DATA_ROOT = "./my_dataset"
        
        current_video_path = "Unknown"
        if inputs[0]['data_type'] == 'image':
            input_copy[1]['content'][0]['image'] = inputs[0]['image_path']
        elif inputs[0]['data_type'] == 'video':
            raw_path = inputs[0]['video_path']
            
            if raw_path.startswith("./"):
                abs_path = DATA_ROOT + raw_path[1:]
            else:
                import os
                abs_path = os.path.join(DATA_ROOT, raw_path)
            current_video_path = abs_path
            input_copy[1]['content'][0]['video'] = abs_path
            
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
        
        except Exception as e:
            print(f"process_vision_info error, using fixed data, {e}")
            raise RuntimeError(f"Failed to process visual inputs for data type: {inputs[0].get('data_type')}. "
                               f"Check if the media files exist or if 'process_vision_info' is compatible.") from e
                
        prompt_inputs = self.processing_class(
            text=copy.deepcopy(prompts_text),
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        
        
        prompt_inputs = super()._prepare_inputs(prompt_inputs)


        # fix prompt_inputs["input_ids"] length issue
        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length :]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length :]

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]
            
            
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
           
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            

        # print('path:', input_copy[0]['content'][0][inputs[0]['data_type']])   
        print('path:', input_copy[1]['content'][0][inputs[0]['data_type']])   
        print('problem_id:', inputs[0]['problem_id'])       
        print('prompt_length:', prompt_length)
                
        
        
        if hasattr(self.processing_class, "tokenizer") and self.processing_class.tokenizer is not None:
            current_eos_token_id = self.processing_class.tokenizer.eos_token_id
        else:
            current_eos_token_id = self.processing_class.eos_token_id
            
        is_eos = completion_ids == current_eos_token_id      
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")
        
        if inputs[0]['data_type'] == 'video':
            prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)
            if 'second_per_grid_ts' in prompt_inputs: 
                del prompt_inputs["second_per_grid_ts"]

        try:
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
            per_token_logps = per_token_logps[:, prompt_length - 1 :]
        except Exception as e:
            print(f"Error computing per_token_logps: {e}. Setting output to zero.")
            # per_token_logps = torch.tensor(0.0, device=prompt_completion_ids.device, requires_grad=True)
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
        
        from contextlib import nullcontext
        
        with torch.no_grad():
            try:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, **prompt_inputs)
                else:
                    unwrapped_model = self.accelerator.unwrap_model(model)

                    if hasattr(unwrapped_model, "disable_adapter"):
                        adapter_ctx = unwrapped_model.disable_adapter()
                    elif hasattr(unwrapped_model, "disable_adapters"):
                        adapter_ctx = unwrapped_model.disable_adapters()
                    else:
                        adapter_ctx = nullcontext()
                    
                    with adapter_ctx:
                        ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
                
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
                
            except Exception as e:
                print(f"Error computing ref_per_token_logps: {e}. Setting output to zero.")
                ref_per_token_logps = torch.zeros_like(per_token_logps)
                
        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10) 
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1
        
    
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
           
        completion_texts = [
            c[0]["content"] if isinstance(c, list) else c
            for c in completions
        ]
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
                
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
            for key in reward_kwargs:
                for example in inputs:
                    # Repeat each value in the column for `num_generations` times
                    reward_kwargs[key].extend([example[key]] * self.num_generations)
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
            

        gt_raw = inputs[0].get(
            "solution",
            inputs[0].get("answer", inputs[0].get("reference", "")),
        )
        if isinstance(gt_raw, list):
            gt_raw = gt_raw[0]
        gt_answer = self._extract_answer(gt_raw)
        pcg_col = torch.zeros(rewards_per_func.size(0), 1, device=device)
        
        MAX_PCG_SEQ_LEN = 16384
        
        # PCG filtering metrics (initialized here, updated inside the loop)
        _pcg_total_sents = 0
        _pcg_removed_sents = 0
        _pcg_max_sims = []
        
        _precomputed_filtered_vt = {}
        if gt_answer:
            for i, content in enumerate(completion_texts):
                try:
                    filtered_vt_block, num_removed, num_total, max_sim = \
                        self._extract_vt_block_filtered(content, gt_answer)
                    _precomputed_filtered_vt[i] = (filtered_vt_block, num_removed, num_total, max_sim)
                    _pcg_total_sents += num_total
                    _pcg_removed_sents += num_removed
                    if num_total > 0:
                        _pcg_max_sims.append(max_sim)
                except Exception as e:
                    print(f"[PCG Filter] Error at idx {i}: {e}")
                    _precomputed_filtered_vt[i] = ("", 0, 0, 0.0)

        if gt_answer:
            try:
                raw_prompt_text = prompts_text[0]
                is_image = inputs[0]["data_type"] == "image" and image_inputs is not None
                is_video = inputs[0]["data_type"] == "video" and video_inputs is not None
                current_images = image_inputs if is_image else None
                current_videos = video_inputs if is_video else None

                target_model_to_unwrap = self.ref_model if self.ref_model is not None else model
                ctx = unwrap_model_for_generation(target_model_to_unwrap, self.accelerator)

                with ctx as teacher:
                    if hasattr(teacher, "disable_adapter"):
                        inner_ctx = teacher.disable_adapter()
                    else:
                        inner_ctx = nullcontext()

                    with inner_ctx: 
                        text_base = raw_prompt_text + gt_answer
                        inputs_base = self.processing_class(
                            text=[text_base],
                            images=current_images,
                            videos=current_videos,
                            return_tensors="pt",
                            padding=True,
                        ).to(device)

                        inputs_prefix_base = self.processing_class(
                            text=[raw_prompt_text],
                            images=current_images,
                            videos=current_videos,
                            return_tensors="pt",
                            padding=True,
                        )
                        prompt_len = inputs_prefix_base["input_ids"].shape[1]
                        del inputs_prefix_base  

                        labels_base = inputs_base["input_ids"].clone()
                        safe_len_base = min(prompt_len, labels_base.shape[1] - 1)
                        labels_base[:, :safe_len_base] = -100
                        
                        visual_kwargs_base = {
                            k: v for k, v in inputs_base.items()
                            if k not in ["input_ids", "attention_mask", "labels"]
                        }
                        
                        logp_base = self._compute_log_likelihood(
                            teacher,
                            inputs_base["input_ids"],
                            inputs_base["attention_mask"],
                            labels_base,
                            visual_kwargs_base,
                        )
                        del inputs_base, labels_base, visual_kwargs_base
   
                        # for i, content in enumerate(completion_texts):
                        #     vt_block = self._extract_vt_block_clean(content)
                            
                        #     if not vt_block:
                        #         pcg_col[i] = 0.0 
                        #         continue
                        
                        
                        # ---- PCG Filtering Metrics ----
                        # _pcg_total_sents = 0
                        # _pcg_removed_sents = 0
                        # _pcg_max_sims = []

                        for i, content in enumerate(completion_texts):
                            # # === MODIFIED: use filtered vt_block instead of raw ===
                            # filtered_vt_block, num_removed, num_total, max_sim = \
                            #     self._extract_vt_block_filtered(content, gt_answer)
                            
                            # _pcg_total_sents += num_total
                            # _pcg_removed_sents += num_removed
                            # if num_total > 0:
                            #     _pcg_max_sims.append(max_sim)
                            
                            # if not filtered_vt_block:
                            #     pcg_col[i] = 0.0
                            #     continue
                            # # === END MODIFIED ===
                            
                            # === Read pre-computed filtered results ===
                            filtered_vt_block, _, _, _ = _precomputed_filtered_vt.get(
                                i, ("", 0, 0, 0.0)
                            )
                            
                            if not filtered_vt_block:
                                pcg_col[i] = 0.0
                                continue
                            # === END ===

                            try:
                                # text_aug_full = raw_prompt_text + vt_block + "\n" + gt_answer
                                text_aug_full = raw_prompt_text + filtered_vt_block + "\n" + gt_answer
                                
                                inputs_aug = self.processing_class(
                                    text=[text_aug_full],
                                    images=current_images,
                                    videos=current_videos,
                                    return_tensors="pt",
                                    padding=True,
                                ).to(device)
                                
                                seq_len = inputs_aug["input_ids"].shape[1]
                                if seq_len > MAX_PCG_SEQ_LEN:
                                    del inputs_aug
                                    continue 

                                # text_aug_prefix = raw_prompt_text + vt_block + "\n"
                                text_aug_prefix = raw_prompt_text + filtered_vt_block + "\n"
                                inputs_prefix_aug = self.processing_class(
                                    text=[text_aug_prefix],
                                    images=current_images,
                                    videos=current_videos,
                                    return_tensors="pt",
                                    padding=True,
                                )
                                prefix_len = inputs_prefix_aug["input_ids"].shape[1]
                                del inputs_prefix_aug 

                                labels_aug = inputs_aug["input_ids"].clone()
                                safe_len_aug = min(prefix_len, labels_aug.shape[1] - 1)
                                labels_aug[:, :safe_len_aug] = -100
                                
                                visual_kwargs_aug = {
                                    k: v for k, v in inputs_aug.items()
                                    if k not in ["input_ids", "attention_mask", "labels"]
                                }
                                
                                logp_aug = self._compute_log_likelihood(
                                    teacher,
                                    inputs_aug["input_ids"],
                                    inputs_aug["attention_mask"],
                                    labels_aug,
                                    visual_kwargs_aug,
                                )
                                
                                pcg_col[i] = logp_aug - logp_base
                                
                                del inputs_aug, labels_aug, visual_kwargs_aug

                            except RuntimeError as e:
                                if "out of memory" in str(e).lower():
                                    print(f"[PCG] OOM at idx {i}: {e}")
                                    torch.cuda.empty_cache() 
                                pcg_col[i] = 0.0
                            except Exception as e:
                                print(f"[PCG] Error at idx {i}: {e}")
                                pcg_col[i] = 0.0

            except Exception as e_base:
                print(f"[PCG] Critical Error building base: {e_base}")

        # rewards_per_func = torch.cat([rewards_per_func, pcg_col], dim=1)
        # ---- Log PCG filtering stats ----
        if _pcg_total_sents > 0:
            self._metrics["pcg_filter/total_sentences"].append(_pcg_total_sents)
            self._metrics["pcg_filter/removed_sentences"].append(_pcg_removed_sents)
            self._metrics["pcg_filter/removal_rate"].append(_pcg_removed_sents / _pcg_total_sents)
        if _pcg_max_sims:
            self._metrics["pcg_filter/mean_max_sim"].append(sum(_pcg_max_sims) / len(_pcg_max_sims))
            self._metrics["pcg_filter/max_max_sim"].append(max(_pcg_max_sims))
            
        # Per-completion: how many had sentences removed vs not
        n_affected = sum(1 for v in _precomputed_filtered_vt.values() if v[1] > 0)  # num_removed > 0
        n_total_completions = len(_precomputed_filtered_vt) if _precomputed_filtered_vt else 0
        if n_total_completions > 0:
            self._metrics["pcg_filter/affected_completion_rate"].append(n_affected / n_total_completions)
            self._metrics["pcg_filter/affected_completions"].append(n_affected)
        
        # PCG reward stats (after filtering)
        pcg_values = pcg_col.detach().cpu().tolist()
        pcg_nonzero = [v[0] if isinstance(v, list) else v for v in pcg_values if (v[0] if isinstance(v, list) else v) != 0.0]
        if pcg_nonzero:
            self._metrics["pcg_reward/mean_nonzero"].append(sum(pcg_nonzero) / len(pcg_nonzero))
            self._metrics["pcg_reward/max"].append(max(pcg_nonzero))
            self._metrics["pcg_reward/min"].append(min(pcg_nonzero))
            self._metrics["pcg_reward/num_nonzero"].append(len(pcg_nonzero))

        rewards_per_func = torch.cat([rewards_per_func, pcg_col], dim=1)
        
        
        batch_problem_ids = [x.get('id', x.get('problem_id', 'N/A')) for x in inputs]
        try:
            self._log_comprehensive_metrics(
                inputs=inputs, 
                completion_texts=completion_texts, 
                gt_answer=gt_answer, 
                rewards_tensor=rewards_per_func,   
                problem_ids=batch_problem_ids,
                pcg_filter_info=_precomputed_filtered_vt,
            )
        except Exception as e:
            print(f"Logging Error: {e}") 
        batch_size = len(inputs)
        G = self.num_generations
        num_funcs = rewards_per_func.shape[1]

        reshaped_rewards = rewards_per_func.view(batch_size, G, num_funcs)
        means = reshaped_rewards.mean(dim=1, keepdim=True)
        stds = reshaped_rewards.std(dim=1, keepdim=True)
        normalized_rewards = (reshaped_rewards - means) / (stds + 1e-6)
        
        # order：[Accuracy, Format, Clip, PCG]
        w_acc, w_fmt, w_clip, w_pcg = 1.0, 1.0, 0.8, 0.8
        weights = torch.tensor([w_acc, w_fmt, w_clip, w_pcg], device=device).view(1, 1, -1)
        
        # Weighted sum yields Advantage
        advantages = (normalized_rewards * weights).sum(dim=2).view(-1)

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        
        # Mean Loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        gathered_raw_rewards = self.accelerator.gather_for_metrics(rewards_per_func)
        mean_raw_per_func = gathered_raw_rewards.mean(0)
        
        for i, reward_func in enumerate(self.reward_funcs):
            if hasattr(reward_func, "config") and hasattr(reward_func.config, "_name_or_path"):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            elif hasattr(reward_func, "__name__"):
                reward_func_name = reward_func.__name__
            else:
                reward_func_name = f"reward_{i}"
            self._metrics[f"rewards_raw/{reward_func_name}"].append(mean_raw_per_func[i].item())           
            
        self._metrics["rewards_raw/CoT PCG"].append(mean_raw_per_func[-1].item())


        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        
        gathered_advantages = self.accelerator.gather_for_metrics(advantages)
        self._metrics["advantage_std"].append(gathered_advantages.std().item())
        
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
