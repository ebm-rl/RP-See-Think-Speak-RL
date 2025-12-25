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

from contextlib import nullcontext

# ========= 统一的标签约束 =========
ANSWER_PATTERN = r"<answer>\s*(.*?)\s*</answer>"

STRICT_PATTERN = r"^\s*<vision>.*?</vision>\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
STRICT_RE = re.compile(STRICT_PATTERN, re.DOTALL)

# ========= ICLG & Format 相关超参（可按需调）=========
ICLG_CLAMP = 2.0          # 对数似然差值截断范围 [-2, 2]
ICLG_SCALE = 0.5          # ICLG 整体缩放
FORMAT_GATE_PENALTY = -1.0   # 格式错时，所有 reward 打成这个值


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb
    

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

from contextlib import contextmanager


class Qwen2VLGRPOTrainer(Trainer):
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
            
        # import pdb; pdb.set_trace() 
        
        # Models
        # Trained model
        # 加载多模态大模型
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

        #self.ref_model = None
        # Reference model
        # 参考模型（grpo中需要一个reference model来计算KL散度）
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
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        # 数据处理类(tokenizer + image processor)
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id or True:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        # 奖励
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

        # Training arguments
        # generation config和temporal/len control，自定义生成三个生成配置
        # self.generation_config正常grpo采样配置；self.shuffled_generation_config只在temporal模式下用；self.dummy_generation_config在没有视频时占位用
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
            pad_token_id=pad_token_id,
        )
        self.shuffled_num_generations = self.num_generations // 2
        self.shuffled_generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=self.shuffled_num_generations,
            pad_token_id=pad_token_id,
        )
        
        self.dummy_generation_config = GenerationConfig(
            max_new_tokens=1,
            do_sample=True,
            top_p=0.95,  
            temperature=1, # HACK
            num_return_sequences=1,
            pad_token_id=pad_token_id,
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

    # 下面的四个函数_set_signature_columns_if_needed、_get_per_token_logps、remove_none_from_data、_prepare_inputs都属于辅助函数
    # 辅助函数
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # 辅助函数
    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, **kwargs):
        # logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)
        # import pdb
        # pdb.set_trace()
        logits = model(input_ids, **kwargs).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)
    
    # 辅助函数
    def remove_none_from_data(self, data):
        for entry in data:
            if "content" in entry and isinstance(entry["content"], list):
                for sub_entry in entry["content"]:
                    if isinstance(sub_entry, dict):
                        keys_to_remove = [k for k, v in sub_entry.items() if v is None]
                        for k in keys_to_remove:
                            del sub_entry[k]
        return data


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    # 辅助函数
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs
    
    
    
    # ========================================================================
    # [Helper 1] 提取 GT Answer (保持与 Accuracy Reward 一致)
    # ========================================================================
    # def _extract_answer(self, text: str) -> str:
    #     # 尝试提取 <answer>...</answer>
    #     pattern = r"<answer>\s*(.*?)\s*</answer>"
    #     match = re.search(pattern, text, re.DOTALL)
    #     if match:
    #         return match.group(1).strip()
    #     # 如果没有 tag，视具体情况返回原文本或空，这里假设数据集中 GT 是纯净的
    #     return text.strip()
    
    def _extract_answer(self, text: str) -> str:
        m = re.search(ANSWER_PATTERN, text, re.DOTALL)
        return m.group(1).strip() if m else ""

    # ========================================================================
    # [Helper 2] 稳健提取 Vision 和 Think 块
    # ========================================================================
    # def _extract_vt_block(self, content: str) -> Optional[str]:
    #     # 分别提取，防止连在一起写正则挂掉
    #     m_v = re.search(r"<vision>(.*?)</vision>", content, re.DOTALL)
    #     m_t = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        
    #     # 这里定义策略：必须两者都有才算有效 ICLG，否则视为无效推理
    #     if not (m_v and m_t):
    #         return None
            
    #     v = m_v.group(1).strip()
    #     t = m_t.group(1).strip()
    #     # 重组标准格式用于拼接
    #     return f"<vision>{v}</vision>\n<think>{t}</think>"
    
    def _extract_vt_block(self, text: str) -> str:
        """提取 <vision>...<think>... 块，用于 ICLG"""
        v_match = re.search(r"<vision>.*?</vision>", text, re.DOTALL)
        t_match = re.search(r"<think>.*?</think>", text, re.DOTALL)
        if v_match and t_match:
            # 截取从 vision 开始到 think 结束
            return text[v_match.start():t_match.end()]
        return ""

    # # ========================================================================
    # # [Helper 3] 底层手动 Forward 计算 Loss (No Grad, CPU return)
    # # ========================================================================
    # def _compute_manual_loss(self, model, input_ids, attention_mask, labels, visual_inputs):
    #     """
    #     手动构造 Forward Pass 计算 Loss。
    #     """
    #     # 1. 构造 kwargs
    #     kwargs = {
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask,
    #         "labels": labels
    #     }
    #     # 2. 合并视觉特征
    #     kwargs.update(visual_inputs)
        
    #     # 3. Forward (No Grad)
    #     with torch.no_grad():
    #         # 必须 unwrapped 才能跳过 DDP 的一些同步检查，且支持单卡 forward
    #         # 但这里 model 可能是 DeepSpeed engine，直接 call 即可
    #         outputs = model(**kwargs)
    #         loss = outputs.loss 
        
    #     # 返回 float，减少显存占用
    #     return loss.item()
    
    # [Helper 3] 底层计算 Log Likelihood (对数似然)
    # ========================================================================
    def _compute_log_likelihood(self, model, input_ids, attention_mask, labels, visual_inputs):
        """
        计算给定输入的平均对数似然 (Average Log-Likelihood per token).
        论文公式对应: 1/|d| * log π(d | x)
        实现方式: - CrossEntropyLoss
        """
        # 1. 构造 kwargs
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        # 2. 合并视觉特征
        kwargs.update(visual_inputs)
        
        # 3. Forward (No Grad)
        with torch.no_grad():
            outputs = model(**kwargs)
            # CrossEntropyLoss 计算的是 Negative Log-Likelihood
            nll = outputs.loss 
        
        # 返回 Log-Likelihood = -NLL
        return -nll.item()


    # 核心GRPO+temporal部分
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        # device = self.accelerator.device
        
        # # === 1. 初始化有效性权重 (默认有效) ===
        # sample_weight = 1.0 
        
        # # 请确保这个文件真实存在！！
        # fallback_video_path = "/data/wm/simple-subtitling/Processed_Dialogue/The_Twilight_Saga/video/1~4_NEW_fixed_videos/1d37ccbacf96f997_sub1_2.mp4"
    
        

        # (1) 准备prompts & 文本版本
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

                
        
        # (2) 图像/视频处理 + 构造 Processor 输入
        input_copy = copy.deepcopy(inputs[0]['prompt'])
        
        input_copy = self.remove_none_from_data(input_copy)
        
        current_video_path = "Unknown"
        if inputs[0]['data_type'] == 'image':
            # input_copy[0]['content'][0]['image'] = os.getcwd() + "/Video-R1-data" + inputs[0]['path'][1:] 
            input_copy[1]['content'][0]['image'] = inputs[0]['image_path']
        elif inputs[0]['data_type'] == 'video':
            # input_copy[0]['content'][0]['video'] = os.getcwd() + "/Video-R1-data" + inputs[0]['path'][1:] 
            current_video_path = inputs[0]['video_path']
            input_copy[1]['content'][0]['video'] = inputs[0]['video_path']
            
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
             
            # # 二次检查：防止 process_vision_info 返回空但不报错
            # if inputs[0]['data_type'] == 'video':
            #     if video_inputs is None or (isinstance(video_inputs, list) and len(video_inputs) == 0):
            #         raise ValueError("Read success but video_inputs is Empty")
        
        except Exception as e:
            print(f"process_vision_info error, using fixed data, {e}")
            if inputs[0]['data_type'] == 'image':
                input_copy[0]['content'][0]['image'] = os.getcwd() + "/Video-R1-data" + '/Math/Multimath-300k/17ff4c7d14c388134de02381b1fc2824.png'
            elif inputs[0]['data_type'] == 'video':
                input_copy[0]['content'][0]['video'] = os.getcwd() + "/Video-R1-data" + '/LLaVA-Video-178K/liwei_youtube_videos/videos/youtube_video_2024/ytb_7nRmsEw7nsE.mp4'
                
            image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)

            # ################# Modified Part #################
            # # 标记为坏样本，权重置 0
            # sample_weight = 0.0
            # print(f"❌ [Rank {self.accelerator.process_index}] Bad Video: {current_video_path} | Err: {e}")
                       
            # log_path = "/data/wm/Video-R1/src/r1-v/src/open_r1/myself_grpo_training_bad_videos.txt"

            # try:
            #     with open(log_path, "a") as f:
            #         f.write(f"{current_video_path}\n")
            # except Exception as e_file: 
            #     print(f"⚠️ Logging Failed! Could not write to {log_path}. Error: {e_file}")
            # # ----------------------------------------
            
            # # 使用 fallback 视频（仅为了占位，不参与训练）
            # if inputs[0]['data_type'] == 'video':
            #     input_copy[1]['content'][0]['video'] = fallback_video_path
            
            # try:
            #     image_inputs, video_inputs, video_kwargs = process_vision_info(input_copy, return_video_kwargs=True)
            # except Exception as e2:
            #     # 终极兜底：手动造假 Tensor 防止 Crash
            #     print(f"Critical: Fallback failed! {e2}")
            #     device = self.accelerator.device
            #     if inputs[0]['data_type'] == 'video':
            #         # 假设模型需要 [8, 3, 224, 224]，dtype跟模型一致
            #         video_inputs = [torch.zeros((8, 3, 224, 224), device=device, dtype=model.dtype)]
            #         video_kwargs = {"video_grid_thw": torch.tensor([[8, 224, 224]], device=device, dtype=torch.long)}
            #     else:
            #         video_inputs = None

        
        # 用 AutoProcessor 把文本+图像/视频打包成张量，再截断到 max_prompt_length：
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
            
        # (3) temporal：打乱视频帧版本（视频 & 开启temporal）：
        if self.temporal and video_inputs:
            indices = torch.randperm(video_inputs[0].size(0))
            shuffled_video_inputs = [video_inputs[0][indices]]
            shuffled_prompt_inputs = self.processing_class(
                text=copy.deepcopy(prompts_text),
                images=image_inputs,
                videos=shuffled_video_inputs,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            shuffled_prompt_inputs = super()._prepare_inputs(shuffled_prompt_inputs)
            shuffled_prompt_ids, shuffled_prompt_mask = shuffled_prompt_inputs["input_ids"], shuffled_prompt_inputs["attention_mask"]
            if self.max_prompt_length is not None:
                shuffled_prompt_ids = shuffled_prompt_ids[:, -self.max_prompt_length :]
                shuffled_prompt_mask = shuffled_prompt_mask[:, -self.max_prompt_length :]
        
        
        # Generate completions
        # (4) 用模型 generate
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # prompt_completion_ids里包含了[prompt tokens][completion tokens]
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
           
            # 用prompt_length把它们拆成prompt_ids和completion_ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            
            # temporal 模式下还会为打乱时序的视频生成一份suffled_prompt_completion_ids
            if self.temporal:
                
                if video_inputs:
            
                    shuffled_prompt_completion_ids = unwrapped_model.generate(**shuffled_prompt_inputs, generation_config=self.shuffled_generation_config)
                    shuffled_prompt_length = shuffled_prompt_ids.size(1)
                    shuffled_prompt_ids = shuffled_prompt_completion_ids[:, :shuffled_prompt_length]
                    shuffled_completion_ids = shuffled_prompt_completion_ids[:, shuffled_prompt_length:]
                    shuffled_prompt_mask = prompt_mask.repeat_interleave(self.shuffled_num_generations, dim=0)
                    
                else:
                    
                    shuffled_prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.dummy_generation_config)

        
        # print('path:', input_copy[0]['content'][0][inputs[0]['data_type']])   
        print('path:', input_copy[1]['content'][0][inputs[0]['data_type']])   
        print('problem_id:', inputs[0]['problem_id'])       
        print('prompt_length:', prompt_length)
                
        
        
        
        # Mask everything after the first EOS token
        # (5) 计算completion_mask：根据第一个EOS的位置构造completion_mask，把EOS 之后的 token 屏蔽掉，避免无效的尾巴影响 loss / reward 等：
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        # attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        # pixel_values = prompt_inputs["pixel_values"].repeat(self.num_generations, 1)
        # image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)
        

        
        # (6) 为B*G复制视觉输入：
        # 因为 generate 返回了B*G条序列，而原来的pixel_values只有B条（通常就是1）
        # 这里通过repeat把视觉特征复制到每一个样本对应的行上，这样后面(7)中_get_per_token_logps在算logits时能对齐
        prompt_inputs.pop("input_ids")
        prompt_inputs.pop("attention_mask")
        
        if inputs[0]['data_type'] == 'image':
            prompt_inputs["pixel_values"] = prompt_inputs["pixel_values"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"].repeat(len(prompt_completion_ids), 1)
        # import pdb; pdb.set_trace()
        

        if inputs[0]['data_type'] == 'video':
            prompt_inputs["pixel_values_videos"] = prompt_inputs["pixel_values_videos"].repeat(len(prompt_completion_ids), 1)
            prompt_inputs["video_grid_thw"] = prompt_inputs["video_grid_thw"].repeat(len(prompt_completion_ids), 1)
            if 'second_per_grid_ts' in prompt_inputs: # this two lines could be deleted: https://github.com/tulerfeng/Video-R1/issues/73
                del prompt_inputs["second_per_grid_ts"]
                # prompt_inputs["second_per_grid_ts"] = torch.tensor(prompt_inputs["second_per_grid_ts"]).repeat(len(prompt_completion_ids), 1)
        
        
        
        
        # (7) 计算per_token_logps & ref_per_token_logps：
        # 只保留completion部分的logp
        # ref_model一般是初始模型、或者关闭adapter的版本

        try:
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
            per_token_logps = per_token_logps[:, prompt_length - 1 :]
        except Exception as e:
            print(f"Error computing per_token_logps: {e}. Setting output to zero.")
            # per_token_logps = torch.tensor(0.0, device=prompt_completion_ids.device, requires_grad=True)
            per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
        
        with torch.inference_mode():
            try:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, **prompt_inputs)
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, **prompt_inputs)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]
            except Exception as e:
                print(f"Error computing ref_per_token_logps: {e}. Setting output to zero.")
                # ref_per_token_logps = torch.tensor(0.0, device=prompt_completion_ids.device)
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # Compute the KL divergence between the model and the reference model
        
        # 然后计算per-token KL（这里是GRPO论文里的KL近似形式）：
        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)  # 限制 x 的范围
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1
        
        # (8) temporal reward时序奖励：打乱 vs 原视频
        if self.temporal and video_inputs:
            # (8-1)对打乱后的视频生成的completion，解码成shuffled_completions
            shuffled_completions = self.processing_class.batch_decode(shuffled_completion_ids, skip_special_tokens=True)
            if is_conversational(inputs[0]):
                shuffled_completions = [[{"role": "assistant", "content": shuffled_completion}] for shuffled_completion in shuffled_completions]
                
            # Compute the rewards
            # (8-2)用reward_funcs分别算reward，得到shuffled_rewards_per_func：
            shuffled_prompts = [prompt for prompt in prompts for _ in range(self.shuffled_num_generations)]
            shuffled_rewards_per_func = torch.zeros(len(shuffled_prompts), len(self.reward_funcs), device=device)
            for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
            ):
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                shuffled_reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in shuffled_reward_kwargs:
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        shuffled_reward_kwargs[key].extend([example[key]] * self.shuffled_num_generations)
                shuffled_output_reward_func = reward_func(prompts=shuffled_prompts, completions=shuffled_completions, **shuffled_reward_kwargs)
                shuffled_rewards_per_func[:, i] = torch.tensor(shuffled_output_reward_func, dtype=torch.float32, device=device)

        
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]
           
        ###################### NEW #######################  
        # 提前抽出纯文本 content，便于做格式检查与 ICLG
        completion_texts = [
            c[0]["content"] if isinstance(c, list) else c
            for c in completions
        ]
        # 用统一 STRICT_PATTERN 做格式判定
        format_mask_list = [bool(STRICT_RE.fullmatch(t)) for t in completion_texts]
        format_mask = torch.tensor(format_mask_list, device=device)        
        ###################### NEW #######################  
        
        # (8-3)对原视频的completion做同样的rewards_per_func计算:
        # Compute the rewards
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
            

        # ======================================================================
        # ICLG Integration Start (集成 ICLG)
        # ======================================================================
        # import torch.distributed as dist
        # rank = dist.get_rank() if dist.is_initialized() else 0
        # if int(rank) == 0:
        #     print(f"[rank0] RemotePdb waiting at 127.0.0.1:4444 ...", flush=True)
        #     from remote_pdb import RemotePdb
        #     RemotePdb('127.0.0.1', 4444).set_trace()
        
        
        # # 1. 准备 GT
        # gt_raw = inputs[0].get('solution', inputs[0].get('answer', inputs[0].get('reference', "")))
        # if isinstance(gt_raw, list): gt_raw = gt_raw[0] 
        # gt_answer = self._extract_answer(gt_raw)

        # # 2. 准备 ICLG 列容器
        # iclg_col = torch.zeros(rewards_per_func.size(0), 1, device=device)
        
        # MAX_ICLG_SEQ_LEN = 16384


        # if gt_answer:
        #     # ==================================================================
        #     # 【优化】使用 unwrap 确保模型处于推理模式，显存更安全
        #     # ==================================================================            
        #     with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
        #         try:
        #             # ----------------------------------------------------------
        #             # 准备 Base 数据
        #             # ----------------------------------------------------------
        #             raw_prompt_text = prompts_text[0]
        #             is_image = (inputs[0]['data_type'] == 'image' and image_inputs is not None)
        #             is_video = (inputs[0]['data_type'] == 'video' and video_inputs is not None)
        #             current_images = image_inputs if is_image else None
        #             current_videos = video_inputs if is_video else None

        #             # 计算 Base LogP
        #             text_base = raw_prompt_text + gt_answer
        #             inputs_base = self.processing_class(
        #                 text=[text_base],
        #                 images=current_images,
        #                 videos=current_videos,
        #                 return_tensors="pt",
        #                 padding=True,
        #             ).to(device)

        #             # 为了 Mask 计算 Prompt Length
        #             inputs_prefix_base = self.processing_class(
        #                 text=[raw_prompt_text],
        #                 images=current_images,
        #                 videos=current_videos,
        #                 return_tensors="pt",
        #                 padding=True,
        #             )
        #             prompt_len = inputs_prefix_base["input_ids"].shape[1]
        #             labels_base = inputs_base["input_ids"].clone()
        #             safe_len_base = min(prompt_len, labels_base.shape[1] - 1)
        #             labels_base[:, :safe_len_base] = -100
                    
        #             visual_kwargs_base = {
        #                 k: v for k, v in inputs_base.items() 
        #                 if k not in ["input_ids", "attention_mask", "labels"]
        #             }
                    
        #             # 使用 unwrapped_model 计算
        #             logp_base = self._compute_log_likelihood(
        #                 unwrapped_model, 
        #                 inputs_base["input_ids"], 
        #                 inputs_base["attention_mask"], 
        #                 labels_base, 
        #                 visual_kwargs_base
        #             )
                    
        #             # 立即清理
        #             del inputs_base, labels_base, visual_kwargs_base, inputs_prefix_base
        #             # with 内部不要频繁 empty_cache，容易拖慢速度，靠 del 就够了

        #             # ----------------------------------------------------------
        #             # D. 计算 Aug Log-Likelihood 
        #             # ----------------------------------------------------------
        #             for i, completion_text in enumerate(completions): 
        #                 content = completion_text[0]['content'] if isinstance(completion_text, list) else completion_text
        #                 vt_block = self._extract_vt_block(content)
                        
        #                 if vt_block:
        #                     try:
        #                         text_aug_full = raw_prompt_text + vt_block + "\n" + gt_answer
        #                         text_aug_prefix = raw_prompt_text + vt_block + "\n"
                                
        #                         inputs_aug = self.processing_class(
        #                             text=[text_aug_full],
        #                             images=current_images,
        #                             videos=current_videos,
        #                             return_tensors="pt",
        #                             padding=True,
        #                         ).to(device)
                                
        #                         # 【关键】长度截断
        #                         seq_len = inputs_aug["input_ids"].shape[1]
        #                         if seq_len > MAX_ICLG_SEQ_LEN:
        #                             # print(f"⚠️ Skip ICLG idx {i} (Len {seq_len})")
        #                             del inputs_aug
        #                             iclg_col[i] = 0.0
        #                             continue 

        #                         inputs_prefix_aug = self.processing_class(
        #                             text=[text_aug_prefix],
        #                             images=current_images,
        #                             videos=current_videos,
        #                             return_tensors="pt",
        #                             padding=True,
        #                         )
        #                         prefix_len = inputs_prefix_aug["input_ids"].shape[1]
        #                         del inputs_prefix_aug # 用完即弃

        #                         labels_aug = inputs_aug["input_ids"].clone()
        #                         safe_len_aug = min(prefix_len, labels_aug.shape[1] - 1)
        #                         labels_aug[:, :safe_len_aug] = -100
                                
        #                         visual_kwargs_aug = {
        #                             k: v for k, v in inputs_aug.items() 
        #                             if k not in ["input_ids", "attention_mask", "labels"]
        #                         }
                                
        #                         logp_aug = self._compute_log_likelihood(
        #                             unwrapped_model, 
        #                             inputs_aug["input_ids"], 
        #                             inputs_aug["attention_mask"], 
        #                             labels_aug, 
        #                             visual_kwargs_aug
        #                         )
                                
        #                         iclg_col[i] = logp_aug - logp_base
                                
        #                         del inputs_aug, labels_aug, visual_kwargs_aug
                                
        #                     except RuntimeError as e:
        #                         if "out of memory" in str(e):
        #                             print(f"⚠️ ICLG OOM at idx {i}")
        #                             iclg_col[i] = 0.0
        #                         else:
        #                             iclg_col[i] = 0.0
        #                     except Exception:
        #                         iclg_col[i] = 0.0
        #                 else:
        #                     iclg_col[i] = 0.0
                
        #         except Exception as e_base:
        #             print(f"[ICLG] Critical Error: {e_base}")
        #             pass
            
        #     # 退出 unwrap 后，做一次彻底清理
        #     torch.cuda.empty_cache()



        # ======================================================================
        #  Teacher-Forced ICLG 集成
        # ======================================================================
        # 1. 准备 GT (Ground Truth)
        gt_raw = inputs[0].get(
            "solution",
            inputs[0].get("answer", inputs[0].get("reference", "")),
        )
        if isinstance(gt_raw, list):
            gt_raw = gt_raw[0]
        gt_answer = self._extract_answer(gt_raw)

        # 2. 准备 ICLG 列容器 (默认全为0)
        # rewards_per_func 的形状是 [Batch*Generations, Num_Rewards]
        iclg_col = torch.zeros(rewards_per_func.size(0), 1, device=device)
        
        MAX_ICLG_SEQ_LEN = 16384

        # 仅当存在有效 GT 时才计算 ICLG
        if gt_answer:
            try:
                # 准备基础数据上下文
                raw_prompt_text = prompts_text[0]
                is_image = inputs[0]["data_type"] == "image" and image_inputs is not None
                is_video = inputs[0]["data_type"] == "video" and video_inputs is not None
                current_images = image_inputs if is_image else None
                current_videos = video_inputs if is_video else None

                # -------------------------------------------------------------------------
                # 【关键修复】: 选择 Teacher 模型并全量解包 (Unwrap)
                #  这会将参数 Gather 到本地显存并驻留，避免循环中的反复通信导致超时。
                # -------------------------------------------------------------------------
                
                # A. 确定 Teacher: 优先用 ref_model，没有则用 model
                target_model_to_unwrap = self.ref_model if self.ref_model is not None else model
                
                # B. 创建解包上下文
                ctx = unwrap_model_for_generation(target_model_to_unwrap, self.accelerator)

                with ctx as teacher:
                    # C. 处理 Adapter: 
                    # 如果用的是 model (target_model_to_unwrap == model)，它带有 Adapter，需禁用以作为 Base。
                    # 如果用的是 ref_model，通常没有 Adapter，disable_adapter() 可能不存在或无需调用。
                    # 为了兼容性，尝试禁用 adapter (如果存在该方法)。
                    if hasattr(teacher, "disable_adapter"):
                        inner_ctx = teacher.disable_adapter()
                    else:
                        inner_ctx = nullcontext()

                    with inner_ctx:
                        # ======================================================
                        # 3. 计算 Base Log-Likelihood: p(d_gt | x)
                        # ======================================================
                        text_base = raw_prompt_text + gt_answer
                        inputs_base = self.processing_class(
                            text=[text_base],
                            images=current_images,
                            videos=current_videos,
                            return_tensors="pt",
                            padding=True,
                        ).to(device)

                        # 计算 Prompt 部分的长度，用于 Mask 掉 Prompt 的 Loss
                        inputs_prefix_base = self.processing_class(
                            text=[raw_prompt_text],
                            images=current_images,
                            videos=current_videos,
                            return_tensors="pt",
                            padding=True,
                        )
                        prompt_len = inputs_prefix_base["input_ids"].shape[1]
                        del inputs_prefix_base  # 及时清理

                        # 构造 Labels 并 Mask 掉 Prompt 部分
                        labels_base = inputs_base["input_ids"].clone()
                        safe_len_base = min(prompt_len, labels_base.shape[1] - 1)
                        labels_base[:, :safe_len_base] = -100
                        
                        # 提取视觉参数
                        visual_kwargs_base = {
                            k: v for k, v in inputs_base.items()
                            if k not in ["input_ids", "attention_mask", "labels"]
                        }
                        
                        # 计算 Base 分数
                        logp_base = self._compute_log_likelihood(
                            teacher,  # 使用解包后的 teacher
                            inputs_base["input_ids"],
                            inputs_base["attention_mask"],
                            labels_base,
                            visual_kwargs_base,
                        )
                        # 清理 Base 中间变量
                        del inputs_base, labels_base, visual_kwargs_base

                        # ======================================================
                        # 4. 计算 Augmented Log-Likelihood: p(d_gt | x, c)
                        # ======================================================
                        for i, content in enumerate(completion_texts):
                            # 如果格式检查未通过 (Format Gate)，跳过计算，节省时间
                            if not format_mask_list[i]:
                                continue

                            # 提取 <vision>...<think>... 块
                            vt_block = self._extract_vt_block(content)
                            if not vt_block:
                                continue

                            try:
                                # 构造 Aug 文本: Prompt + VT_Block + GT
                                text_aug_full = raw_prompt_text + vt_block + "\n" + gt_answer
                                
                                inputs_aug = self.processing_class(
                                    text=[text_aug_full],
                                    images=current_images,
                                    videos=current_videos,
                                    return_tensors="pt",
                                    padding=True,
                                ).to(device)
                                
                                # 长度截断检查
                                seq_len = inputs_aug["input_ids"].shape[1]
                                if seq_len > MAX_ICLG_SEQ_LEN:
                                    del inputs_aug
                                    continue 

                                # 计算前缀长度 (Prompt + VT_Block)
                                text_aug_prefix = raw_prompt_text + vt_block + "\n"
                                inputs_prefix_aug = self.processing_class(
                                    text=[text_aug_prefix],
                                    images=current_images,
                                    videos=current_videos,
                                    return_tensors="pt",
                                    padding=True,
                                )
                                prefix_len = inputs_prefix_aug["input_ids"].shape[1]
                                del inputs_prefix_aug # 用完即弃

                                # 构造 Labels 并 Mask 掉前缀
                                labels_aug = inputs_aug["input_ids"].clone()
                                safe_len_aug = min(prefix_len, labels_aug.shape[1] - 1)
                                labels_aug[:, :safe_len_aug] = -100
                                
                                visual_kwargs_aug = {
                                    k: v for k, v in inputs_aug.items()
                                    if k not in ["input_ids", "attention_mask", "labels"]
                                }
                                
                                # 计算 Aug 分数
                                logp_aug = self._compute_log_likelihood(
                                    teacher, # 使用解包后的 teacher
                                    inputs_aug["input_ids"],
                                    inputs_aug["attention_mask"],
                                    labels_aug,
                                    visual_kwargs_aug,
                                )
                                
                                # 计算 ICLG Gain
                                iclg_col[i] = logp_aug - logp_base
                                
                                # 清理 Aug 中间变量
                                del inputs_aug, labels_aug, visual_kwargs_aug

                            except RuntimeError as e:
                                if "out of memory" in str(e).lower():
                                    print(f"[ICLG] OOM at idx {i}: {e}")
                                    # OOM 时保持为 0，并清理显存
                                    torch.cuda.empty_cache() 
                                iclg_col[i] = 0.0
                            except Exception as e:
                                print(f"[ICLG] Error at idx {i}: {e}")
                                iclg_col[i] = 0.0

            except Exception as e_base:
                print(f"[ICLG] Critical Error building base: {e_base}")

                        
        # 3. 合并 Reward 到总表（列顺序：0=bertscore acc,1=format,2=clip,3=iclg）
        # rewards_per_func: [B*G, Old_Num] -> [B*G, Old_Num + 1]
        rewards_per_func = torch.cat([rewards_per_func, iclg_col], dim=1)
        
        # ======================================================================
        # NEW-Format Gating：一票否决
        # ======================================================================
        # 只要整体输出不满足 STRICT_PATTERN，就把所有 reward 打成负常数
        bad_mask = ~format_mask  # shape: (B*G,)
        rewards_per_func[bad_mask, :] = FORMAT_GATE_PENALTY
        
        # ----------------------------------------------------------------------
        # NEW-统计与 Advantage 
        # ---------------------------------------------------------------------
        
        # # 1. 准备数据维度
        # # rewards_per_func shape: (Batch * G, Num_Functions)
        # batch_size = len(inputs)
        # G = self.num_generations
        # num_funcs = rewards_per_func.shape[1]
        
        # # Reshape 成 (Batch, G, Num_Functions)
        # reshaped_rewards = rewards_per_func.view(batch_size, G, num_funcs)
        
        # # 2. 对每个 Reward 函数，分别计算组内均值和标准差
        # # mean/std shape: (Batch, 1, Num_Functions)
        # mean_rewards = reshaped_rewards.mean(dim=1, keepdim=True)
        # std_rewards = reshaped_rewards.std(dim=1, keepdim=True)
        
        # # 3. 组内归一化 (Z-Score)
        # # 这样所有 Reward 都会变成 0均值、1方差的分布，量级被拉平
        # normalized_rewards = (reshaped_rewards - mean_rewards) / (std_rewards + 1e-4)
        
        # # 4. 定义重要性权重
        # # [1.0, 1.0, 1.0] 代表三个reward在“相对进步幅度”上的贡献完全一致
        # # func_weights = torch.tensor([1.0, 1.0, 1.0], device=per_token_logps.device)
        # func_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], device=per_token_logps.device)
        # func_weights = func_weights.view(1, 1, -1)
        
        # # 5. 加权求和得到最终 Advantage
        # # shape: (Batch, G) -> Flatten to (Batch * G)
        # advantages = (normalized_rewards * func_weights).sum(dim=2).view(-1)
        
        # ======================================================================
        # NEW-Advantage：只对 [acc, clip, iclg] 做 z-score + 加权
        # ======================================================================
        batch_size = len(inputs)
        G = self.num_generations
        num_funcs = rewards_per_func.shape[1]

        reshaped_rewards = rewards_per_func.view(batch_size, G, num_funcs)

        # 列索引：0=acc,1=format,2=clip,3=iclg
        acc_idx, fmt_idx, clip_idx, iclg_idx = 0, 1, 2, 3

        main_indices = [acc_idx, clip_idx, iclg_idx]
        main_rewards = reshaped_rewards[:, :, main_indices]

        main_mean = main_rewards.mean(dim=1, keepdim=True)
        main_std = main_rewards.std(dim=1, keepdim=True)
        normalized_main = (main_rewards - main_mean) / (main_std + 1e-4)

        # 权重：嘴，眼，脑（ICLG 是 shaping 信号）
        w_acc, w_clip, w_iclg = 1.0, 0.6, 0.6
        func_weights = torch.tensor([w_acc, w_clip, w_iclg], device=device).view(1, 1, -1)

        advantages = (normalized_main * func_weights).sum(dim=2).view(-1)
        
        # # 为了日志，把归一化结果嵌回完整矩阵（format 那一列保持 0）
        # normalized_rewards = torch.zeros_like(reshaped_rewards)
        # normalized_rewards[:, :, acc_idx] = normalized_main[:, :, 0]
        # normalized_rewards[:, :, clip_idx] = normalized_main[:, :, 1]
        # normalized_rewards[:, :, iclg_idx] = normalized_main[:, :, 2]     
        
        
        # ==============================================================================
        # [新逻辑] 计算 Loss
        # ==============================================================================
        
        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        
        # Mean Loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # ==============================================================================
        # [新逻辑] 日志记录 (Metrics Logging)
        # ==============================================================================
        
        # 1. 记录生成长度
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        # # normalized_rewards 是 (Batch, G, Num_Funcs)，我们需要 Flatten 回 (Batch*G, Num_Funcs) 以便 Gather
        # flat_normalized_rewards = normalized_rewards.reshape(-1, num_funcs)
        
        # # Gather 多卡数据
        # gathered_normalized_rewards = self.accelerator.gather_for_metrics(flat_normalized_rewards)
        # # 计算所有样本的均值，得到每个 Reward 函数的平均归一化分数
        # mean_normalized_per_func = gathered_normalized_rewards.mean(0) # shape: (Num_Functions,)
 
        # 保留 Raw Reward 的记录用于对比
        gathered_raw_rewards = self.accelerator.gather_for_metrics(rewards_per_func)
        mean_raw_per_func = gathered_raw_rewards.mean(0)
        
        # 2. 遍历记录每个 Reward 函数 (Accuracy, Format, Clip)
        for i, reward_func in enumerate(self.reward_funcs):
            # 获取函数名
            if hasattr(reward_func, "config") and hasattr(reward_func.config, "_name_or_path"):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            elif hasattr(reward_func, "__name__"):
                reward_func_name = reward_func.__name__
            else:
                reward_func_name = f"reward_{i}"
            
            # # === 修改：记录 Normalized 值 ===
            # # key 加上 "_norm" 后缀以示区分，或者直接覆盖原名
            # self._metrics[f"rewards_normalized/{reward_func_name}"].append(mean_normalized_per_func[i].item())
 
            # 同时记录 Raw 值
            self._metrics[f"rewards_raw/{reward_func_name}"].append(mean_raw_per_func[i].item())           
            
        # 3. 显式记录 ICLG
        # ICLG 是最后一列
        # self._metrics["rewards_normalized/CoT ICLG"].append(mean_normalized_per_func[-1].item())
        self._metrics["rewards_raw/CoT ICLG"].append(mean_raw_per_func[-1].item())

            
        # 4. 记录加权后的总分 (这里我们也记录 Normalized 的总分，反映 Advantage 的水平)
        # 注意：Raw Sum Reward 其实是没有任何归一化的总分，Advantage 才是归一化加权后的总分
        # 这里我们记录 平均 Advantage，它等于 Sum(Normalized_Reward * Weight)
        mean_advantage = self.accelerator.gather_for_metrics(advantages).mean().item()
        self._metrics["Total_reward"].append(mean_advantage) 
        
        # 5. 记录 KL 散度
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        
        # 6. 记录 Advantage 的标准差
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
