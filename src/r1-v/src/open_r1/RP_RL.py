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
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from trainer import Qwen2VLEBMGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

import requests
import time

import torch
from transformers import CLIPProcessor, CLIPModel
from movie_chara_profiles import resolve_profile, normalize_movie_name

    
CLIP_SERVER_URL = "http://127.0.0.1:5001/score"

BERT_SERVER_URL = "http://127.0.0.1:5000/score"

VISION_EXTRACT_PATTERN = re.compile(r"<vision>(.*?)</vision>", re.DOTALL)
ANSWER_EXTRACT_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

STRICT_TAG_ORDER_PATTERN = re.compile(
    r"<vision>.*?</vision>[^<]*<think>.*?</think>[^<]*<answer>.*?</answer>", 
    re.DOTALL
)

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "clip"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format', 'clip'."},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )
    base_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the base model (containing tokenizer_config.json, etc.) if SFT checkpoint is missing them."},
    )

class CLIPRewardClient:
    def __init__(self):
        pass

    def _write_debug(self, msg):
        if not self.debug_mode:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception as e:
            print(f"[CLIP Client Rank {self.rank}] Logging Error: {e}")

    def _get_full_vision_content(self, text):
        pattern = re.compile(r"<vision>(.*?)</vision>", re.DOTALL)
        content = text[0]['content'] if isinstance(text, list) else str(text)
        match = pattern.search(content)
        if match:
            return match.group(1).strip()
        return "[No <vision> tag detected]"

    def _parse_vision(self, text_list):
        parsed = []
        lengths = []
        mask = []
        # pattern = re.compile(r"<vision>(.*?)</vision>", re.DOTALL)
        match_pattern = VISION_EXTRACT_PATTERN
        for t in text_list:
            content = t[0]['content'] if isinstance(t, list) else str(t)
            # match = pattern.search(content)
            match = match_pattern.search(content)
            if match and match.group(1).strip():
                txt = match.group(1).strip()
                lengths.append(len(txt.split()))
                # parsed.append(txt[:350])
                parsed.append(txt)
                mask.append(True)
            else:
                parsed.append("")
                lengths.append(0)
                mask.append(False)
        return parsed, lengths, mask

    def __call__(self, completions, **kwargs):
        final_scores = [0.0] * len(completions)
        
        embed_paths = kwargs.get('clip_embed_path', [])
        problem_ids = kwargs.get('id', kwargs.get('problem_id', []))
        
        if not embed_paths:
            # self._log_batch(completions, final_scores, problem_ids)
            return final_scores

        texts, lengths, mask = self._parse_vision(completions)
        
        valid_indices = [i for i, m in enumerate(mask) if m and embed_paths[i]]
        valid_texts = [texts[i] for i in valid_indices]
        valid_paths = [embed_paths[i] for i in valid_indices]
        
        # final_scores = [0.0] * len(completions)
        if not valid_indices:
            # self._log_batch(completions, final_scores, problem_ids) 
            return final_scores

        try:
            resp = requests.post(
                CLIP_SERVER_URL, 
                json={"texts": valid_texts, "embed_paths": valid_paths, "indices": valid_indices},
                timeout=30
            )
            if resp.status_code == 200:
                raw_scores = resp.json().get("raw_scores", [])
            else:
                print(f"[CLIP Client] Server Error: {resp.status_code}")
                # self._log_batch(completions, final_scores, problem_ids)
                return final_scores
        except Exception as e:
            print(f"[CLIP Client] Connection Failed: {e}")
            # self._log_batch(completions, final_scores, problem_ids)
            return final_scores

        if raw_scores:           
            for k, original_idx in enumerate(valid_indices):
                final_scores[original_idx] = float(raw_scores[k])
        # self._log_batch(completions, final_scores, problem_ids)
        
        return final_scores

_GLOBAL_CLIP_CLIENT = None
def clip_reward_wrapper(prompts, completions, **kwargs):
    global _GLOBAL_CLIP_CLIENT
    if _GLOBAL_CLIP_CLIENT is None:
        _GLOBAL_CLIP_CLIENT = CLIPRewardClient()
    
    return _GLOBAL_CLIP_CLIENT(completions, **kwargs)



def accuracy_reward(completions, solution, **kwargs):
    import torch.distributed as dist
    import os
    from datetime import datetime
    import re
    from rouge_score import rouge_scorer
    import json
    import subprocess
    import uuid
    
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0


    def extract_answer(text: str) -> str:
        m = ANSWER_EXTRACT_PATTERN.search(text)
        return m.group(1).strip() if m else ""

    question_type = kwargs['problem_type'][0]
    
    contents = [completion[0]["content"] for completion in completions]

    final_scores = [0.0] * len(contents)
    if question_type == "free-form":
        outputs = []
        gts = []
        valid_indices = []
        
        for idx, (content, sol) in enumerate(zip(contents, solution)):           
            out_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            
            if out_ans and gt_ans:
                outputs.append(out_ans)
                gts.append(gt_ans)
                valid_indices.append(idx)

        if outputs:           
            try:
                response = requests.post(
                    BERT_SERVER_URL, 
                    json={"cands": outputs, "refs": gts},
                    timeout=60 
                )
                
                if response.status_code == 200:
                    scores = response.json().get("scores", [])
                    
                    if len(scores) == len(outputs):
                        for list_idx, score in enumerate(scores):
                            final_scores[valid_indices[list_idx]] = score
                    else:
                        print(f"[Rank {rank}] Server returned mismatching scores.")
                else:
                    print(f"[Rank {rank}] Server returned status: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"[Rank {rank}] BERTScore Server Connection Failed: {e}")
                print(f"[Rank {rank}] MAKE SURE you started bertscore_server.py!")
    
    return final_scores


def format_reward(completions, **kwargs):
    tags = ["<vision>", "</vision>", "<think>", "</think>", "<answer>", "</answer>"]
    
    start_pattern = re.compile(r"^\s*<vision>", re.DOTALL)
    end_pattern = re.compile(r"</answer>\s*$", re.DOTALL)

    scores = []
    
    for c in completions:
        content = c[0]["content"] if isinstance(c, list) else c
        score = 0.0
        
        for tag in tags:
            count = content.count(tag)
            if count == 1: score += 0.5 
            else: score -= 0.5 
        
        if STRICT_TAG_ORDER_PATTERN.search(content):
            score += 1.0 
        
        if start_pattern.match(content): score += 0.5
        else: score -= 1.0 
            
        if end_pattern.search(content): score += 0.5
        else: score -= 1.0 
            
        scores.append(score)
            
    return scores

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "clip": clip_reward_wrapper,
}

SYSTEM_PROMPT = (
    "You are an expert Role-Play Dialogue AI.\n"
    "Your goal is to immerse yourself in a specific character (the ASSISTANT) and respond to a USER, "
    "based on a video input and dialogue history.\n"
    "\n"
    "### 1. THE SHARED REALITY (Crucial Rule)\n"
    "The video represents a **Live Event** that occurred immediately before (or during) the current conversation.\n"
    "- **Physical Presence**: Both User and Assistant are physically present in this scene, fully involved in the context.\n"
    "- **The Identity Separation Rule (NO FORCED BINDING)**: \n"
    "  *   The figures visible in the video **might NOT be** the User or Assistant, they could be the other people present. The User/Assistant might be standing just off-camera.\n"
    "  *   **But At least ONE** of the speakers (User or Assistant) was definitely **ON SCREEN** experiencing the event directly.\n"
    "  *   The other speaker was either also on screen, OR standing right next to the action witnessing it.\n"
    "  *   **Strict Constraint**: Do NOT forcefully assume the visible figures are the speakers. \n"
    "  *   *However*: The User and Assistant are **NOT random observers**. They are deeply connected to this event (either experiencing it directly or witnessing it from right next to the action).\n"
    "\n"
    "### 2. CORE RESPONSE LOGIC (The Flow)\n"
    "You must determine the Assistant's response based on the **Dialogue Direction** established in the history:\n"
    "   *   **Direction A (Topic Continuation)**: The dialogue directly **CONTINUES the conversation or interaction** shown in the video. (e.g., The video shows people talking/interacting; the current dialogue picks up right where the video left off, continuing the same specific topic).\n"
    "   *   **Direction B (Inquiry/Reflection)**: The dialogue is a **Reaction/Inquiry** regarding the video event. (e.g., One person is asking the other about their thoughts, feelings, or reasons behind what they just did/said in the video).\n"
    "\n"
    "### 3. THE PERSONALITY FILTER (High-Stakes Logic)\n"
    "Do NOT just output generic emotions. You must filter the observable reality through the Assistant's Profile:\n"
    "- **Standard Rule**: In normal situations, apply standard traits (e.g., A humorous character makes jokes).\n"
    "- **High-Stakes Rule (Crucial)**: Adjust for the **intensity** of the atmosphere.\n"
    "  * If a humorous character faces **mortal danger**, they don't tell stand-up jokes -> their humor becomes nervous, OR they drop the jokes to show unexpected **bravery**.\n"
    "  * If a wise character faces **tragedy**, they don't lecture -> their wisdom becomes gentle silence.\n"
    "\n"
    "### 4. SPOKEN-STYLE REQUIREMENT (MANDATORY)\n"
    "- The final `<answer>` MUST sound like **spoken dialogue**, not prose.\n"
    "- Use **natural colloquial phrasing**, including casual connectors (e.g., \"look\", \"okay\", \"yeah\", \"come on\") and character-appropriate slang/idioms.\n"
    "- **STRICTLY FORBIDDEN**: bookish/formal writing, essay tone, academic vocabulary, structured exposition, or moralizing speeches.\n"
    "- Keep it **short**: one concise line (or two short sentences max).\n"
    "\n"
    "### OUTPUT FORMAT (Strict Step-by-Step)\n"
    "You must output XML-like tags in this exact order:\n"
    "\n"
    "<vision>\n"
    "Describe the **EVENT** objectively. **STRICTLY PROHIBITED: naming figures.**\n"
    "1. **The Core Event (Action & Expression)**: \n"
    "   - Describe the specific interactions/conversation dynamics. (e.g., 'People are having a tense discussion', 'Someone is crying while another comforts them', 'A physical confrontation').\n"
    "   - **Expression Check**: Describe the **visible emotions** of the figures. (e.g., 'One looks desperate, the other looks cold', 'Both seem happy').\n"
    "2. **Key Objects**: Identify items driving the plot (e.g., a wand, a ring, a letter).\n"
    "3. **Atmosphere**: Describe the tension level (Safe vs. Dangerous) and lighting/vibe strictly to set the scene's emotional baseline.\n"
    "</vision>\n"
    "\n"
    "<think>\n"
    "Synthesize Vision + Dialogue History to determine the response:\n"
    "1. **Analyze Vision**: What is the physical reality? (e.g., 'A warm conversation', 'A dangerous battle').\n"
    "2. **Analyze Dialogue History**: Look at the CONTEXT of the conversation so far (User + Assistant):\n"
    "   - Are they **continuing the specific topic** from the video? (Direction A)\n"
    "   - Are they **discussing the aftermath/feelings** of the event? (Direction B)\n"
    "3. **Determine Topic**: Combine [Event] + [Dialogue Direction] to define the current topic. (e.g., 'Continuing the discussion about the plan', 'Asking why they said that').\n"
    "4. **Drafting (Personality Filter)**: \n"
    "   - Internalize ASSISTANT-{assistant_name}'s mindset. Apply the **THE PERSONALITY ANALYSIS** (e.g., humor turns to bravery in danger) while thinking as the assistant. Based on the current situation and the topic of the conversation, and considering the personality and tone of the assistant, think about possible directions and responses for the assistant's next sentence.\n"   
    "   - *Check*: If the Vision is dangerous, does the character show bravery/nervousness instead of casual traits?\n"
    "</think>\n"
    "\n"
    "<answer>\n"
    "The final natural spoken line by the Assistant. No speaker name. No quotes.\n"
    "Sound fully **in-world** and unmistakably like {assistant_name}'s voice (personality, tone, cadence, formality, signature phrasing, values/social stance).\n"
    "- **Video-Text Relevance**: The line must feel constrained by the **video's atmosphere and visible emotional pressure** (safe vs dangerous; warm vs cold; comedic vs tragic). \n "
    "Maintain emotional/stakes alignment, and do NOT invent specific visual claims (objects/events/locations) that the frames do not support.\n"
    "</answer>"
)



def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
        
    
    def _format_character_block(movie: str, name: str) -> str:
        """
        Retrieves character profile (by movie + name) and formats all fields in original order.
        """
        profile = resolve_profile(movie, name)

        lines = [f"Name: {name}", "Profile:"]

        if not profile or not isinstance(profile, dict):
            lines.append("No profile available.")
            return "\n".join(lines) + "\n"

        for k, v in profile.items():
            if v is None:
                continue

            key_name = k.replace("_", " ").title()

            if isinstance(v, (list, tuple)):
                val_text = ", ".join([str(x) for x in v if str(x).strip() != ""])
            else:
                val_text = str(v).strip()

            if not val_text:
                continue

            lines.append(f"   - {key_name}: {val_text}")

        return "\n".join(lines) + "\n"


    def _format_dialogue_block(dialogue_list) -> str:
        """
        dialogue_list: List[{'speaker': str, 'utterance': str}]
        """
        lines = []
        for turn in dialogue_list:
            spk = turn.get("speaker", "Unknown")
            utt = turn.get("utterance", "")
            lines.append(f"{spk}: {utt}")
        return "\n".join(lines)
    
    
    def _build_roleplay_prompt_text(example) -> str:
        movie_raw = example.get("data_source", "Unknown")
        movie = normalize_movie_name(movie_raw)

        user_name = example["user"]
        assistant_name = example["assistant"]

        user_block = _format_character_block(movie, user_name)
        assistant_block = _format_character_block(movie, assistant_name)
        dialogue_block = _format_dialogue_block(example["dialogue"])
        
        text = (
            f"You are role-playing as **{assistant_name}** from the universe of '{movie}'.\n"
            "**Context**: You are physically present in the scene, witnessing/experiencing the events shown below.\n\n"
            "### CHARACTER PROFILES\n"
            f"USER ({user_name}):\n{user_block}\n\n"
            f"ASSISTANT ({assistant_name}):\n{assistant_block}\n\n"
            "### INPUT CONTEXT\n"
            "**[Visual Reality]**: The raw recording of the immediate event. Contains Actions, Objects, and Atmosphere.\n"
            "**[Dialogue Context]**:\n"
            f"{dialogue_block}\n"
            "### INSTRUCTION\n"
            f"Generate the next line for **{assistant_name}**.\n"
            "(1) **Step 1 <vision>...</vision> (Camera Mode)**: Strictly describe **observable facts** ONLY inside these tags. Keep it concise.\n"
            "   - 1. **The Core Event (Action & Expression)**: Describe the event objectively (no guessing). Do NOT name any figures.\n"
            "   - 2. **Key Objects**: List the plot-driving objects.\n"
            "   - 3. **Atmosphere**: State tension level (Safe vs Dangerous) and lighting/vibe briefly.\n"
            "   - **NO ANALYSIS OR GUESSING HERE.** Just report what is seen.\n"
            "(2) **Step 2 <think>...</think> (Analytic Mode)**: Analyze the visual clues and conversation context to explain the logic behind the target response. \n"
            "   - 1. **Analyze Vision**: Summarize the physical reality implied by the vision.\n"
            "   - 2. **Analyze Dialogue History**: Determine whether the conversation is (A) continuing the same topic or (B) discussing reactions/aftermath.\n"
            "   - 3. **Determine Topic**: Combine the event + dialogue direction to determine the current topic.\n"
            f"   - 4. **Drafting (Personality Filter)**: Internalize {assistant_name}'s mindset. Apply the **High-Stakes Personality Rule** (e.g., humor turns to bravery in danger) while thinking as the assistant. Based on the current situation and the topic of the conversation, and considering the personality and tone of the assistant, think about possible directions and responses for the assistant's next sentence.\n"   
            f"(3) **Step 3 <answer>...</answer>**: Reply to {user_name} in character. Apply the **High-Stakes Personality Rule** (e.g., humor turns to bravery in danger).\n"
            "   - **Spoken-only**: Use colloquial spoken language in the <answer>; forbid formal/essay-like writing.\n"
            f"   - Sound fully **in-world** and unmistakably like {assistant_name}'s voice (personality, tone, cadence, formality, signature phrasing, values/social stance). "
            "   - **Video-Text Relevance**: The line must feel constrained by the **video's atmosphere and visible emotional pressure** (safe vs dangerous; warm vs cold; comedic vs tragic). "
            "   - Maintain emotional/stakes alignment, and do NOT invent specific visual claims (objects/events/locations) that the frames do not support.\n"
            "**Remember to close all tags.** Start immediately with <vision>..."
        )
        return text

    def make_conversation_roleplay(example):
        # import pdb; pdb.set_trace()
        prompt_text = _build_roleplay_prompt_text(example)
        data_type = example.get("data_type", None)


        current_assistant_name = example.get("assistant", "")
        formatted_system_prompt = SYSTEM_PROMPT.replace("{assistant_name}", current_assistant_name)
        system_msg = {
            "role": "system",
            "content": [
                {"type": "text", "text": formatted_system_prompt}
            ]
        }

        if data_type in ("image", "video"):
            user_msg = {
                "role": "user",
                "content": [
                    {"type": data_type},
                    {"type": "text", "text": prompt_text}
                ]
            }
        else:
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text}
                ]
            }

        return {"prompt": [system_msg, user_msg]}


    dataset = dataset.map(make_conversation_roleplay)

    processor_path = script_args.base_model_path if script_args.base_model_path else model_args.model_name_or_path
    
    print(f"Loading processor from: {processor_path}")
    try:
        processing_class = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=model_args.trust_remote_code,
            revision=model_args.model_revision,
        )
    except OSError as e:
        print(f"\n[Error] Failed to load processor from {processor_path}.")
        print("Hint: If your SFT checkpoint lacks config files, please provide --base_model_path pointing to the original Qwen2.5-VL directory.\n")
        raise e

    trainer_cls = Qwen2VLEBMGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        processing_class=processing_class,
    )
    
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
