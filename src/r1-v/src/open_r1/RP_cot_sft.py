# Copyright 2024. All rights reserved.
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
import json
import random
import requests
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info

from datasets import Dataset, DatasetDict

import wandb

from typing import List, Dict, Any
import pdb
from transformers import TrainerCallback

try:
    from movie_chara_profiles import resolve_profile, normalize_movie_name
except ImportError:
    print("Warning: 'movie_chara_profiles' module not found. Using fallback functions.")
    def normalize_movie_name(n): return n
    def resolve_profile(m, n): return {}
    
acc = Accelerator()

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

def _format_character_block(movie: str, name: str) -> str:
    profile = resolve_profile(movie, name)
    lines = [f"Name: {name}", "Profile:"]
    if not profile or not isinstance(profile, dict):
        lines.append("No profile available.")
        return "\n".join(lines) + "\n"
    for k, v in profile.items():
        if v is None: continue
        key_name = k.replace("_", " ").title()
        if isinstance(v, (list, tuple)):
            val_text = ", ".join([str(x) for x in v if str(x).strip() != ""])
        else:
            val_text = str(v).strip()
        if not val_text: continue
        lines.append(f"   - {key_name}: {val_text}")
    return "\n".join(lines) + "\n"

def _format_dialogue_block(dialogue_list) -> str:
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
        f"(3) **Step 3 <answer>...</answer>**: Reply to {user_name} in character. Apply the **High-Stakes Personality Rule** (e.g., humor turns to bravery in danger).\n\n"
        "   - **Spoken-only**: Use colloquial spoken language in the <answer>; forbid formal/essay-like writing.\n"
        f"   - Sound fully **in-world** and unmistakably like {assistant_name}'s voice (personality, tone, cadence, formality, signature phrasing, values/social stance). "
        "   - **Video-Text Relevance**: The line must feel constrained by the **video's atmosphere and visible emotional pressure** (safe vs dangerous; warm vs cold; comedic vs tragic). "
        "   - Maintain emotional/stakes alignment, and do NOT invent specific visual claims (objects/events/locations) that the frames do not support.\n"
        "**Remember to close all tags.** Start immediately with <vision>..."
    )
    return text
    
def prepare_dataset(example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    - Input: System Prompt + Video + Roleplay Prompt
    - Label: Process (<vision>+<think>) + Solution (<answer>)
    """
    
    prompt_text = _build_roleplay_prompt_text(example)
    
    process_content = example.get('process', '')
    solution_content = example.get('solution', '')
    
    full_response = f"{process_content}\n\n{solution_content}"


    video_path = example.get('video_path', None)
    data_type = example.get('data_type', 'video')

    user_content_list = []
    

    if data_type in ['video', 'image'] and video_path:
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}")
        
        user_content_list.append({
            "type": data_type,
            data_type: video_path,
        })
    
    user_content_list.append({
        "type": "text", 
        "text": prompt_text
    })

    current_assistant_name = example.get("assistant", "")
    formatted_system_prompt = SYSTEM_PROMPT.replace("{assistant_name}", current_assistant_name)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": formatted_system_prompt}]},
        {"role": "user", "content": user_content_list},
        {"role": "assistant", "content": [{"type": "text", "text": full_response}]}
    ]

    return {"messages": messages}

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Collate batch of examples for training."""
    texts = []
    # video_inputs = []
    # image_inputs = []

    # if acc.is_main_process:
    #     import pdb; pdb.set_trace()

    for i, example in enumerate(examples):
        try:

            texts.append(processor.apply_chat_template(example["messages"], tokenize=False))
            image_inputs, video_inputs, video_kwargs = process_vision_info(example["messages"], return_video_kwargs=True)
            
        except Exception as e:
            raise ValueError(f"Failed to process example {i}: {e}")

    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    )

    # labels = inputs["input_ids"].clone()
    # labels[labels == processor.tokenizer.pad_token_id] = -100
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Handle visual tokens based on processor type
    visual_tokens = [151652, 151653, 151656] if isinstance(processor, Qwen2VLProcessor) else [
        processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    ]

    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100
        
 
    IM_START_ID = 151644  # <|im_start|>
    
    for i in range(len(input_ids)):
        start_indices = (input_ids[i] == IM_START_ID).nonzero(as_tuple=True)[0]
        
        if len(start_indices) > 0:
            last_start_idx = start_indices[-1]
            mask_len = last_start_idx + 3
            mask_len = min(mask_len, len(labels[i]))
    
            labels[i, :mask_len] = -100

    inputs["labels"] = labels  
    return inputs

class VisualizerCallback(TrainerCallback):
    def __init__(self, eval_dataset, tokenizer, processor, num_samples=2):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.processor = processor
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, model, **kwargs):
        if not args.should_save:
            return

        print(f"\n[Visualizer] Generating {self.num_samples} samples for inspection (Full COT)...", flush=True)
        
        import random
        safe_num = min(len(self.eval_dataset), self.num_samples)
        indices = random.sample(range(len(self.eval_dataset)), safe_num)
        samples = [self.eval_dataset[i] for i in indices]

        model.eval()
        results = []

        for example in samples:
            messages = example["messages"]
            
            user_messages = messages[:-1] 
            ground_truth_response = messages[-1]['content'][0]['text']
            
            text = self.processor.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(user_messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=4096,  
                    temperature=0.7,     
                    do_sample=True
                )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            results.append({
                "step": state.global_step,
                "input_prompt_snippet": text,
                "ground_truth_full": ground_truth_response,
                "model_prediction_full": output_text
            })


        log_file = os.path.join(args.output_dir, "prediction_logs.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

        print(f"[Visualizer] Saved predictions to {log_file}\n")
        model.train()
        
if __name__ == "__main__":
    # Parse arguments
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))

    # if acc.is_main_process:
    #     import pdb; pdb.set_trace()

    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # Configure training args
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

   
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        raw_dataset = Dataset.from_json(script_args.dataset_name)
    else:
        raw_val = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
        if isinstance(raw_val, DatasetDict):
            if "train" in raw_val:
                raw_dataset = raw_val["train"]
            else:
                raw_dataset = raw_val[list(raw_val.keys())[0]]
        else:
            raw_dataset = raw_val

    print(f"Total raw samples: {len(raw_dataset)}")
    
    all_context = raw_dataset[:] 
    session_ids = all_context.get('session_id', [])
    
    if not session_ids:
        print("Warning: 'session_id' not found in dataset. Falling back to random split.")
        dataset_splits = raw_dataset.train_test_split(test_size=0.1, seed=42)
    else:
        unique_sessions = list(set(session_ids))
        unique_sessions.sort() 
        
        random.seed(42) 
        random.shuffle(unique_sessions)
        
        split_idx = int(len(unique_sessions) * 0.9)
        train_sessions = set(unique_sessions[:split_idx])
        eval_sessions = set(unique_sessions[split_idx:])
        
        print(f"Unique Sessions: {len(unique_sessions)}")
        print(f"Train Sessions: {len(train_sessions)}, Eval Sessions: {len(eval_sessions)}")
        
        train_data = raw_dataset.filter(lambda x: x['session_id'] in train_sessions)
        eval_data = raw_dataset.filter(lambda x: x['session_id'] in eval_sessions)
        
        dataset_splits = {
            'train': train_data,
            'test': eval_data
        }

    print(f"Train set size (Samples): {len(dataset_splits['train'])}")
    print(f"Eval set size (Samples): {len(dataset_splits['test'])}")
    
    
    
    train_dataset = [prepare_dataset(example) for example in dataset_splits['train']]
    eval_dataset = [prepare_dataset(example) for example in dataset_splits['test']]
    
    
    # Setup model
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # Model initialization
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        # device_map=get_kbit_device_map(),
        # quantization_config=bnb_config,
    )
    
    
    if "Qwen2-VL" in model_config.model_name_or_path:
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    elif "Qwen2.5-VL" in model_config.model_name_or_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_config.model_name_or_path, **model_kwargs)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )

    # if acc.is_main_process:
    #     import pdb; pdb.set_trace()

    # # Prepare dataset
    # prepared_dataset = [prepare_dataset(example) for example in dataset['train']]

    # Initialize wandb if specified
    if training_args.report_to == "wandb":
        wandb.init(project="Role-Playing-COT-SFT")

    # Initialize callback
    visualizer_callback = VisualizerCallback(
        eval_dataset=eval_dataset, 
        tokenizer=processor.tokenizer, 
        processor=processor, 
        num_samples=4
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_config),
        callbacks=[]
        # tokenizer=processor.tokenizer
    )

    # Train model
    trainer.train()

    # Save final model

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = False
        trainer.model.config.save_pretrained(training_args.output_dir)

    # Cleanup
    del model
    del trainer
    torch.cuda.empty_cache()
    wandb.finish()
