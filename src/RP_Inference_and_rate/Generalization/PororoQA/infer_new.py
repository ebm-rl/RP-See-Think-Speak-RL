import sys
import os
import json
import re
import argparse
import pandas as pd
import time
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import random
from PIL import Image

def get_frames_from_gif(gif_path, nframes=2):
    try:
        with Image.open(gif_path) as im:
            frames = []
            total_frames = im.n_frames
            if total_frames <= 1:
                indices = [0] * nframes
            else:
                indices = [int(total_frames * 0.33), int(total_frames * 0.66)]
            
            for idx in indices:
                im.seek(idx)
                frames.append(im.convert("RGB"))
            return frames
    except Exception as e:
        print(f"Warning: Failed to read {gif_path}: {e}")
        return []

current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(current_dir, "../r1-v/src/open_r1")
target_dir = os.path.abspath(target_dir)
if target_dir not in sys.path:
    sys.path.append(target_dir)

SYSTEM_HEADER_SHARED = (
    "You are a Video Expert. Identify characters in the Pororo series using these STICKY VISUAL FEETS:\n"
    "- **Pororo**: Blue/white penguin, YELLOW racing helmet, ORANGE goggles.\n"
    "- **Crong**: Small BRIGHT GREEN dinosaur. Usually shirtless.\n"
    "- **Loopy**: Light PINK beaver. Small PINK FLOWER clip on head.\n"
    "- **Eddy**: ORANGE fox, white belly. Often carries gadgets.\n"
    "- **Poby**: Large WHITE polar bear. Wears BLUE dungaree/overalls.\n"
    "- **Petty**: Blue/white girl penguin. PURPLE winter hat & PURPLE coat.\n"
    "- **Harry**: Tiny PINK hummingbird, PURPLE bowtie. Often flying.\n"
    "- **Tongtong**: ORANGE dragon, wears a small TUXEDO.\n"
    "- **Rody**: Bright YELLOW robot, long mechanical arms.\n\n"
    "Analyze the 32-frame sequence as a continuous event. Identify characters by COLOR and ACCESSORIES first."
)

SYSTEM_SUFFIX_OURS = (    
    "### OUTPUT FORMAT (Strict Step-by-Step)\n"
    "You must output XML-like tags in this exact order:\n"
    "<vision>: List characters by their KEY COLORS (e.g., 'Yellow helmet character', 'Pink beaver'). Describe their movement trajectory.\n"
    "<think>\nAnalyze the visual evidence to find the correct answer among the options.\n</think>\n"
    "<answer>\nState the correct option index (0-4) and the full text.\n</answer>"
)

SYSTEM_SUFFIX_NORMAL_COT = (
    "### OUTPUT FORMAT (Chain-of-Thought)\n"
    "You must output XML-like tags in this exact order:\n"
    "<think>\nAnalyze the visual information and conversation context step-by-step to explain how you arrive at the final answer.\n</think>\n"
    "<answer>\nState the correct option index (0-4).\n</answer>"
)

SYSTEM_SUFFIX_VANILLA = (
    "Directly output the final answer based on the video.\n"
    "**STRICT OUTPUT CONSTRAINT**: Do NOT output any thinking. Just the answer."
)

def _build_system_prompt(mode, assistant_name):
    if mode == "ours":
        return SYSTEM_HEADER_SHARED + SYSTEM_SUFFIX_OURS
    elif mode == "normal_cot":
        return SYSTEM_HEADER_SHARED + SYSTEM_SUFFIX_NORMAL_COT
    elif mode == "vanilla":
        return SYSTEM_HEADER_SHARED + SYSTEM_SUFFIX_VANILLA
    return SYSTEM_HEADER_SHARED

def _build_user_prompt_pororo(example, mode="ours") -> str:
    video_id = example.get("video_name", "Unknown")
    question = example.get("question", "")
    
    options = []
    for i in range(5):
        opt = example.get(f"answer{i}")
        if pd.notna(opt):
            options.append(f"({i}) {opt}")
    options_str = "\n".join(options)

    user_name = "Inquirer"
    assistant_name = "Video Expert"

    base_text = (
        f"You are **{assistant_name}**. You are watching multiple clips from {video_id} with the {user_name}.\n"
        f"### INPUT CONTEXT\n"
        f"**{user_name}'s Question**: {question}\n\n"
        f"**Multiple Choice Options**:\n{options_str}\n\n"
    )

    if mode == "ours":
        instruction = (
            "### INSTRUCTION\n"
            "Follow these steps strictly to find the correct answer:\n"
            "1. <vision>: Identify the characters appearing in the clips based on the Character Encyclopedia (e.g., check colors and species). Describe their key actions.\n"
            "2. <think>: Match the visual evidence with the character descriptions and the provided options. Reason through which option is most consistent with the scene.\n"
            "3. <answer>: Output ONLY the correct option index (e.g., '3').\n"
            "Start with <vision>..."
            "FINAL REMINDER: The content between <answer> and </answer> MUST be a single digit (0-4)."
        )
    elif mode == "normal_cot":
        instruction = (
            "### INSTRUCTION\n"
            "Identify the characters using the Encyclopedia and analyze the video clips carefully.\n"
            "1. <think>: Provide your reasoning process, linking visual actions to specific characters.\n"
            "2. <answer>: Output ONLY the correct option index (e.g., '1').\n"
            "Start with <think>..."
            "FINAL REMINDER: The content between <answer> and </answer> MUST be a single digit (0-4)."
        )
    else: # vanilla
        instruction = (
            "### INSTRUCTION\n"
            "Use the visual info and character guide to answer. Output ONLY the correct option index. **NO OTHER TEXT.**"
        )
        
    return base_text + instruction

def get_pororo_video_paths(base_dir, video_name, supporting_num):
    parent_dir = "_".join(video_name.split("_")[:-1])
    folder_path = os.path.join(base_dir, parent_dir, video_name)
    
    try:
        center_idx = int(supporting_num)
    except:
        center_idx = 1
        
    valid_paths = []
    for i in range(center_idx - 8, center_idx + 8):
        if i < 1: continue 
        gif_path = os.path.join(folder_path, f"{i}.gif")
        if os.path.exists(gif_path):
            valid_paths.append(gif_path)
    return valid_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--base_model_path', type=str, default=None)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, default="results.json")
    parser.add_argument('--mode', type=str, default='ours')
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--sample_size', type=int, default=None)
    parser.add_argument('--sample_seed', type=int, default=42)
    args = parser.parse_args()

    video_base_dir = "./Scenes_Dialogues"

    print(f"Loading model on local rank {args.shard_id}...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2", 
        device_map={"": "cuda:0"} 
    )
    model.config.use_cache = True
    model.eval()

    processor_path = args.base_model_path if args.base_model_path else args.model_path
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

    with open(args.input_file, "r", encoding="utf-8") as f:
        full_json = json.load(f)
        data = full_json.get("PororoQA", [])

    total_size = len(data)
    print(f"Total samples: {total_size}", flush=True)

    if args.sample_size is not None and args.sample_size < total_size:
        rng = random.Random(args.sample_seed)
        sampled_indices = rng.sample(range(total_size), args.sample_size)
        sampled_indices.sort()
        selected_data = [data[i] for i in sampled_indices]
    else:
        selected_data = data

    local_data = selected_data[args.shard_id :: args.num_shards]
    actual_output_file = args.output_file.replace(".json", f"_part{args.shard_id}.json")

    final_results = []
    
    for i, item in tqdm(enumerate(local_data), total=len(local_data), desc=f"Shard {args.shard_id}"):
        user_prompt = _build_user_prompt_pororo(item, mode=args.mode)
        sys_prompt = _build_system_prompt(args.mode, "Video Expert")

        gif_paths = get_pororo_video_paths(video_base_dir, item['video_name'], item['supporting_num'])
        if not gif_paths: continue

        all_frames = []
        for g_path in gif_paths:
            gif_frames = get_frames_from_gif(g_path, nframes=2)
            all_frames.extend(gif_frames)
        
        if not all_frames: continue

        messages = [
            {
                "role": "system", 
                "content": [{"type": "text", "text": sys_prompt}]
            },
            {
                "role": "user", 
                "content": [
                    {
                        "type": "video", 
                        "video": all_frames,  
                        "fps": 2.0,        
                    },
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]

        try:
            _, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=1536, 
                    temperature=0.1,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
            
            output_text = processor.batch_decode(
                [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)], 
                skip_special_tokens=True
            )[0].strip()

            res_item = item.copy()
            res_item["model_raw_output"] = output_text
            final_results.append(res_item)

        except Exception as e:
            print(f"Error at shard {args.shard_id}: {e}")

    with open(actual_output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()