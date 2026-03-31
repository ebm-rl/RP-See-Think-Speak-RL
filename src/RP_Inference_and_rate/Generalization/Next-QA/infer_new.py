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

SYSTEM_HEADER_SHARED = (
    "You are an expert Role-Play Dialogue AI.\n"
    "Your goal is to immerse yourself in a specific character (the ASSISTANT) and respond to a USER.\n"
    "### THE SHARED REALITY\n"
    "The video represents a Live Event you are witnessing. You must answer questions based on the video evidence.\n"
)

SYSTEM_SUFFIX_OURS = (    
    "### OUTPUT FORMAT (Strict Step-by-Step)\n"
    "You must output XML-like tags in this exact order:\n"
    "<vision>\nDescribe the objective facts and actions in the video.\n</vision>\n"
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

def _build_user_prompt(example, mode="ours") -> str:
    video_id = example.get("video", "Unknown")
    question = example.get("question", "")
    
    options = []
    for i in range(5):
        opt = example.get(f"a{i}")
        if pd.notna(opt):
            options.append(f"({i}) {opt}")
    options_str = "\n".join(options)

    user_name = "Inquirer"
    assistant_name = "Video Expert"

    base_text = (
        f"You are **{assistant_name}**. You are watching video {video_id} with the {user_name}.\n"
        f"### INPUT CONTEXT\n"
        f"**{user_name}'s Question**: {question}\n\n"
        f"**Multiple Choice Options**:\n{options_str}\n\n"
    )

    if mode == "ours":
        instruction = (
            "### INSTRUCTION\n"
            "Follow these steps strictly:\n"
            "1. <vision>: Describe key visual movements and objects briefly.\n"
            "2. <think>: Match visual facts with the provided options. Eliminate wrong ones.\n"
            "3. <answer>:\n"
            "   **Output ONLY the correct option index.** \n"
            "   Example: '2'. \n"
            "   Do not add any preamble or conversational filler.\n"
            "Start with <vision>..."
        )
    elif mode == "normal_cot":
        instruction = (
            "### INSTRUCTION\n"
            "Analyze the video and options carefully.\n"
            "1. <think>: Provide your step-by-step reasoning process. Analyze the visual evidence and evaluate each option.\n"
            "2. <answer>: Output ONLY the correct option index (e.g., '2').\n"
            "Start with <think>..."
        )
    else:
        instruction = (
            "### INSTRUCTION\n"
            "Output ONLY the correct option index and text (e.g., '1'). **NO OTHER TEXT.**"
        )

    return base_text + instruction


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

    video_base_dir = "./NExTQA/NExTVideo"

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

    if args.input_file.endswith('.csv'):
        df = pd.read_csv(args.input_file)
        data = df.to_dict(orient='records')
    else:
        with open(args.input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

    total_size = len(data)
    print(f"Total samples in input file: {total_size}", flush=True)

    if args.sample_size is not None:
        if args.sample_size <= 0:
            raise ValueError("sample_size must be a positive integer")

        if args.sample_size > total_size:
            print(
                f"Warning: sample_size={args.sample_size} is greater than the total number of samples {total_size},"
                f" using all samples.",
                flush=True
            )
            selected_data = data
        else:
            rng = random.Random(args.sample_seed)
            sampled_indices = rng.sample(range(total_size), args.sample_size)
            sampled_indices.sort()
            selected_data = [data[i] for i in sampled_indices]

        print(
            f"Randomly selected {len(selected_data)} samples "
            f"(seed={args.sample_seed}) before sharding.",
            flush=True
        )
    else:
        selected_data = data
        print("No random sampling applied. Using all samples.", flush=True)

    local_data = selected_data[args.shard_id :: args.num_shards]
    print(f"Shard {args.shard_id} will process {len(local_data)} samples.", flush=True)

    actual_output_file = args.output_file.replace(".json", f"_part{args.shard_id}.json")

    final_results = []
    
    for i, item in tqdm(enumerate(local_data), total=len(local_data), desc=f"Shard {args.shard_id}"):
        user_prompt = _build_user_prompt(item, mode=args.mode)
        sys_prompt = _build_system_prompt(args.mode, "Video Expert")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": [
                {
                    "type": "video", 
                    "video": os.path.join(video_base_dir, f"{item['video']}.mp4"), 
                    "nframes": 32,  
                    # "fps": 1.0 
                },
                {"type": "text", "text": user_prompt}
            ]}
        ]

        if not os.path.exists(messages[1]["content"][0]["video"]):
            continue

        inputs = processor(
            text=[processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
            videos=process_vision_info(messages)[1],
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=1024, 
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
            print(f"Error: {e}")

    with open(actual_output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()