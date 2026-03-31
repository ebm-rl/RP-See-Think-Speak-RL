import sys
import os
import json
import re
import glob
import argparse
import random
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(current_dir, "../r1-v/src/open_r1")
target_dir = os.path.abspath(target_dir)
if target_dir not in sys.path:
    sys.path.append(target_dir)

SYSTEM_HEADER_SHARED = (
    "You are an expert Video Question Answering AI.\n"
    "You must answer the user's question only from the visual evidence in the video.\n"
    "### THE SHARED REALITY\n"
    "The video represents a Live Event you are witnessing. You must answer questions based on the video evidence.\n"
)

SYSTEM_SUFFIX_NORMAL_COT = (
    "### OUTPUT FORMAT\n"
    "You must output XML-like tags in this exact order:\n"
    "<think>\n"
    " Analyze the visual information and conversation context step-by-step to explain how you arrive at the final answer.\n"
    "</think>\n"
    "<answer>\n"
    "Output only yes or no.\n"
    "</answer>"
)

SYSTEM_SUFFIX_VANILLA = (
    "### OUTPUT FORMAT\n"
    "Output only one word: yes or no.\n"
    "Do not output any extra text."
)

SYSTEM_SUFFIX_OURS = (    
    "### OUTPUT FORMAT (Strict Step-by-Step)\n"
    "You must output XML-like tags in this exact order:\n"
    "<vision>\nDescribe the objective facts and actions in the video.\n</vision>\n"
    "<think>\nAnalyze the visual evidence to find the correct answer among the options.\n</think>\n"
    "<answer>\nOutput only yes or no.\n</answer>"
)


def _build_system_prompt(mode: str):
    if mode == "ours":
        return SYSTEM_HEADER_SHARED + SYSTEM_SUFFIX_OURS
    elif mode == "normal_cot":
        return SYSTEM_HEADER_SHARED + SYSTEM_SUFFIX_NORMAL_COT
    elif mode == 'vanilla':
        return SYSTEM_HEADER_SHARED + SYSTEM_SUFFIX_VANILLA
    return SYSTEM_HEADER_SHARED


def _build_user_prompt(example, mode='ours') -> str:
    question = example.get('question', '').strip()
    video_name = example.get('video_name', 'Unknown')

    base_text = (
        f"You are watching video {video_name}.\n"
        f"Question: {question}\n\n"
    )

    if mode == "ours":
        instruction = (
            "### INSTRUCTION\n"
            "Follow these steps strictly:\n"
            "1. <vision>: Briefly describe the key visual objects, actions, and movements in the video.\n"
            "2. <think>: Reason step by step based on the visual evidence, and determine whether the answer is yes or no.\n"
            "3. <answer>:\n"
            "   **Output ONLY 'yes' or 'no'.**\n"
            "   Do not add any preamble or conversational filler.\n"
            "Start with <vision>..."
        )
    elif mode == "normal_cot":
        instruction = (
            "### INSTRUCTION\n"
            "Analyze the video carefully.\n"
            "1. <think>: Provide your step-by-step reasoning process based on the visual evidence, and determine whether the answer is yes or no.\n"
            "2. <answer>: Output ONLY 'yes' or 'no'.\n"
            "Start with <think>..."
        )   
    else:
        instruction = (
            "Answer this yes/no question using only the video.\n"
            "First give brief reasoning in <think>, then output only yes or no in <answer>."
        )

    return base_text + instruction

def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_video_path(video_base_dir: str, video_name: str):
    name_variants = [video_name, f"v_{video_name}", f"v__{video_name}"]
    
    for name in name_variants:
        direct_path = os.path.join(video_base_dir, name)
        if os.path.exists(direct_path):
            return direct_path

        for ext in ['.mp4', '.mkv', '.webm', '.avi', '.mov']:
            p = os.path.join(video_base_dir, name + ext)
            if os.path.exists(p):
                return p
            
        matches = glob.glob(os.path.join(video_base_dir, name + '.*'))
        if matches:
            return matches[0]
    return None


def extract_yes_no(output_text: str):
    text = output_text.strip().lower()

    m = re.search(r'<answer>\s*(yes|no)\s*</answer>', text, flags=re.S)
    if m:
        return m.group(1)

    matches = re.findall(r'\b(yes|no)\b', text)
    if matches:
        return matches[-1]

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--base_model_path', type=str, default=None)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, default='results.json')
    parser.add_argument('--video_base_dir', type=str,
                        default='./ActivityNetQA/all_test')
    parser.add_argument('--mode', type=str, default='normal_cot', choices=['normal_cot', 'vanilla', 'ours'])
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--sample_size', type=int, default=None)
    parser.add_argument('--sample_seed', type=int, default=42)
    parser.add_argument('--nframes', type=int, default=32)
    args = parser.parse_args()

    print(f"Loading model on local rank {args.shard_id}...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
        device_map={'': 'cuda:0'}
    )
    model.config.use_cache = True
    model.eval()

    processor_path = args.base_model_path if args.base_model_path else args.model_path
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

    data = load_json(args.input_file)
    total_size = len(data)
    print(f'Total samples in input file: {total_size}', flush=True)

    if args.sample_size is not None:
        if args.sample_size <= 0:
            raise ValueError('sample_size must larger than 0')
        if args.sample_size > total_size:
            print(f'Warning: sample_size={args.sample_size} is greater than the total sample size {total_size}. Therefore, all samples will be used.', flush=True)
            selected_data = data
        else:
            rng = random.Random(args.sample_seed)
            sampled_indices = rng.sample(range(total_size), args.sample_size)
            sampled_indices.sort()
            selected_data = [data[i] for i in sampled_indices]
        print(f'Randomly selected {len(selected_data)} samples (seed={args.sample_seed}) before sharding.', flush=True)
    else:
        selected_data = data
        print('No random sampling applied. Using all samples.', flush=True)

    local_data = selected_data[args.shard_id::args.num_shards]
    print(f'Shard {args.shard_id} will process {len(local_data)} samples.', flush=True)

    actual_output_file = args.output_file.replace('.json', f'_part{args.shard_id}.json')
    final_results = []

    for item in tqdm(local_data, total=len(local_data), desc=f'Shard {args.shard_id}'):
        video_path = find_video_path(args.video_base_dir, item['video_name'])
        if video_path is None:
            print(f"Video not found: {item['video_name']}", flush=True)
            continue

        user_prompt = _build_user_prompt(item, mode=args.mode)
        sys_prompt = _build_system_prompt(args.mode)

        messages = [
            {'role': 'system', 'content': [{'type': 'text', 'text': sys_prompt}]},
            {'role': 'user', 'content': [
                {
                    'type': 'video',
                    'video': video_path,
                    'nframes': args.nframes,
                },
                {'type': 'text', 'text': user_prompt}
            ]}
        ]

        try:
            inputs = processor(
                text=[processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)],
                videos=process_vision_info(messages)[1],
                padding=True,
                return_tensors='pt'
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.1,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )

            output_text = processor.batch_decode(
                [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)],
                skip_special_tokens=True
            )[0].strip()

            pred_answer = extract_yes_no(output_text)

            res_item = item.copy()
            res_item['video_path'] = video_path
            res_item['model_raw_output'] = output_text
            res_item['pred_answer'] = pred_answer
            res_item['is_correct'] = (pred_answer == str(item.get('answer', '')).strip().lower()) if pred_answer is not None else False
            final_results.append(res_item)

        except Exception as e:
            print(f"Error on {item.get('question_id', 'unknown')}: {e}", flush=True)

    with open(actual_output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f'Saved shard result to: {actual_output_file}', flush=True)


if __name__ == '__main__':
    main()
