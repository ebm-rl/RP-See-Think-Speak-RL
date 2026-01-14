import sys
import os
import json
import re
import argparse
from tqdm import tqdm
import torch
import torch.distributed as dist

# ==================== 1. 路径修复 (最优先执行) ====================
# 获取当前脚本绝对路径: /data/wm/Video-R1/src/Myself_Inference
current_dir = os.path.dirname(os.path.abspath(__file__))

# 计算目标目录: 往上两级找到 src, 再进入 r1-v/src/open_r1
# 相对路径: ../r1-v/src/open_r1
target_dir = os.path.join(current_dir, "../r1-v/src/open_r1")
target_dir = os.path.abspath(target_dir)

# 将其加入 Python 搜索路径
if target_dir not in sys.path:
    sys.path.append(target_dir)
    print(f"[Setup] Added {target_dir} to sys.path")

try:
    from movie_chara_profiles import resolve_profile, normalize_movie_name
    print("[Setup] Successfully imported 'movie_chara_profiles'.")
except ImportError as e:
    print(f"[Error] Failed to import 'movie_chara_profiles': {e}")
    print(f"[Error] Current sys.path: {sys.path}")
    # Mock 函数防止程序直接崩溃，但建议检查路径是否正确
    def resolve_profile(movie, name): return {}
    def normalize_movie_name(name): return name
# ================================================================

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# ==================== 配置区域 ====================
BSZ = 8  # 根据显存调整 Batch Size

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
    "   - Generate the response. Apply the **High-Stakes Rule** defined in Section 3.\n"
    "   - *Check*: If the Vision is dangerous, does the character show bravery/nervousness instead of casual traits?\n"
    "</think>\n"
    "\n"
    "<answer>\n"
    "The final natural spoken line by the Assistant. No speaker name. No quotes.\n"
    "</answer>"
)


# ==================== 辅助函数 (Prompt 构建) ====================

def _format_character_block(movie: str, name: str) -> str:
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
        "1. **Step 1 <vision>...</vision> (Camera Mode)**: Strictly describe **observable facts** ONLY inside these tags. \n"
        "   - List specific physical actions and visible expressions.\n"
        "   - Identify key objects.\n"
        "   - **NO ANALYSIS OR GUESSING HERE.** Just report what is seen.\n"
        "2. **Step 2 <think>...</think> (Analytic Mode)**: Decode the event inside these tags.\n"
        "   - **Analyze the Clues**: Use the *Key Objects* and *Action Intensity* from Vision to determine **what event is exactly happening that see just now**.\n"
        "   - **Synthesize with Dialogue**: Combine this Event Analysis with the [Dialogue History] to define the precise **Topic**.\n"
        "3. **Step 3 <answer>...</answer>**: Reply to {user_name} in character. Apply the **High-Stakes Personality Rule** (e.g., humor turns to bravery in danger).\n\n"
        "**Remember to close all tags.** Start immediately with <vision>..."
    )
    return text

def extract_answer_content(text):
    """提取 <answer> 标签内部的内容"""

    # 尝试匹配完整的 <answer>...</answer>
    pattern_full = r'<answer>\s*(.*?)\s*</answer>'
    match = re.search(pattern_full, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # 如果没找到闭合标签，尝试匹配 <answer>... 到底
    # 这专门用来应对 include_stop_token_in_output=False 的情况
    pattern_partial = r'<answer>\s*(.*)'
    match = re.search(pattern_partial, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return ""  # 提取失败返回空

# ==================== 主推理逻辑 ====================

def _init_dist_if_needed():
    """只负责并行方式初始化：torchrun 下启用 Data Parallel；普通 python 下就是单卡。"""
    use_dist = ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ)
    if use_dist and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)  # 每个进程绑定一张可见 GPU（CUDA_VISIBLE_DEVICES 内部编号 0/1）
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return True, rank, world_size
    return False, 0, 1

def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to the fine-tuned model checkpoint")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the test dataset JSON file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the result JSON file")
    args = parser.parse_args()

    # ===== Data Parallel 初始化（只改并行方式）=====
    use_dist, rank, world_size = _init_dist_if_needed()

    # 1. 加载数据（每个进程都读一遍，逻辑不动；只是后面按 rank 切分）
    if rank == 0:
        print(f"[Info] Loading data from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if rank == 0:
        print(f"[Info] Total samples: {len(data)}")

    # ===== 按 rank 切分数据：Data Parallel 的关键 =====
    if world_size > 1:
        shard_indices = list(range(rank, len(data), world_size))
        data_shard = [data[i] for i in shard_indices]
    else:
        shard_indices = list(range(len(data)))
        data_shard = data

    # 2. 初始化 vLLM 模型
    if rank == 0:
        print(f"[Info] Initializing vLLM model from {args.model_path}...")
    llm = LLM(
        model=args.model_path,
        # ===== Data Parallel：每个进程只用 1 张 GPU，关闭 TP =====
        tensor_parallel_size=1,
        max_model_len=16384,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=4096,
        stop=["<|endoftext|>", "<|im_end|>", "</answer>"],
    )

    # 用于预处理 vision data 的 processor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # 3. 准备 Inputs
    if rank == 0:
        print("[Info] preparing message inputs...")
    messages_list = []

    for item in data_shard:
        prompt_text = _build_roleplay_prompt_text(item)
        data_type = item.get("data_type", "video")  # 默认当作 video

        system_msg = {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        }

        user_content = []
        # 添加 Vision 输入
        if data_type == "video" and "video_path" in item:
            user_content.append({"type": "video", "video": item["video_path"]})
        elif data_type == "image" and "image_path" in item:
            user_content.append({"type": "image", "image": item["image_path"]})

        # 添加 Text Prompt
        user_content.append({"type": "text", "text": prompt_text})

        user_msg = {
            "role": "user",
            "content": user_content
        }

        messages_list.append([system_msg, user_msg])

    # 4. 批量推理
    final_output_indexed = []  # 存 (global_idx, item)，最后合并还原顺序
    if rank == 0:
        print(f"[Info] Starting inference with Batch Size {BSZ}...")

    for i in tqdm(
        range(0, len(messages_list), BSZ),
        desc="Inferencing",
        disable=(world_size > 1 and rank != 0),  # 只让 rank0 打进度条
    ):
        batch_msgs = messages_list[i: i + BSZ]
        batch_original_data = data_shard[i: i + BSZ]
        batch_global_indices = shard_indices[i: i + BSZ]

        # 工具函数处理: 转换成 vLLM 支持的 prompt string 和 mm_data
        prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_msgs]

        image_inputs, video_inputs, video_kwargs = process_vision_info(batch_msgs, return_video_kwargs=True)

        llm_inputs = []

        # 重新组装输入
        img_idx = 0
        vid_idx = 0

        for idx, prompt in enumerate(prompts):
            sample_mm_data = {}
            msg_content = batch_msgs[idx][1]['content']
            has_video = any(x['type'] == 'video' for x in msg_content)
            has_image = any(x['type'] == 'image' for x in msg_content)

            if has_video:
                sample_mm_data["video"] = video_inputs[vid_idx]
                vid_idx += 1
            elif has_image:
                sample_mm_data["image"] = image_inputs[img_idx]
                img_idx += 1

            llm_inputs.append({
                "prompt": prompt,
                "multi_modal_data": sample_mm_data,
            })

        try:
            outputs = llm.generate(llm_inputs, sampling_params=sampling_params, use_tqdm=False)

            for j, output_item in enumerate(outputs):
                original_item = batch_original_data[j]
                generated_text = output_item.outputs[0].text
                # import pdb; pdb.set_trace()  # 现在的这版代码调试不能用pdb
                answer_content = extract_answer_content(generated_text)

                new_item = original_item.copy()
                new_item["model_prediction"] = answer_content

                final_output_indexed.append((batch_global_indices[j], new_item))

        except Exception as e:
            if rank == 0:
                print(f"[Error] Batch {i} failed: {e}")
            for j in range(len(batch_msgs)):
                new_item = batch_original_data[j].copy()
                new_item["model_prediction"] = "Error"
                final_output_indexed.append((batch_global_indices[j], new_item))

    # 5. 保存结果：DP 下只让 rank0 合并写文件
    if world_size > 1:
        gathered = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(final_output_indexed, gathered, dst=0)
        dist.barrier()

        if rank == 0:
            merged = []
            for part in gathered:
                merged.extend(part)
            merged.sort(key=lambda x: x[0])  # 还原原始顺序
            final_output = [it for _, it in merged]

            output_dir = os.path.dirname(args.output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            print(f"[Info] Saving results to {args.output_file}...")
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=4, ensure_ascii=False)

            print("[Success] Inference completed.")

        dist.destroy_process_group()
    else:
        final_output_indexed.sort(key=lambda x: x[0])
        final_output = [it for _, it in final_output_indexed]

        output_dir = os.path.dirname(args.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print(f"[Info] Saving results to {args.output_file}...")
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)

        print("[Success] Inference completed.")

if __name__ == "__main__":
    run_inference()


# --nproc_per_node=2 参数表示用几张卡

# CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \
#   /data/wm/Video-R1/src/Myself_Inference/infer_my_prompt_res.py \
#   --model_path /data/wm/Video-R1/Qwen2.5-VL-7B-Instruct \
#   --input_file /data/wm/simple-subtitling/Processed_Dialogue/RP-RL-Dataset/test_api_and_raw.json \
#   --output_file /data/wm/Video-R1/src/Myself_Inference/Inference_result/qwen_my_cot.json
