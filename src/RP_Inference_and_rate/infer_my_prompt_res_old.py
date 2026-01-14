import sys
import os
import json
import re
import argparse
from tqdm import tqdm
import torch

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

# def extract_answer_content(text):
#     """提取 <answer> 标签内部的内容, 忽略 vision 和 think"""
#     pattern = r'<answer>\s*(.*?)\s*</answer>'
#     match = re.search(pattern, text, re.DOTALL)
#     if match:
#         return match.group(1).strip()
#     return "" # 提取失败返回空

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
        
    return "" # 提取失败返回空

# ==================== 主推理逻辑 ====================

def run_inference():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to the fine-tuned model checkpoint")
    parser.add_argument('--input_file', type=str, required=True, help="Path to the test dataset JSON file")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the result JSON file")
    args = parser.parse_args()

    # 1. 加载数据
    print(f"[Info] Loading data from {args.input_file}...")
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[Info] Total samples: {len(data)}")

    # 2. 初始化 vLLM 模型
    print(f"[Info] Initializing vLLM model from {args.model_path}...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=16384, 
        gpu_memory_utilization=0.8,
        # limit_mm_per_prompt={"image": 1, "video": 1}, # 如果 vLLM 版本较高(0.6.3+), 可以开启此优化
        trust_remote_code=True,
    )
    
    # 采样参数: 保持一定的 temperature 以避免死板，但要可控
    # sampling_params = SamplingParams(
    #     temperature=0.1,    
    #     top_p=0.9,
    #     max_tokens=2048,    # 留足够空间给 CoT
    #     stop_token_ids=[],
    # )
    
    sampling_params = SamplingParams(
    temperature=0.1,    
    top_p=0.9,
    max_tokens=4096,  # (或者建议改成 4096)
    stop=["<|endoftext|>", "<|im_end|>", "</answer>"], 
    )

    
    # 用于预处理 vision data 的 processor
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # 3. 准备 Inputs
    print("[Info] preparing message inputs...")
    messages_list = []
    
    for item in data:
        prompt_text = _build_roleplay_prompt_text(item)
        data_type = item.get("data_type", "video") # 默认当作 video

        system_msg = {
            "role": "system", 
            "content": [{"type": "text", "text": SYSTEM_PROMPT}]
        }
        
        user_content = []
        # 添加 Vision 输入
        if data_type == "video" and "video_path" in item:
            # vLLM/qwen_utils 需要正确的路径
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
    final_output = []
    print(f"[Info] Starting inference with Batch Size {BSZ}...")

    for i in tqdm(range(0, len(messages_list), BSZ), desc="Inferencing"):
        batch_msgs = messages_list[i : i + BSZ]
        batch_original_data = data[i : i + BSZ]
        
        # 工具函数处理: 转换成 vLLM 支持的 prompt string 和 mm_data
        prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_msgs]
        
        image_inputs, video_inputs, video_kwargs = process_vision_info(batch_msgs, return_video_kwargs=True)

        
        llm_inputs = []
        
        # 重新组装输入
        img_idx = 0
        vid_idx = 0
        
        for idx, prompt in enumerate(prompts):
            sample_mm_data = {}
            # 查看该样本包含什么类型的多模态数据
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
            # 生成
            # import pdb; pdb.set_trace()
            outputs = llm.generate(llm_inputs, sampling_params=sampling_params, use_tqdm=False)
            
            for j, output_item in enumerate(outputs):
                original_item = batch_original_data[j]
                
                
                # import pdb; pdb.set_trace()
                # 获取完整生成的文本 (<vision>...<think>...<answer>...)
                generated_text = output_item.outputs[0].text
                
                # 提取 <answer>
                answer_content = extract_answer_content(generated_text)
                
                # 构建结果 Item
                new_item = original_item.copy()
                new_item["model_prediction"] = answer_content
                
                # 可选: 如果你调试时需要看完整思维链，可以解开下面这行
                # new_item["model_raw_full_response"] = generated_text
                
                final_output.append(new_item)
                
        except Exception as e:
            print(f"[Error] Batch {i} failed: {e}")
            # 填入 Error 占位防止错位
            for j in range(len(batch_msgs)):
                new_item = batch_original_data[j].copy()
                new_item["model_prediction"] = "Error"
                final_output.append(new_item)

    # 5. 保存结果
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"[Info] Saving results to {args.output_file}...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    
    print("[Success] Inference completed.")

if __name__ == "__main__":
    run_inference()



# 运行命令示例:
# python infer_my_prompt_res.py \
#     --model_path /data/wm/Video-R1/your_trained_checkpoint \
#     --input_file /data/wm/Video-R1/data/your_test_data.json \
#     --output_file /data/wm/Video-R1/results/inference_result.json

# # 推荐使用这个命令
# CUDA_VISIBLE_DEVICES=6,7 python infer_my_prompt_res.py \
#     --model_path /你的模型路径 \
#     --input_file /你的输入.json \
#     --output_file /你的输出.json