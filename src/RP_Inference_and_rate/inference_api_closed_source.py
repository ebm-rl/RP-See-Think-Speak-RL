import sys
import os
import json
import base64
import cv2
import argparse
import time
import threading
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm
import random

# ================= 0. 路径环境配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
# /src/RP_Inference -> /src/r1-v/src/open_r1
target_dir = os.path.join(current_dir, "../r1-v/src/open_r1")
target_dir = os.path.abspath(target_dir)

if target_dir not in sys.path:
    sys.path.append(target_dir)

try:
    from movie_chara_profiles import resolve_profile, normalize_movie_name
    print("[Setup] Successfully imported 'movie_chara_profiles'.")
except ImportError as e:
    print(f"[Warn] Failed to import 'movie_chara_profiles'. Using mocks.")
    def resolve_profile(movie, name): return {}
    def normalize_movie_name(name): return name

# ================= 1. 配置区域 (请在此修改API信息) =================

# API 配置 (Gemini / Claude / GPT)
API_KEY = "sk-live-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJNZXRhQ2hhdCIsInN1YiI6IjY5MWJkODI2M2E1NGE3MDkxMmU3OTRmOSIsImNsaWVudF9pZCI6ImEwYWZkNjllOTYzNzJmMzUwMGEzMmQzMDRjODMyOTYxIiwic2NvcGUiOiJtaWRqb3VybmV5IGFnZW50IiwiaWF0IjoxNzYzNDMyNjAwfQ.WZZB02qWSGz1HwXiZRDjMS9iLhpE-gkOHsykh8pFGw4" 
BASE_URL = "https://llm-api.mmchat.xyz/v1"
MODEL_NAME = "gpt-5.2"

# 视频处理与并发配置
MAX_NUM_FRAMES = 8       # 抽帧数量，保持与开源模型训练/推理时的一致性
FRAME_RESIZE = 512       # 图片缩放尺寸
MAX_RETRIES = 3          # API 重试次数
RETRY_DELAY = 2          # 重试延迟 (秒)
MAX_WORKERS = 1          # <--- 并发线程数，API通常IO密集，可设置较高 (如 8-20)

# ================= 2. Prompt 构建模块 (与开源推理脚本保持完全一致) =================

# --- A. System Prompt 共享部分 ---
SYSTEM_HEADER_SHARED = (
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
)

# --- B. System Prompt 差异化部分 ---

# 1. Ours Mode
SYSTEM_SUFFIX_OURS = (
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
    "   - Internalize {assistant_name}'s mindset. Apply the **THE PERSONALITY ANALYSIS** (e.g., humor turns to bravery in danger) while thinking as the assistant. Based on the current situation and the topic of the conversation, and considering the personality and tone of the assistant, think about possible directions and responses for the assistant's next sentence.\n"   
    "   - *Check*: If the Vision is dangerous, does the character show bravery/nervousness instead of casual traits?\n"
    "</think>\n"
    "\n"
    "<answer>\n"
    "The final natural spoken line by the Assistant. No speaker name. No quotes.\n"
    "</answer>"
)

# 2. CoT Mode
SYSTEM_SUFFIX_COT = (
    "### OUTPUT FORMAT (Chain-of-Thought)\n"
    " First analyze visual information and conversation context to explain the logic step by step, then output the prediction."
    "\n"
    "### Thinking Hint:\n"
    "** You CAN refer to the following aspects while thinking: \n"
    "1. **Analyze Vision**: Based on the input visual information and the overall visual atmosphere, analyze what is happening at present? (e.g., 'A warm conversation', 'A dangerous battle').\n"
    "2. **Analyze Dialogue History**: Look at the CONTEXT of the conversation so far (User + Assistant):\n"
    "   - Are they **continuing the specific topic** from the video? (Direction A)\n"
    "   - Are they **discussing the aftermath/feelings** of the event? (Direction B)\n"
    "3. **Determine Topic**: Combine [Visual Information] + [Dialogue Direction] to define the current topic.\n"
    "4. **Drafting (Personality Filter)**: \n"
    "   - Internalize {assistant_name}'s mindset. Apply the **High-Stakes Personality Rule** (e.g., humor turns to bravery in danger) while thinking as the assistant. Based on the current situation and the topic of the conversation, and considering the personality and tone of the assistant, think about possible directions and responses for the assistant's next sentence.\n"   
    "   - *Check*: If the Vision is dangerous, does the character show bravery/nervousness instead of casual traits?\n"
    "\n"
    "<think>\n"
    " Put your thinking process here.\n"
    " Analyze step by step to explain how you arrive at the final answer.\n"
    " Synthesize the Observable Facts and the Dialogue History together, such as: \n"
    " 1. Start by analyzing the visual events...\n"
    " 2. Then review the conversation history...\n"
    " 3. Combine these insights to determine the current topic...\n"
    " 4. Finally, determine the response strategy considering the personality and tone of the assistant...\n"
    "</think>\n"
    "\n"
    "<answer>\n"
    "The final natural spoken line by the Assistant. No speaker name. No quotes.\n"
    "</answer>"
)

# 3. Vanilla Mode
SYSTEM_SUFFIX_VANILLA = (
    "Based on the video and dialogue, **immediately** generate the response.\n"
    "### Thinking Hint:\n"
    "**Internal Processing Requirement**: Before generating the final answer, you CAN internally process the following steps (Do NOT output them):\n"
    "1. **Analyze Vision**: Based on the input visual information and the overall visual atmosphere, assess internally what is happening at present? (e.g., 'A warm conversation', 'A dangerous battle').\n"
    "2. **Analyze Dialogue History**: Look at the CONTEXT of the conversation so far (User + Assistant), and check internally if the topic is continuing or reacting.\n"
    "3. **Determine Topic**: Combine vision and context to define the current topic.\n"
    "4. **Drafting (Personality Filter)**: Internalize {assistant_name}'s mindset. Apply the **High-Stakes Personality Rule** (e.g., humor turns to bravery in danger) while thinking as the assistant. Based on the current situation and the topic of the conversation, and considering the personality and tone of the assistant, think about possible directions and responses for the assistant's next sentence.\n"  
    "\n"
    "**STRICT OUTPUT CONSTRAINT**: \n"
    "- Do NOT output any thinking process.\n"
    "- Output ONLY the final natural spoken line by the Assistant inside <answer>...</answer>.\n"
    "- No speaker name. No quotes."
)

def _build_system_prompt(mode, assistant_name):
    """根据模式组合 System Prompt"""
    if mode == "ours":
        return SYSTEM_HEADER_SHARED + SYSTEM_SUFFIX_OURS.format(assistant_name=assistant_name)
    elif mode == "cot":
        return SYSTEM_HEADER_SHARED + SYSTEM_SUFFIX_COT.format(assistant_name=assistant_name)
    elif mode == "vanilla":
        return SYSTEM_HEADER_SHARED + SYSTEM_SUFFIX_VANILLA.format(assistant_name=assistant_name)
    else:
        raise ValueError(f"Unknown mode: {mode}")

# --- C. User Prompt 差异化构建 ---

def _format_character_block(movie, name):
    profile = resolve_profile(movie, name)
    lines = [f"Profile:"]
    if not profile or not isinstance(profile, dict):
        return "(No profile)"
    for k, v in profile.items():
        if v:
            lines.append(f"   - {k}: {v}")
    return "\n".join(lines)

def _format_dialogue_block(dialogue_list):
    lines = []
    for turn in dialogue_list:
        spk = turn.get("speaker", "Unknown")
        utt = turn.get("utterance", "")
        lines.append(f"{spk}: {utt}")
    return "\n".join(lines)

def _build_user_prompt(example, mode="ours") -> str:
    """构建 User Prompt"""
    movie = normalize_movie_name(example.get("data_source", "Unknown"))
    user_name = example["user"]
    assistant_name = example["assistant"]
    
    user_block = _format_character_block(movie, user_name)
    assistant_block = _format_character_block(movie, assistant_name)
    dialogue_block = _format_dialogue_block(example["dialogue"])

    # 1. 共享的 Context 部分
    base_text = (
        f"You are role-playing as **{assistant_name}** from the universe of '{movie}'.\n"
        "**Context**: You are physically present in the scene, witnessing/experiencing the events shown below.\n\n"
        "### CHARACTER PROFILES\n"
        f"USER ({user_name}):\n{user_block}\n\n"
        f"ASSISTANT ({assistant_name}):\n{assistant_block}\n\n"
        "### INPUT CONTEXT\n"
        "**[Visual Reality]**: The raw recording of the immediate event. Contains Actions, Objects, and Atmosphere.\n"
        "**[Dialogue Context]**:\n"
        f"{dialogue_block}\n"
    )

    # 2. 差异化的 Instruction 部分
    if mode == "ours":
        instruction = (
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
            "**Remember to close all tags.** Start immediately with <vision>..."         
        )
    
    elif mode == "cot":
        instruction = (
            "### INSTRUCTION\n"
            f"Generate the next line for **{assistant_name}**.\n"
            "(1) **Step 1 <think>...</think> (Analytic Mode)**: Analyze visual information and conversation context to explain the logic step by step. Synthesize the Observable Facts and the Dialogue History together.\n"
            f"(2) **Step 2 <answer>...</answer>**: Reply to {user_name} in character. Apply the **High-Stakes Personality Rule** (e.g., humor turns to bravery in danger).\n\n"
            "**Remember to close all tags.** Start immediately with <think>..."      
        )
    
    else: # vanilla
        instruction = (
            "### INSTRUCTION\n"
            f"Generate the next line for **{assistant_name}**.\n"
            "**Mental Check**: Internally analyze the **Visual Reality**, **Dialogue Direction**, and **Personality Filter** before speaking. Do not output the analysis.\n"
            f"(1) **<answer>...</answer>**: Reply to {user_name} in character. Apply the **High-Stakes Personality Rule** (e.g., humor turns to bravery in danger).\n\n"
            "**Remember to close the tag.** Start immediately with <answer>..."
        )

    return base_text + instruction

def extract_answer_content(text):
    """
    严格提取模式：只接受完整的 <answer>...</answer>
    """
    if not text: return ""
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

# ================= 3. 视频处理 & API 工具 =================

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def process_video_frames(video_path):
    if not os.path.exists(video_path):
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): 
        return None

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 策略优化：
        # 1. 如果视频帧数很少 (<= 8)，直接提取所有帧
        if 0 < total_frames <= MAX_NUM_FRAMES:
            indices = list(range(total_frames))
        # 2. 如果是长视频，进行均匀抽样
        elif total_frames > MAX_NUM_FRAMES:
            indices = [int(i * total_frames / MAX_NUM_FRAMES) for i in range(MAX_NUM_FRAMES)]
            indices = sorted(list(set(indices))) # 去重并排序，提高效率
        # 3. 如果读取不到帧数 (total_frames<=0)，则放弃该视频 (或者你可以选择盲读，这里保持放弃)
        else:
            return None

        frames_base64 = []
        current_frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: 
                break
            
            # 判断当前帧是否需要保留
            if current_frame_idx in indices:
                h, w = frame.shape[:2]
                scale = FRAME_RESIZE / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame_resized = cv2.resize(frame, (new_w, new_h))
                _, buffer = cv2.imencode(".jpg", frame_resized)
                b64_str = base64.b64encode(buffer).decode("utf-8")
                frames_base64.append(b64_str)
            
            current_frame_idx += 1
            
            # 如果已经读完了我们需要的所有帧，提前结束循环，节省时间
            if current_frame_idx > indices[-1]:
                break
                
        # 只要读到了至少一帧，就算成功
        if len(frames_base64) > 0:
            return frames_base64
        else:
            return None

    finally:
        cap.release()


def call_llm(system_prompt, user_prompt, video_frames):
    user_content = []
    
    # 闭源模型通常在这里传入图片/视频帧
    if video_frames:
        for b64 in video_frames:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
            })
            
    user_content.append({"type": "text", "text": user_prompt})
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=messages,
                temperature=0.7, # 保持与开源模型一致
                max_tokens=2048, # 保持与开源模型生成长度一致
                timeout=180
            )
            content = response.choices[0].message.content
            # Cleanup potential Markdown block wrappers
            if "```xml" in content: content = content.split("```xml")[1].split("```")[0].strip()
            elif "```" in content: content = content.split("```")[1].split("```")[0].strip()
            return content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return None

# ================= 4. 处理核心 (Worker) =================

def process_item(item, output_lock, output_file, error_lock, error_file, mode="ours", video_base_dir=""):
    pid = item.get("problem_id", str(item.get("id", "unknown")))
    raw_video_path = item.get("video_path", "")
    
    if raw_video_path and not os.path.isabs(raw_video_path):
        video_path = os.path.join(video_base_dir, raw_video_path)
        video_path = os.path.normpath(video_path)
    else:
        video_path = raw_video_path
        
    # 视频抽帧
    frames = process_video_frames(video_path)
    
    # 定义错误日志函数
    def log_error(reason, raw_output=None):
        error_entry = item.copy()
        error_entry["error_reason"] = reason
        error_entry["resolved_video_path"] = video_path
        if raw_output:
            error_entry["raw_llm_output"] = raw_output
        json_str = json.dumps(error_entry, ensure_ascii=False, indent=2)
        with error_lock:
            with open(error_file, "a", encoding="utf-8") as f:
                f.write(json_str + ",\n")

    # =========================================================================
    # <--- 【修改点】：闭源模型 - 严格检查视频读取结果 (frames)
    # 只有当 frames 非空（成功读取到内容）才继续，否则认为是错误数据直接Drop
    # =========================================================================
    if not frames: 
        # frames 为 None (路径不存在) 或 Empty List (读取失败)
        # 记录到 Error log 并返回 False (Drop Sample)
        # print(f"[Skip] ID {pid}: Video not found or empty.") # 可选打印
        log_error("Video read failed: invalid path or corrupted file")
        return False
    # =========================================================================

    # 1. 构建 Prompt
    current_assistant = item["assistant"]
    sys_prompt_text = _build_system_prompt(mode, current_assistant)
    user_prompt_text = _build_user_prompt(item, mode=mode)
    
    # 2. 调用 API
    # import pdb; pdb.set_trace()
    llm_output = call_llm(sys_prompt_text, user_prompt_text, frames)
    
    if llm_output:
        # 3. 提取答案
        ans = extract_answer_content(llm_output)
        
        # 结果构建
        res_item = item.copy()
        res_item["model_prediction"] = ans
        res_item["raw_output"] = llm_output
        
        # 4. 写入成功结果
        json_str = json.dumps(res_item, ensure_ascii=False, indent=2)
        with output_lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json_str + ",\n")
        return True
    else:
        # API 彻底失败
        log_error("API Call Failed (Network/RateLimit/Safety)")
        return False

# ================= 5. 主程序入口 =================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input data json path")
    parser.add_argument("--output_file", type=str, required=True, help="Result json path")
    parser.add_argument("--error_file", type=str, default=None, help="Failed log path")
    parser.add_argument('--mode', type=str, default='ours', choices=['ours', 'cot', 'vanilla'], help="Prompt Strategy")
    args = parser.parse_args()

    if not args.error_file:
        base, ext = os.path.splitext(args.output_file)
        args.error_file = f"{base}_errors{ext}"
        
    video_base_dir = os.path.dirname(os.path.abspath(args.input_file))

    print(f"[Info] Mode: {args.mode.upper()}")
    print(f"[Info] Model: {MODEL_NAME}")
    print(f"[Info] Input: {args.input_file}")
    print(f"[Info] Video Base Dir: {video_base_dir}")
    
    with open(args.input_file, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    
    # 随机乱序以平衡各种电影的请求负载
    random.shuffle(data_list)
    print(f"[Info] Total samples: {len(data_list)}")

    # 初始化文件 (JSON Array Start)
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write("[\n") 
    with open(args.error_file, "w", encoding="utf-8") as f:
        f.write("[\n") 

    output_lock = threading.Lock()
    error_lock = threading.Lock()
    success_count = 0
    
    print(f"[Info] Starting tasks with MAX_WORKERS={MAX_WORKERS}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_item, item, output_lock, args.output_file, error_lock, args.error_file, args.mode, video_base_dir) 
            for item in data_list
        ]
        
        for f in tqdm(as_completed(futures), total=len(data_list), desc=f"Inference ({args.mode})"):
            if f.result():
                success_count += 1
    
    
    # # 方案 B: 单线程 Debug 模式
    # print("!!! DEBUG MODE: Running sequentially in main thread !!!")
    # import pdb 
    
    # for i, item in enumerate(data_list):
    #     print(f"Processing item {i}...")
        
    #     # 可选：在这里设置断点，或者在 process_item 内部设置
    #     # pdb.set_trace() 
        
    #     # 直接调用函数
    #     result = process_item(item, output_lock, args.output_file, error_lock, args.error_file, args.mode, video_base_dir)
        
    #     if result:
    #         success_count += 1
            
    #     # 可选：只跑前1个就退出，方便快速验证
    #     # if i >= 0: break 
    
    # ============================================================

    # 修复文件结尾 (JSON Array End)
    for filepath in [args.output_file, args.error_file]:
        if os.path.exists(filepath):
            with open(filepath, "rb+") as f:
                try:
                    f.seek(-2, os.SEEK_END)
                    if f.read() == b",\n":
                        f.seek(-2, os.SEEK_END)
                        f.truncate()
                except OSError: pass 
                f.write(b"\n]")

    print(f"\n[Done] Success: {success_count}/{len(data_list)}")
    print(f"[Done] Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()



# # 使用示例：
# python /data/wm/Video-R1/src/RP_Inference/inference_api_closed_source.py \
#     --input_file /data/wm/Video-R1/data/test_data.json \
#     --output_file ./results/result_api_gemini_ours.json \
#     --mode ours  choices=['ours', 'cot', 'vanilla']
