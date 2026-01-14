import sys
import os
import json
import re
import argparse
from tqdm import tqdm
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ==================== 1. 路径环境配置 ====================
current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(current_dir, "../r1-v/src/open_r1")
target_dir = os.path.abspath(target_dir)

if target_dir not in sys.path:
    sys.path.append(target_dir)
    print(f"[Setup] Added {target_dir} to sys.path")

try:
    from movie_chara_profiles import resolve_profile, normalize_movie_name
    print("[Setup] Successfully imported 'movie_chara_profiles'.")
except ImportError as e:
    print(f"[Warn] Failed to import 'movie_chara_profiles'. Using mocks.")
    def resolve_profile(movie, name): return {}
    def normalize_movie_name(name): return name

# ==================== 2. Prompt 构建模块 (HEAVILY MODIFIED) ====================

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

# 1. Ours Mode: Vision -> Think -> Answer
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

# 2. CoT Mode: Output only <think> and <answer>. <think> mixes vision and think logic together.
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

# 3. Vanilla Mode: Output only <answer>.
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
        # 你的 Prompt 里有 {assistant_name} 变量，这里需要 format
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
    """构建包含 Shared Header + Mode Specific Instruction 的 User Prompt"""
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
            "**Remember to close all tags.\n**"
            "**CRITICAL FORMATTING RULE**: You MUST close your response with </answer>. Do not stop until you write this tag.\n"
            "Start immediately with <vision>..."         
        )
    
    elif mode == "cot":
        instruction = (
            "### INSTRUCTION\n"
            f"Generate the next line for **{assistant_name}**.\n"
            "(1) **Step 1 <think>...</think> (Analytic Mode)**: Analyze visual information and conversation context to explain the logic step by step. Synthesize the Observable Facts and the Dialogue History together.\n"
            f"(2) **Step 2 <answer>...</answer>**: Reply to {user_name} in character. Apply the **High-Stakes Personality Rule** (e.g., humor turns to bravery in danger).\n\n"
            "**Remember to close all tags.\n**"
            "**CRITICAL FORMATTING RULE**: You MUST close your response with </answer>. Do not stop until you write this tag.\n"
            "Start immediately with <think>..." 
        )
    
    else: # vanilla
        instruction = (
            "### INSTRUCTION\n"
            f"Generate the next line for **{assistant_name}**.\n"
            "**Mental Check**: Internally analyze the **Visual Reality**, **Dialogue Direction**, and **Personality Filter** before speaking. Do not output the analysis.\n"
            f"(1) **<answer>...</answer>**: Reply to {user_name} in character. Apply the **High-Stakes Personality Rule** (e.g., humor turns to bravery in danger).\n\n"
            "**Remember to close the tag.\n**"
            "**CRITICAL FORMATTING RULE**: You MUST close your response with </answer>. Do not stop until you write this tag.\n"
            "Start immediately with <answer>..." 
        )

    return base_text + instruction

def extract_answer_content(text):
    """
    严格提取模式：只接受完整的 <answer>...</answer>
    """
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""
# ==================== 3. 主逻辑 ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Checkpoint Path")
    parser.add_argument('--base_model_path', type=str, default=None, 
                        help="Base Model Path (for Configs/Processor), if checkpoint lacks them")
    parser.add_argument('--input_file', type=str, required=True, help="Input JSON")
    parser.add_argument('--output_file', type=str, default="inference_results.json", help="Output JSON")
    parser.add_argument('--mode', type=str, default='ours', choices=['ours', 'cot', 'vanilla'], help="Prompt Strategy")
    
    parser.add_argument('--num_shards', type=int, default=1, help="总共分成几份")
    parser.add_argument('--shard_id', type=int, default=0, help="当前是第几份 (从0开始)")
    args = parser.parse_args()
    
    video_base_dir = os.path.dirname(os.path.abspath(args.input_file))
    print(f"[Info] Video Base Directory: {video_base_dir}")

    print(f"[Info] Mode: {args.mode.upper()}")
    print(f"[Info] Loading Model from {args.model_path} ...")

    # 1. Load Model (Hugging Face style)
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     args.model_path,
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2",
    #     device_map="auto" 
    # )
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto"
        )
        # 【修改点】强制开启 use_cache 以解决推理慢的问题
        model.config.use_cache = True
        print("[Info] Forced model.config.use_cache = True for inference speed.")
        
    except Exception as e:
        print(f"[Error] Failed to load model weights: {e}")
        return

    # 2. Load Processor
    # try:
    #     processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    # except Exception as e:
    #     print(f"[Error] Failed to load processor: {e}")
    #     return
    
    
    # ========================== 修改点开始 ==========================
    # 逻辑：如果 bash 里传了 base_model_path，就用它。如果没传，就尝试用 model_path。
    # 之前你报错的原因是因为代码里只用了 args.model_path
    
    if args.base_model_path:
        processor_path = args.base_model_path
        print(f"[Info] Base model path provided. Loading Processor from: {processor_path}")
    else:
        processor_path = args.model_path
        print(f"[Info] No Base model path provided. Trying to load Processor from checkpoint: {processor_path}")

    try:
        # 这里使用计算出来的 processor_path，而不是 args.model_path
        processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        print("[Info] Processor loaded successfully.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to load processor from: {processor_path}")
        print(f"Error Details: {e}")
        print(">> Hint: If loading from a checkpoint, make sure to provide '--base_model_path' via the bash script.")
        return
    # ========================== 修改点结束 ==========================

    model.eval()

    # 3. Load Data
    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    local_data = data[args.shard_id :: args.num_shards]
    
    print(f"[Info] Worker {args.shard_id}/{args.num_shards}: Processing {len(local_data)} samples (Total: {len(data)})")
    
    # 修改输出文件名，避免多个进程写同一个文件冲突
    # inference_results.json -> inference_results_part0.json
    name, ext = os.path.splitext(args.output_file)
    actual_output_file = f"{name}_part{args.shard_id}{ext}"
    backup_file = f"{name}_part{args.shard_id}.jsonl"

    print(f"STARTING INFERENCE | Output File: {actual_output_file}")
    
    final_results = []

    # 用于实时写入备份的文件 (JSON Lines 格式)
    with open(backup_file, "w", encoding="utf-8") as f_backup:
        
        # for i, item in tqdm(enumerate(local_data), total=len(local_data), position=args.shard_id, desc=f"Shard {args.shard_id}"):
        for i, item in tqdm(enumerate(local_data), 
                            total=len(local_data), 
                            position=args.shard_id, 
                            desc=f"Shard {args.shard_id}",
                            file=sys.stdout,     # 强制输出到标准输出 (会被重定向到log)
                            disable=False,       # 强制开启进度条，防止被自动关闭
                            mininterval=1.0,
                            ascii=True,    
                            ncols=100):

            # (MODIFIED) 不同模式生成不同的 User Prompt
            user_prompt_text = _build_user_prompt(item, mode=args.mode)
            
            # (MODIFIED) 不同模式，为每个独立的样本生成包含正确 assistant_name 的 System Prompt
            current_assistant = item["assistant"]
            sys_prompt_text = _build_system_prompt(args.mode, current_assistant)

            # 构建 messages
            messages = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt_text}]},
                {"role": "user", "content": []}
            ]
            
            # 处理图像/视频
            data_type = item.get("data_type", "video") # 默认 video
            
            has_media = False
            # if data_type == "video" and "video_path" in item:
            #     if os.path.exists(item["video_path"]):
            #         messages[1]["content"].append({"type": "video", "video": item["video_path"]})
            #         has_media = True
            #     else:
            #         tqdm.write(f"[Warn] Video not found: {item['video_path']}")
            # elif data_type == "image" and "image_path" in item:
            #     if os.path.exists(item["image_path"]):
            #         messages[1]["content"].append({"type": "image", "image": item["image_path"]})
            #         has_media = True
            #     else:
            #         tqdm.write(f"[Warn] Image not found: {item['image_path']}")
            
            if data_type == "video":
                raw_video_path = item.get("video_path")
                if raw_video_path:
                    if not os.path.isabs(raw_video_path):
                        video_path = os.path.join(video_base_dir, raw_video_path)
                        video_path = os.path.normpath(video_path)
                    else:
                        video_path = raw_video_path

                    if os.path.exists(video_path):
                        messages[1]["content"].append({"type": "video", "video": video_path})
                        has_media = True
                    else:
                        tqdm.write(f"[Skipping] Video not found: {video_path} (Raw: {raw_video_path})")
                        continue 
                else:
                     tqdm.write(f"[Skipping] No video_path in item {i}")
                     continue
                    
            elif data_type == "image":
                raw_image_path = item.get("image_path")
                if raw_image_path:
                    if not os.path.isabs(raw_image_path):
                        image_path = os.path.join(video_base_dir, raw_image_path)
                        image_path = os.path.normpath(image_path)
                    else:
                        image_path = raw_image_path
                        
                    if os.path.exists(image_path):
                        messages[1]["content"].append({"type": "image", "image": image_path})
                        has_media = True
                    else:
                        tqdm.write(f"[Skipping] Image not found: {image_path}")
                        continue
                else:
                    tqdm.write(f"[Skipping] No image_path in item {i}")
                    continue
            
            messages[1]["content"].append({"type": "text", "text": user_prompt_text})

            # 准备输入
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            # 生成 (移除 stop_tokens, 依靠 max_new_tokens 或 EOS 自行停止)
            try:
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=2048, # 足够长，让模型自己生成 EOS
                        temperature=0.7, 
                        top_p=0.9,
                        do_sample=True,
                        use_cache=True  # # 【修改点】在 generate 参数中显式确保 use_cache=True 开启 KV Cache 加速(双重保险)
                    )
                
                # 解码
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0].strip()

                # 提取答案
                ans = extract_answer_content(output_text)

                # 记录结果
                res_item = item.copy()
                res_item["model_prediction"] = ans
                res_item["raw_output"] = output_text # 输出原始文本供检查格式

                final_results.append(res_item)

                # 实时写入备份 (JSONLine)
                # f_backup.write(json.dumps(res_item, ensure_ascii=False) + "\n")
                f_backup.write(json.dumps(res_item, indent=4, ensure_ascii=False) + "\n")
                f_backup.flush()

            except Exception as e:
                tqdm.write(f"[Error] Sample {i} failed: {e}")

    # 4. 保存最终 JSON List
    # with open(actual_output_file, "w", encoding="utf-8") as f:
    #     json.dump(final_results, f, indent=4, ensure_ascii=False)    
    with open(actual_output_file, "w", encoding="utf-8") as f:
        json.dump(
            final_results, 
            f, 
            indent=4,               # 【关键】缩进4空格，实现每行一个key
            ensure_ascii=False,     # 【关键】显示中文而不是 \uXXXX
            separators=(',', ': ')  # 去除无关的尾部空格，更美观
        )
    print(f"\n[Success] Inference done. Saved {len(final_results)} results to {actual_output_file}")

if __name__ == "__main__":
    main()