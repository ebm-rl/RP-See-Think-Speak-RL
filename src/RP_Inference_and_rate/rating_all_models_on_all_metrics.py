import sys
import os
import json
import base64
import cv2
import argparse
import time
import threading
import re
import random
import copy
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 0. 路径环境配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
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

# ================= 1. 配置区域 =================

API_KEY = "xxx" 
BASE_URL = "xxx" 
MODEL_NAME = "gpt-4o"   # 调用的评测模型

# 视频与并发配置
MAX_NUM_FRAMES = 8       
FRAME_RESIZE = 512       
MAX_JUDGE_RETRIES = 3    # Judge 漏评时的最大重试次数
RETRY_DELAY = 1          
MAX_WORKERS = 8 

# Judge System Prompt
JUDGE_SYSTEM_PROMPT = (
    "You are an expert Role-Play Evaluation Critic. "
    "Your task is to score multiple AI models' responses based on specific metrics. "
    "Maintain absolute objectivity."
)

# ================= 2. 评测 Prompt 模板 =================

COMMON_PROFILE_INSTRUCTION = (
    "**Note on Profiles**: Two profiles are provided below.\n"
    "- **[USER INFO]** is provided ONLY to help you understand the relationship and background of the conversation.\n"
    "- **[ASSISTANT INFO]** is the Target Persona. Your evaluation must focus on how well the model portrays this specific Assistant character."
)

FORCE_COMPLETE_INSTRUCTION = (
    "\n\n**CRITICAL REQUIREMENT**:\n"
    "You provided {num_models} candidate responses (labeled as [[ModelName]]). \n"
    "You **MUST** output a score block for **EVERY SINGLE MODEL** provided. \n"
    "Do not miss any model. Double check before outputting."
)

PROMPT_TEMPLATES = {
    "character_fidelity": """
Attention:
You are a strict evaluator. You have received {num_models} candidate responses.
You MUST generate a scoring block for ALL {num_models} models. Do not stop until you satisfy the count.
Task: Rate the **Character Fidelity** of {assistant_name}'s performance.

**Important Notes (Read Carefully):**
1) The Ground Truth is for reference only. Do NOT score based on matching the Ground Truth wording.
2) This is an **immersive role-playing** task: the assistant must sound like they are **in-world**, speaking naturally in the moment.
3) **High-Stakes Personality Rule (Crucial):**
   - In dangerous/tragic/tense scenes, the character may **modulate** their usual traits.
   - Humor is OPTIONAL; it can become nervous, restrained, brave, or disappear briefly.
   - Do NOT penalize reduced humor if it is consistent with the character and the stakes.

{profile_note}

**Profile Context**:
{context}

**Conversation History**:
{conversation}

**Real Answer (Ground Truth)**:
{gt_answer}

**Candidate Responses to Evaluate**:
{model_responses_block}

**Evaluation Criteria (0-100)**:
Score based on whether the response embodies {assistant_name} in terms of **voice + values + social stance + in-world plausibility**.

Judge these dimensions:
1) **Voice & Speech Style (Signature)**
   - Does it match the character’s vocabulary, cadence, formality, catchphrases/verbal tics (if any)?
   - Avoids generic “assistant-like” tone unless that is the character.

2) **Personality & Decision Consistency (with Stakes Modulation)**
   - Are emotions/values/attitude consistent with the profile and the conversation context?
   - High-stakes scenes may legitimately shift the surface tone (e.g., joking -> urgent bravery), but the “inner self” should still feel like the same person.

3) **Relationship & Status Appropriateness**
   - Does the attitude toward the User fit their relationship/status in the context?
   - Avoids sudden intimacy, hostility, or submissiveness not supported by the profile/history.

4) **Grounding vs Hallucination (Balanced for Role-Play)**
   - **Hard Constraint (must not violate):** Do NOT contradict explicit facts in Profile or Conversation History.
   - **Soft Allowance (role-play realism):** Minor in-character color/phrasing is allowed if it does NOT introduce new decisive facts (new events, new backstory, new relationships) that change the reality.
   - “Hallucination” here means: inventing key facts that materially alter who/what/why, or contradicting the provided data.

Key Rules (VERY IMPORTANT):
1) **Dialogue-Only Constraint**: The response MUST be first-person spoken dialogue. Heavy penalties for narration/stage directions/meta (e.g., "*sighs*", "She said...", "As an AI...").
2) **In-World Only**: No mention of "video", "camera", "scene", "script", "prompt", "evaluation".

Scoring anchors:
- **0-20 (Severe Break / Contradiction)**: Not {assistant_name} at all, breaks immersion, heavy meta/narration, or contradicts core profile/history.
- **21-40 (Poor)**: Some role-play intent but mostly generic; wrong social stance; major unsupported claims.
- **41-60 (Moderate)**: Generally in character but lacks signature voice; minor inconsistencies; weak emotional fit.
- **61-80 (Good)**: Strong character voice and appropriate stance; stakes modulation feels believable; no major hallucination.
- **81-100 (Excellent)**: Distinct, nuanced, and fully in-character; subtle emotional subtext; stakes-aware modulation; perfectly grounded.

**Output Format**:
For EACH model, output a block strictly following this format:
[[Model Name]]
Score: <Number 0-100>
Reason: <Short explanation focusing on voice + personality consistency + grounding>
{force_complete_msg}
""",

    "video_text_relevance": """
Attention:
You are a strict evaluator for **Immersive Scene-Grounded Role-Play Dialogue**.
You will be shown VIDEO FRAMES and multiple MODEL RESPONSES.

Task: Rate the **Video-Text Relevance** of {assistant_name}'s performance — i.e., how strongly the response feels
**situated in the same scene and stakes**, like a person physically present nearby.

**Important Notes (Read Carefully):**
1) Do NOT score based on similarity to the Ground Truth. Use it only as weak reference for conversational continuity.
2) **Temporal Flexibility (Crucial for this dataset):**
   - The dialogue may occur **immediately after** the captured video moment.
   - Therefore, do NOT require the response to explicitly mention exact on-screen objects/actions.
   - The video is primarily an anchor for **atmosphere, stakes, emotional pressure, and immediate situation**, not a strict transcript.
3) Identity Separation:
   - The visible figures may or may not be the speakers.
   - Do NOT penalize the response for avoiding forced identity binding.
4) High-Stakes Atmosphere Rule:
   - If the scene is dangerous/tragic/tense, tone should naturally shift (urgent, cautious, restrained, gentle).
   - Humor is OPTIONAL; do NOT penalize reduced humor in high-stakes scenes.

{profile_note}

**Profile Context**:
{context}

**Conversation History**:
{conversation}

**Real Answer (Ground Truth)**:
{gt_answer}

**Candidate Responses to Evaluate**:
{model_responses_block}

What to judge (core of this metric):
A) **Atmosphere & Stakes Alignment (Primary)**
   - Does the line match the visible mood (safe vs dangerous; warm vs cold; comedic vs tragic)?
   - Penalize obvious mismatch (casual chit-chat during imminent threat; stand-up comedy during visible tragedy).

B) **Situatedness / “Being There” (Secondary)**
   - Does it feel like something said by a participant/witness right now (or right after), given the scene pressure?
   - Good: reacts to tension/urgency/comfort/confrontation/silence in a scene-appropriate way.
   - Bad: generic role-play sentence that could fit almost any scene.

C) **No Hard Visual Contradictions**
   - The response should not rely on **specific visual claims** that are not supported by frames.
   - Heavy penalty if it invents major objects/events/locations/emotions that contradict what is shown.
   - Note: Implicit grounding is valid; explicit object listing is NOT required.

D) **Safe Abstraction for Weak Video-Text Match**
   - If frames are ambiguous or the dialogue is slightly after the event, it is acceptable for the response to stay at a higher level (tone-driven, relationship-driven).
   - However, a high score still requires the line to feel constrained by the scene’s stakes, not a free-floating fantasy.

**Evaluation Criteria (0-100)**:
- **0-20 (Contradictory / Unanchored)**:
  Ignores the scene’s mood OR contradicts key visible facts; invents major scene elements.
- **21-40 (Mostly Generic)**:
  Could be said in many unrelated scenes; weak sense of the current stakes.
- **41-60 (Plausible but Soft)**:
  Fits the broad mood but lacks scene pressure or misses critical tension cues.
- **61-80 (Immersive Grounding)**:
  Clearly shaped by the visible stakes/atmosphere; feels in-the-moment; no major contradictions.
- **81-100 (Scene-Driven & Nuanced)**:
  Strongly situated, stakes-aware, emotionally precise; uses scene-constrained cues (implicit or explicit) without hallucination.

**Output Format**:
For EACH model, output a block strictly following this format:
[[Model Name]]
Score: <Number 0-100>
Reason: <Short explanation focusing on stakes alignment + situatedness + contradiction risk>
{force_complete_msg}
""",


    "utterance_fluency": """
Attention:
You are a strict evaluator. You have received {num_models} candidate responses.
You MUST generate a scoring block for ALL {num_models} models. Do not stop until you satisfy the count.
Task: Rate the **Utterance Fluency** of {assistant_name}'s performance.

{profile_note}

**Profile Context**:
{context}

**Conversation History**:
{conversation}

**Real Answer (Ground Truth)**:
{gt_answer}

**Candidate Responses to Evaluate**:
{model_responses_block}

**Evaluation Criteria (0-100)**:
Score based on grammatical correctness, natural phrasing, and smooth readability. Does the dialogue flow naturally?

Key rules (VERY IMPORTANT):
1) **Character-Specific Dialects**: If the character profile specifies broken English, slang, or a specific dialect (e.g., Yoda, a pirate, a child), adhering to that style is **fluent for that character**. Do not penalize intentional stylistic grammar "errors" required by the persona.
2) **Coherence**: The response must make logical syntax sense.

Scoring anchors:
- **0-20 (Low Fluency)**: Riddled with severe grammatical errors (unrelated to persona), unnatural phrasing, or incoherent sentence structures; largely unreadable.
- **21-40 (Poor Fluency)**: Significant errors and awkward, robotic phrasing making it difficult to read; feels like a bad translation.
- **41-60 (Moderate Fluency)**: Some noticeable errors or awkward phrasing that impede flow, but overall meaning is clear.
- **61-80 (Good Fluency)**: Largely grammatically correct with mostly natural sentence structures; minor non-disruptive errors.
- **81-100 (High Fluency)**: Grammatically flawless (within persona constraints), natural smooth structure, excellent readability and effortless flow.

**Output Format**:
For EACH model, output a block strictly following this format:
[[Model Name]]
Score: <Number 0-100>
Reason: <Short explanation>

Do not output JSON. Just output the text blocks.
{force_complete_msg}
""",

    "instructional_adherance": """
Attention:
You are a strict evaluator. You have received {num_models} candidate responses.
You MUST generate a scoring block for ALL {num_models} models. Do not stop until you satisfy the count.
Task: Rate the **Instructional Adherence** of {assistant_name}'s performance.

{profile_note}

**Profile Context**:
{context}

**Conversation History**:
{conversation}

**Real Answer (Ground Truth)**:
{gt_answer}

**Candidate Responses to Evaluate**:
{model_responses_block}

**Evaluation Criteria (0-100)**:
Score based on how strictly the model stays in character without adding AI explanations, refusal messages, or meta-commentary.

Key rules (VERY IMPORTANT):
1) **No AI-Signposting**: Penalize HEAVILY if the response starts with "As [Character Name]...", "Here is a response...", or "I will act as...".
2) **No Refusals**: Unless the character would refuse, the model should not output AI safety refusals (e.g., "I cannot generate that content").
3) **Pure Diegesis**: The output must be *only* what the character says/does.

Scoring anchors:
- **0-20 (Low Adherence)**: Ignores role-play entirely; uses generic AI assistant phrasing; breaks character to explain the response.
- **21-40 (Poor Adherence)**: Partially role-plays but frequently includes explanatory prefixes/suffixes or neutral "assistant" language.
- **41-60 (Moderate Adherence)**: Mostly character voice but slips into descriptive/instructional language or includes non-diegetic elements.
- **61-80 (Good Adherence)**: Consistently stays in-character with no explanatory framing; deviations are rare and subtle.
- **81-100 (High Adherence)**: Perfectly embodies the character without any AI-like signposts, explanations, or out-of-role content; pure immersion.

**Output Format**:
For EACH model, output a block strictly following this format:
[[Model Name]]
Score: <Number 0-100>
Reason: <Short explanation>

Do not output JSON. Just output the text blocks.
{force_complete_msg}
""",


    "response_accuracy": """
Attention:
You are a strict evaluator. You have received {num_models} candidate responses.
You MUST generate a scoring block for ALL {num_models} models. Do not stop until you satisfy the count.
Task: Rate the **Response Accuracy** of {assistant_name}'s performance.

{profile_note}

**Profile Context**:
{context}

**Conversation History**:
{conversation}

**Real Answer (Ground Truth)**:
{gt_answer}

**Candidate Responses to Evaluate**:
{model_responses_block}

**Evaluation Criteria (0-100)**:
Score based on whether the response accurately addresses the user's input/question and engages appropriately in the current conversational context.

Key rules (VERY IMPORTANT):
1) **Relevance**: The response must directly follow the logical flow of the Conversation History.
2) **Intent**: Did the character understand what the user said? (e.g., if User asks "Where are we?", does the Assistant answer the location, or say something random?)

Scoring anchors:
- **0-20 (Low Accuracy)**: Completely fails to address the question/input; entirely irrelevant to the context; logic hallucination.
- **21-40 (Poor Accuracy)**: Tangentially addresses the context but misses the core intent; introduces significant irrelevant information.
- **41-60 (Moderate Accuracy)**: Generally engages appropriately but may overlook nuances or be slightly incomplete/vague.
- **61-80 (Good Accuracy)**: Accurately addresses the main aspects of the input; engages well with the context with minor omissions.
- **81-100 (High Accuracy)**: Perfectly and comprehensively addresses the input; engages flawlessly within the conversational context.

**Output Format**:
For EACH model, output a block strictly following this format:
[[Model Name]]
Score: <Number 0-100>
Reason: <Short explanation>

Do not output JSON. Just output the text blocks.
{force_complete_msg}
""",

    "human_likeness": """
Attention:
You are a strict evaluator. You have received {num_models} candidate responses.
You MUST generate a scoring block for ALL {num_models} models. Do not stop until you satisfy the count.
Task: Rate the **Human Likeness** of {assistant_name}'s performance.

{profile_note}

**Profile Context**:
{context}

**Conversation History**:
{conversation}

**Real Answer (Ground Truth)**:
{gt_answer}

**Candidate Responses to Evaluate**:
{model_responses_block}

**Evaluation Criteria (0-100)**:
Score based on whether the response conveys a sense of human-like interaction rather than presenting an AI style.

Key rules (VERY IMPORTANT):
1) **Avoid AI Structure**: Penalize lists, bullet points, or "essay-style" structure unless the character is specifically a robot/computer.
2) **Emotional Nuance**: Human-like responses contain subtext, hesitation, or emotional coloring, whereas AI responses are often overly polite and flat.

Scoring anchors:
- **0-20 (Low Human Likeness)**: Distinctly artificial, robotic, or overly formulaic; clear "AI assistant" vibe.
- **21-40 (Poor Human Likeness)**: Noticeable AI-like characteristics (unnatural phrasing, lack of nuance, mechanical tone).
- **41-60 (Moderate Human Likeness)**: Some human qualities but still contains artificial/formal elements; doesn't feel alive.
- **61-80 (Good Human Likeness)**: Generally sounds natural and conversational; avoids obvious AI tells.
- **81-100 (High Human Likeness)**: Indistinguishable from human expression; natural tone, nuance, and conversational style; zero AI stiffness.

**Output Format**:
For EACH model, output a block strictly following this format:
[[Model Name]]
Score: <Number 0-100>
Reason: <Short explanation>

Do not output JSON. Just output the text blocks.
{force_complete_msg}
"""
}

METRICS_NEED_VIDEO = ["video_text_relevance"]



client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def _format_character_block(movie, name):
    profile = resolve_profile(movie, name)
    lines = [f"Name: {name}", "Profile:"]
    if not profile or not isinstance(profile, dict):
        lines.append("   (No specific profile details)")
        return "\n".join(lines)
    for k, v in profile.items():
        if v:
            lines.append(f"   - {k}: {v}")
    return "\n".join(lines)

def _format_context_block(example):
    movie = normalize_movie_name(example.get("data_source", "Unknown"))
    user_name = example.get("user", "User")
    assistant_name = example.get("assistant", "Assistant")
    
    user_block = _format_character_block(movie, user_name)
    asst_block = _format_character_block(movie, assistant_name)
    
    return (
        f"--- Movie: {movie} ---\n"
        f"[USER INFO] ({user_name})\n{user_block}\n\n"
        f"[ASSISTANT INFO] Target Persona: ({assistant_name})\n{asst_block}\n"
    )

def _format_dialogue(dialogue_list):
    lines = []
    for turn in dialogue_list:
        spk = turn.get("speaker", "Unknown")
        utt = turn.get("utterance", "")
        lines.append(f"{spk}: {utt}")
    return "\n".join(lines)

def extract_answer_tag_robust(text):
    """
    【核心修改】实现解耦：内容提取宽容，格式判断严格。
    
    Returns:
        content (str): 提取出的内容（用于 LLM 评分，越全越好）
        is_strict_format (bool): 是否符合 strict <answer>...</answer> 标准
    """
    if not text: return "", False

    # 1. 严格格式检查 (Strict Check)
    # 必须同时包含 <answer> 和 </answer> 才算格式正确
    is_strict_fmt = bool(re.search(r'<answer>\s*\S.*?\s*</answer>', text, re.DOTALL | re.IGNORECASE))

    # 2. 内容提取 (Content Extraction - 宽容模式)
    
    # 策略 A: 尝试提取 <answer> 开头的内容 (无论有没有结尾)
    match = re.search(r'<answer>\s*(.*?)(?:</answer>|$)', text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        # 只要能提取出内容，就返回内容；格式分由 is_strict_fmt 决定
        return content, is_strict_fmt
    
    # 策略 B: 【Fallback】针对 Vanilla/CoT 格式错误 (如只写了 </think>)
    # 尝试找到 </think>，取其后面的内容。
    think_end_match = re.search(r'</think>\s*(.*)', text, re.DOTALL | re.IGNORECASE)
    if think_end_match:
        content = think_end_match.group(1).strip()
        # 这里肯定格式错误，因为没 <answer>
        return content, False 

    # 策略 C: 实在没标签，返回全文
    # 【核心修复】：在返回全文前，必须尝试移除 <think>...</think> 或 <vision>...</vision>
    clean_text = text
    
    # 移除 <think>...</think> (非贪婪匹配)
    clean_text = re.sub(r'<think>.*?</think>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    # 移除 <vision>...</vision>
    clean_text = re.sub(r'<vision>.*?</vision>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    
    # 再清理一下可能残留的空标签
    clean_text = clean_text.replace('<think>', '').replace('</think>', '')
    clean_text = clean_text.replace('<vision>', '').replace('</vision>', '')
    
    clean_text = clean_text.strip()
    
    return clean_text, False

def process_video_frames(video_path):
    if not os.path.exists(video_path): return None
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if 0 < total_frames <= MAX_NUM_FRAMES:
            target_indices = list(range(total_frames))
        elif total_frames > MAX_NUM_FRAMES:
            target_indices = [int(i * total_frames / MAX_NUM_FRAMES) for i in range(MAX_NUM_FRAMES)]
            target_indices = sorted(list(set(target_indices))) 
        else:
            target_indices = None 
        frames_base64 = []
        current_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            keep_frame = False
            if target_indices is None:
                if len(frames_base64) < MAX_NUM_FRAMES: keep_frame = True
                else: break
            else:
                if current_idx in target_indices: keep_frame = True
            if keep_frame:
                h, w = frame.shape[:2]
                scale = FRAME_RESIZE / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame_resized = cv2.resize(frame, (new_w, new_h))
                _, buffer = cv2.imencode(".jpg", frame_resized)
                b64_str = base64.b64encode(buffer).decode("utf-8")
                frames_base64.append(b64_str)
            current_idx += 1
            if target_indices is not None and len(frames_base64) >= len(target_indices): break
        cap.release()
        if not frames_base64: return None
        return frames_base64
    except Exception as e:
        if cap: cap.release()
        return None

def call_judge_llm_text(messages):
    for attempt in range(3): 
        try:
            # print(f"[Debug] Attempt {attempt+1}...")
            response = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=messages,
                temperature=0.0, 
                max_tokens=4096, 
                timeout=120
            )
            content = response.choices[0].message.content
            if content is None or content.strip() == "":
                return None
            return content
        except Exception as e:
            if attempt < 2:
                time.sleep(1)
            else:
                return None

def parse_judge_response(text, anon_names):
    if not text: return {}
    results = {}
    for anon in anon_names:
        pattern = rf"\[\[{re.escape(anon)}\]\]\s*Score:\s*(\d+(\.\d+)?).*?Reason:\s*(.*?)(?=\[\[|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                reason = match.group(3).strip()
                results[anon] = {"score": score, "reason": reason}
            except:
                results[anon] = None 
        else:
            results[anon] = None 
    return results

# ================= 评测核心逻辑 =================
def evaluate_sample(unified_item, metric_name, model_names, video_base_dir=None):
    context_str = _format_context_block(unified_item)
    dialogue_str = _format_dialogue(unified_item.get("dialogue", []))
    gt_answer = unified_item.get("target", {}).get("utterance", "")
    assistant_name = unified_item.get("assistant", "Assistant")
    
    raw_video_path = unified_item.get("video_path", "")
    video_path = raw_video_path
    if video_base_dir and raw_video_path and not os.path.isabs(raw_video_path):
        video_path = os.path.join(video_base_dir, raw_video_path)
        video_path = os.path.normpath(video_path)

    model_responses_block = ""
    valid_anon_payload = [] 
    
    # 记录详细的临时状态：{RealName: {'ans': str, 'strict_fmt': bool}}
    temp_extraction_status = {} 
    raw_outputs_snapshot = {}

    # ================= 匿名化处理 (Strict logic) =================
    # 1. 固定匿名标签列表 (Model A, Model B...)
    anon_labels = [f"Model {chr(65+i)}" for i in range(len(model_names))]
    
    # 2. 随机打乱真实模型名称
    shuffled_real_names = list(model_names)
    random.shuffle(shuffled_real_names) 
    
    # 3. 建立映射: Slot 0 (Model A) -> RandomRealModel
    anon_to_real_map = {} 
    
    for idx, anon_label in enumerate(anon_labels):
        real_model_name = shuffled_real_names[idx]
        anon_to_real_map[anon_label] = real_model_name
        
        # 获取该模型的原始输出
        raw_output = unified_item.get(f"res_{real_model_name}", {}).get("raw_output", "")
        if raw_output is None: raw_output = ""
        raw_outputs_snapshot[real_model_name] = raw_output.replace("\n", " ") + "..."

        # 【调用提取函数】
        extracted_content, is_strict_fmt = extract_answer_tag_robust(raw_output)
        
        # 记录提取状态 (保留原始提取结果)
        temp_extraction_status[real_model_name] = {
            "content": extracted_content,
            "is_strict_fmt": is_strict_fmt
        }

        # 处理空内容：填入占位符，保证所有模型都在 Prompt 里出现
        final_prompt_content = extracted_content.strip()
        if not final_prompt_content:
            final_prompt_content = "(EMPTY RESPONSE / NO OUTPUT)"
        
        # 构建 Block
        model_responses_block += f"[[{anon_label}]]\nResponse: {final_prompt_content}\n\n"
        valid_anon_payload.append(anon_label)
        
    # ==========================================================

    # 3. 视频检查 
    video_frames = None
    if metric_name in METRICS_NEED_VIDEO:
        video_frames = process_video_frames(video_path)
        if not video_frames:
             return None, f"Video Missing or Corrupted: {video_path}"

    # 4. 全挂检查 (防御性代码，正常不会进，因为上面做了空填补)
    if not valid_anon_payload:
        ordered_scores_map = {}
        for m in model_names:
             ordered_scores_map[m] = {
                "score": 0, "reason": "System Error: No models loaded.",
                "format_pass": False, "raw_snapshot": ""
             }
        return ordered_scores_map, None

    # 5. 构建 Prompt
    num_models_count = len(valid_anon_payload)
    force_msg = FORCE_COMPLETE_INSTRUCTION.format(num_models=num_models_count)
    prompt_template = PROMPT_TEMPLATES[metric_name]
    
    user_prompt = prompt_template.format(
        num_models=num_models_count,
        profile_note=COMMON_PROFILE_INSTRUCTION,
        assistant_name=assistant_name,
        context=context_str,
        conversation=dialogue_str,
        gt_answer=gt_answer,
        model_responses_block=model_responses_block,
        force_complete_msg=force_msg
    )
    
    if video_frames:
        user_prompt = (
            f"(You will receive {len(video_frames)} video frames as images ABOVE in this same message. "
            f"Use them for scoring.)\n\n"
            + user_prompt
        )

    base_messages = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}]
    input_content = []
    if video_frames:
        for b64 in video_frames:
            input_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
            })
    input_content.append({"type": "text", "text": user_prompt})
    base_messages.append({"role": "user", "content": input_content})

    # 6. 重试逻辑
    final_scores_map = {}
    best_attempt_result = {} 
    missing_models = None

    for attempt_idx in range(MAX_JUDGE_RETRIES):
        current_messages = copy.deepcopy(base_messages)
        
        if attempt_idx > 0 and missing_models:
             remind_msg = (
                 f"\n\nWARNING: In your previous attempt, you missed rating the following models: {missing_models}. "
                 f"You MUST provide scores for {len(missing_models)} missing models immediately."
             )
             current_messages.append({"role": "user", "content": [{"type": "text", "text": remind_msg}]})

        judge_text_output = call_judge_llm_text(current_messages)
        if not judge_text_output: continue 
            
        parsed_results = parse_judge_response(judge_text_output, valid_anon_payload)
        
        for k, v in parsed_results.items():
            if v is not None: best_attempt_result[k] = v

        missing_models = [m for m in valid_anon_payload if best_attempt_result.get(m) is None]
        if not missing_models: break 
        else: continue

    if not best_attempt_result:
        return None, "Judge API Call Failed"

    missing_finally = [m for m in valid_anon_payload if best_attempt_result.get(m) is None]
    if missing_finally:
        return None, f"Judge incomplete. Missing: {missing_finally}"

    # 7. 组装结果 & 【还原与强制覆盖】
    for anon_name in valid_anon_payload:
        real_name = anon_to_real_map[anon_name]
        judge_result = best_attempt_result.get(anon_name)
        
        temp_status = temp_extraction_status[real_name]
        is_empty_originally = not temp_status["content"].strip()
        

        if is_empty_originally:
            final_score = 0.0
            final_reason = "Evaluator Override: Original output was empty."
        else:
            final_score = judge_result["score"]
            final_reason = judge_result["reason"]
        
        final_scores_map[real_name] = {
            "score": final_score,
            "reason": final_reason,
            "format_pass": temp_status["is_strict_fmt"], 
            "raw_snapshot": raw_outputs_snapshot.get(real_name, "")
        }

    # 8. 排序输出 (按输入的 model_names 顺序)
    ordered_scores_map = {}
    for m in model_names:
        if m in final_scores_map:
            ordered_scores_map[m] = final_scores_map[m]
        else:
            ordered_scores_map[m] = {
                "score": 0, "reason": "System Error: Missing in final map", 
                "format_pass": False, "raw_snapshot": ""
            }

    return ordered_scores_map, None

def load_and_align_data(model_files, dropped_list): 
    """
    读取所有模型文件，并只保留所有文件都存在的 Problem ID (Intersection).
    """
    raw_data_map = {} 
    all_pids_sets = []
    
    print("[Data] Loading model results...")
    
    for m_name, f_path in model_files.items():
        if not os.path.exists(f_path):
            print(f"[Error] File not found: {f_path}")
            sys.exit(1)
            
        with open(f_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        current_model_pids = set()
        raw_data_map[m_name] = {}
        
        for item in data:
            pid = str(item.get("problem_id", item.get("id")))
            current_model_pids.add(pid)
            raw_data_map[m_name][pid] = item
            
        all_pids_sets.append(current_model_pids)
        print(f"   - {m_name}: Loaded {len(data)} samples from {f_path}")

    if not all_pids_sets:
        return []
    
    common_pids = set.intersection(*all_pids_sets)
    print(f"[Data] Found {len(common_pids)} common samples across all {len(model_files)} models.")

    unique_dropped_pids = set()
    for m_name, pids_set in zip(model_files.keys(), all_pids_sets):
        diff = pids_set - common_pids
        for pid in diff:
            if pid not in unique_dropped_pids:
                unique_dropped_pids.add(pid)
                dropped_list.append({
                    "problem_id": pid,
                    "error_reason": f"Missing in some models (Present in {m_name}, but not all)"
                })
    
    print(f"[Data] Dropped {len(unique_dropped_pids)} samples (missing in at least one model).")

    aligned_list = []
    for pid in common_pids:
        first_model = list(model_files.keys())[0]
        base_item = raw_data_map[first_model][pid]
        
        unified_item = {
            "problem_id": pid,
            "video_path": base_item.get("video_path"),
            "data_source": base_item.get("data_source"),
            "user": base_item.get("user"),
            "assistant": base_item.get("assistant"),
            "dialogue": base_item.get("dialogue"),
            "target": base_item.get("target"),
        }
        
        for m_name in model_files.keys():
            m_item = raw_data_map[m_name][pid]
            unified_item[f"res_{m_name}"] = {
                "raw_output": m_item.get("raw_output", m_item.get("model_prediction", ""))
            }
        
        aligned_list.append(unified_item)
            
    return aligned_list

def worker_wrapper(item, metric, model_names, pbar, result_lock, details_list, dropped_list, video_base_dir=None):
    pid = item["problem_id"]
    try:
        scores_map, error_reason = evaluate_sample(item, metric, model_names, video_base_dir=video_base_dir)
        
        # import pdb; pdb.set_trace()
        
        with result_lock:
            if error_reason:
                dropped_list.append({
                    "problem_id": pid,
                    "error_reason": error_reason
                })
            else:
                details_list.append({
                    "problem_id": pid,
                    "metric_scores": scores_map
                })
            
    except Exception as e:
        with result_lock:
            dropped_list.append({
                "problem_id": pid,
                "error_reason": f"Code Exception: {str(e)}"
            })
    finally:
        pbar.update(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs='+', required=True, help="List of model results: CustomName=Path")
    parser.add_argument("--metric", type=str, required=True, choices=PROMPT_TEMPLATES.keys())
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--error_file", type=str, required=False, default=None, help="File to save dropped samples")
    parser.add_argument("--video_base_dir", type=str, default=None, help="Root directory for relative video paths")
    args = parser.parse_args()

    if not args.error_file:
        base, ext = os.path.splitext(args.output_file)
        args.error_file = f"{base}_drops{ext}"

    model_files = {}
    model_names = []
    for pair in args.models:
        name, path = pair.split("=", 1)
        model_files[name] = path
        model_names.append(name)

    print(f"[Info] Metric: {args.metric}")
    print(f"[Info] Judge Retries (per sample): {MAX_JUDGE_RETRIES}")
    
    details_list = []
    dropped_list = []
    result_lock = threading.Lock()
    
    unified_data = load_and_align_data(model_files, dropped_list)
    print(f"[Info] Ready to evaluate {len(unified_data)} samples.")
    
    print(f"[Info] Starting Evaluation with {MAX_WORKERS} workers...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with tqdm(total=len(unified_data)) as pbar:
            futures = []
            for item in unified_data:
                futures.append(
                    executor.submit(worker_wrapper, item, args.metric, model_names, pbar, result_lock, details_list, dropped_list, args.video_base_dir)
                )
            for f in as_completed(futures): pass
    
    
    # print("!!! DEBUG MODE: Running sequentially in main thread !!!")
    # import pdb 
    
    # # 创建一个假的进度条类，防止 worker_wrapper 报错
    # class MockPbar:
    #     def update(self, n): pass
    
    # pbar = MockPbar() # 实例化假进度条
    
    # for i, item in enumerate(unified_data):
    #     print(f"Processing item {i} | Problem ID: {item.get('problem_id')}...")
        
    #     # 【调试点 1】：设置断点，程序会暂停在处理每个样本之前
    #     # pdb.set_trace() 
        
    #     # 直接调用 worker_wrapper (不通过线程池)
    #     worker_wrapper(item, args.metric, model_names, pbar, result_lock, details_list, dropped_list, args.video_base_dir)
        
    #     # 【调试点 2】：只跑 1 个样本测试流程，取消下面两行的注释
    #     # if i >= 0: 
    #     #     break
    
    # ================= 统计汇总 =================
    stats = {
        m: {"total_score": 0.0, "format_pass_count": 0, "valid_samples": 0} 
        for m in model_names
    }

    # 仅统计 Valid (未被 Drop) 的样本
    for item in details_list:
        scores_map = item["metric_scores"]
        for m in model_names:
            if m in scores_map:
                record = scores_map[m]
                stats[m]["valid_samples"] += 1
                stats[m]["total_score"] += record["score"] 
                if record["format_pass"]:
                    stats[m]["format_pass_count"] += 1

    final_summary = {}
    print("\n" + "="*90)
    print(f"{'Model Name':<20} | {'Avg Score':<10} | {'Pass Rate':<10} | {'Valid'} | {'Drops'}")
    print("-" * 90)
    
    for m in model_names:
        total = stats[m]["valid_samples"]
        if total > 0:
            avg_score = stats[m]["total_score"] / total
            pass_rate = (stats[m]["format_pass_count"] / total) * 100
        else:
            avg_score = 0.0
            pass_rate = 0.0
            
        final_summary[m] = {
            "average_score": round(avg_score, 2),
            "format_pass_rate": f"{pass_rate:.2f}%",
            "format_pass_count": stats[m]["format_pass_count"],
            "total_valid_samples": total,
            "dropped_samples_count": len(dropped_list)
        }
        
        print(f"{m:<20} | {avg_score:<10.2f} | {pass_rate:<9.2f}% | {total:<5} | {len(dropped_list)}")
    print("="*90 + "\n")

    output_data = {
        "meta_info": {
            "metric": args.metric,
            "judge_model": MODEL_NAME,
            "dropped_count": len(dropped_list)
        },
        "statistics_summary": final_summary,
        "samples_details": details_list
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    
    with open(args.error_file, "w", encoding="utf-8") as f:
        json.dump(dropped_list, f, indent=4, ensure_ascii=False)
        
    print(f"[Done] Report: {args.output_file}")
    print(f"[Done] Drops:  {args.error_file}")

if __name__ == "__main__":
    main()


# # 使用示例：
# python rating_all_models_on_all_metrics.py \
#   --models Ours=/path/to/ours.json GPT4=/path/to/gpt4.json \
#   --metric character_fidelity \
#   --output_file ./eval_results/eval_character_stat.json
#   --video_base_dir /path/to/videos
