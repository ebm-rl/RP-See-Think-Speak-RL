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


API_KEY="xxx"  # your api key
BASE_URL = "xxx"  # api base url
MODEL_NAME = "gpt-5-mini"   # evaluation model name


MAX_NUM_FRAMES = 8       
FRAME_RESIZE = 512       
MAX_JUDGE_RETRIES = 3    # Judge retry times
RETRY_DELAY = 1          
MAX_WORKERS = 20

# Judge System Prompt
JUDGE_SYSTEM_PROMPT = (
    "You are an expert Role-Play Evaluation Critic. "
    "Your task is to score multiple AI models' responses based on specific metrics. "
    "Maintain absolute objectivity."
    "**CRITICAL SCORING RULE**: \n"
    "Candidate responses may contain reasoning steps, chain-of-thought, or internal vision analysis tags.\n"    
    "1. You MUST IGNORE all such meta-content.\n"
    "2. You MUST first identify the actual response part and evaluate the score based exclusively on the final utterance of the character.\n"
    "3. DO NOT reward models for 'thinking' steps. Detailed reasoning DOES NOT equal higher character fidelity or fluency."
)


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
    "Visual_Evidence_Grounding": """
Attention:
You are a "Visual Evidence Auditor". Your task is to evaluate the physical and factual alignment between the video and the dialogue. 
This is a test of "Observation Logic," not personality.

{profile_note}

**Profile Context (for baseline temperament only):**
{context}

**Conversation History (for current emotional context):**
{conversation}

**Real Answer (Ground Truth): Reference only**:
{gt_answer}

**Candidate Responses (ignore tags; judge spoken dialogue only):**
{model_responses_block}

Scoring (0-100), compute 3 subscores then weighted total:

**EVALUATION CRITERIA**:
1. **Visual Triggering (50%)**: Does the dialogue contain a direct, logical link to a specific action, object, or expression shown in the video?
2. **Object/Scene Integrity (30%)**: Does the response respect the physical limits of the scene? Penalize models mentioning invisible items.
3. **Temporal Realism (20%)**: Does the utterance match the immediacy of visual perception? Real-time visual grounding is typically reactive and sharp. Penalize excessive length if it drifts into “describing the video” instead of “reacting to the video”.

**SCORING ANCHORS (5 Tiers)**:
- 0-20 (Tier 1): Hallucinates elements not in video or provides AI refusal.
- 21-40 (Tier 2): Overly descriptive/bookish; describes the scene rather than being in it.
- 41-60 (Tier 3): Factually safe but "video-blind"; relies purely on text history.
- 61-80 (Tier 4): Shows clear awareness of the visual environment; situated and reactive.
- 81-100 (Tier 5): Precision sensing; captures micro-interactions only possible via vision.

**Output Format**:
For EACH model:
[[Model Name]]
Subscores: Triggering=<0-100>, Integrity=<0-100>, Temporal=<0-100>
Score: <0-100>
Reason: <explain how the response reflects visual precision vs. descriptive hallucination>
{force_complete_msg}
""",


    "Conversational_Naturalism": """
Attention:
You are a "Linguistic Stylist". Your task is to evaluate the "Spoken Quality" of the response. 
Is it a real person talking, or an AI writing a book?

{profile_note}

**Character Profile (for linguistic habits):**
{context}

**Conversation History:**
{conversation}

**Real Answer (Ground Truth): Reference only**:
{gt_answer}

**Candidate Responses (judge ONLY the spoken line):**
{model_responses_block}

Scoring (0-100), compute 2 subscores then weighted total:

**EVALUATION CRITERIA**:
1. **Oral Realism (60%)**: Use of colloquial phrasing and natural sentence structures. 
2. **Anti-Narrative Voice (40%)**: Penalize stage directions (*smiles*), "he said", or bookish monologues.

**SCORING ANCHORS (5 Tiers)**:
- 0-20 (Tier 1): Script/AI Voice; feels like a robot or contains "As an AI...".
- 21-40 (Tier 2): Narrative Prose; structured like a book paragraph, too formal/polished for speech.
- 41-60 (Tier 3): Formal Dialogue; grammatically correct but stiff, like a formal interview.
- 61-80 (Tier 4): Colloquial Spoken; good flow, uses contractions and natural slang.
- 81-100 (Tier 5): Situated Utterance; feels like a "snippet" of real life—short and impactful.

**Output Format**:
For EACH model:
[[Model Name]]
Subscores: Oral=<0-100>, Anti-Narrative=<0-100>
Score: <0-100>
Reason: <explain how the response avoids narrator-voice and achieves natural speech>
{force_complete_msg}
""",


    "Situational_Persona_Compatibility": """
Attention:
You are an expert Casting Director and Acting Coach. 
Your task is to evaluate {num_models} candidate responses. 
Your goal is to evaluate if the line sounds like the character would ACTUALLY speak in the specific atmosphere shown in the video. 

**EVALUATION PHILOSOPHY**:
- **Dynamic Persona (Core Insight)**: Character consistency is NOT static. A "humorous" character in mortal danger should show nervous grit or brevity, NOT tell jokes. A "wise" leader in a high-pressure scene should be decisive and sharp, NOT deliver a poetic lecture. 
- **The "Presence" Test**: Does the model sound like someone physically in the room? 
- **Authenticity over Caricature**: Avoid rewarding models that simply use catchphrases or "lore-dumping" if it breaks the immediate scene tension.


{profile_note}

**Character Profile & Values**:
{context}

**Conversation History**:
{conversation}

**Real Answer (Ground Truth): Just for reference only**:
{gt_answer}

**Candidate Responses (may include extra tags; apply the rule above)**:
{model_responses_block}

Scoring (0-100), compute 3 subscores and then a weighted total:

1) **Situational Value Expression (40%)**: 
   - Does the character express their core identity THROUGH the lens of the current event?
   - **Crucial**: High scores go to characters who adapt their expression. (e.g., A proud villain showing rare caution when the video shows an overwhelming threat).
   - Penalize "Static Labeling" where characters repeat slogans regardless of the visual pressure.

2) **Acting Realism & Rhythm (30%)**: 
   - **Hard Penalty**: Over-dramatized monologues or “stagey” declamations that destroy the scene's tension.
   - **Focus**: Does the response respect the pacing of the moment? High-stakes/urgent scenes usually require sharp, reactive lines, long-winded explanations in tense moments break immersion.

3) **Sensory-Driven Modulation (30%)**: 
   - Does the voice feel like it was filtered through the "Eyes"? 
   - If the video shows the character is exhausted, is the line shorter and lower energy?
   
**Scoring Anchors (5 Tiers)**:

- **0-20 (Tier 1: Disconnected)**: Meta-responses, refusals, or completely out-of-universe logic.
- **21-40 (Tier 2: Static Caricature)**: Sounds like a generic fan-fiction description. Overly wordy, ignores the video's urgency, or uses "book-narrator" prose instead of speech.
- **41-60 (Tier 3: Rigid Profile)**: Accurate to the text-only Profile, but "blind" to the video. The character acts the same in a forest battle as they would in a classroom.
- **61-80 (Tier 4: Believable Actor)**: Consistent and shows clear awareness of the scene. The tone shifts correctly with the visual tension. Minor style gaps.
- **81-100 (Tier 5: Immersive Soul)**: Elite performance. The line is concise, perfectly situated, and captures the character's unique voice under the *exact* pressure shown in the video.

Output (for EACH model):
[[Model Name]]
Subscores: Values=<0-100>, Style=<0-100>, Stakes=<0-100>
Score: <0-100>
Reason: <Focus on how the character's static profile evolved/modulated in response to the specific video atmosphere>
{force_complete_msg}
"""
}

METRICS_NEED_VIDEO = ["Situational_Persona_Compatibility", "Visual_Evidence_Grounding"]



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
    if not text: return "", False

    match = re.search(r'<answer>\s*(.*?)(?:</answer>|$)', text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        return content, True
    
    clean_text = text
    clean_text = re.sub(r'<(think|vision)>.*?</\1>', '', clean_text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r'</?(think|vision|answer)>', '', clean_text, flags=re.IGNORECASE)
    
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
    last_error_log = "Unknown API Error"
    for attempt in range(3): 
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=messages,
                temperature=0.0, 
                max_completion_tokens=4096, 
                timeout=120
            )
            # import pdb; pdb.set_trace()
            content = response.choices[0].message.content
            if content and content.strip():
                return content, None
            last_error_log = "API response was successful but the content is empty."
        except Exception as e:
            last_error_log = str(e)
            tqdm.write(f"\n[API attempt {attempt+1}/3 failed]: {last_error_log}")
            if attempt < 2:
                time.sleep(2) 

    return None, last_error_log


def parse_judge_response(text, anon_names):
    if not text: return {}
    results = {}
    for anon in anon_names:
        pattern = rf"\[\[{re.escape(anon)}\]\].*?Score:\s*(\d+(\.\d+)?).*?Reason:\s*(.*?)(?=\[\[|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            try:
                score = float(match.group(1))
                reason = match.group(3).strip()
                results[anon] = {"score": score, "reason": reason}
            except Exception as e:
                print(f"Parsing {anon} failed: {e}")
                results[anon] = None 
        else:
            results[anon] = None 
    return results

def evaluate_sample(unified_item, metric_name, model_names, video_base_dir=None):
    context_str = _format_context_block(unified_item)
    dialogue_str = _format_dialogue(unified_item.get("dialogue", []))
    gt_answer = unified_item.get("target", {}).get("utterance", "")
    assistant_name = unified_item.get("assistant", "Assistant")
    
    raw_video_path = unified_item.get("video_path", "")
    video_path = raw_video_path
    if video_base_dir and raw_video_path and not os.path.isabs(raw_video_path):
        video_path = os.path.join(video_base_dir, raw_video_path)

    model_responses_block = ""
    valid_anon_payload = [] 
    temp_extraction_status = {} 
    raw_outputs_snapshot = {}

    anon_labels = [f"Model {chr(65+i)}" for i in range(len(model_names))]
    shuffled_real_names = list(model_names)
    random.shuffle(shuffled_real_names) 
    anon_to_real_map = {anon: real for anon, real in zip(anon_labels, shuffled_real_names)}
    
    for anon_label, real_model_name in anon_to_real_map.items():
        raw_output = unified_item.get(f"res_{real_model_name}", {}).get("raw_output", "") or ""
        raw_outputs_snapshot[real_model_name] = raw_output.replace("\n", " ")
        extracted_content, is_strict_fmt = extract_answer_tag_robust(raw_output)
        
        temp_extraction_status[real_model_name] = {"content": extracted_content, "is_strict_fmt": is_strict_fmt}
        final_prompt_content = extracted_content.strip() or "(EMPTY RESPONSE)"
        model_responses_block += f"[[{anon_label}]]\nResponse: {final_prompt_content}\n\n"
        valid_anon_payload.append(anon_label)
        
    video_frames = None
    if metric_name in METRICS_NEED_VIDEO:
        video_frames = process_video_frames(video_path)
        if not video_frames:
            return None, f"The video file is missing or corrupted: {video_path}"

    force_msg = FORCE_COMPLETE_INSTRUCTION.format(num_models=len(valid_anon_payload))
    prompt_template = PROMPT_TEMPLATES.get(metric_name, "")
    user_prompt = prompt_template.format(
        num_models=len(valid_anon_payload), profile_note=COMMON_PROFILE_INSTRUCTION,
        assistant_name=assistant_name, context=context_str, conversation=dialogue_str,
        gt_answer=gt_answer, model_responses_block=model_responses_block, force_complete_msg=force_msg
    )
    
    base_messages = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}]
    input_content = []
    if video_frames:
        for b64 in video_frames:
            input_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}})
    input_content.append({"type": "text", "text": user_prompt})
    base_messages.append({"role": "user", "content": input_content})

    best_attempt_result = {} 
    missing_models = None

    final_api_error_reason = "All retry attempts failed to return error details."

    for attempt_idx in range(MAX_JUDGE_RETRIES):
        current_messages = copy.deepcopy(base_messages)
        if attempt_idx > 0 and missing_models:
            remind_msg = f"\n\nWARNING: Missing models: {missing_models}. Provide them now."
            current_messages.append({"role": "user", "content": [{"type": "text", "text": remind_msg}]})

        judge_text_output, api_error = call_judge_llm_text(current_messages)
        
        if not judge_text_output: 
            if api_error:
                final_api_error_reason = api_error
            continue 
            
        parsed_results = parse_judge_response(judge_text_output, valid_anon_payload)
        for k, v in parsed_results.items():
            if v: best_attempt_result[k] = v

        missing_models = [m for m in valid_anon_payload if m not in best_attempt_result]
        if not missing_models: break 

    if not best_attempt_result:
        return None, f"The Judge API call ultimately failed. Details: {final_api_error_reason}"
    
    final_scores_map = {}
    for anon_name, real_name in anon_to_real_map.items():
        judge_result = best_attempt_result.get(anon_name)
        if not judge_result or not temp_extraction_status[real_name]["content"].strip():
            final_scores_map[real_name] = {"score": 0.0, "reason": "Result is empty or parsing failed", "format_pass": False, "raw_snapshot": raw_outputs_snapshot.get(real_name, "")}
        else:
            final_scores_map[real_name] = {
                "score": judge_result["score"], "reason": judge_result["reason"],
                "format_pass": temp_extraction_status[real_name]["is_strict_fmt"], 
                "raw_snapshot": raw_outputs_snapshot.get(real_name, "")
            }
    
    return {m: final_scores_map.get(m, {"score":0, "reason":"Missing rating data"}) for m in model_names}, None


def load_and_align_data(model_files, dropped_list, sample_limit=100): 
    raw_data_map = {} 
    all_pids_sets = []
    print("[Data] Begin aligning model results...")
    
    for m_name, f_path in model_files.items():
        if not os.path.exists(f_path):
            print(f"[Error] Model file not found: {f_path}")
            return []
        with open(f_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            current_model_pids = set()
            raw_data_map[m_name] = {}
            for item in data:
                pid = str(item.get("problem_id", item.get("id", "")))
                if pid:
                    current_model_pids.add(pid)
                    raw_data_map[m_name][pid] = item
            all_pids_sets.append(current_model_pids)
            print(f"   - {m_name}: {len(current_model_pids)} samples loaded")

    if not all_pids_sets: return []
    common_pids = sorted(list(set.intersection(*all_pids_sets)))
    print(f"[Data] Number of samples jointly owned by models: {len(common_pids)}")
    
    # random.seed(42)
    random.shuffle(common_pids)
    if len(common_pids) > sample_limit:
        common_pids = common_pids[:sample_limit]
    
    aligned_list = []
    for pid in common_pids:
        first_model = list(model_files.keys())[0]
        base_item = raw_data_map[first_model][pid]
        unified_item = {k: base_item.get(k) for k in ["video_path", "data_source", "user", "assistant", "dialogue", "target"]}
        unified_item["problem_id"] = pid
        for m_name in model_files.keys():
            unified_item[f"res_{m_name}"] = {"raw_output": raw_data_map[m_name][pid].get("raw_output", raw_data_map[m_name][pid].get("model_prediction", ""))}
        aligned_list.append(unified_item)
    return aligned_list

def save_checkpoint(output_file, error_file, details, drops, metric, model_names):
    stats = {m: {"total_score": 0.0, "valid_samples_count": 0} for m in model_names}
    for item in details:
        for m in model_names:
            if m in item["metric_scores"]:
                stats[m]["valid_samples_count"] += 1
                stats[m]["total_score"] += item["metric_scores"][m]["score"]
    
    summary = {m: {
        "average_score": round(stats[m]["total_score"] / stats[m]["valid_samples_count"], 2) if stats[m]["valid_samples_count"] > 0 else 0,
        "total_valid_samples": stats[m]["valid_samples_count"]
    } for m in model_names}
    
    final_data = {"meta_info": {"metric": metric, "judge_model": MODEL_NAME, "dropped_count": len(drops)}, "statistics_summary": summary, "samples_details": details}
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)
    if drops:
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(drops, f, indent=4, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs='+', required=True)
    parser.add_argument("--metric", type=str, required=True, choices=PROMPT_TEMPLATES.keys())
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--error_file", type=str, default=None)
    parser.add_argument("--video_base_dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    if not args.error_file:
        base, ext = os.path.splitext(args.output_file)
        args.error_file = f"{base}_drops{ext}"

    for fpath in [args.output_file, args.error_file]:
        fdir = os.path.dirname(os.path.abspath(fpath))
        if fdir and not os.path.exists(fdir):
            os.makedirs(fdir, exist_ok=True)
            print(f"[Info] The directory has been automatically created: {fdir}")

    model_files = {p.split("=")[0]: p.split("=")[1] for p in args.models}
    model_names = list(model_files.keys())

    details_list, dropped_list, processed_pids = [], [], set()
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                details_list = data.get("samples_details", [])
                processed_pids = {str(d["problem_id"]) for d in details_list}
                print(f"[Resume] Progress loaded, skipping {len(processed_pids)} processed samples.")
        except: pass

    all_aligned_data = load_and_align_data(model_files, [], sample_limit=args.limit)
    unified_data = [item for item in all_aligned_data if str(item["problem_id"]) not in processed_pids]
    
    if not unified_data:
        print("[Done] All samples have been processed.")
        return

    result_lock = threading.Lock()
    print(f"[Info] Starting evaluation. Concurrent threads: {MAX_WORKERS}. Samples pending: {len(unified_data)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        with tqdm(total=len(unified_data)) as pbar:
            future_to_item = {executor.submit(evaluate_sample, item, args.metric, model_names, args.video_base_dir): item for item in unified_data}
            
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                pid = item["problem_id"]
                try:
                    scores_map, error_reason = future.result()
                    with result_lock:
                        if error_reason:
                            dropped_list.append({"problem_id": pid, "error_reason": error_reason})
                        else:
                            details_list.append({"problem_id": pid, "metric_scores": scores_map})
                        
                        try:
                            save_checkpoint(args.output_file, args.error_file, details_list, dropped_list, args.metric, model_names)
                        except Exception as se:
                            print(f"\n [Write failed] {se}")
                except Exception as e:
                    print(f"\n [Runtime Crash] Sample {pid}: {e}")
                    dropped_list.append({"problem_id": pid, "error_reason": str(e)})
                pbar.update(1)

    print("\n" + "="*50 + "\nFinal Score Statistics:")
    for m in model_names:
        valid = [d for d in details_list if m in d["metric_scores"]]
        avg = sum(d["metric_scores"][m]["score"] for d in valid)/len(valid) if valid else 0
        print(f"{m:<20} | Average score: {avg:>6.2f} | Valid samples: {len(valid)}")

if __name__ == "__main__":
    main()

