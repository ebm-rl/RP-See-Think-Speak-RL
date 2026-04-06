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
import math
from copy import deepcopy
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional: remove proxy vars if your environment has problematic proxy settings
for key in ["ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"]:
    os.environ.pop(key, None)

current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(current_dir, "../r1-v/src/open_r1")
target_dir = os.path.abspath(target_dir)

if target_dir not in sys.path:
    sys.path.append(target_dir)

try:
    from movie_chara_profiles import resolve_profile, normalize_movie_name
    print("[Setup] Successfully imported 'movie_chara_profiles'.")
except ImportError:
    print("[Warn] Failed to import 'movie_chara_profiles'. Using mocks.")
    def resolve_profile(movie, name):
        return {}
    def normalize_movie_name(name):
        return name

API_KEY = "xxx"
BASE_URL = "xxx"
MODEL_NAME = "gpt-5-mini"

MAX_NUM_FRAMES = 8
FRAME_RESIZE = 512
MAX_JUDGE_RETRIES = 3
RETRY_DELAY = 1
MAX_WORKERS = 20


JUDGE_SYSTEM_PROMPT = (
    "You are an expert Role-Play Evaluation Critic performing pairwise preference comparisons. "
    "Your task is to compare TWO candidate responses and decide which one is better under the provided metric. "
    "Maintain absolute objectivity.\n"
    "IMPORTANT RULES:\n"
    "1. Candidate responses may contain reasoning, chain-of-thought, internal vision analysis, or extra tags. "
    "You MUST ignore all such meta-content.\n"
    "2. Evaluate ONLY the final spoken utterance of the character.\n"
    "3. Do NOT assign numeric scores.\n"
    "4. Do NOT rely on explicit score intervals such as 0-20, 21-40, etc.\n"
    "5. Instead, use the metric criteria and preference guidance below: "
    "prefer the response that more closely matches the qualities of a stronger response under this metric, "
    "and disprefer the response that more closely matches the qualities of a weaker response.\n"
    "6. Choose Tie only when the two responses are genuinely too close to distinguish under the metric."
)

COMMON_PROFILE_INSTRUCTION = (
    "**Note on Profiles**: Two profiles are provided below.\n"
    "- **[USER INFO]** is provided ONLY to help you understand the relationship and background of the conversation.\n"
    "- **[ASSISTANT INFO]** is the Target Persona. Your evaluation must focus on how well the model portrays this specific Assistant character."
)

METRIC_CRITERIA = {
    "Visual_Evidence_Grounding": """
**Metric Goal**:
Decide which response is better grounded in the visible scene as immediate in-world dialogue.

**Compare using these criteria**:
1. **Visual Triggering**: Does the dialogue contain a direct, logical link to a specific visible action, object, expression, or event?
2. **Object / Scene Integrity**: Does the response respect the physical limits of the scene and avoid unsupported visual claims?
3. **Temporal Realism**: Does the amount and style of speech fit the duration and urgency of the visible moment?

**Preference Guidance**:
Prefer responses that:
- react directly and logically to what is visible,
- stay constrained by the actual scene,
- fit the temporal window with concise and scene-appropriate speech.

Disprefer responses that:
- hallucinate objects, actions, or scene facts,
- rely on generic text-only role-play that could fit many scenes,
- become overly descriptive, bookish, or temporally unrealistic.
""",

    "Conversational_Naturalism": """
**Metric Goal**:
Decide which response sounds more like natural spoken dialogue rather than written prose.

**Compare using these criteria**:
1. **Oral Realism**: Is the phrasing colloquial, natural, and speech-like?
2. **Anti-Narrative Voice**: Does it avoid narrator-style prose, stage-direction dependence, and bookish monologue?

**Preference Guidance**:
Prefer responses that:
- sound like something a real person would say in the moment,
- are concise, oral, and conversational,
- avoid literary or essay-like delivery.

Disprefer responses that:
- read like narration or prose,
- sound overly formal, polished, or stiff,
- feel more written than spoken.
""",

    "Situational_Persona_Compatibility": """
**Metric Goal**:
Decide which response better expresses the character's identity under the specific visual pressure of the scene.

**Compare using these criteria**:
1. **Situational Value Expression**: Does the line express core identity through the current event rather than through static slogan-like behavior?
2. **Acting Realism & Rhythm**: Does it avoid overlong theatrical speech and remain plausible as spoken dialogue in the scene?
3. **Sensory-Driven Modulation**: Does the delivery feel shaped by what is visually happening right now?

**Preference Guidance**:
Prefer responses that:
- feel like believable in-scene performance,
- adapt persona dynamically under pressure,
- stay concise and physically plausible as spoken dialogue.

Disprefer responses that:
- behave like static caricatures,
- ignore the specific scene pressure,
- become fan-fiction-like, theatrical, or overlong.
""",

}

METRICS_NEED_VIDEO = [
    "Situational_Persona_Compatibility", "Visual_Evidence_Grounding"
]

PAIRWISE_PROMPT_TEMPLATE = """
You are comparing two candidate responses for the role of **{assistant_name}**.

Your judgment must be **preference-based, not score-based**:
- Use ONLY the metric criteria and preference guidance below.
- Do NOT assign any numeric score.
- Do NOT think in terms of explicit score intervals.
- Prefer the response that more closely matches the qualities of a stronger response under this metric.
- Disprefer the response that more closely matches the qualities of a weaker response under this metric.
- Choose **Tie** only when the two responses are genuinely too close to distinguish under the metric.

{profile_note}

**Profile Context**:
{context}

**Conversation History**:
{conversation}

**Real Answer (Ground Truth): Reference only**:
{gt_answer}

--- Metric: {metric_name} ---
{metric_criteria}

--- Candidate Responses (evaluate ONLY the final spoken dialogue, ignore all tags) ---

**[[Response A]]**:
{response_a}

**[[Response B]]**:
{response_b}

--- Your Task ---
Compare Response A and Response B based ONLY on the criteria and preference guidance above.

**Output Format (you MUST follow this exactly)**:
Winner: <A or B or Tie>
Reason: <1-3 sentences explaining why one is better, or why they are tied>
Confidence: <high or medium or low>
"""

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
    if not text:
        return "", False

    match = re.search(r"<answer>\s*(.*?)(?:</answer>|$)", text, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        return content, True

    clean_text = text
    clean_text = re.sub(r"<(think|vision)>.*?</\1>", "", clean_text, flags=re.DOTALL | re.IGNORECASE)
    clean_text = re.sub(r"</?(think|vision|answer)>", "", clean_text, flags=re.IGNORECASE)
    clean_text = clean_text.strip()
    return clean_text, False

def process_video_frames(video_path):
    if not os.path.exists(video_path):
        return None
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

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
            if not ret:
                break

            keep_frame = False
            if target_indices is None:
                if len(frames_base64) < MAX_NUM_FRAMES:
                    keep_frame = True
                else:
                    break
            else:
                if current_idx in target_indices:
                    keep_frame = True

            if keep_frame:
                h, w = frame.shape[:2]
                scale = FRAME_RESIZE / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame_resized = cv2.resize(frame, (new_w, new_h))
                _, buffer = cv2.imencode(".jpg", frame_resized)
                b64_str = base64.b64encode(buffer).decode("utf-8")
                frames_base64.append(b64_str)

            current_idx += 1
            if target_indices is not None and len(frames_base64) >= len(target_indices):
                break

        cap.release()
        if not frames_base64:
            return None
        return frames_base64

    except Exception:
        if cap:
            cap.release()
        return None

def call_judge_llm_text(messages):
    last_error_log = "Unknown API error"
    for attempt in range(MAX_JUDGE_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_completion_tokens=1024,
                timeout=120
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content, None
            last_error_log = "API returned empty content"
        except Exception as e:
            last_error_log = str(e)
            tqdm.write(f"\n[API attempt {attempt+1}/{MAX_JUDGE_RETRIES} failed]: {last_error_log}")
            if attempt < MAX_JUDGE_RETRIES - 1:
                time.sleep(RETRY_DELAY * 2)
    return None, last_error_log

def parse_pairwise_response(text):
    if not text:
        return None, "", ""

    winner_match = re.search(r"Winner:\s*(A|B|Tie)\b", text, re.IGNORECASE)
    if not winner_match:
        if re.search(r"Response\s+A\s+is\s+better", text, re.IGNORECASE):
            winner = "A"
        elif re.search(r"Response\s+B\s+is\s+better", text, re.IGNORECASE):
            winner = "B"
        elif re.search(r"\btie\b", text, re.IGNORECASE):
            winner = "Tie"
        else:
            return None, text.strip(), ""
    else:
        winner = winner_match.group(1).upper()
        if winner == "TIE":
            winner = "Tie"

    reason_match = re.search(r"Reason:\s*(.*?)(?=Confidence:|$)", text, re.DOTALL | re.IGNORECASE)
    reason = reason_match.group(1).strip() if reason_match else ""

    conf_match = re.search(r"Confidence:\s*(high|medium|low)", text, re.IGNORECASE)
    confidence = conf_match.group(1).lower() if conf_match else ""

    return winner, reason, confidence

def evaluate_pairwise_single(unified_item, metric_name, model_a_name, model_b_name,
                             video_base_dir=None, swap_order=False):
    context_str = _format_context_block(unified_item)
    dialogue_str = _format_dialogue(unified_item.get("dialogue", []))
    gt_answer = unified_item.get("target", {}).get("utterance", "")
    assistant_name = unified_item.get("assistant", "Assistant")

    raw_video_path = unified_item.get("video_path", "")
    video_path = raw_video_path
    if video_base_dir and raw_video_path and not os.path.isabs(raw_video_path):
        video_path = os.path.join(video_base_dir, raw_video_path)

    raw_a = unified_item.get(f"res_{model_a_name}", {}).get("raw_output", "") or ""
    raw_b = unified_item.get(f"res_{model_b_name}", {}).get("raw_output", "") or ""

    content_a, _ = extract_answer_tag_robust(raw_a)
    content_b, _ = extract_answer_tag_robust(raw_b)

    content_a = content_a.strip() or "(EMPTY RESPONSE)"
    content_b = content_b.strip() or "(EMPTY RESPONSE)"

    if swap_order:
        prompt_response_a = content_b
        prompt_response_b = content_a
    else:
        prompt_response_a = content_a
        prompt_response_b = content_b

    video_frames = None
    if metric_name in METRICS_NEED_VIDEO:
        video_frames = process_video_frames(video_path)
        if not video_frames:
            return None, "", "", f"Video file missing or unreadable: {video_path}"

    metric_criteria = METRIC_CRITERIA.get(metric_name, "")
    user_prompt = PAIRWISE_PROMPT_TEMPLATE.format(
        assistant_name=assistant_name,
        profile_note=COMMON_PROFILE_INSTRUCTION,
        context=context_str,
        conversation=dialogue_str,
        gt_answer=gt_answer,
        metric_name=metric_name,
        metric_criteria=metric_criteria,
        response_a=prompt_response_a,
        response_b=prompt_response_b,
    )

    messages = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}]
    input_content = []

    if video_frames:
        for b64 in video_frames:
            input_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "low"
                }
            })

    input_content.append({"type": "text", "text": user_prompt})
    messages.append({"role": "user", "content": input_content})

    judge_output, api_error = call_judge_llm_text(messages)
    if not judge_output:
        return None, "", "", f"Judge API failed: {api_error}"

    winner_label, reason, confidence = parse_pairwise_response(judge_output)
    if winner_label is None:
        return None, reason, confidence, f"Failed to parse winner: {judge_output[:300]}"

    if winner_label == "Tie":
        winner_real = "Tie"
    elif swap_order:
        winner_real = model_b_name if winner_label == "A" else model_a_name
    else:
        winner_real = model_a_name if winner_label == "A" else model_b_name

    return winner_real, reason, confidence, None

def evaluate_pairwise_with_swap(unified_item, metric_name, model_a_name, model_b_name,
                                video_base_dir=None):
    w1, r1, c1, e1 = evaluate_pairwise_single(
        unified_item, metric_name, model_a_name, model_b_name,
        video_base_dir=video_base_dir, swap_order=False
    )
    if e1:
        return None, {"error_run1": e1}, e1

    w2, r2, c2, e2 = evaluate_pairwise_single(
        unified_item, metric_name, model_a_name, model_b_name,
        video_base_dir=video_base_dir, swap_order=True
    )
    if e2:
        return w1, {
            "run1": {"winner": w1, "reason": r1, "confidence": c1},
            "run2": {"error": e2},
            "agreement": "run2_failed"
        }, None

    details = {
        "run1": {
            "winner": w1,
            "reason": r1,
            "confidence": c1,
            "order": f"A={model_a_name}, B={model_b_name}"
        },
        "run2": {
            "winner": w2,
            "reason": r2,
            "confidence": c2,
            "order": f"A={model_b_name}, B={model_a_name}"
        },
    }

    if w1 == w2:
        final_winner = w1
        details["agreement"] = "consistent"
    elif w1 == "Tie" or w2 == "Tie":
        final_winner = w1 if w2 == "Tie" else w2
        details["agreement"] = "partial_tie"
    else:
        final_winner = "Tie"
        details["agreement"] = "contradictory_to_tie"

    details["final_winner"] = final_winner
    return final_winner, details, None

def load_and_align_data(model_files, sample_limit=100):
    raw_data_map = {}
    all_pids_sets = []
    print("[Data] Aligning model outputs...")

    for m_name, f_path in model_files.items():
        if not os.path.exists(f_path):
            print(f"[Error] Missing model file: {f_path}")
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
            print(f"   - {m_name}: loaded {len(current_model_pids)} samples")

    if not all_pids_sets:
        return []

    common_pids = sorted(list(set.intersection(*all_pids_sets)))
    print(f"[Data] Common samples across all models: {len(common_pids)}")

    random.shuffle(common_pids)
    if len(common_pids) > sample_limit:
        common_pids = common_pids[:sample_limit]

    aligned_list = []
    for pid in common_pids:
        first_model = list(model_files.keys())[0]
        base_item = raw_data_map[first_model][pid]
        unified_item = {
            k: base_item.get(k)
            for k in ["video_path", "data_source", "user", "assistant", "dialogue", "target"]
        }
        unified_item["problem_id"] = pid
        for m_name in model_files.keys():
            unified_item[f"res_{m_name}"] = {
                "raw_output": raw_data_map[m_name][pid].get(
                    "raw_output",
                    raw_data_map[m_name][pid].get("model_prediction", "")
                )
            }
        aligned_list.append(unified_item)

    return aligned_list

def exact_binomial_two_sided_pvalue(wins, losses):
    """
    Two-sided exact binomial test under H0: p=0.5, ignoring ties.
    """
    n = wins + losses
    if n == 0:
        return None

    observed_pmf = math.comb(n, wins) * (0.5 ** n)
    p_value = 0.0
    eps = 1e-15

    for k in range(n + 1):
        pmf_k = math.comb(n, k) * (0.5 ** n)
        if pmf_k <= observed_pmf + eps:
            p_value += pmf_k

    return min(1.0, p_value)

def compute_win_rate_table(pair_results, anchor, opponents):
    table = {}

    for opp in opponents:
        pair_key = f"{anchor}_vs_{opp}"
        if pair_key not in pair_results:
            continue

        results = pair_results[pair_key]
        total = len(results)
        if total == 0:
            continue

        anchor_wins = sum(1 for r in results if r["final_winner"] == anchor)
        opp_wins = sum(1 for r in results if r["final_winner"] == opp)
        ties = sum(1 for r in results if r["final_winner"] == "Tie")
        errors = sum(1 for r in results if r["final_winner"] is None)

        valid = total - errors
        decisive = anchor_wins + opp_wins
        decisive_win_rate = round(anchor_wins / decisive * 100, 1) if decisive > 0 else None
        p_value = exact_binomial_two_sided_pvalue(anchor_wins, opp_wins)

        table[opp] = {
            "anchor_wins": anchor_wins,
            "opponent_wins": opp_wins,
            "ties": ties,
            "errors": errors,
            "total": total,
            "valid": valid,
            "anchor_win_rate": round(anchor_wins / valid * 100, 1) if valid > 0 else 0.0,
            "opponent_win_rate": round(opp_wins / valid * 100, 1) if valid > 0 else 0.0,
            "tie_rate": round(ties / valid * 100, 1) if valid > 0 else 0.0,
            "net_win_rate": round((anchor_wins - opp_wins) / valid * 100, 1) if valid > 0 else 0.0,
            "decisive_win_rate": decisive_win_rate,
            "p_value_two_sided": p_value,
        }

    return table

def save_results(output_file, pair_results, win_rate_table, anchor, metric_name):
    output_data = {
        "meta_info": {
            "metric": metric_name,
            "judge_model": MODEL_NAME,
            "anchor_model": anchor,
            "evaluation_type": "pairwise_preference_with_position_swap",
        },
        "win_rate_table": win_rate_table,
        "pair_details": pair_results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

def format_p_value(p):
    if p is None:
        return "NA"
    if p >= 0.001:
        return f"{p:.4f}"
    return f"{p:.2e}"

def print_win_rate_table(win_rate_table, anchor, metric_name):
    print(f"\n{'=' * 122}")
    print(f"  Preference-based Pairwise Win Rate — Metric: {metric_name}")
    print(f"  Anchor Model: {anchor}")
    print(f"{'=' * 122}")
    print(f"  {'Opponent':<22} {'Win':>7} {'Loss':>7} {'Tie':>7} {'Net':>8} {'Decisive':>11} {'p-value':>12} {'Counts(W/L/T)':>18}")
    print(f"  {'-' * 110}")

    for opp, stats in win_rate_table.items():
        decisive_str = "NA" if stats["decisive_win_rate"] is None else f"{stats['decisive_win_rate']:.1f}%"
        counts_str = f"{stats['anchor_wins']}/{stats['opponent_wins']}/{stats['ties']}"
        print(
            f"  {opp:<22} "
            f"{stats['anchor_win_rate']:>6.1f}% "
            f"{stats['opponent_win_rate']:>6.1f}% "
            f"{stats['tie_rate']:>6.1f}% "
            f"{stats['net_win_rate']:>7.1f}% "
            f"{decisive_str:>11} "
            f"{format_p_value(stats['p_value_two_sided']):>12} "
            f"{counts_str:>18}"
        )
    print(f"{'=' * 122}\n")

def main():
    parser = argparse.ArgumentParser(description="Preference-based pairwise win rate evaluation")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model name=filepath pairs, e.g. EBM=/path/to/ebm.json")
    parser.add_argument("--metric", type=str, required=True,
                        choices=list(METRIC_CRITERIA.keys()))
    parser.add_argument("--anchor", type=str, required=True,
                        help="Anchor model name (must match one of --models names)")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--video_base_dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=500,
                        help="Max samples to evaluate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sample shuffle")
    args = parser.parse_args()

    random.seed(args.seed)

    model_files = {}
    for p in args.models:
        parts = p.split("=", 1)
        if len(parts) != 2:
            print(f"Format error: {p}. Expected ModelName=/path/to/file.json")
            return
        model_files[parts[0]] = parts[1]

    model_names = list(model_files.keys())
    if args.anchor not in model_names:
        print(f"Anchor model '{args.anchor}' is not in --models. Available: {model_names}")
        return

    opponents = [m for m in model_names if m != args.anchor]
    if not opponents:
        print("At least 2 models are required for pairwise comparison.")
        return

    out_dir = os.path.dirname(os.path.abspath(args.output_file))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    all_data = load_and_align_data(model_files, sample_limit=args.limit)
    if not all_data:
        print("No usable data found.")
        return

    print(f"\n[Info] Anchor: {args.anchor}")
    print(f"[Info] Opponents: {opponents}")
    print(f"[Info] Metric: {args.metric}")
    print(f"[Info] Samples: {len(all_data)}")
    print(f"[Info] Total API calls: ~{len(all_data) * len(opponents) * 2} (pairwise × 2 for order swap)")

    pair_results = {}
    processed_pids_per_pair = {}

    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
                pair_results = existing.get("pair_details", {})
                for pair_key, results_list in pair_results.items():
                    processed_pids_per_pair[pair_key] = {str(r["problem_id"]) for r in results_list}
                total_done = sum(len(v) for v in pair_results.values())
                print(f"[Resume] Loaded existing progress. Skipping {total_done} completed comparisons.")
        except Exception as e:
            print(f"[Resume] Failed to read existing file: {e}. Starting from scratch.")

    result_lock = threading.Lock()

    for opp in opponents:
        pair_key = f"{args.anchor}_vs_{opp}"
        if pair_key not in pair_results:
            pair_results[pair_key] = []

        existing_pids = processed_pids_per_pair.get(pair_key, set())
        pending_items = [item for item in all_data if str(item["problem_id"]) not in existing_pids]

        if not pending_items:
            print(f"\n[Skip] {pair_key}: all samples already completed")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Evaluating: {args.anchor} vs {opp}")
        print(f"  Pending samples: {len(pending_items)}")
        print(f"{'=' * 60}")

        def _eval_one_pair(item, anchor=args.anchor, opponent=opp, metric=args.metric, vdir=args.video_base_dir):
            winner, details, error = evaluate_pairwise_with_swap(
                item, metric, anchor, opponent, video_base_dir=vdir
            )
            return item["problem_id"], winner, details, error

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(_eval_one_pair, item): item for item in pending_items}

            with tqdm(total=len(pending_items), desc=f"{args.anchor} vs {opp}") as pbar:
                for future in as_completed(futures):
                    try:
                        pid, winner, details, error = future.result()

                        result_entry = {
                            "problem_id": str(pid),
                            "final_winner": winner,
                            "details": details,
                        }
                        if error:
                            result_entry["error"] = error

                        with result_lock:
                            pair_results[pair_key].append(result_entry)

                            if len(pair_results[pair_key]) % 10 == 0:
                                try:
                                    win_table = compute_win_rate_table(pair_results, args.anchor, opponents)
                                    save_results(args.output_file, pair_results, win_table, args.anchor, args.metric)
                                except Exception as se:
                                    tqdm.write(f"[Save Error] {se}")

                    except Exception as e:
                        tqdm.write(f"[Runtime Error] {e}")

                    pbar.update(1)

        win_table = compute_win_rate_table(pair_results, args.anchor, opponents)
        save_results(args.output_file, pair_results, win_table, args.anchor, args.metric)

    win_table = compute_win_rate_table(pair_results, args.anchor, opponents)
    save_results(args.output_file, pair_results, win_table, args.anchor, args.metric)
    print_win_rate_table(win_table, args.anchor, args.metric)

if __name__ == "__main__":
    main()
