# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

from datasets import Dataset, DatasetDict

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

import requests
import time

import torch
from transformers import CLIPProcessor, CLIPModel
from movie_chara_profiles import resolve_profile, normalize_movie_name


    
# # 改为八卡的时候要删除下方这一小块内容
# import torch
# torch.backends.cudnn.benchmark = False
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.deterministic = False



# ==================== 权重配置区 ====================
# CLIP_REWARD_WEIGHT = 1.0   # 画面对齐的权重 (可以先从 0.5 ~ 1.0 试起)
CLIP_SERVER_URL = "http://127.0.0.1:5001/score"

# BERT_REWARD_WEIGHT = 4.0   # 文本准确性的权重 (建议 3.0 ~ 5.0)
BERT_SERVER_URL = "http://127.0.0.1:5000/score"

# # 长度惩罚参数 (只在 Client 端生效)
# LEN_SAFE_LIMIT = 60
# LEN_HARD_LIMIT = 70
# ===================================================

# ================== 全局格式约束 ==================
# 严格要求：必须是 <vision>...</vision><think>...</think><answer>...</answer> 结构
# 且首尾允许空白，中间不允许夹杂其他内容
# STRICT_PATTERN = r"^\s*<vision>.*?</vision>\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
# ANSWER_PATTERN = r"<answer>\s*(.*?)\s*</answer>"
STRICT_PATTERN = r"^\s*<vision>.*?</vision>\s*<think>.*?</think>\s*<answer>.*?</answer>"
ANSWER_PATTERN = r"<answer>\s*(.*?)\s*</answer>"

def _truncate_after_end_tag(text: str, end_tag: str = "</answer>") -> str:
    """双保险：确保 Reward 计算时看到的也是干净的文本"""
    if not isinstance(text, str):
        return text
    idx = text.lower().find(end_tag)
    if idx < 0:
        return text.strip()
    return text[: idx + len(end_tag)].strip()
# =================================================

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "clip"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format', 'clip'."},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    temporal: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using temporal GRPO"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )


class CLIPRewardClient:
    """
    负责维护状态、发请求、计算最终奖励逻辑
    """
    def __init__(self):
        # self.momentum = 0.99 
        # self.clip_mean = None
        # self.clip_var = None
        
        # # ================== 【新增开始】日志初始化 ==================
        # import torch.distributed as dist
        
        # # 定义日志目录
        # self.log_dir = "/data/wm/Video-R1/src/r1-v/src/open_r1/debug_log"
        # os.makedirs(self.log_dir, exist_ok=True)

        # # 获取 Rank，区分不同 GPU 的日志文件
        # if dist.is_initialized():
        #     self.rank = dist.get_rank()
        # else:
        #     self.rank = 0
            
        # # 定义文件名，例如: debug_log_rank_CLIP_0.txt
        # self.log_path = f"{self.log_dir}/debug_log_rank_CLIP_{self.rank}.txt"
        
        # # 读取 DEBUG 模式开关
        # self.debug_mode = os.getenv("DEBUG_MODE", "").lower().strip() in ["true", "1", "yes"]
        # # ================== 【新增结束】 ==================
        pass

    # ================== 【新增开始】辅助函数 ==================
    def _write_debug(self, msg):
        """写入日志文件的底层函数"""
        if not self.debug_mode:
            return
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception as e:
            print(f"[CLIP Client Rank {self.rank}] Logging Error: {e}")

    def _get_full_vision_content(self, text):
        """提取完整的 <vision> 内容（不做截断），专门用于日志记录"""
        pattern = re.compile(r"<vision>(.*?)</vision>", re.DOTALL)
        content = text[0]['content'] if isinstance(text, list) else str(text)
        match = pattern.search(content)
        if match:
            return match.group(1).strip()
        return "[No <vision> tag detected]"

    def _log_batch(self, completions, scores, problem_ids):
        """批量格式化并写入日志的核心逻辑"""
        if not self.debug_mode:
            return

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        self._write_debug("====================================================")
        self._write_debug(f"[Rank {self.rank}] CLIP REWARD LOG | Time: {now_str}")
        
        if not problem_ids:
            problem_ids = ["N/A"] * len(completions)
        
        for idx, (completion, score) in enumerate(zip(completions, scores)):
            pid = problem_ids[idx] if idx < len(problem_ids) else "Unknown"
            vision_content = self._get_full_vision_content(completion)
            
            self._write_debug(f"Batch_Idx     = {idx}")
            self._write_debug(f"PROBLEM_ID    = {pid}")
            self._write_debug(f"VISION_CONTENT= {vision_content}")
            self._write_debug(f"CLIP_REWARD   = {score:.6f}")
            self._write_debug("-" * 30)
    # ================== 【新增结束】 ==================

    def _parse_vision(self, text_list):
        parsed = []
        lengths = []
        mask = []
        pattern = re.compile(r"<vision>(.*?)</vision>", re.DOTALL)
        for t in text_list:
            content = t[0]['content'] if isinstance(t, list) else str(t)
            match = pattern.search(content)
            if match and match.group(1).strip():
                txt = match.group(1).strip()
                lengths.append(len(txt.split()))
                # parsed.append(txt[:350])
                parsed.append(txt)
                mask.append(True)
            else:
                parsed.append("")
                lengths.append(0)
                mask.append(False)
        return parsed, lengths, mask

    def __call__(self, completions, **kwargs):
        final_scores = [0.0] * len(completions)
        
        embed_paths = kwargs.get('clip_embed_path', [])
        problem_ids = kwargs.get('id', kwargs.get('problem_id', []))
        
        if not embed_paths:
            # self._log_batch(completions, final_scores, problem_ids)
            return final_scores

        texts, lengths, mask = self._parse_vision(completions)
        
        valid_indices = [i for i, m in enumerate(mask) if m and embed_paths[i]]
        valid_texts = [texts[i] for i in valid_indices]
        valid_paths = [embed_paths[i] for i in valid_indices]
        
        # final_scores = [0.0] * len(completions)
        if not valid_indices:
            # self._log_batch(completions, final_scores, problem_ids) 
            return final_scores

        try:
            # === 发送请求给 Server ===
            resp = requests.post(
                CLIP_SERVER_URL, 
                json={"texts": valid_texts, "embed_paths": valid_paths, "indices": valid_indices},
                timeout=30
            )
            if resp.status_code == 200:
                raw_scores = resp.json().get("raw_scores", [])
            else:
                print(f"[CLIP Client] Server Error: {resp.status_code}")
                # self._log_batch(completions, final_scores, problem_ids)
                return final_scores
        except Exception as e:
            print(f"[CLIP Client] Connection Failed: {e}")
            # self._log_batch(completions, final_scores, problem_ids)
            return final_scores

        # === Client 端后处理 (EMA + Norm + Penalty) ===
        if raw_scores:           
            for k, original_idx in enumerate(valid_indices):
                final_scores[original_idx] = float(raw_scores[k])
        # self._log_batch(completions, final_scores, problem_ids)
        
        return final_scores

# 注册 Wrapper
_GLOBAL_CLIP_CLIENT = None
def clip_reward_wrapper(prompts, completions, **kwargs):
    global _GLOBAL_CLIP_CLIENT
    if _GLOBAL_CLIP_CLIENT is None:
        _GLOBAL_CLIP_CLIENT = CLIPRewardClient()
    
    return _GLOBAL_CLIP_CLIENT(completions, **kwargs)



def accuracy_reward(completions, solution, **kwargs): # 计算的是一个样本的G个生成的reward
    import torch.distributed as dist
    import os
    from datetime import datetime
    import re
    from rouge_score import rouge_scorer
    import json
    import subprocess
    import uuid
    
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # # -----------------------------
    # # (1) Rank & 日志文件路径
    # # -----------------------------
    # log_dir = "/data/wm/Video-R1/src/r1-v/src/open_r1/debug_log"
    # os.makedirs(log_dir, exist_ok=True)

    # # 单卡 / 多卡自动区分
    # if dist.is_initialized():
    #     log_path = f"{log_dir}/debug_log_rank_NEW_BERTScore{rank}.txt"   # 多卡时每个 rank 单独一个文件，避免冲突
    # else:
    #     log_path = f"{log_dir}/debug_log_single_gpu.txt"

    # # 只读环境变量，不覆盖路径
    # DEBUG = os.getenv("DEBUG_MODE", "").lower().strip() in ["true", "1", "yes"]

    # # 写日志函数（带异常打印）
    # def write_debug(msg):
    #     if not DEBUG:
    #         return
    #     try:
    #         with open(log_path, "a", encoding="utf-8") as f:
    #             f.write(msg + "\n")
    #     except Exception as e:
    #         print(f"[Rank {rank}] LOGGING ERROR:", e)
    #         print(f"[Rank {rank}] Tried writing to:", log_path)
            
    # # 尝试从 kwargs 中获取 id 或 problem_id
    # problem_ids = kwargs.get('id', kwargs.get('problem_id', []))
    # # 如果没获取到，填充 N/A
    # if not problem_ids:
    #     problem_ids = ["N/A"] * len(completions)

    # # -----------------------------
    # # 打印 enter 信息
    # # -----------------------------
    # write_debug("====================================================")
    # write_debug(f"[Rank {rank}] ENTER accuracy_reward()")
    # write_debug(f"Time = {datetime.now()}")
    # write_debug(f"problem_type = {kwargs.get('problem_type')}")
    # write_debug(f"Problem IDs batch = {problem_ids}") 
    
    # ------- 统一的正则 -------
    full_pattern = re.compile(STRICT_PATTERN, re.DOTALL)
    ans_pattern = re.compile(ANSWER_PATTERN, re.DOTALL)

    def extract_answer(text: str) -> str:
        m = ans_pattern.search(text)
        return m.group(1).strip() if m else ""



    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m+1):
            d[i][0] = i
        for j in range(n+1):
            d[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if ref_words[i-1] == hyp_words[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    d[i][j] = 1 + min(d[i-1][j], d[i][j-1], d[i-1][j-1])
        return d[m][n] / max(1, m)


    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=use_stemmer)
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (scores['rouge1'].fmeasure + scores['rouge2'].fmeasure + scores['rougeL'].fmeasure) / 3
        return average_fmeasure
    

    question_type = kwargs['problem_type'][0]
    
    contents = [completion[0]["content"] for completion in completions]
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    
    ############# New BERTScore 计算 #############
    final_scores = [0.0] * len(contents)
    if question_type == "free-form":
        outputs = []
        gts = []
        valid_indices = []
        
        # 预处理
        for idx, (content, sol) in enumerate(zip(contents, solution)):           
            # 1. 格式硬门槛：没通过 STRICT_PATTERN 的，直接 0 分
            if not full_pattern.fullmatch(content):
                # final_scores[idx] = 0.0
                continue
                
            out_ans = extract_answer(content)
            gt_ans = extract_answer(sol)
            
            # 2. 答案非空检查
            if out_ans and gt_ans:
                outputs.append(out_ans)
                gts.append(gt_ans)
                valid_indices.append(idx)

        if outputs:           
            try:
                response = requests.post(
                    BERT_SERVER_URL, 
                    json={"cands": outputs, "refs": gts},
                    timeout=60 
                )
                
                if response.status_code == 200:
                    scores = response.json().get("scores", [])
                    
                    if len(scores) == len(outputs):
                        for list_idx, score in enumerate(scores):
                            final_scores[valid_indices[list_idx]] = score
                    else:
                        print(f"[Rank {rank}] Server returned mismatching scores.")
                else:
                    print(f"[Rank {rank}] Server returned status: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                # 如果连接失败，打印错误并返回 0 分，避免训练崩溃
                # 这里打印比较重要，提示你可能忘开 server 了
                print(f"[Rank {rank}] BERTScore Server Connection Failed: {e}")
                print(f"[Rank {rank}] MAKE SURE you started bertscore_server.py!")
    
    
    # # 无论是否 free-form，无论是否计算成功，都遍历一遍记录状态
    # for idx, (content, sol) in enumerate(zip(contents, solution)):
        
    #     # 获取当前样本对应的 problem_id
    #     # 因为 completions 是 list，kwargs['id'] 也是 list，它们的索引是一一对应的
    #     pid = problem_ids[idx] if idx < len(problem_ids) else "Unknown"

    #     output_ans = extract_answer(content)
    #     gt_ans = extract_answer(sol)
    #     score = final_scores[idx]

    #     write_debug(f"[{now}] Batch_Idx={idx} | Rank={rank}")
    #     write_debug(f"PROBLEM_ID = {pid}")   # <--- 重点：记录 Problem ID
    #     write_debug(f"GT         = {gt_ans}")   
    #     write_debug(f"PRED       = {output_ans}")   
    #     write_debug(f"REWARD     = {score:.6f}") # 记录当前样本的具体分数
    #     write_debug("----------------------------------------------------")
        
    return final_scores



# def format_reward(completions, **kwargs):
#     """
#     格式检查：必须严格匹配 STRICT_PATTERN。
#     返回 1.0 (合格) 或 0.0 (不合格)。
#     """
#     pattern = re.compile(STRICT_PATTERN, re.DOTALL)
#     completion_contents = [c[0]["content"] if isinstance(c, list) else c for c in completions]
#     matches = [pattern.fullmatch(content) for content in completion_contents]
#     return [1.0 if m else 0.0 for m in matches]

def format_reward(completions, **kwargs):
    """
    格式检查：与 Trainer 的 Quality Mask 逻辑对齐。
    1. 先截断垃圾尾巴
    2. 再正则匹配
    返回 1.0 (合格) 或 0.0 (不合格)。
    """
    pattern = re.compile(STRICT_PATTERN, re.DOTALL)
    scores = []
    
    for c in completions:
        # 提取文本内容
        content = c[0]["content"] if isinstance(c, list) else c
        
        # [关键步骤] 先截断，与 Trainer 内部逻辑保持一致
        clean_content = _truncate_after_end_tag(content)
        
        # 使用 match 而不是 fullmatch (配合去掉了 $ 的正则)
        # 只要开头对，中间全，就给分
        if pattern.match(clean_content):
            scores.append(1.0)
        else:
            scores.append(0.0)
            
    return scores


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "clip": clip_reward_wrapper,
}

# SYSTEM_PROMPT = (
#     "You are an expert Role-Play Dialogue AI.\n"
#     "Your goal is to immerse yourself in a specific character (the ASSISTANT) and respond to a USER, "
#     "based on a video scene and dialogue context.\n"
#     "\n"
#     "### 1. SCENE PRESENCE & REALITY\n"
#     "The video represents the immediate reality unfolding **right in front of you**.\n"
#     "- **Your Position**: You are strictly a participant in this scene. You (or the User) might be **visible in the frame**, "
#     "OR you (or the User) might be **off-camera**. **Crucially, even if off-camera, you and the User are fully involved in the scene—witnessing the events directly and potentially conversing with those inside the frame.**\n"
#     "- **The Connection**: The Video provides the *Stimulus* (Visuals). The Dialogue is your *Reaction*.\n"
#     "\n"
#     "### 2. CORE LOGIC: THE PERSONALITY FILTER\n"
#     "Do NOT just output generic emotions. You must filter the observable reality through your Character Profile:\n"
#     "- **Standard Rule**: A humorous character deflects fear with jokes; a warrior faces it with grit. (Same input -> Different output based on ID).\n"
#     "- **High-Stakes Rule (Crucial)**: Adjust for the *intensity* of the atmosphere.\n"
#     "  * If a humorous character faces **mortal danger**, they don't tell stand-up jokes -> their humor becomes nervous or they show unexpected bravery.\n"
#     "  * If a wise character faces **tragedy**, they don't lecture -> their wisdom becomes gentle silence.\n"
#     "\n"
#     "### OUTPUT FORMAT (Strict Step-by-Step)\n"
#     "You must output XML-like tags in this exact order:\n"
#     "\n"
#     "<vision>\n"
#     "Describe the immediate reality. Focus on tangible drivers for conversation:\n"
#     "1. **Setting, Lighting & Atmosphere**: \n"
#     "   - If details are clear: Describe the location using environment and weather.\n"
#     "   - If unclear, dark, or generic: Focus strictly on **Color Tone**, **Brightness**, and **Visibility** (e.g., 'pitch black', 'dim warm candlelight', 'harsh white glare', 'blurry surroundings'). Do NOT invent furniture/weather/environment if invisible.\n"
#     "2. **Visual Figures & Actions**: \n"
#     "   - **Identity Rule**: The figures visible in the video **might NOT be** the current User or Assistant. They could be third parties who also in this scene together.\n"
#     "   - **Identify by Trait ONLY**: Describe meaningful figures by count/gender/appearance (e.g., 'a tall bearded figure', 'a young person in robes', 'a group of soldiers'). **Do NOT map them to the User/Assistant unless certain.**\n"
#     "   - **Focus on Tension/Action**: Describe what is happening (e.g., 'one character torturing another', 'two people arguing', 'one pointing a weapon', 'looking terrified', 'running away'). This context matters more than who they are.\n"
#     "3. **Manifested Emotions**: What raw feelings are freely on display in the video? (e.g., tense silence, crying, laughter).\n"
#     "4. **Key Objects/Dialogue Topics**: Identify pivotal items that define the subject (e.g., the One Ring, a diary, a visible wound).\n"
#     "</vision>\n"
#     "\n"
#     "<think>\n"
#     "Apply the 'Personality Filter' to the Vision:\n"
#     "1. **Analyze Situation Intensity**: Based on <vision>, is this a normal conversation, a tense moment, or a life-or-death crisis?\n"
#     "2. **Profile Integration**:\n"
#     "   - If Normal/Tense: How does the Assistant typically respond? (e.g., 'I will make a joke to ease the tension').\n"
#     "   - If Extreme/Strange: How does their coping mechanism shift? (e.g., 'It is too dangerous for jokes, I must be brave/alert').\n"
#     "3. **Drafting**: Plan the reply to the User, ensuring the tone matches above analysis.\n"
#     "</think>\n"
#     "\n"
#     "<answer>\n"
#     "The final natural spoken line by the Assistant. No speaker name. No quotes.\n"
#     "</answer>"
# )



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
    "   - Internalize {assistant_name}'s mindset. Apply the **THE PERSONALITY ANALYSIS** (e.g., humor turns to bravery in danger) while thinking as the assistant. Based on the current situation and the topic of the conversation, and considering the personality and tone of the assistant, think about possible directions and responses for the assistant's next sentence.\n"   
    "   - *Check*: If the Vision is dangerous, does the character show bravery/nervousness instead of casual traits?\n"
    
    "</think>\n"
    "\n"
    "<answer>\n"
    "The final natural spoken line by the Assistant. No speaker name. No quotes.\n"
    "</answer>"
)



def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
        
    
    def _format_character_block(movie: str, name: str) -> str:
        """
        读取角色 profile（按 movie + name），并把 profile dict 里所有字段按原顺序输出。
        """
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
        """
        dialogue_list: List[{'speaker': str, 'utterance': str}]
        """
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

        # 简化的 Block 构建
        user_block = _format_character_block(movie, user_name)
        assistant_block = _format_character_block(movie, assistant_name)
        dialogue_block = _format_dialogue_block(example["dialogue"])

        # text = (
        #     f"You are role-playing as **{assistant_name}** from the universe of '{movie}'.\n"
        #     "**Context**: You are physically present in the scene, witnessing/experiencing the events shown below.\n\n"
        #     "### CHARACTER PROFILES\n"
        #     f"USER ({user_name}):\n{user_block}\n\n"
        #     f"ASSISTANT ({assistant_name}):\n{assistant_block}\n\n"
        #     "### INPUT CONTEXT\n"
        #     "**[Visual Reality]**: The raw recording of the immediate event. Contains Actions, Objects, and Atmosphere.\n"
        #     "**[Dialogue Context]**:\n"
        #     f"{dialogue_block}\n"
        #     "### INSTRUCTION\n"
        #     f"Generate the next line for **{assistant_name}**.\n"
        #     "1. **Step 1 <vision>...</vision> (Camera Mode)**: Strictly describe **observable facts** ONLY inside these tags. \n"
        #     "   - List specific physical actions and visible expressions.\n"
        #     "   - Identify key objects.\n"
        #     "   - **NO ANALYSIS OR GUESSING HERE.** Just report what is seen.\n"
        #     "2. **Step 2 <think>...</think> (Analytic Mode)**: Decode the event inside these tags.\n"
        #     "   - **Analyze the Clues**: Use the *Key Objects* and *Action Intensity* from Vision to determine **what event is exactly happening that see just now**.\n"
        #     "   - **Synthesize with Dialogue**: Combine this Event Analysis with the [Dialogue History] to define the precise **Topic**.\n"
        #     f"   - **Thinking as assistant**: **Internalize {assistant_name}'s mindset.** Apply the **High-Stakes Personality Rule** (e.g., humor turns to bravery in danger) while thinking as the assistant. Based on the current situation and the topic of the conversation, and considering the personality and tone of the assistant, think about possible directions and responses for the assistant's next sentence.\n"
        #     f"3. **Step 3 <answer>...</answer>**: Reply to {user_name} in character. Apply the **High-Stakes Personality Rule** (e.g., humor turns to bravery in danger).\n\n"
        #     "**Remember to close all tags.** Start immediately with <vision>..."
        # )
        
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
        return text

    def make_conversation_roleplay(example):
        """
        构造 Qwen2VL 所需的 multi-modal prompt:
        - 对于 video 数据：content 里先给一个 'video' 占位，再给文本 prompt
        - 对于纯文本数据：只给文本 prompt
        """
        # import pdb; pdb.set_trace()
        prompt_text = _build_roleplay_prompt_text(example)
        data_type = example.get("data_type", None)

        system_msg = {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT}
            ]
        }

        if data_type in ("image", "video"):
            user_msg = {
                "role": "user",
                "content": [
                    {"type": data_type},
                    {"type": "text", "text": prompt_text}
                ]
            }
        else:
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text}
                ]
            }

        return {"prompt": [system_msg, user_msg]}


    # dataset = dataset.map(make_conversation_image_and_video)
    dataset = dataset.map(make_conversation_roleplay)


    # import pdb; pdb.set_trace()
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    
    # import pdb; pdb.set_trace()
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    # import pdb; pdb.set_trace()
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
