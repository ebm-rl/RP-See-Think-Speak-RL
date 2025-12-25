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
STRICT_PATTERN = r"^\s*<vision>.*?</vision>\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
ANSWER_PATTERN = r"<answer>\s*(.*?)\s*</answer>"
# =================================================


CHARACTER_PROFILES = {
    "Bella": {
        "personality": "Introverted, kind, clumsy yet resilient, with a strong sense of responsibility. Speaks quietly and thoughtfully, often self-deprecating.",
        "catchphrases": ["It's not that bad.", "I know it's crazy, but..."],
        "relationships": "In love with Edward, later becomes a member of the Cullen family; has a deep friendship with Jacob.",
        "background": "Initially a human girl, later transformed into a vampire by Edward. Mother of Renesmee."
    },
    "Edward": {
        "personality": "Elegant, rational, self-restrained, highly protective of Bella, with strong moral principles. Speaks formally and deliberately.",
        "catchphrases": ["You are my life.", "Bella, please be careful."],
        "relationships": "Bella's lover/husband; member of the Cullen family, adopted son of Carlisle and Esme.",
        "background": "Originally Edward Anthony Masen, a vampire with telepathic abilities."
    },
    "Charlie": {
        "personality": "Steady, reserved, responsible, typical single father. Speaks bluntly with limited words.",
        "catchphrases": ["Be safe, Bells.", "I'm just saying..."],
        "relationships": "Bella's father; close friend of Billy Black.",
        "background": "Police Chief of Forks, Washington."
    },
    "Billy_Black": {
        "personality": "Wise, traditional, deeply respects Quileute tribe history and legends. Speaks with authority and warmth.",
        "catchphrases": ["The legends say...", "It's in our blood."],
        "relationships": "Jacob's father; friend of Charlie Swan.",
        "background": "Quileute tribal elder, former fisherman, uses wheelchair due to diabetes."
    },
    "Jacob": {
        "personality": "Passionate, loyal, quick-tempered but gentle; highly protective. Speaks informally and energetically.",
        "catchphrases": ["It's a wolf thing.", "I can't just sit around."],
        "relationships": "Close friend and former crush of Bella; later imprints on Renesmee; son of Billy Black.",
        "background": "From Quileute tribe, can shapeshift into a wolf (therianthrope), becomes pack alpha."
    },
    "Rosalie": {
        "personality": "Proud, beautiful, strong-willed, deeply regrets losing human life. Speaks sharply and elegantly.",
        "catchphrases": ["Don't be a fool.", "I had dreams too."],
        "relationships": "Cullen family member, married to Emmett; adoptive sister to Bella.",
        "background": "Former human turned vampire, possesses extraordinary beauty but mourns inability to bear children."
    },
    "Emmett": {
        "personality": "Outgoing, humorous, extremely strong, the most physically powerful Cullen. Speaks playfully and loudly.",
        "catchphrases": ["Bring it on!", "Nice one, Rose!"],
        "relationships": "Rosalie's husband; Cullen family member (adoptive brother to Edward, Alice, Jasper).",
        "background": "Vampire (Cullen family), fiercely loyal and protective of family."
    },
    "Alice": {
        "personality": "Bubbly, optimistic, friendly, clever, highly protective. Speaks quickly and excitedly.",
        "catchphrases": ["I saw it coming!", "Trust me, I've seen it."],
        "relationships": "Cullen family member, Jasper's mate; close to Bella.",
        "background": "Vampire with precognitive abilities (can see the future)."
    },
    "Jasper": {
        "personality": "Calm, introspective, empathetic, but struggles with blood thirst. Speaks softly and cautiously.",
        "catchphrases": ["I can feel your mood.", "It's... intense."],
        "relationships": "Alice's mate; Cullen family member (adoptive brother).",
        "background": "Vampire (Cullen family) with empathic abilities (can sense and manipulate emotions)."
    },
    "Dr_Cullen": {
        "personality": "Compassionate, wise, empathetic, moral pillar of the family. Speaks gently and authoritatively.",
        "catchphrases": ["We have a choice.", "There is always another way."],
        "relationships": "Leader of Cullen family; husband to Esme; adoptive father to Edward, Alice, Jasper, Rosalie, Emmett.",
        "background": "Born in 17th century, vampire, former doctor. Advocates 'vegetarian' diet (animal blood only)."
    },
    "Esme": {
        "personality": "Gentle, loving, highly maternal, deeply cares for family. Speaks warmly and soothingly.",
        "catchphrases": ["You're always welcome here.", "My dear..."],
        "relationships": "Carlisle's wife; maternal figure to all Cullen family members including Bella.",
        "background": "Originally Esme Anne Platt/Evenson, turned vampire. No special powers but deeply empathetic."
    },
    "James": {
        "personality": "Cunning, dangerous, predatory, highly aggressive. Speaks with mocking confidence.",
        "catchphrases": ["The chase is on.", "You can't hide from me."],
        "relationships": "Antagonist vampire (nomad), initially hunts Bella.",
        "background": "Nomadic vampire, expert tracker and hunter of humans (especially Bella)."
    },
    "Victoria": {
        "personality": "Cold, fiercely vengeful, ambitious. Speaks with sharp intensity.",
        "catchphrases": ["This isn't over.", "He will be avenged."],
        "relationships": "James's companion (later seeks revenge for his death); antagonist to Bella.",
        "background": "Nomadic vampire, threatens Cullen family to avenge James's death."
    },
    "Laurent": {
        "personality": "More peaceful (compared to James and Victoria) but still dangerous. Speaks with smooth deception.",
        "catchphrases": ["I mean no harm.", "Circumstances change."],
        "relationships": "Formerly with James and Victoria, but not always aligned; nomadic vampire.",
        "background": "Nomadic vampire, observes Cullen family and Bella in the story."
    },
    "Aro": {
        "personality": "Ancient, charismatic, calculating, obsessed with power and unique abilities. Speaks in a theatrical, overly polite manner, often hiding threats beneath courtesy.",
        "catchphrases": ["How extraordinary.", "Let us not be hasty."],
        "relationships": "One of the three Volturi leaders; 'brother' to Marcus and Caius; seeks to control the Cullen family, especially Bella, Edward, and Renesmee because of their gifts.",
        "background": "Ancient vampire and de facto ruler of the Volturi in Volterra. Possesses tactile telepathy, allowing him to read every thought a person has ever had through physical touch."
    },
    "Jane": {
        "personality": "Childlike in appearance but cruel and sadistic, fiercely loyal to Aro. Speaks softly and calmly, often enjoying others' fear and pain.",
        "catchphrases": ["This won't take long.", "Pain is very convincing."],
        "relationships": "Member of the Volturi guard; twin sister of Alec; one of Aro's most trusted enforcers; enemy of the Cullens and their allies.",
        "background": "Powerful Volturi guard turned into a vampire at a young age along with her twin brother Alec. Has the ability to project incapacitating illusions of pain onto others."
    }
}


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
        
        # ================== 【新增开始】日志初始化 ==================
        import torch.distributed as dist
        
        # 定义日志目录
        self.log_dir = "/data/wm/Video-R1/src/r1-v/src/open_r1/debug_log"
        os.makedirs(self.log_dir, exist_ok=True)

        # 获取 Rank，区分不同 GPU 的日志文件
        if dist.is_initialized():
            self.rank = dist.get_rank()
        else:
            self.rank = 0
            
        # 定义文件名，例如: debug_log_rank_CLIP_0.txt
        self.log_path = f"{self.log_dir}/debug_log_rank_CLIP_{self.rank}.txt"
        
        # 读取 DEBUG 模式开关
        self.debug_mode = os.getenv("DEBUG_MODE", "").lower().strip() in ["true", "1", "yes"]
        # ================== 【新增结束】 ==================


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



    # def _update_stats(self, tensor_val):
    #     """在 CPU 上维护 EMA 统计量"""
    #     if tensor_val.numel() == 0: return
    #     tensor_val = tensor_val.cpu()
        
    #     batch_mean = tensor_val.mean()
    #     batch_var = tensor_val.var(unbiased=False)
        
    #     if self.clip_mean is None:
    #         self.clip_mean = batch_mean
    #         self.clip_var = batch_var
    #     else:
    #         self.clip_mean = self.momentum * self.clip_mean + (1 - self.momentum) * batch_mean
    #         self.clip_var = self.momentum * self.clip_var + (1 - self.momentum) * batch_var

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

    # def _calc_len_penalty(self, length):
    #     """【逻辑保留在 Client】计算长度惩罚"""
    #     if length <= LEN_SAFE_LIMIT: return 0.0
    #     if length <= LEN_HARD_LIMIT:
    #         return (length - LEN_SAFE_LIMIT) * 0.01
    #     base = (LEN_HARD_LIMIT - LEN_SAFE_LIMIT) * 0.01
    #     return min(2.0, base + (length - LEN_HARD_LIMIT) * 0.03)

    def __call__(self, completions, **kwargs):
        final_scores = [0.0] * len(completions)
        
        embed_paths = kwargs.get('clip_embed_path', [])
        problem_ids = kwargs.get('id', kwargs.get('problem_id', []))
        
        if not embed_paths:
            self._log_batch(completions, final_scores, problem_ids)
            return final_scores

        texts, lengths, mask = self._parse_vision(completions)
        
        valid_indices = [i for i, m in enumerate(mask) if m and embed_paths[i]]
        valid_texts = [texts[i] for i in valid_indices]
        valid_paths = [embed_paths[i] for i in valid_indices]
        
        # final_scores = [0.0] * len(completions)
        if not valid_indices:
            self._log_batch(completions, final_scores, problem_ids) 
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
                self._log_batch(completions, final_scores, problem_ids)
                return final_scores
        except Exception as e:
            print(f"[CLIP Client] Connection Failed: {e}")
            self._log_batch(completions, final_scores, problem_ids)
            return final_scores

        # === Client 端后处理 (EMA + Norm + Penalty) ===
        if raw_scores:
            # raw_tensor = torch.tensor(raw_scores)
            # self._update_stats(raw_tensor) # 更新 EMA
            
            # mu = self.clip_mean if self.clip_mean is not None else 0.0
            # std = (self.clip_var.sqrt() + 1e-6) if self.clip_var is not None else 1.0
            
            for k, original_idx in enumerate(valid_indices):
                # raw = raw_scores[k]
                
                # # 1. Z-Score 归一化
                # norm = (raw - mu) / std
                
                # # 2. 长度惩罚
                # lp = self._calc_len_penalty(lengths[original_idx])
                
                # # 3. 截断与加权
                # base_score = max(-3.0, min(3.0, norm - lp))
                # final_scores[original_idx] = CLIP_REWARD_WEIGHT * base_score
                final_scores[original_idx] = float(raw_scores[k])
        self._log_batch(completions, final_scores, problem_ids)
        
        return final_scores

# 注册 Wrapper
_GLOBAL_CLIP_CLIENT = None
def clip_reward_wrapper(prompts, completions, **kwargs):
    global _GLOBAL_CLIP_CLIENT
    if _GLOBAL_CLIP_CLIENT is None:
        _GLOBAL_CLIP_CLIENT = CLIPRewardClient()
        
        
    # import torch.distributed as dist
    # rank = dist.get_rank() if dist.is_initialized() else 0
    # if int(rank) == 0:
    #     print(f"[rank0] RemotePdb waiting at 127.0.0.1:4444 ...", flush=True)
    #     from remote_pdb import RemotePdb
    #     RemotePdb('127.0.0.1', 4444).set_trace()
        
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

    # -----------------------------
    # (1) Rank & 日志文件路径
    # -----------------------------
    log_dir = "/data/wm/Video-R1/src/r1-v/src/open_r1/debug_log"
    os.makedirs(log_dir, exist_ok=True)

    # 单卡 / 多卡自动区分
    if dist.is_initialized():
        log_path = f"{log_dir}/debug_log_rank_NEW_BERTScore{rank}.txt"   # 多卡时每个 rank 单独一个文件，避免冲突
    else:
        log_path = f"{log_dir}/debug_log_single_gpu.txt"

    # 只读环境变量，不覆盖路径
    DEBUG = os.getenv("DEBUG_MODE", "").lower().strip() in ["true", "1", "yes"]

    # 写日志函数（带异常打印）
    def write_debug(msg):
        if not DEBUG:
            return
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception as e:
            print(f"[Rank {rank}] LOGGING ERROR:", e)
            print(f"[Rank {rank}] Tried writing to:", log_path)
            
    # 尝试从 kwargs 中获取 id 或 problem_id
    problem_ids = kwargs.get('id', kwargs.get('problem_id', []))
    # 如果没获取到，填充 N/A
    if not problem_ids:
        problem_ids = ["N/A"] * len(completions)

    # -----------------------------
    # 打印 enter 信息
    # -----------------------------
    write_debug("====================================================")
    write_debug(f"[Rank {rank}] ENTER accuracy_reward()")
    write_debug(f"Time = {datetime.now()}")
    write_debug(f"problem_type = {kwargs.get('problem_type')}")
    write_debug(f"Problem IDs batch = {problem_ids}") 

    # def extract_answer(text):
    #     pattern = r'<answer>\s*(.*?)\s*</answer>'
    #     match = re.search(pattern, text, re.DOTALL)
    #     if match:
    #         return match.group(1).strip()
    #     return ""
    
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
            # output_ans = extract_answer(content)
            # gt_ans = extract_answer(sol)
            
            # if (output_ans and output_ans.strip()) and (gt_ans and gt_ans.strip()): 
            #     outputs.append(output_ans)
            #     gts.append(gt_ans)
            #     valid_indices.append(idx)
            
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
            # else:
            #     final_scores[idx] = 0.0            
            
        
        # 计算
        # if outputs:
        #     # 定义临时文件路径
        #     # 使用 /tmp 或者当前目录下的 tmp 文件夹，确保有写权限
        #     # 加上 rank 和 uuid 防止冲突
        #     tmp_id = uuid.uuid4().hex
        #     tmp_dir = "/data/wm/Video-R1/src/r1-v/src/open_r1/tmp_bertscore" # 建议用项目内的 tmp 目录
        #     os.makedirs(tmp_dir, exist_ok=True)
            
        #     in_path = os.path.join(tmp_dir, f"in_rank{rank}_{tmp_id}.json")
        #     out_path = os.path.join(tmp_dir, f"out_rank{rank}_{tmp_id}.json")
            
        #     # Worker 脚本路径 (请确保路径正确!)
        #     worker_script = "/data/wm/Video-R1/src/r1-v/src/open_r1/bertscore_worker.py"
            
        #     try:
        #         # 1. 写入输入
        #         with open(in_path, "w", encoding="utf-8") as f:
        #             json.dump({"cands": outputs, "refs": gts}, f)
                
        #         # 2. 调用子进程
        #         # 关键点：这个 subprocess 是纯净的，不受当前 DeepSpeed 环境影响
        #         cmd = ["python", worker_script, in_path, out_path]
        #         subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                
        #         # 3. 读取输出
        #         if os.path.exists(out_path):
        #             with open(out_path, "r", encoding="utf-8") as f:
        #                 result_data = json.load(f)
        #             scores = result_data.get("scores", [])
                    
        #             # 4. 填回分数
        #             if len(scores) == len(outputs):
        #                 for list_idx, score in enumerate(scores):
        #                     original_idx = valid_indices[list_idx]
        #                     final_scores[original_idx] = score
        #             else:
        #                 print(f"[Rank {rank}] Error: BERTScore worker returned {len(scores)} scores, expected {len(outputs)}")
        #         else:
        #             print(f"[Rank {rank}] Error: BERTScore worker did not produce output file.")
                    
        #     except subprocess.CalledProcessError as e:
        #         print(f"[Rank {rank}] BERTScore Worker Failed: {e}")
        #         if e.stderr:
        #             print(f"[Rank {rank}] Worker Stderr: {e.stderr.decode()}")
        #     except Exception as e:
        #         print(f"[Rank {rank}] BERTScore Wrapper Error: {e}")
        #     finally:
        #         # 清理临时文件
        #         if os.path.exists(in_path): os.remove(in_path)
        #         if os.path.exists(out_path): os.remove(out_path)

        # return final_scores
        
        ###################### 上述计算bertscore的代码块可以运行的通，但是运行后容易卡死 ################################
        
        if outputs:
            # 定义 Server 地址
            # server_url = "http://127.0.0.1:5000/score"
            
            try:
                # 发送 HTTP 请求
                # timeout 设置短一点，防止卡死
                # response = requests.post(
                #     server_url, 
                #     json={"cands": outputs, "refs": gts},
                #     timeout=60 
                # )
                
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

        # return final_scores
                    
    # for idx, (content, sol) in enumerate(zip(contents, solution)):   # ← 加入 idx
    
    #     try:
    #         output_ans = extract_answer(content)
    #         gt_ans = extract_answer(sol)
    #         if question_type == "multiple choice":
    #             reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
    #         elif question_type == "numerical":
    #             gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
    #             out_has_decimal = ("." in output_ans) or ("," in output_ans)
    #             if gt_has_decimal != out_has_decimal:
    #                 reward = 0.0
    #             else:
    #                 gt_number = normalize_number(gt_ans)
    #                 out_number = normalize_number(output_ans)
    #                 if gt_number is None or out_number is None:
    #                     reward = 0.0
    #                 else:
    #                     reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
    #         elif question_type == "OCR":
    #             error_rate = wer(gt_ans, output_ans)
    #             reward = 1 - error_rate
    #             reward = max(0.0, min(1.0, reward))
    #         # elif question_type == "free-form":
    #         #     score = compute_rouge_score(gt_ans, output_ans)
    #         #     reward = max(0.0, min(1.0, score))
    #         elif question_type == "regression":
    #             gt_number = normalize_number(gt_ans)
    #             out_number = normalize_number(output_ans)
    #             if gt_number is None or out_number is None:
    #                 reward = 0.0
    #             rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
    #             rel_diff = min(1.0, max(0.0, rel_diff))
    #             reward = 1 - rel_diff
    #         else:
    #             reward = 0.0
    #     except Exception as e:
    #         print(f"Error in reward_fn for question_type '{question_type}': {e}")
    #         reward = 0.0
    
    #     rewards.append(reward)
        
    #     # 写详细日志
    #     # -----------------------------
    #     write_debug(f"[{now}] sample_index={idx} rank={rank}")
    #     write_debug(f"GT     = {gt_ans}")   
    #     write_debug(f"PRED   = {output_ans}")   

    #     write_debug(f"REWARD = {final_scores}")
    #     write_debug("----------------------------------------------------")
        
    # return rewards
    
    
    
    # 无论是否 free-form，无论是否计算成功，都遍历一遍记录状态
    for idx, (content, sol) in enumerate(zip(contents, solution)):
        
        # 获取当前样本对应的 problem_id
        # 因为 completions 是 list，kwargs['id'] 也是 list，它们的索引是一一对应的
        pid = problem_ids[idx] if idx < len(problem_ids) else "Unknown"

        output_ans = extract_answer(content)
        gt_ans = extract_answer(sol)
        score = final_scores[idx]

        write_debug(f"[{now}] Batch_Idx={idx} | Rank={rank}")
        write_debug(f"PROBLEM_ID = {pid}")   # <--- 重点：记录 Problem ID
        write_debug(f"GT         = {gt_ans}")   
        write_debug(f"PRED       = {output_ans}")   
        write_debug(f"REWARD     = {score:.6f}") # 记录当前样本的具体分数
        write_debug("----------------------------------------------------")
        
    return final_scores


# def format_reward(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""
#     # pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
#     pattern = r"\s*<vision>.*?</vision>\s*<think>.*?</think>\s*<answer>.*?</answer>\s*"
#     completion_contents = [completion[0]["content"] for completion in completions]
#     matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
#     return [1.0 if match else 0.0 for match in matches]


def format_reward(completions, **kwargs):
    """
    格式检查：必须严格匹配 STRICT_PATTERN。
    返回 1.0 (合格) 或 0.0 (不合格)。
    """
    pattern = re.compile(STRICT_PATTERN, re.DOTALL)
    completion_contents = [c[0]["content"] if isinstance(c, list) else c for c in completions]
    matches = [pattern.fullmatch(content) for content in completion_contents]
    return [1.0 if m else 0.0 for m in matches]



reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "clip": clip_reward_wrapper,
}

# SYSTEM_PROMPT = (
#     "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
#     "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
#     "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
#     "<think> reasoning process here </think><answer> answer here </answer>"
# )

# role-playing 人物场景的SYSTEM_PROMPT设计：
# SYSTEM_PROMPT = (
#     "You are a movie role-play dialogue generation assistant.\n"
#     "Given character profiles and a partial dialogue between two characters from a movie, "
#     "you must generate the next line spoken by the assistant character.\n"
#     "Always stay in character, respect the personalities, relationships, and background of the characters, "
#     "and make the reply coherent, natural, and emotionally appropriate.\n"
#     "You must first think about the reasoning process inside <think> </think>, and then provide ONLY the final "
#     "utterance inside <answer> </answer>.\n"
#     "Do not include the speaker name or quotation marks inside <answer>."
# )

SYSTEM_PROMPT = (
    "You are a movie role-play dialogue generation assistant.\n"
    "Given character profiles and a partial dialogue between two characters from a movie, "
    "you must generate the next line spoken by the assistant character.\n"
    "Always stay in character, respect the personalities, relationships, and background of the characters, "
    "and make the reply coherent, natural, and emotionally appropriate.\n"
    "\n"
    "If there is visual input (image or video), you must first describe what you see.\n"
    "Your output must strictly follow this structure:\n"
    "<vision> Describe the character's facial expression, body posture/gestures, and the surrounding environment "
    "seen in the image or video. Focus on concrete visual details rather than abstract interpretation. </vision>\n"
    "<think> Based on the visual cues and the dialogue history, analyze the character's internal emotional state, "
    "motivation, and what they are likely to say next. Reason step by step and verify that your planned reply "
    "matches the character's personality and relationships. </think>\n"
    "<answer> Your final response as the character, written as a single natural line of dialogue, without "
    "including the speaker name or quotation marks. </answer>\n"
    "\n"
    "Do not include the speaker name or quotation marks inside <answer> </answer>."
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        dataset =  DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # Load the dataset
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    
    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
    }

    def make_conversation_image(example):
        
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
        }
    
        
    def make_conversation_video(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["problem"])},
                    ],
                },
            ],
    }
        
    def make_conversation_image_and_video(example):
        if example["problem_type"] == 'multiple choice':
            question = example['problem'] + "Options:\n"
            for op in example["options"]:
                question += op + "\n"
        else:
            question = example['problem']

        
        msg ={
            "prompt": 
               [{
                    "role": "user",
                    "content": [
                        {
                            "type": example['data_type'],
                            # example['data_type']: os.getcwd() + "/Video-R1-data" + example['path'][1:]
                        },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE[example['problem_type']]
                        }
                        ]
                }]
            }
        
        return msg
    
    
    
    def _format_character_block(name: str) -> str:
        profile = CHARACTER_PROFILES.get(name, None)
        if profile is None:
            return f"{name}: (no profile available)\n"

        catchphrases = ", ".join(profile.get("catchphrases", []))
        text = (
            f"Name: {name}\n"
            f"Personality: {profile.get('personality', '')}\n"
            f"Speaking style and catchphrases: {catchphrases}\n"
            f"Relationships: {profile.get('relationships', '')}\n"
            f"Background: {profile.get('background', '')}\n"
        )
        return text

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
        """
        根据单条样本构造文本 prompt：
        - 引入 user / assistant 的人物档案
        - 列出已有 dialogue
        - 明确任务：预测 assistant 的下一句台词
        """
        user_name = example["user"]
        assistant_name = example["assistant"]

        user_block = _format_character_block(user_name)
        assistant_block = _format_character_block(assistant_name)
        dialogue_block = _format_dialogue_block(example["dialogue"])

        # text = (
        #     "You are continuing a movie dialogue between two characters from 'The Twilight Saga'.\n\n"
        #     "[Character profiles]\n"
        #     f"USER ({user_name}):\n{user_block}\n"
        #     f"ASSISTANT ({assistant_name}):\n{assistant_block}\n"
        #     "[Dialogue so far]\n"
        #     f"{dialogue_block}\n\n"
        #     "Task:\n"
        #     f"Write the next line of dialogue that {assistant_name} would say in this scene.\n"
        #     f"The line must:\n"
        #     f"- Match {assistant_name}'s personality and speaking style.\n"
        #     f"- Be coherent with the previous lines and smoothly continue the conversation.\n"
        #     f"- Sound natural and fluent.\n\n"
        #     "Output format:\n"
        #     "- First, think step by step about what the assistant character would say. Put this reasoning inside <think> </think>.\n"
        #     "- Then, output ONLY the final utterance text (without speaker name or quotation marks) inside <answer> </answer>.\n"
        # )
        text = (
            "You are continuing a movie dialogue between two characters from 'The Twilight Saga'.\n\n"
            "[Character profiles]\n"
            f"USER ({user_name}):\n{user_block}\n"
            f"ASSISTANT ({assistant_name}):\n{assistant_block}\n"
            "[Dialogue so far]\n"
            f"{dialogue_block}\n\n"
            "Task:\n"
            f"Write the next line of dialogue that {assistant_name} would say in this scene.\n"
            "The line must:\n"
            f"- Match {assistant_name}'s personality and speaking style.\n"
            f"- Be coherent with the previous lines and smoothly continue the conversation.\n"
            f"- Sound natural and fluent.\n\n"
            "Output format:\n"
            "You MUST structure your output using the following tags:\n"
            "<vision> Describe the character's facial expression, gestures, and the environment "
            "seen in the image or video. Mention only what is visually observable. </vision>\n"
            "<think> Based on the visual cues and the dialogue history, analyze the character's "
            "internal emotional state and decide what they are most likely to say next. "
            "Think step by step, and check that your plan is consistent with the character profiles. </think>\n"
            "<answer> Your final response as the character: a single line of dialogue that they would speak "
            "in this moment, without the speaker name or quotation marks. </answer>\n"
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
