# src/open_r1/bertscore_worker.py
import json
import sys
import os
import torch

# 关键：这里完全是一个干净的 Python 环境，没有 DeepSpeed，没有 ZeRO-3
# 所以 BERTScorer 会正常工作，不会被切片。

def main(in_path, out_path):
    # 1. 读取输入
    try:
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Worker Error] Failed to read input: {e}", file=sys.stderr)
        return

    cands = data.get("cands", [])
    refs = data.get("refs", [])
    
    if not cands:
        # 空数据直接写回空结果
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"scores": []}, f)
        return

    # 2. 加载模型
    # 建议使用 CPU 以避免与训练进程抢显存，且绝对稳定
    # batch size 很小 (8)，CPU 速度足够快 (0.x秒)
    device = "cpu"
    
    # 如果你确定显存极其充裕，也可以尝试用 "cuda"
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # print(f"[Worker] Loading BERTScore on {device}...", file=sys.stderr)

    try:
        # 在这里 import 避免影响主进程（其实是独立的，无所谓）
        from bert_score import BERTScorer
        
        # 加载模型
        scorer = BERTScorer(
            model_type="microsoft/deberta-xlarge-mnli",
            device=device,
            # 显式关闭 idf 和 rescale，防止哈希报错 (int too big)
            idf=False,
            rescale_with_baseline=False,
            lang="en" # 显式指定语言通常是个好习惯，或者用 model_type
        )
        
        # 3. 计算
        # P, R, F1
        P, R, F1 = scorer.score(cands, refs)
        
        # 转换为 float list
        scores = [max(0.0, min(1.0, float(s.item()))) for s in F1]
        
    except Exception as e:
        print(f"[Worker Error] Calculation failed: {e}", file=sys.stderr)
        scores = [0.0] * len(cands)

    # 4. 写回输出
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"scores": scores}, f)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python bertscore_worker.py <in_json> <out_json>", file=sys.stderr)
        sys.exit(1)
    
    main(sys.argv[1], sys.argv[2])
