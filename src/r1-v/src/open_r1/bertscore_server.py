# src/open_r1/bertscore_server.py
import torch
from fastapi import FastAPI, Request
from bert_score import BERTScorer
import uvicorn
import os
import sys

app = FastAPI()

# 全局变量
scorer = None

@app.on_event("startup")
async def startup_event():
    global scorer
    print("Initialize BERTScore Server...", flush=True)
    
    # 策略：如果有空闲显卡，就用显卡；否则用 CPU
    # 建议：指定到一张卡上，比如 CUDA_VISIBLE_DEVICES=0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...", flush=True)
    
    try:
        scorer = BERTScorer(
            model_type="microsoft/deberta-xlarge-mnli",
            device=device,
            idf=False, # 关闭 idf 防止 hash 错误
            rescale_with_baseline=False,
            lang="en"
        )
        # 预热一下，防止第一次请求慢
        scorer.score(["hello"], ["hello"])
        print("BERTScore Model Loaded Successfully!", flush=True)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

@app.post("/score")
async def calculate_score(request: Request):
    if scorer is None:
        return {"error": "Model not loaded"}
        
    try:
        data = await request.json()
        cands = data.get("cands", [])
        refs = data.get("refs", [])
        
        if not cands:
            return {"scores": []}
        
        # 计算逻辑
        # 因为 scorer 已经初始化好了，这里仅仅是 forward，速度极快
        P, R, F1 = scorer.score(cands, refs)
        
        # 转 list
        scores = [max(0.0, min(1.0, float(s.item()))) for s in F1]
        return {"scores": scores}
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return {"scores": [0.0] * len(cands)}

if __name__ == "__main__":
    # 监听 127.0.0.1:5000
    uvicorn.run(app, host="127.0.0.1", port=5000)
