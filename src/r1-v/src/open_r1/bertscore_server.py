# src/open_r1/bertscore_server.py
import torch
from fastapi import FastAPI, Request
from bert_score import BERTScorer
import uvicorn
import os
import sys
from transformers import AutoConfig

app = FastAPI()

scorer = None

@app.on_event("startup")
async def startup_event():
    global scorer
    print("Initialize BERTScore Server...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...", flush=True)

    try:
        scorer = BERTScorer(
            model_type="microsoft/deberta-xlarge-mnli",
            device=device,
            idf=False,
            rescale_with_baseline=False,
            lang="en"
        )

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
        
        P, R, F1 = scorer.score(cands, refs)
        
        scores = [max(0.0, min(1.0, float(s.item()))) for s in F1]
        return {"scores": scores}
        
    except Exception as e:
        print(f"Error processing request: {e}")
        return {"scores": [0.0] * len(cands)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
