import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict
import re

app = FastAPI()

MODEL_ID = "openai/clip-vit-large-patch14"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = None
PROCESSOR = None

class CLIPRequest(BaseModel):
    texts: List[str]    
    embed_paths: List[str] 
    indices: List[int]   

@app.on_event("startup")
def load_model():
    global MODEL, PROCESSOR
    print(f"[CLIP Server] Loading model on {DEVICE}...")
    try:
        MODEL = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
        PROCESSOR = CLIPProcessor.from_pretrained(MODEL_ID)
        print("[CLIP Server] Model loaded successfully!")
    except Exception as e:
        print(f"[CLIP Server] FATAL ERROR: {e}")
        raise e

@app.post("/score")
async def calculate_score(req: CLIPRequest):
    if not req.texts:
        return {"raw_scores": []}
    
    try:
        results_map = {} 
        
        path_groups = defaultdict(list)
        for batch_idx, path in enumerate(req.embed_paths):
            path_groups[path].append(batch_idx)
            
        for path, batch_indices in path_groups.items():
            
            if not os.path.exists(path):
                for b_idx in batch_indices:
                    results_map[req.indices[b_idx]] = -1.0
                continue
            
            v_embeds = torch.load(path, map_location=DEVICE).float()
            if v_embeds.dim() == 1: v_embeds = v_embeds.unsqueeze(0)
            v_embeds = v_embeds / v_embeds.norm(p=2, dim=-1, keepdim=True)
            n_frames = v_embeds.size(0)
            
            for b_idx in batch_indices:
                text_content = req.texts[b_idx]
                original_idx = req.indices[b_idx]
                sentences = re.split(r'[.!?。！？\n]', text_content)

                sentences = [
                    s.strip() for s in sentences 
                    if len(s.strip()) > 5 and not re.match(r'^\d+\.?$', s.strip())
                ]
                sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
                
                if not sentences:
                    sentences = [text_content]
                inputs_sent = PROCESSOR(
                    text=sentences, return_tensors="pt", padding=True, truncation=True, max_length=77
                ).to(DEVICE)
                
                with torch.no_grad():
                    # sent_embeds: [Num_Sentences, Dim]
                    sent_embeds = MODEL.get_text_features(**inputs_sent)
                    sent_embeds = sent_embeds / sent_embeds.norm(p=2, dim=-1, keepdim=True)
                    sim_matrix = torch.matmul(v_embeds, sent_embeds.T)
                    
                    if n_frames == 1:
                        sentence_scores = sim_matrix.squeeze(0)
                    else:
                        top_k_ratio = 0.2
                        k = max(1, int(n_frames * top_k_ratio))
                        
                        topk_values, _ = sim_matrix.topk(k=k, dim=0)
                        sentence_scores = topk_values.mean(dim=0)

                    final_score = sentence_scores.mean().item()
                    
                    results_map[original_idx] = final_score

        results = []
        for idx in req.indices:
            results.append(results_map.get(idx, 0.0))
            
        return {"raw_scores": results}

    except Exception as e:
        print(f"[CLIP Server] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)