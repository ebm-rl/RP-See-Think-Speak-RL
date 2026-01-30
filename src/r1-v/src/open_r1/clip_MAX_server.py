import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict

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
        inputs = PROCESSOR(
            text=req.texts, return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(DEVICE)
        
        scores_map = {}
        
        with torch.no_grad():
            all_t_embeds = MODEL.get_text_features(**inputs)
            all_t_embeds = all_t_embeds / all_t_embeds.norm(p=2, dim=-1, keepdim=True)
            
            path_groups = defaultdict(list)
            for batch_idx, path in enumerate(req.embed_paths):
                path_groups[path].append(batch_idx)
            
            for path, batch_indices in path_groups.items():
                if not os.path.exists(path):
                    for b_idx in batch_indices:
                        original_idx = req.indices[b_idx]
                        scores_map[original_idx] = -1.0
                    continue
                
                v_embeds = torch.load(path, map_location=DEVICE).float()
                if v_embeds.dim() == 1: 
                    v_embeds = v_embeds.unsqueeze(0)
                
                v_embeds = v_embeds / v_embeds.norm(p=2, dim=-1, keepdim=True)
                group_t_embeds = all_t_embeds[batch_indices]
                sim_matrix = torch.matmul(v_embeds, group_t_embeds.T)
                best_scores = sim_matrix.max(dim=0).values
                

                for i, score in enumerate(best_scores):
                    b_idx = batch_indices[i]
                    original_idx = req.indices[b_idx] 
                    scores_map[original_idx] = score.item()

        results = []
        for idx in req.indices:
            results.append(scores_map.get(idx, 0.0))
            
        return {"raw_scores": results}

    except Exception as e:
        print(f"[CLIP Server] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)
