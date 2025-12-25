import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from transformers import CLIPProcessor, CLIPModel
from collections import defaultdict

app = FastAPI()

# ================= 配置区 =================
MODEL_ID = "openai/clip-vit-large-patch14"
# 自动检测设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================

MODEL = None
PROCESSOR = None

class CLIPRequest(BaseModel):
    texts: List[str]       # 文本内容
    embed_paths: List[str] # 图片 embedding 路径
    indices: List[int]     # 原始索引（用于对应）

@app.on_event("startup")
def load_model():
    global MODEL, PROCESSOR
    print(f"[CLIP Server] Loading model on {DEVICE}...")
    try:
        # 这是一个纯净的 Python 进程，没有任何 DeepSpeed 干扰
        # 直接加载完整模型
        MODEL = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
        PROCESSOR = CLIPProcessor.from_pretrained(MODEL_ID)
        print("[CLIP Server] Model loaded successfully!")
    except Exception as e:
        print(f"[CLIP Server] FATAL ERROR: {e}")
        raise e

# @app.post("/score")
# async def calculate_score(req: CLIPRequest):
#     """
#     只负责计算：Text Embedding 和 Image Embedding 的原始 Cosine Similarity。
#     不包含任何惩罚、归一化或权重逻辑。
#     """
#     if not req.texts:
#         return {"raw_scores": []}
    
#     try:
#         # 1. 处理文本
#         inputs = PROCESSOR(
#             text=req.texts, return_tensors="pt", padding=True, truncation=True, max_length=77
#         ).to(DEVICE)
        
#         scores_map = {}
        
#         with torch.no_grad():
#             # 获取文本特征并归一化
#             t_embeds = MODEL.get_text_features(**inputs)
#             t_embeds = t_embeds / t_embeds.norm(p=2, dim=-1, keepdim=True)
            
#             # 2. 逐个处理对应的图片
#             for i, path in enumerate(req.embed_paths):
#                 if not os.path.exists(path):
#                     # 如果找不到文件，给个极低分
#                     scores_map[req.indices[i]] = -1.0 
#                     continue
                
#                 # 加载图片特征 (确保是 float32)
#                 v_embeds = torch.load(path, map_location=DEVICE).float()
#                 if v_embeds.dim() == 1: v_embeds = v_embeds.unsqueeze(0)
                
#                 # 计算点积 (Cosine Similarity)
#                 sim = torch.matmul(v_embeds, t_embeds[i].unsqueeze(0).T)
#                 raw_score = sim.max().item()
                
#                 scores_map[req.indices[i]] = raw_score

#         # 按请求顺序返回列表
#         results = []
#         for idx in req.indices:
#             results.append(scores_map.get(idx, 0.0))
            
#         return {"raw_scores": results}

#     except Exception as e:
#         print(f"[CLIP Server] Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


@app.post("/score")
async def calculate_score(req: CLIPRequest):
    """
    【优化版】针对 GRPO 场景优化：
    1. 自动将相同 embed_paths 的请求合并。
    2. 同一个视频只加载一次，保证计算结果完全一致且高效。
    """
    if not req.texts:
        return {"raw_scores": []}
    
    try:
        # 1. 预处理文本 (一次性处理所有文本)
        # inputs: {'input_ids': ..., 'attention_mask': ...}
        inputs = PROCESSOR(
            text=req.texts, return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(DEVICE)
        
        scores_map = {}
        
        with torch.no_grad():
            # 2. 获取所有文本的 Feature 并归一化
            # t_embeds shape: [Batch_Size, Embed_Dim]
            all_t_embeds = MODEL.get_text_features(**inputs)
            all_t_embeds = all_t_embeds / all_t_embeds.norm(p=2, dim=-1, keepdim=True)
            
            # 3. 按视频路径分组
            # 结构: { "path/to/video.pt": [batch_idx_0, batch_idx_1, ...] }
            path_groups = defaultdict(list)
            for batch_idx, path in enumerate(req.embed_paths):
                path_groups[path].append(batch_idx)
            
            # 4. 逐组计算 (通常 GRPO 场景下只有 1 组)
            for path, batch_indices in path_groups.items():
                
                # (A) 处理文件不存在的情况
                if not os.path.exists(path):
                    for b_idx in batch_indices:
                        original_idx = req.indices[b_idx]
                        scores_map[original_idx] = -1.0
                    continue
                
                # (B) 加载视频特征 (只加载一次！)
                # v_embeds shape: [N_frames, Embed_Dim] 或 [1, Embed_Dim]
                v_embeds = torch.load(path, map_location=DEVICE).float()
                if v_embeds.dim() == 1: 
                    v_embeds = v_embeds.unsqueeze(0)
                
                # (C) 归一化视频特征 (CLIP 计算 Cosine Sim 必须两边都归一化)
                # 这一步很重要，原代码是在 matmul 后算的，这里提前算更稳
                v_embeds = v_embeds / v_embeds.norm(p=2, dim=-1, keepdim=True)

                # (D) 提取当前组对应的文本特征
                # group_t_embeds shape: [Group_Size, Embed_Dim]
                group_t_embeds = all_t_embeds[batch_indices]
                
                # (E) 批量计算相似度
                # [N_frames, Dim] @ [Group_Size, Dim].T  ->  [N_frames, Group_Size]
                sim_matrix = torch.matmul(v_embeds, group_t_embeds.T)
                
                # (F) 取每个文本对应的最大相似度 (Max over frames)
                # best_scores shape: [Group_Size]
                best_scores = sim_matrix.max(dim=0).values
                
                # (G) 填回结果
                for i, score in enumerate(best_scores):
                    b_idx = batch_indices[i] # 当前 batch 中的索引
                    original_idx = req.indices[b_idx] # 原始请求中的 ID
                    scores_map[original_idx] = score.item()

        # 5. 按原始请求顺序返回
        results = []
        for idx in req.indices:
            results.append(scores_map.get(idx, 0.0))
            
        return {"raw_scores": results}

    except Exception as e:
        print(f"[CLIP Server] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    # 运行在 5001 端口 (避开 BERTScore 的 5000)
    uvicorn.run(app, host="0.0.0.0", port=5001)
