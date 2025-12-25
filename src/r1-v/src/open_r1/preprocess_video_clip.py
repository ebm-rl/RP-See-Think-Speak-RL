import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch.multiprocessing as mp

# ================= 配置 =================
MODEL_ID = "openai/clip-vit-large-patch14"
INPUT_JSON = "/data/wm/simple-subtitling/Processed_Dialogue/The_Twilight_Saga/final_dialog_samples_data.json"
OUTPUT_JSON = "/data/wm/simple-subtitling/Processed_Dialogue/The_Twilight_Saga/final_dialog_samples_data_pre_video_clip.json"
EMBED_DIR = "/data/wm/Video-R1/src/r1-v/src/open_r1/clip_embeddings"
# ========================================

def get_dynamic_indices(total_frames, fps):
    if fps <= 0: fps = 25.0
    duration = total_frames / fps
    if duration < 5: n = 4
    elif duration < 15: n = 8
    elif duration < 60: n = 16
    else: n = 32
    if total_frames <= n: return list(range(total_frames))
    return np.linspace(0, total_frames-1, n, dtype=int).tolist()

def process_video(video_path, model, processor, device):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        indices = get_dynamic_indices(total, fps)
        
        images = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        
        if not images: return None

        with torch.no_grad():
            # 注意：这里要用传入的 device
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            
        return feats.cpu()
    except Exception as e:
        # print(f"Error: {e}") 
        return None

def worker(gpu_id, sub_data, result_queue):
    """
    每个 GPU 进程运行的函数
    """
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading model...")
    # 每个进程独立加载模型到对应的卡
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    
    processed_items = []
    
    # 只需要显示 GPU 0 的进度条，不然屏幕会花
    iterator = tqdm(sub_data, desc=f"GPU {gpu_id}", position=gpu_id) if gpu_id == 0 else sub_data
    
    for item in iterator:
        v_path = item.get('video_path')
        if not v_path: continue
        
        file_name_no_ext = os.path.basename(v_path).rsplit('.', 1)[0]
        save_name = f"{file_name_no_ext}.pt"
        save_path = os.path.join(EMBED_DIR, save_name)
        
        # 即使已经存在，也要更新 item 里的路径，方便最后汇总
        if os.path.exists(save_path):
            item['clip_embed_path'] = save_path
            processed_items.append(item)
            continue
            
        emb = process_video(v_path, model, processor, device)
        if emb is not None:
            torch.save(emb, save_path)
            item['clip_embed_path'] = save_path
            processed_items.append(item)
            
    # 把处理好的数据传回主进程
    result_queue.put(processed_items)
    print(f"[GPU {gpu_id}] Done.")

def main():
    # 必须设置启动方式为 spawn，否则 CUDA 初始化会报错
    mp.set_start_method('spawn', force=True)
    
    os.makedirs(EMBED_DIR, exist_ok=True)
    
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)
    
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs. Splitting workload...")
    
    # 将数据切分成 N 份
    chunk_size = len(data) // num_gpus + 1
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    
    # 确保 chunks 长度等于 gpu 数量 (防止最后一部分数据越界或不够分)
    if len(chunks) > num_gpus:
        chunks[num_gpus-1].extend(sum(chunks[num_gpus:], []))
        chunks = chunks[:num_gpus]
    
    queue = mp.Queue()
    processes = []
    
    # 启动多进程
    for rank in range(num_gpus):
        p = mp.Process(target=worker, args=(rank, chunks[rank], queue))
        p.start()
        processes.append(p)
        
    # 收集结果
    final_data = []
    for _ in range(num_gpus):
        final_data.extend(queue.get())
        
    for p in processes:
        p.join()
        
    # 保存合并后的 JSON
    print(f"All done! Saving {len(final_data)} items to {OUTPUT_JSON}")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_data, f, indent=2)

if __name__ == "__main__":
    main()
