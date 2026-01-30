import os
import json
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import torch.multiprocessing as mp

# ================= Config =================
MODEL_ID = "openai/clip-vit-large-patch14"
INPUT_JSON = "../simple-subtitling/Processed_Dialogue/RP-RL-Dataset/train.json"
OUTPUT_JSON = "../simple-subtitling/Processed_Dialogue/RP-RL-Dataset/train_pre_video_clip.json"
EMBED_DIR = "./src/r1-v/src/open_r1/clip_embeddings/final_train"


VIDEO_ROOT = "../simple-subtitling/Processed_Dialogue/RP-EBM-Dataset"
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
            inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            
        return feats.cpu()
    except Exception as e:
        # print(f"Error: {e}") 
        return None

def worker(gpu_id, sub_data, result_queue):
    device = f"cuda:{gpu_id}"
    print(f"[GPU {gpu_id}] Loading model...")
    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    
    processed_items = []
    
    iterator = tqdm(sub_data, desc=f"GPU {gpu_id}", position=gpu_id) if gpu_id == 0 else sub_data
    
    for item in iterator:
        # v_path = item.get('video_path')
        # if not v_path: continue
        
        rel_path = item.get('video_path')
        if not rel_path: continue
        
        clean_rel_path = rel_path.lstrip('./').lstrip('/') 
        full_video_path = os.path.join(VIDEO_ROOT, clean_rel_path)
        
        if not os.path.exists(full_video_path):
            # print(f"[Skip] File not found: {full_video_path}")
            continue
        
        unique_name = clean_rel_path.replace('/', '_').replace('\\', '_').rsplit('.', 1)[0]
        save_name = f"{unique_name}.pt"
        save_path = os.path.join(EMBED_DIR, save_name)
        
        if os.path.exists(save_path):
            item['clip_embed_path'] = save_path
            processed_items.append(item)
            continue
        
        emb = process_video(full_video_path, model, processor, device)
        if emb is not None:
            torch.save(emb, save_path)
            item['clip_embed_path'] = save_path
            processed_items.append(item)
            
    result_queue.put(processed_items)
    print(f"[GPU {gpu_id}] Done.")

def main():
    mp.set_start_method('spawn', force=True)
    
    os.makedirs(EMBED_DIR, exist_ok=True)
    
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)
    
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs. Splitting workload...")
    
    chunk_size = len(data) // num_gpus + 1
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    if len(chunks) > num_gpus:
        chunks[num_gpus-1].extend(sum(chunks[num_gpus:], []))
        chunks = chunks[:num_gpus]
    
    queue = mp.Queue()
    processes = []

    for rank in range(num_gpus):
        p = mp.Process(target=worker, args=(rank, chunks[rank], queue))
        p.start()
        processes.append(p)
        
    final_data = []
    for _ in range(num_gpus):
        final_data.extend(queue.get())
        
    for p in processes:
        p.join()
        
    print(f"All done! Saving {len(final_data)} items to {OUTPUT_JSON}")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_data, f, indent=2)

if __name__ == "__main__":
    main()
