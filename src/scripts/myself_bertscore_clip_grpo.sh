#!/bin/bash
cd src/r1-v

# ================= 资源分配策略 =================
# 策略：GPU 0 跑 Server，GPU 1-7 跑训练 (共7卡)
# 这样能最大程度避免显存碎片化和 DeepSpeed 冲突
export SERVER_GPU="0"
export TRAINING_GPUS="1,2,3,4,5,6,7"
export NUM_GPUS=7  # 必须与 TRAINING_GPUS 数量一致
# ================================================

# 【新增】设置 NCCL 超时时间为 3600 秒 (1小时)
# 默认是 1800 秒 (30分钟)，这能防止长样本 Forward 时被误杀
export NCCL_TIMEOUT="3600"  
# ================================================

# 清理函数：脚本退出时杀掉后台进程
cleanup() {
    echo "[Script] Cleaning up background servers..."
    if [ ! -z "$BERT_PID" ]; then kill $BERT_PID 2>/dev/null; fi
    if [ ! -z "$CLIP_PID" ]; then kill $CLIP_PID 2>/dev/null; fi
}
trap cleanup EXIT

echo "=================================================="
echo "[Script] Step 1: Starting Reward Servers on GPU $SERVER_GPU..."
echo "=================================================="

# 1. 启动 BERTScore Server (Port 5000)
CUDA_VISIBLE_DEVICES=$SERVER_GPU python src/open_r1/bertscore_server.py > bert_server.log 2>&1 &
BERT_PID=$!
echo "[Script] BERTScore Server PID: $BERT_PID"

# 2. 启动 CLIP Server (Port 5001)
CUDA_VISIBLE_DEVICES=$SERVER_GPU python src/open_r1/clip_server.py > clip_server.log 2>&1 &
CLIP_PID=$!
echo "[Script] CLIP Server PID: $CLIP_PID"

echo "Waiting 30 seconds for models to load..."
sleep 30

# 简单检查一下端口是否通了
if ! nc -z 127.0.0.1 5000; then echo "WARNING: BERTScore server port 5000 not open yet!"; fi
if ! nc -z 127.0.0.1 5001; then echo "WARNING: CLIP server port 5001 not open yet!"; fi


echo "=================================================="
echo "[Script] Step 2: Starting GRPO Training on GPUs $TRAINING_GPUS..."
echo "=================================================="

# 注意：
# --nproc_per_node=$NUM_GPUS  (这里是 7)
# CUDA_VISIBLE_DEVICES=$TRAINING_GPUS (这里是 1,2,3,4,5,6,7)
# 这样对于 PyTorch 来说，它只看得到 7 张卡，逻辑编号是 0-6，不会意识到物理 GPU 0 的存在

export DEBUG_MODE="true"

CUDA_VISIBLE_DEVICES=$TRAINING_GPUS torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12365" \
    src/open_r1/myself_grpo.py \
    --output_dir "./log/Qwen2.5-VL-7B-GRPO-Role-Playing" \
    --model_name_or_path '/data/wm/Video-R1/Qwen2.5-VL-7B-Instruct' \
    --dataset_name "/data/wm/simple-subtitling/Processed_Dialogue/The_Twilight_Saga/final_dialog_samples_data_pre_video_clip.json" \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length 768 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 True \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --temporal False \
    --len_control False \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --max_steps 300 \
    --run_name Myself-GRPO-Role-Playing \
    --save_steps 100 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 8

echo "=================================================="
echo "[Script] Training finished."
