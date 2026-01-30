#!/bin/bash
set -e
cd src/r1-v

cleanup () {
  pkill -9 -f bertscore_server.py || true
  pkill -9 -f clip_MAX_server.py || true
}
trap cleanup EXIT

# ============== Servers on GPU0 ==============
CUDA_VISIBLE_DEVICES=0 python src/open_r1/bertscore_server.py > bert_server.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python src/open_r1/clip_MAX_server.py      > clip_MAX_server.log 2>&1 &
# sleep 30

echo "=================================================="
echo "[Script] Waiting for servers to initialize..."

wait_for_port() {
    local port=$1
    local name=$2
    local timeout=300 

    echo "[Script] Checking $name on port $port..."
    for ((i=1; i<=timeout; i++)); do
        if (echo > /dev/tcp/127.0.0.1/$port) >/dev/null 2>&1; then
            echo -e "\n[Script] ✅ $name (Port $port) is READY after $i seconds."
            return 0
        fi
        echo -ne "Waiting for $name... time elapsed: ${i}s\r"
        sleep 1
    done

    echo -e "\n[Script] ❌ ERROR: $name failed to start on port $port within $timeout seconds."
    if [ "$name" == "BertScore" ]; then tail -n 20 bert_server.log; fi
    if [ "$name" == "ClipServer" ]; then tail -n 20 clip_MAX_server.log; fi
    exit 1
}

wait_for_port 5000 "BertScore"
wait_for_port 5001 "ClipServer"

echo "[Script] All servers are ready. Starting training..."
echo "=================================================="

export NCCL_DEBUG=WARN
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
export OMP_NUM_THREADS=1

export NCCL_IB_DISABLE=1
unset NCCL_P2P_DISABLE
unset NCCL_SOCKET_IFNAME
export TORCH_NCCL_EAGER_CONNECT=0
unset TORCH_NCCL_USE_COMM_NONBLOCKING
export NCCL_NVLS_ENABLE=0
export TORCH_DISTRIBUTED_DEBUG=INFO 
export DEBUG_MODE="true"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


torchrun --standalone \
    --nproc_per_node=7 \
    --master_port="12365" \
    src/open_r1/RP_RL.py \
    --output_dir "./log/Qwen2.5-VL-7B-Role-Playing-GRPO-EBM-w-1-1-0.8-0.8" \
    --model_name_or_path './log/Qwen2.5-VL-7B-Rolae-Playing-COT-SFT-2/checkpoint-360' \
    --base_model_path '../../Qwen2.5-VL-7B-Instruct' \
    --dataset_name "../../../simple-subtitling/Processed_Dialogue/RP-EBM-Dataset/RP-EBM-train.json" \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 16384 \
    --max_completion_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-7 \
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
    --run_name Qwen2.5-VL-Role-Playing-E-B-M-GRPO-w-1-1-0.8-0.8 \
    --save_steps 500 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 8

pkill -9 -f bertscore_server.py
pkill -9 -f clip_MAX_server.py

echo "=================================================="
echo "[Script] Training finished."
