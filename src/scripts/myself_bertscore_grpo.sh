#!/bin/bash
cd src/r1-v  # 确保进入正确的目录

# ================= 配置区 =================
# Server 端口
SERVER_PORT=5000
SERVER_HOST="127.0.0.1"
# 指定 Server 运行在哪张卡（建议选一张显存较空的卡，或者如果没有空闲卡就用 CPU）
# 如果想用 CPU，把下面这行改成 export CUDA_VISIBLE_DEVICES=""
export SERVER_GPU="0" 

# 训练用的 GPU (例如用前8张卡)
export TRAINING_GPUS="0,1,2,3,4,5,6,7"
# ==========================================

# 1. 定义清理函数：脚本退出时杀掉后台的 Server
cleanup() {
    echo "[Script] Cleaning up..."
    if [ ! -z "$SERVER_PID" ]; then
        echo "[Script] Killing BERTScore Server (PID: $SERVER_PID)"
        kill $SERVER_PID 2>/dev/null
    fi
}
# 捕获 EXIT 信号（无论正常退出还是 Ctrl+C 中断，都会执行 cleanup）
trap cleanup EXIT

echo "=================================================="
echo "[Script] Step 1: Starting BERTScore Server..."
echo "=================================================="

# 启动 Server (注意最后的 & 符号，让它在后台运行)
# 我们将 Server 的日志重定向到 server.log 以免干扰训练界面
CUDA_VISIBLE_DEVICES=$SERVER_GPU python src/open_r1/bertscore_server.py > server.log 2>&1 &
SERVER_PID=$! # 获取刚才后台进程的 PID

echo "[Script] Server PID is $SERVER_PID. Logs are in server.log."
echo "[Script] Waiting for Server to be ready on port $SERVER_PORT..."

# 循环检查 Server 是否启动 (最多等 60 秒)
MAX_RETRIES=30
count=0
while true; do
    # 尝试访问 Server 的根路径或健康检查接口 (假设 server 代码里监听了端口)
    # 这里我们简单检查端口是否被监听 (利用 nc 或 netstat 或 python)
    # 或者更直接点，用 curl 试探一下
    
    # 尝试 curl 发送一个空请求，如果连接上了说明端口开了
    if curl -s "http://$SERVER_HOST:$SERVER_PORT/docs" > /dev/null; then
        echo "[Script] Server is UP and READY!"
        break
    fi
    
    # 检查进程是否意外挂了
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "[Script] Error: Server process died unexpectedly! Check server.log."
        cat server.log
        exit 1
    fi

    count=$((count+1))
    if [ $count -ge $MAX_RETRIES ]; then
        echo "[Script] Error: Server timed out waiting for start."
        cat server.log
        exit 1
    fi
    
    echo -n "."
    sleep 2
done

echo ""
echo "=================================================="
echo "[Script] Step 2: Starting Training..."
echo "=================================================="

export DEBUG_MODE="true"

# 启动训练
CUDA_VISIBLE_DEVICES=$TRAINING_GPUS torchrun --nproc_per_node="8" \
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
    --max_steps 1500 \
    --run_name Myself-GRPO-Role-Playing \
    --save_steps 100 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 8

# 训练结束后，trap 会自动执行 cleanup 杀掉 server
echo "=================================================="
echo "[Script] Training finished."
