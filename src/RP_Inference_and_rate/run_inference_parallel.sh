#!/bin/bash
set -u # 使用未初始化变量时报错

# 配置路径
MODEL_PATH="/data/wm/Video-R1/Qwen2.5-VL-7B-Instruct"  # 改
# BASE_MODEL_PATH="/data/wm/Video-R1/Qwen2.5-VL-7B-Instruct"  # 改/不需要的话设为空
BASE_MODEL_PATH=""   # 改/需要的话加上路径
INPUT_FILE="/data/wm/simple-subtitling/Processed_Dialogue/RP-EBM-Dataset/RP-test.json"
OUTPUT_FILE="/data/wm/Video-R1/src/RP_Inference_and_rate/Inference_result/QWEN2_5_our_cot_no_training.json"  # 改
MODE="ours"  # 改
SCRIPT_PATH="/data/wm/Video-R1/src/RP_Inference_and_rate/inference_all_prompt_res.py"

# 确保输出目录存在
mkdir -p "$(dirname "$OUTPUT_FILE")"

# 显卡数量
NUM_GPUS=8

echo "Starting $NUM_GPUS parallel workers..."

# 构建 base_model_arg 字符串
# 如果 BASE_MODEL_PATH 非空，则拼接参数；否则为空
if [ -n "$BASE_MODEL_PATH" ]; then
    BASE_MODEL_ARG="--base_model_path $BASE_MODEL_PATH"
    echo "Using Base Model Path for Configs: $BASE_MODEL_PATH"
else
    BASE_MODEL_ARG=""
    echo "No Base Model Path provided. Will try loading configs from Checkpoint."
fi

for ((i=0; i<NUM_GPUS; i++)); do
    echo "Launching worker $i on GPU $i..."

    # # 使用括号启动子shell，确保后台运行正确
    # (
    #     export CUDA_VISIBLE_DEVICES=$i
    #     # 去掉 nohup，直接重定向
    #     python "$SCRIPT_PATH" \
    #         --model_path "$MODEL_PATH" \
    #         --input_file "$INPUT_FILE" \
    #         --output_file "$OUTPUT_FILE" \
    #         --mode "$MODE" \
    #         --num_shards $NUM_GPUS \
    #         --shard_id $i \
    #         > "log_worker_$i.log" 2>&1
    # ) & 

    
    # 使用括号启动子shell
    (
        export CUDA_VISIBLE_DEVICES=$i
        
        # === 核心修改区 ===
        if [ "$i" -eq 0 ]; then
            # 【GPU 0】: 使用 tee 命令。
            # 作用：既把输出写入 log 文件，同时也在屏幕上显示（让你看到进度条）
            python -u "$SCRIPT_PATH" \
                --model_path "$MODEL_PATH" \
                $BASE_MODEL_ARG \
                --input_file "$INPUT_FILE" \
                --output_file "$OUTPUT_FILE" \
                --mode "$MODE" \
                --num_shards $NUM_GPUS \
                --shard_id $i \
                2>&1 | tee "log_worker_$i.log"
        else
            # 【GPU 1-7】: 保持静默，只写入文件。
            # 防止 8 个进度条在屏幕上打架
            python -u "$SCRIPT_PATH" \
                --model_path "$MODEL_PATH" \
                $BASE_MODEL_ARG \
                --input_file "$INPUT_FILE" \
                --output_file "$OUTPUT_FILE" \
                --mode "$MODE" \
                --num_shards $NUM_GPUS \
                --shard_id $i \
                > "log_worker_$i.log" 2>&1
        fi
        # ==================
    ) & 
done

echo "All workers launched! Waiting for them to finish..."
wait
echo "All inference tasks completed or crashed."

# 检查是否真的有文件生成了
base_name="${OUTPUT_FILE%.*}"
count=$(ls ${base_name}_part*.json 2>/dev/null | wc -l)

if [ "$count" -eq 0 ]; then
    echo "❌ ERROR: No result files found! Please check log_worker_0.log for details."
    echo "--- Last 20 lines of log_worker_0.log ---"
    tail -n 20 log_worker_0.log
    exit 1
fi

echo "Merging results..."
python -c "
import json, glob, os
base_name = '${OUTPUT_FILE%.*}'
pattern = f'{base_name}_part*.json'

print(f'Looking for files matching: {pattern}')
files = sorted(glob.glob(pattern))
all_res = []
if not files:
    print('No files found.')
else:
    for f in files:
        if os.path.exists(f): 
            try:
                print(f'Loading {f}...')
                with open(f) as fp: 
                    data = json.load(fp)
                    all_res.extend(data)
            except Exception as e:
                print(f'Error loading {f}: {e}')

    if all_res:
        with open('${OUTPUT_FILE}', 'w') as fp:
            json.dump(all_res, fp, indent=4, ensure_ascii=False)
        print(f'Merged {len(all_res)} samples into ${OUTPUT_FILE}')
    else:
        print('Merged data is empty.')
"
