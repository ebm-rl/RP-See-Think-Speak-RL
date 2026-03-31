#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -u

MODEL_PATH="./Models/Qwen2.5-VL-7B-Instruct"  
BASE_MODEL_PATH=""
INPUT_FILE="./src/RP_Inference_and_rate/Generalization/Next-QA/infer_result/NExTQA_CH_Inference-7b-instruct-ALL.json" 
OUTPUT_FILE="./src/RP_Inference_and_rate/Generalization/Next-QA/infer_result/NExTQA_CH_Inference-7b-instruct-ALL.json"
MODE="normal_cot" 
SCRIPT_PATH="./src/RP_Inference_and_rate/Generalization/Next-QA/infer_new.py"
SAMPLE_SIZE=1173
SAMPLE_SEED=42

mkdir -p "$(dirname "$OUTPUT_FILE")"


TARGET_GPUS=(0 1)

NUM_GPUS=${#TARGET_GPUS[@]}

echo "Selected Physical GPUs: ${TARGET_GPUS[*]}"
echo "Total Workers: $NUM_GPUS"

if [ -n "$BASE_MODEL_PATH" ]; then
    BASE_MODEL_ARG="--base_model_path $BASE_MODEL_PATH"
    echo "Using Base Model Path for Configs: $BASE_MODEL_PATH"
else
    BASE_MODEL_ARG=""
    echo "No Base Model Path provided. Will try loading configs from Checkpoint."
fi

for ((i=0; i<NUM_GPUS; i++)); do
    MY_GPU=${TARGET_GPUS[$i]}
    echo "Launching worker $i on GPU $i..."

    (
        # export CUDA_VISIBLE_DEVICES=$i
        export CUDA_VISIBLE_DEVICES=$MY_GPU
        
        if [ "$i" -eq 0 ]; then
            python -u "$SCRIPT_PATH" \
                --model_path "$MODEL_PATH" \
                $BASE_MODEL_ARG \
                --input_file "$INPUT_FILE" \
                --output_file "$OUTPUT_FILE" \
                --mode "$MODE" \
                --num_shards $NUM_GPUS \
                --shard_id $i \
                --sample_size $SAMPLE_SIZE \
                --sample_seed $SAMPLE_SEED \
                2>&1 | tee "log_worker_$i.log"
        else
            python -u "$SCRIPT_PATH" \
                --model_path "$MODEL_PATH" \
                $BASE_MODEL_ARG \
                --input_file "$INPUT_FILE" \
                --output_file "$OUTPUT_FILE" \
                --mode "$MODE" \
                --num_shards $NUM_GPUS \
                --shard_id $i \
                --sample_size $SAMPLE_SIZE \
                --sample_seed $SAMPLE_SEED \
                > "log_worker_$i.log" 2>&1
        fi
    ) & 
done

echo "All workers launched! Waiting for them to finish..."
wait
echo "All inference tasks completed or crashed."

base_name="${OUTPUT_FILE%.*}"
count=$(ls ${base_name}_part*.json 2>/dev/null | wc -l)

if [ "$count" -eq 0 ]; then
    echo " ERROR: No result files found! Please check log_worker_0.log for details."
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

if [ -s "$OUTPUT_FILE" ]; then
    echo "Merge SUCCESS! Final file created: $OUTPUT_FILE"
    echo "Cleaning up temporary part files..."
    
    base_name="${OUTPUT_FILE%.*}"
    rm -f "${base_name}"_part*.json
    rm -f "${base_name}"_part*.jsonl
    echo " Temporary files deleted."
else
    echo " WARNING: Merged file is missing or empty. Keeping part files for debugging."
fi
