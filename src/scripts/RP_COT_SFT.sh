cd src/r1-v

export NCCL_NVLS_ENABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export IGNORE_MISMATCHED_SIZES=true
export ACCELERATE_USE_DEEPSPEED=true
export DEEPSPEED_CONFIG_FILE="local_scripts/zero3.json"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12349" \
    src/open_r1/RP_cot_sft.py \
    --output_dir "./log/Qwen2.5-VL-7B-Role-Playing-COT-SFT" \
    --model_name_or_path "../../Qwen2.5-VL-7B-Instruct" \
    --dataset_name "../../../simple-subtitling/Processed_Dialogue/RP-RL-Dataset/final_train_COT_SFT.json" \
    --deepspeed local_scripts/zero3.json \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 3e-6 \
    --warmup_ratio 0.1 \
    --logging_steps 1 \
    --bf16 True\
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 4 \
    --run_name Qwen2.5-VL-Role-Playing-COT-SFT \
    --eval_strategy "steps" \
    --eval_steps 40 \
    --save_strategy "steps" \
    --save_steps 40 \
    --per_device_eval_batch_size 1 \
    --max_grad_norm 5 \