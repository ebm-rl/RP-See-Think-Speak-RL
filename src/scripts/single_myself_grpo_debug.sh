#!/bin/bash
cd src/r1-v

# 单卡调试
export CUDA_VISIBLE_DEVICES=0
export DEBUG_MODE="true" 
export PYTHONUNBUFFERED=1

# 关键：不要用 torchrun / deepspeed
python src/open_r1/myself_grpo.py \
    --output_dir "./myself_grpo_sigle_debug_run" \
    --model_name_or_path '/data/wm/Video-R1/Qwen2.5-VL-7B-Instruct' \
    --dataset_name "/data/wm/simple-subtitling/Processed_Dialogue/The_Twilight_Saga/final_dialog_samples_data.json" \
    --max_prompt_length 4096 \
    --max_completion_length 256 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --bf16 True \
    --logging_steps 1 \
    --gradient_checkpointing True \
    --temporal False \
    --len_control False \
    --max_pixels 401408 \
    --num_train_epochs 1 \
    --max_steps 1500 \
    --run_name "debug-myself-grpo-single-gpu" \
    --save_steps 100 \
    --beta 0.04 \
    --max_grad_norm 5 \
    --save_only_model false \
    --num_generations 8  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
