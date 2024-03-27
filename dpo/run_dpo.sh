#!/bin/bash
export WANDB_ENTITY='dpo'
export WANDB_PROJECT='code-dpo'

run_name="dpo-run"
sft_model_checkpoint="dpo/checkpoints/sft_checkpoint"
config_file="dpo/config_dpo.yaml"
output_dir="dpo/checkpoints/${run_name}"
data_path="dpo/data"

accelerate launch \
    --config_file $config_file dpo/run_dpo.py \
    --model_name_or_path=${sft_model_checkpoint} \
    --data_dir=$data_path \
    --use_local 0 \
    --output_dir $output_dir \
    --logging_steps 5 \
    --max_steps 2000 \
    --save_steps 50 \
    --evaluation_strategy steps \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --max_length 2048 \
    --report_to wandb \
    --eval_steps 25 \
    --run_name $run_name 

