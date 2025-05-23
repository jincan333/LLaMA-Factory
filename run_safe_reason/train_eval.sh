#!/bin/bash
current_project="safe_reason"
output_dir=$MY_PROJECT/${current_project}
mkdir -p $output_dir
echo "num_nodes: $SLURM_NNODES"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=29501

experiment_name=$1
output_dir=${output_dir}/${experiment_name}

dataset="bbh_zeroshot"
n_shot=0
batch_size=128
echo "evaldataset: $dataset"
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${output_dir},dtype=auto \
    --tasks $dataset \
    --num_fewshot $n_shot \
    --gen_kwargs temperature=0 \
    --output_path ${output_dir}/${dataset} \
    --apply_chat_template \
    --batch_size $batch_size
# dataset="gsm8k"
# n_shot=5
# batch_size=128
# export ACCELERATE_LOG_LEVEL=info
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# echo "evaldataset: $dataset"
# accelerate launch -m lm_eval --model hf \
#     --model_args pretrained=${model_name_or_path},dtype=auto \
#     --tasks $dataset \
#     --num_fewshot $n_shot \
#     --gen_kwargs top_p=1,temperature=0 \
#     --output_path ${output_dir}/${dataset} \
#     --log_samples \
#     --write_out \
#     --apply_chat_template \
#     --wandb_args project=$current_project,name=${experiment_name}_${suffix} \
#     --batch_size $batch_size

dataset="mmlu"
n_shot=5
batch_size=16
echo "evaldataset: $dataset"
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${output_dir},dtype=auto \
    --tasks $dataset \
    --num_fewshot $n_shot \
    --gen_kwargs temperature=0 \
    --output_path ${output_dir}/${dataset} \
    --apply_chat_template \
    --batch_size $batch_size

# dataset="hellaswag"
# n_shot=10
# batch_size=64
# echo "evaldataset: $dataset"
# accelerate launch -m lm_eval --model hf \
#     --model_args pretrained=${model_name_or_path},dtype=auto \
#     --tasks $dataset \
#     --num_fewshot $n_shot \
#     --gen_kwargs temperature=0 \
#     --output_path ${output_dir}/${dataset} \
#     --log_samples \
#     --write_out \
#     --apply_chat_template \
#     --wandb_args project=$current_project,name=${experiment_name}_${dataset}_${suffix} \
#     --batch_size $batch_size
