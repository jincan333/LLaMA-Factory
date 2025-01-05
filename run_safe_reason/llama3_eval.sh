#!/bin/bash
current_project="safe_reason"
output_dir=$MY_PROJECT/${current_project}
mkdir -p $output_dir
echo "num_nodes: $SLURM_NNODES"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=29501
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=3

suffix="eval"
experiment_name="llama3_sft_gsm8k_tbs16_ebs16_lr1e-6_cl2048"
model_name_or_path="$output_dir/${experiment_name}"
# model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct"
dataset="gsm8k"
n_shot=5
batch_size=64
output_dir=$output_dir/${experiment_name}

mkdir -p logs
mkdir -p eval

# nohup accelerate launch -m lm_eval --model hf \
#     --model_args pretrained=${model_name_or_path},dtype=auto \
#     --tasks $dataset \
#     --num_fewshot $n_shot \
#     --gen_kwargs top_p=1,temperature=0 \
#     --output_path $output_dir/${experiment_name}/${dataset} \
#     --log_samples \
#     --write_out \
#     --apply_chat_template \
#     --wandb_args project=$current_project,name=${experiment_name}_${suffix} \
#     --batch_size $batch_size \
#     > logs/${experiment_name}_${suffix}.log 2>&1 &

nohup lm_eval --model vllm \
    --model_args pretrained=${model_name_or_path},tensor_parallel_size=1,dtype=auto \
    --tasks $dataset \
    --num_fewshot $n_shot \
    --gen_kwargs top_p=1,temperature=0 \
    --output_path $output_dir/${experiment_name}/${dataset} \
    --log_samples \
    --write_out \
    --apply_chat_template \
    --wandb_args project=$current_project,name=${experiment_name}_${suffix} \
    --batch_size $batch_size \
    > logs/${experiment_name}_${suffix}.log 2>&1 &
