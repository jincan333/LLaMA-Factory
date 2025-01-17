#!/bin/bash
current_project="safe_reason"
output_dir=$MY_PROJECT/${current_project}
mkdir -p $output_dir
echo "num_nodes: $SLURM_NNODES"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export MASTER_ADDR=$HOSTNAME
export MASTER_PORT=29502
export OMP_NUM_THREADS=8

prefix="llama3_dpo"
export dataset="gsm8k_rectify_dpo"
export per_device_train_batch_size=4
export per_device_eval_batch_size=8
export gradient_accumulation_steps=4
export learning_rate=1e-6
export cutoff_len=4096
export experiment_name=${prefix}_${dataset}_tbs${per_device_train_batch_size}_ebs${per_device_eval_batch_size}_gas${gradient_accumulation_steps}_lr${learning_rate}_cl${cutoff_len}
export output_dir=${output_dir}/${experiment_name}
export WANDB_PROJECT=${current_project}

mkdir -p $output_dir
echo "output_dir: $output_dir"
mkdir -p logs

envsubst < examples/train_full/llama3_full_dpo.yaml > safe_reason_logs/${experiment_name}.yaml

FORCE_TORCHRUN=1 NNODES=1 llamafactory-cli train safe_reason_logs/${experiment_name}.yaml
#  > logs/${experiment_name}.log 2>&1

suffix="eval"
model_name_or_path=${output_dir}
dataset="gsm8k"
n_shot=5
batch_size=128
export ACCELERATE_LOG_LEVEL=info
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "evaldataset: $dataset"
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${model_name_or_path},dtype=auto \
    --tasks $dataset \
    --num_fewshot $n_shot \
    --gen_kwargs top_p=1,temperature=0 \
    --output_path ${output_dir}/${dataset} \
    --log_samples \
    --write_out \
    --apply_chat_template \
    --wandb_args project=$current_project,name=${experiment_name}_${suffix} \
    --batch_size $batch_size

dataset="mmlu"
n_shot=5
batch_size=32
echo "evaldataset: $dataset"
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${model_name_or_path},dtype=auto \
    --tasks $dataset \
    --num_fewshot $n_shot \
    --gen_kwargs temperature=0 \
    --output_path ${output_dir}/${dataset} \
    --log_samples \
    --write_out \
    --apply_chat_template \
    --wandb_args project=$current_project,name=${experiment_name}_${dataset}_${suffix} \
    --batch_size $batch_size

dataset="hellaswag"
n_shot=10
batch_size=64
echo "evaldataset: $dataset"
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${model_name_or_path},dtype=auto \
    --tasks $dataset \
    --num_fewshot $n_shot \
    --gen_kwargs temperature=0 \
    --output_path ${output_dir}/${dataset} \
    --log_samples \
    --write_out \
    --apply_chat_template \
    --wandb_args project=$current_project,name=${experiment_name}_${dataset}_${suffix} \
    --batch_size $batch_size

dataset="bbh_zeroshot"
n_shot=0
batch_size=128
echo "evaldataset: $dataset"
accelerate launch -m lm_eval --model hf \
    --model_args pretrained=${model_name_or_path},dtype=auto \
    --tasks $dataset \
    --num_fewshot $n_shot \
    --gen_kwargs temperature=0 \
    --output_path ${output_dir}/${dataset} \
    --log_samples \
    --write_out \
    --apply_chat_template \
    --wandb_args project=$current_project,name=${experiment_name}_${dataset}_${suffix} \
    --batch_size $batch_size
