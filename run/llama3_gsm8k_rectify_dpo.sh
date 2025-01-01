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

prefix="llama3_sft"
export dataset="gsm8k_real_dpo"
export per_device_train_batch_size=16
export per_device_eval_batch_size=16
export gradient_accumulation_steps=1
export learning_rate=1e-6
export cutoff_len=2048
export output_dir=${output_dir}/${experiment_name}
export WANDB_PROJECT=${current_project}
export experiment_name=${prefix}_${dataset}_tbs${per_device_train_batch_size}_ebs${per_device_eval_batch_size}_lr${learning_rate}_cl${cutoff_len}

mkdir -p $output_dir
echo "output_dir: $output_dir"
mkdir -p logs

envsubst < examples/train_full/llama3_full_sft.yaml > logs/${experiment_name}.yaml

FORCE_TORCHRUN=1 NNODES=1 llamafactory-cli train logs/${experiment_name}.yaml \
#  > logs/${experiment_name}.log 2>&1
 