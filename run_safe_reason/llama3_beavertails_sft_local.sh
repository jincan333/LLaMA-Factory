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
export dataset="beavertails_sft_full_5_1_100"
export per_device_train_batch_size=1
export per_device_eval_batch_size=1
export gradient_accumulation_steps=16
export learning_rate=5e-6
export cutoff_len=4096
export experiment_name=${prefix}_${dataset}_tbs${per_device_train_batch_size}_ebs${per_device_eval_batch_size}_gas${gradient_accumulation_steps}_lr${learning_rate}_cl${cutoff_len}
export output_dir=${output_dir}/${experiment_name}
export WANDB_PROJECT=${current_project}

mkdir -p $output_dir
echo "output_dir: $output_dir"
mkdir -p logs

envsubst < examples/train_full/llama3_full_sft.yaml > safe_reason_logs/${experiment_name}.yaml

FORCE_TORCHRUN=1 NNODES=1 llamafactory-cli train safe_reason_logs/${experiment_name}.yaml
#  > logs/${experiment_name}.log 2>&1