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

suffix="chat"
experiment_name="llama3_sft_gsm8k_tbs16_ebs16_lr1e-6_cl2048"
export model_name_or_path="$output_dir/${experiment_name}"
export dataset="gsm8k"

mkdir -p logs
experiment_name=${experiment_name}_${suffix}

envsubst < examples/inference/llama3_vllm.yaml > logs/${experiment_name}.yaml

script -f -c "FORCE_TORCHRUN=1 NNODES=1  llamafactory-cli chat logs/${experiment_name}.yaml" logs/${experiment_name}.log

# script -f -c "python -u scripts/vllm_infer.py --model_name_or_path $model_name_or_path --dataset $dataset" logs/${experiment_name}.log 2>&1

 