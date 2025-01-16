#!/bin/bash
current_project="safe_reason"
output_dir=$MY_PROJECT/${current_project}
mkdir -p $output_dir
echo "num_nodes: $SLURM_NNODES"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=3
mkdir -p server_logs

server_type=""
experiment_name="llama3_sft_beavertails_sft_full_5_1_100_tbs1_ebs1_gas16_lr5e-6_cl4096"
export model_name_or_path=${output_dir}/${experiment_name}
# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --server_type=*) server_type="${1#*=}"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

if [ -z "$server_type" ] || { [ "$server_type" != "api" ] && [ "$server_type" != "chat" ]; }; then
    echo "Error: Invalid or missing server_type. Please specify 'api' or 'chat'."
    exit 1
fi

if [ "$server_type" == "api" ]; then
    # api server
    suffix="generate"
    experiment_name=${experiment_name}_${suffix}
    envsubst < examples/inference/llama3_vllm.yaml > server_logs/${experiment_name}.yaml
    nohup script -f -a -c "API_PORT=8000 llamafactory-cli api server_logs/${experiment_name}.yaml" server_logs/${experiment_name}.log > /dev/null 2>&1 &
fi

if [ "$server_type" == "chat" ]; then
    # chat server
    suffix="chat"
    experiment_name=${experiment_name}_${suffix}
    envsubst < examples/inference/llama3_vllm.yaml > server_logs/${experiment_name}.yaml
    script -f -a -c "FORCE_TORCHRUN=1 NNODES=1 llamafactory-cli chat server_logs/${experiment_name}.yaml" server_logs/${experiment_name}.log
fi