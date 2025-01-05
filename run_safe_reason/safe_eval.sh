#!/bin/bash
current_project="safe_reason"
log_dir=${current_project}_logs
output_dir=${MY_PROJECT}/${current_project}
mkdir -p $output_dir
echo "num_nodes: $SLURM_NNODES"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES=3
mkdir -p server_logs
mkdir -p $log_dir

prefix="safe_eval"
# ['llama3_sft_gsm8k_tbs16_ebs16_lr1e-6_cl2048', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-11-20', 'llama-3-8b-instruct', 'gemma-2-9b-it']
model='llama3_sft_gsm8k_tbs16_ebs16_lr1e-6_cl2048'
# ['gpt-4o-mini-2024-07-18', 'gpt-4o-2024-11-20']
judge_model='gpt-4o-mini-2024-07-18'
# ['strongreject', 'strongreject_small', 'advbench', 'hex_phi', 'xstest']
dataset='hex_phi'
# ['happy_to_help', 'pair']
jailbreak='happy_to_help'
evaluator='strongreject_rubric'
temperature=0.7
top_p=0.95
max_length=2048
experiment_name=${prefix}_${dataset}_${jailbreak}_${model}_${judge_model}_${evaluator}
export model_name_or_path="$output_dir/$model"

if [ ! -f server_logs/${experiment_name}.yaml ] && [[ ${model} != *"gpt"* ]]; then
    envsubst < examples/inference/llama3_vllm.yaml > server_logs/${experiment_name}.yaml
    nohup script -f -a -c "API_PORT=8000 llamafactory-cli api server_logs/${experiment_name}.yaml" server_logs/${experiment_name}.log > /dev/null 2>&1 &
    delay=15
    echo "Waiting for $delay seconds..."
    sleep $delay
fi

nohup python -u ${current_project}/evaluate.py \
    --model $model \
    --judge_model $judge_model \
    --dataset $dataset \
    --jailbreak $jailbreak \
    --evaluator $evaluator \
    --temperature $temperature \
    --top_p $top_p \
    --max_length $max_length \
    --generate "false" \
    --evaluate "false" \
> ${log_dir}/${experiment_name}.log 2>&1 &
