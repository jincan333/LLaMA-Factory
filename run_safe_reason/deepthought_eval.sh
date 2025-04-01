#!/bin/bash
source ~/.bashrc
source "/home/colligo/miniforge3/etc/profile.d/conda.sh"  # Changed conda initialization
conda activate llamafactory
current_project="safe_reason"
log_dir=${current_project}_logs
output_dir=${MY_PROJECT}/${current_project}
mkdir -p $output_dir
echo "num_nodes: $SLURM_NNODES "
echo "SLURM_NODEID: $SLURM_NODEID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES=6
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p server_logs
mkdir -p $log_dir

prefix="rebuttal"
model='deepthought-8b'
# ['llama3_sft_gsm8k_tbs16_ebs16_lr1e-6_cl2048', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-11-20', 'llama-3-8b-instruct', 'gemma-2-9b-it']
model_name_or_path="ruliad/deepthought-8b-llama-v0.01-alpha"
# ['gpt-4o-mini-2024-07-18', 'gpt-4o-2024-11-20']
judge_model='gpt-4o-mini-2024-07-18'
# ['strongreject', 'strongreject_small', 'advbench', 'hex_phi', 'xstest']
# dataset='strongreject'
dataset='harmbench'
# ['happy_to_help', 'pair', 'none', 'wikipedia', 'distractors', 'prefix_injection', 'combination_2', 'pap_misrepresentation']
# jailbreak='none,pair,pap_misrepresentation'
jailbreak='none,gcg_transfer_harmbench,distractors'
# ['none', 'cot_specification', 'cot_classification_specification', 'cot_instruction', 'cot_simple', 'cot_helpful', 'cot_specification_helpful']
cot_prompt='cot_specification'
# cot_prompt='none'
evaluator='strongreject_rubric'
temperature=0
top_p=1
max_length=4096
experiment_name=${prefix}_${dataset}_${jailbreak}_${cot_prompt}_${model}_${judge_model}_${evaluator}
# export model_name_or_path="$output_dir/$model"
export model_name_or_path=$model_name_or_path

# envsubst < examples/inference/llama3_vllm.yaml > server_logs/${experiment_name}.yaml
# nohup script -f -a -c "API_PORT=8001 llamafactory-cli api server_logs/${experiment_name}.yaml" server_logs/${experiment_name}.log > /dev/null 2>&1 &
# delay=40
# echo "Waiting for $delay seconds..."
# sleep $delay

nohup python -u ${current_project}/evaluate.py \
    --model $model_name_or_path \
    --judge_model $judge_model \
    --dataset $dataset \
    --jailbreak $jailbreak \
    --cot_prompt $cot_prompt \
    --evaluator $evaluator \
    --temperature $temperature \
    --top_p $top_p \
    --max_length $max_length \
    --generate "true" \
    --evaluate "true" \
> ${log_dir}/${experiment_name}.log 2>&1 &
