#!/bin/bash
current_project="safe_reason"
log_dir=${current_project}_logs
output_dir=${MY_PROJECT}/${current_project}
mkdir -p $output_dir
echo "num_nodes: $SLURM_NNODES"
echo "SLURM_NODEID: $SLURM_NODEID"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES=1
mkdir -p server_logs
mkdir -p $log_dir

prefix="ours"
model="llama3_sft_sr_meta-llama-Meta-Llama-3-8B-Instruct_20_50_tbs8_ebs8_gas2_lr2e-5_cl4096"
# ['llama3_sft_gsm8k_tbs16_ebs16_lr1e-6_cl2048', 'gpt-4o-mini-2024-07-18', 'gpt-4o-2024-11-20', 'llama-3-8b-instruct', 'gemma-2-9b-it']
model_name_or_path=${output_dir}/${model}
# ['gpt-4o-mini-2024-07-18', 'gpt-4o-2024-11-20']
judge_model='gpt-4o-mini-2024-07-18'
# ['strongreject', 'strongreject_small', 'advbench', 'hex_phi', 'xstest']
dataset='strongreject'
# dataset='xstest'
# ['happy_to_help', 'pair', 'none', 'wikipedia', 'distractors', 'prefix_injection', 'combination_2', 'pap_misrepresentation']
jailbreak='none,pair,pap_misrepresentation'
# jailbreak='none'
# ['none', 'cot_specification', 'cot_specification_simplified', 'cot_instruction', 'cot_simple']
cot_prompt='none'
# cot_prompt='cot_specification'
evaluator='strongreject_rubric'
temperature=0
top_p=1
max_length=4096
experiment_name=${prefix}_${dataset}_${jailbreak}_${cot_prompt}_${model}_${judge_model}_${evaluator}
# export model_name_or_path="$output_dir/$model"
export model_name_or_path=$model_name_or_path

# envsubst < examples/inference/llama3_vllm.yaml > server_logs/${experiment_name}.yaml
# nohup script -f -a -c "API_PORT=8000 llamafactory-cli api server_logs/${experiment_name}.yaml" server_logs/${experiment_name}.log > /dev/null 2>&1 &
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
