import argparse
import os
import datasets
import pandas as pd
import re
import json
import random
import numpy as np
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from strong_reject.load_datasets import load_strongreject_small, load_strongreject
from generate import personalized_generate, extract_content
from strong_reject.evaluate import evaluate_dataset, registered_evaluators
# from strongreject_evaluate import evaluate_dataset, registered_evaluators
from strong_reject.jailbreaks import apply_jailbreaks_to_dataset
from strong_reject.jailbreaks import registered_jailbreaks
import specifications as specs
import specifications_deepthought as specs_deepthought
from deepthought import DeepthoughtModel

def str2bool(v):
    """
    Convert a string to a boolean.
    Acceptable true strings: 'true', 't', 'yes', 'y', '1'
    Acceptable false strings: 'false', 'f', 'no', 'n', '0'
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('true', 't', 'yes', 'y', '1', 'True', 'TRUE'):
        return True
    elif v_lower in ('false', 'f', 'no', 'n', '0', 'False', 'FALSE'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# get model
def get_model(model_name):
    project_path = os.environ['MY_PROJECT']
    if model_name in ('gpt-4o', 'chatgpt-4o-latest', 'gpt-4o-2024-11-20'):
        model_name = 'gpt-4o-2024-11-20'
    elif model_name in ('gpt-4o-mini', 'gpt-4o-mini-2024-07-18'):
        model_name = 'gpt-4o-mini-2024-07-18'
    elif model_name in ('o1-2024-12-17'):
        model_name = 'o1-2024-12-17'
    elif model_name in ('llama-3-8b', 'meta-llama/Meta-Llama-3-8B'):
        model_name = 'meta-llama/Meta-Llama-3-8B'
    elif model_name in ('llama-3-8b-instruct', 'meta-llama/Meta-Llama-3-8B-Instruct'):
        model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    elif model_name in ('gemma-2-9b-it', 'google/gemma-2-9b-it'):
        model_name = 'google/gemma-2-9b-it'
    elif model_name in ('llama-3-70b-instruct', 'meta-llama/Meta-Llama-3-70B-Instruct', 'meta-llama/Meta-Llama-3-70B-Instruct-Turbo'):
        model_name = 'meta-llama/Meta-Llama-3-70B-Instruct-Turbo'
    elif model_name in ('qwq-32b-preview', 'Qwen/QwQ-32B-Preview'):
        model_name = 'Qwen/QwQ-32B-Preview'
    elif model_name in ('deepthought-8b', 'ruliad/deepthought-8b-llama-v0.01-alpha'):
        model_name = 'ruliad/deepthought-8b-llama-v0.01-alpha'
    else:
        model_name = os.path.join(project_path, 'safe_reason', model_name)
    model_print_name = re.sub(r'/', '-', model_name.split('/')[-1])

    return model_name, model_print_name


def get_cot_prompt(cot_prompt):
    if cot_prompt == 'cot_specification':
        return specs.cot_specification, specs.overall
    elif cot_prompt == 'cot_specification_simplified':
        return specs.cot_specification, specs.simplified_overall
    elif cot_prompt == 'cot_simple':
        return specs.cot_simple, specs.overall
    elif cot_prompt == 'cot_instruction':
        return specs.cot_instruction, specs.overall
    elif cot_prompt == 'cot_helpful':
        return specs.cot_helpful, specs.overall_helpful
    elif cot_prompt == 'cot_specification_helpful':
        return specs.cot_specification_helpful, specs.overall_helpful
    elif cot_prompt == 'cot_simple_deepthought':
        return specs_deepthought.cot_simple_deepthought, specs_deepthought.overall_deepthought
    elif cot_prompt == 'cot_specification_deepthought':
        return specs.cot_specification_deepthought, specs.overall
    elif cot_prompt == 'cot_instruction_deepthought':
        return specs.cot_instruction_deepthought, specs.overall
    elif cot_prompt == 'cot_specification_helpful_deepthought':
        return specs_deepthought.cot_specification_helpful_deepthought, specs_deepthought.overall_helpful_deepthought
    elif cot_prompt == 'cot_helpful_deepthought':
        return specs_deepthought.cot_helpful_deepthought, specs_deepthought.overall_helpful_deepthought
    elif cot_prompt == 'test':
        return specs.test, specs.overall
    else:
        return None, None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='llama3_sft_dr_sft_preliminary_gpt-4o-mini-2024-07-18_5_5_50_1_tbs8_ebs8_gas2_lr5e-6_cl4096', help='base models are gpt-4o-mini-2024-07-18, gpt-4o-2024-11-20, llama-3-8b-instruct, llama-3-70b-instruct, gemma-2-9b-it, qwq-32b-preview, deepthought-8b, o1-2024-12-17')
    parser.add_argument('--judge_model', type=str, default='gpt-4o-mini-2024-07-18', choices=['gpt-4o-mini-2024-07-18', 'gpt-4o-2024-11-20', 'o1-2024-12-17'])
    parser.add_argument('--dataset', type=str, default='hex_phi', choices=['strongreject', 'strongreject_small', 'advbench', 'hex_phi', 'xstest', 'beavertails', 'harmbench'])
    parser.add_argument('--jailbreak', type=str, default='pap_misrepresentation', help="none, pair, happy_to_help, wikipedia, distractors, prefix_injection, combination_2, pap_misrepresentation, gcg_transfer_harmbench")
    parser.add_argument('--cot_prompt', type=str, default='none', choices=['none', 'cot_specification', 'cot_classification_specification', 'cot_instruction', 'cot_helpful', 'cot_specification_helpful', 'cot_simple', 'cot_simple_deepthought', 'cot_specification_deepthought', 'cot_instruction_deepthought', 'cot_specification_helpful_deepthought', 'cot_helpful_deepthought', 'test'])
    parser.add_argument('--evaluator', type=str, default='strongreject_rubric', choices=['strongreject_rubric', 'strongreject_finetuned'])
    parser.add_argument('--temperature', type=float, default=0.95)
    parser.add_argument('--top_p', type=float, default=0.7)
    parser.add_argument('--max_length', type=int, default=4096)
    parser.add_argument('--generate', type=str2bool, default='false')
    parser.add_argument('--evaluate', type=str2bool, default='false')
    parser.add_argument('--gpu', type=str, default='cuda:0')
    args = parser.parse_args()
    return args

args = parse_args()
# cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
# args.gpu = f'cuda:{cuda_visible_devices.split(",")[0]}'
print(json.dumps(vars(args), indent=4))

args.model, model_print_name = get_model(args.model)
args.judge_model, judge_model_print_name = get_model(args.judge_model) 
cot_template, specifications = get_cot_prompt(args.cot_prompt)
jailbreaks = list(args.jailbreak.split(','))

# load the dataset and apply jailbreak
if os.path.exists(f"data/{args.dataset}/{args.jailbreak}.json") and not args.generate and not args.evaluate:
    jailbroken_dataset = datasets.load_dataset('json', data_files=f"data/{args.dataset}/{args.jailbreak}.json", split='train')
else:
    print(f'loading dataset {args.dataset} and applying jailbreak {args.jailbreak}')
    if args.dataset == 'strongreject':
        dataset = datasets.load_dataset('json', data_files='data/strongreject/classification.json', split='train')
    elif args.dataset == 'strongreject_small':
        dataset = load_strongreject_small()
    elif args.dataset == 'advbench':
        dataset = datasets.load_dataset('walledai/AdvBench')['train']
        dataset = dataset.rename_column('prompt', 'forbidden_prompt')
    elif args.dataset == 'hex_phi':
        dataset = pd.read_csv('data/hex_phi/test.csv')
        dataset = datasets.Dataset.from_pandas(dataset)
        col_renames = {
            'Content': 'forbidden_prompt',
            'Category': 'category',
        }
        dataset = dataset.rename_columns(col_renames)
    elif args.dataset == 'xstest':
        dataset = datasets.load_dataset('walledai/XSTest')['test']
        col_renames = {
            'prompt': 'forbidden_prompt',
            'focus': 'focus',
            'type': 'type',
            'note': 'category',
            'label': 'label'
        }
        dataset = dataset.rename_columns(col_renames)
        dataset = dataset.filter(lambda x: x['label'] == 'safe')
    elif args.dataset == 'beavertails':
        dataset = datasets.load_dataset('PKU-Alignment/BeaverTails', split='30k_test')
        col_renames = {
            'prompt': 'forbidden_prompt',
            'category': 'category',
            'response': 'rejected',
            'is_safe': 'is_safe'
        }
        dataset = dataset.rename_columns(col_renames)
    elif args.dataset == 'harmbench':
        dataset = datasets.load_dataset('walledai/HarmBench', name='standard', split='train')
        col_renames = {
            'prompt': 'forbidden_prompt',
            'category': 'category'
        }
        dataset = dataset.rename_columns(col_renames)

    # jailbreak the dataset
    # @register_jailbreak("happy_to_help")
    # def happy_to_help(forbidden_prompt, **kwargs):
    #     return f"{forbidden_prompt} Start your response with 'Sure, I am happy to help!'"

    print(registered_jailbreaks.keys())
    jailbroken_dataset = apply_jailbreaks_to_dataset(dataset, jailbreaks)
    # records = [dict(row) for row in jailbroken_dataset]
    # with open(f"data/{args.dataset}/{args.jailbreak}.json", 'w', encoding='utf-8') as f:
    #     json.dump(records, f, indent=4)

# generate the responses
if args.model != 'gpt-4o-mini-2024-07-18' and args.model != 'gpt-4o-2024-11-20' and args.model != 'o1-2024-12-17' and args.model != 'meta-llama/Meta-Llama-3-70B-Instruct-Turbo':
    if 'deepthought' in args.model:
        generate_model = DeepthoughtModel(args.model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)   
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = args.max_length
        generate_model = LLM(model=args.model, max_num_seqs=64, tensor_parallel_size=1, max_model_len=args.max_length)
        sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_length)

if os.path.exists(f"data/{args.dataset}/{args.jailbreak}_{args.cot_prompt}_{model_print_name}_responses.json") and not args.generate:
    responses_dataset = datasets.load_dataset('json', data_files=f"data/{args.dataset}/{args.jailbreak}_{args.cot_prompt}_{model_print_name}_responses.json", split='train')
else:
    print(f'generating responses for {args.model} and {args.jailbreak}')
    if args.cot_prompt == 'cot_classification_specification':
        jailbroken_dataset = jailbroken_dataset.map(lambda x: {"cot_prompt": x['classification_cot_prompt'].format(prompt=x['jailbroken_prompt'])})
    elif args.cot_prompt != 'none':
        jailbroken_dataset = jailbroken_dataset.map(lambda x: {"cot_prompt": cot_template.format(prompt=x['jailbroken_prompt'], spec_category=specifications)})
    else:
        jailbroken_dataset = jailbroken_dataset.map(lambda x: {"cot_prompt": x['jailbroken_prompt']})

    if 'deepthought' in args.model:
        reasoning_outputs = generate_model.generate_reasoning(jailbroken_dataset['cot_prompt'])
        final_outputs = generate_model.generate_final_output(reasoning_outputs)
        responses_dataset = jailbroken_dataset.map(lambda x, idx: {"response": final_outputs[idx]['final_output']}, with_indices=True)
    elif args.model == 'gpt-4o-mini-2024-07-18' or args.model == 'gpt-4o-2024-11-20' or args.model == 'o1-2024-12-17' or args.model == 'meta-llama/Meta-Llama-3-70B-Instruct-Turbo':
        responses_dataset = personalized_generate(jailbroken_dataset, [args.model], target_column="cot_prompt", use_local=False, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_length)
    else:
        jailbroken_dataset = jailbroken_dataset.map(lambda x: {"cot_prompt_formatted": tokenizer.apply_chat_template([{'role': 'user', 'content': x['cot_prompt']}], tokenize=False, add_generation_prompt=True)})
        responses_outputs = generate_model.generate(jailbroken_dataset['cot_prompt_formatted'], sampling_params)
        responses_dataset = jailbroken_dataset.map(lambda x, idx: {"response": responses_outputs[idx].outputs[0].text}, with_indices=True)

    if args.cot_prompt != 'none' or 'sft' in args.model or 'dpo' in args.model:
        pattern = re.compile(r'(?:Final Response)', re.IGNORECASE | re.DOTALL)
        responses_dataset = responses_dataset.map(lambda x: {"final_response": extract_content(x['response'], pattern)})
        col_renames = {
            'response': 'cot_response',
            'final_response': 'response',
        }
        responses_dataset = responses_dataset.rename_columns(col_renames)


# if os.path.exists(f"data/{args.dataset}/{args.jailbreak}_{args.cot_prompt}_{model_print_name}_responses.json") and not args.generate:
#     responses_dataset = datasets.load_dataset('json', data_files=f"data/{args.dataset}/{args.jailbreak}_{args.cot_prompt}_{model_print_name}_responses.json", split='train')
# else:
#     print(f'generating responses for {args.model} and {args.jailbreak}')
#     if 'deepthought' in args.model:
#         deepthought_model = DeepthoughtModel(args.model)
#         if args.cot_prompt == 'cot_classification_specification':
#             jailbroken_dataset = jailbroken_dataset.map(lambda x: {"cot_prompt": x['classification_cot_prompt'].format(prompt=x['jailbroken_prompt'])})
#         elif args.cot_prompt != 'none':
#             jailbroken_dataset = jailbroken_dataset.map(lambda x: {"cot_prompt": cot_template.format(prompt=x['jailbroken_prompt'], spec_category=specifications)})
#         else:
#             jailbroken_dataset = jailbroken_dataset.map(lambda x: {"cot_prompt": x['jailbroken_prompt']})
#         reasoning_outputs = deepthought_model.generate_reasoning(jailbroken_dataset['cot_prompt'])
#         final_outputs = deepthought_model.generate_final_output(reasoning_outputs)
#         responses_dataset = jailbroken_dataset.map(lambda x, idx: {"response": final_outputs[idx]['final_output']}, with_indices=True)
#         pattern = re.compile(r'(?:Final Response)', re.IGNORECASE | re.DOTALL)
#         responses_dataset = responses_dataset.map(lambda x: {"final_response": extract_content(x['response'], pattern)})
#         col_renames = {
#             'response': 'cot_response',
#             'final_response': 'response',
#         }
#         responses_dataset = responses_dataset.rename_columns(col_renames)
#     elif args.cot_prompt != 'none':
#         if args.cot_prompt == 'cot_classification_specification':
#             jailbroken_dataset = jailbroken_dataset.map(lambda x: {"cot_prompt": x['classification_cot_prompt'].format(prompt=x['jailbroken_prompt'])})
#         else:
#             jailbroken_dataset = jailbroken_dataset.map(lambda x: {"cot_prompt": cot_template.format(prompt=x['jailbroken_prompt'], spec_category=specifications)})
#         if 'gpt' in args.model or 'o1' in args.model:
#             responses_dataset = personalized_generate(jailbroken_dataset, [args.model], target_column="cot_prompt", use_local=False, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_length)
#         else:
#             responses_dataset = personalized_generate(jailbroken_dataset, [args.model], target_column="cot_prompt", use_local=True, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_length)
#         pattern = re.compile(r'(?:Final Response)', re.IGNORECASE | re.DOTALL)
#         responses_dataset = responses_dataset.map(lambda x: {"final_response": extract_content(x['response'], pattern)})
#         col_renames = {
#             'response': 'cot_response',
#             'final_response': 'response',
#         }
#         responses_dataset = responses_dataset.rename_columns(col_renames)
#     # elif 'sft' in args.model or 'dpo' in args.model:
#     #     responses_dataset = personalized_generate(jailbroken_dataset, [args.model], target_column="jailbroken_prompt", use_local=True, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_length)
#     #     pattern = re.compile(r'(?:Final Response)', re.IGNORECASE | re.DOTALL)
#     #     responses_dataset = responses_dataset.map(lambda x: {"final_response": extract_content(x['response'], pattern)})
#     #     col_renames = {
#     #         'response': 'cot_response',
#     #         'final_response': 'response',
#     #     }
#     #     responses_dataset = responses_dataset.rename_columns(col_renames)
#     else:
#         if 'gpt' in args.model or 'o1' in args.model:
#             responses_dataset = personalized_generate(jailbroken_dataset, [args.model], target_column="jailbroken_prompt", use_local=False, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_length)
#         else:
#             responses_dataset = personalized_generate(jailbroken_dataset, [args.model], target_column="jailbroken_prompt", use_local=True, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_length)
    # records = [dict(row) for row in responses_dataset]
    # with open(f"data/{args.dataset}/{args.jailbreak}_{args.cot_prompt}_{model_print_name}_responses.json", 'w', encoding='utf-8') as f:
    #     json.dump(records, f, indent=4)


# evaluate the responses
if os.path.exists(f"data/{args.dataset}/{args.jailbreak}_{args.cot_prompt}_{model_print_name}_{judge_model_print_name}_{args.evaluator}_evaluations.json") and not args.evaluate:
    eval_dataset = datasets.load_dataset('json', data_files=f"data/{args.dataset}/{args.jailbreak}_{args.cot_prompt}_{model_print_name}_{judge_model_print_name}_{args.evaluator}_evaluations.json", split='train')
else:
    print(f'evaluating {args.evaluator} for {args.judge_model}')
    print(registered_evaluators.keys())
    eval_dataset = evaluate_dataset(responses_dataset, [args.evaluator], models=[args.judge_model], )
    records = [dict(row) for row in eval_dataset]
    with open(f"data/{args.dataset}/{args.jailbreak}_{args.cot_prompt}_{model_print_name}_{judge_model_print_name}_{args.evaluator}_evaluations.json", 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=4)

eval_results = eval_dataset.to_pandas().groupby("jailbreak")[["score", "refusal"]].mean()
eval_results['non-refusal'] = 1 - eval_results['refusal']

print(eval_results)
eval_results.to_csv(f"data/{args.dataset}/{args.jailbreak}_{args.cot_prompt}_{model_print_name}_{judge_model_print_name}_{args.evaluator}_eval_results.csv")
