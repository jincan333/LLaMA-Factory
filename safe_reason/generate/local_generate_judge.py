import tempfile
import os
import json
import argparse
import json
import wandb
from datetime import datetime
import copy
import tiktoken
import re
import time
import sys
import concurrent.futures
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import random
import torch.distributed as dist

current_script_directory = os.path.dirname(os.path.abspath(__file__))
utils_directory = os.path.abspath(os.path.join(current_script_directory, '..', '..'))
sys.path.append(utils_directory)
from utils.llm_api import OpenaiClient, TogetherClient, ClaudeClient


def parse_args():
    # general
    parser = argparse.ArgumentParser(description='LLM_Alignment1')
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='LLM_Alignment')
    parser.add_argument('--save_path', type=str, default='data/')
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=6)
    parser.add_argument('--task', type=str, default='all', choices=['all', 'gen_y_before', 'jud_y_before','gen_y_rectify','jud_y_rectify', 'post_process'])
    parser.add_argument('--given_data', type=int, default=0)

    # model
    parser.add_argument('--model', type=str, default='llama-3-8b-instruct', choices=['gpt-4o', 'llama-3-8b', 'llama-3-8b-instruct', 'llama-3.1-8b', 'llama-3.1-8b-instruct', 'llama-3.1-70b-instruct-turbo', 'mistral-0.1-7b', 'mistral-0.2-7b-instruct'])
    parser.add_argument('--judge_model', type=str, default='llama-3-8b-instruct', choices=['gpt-4o', 'llama-3-8b', 'llama-3-8b-instruct', 'llama-3.1-8b', 'llama-3.1-8b-instruct', 'llama-3.1-70b-instruct-turbo', 'mistral-0.1-7b', 'mistral-0.2-7b-instruct'])
    parser.add_argument('--is_judge_model_api', type=int, default=0)
    parser.add_argument('--is_rectify_model_api', type=int, default=0)
    parser.add_argument('--n_y_before', type=int, default=1)
    parser.add_argument('--n_y_rectify', type=int, default=3)
    parser.add_argument('--n_judge', type=int, default=3)
    parser.add_argument('--y_before_threshold', type=int, default=1)
    parser.add_argument('--y_rectify_threshold', type=int, default=3)
    parser.add_argument('--total_input_tokens', type=int, default=0)
    parser.add_argument('--total_output_tokens', type=int, default=0)

    # dataset
    parser.add_argument('--dataset', type=str, default='gsm8k', choices=['gsm8k', 'MATH', 'AQuA', 'deepmind', 'mmlu', 'numglue', 'sat', 'simuleq', 'SVAMP'])
    parser.add_argument('--data_path', type=str, default='../data/math_eval_datasets/')
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=int, default=100)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--y_before_path', type=str, default='data/LLM_Alignment/y_before_with_score.json')
    parser.add_argument('--y_rectify_path', type=str, default='data/LLM_Alignment/y_rectify_without_score.json')
    # prompt
    parser.add_argument('--origin_prompt_file', type=str, default='prompt/origin_manual.json', help='origin prompt file location')
    parser.add_argument('--origin_prompt_index', type=int, default=2)
    parser.add_argument('--rectify_prompt_file', type=str, default='prompt/rectify_manual.json', help='rectify prompt file location')
    parser.add_argument('--rectify_prompt_index', type=int, default=18)
    parser.add_argument('--judge_prompt_file', type=str, default='prompt/judge_manual.json', help='judge prompt file location')
    parser.add_argument('--judge_prompt_index', type=int, default=8)
    parser.add_argument('--extract_prompt_file', type=str, default='prompt/extract_manual.json', help='extract prompt file location')
    parser.add_argument('--extract_prompt_index', type=int, default=3)
    parser.add_argument('--error_prompt_file', type=str, default='prompt/error_manual.json', help='find error prompt file location')
    parser.add_argument('--error_prompt_index', type=int, default=0)
    parser.add_argument('--is_find_error', type=int, default=1)

    args = parser.parse_args()

    return args


def num_tokens_from_messages(messages, model="gpt-4-turbo-2024-04-09"):
    """Returns the number of tokens used by a list of messages."""
    if 'gpt-4o' in model:
        tokens_per_message = 4
        tokens_per_name = -1
    else:
        tokens_per_message = 3
        tokens_per_name = 1

    try:
        encoding = tiktoken.get_encoding(model)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    if isinstance(messages, list):
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
    else:
        num_tokens += len(encoding.encode(messages))

    return num_tokens


def max_tokens(model):
    model = model.lower()
    if 'gpt-4' in model:
        return 128000
    elif any(x in model for x in ['gemma', 'llama-3']):
        return 8192
    elif any(x in model for x in ['qwen', 'mistral']):
        return 32768
    else:
        raise ValueError(f"Model {model} not supported")


def run_llm(messages, model_name="gpt-4o-2024-05-13", temperature=0.7, top_p=0.95, n=1, max_tokens=512, retry_attempts=3):
    for _ in range(retry_attempts):
        try:
            if 'gpt-4' in model_name.lower(): 
                agent = OpenaiClient()
                response = agent.chat(model=model_name, messages=messages, temperature=temperature, top_p=top_p, return_text=True, n=n, max_tokens=max_tokens)
            elif 'claude' in model_name.lower():
                agent = ClaudeClient()
                response = agent.chat(model=model_name, messages=messages, temperature=temperature, top_p=top_p, return_text=True, n=n, max_tokens=max_tokens)
            else:
                agent = TogetherClient()
                response = agent.chat(model=model_name, messages=messages, temperature=temperature, top_p=top_p, return_text=True, n=n, max_tokens=max_tokens)
            return response
        
        except Exception as e:
            print(e)
            time.sleep(10)
            response = ['error' for _ in range(n)]

    return response


def get_prompt(prompt_path, prompt_index):
    with open(prompt_path, 'r', encoding='utf-8') as file:
        all_prompt = json.load(file)
    return all_prompt[prompt_index]['prompt']


def get_chat_template(model_name):
    with open('prompt/chat_template.json', 'r', encoding='utf-8') as file:
        chat_templates = json.load(file)
    if 'llama-3' in model_name.lower():
        return chat_templates['llama3']
    elif 'mistral' in model_name.lower():
        if 'instruct' in model_name:
            return chat_templates['mistral-instruct']
        else:
            return chat_templates['mistral-base']
    else:
        return None
    

def get_model(model_name):
    if model_name in ('gpt-4o', 'chatgpt-4o-latest'):
        model_name = 'chatgpt-4o-latest'
    elif model_name in ('llama-3-8b', 'meta-llama/Meta-Llama-3-8B'):
        model_name = 'meta-llama/Meta-Llama-3-8B'
    elif model_name in ('llama-3-8b-instruct', 'meta-llama/Meta-Llama-3-8B-Instruct'):
        model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    elif model_name in ('llama-3.1-8b', 'meta-llama/Meta-Llama-3.1-8B'):
        model_name = 'meta-llama/Meta-Llama-3.1-8B'
    elif model_name in ('llama-3.1-8b-instruct', 'meta-llama/Meta-Llama-3.1-8B-Instruct'):
        model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    elif model_name in ('llama-3.1-70b-instruct-turbo', 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'):
        model_name = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
    elif model_name in ('mistral-0.1-7b', 'mistralai/Mistral-7B-v0.1'):
        model_name = 'mistralai/Mistral-7B-v0.1'
    elif model_name in ('mistral-0.2-7b-instruct', 'mistralai/Mistral-7B-Instruct-v0.2'):
        model_name = 'mistralai/Mistral-7B-Instruct-v0.2'

    return model_name


def get_data(args):
    if args.dataset == 'gsm8k':
        if args.task in ('gen_y_before', 'all'):
            data_path = '../data/gsm8k/train.jsonl'
        elif args.task in ('jud_y_before', 'gen_y_rectify'):
            assert 'y_before' in args.y_before_path
            data_path = args.y_before_path
            args.save_dir = os.path.dirname(args.y_before_path)
        elif args.task in ('jud_y_rectify', 'post_process'):
            assert 'y_rectify' in args.y_rectify_path
            data_path = args.y_rectify_path
            args.save_dir = os.path.dirname(args.y_rectify_path)
    elif args.dataset == 'MATH':
        data_path = 'hendrycks/competition_math'

    try:
        data_original = load_dataset('json', data_files=data_path, split="train", cache_dir='./cache')
    except:
        try:
            data_original = load_dataset(data_path, split="train", cache_dir='./cache')
        except:
            with open(data_path, 'r') as f:
                data_original = json.load(f) 

    if args.dataset == 'gsm8k' and args.task in ('gen_y_before', 'all'):
        data_original = [[{'question': data['query'], 'answer': data['response']}] for data in data_original]
    elif args.dataset == 'MATH' and args.task in ('gen_y_before', 'all'):
        data_original = [[{'question': data['query'], 'answer': data['response']}] for data in data_original]
    
    if args.frac_len > 0 and args.task in ('gen_y_before', 'all'):
        sub_len = args.frac_len 
        if sub_len*(args.data_frac+1) > len(data_original):
            data_frac = data_original[sub_len*args.data_frac:]
        else:
            data_frac = data_original[sub_len*args.data_frac:sub_len*(args.data_frac+1)]
    else:
        data_frac = data_original[:]

    extract_data_lambda = lambda example: {
            "question": example['question'].strip(), 
            "answer": example['answer'].strip(), 
            "input": example['input'].strip() if 'input' in example and example['input'] is not None else None,
            "y_before": example['y_before'].strip() if 'y_before' in example and example['y_before'] is not None else None,
            "y_before_score": example['y_before_score'] if 'y_before_score' in example and example['y_before_score'] is not None else None,
            "y_rectify": example['y_rectify'].strip() if 'y_rectify' in example and example['y_rectify'] else None,
            "y_rectify_score": example['y_rectify_score'] if 'y_rectify_score' in example and example['y_rectify_score'] is not None else None,
            "error": example['error'].strip() if 'error' in example and example['error'] is not None else None,
        }
    
    data = list(map(lambda x: list(map(extract_data_lambda, x)), data_frac))

    return data

    
def get_tokenizer(tokenizer):
    return lambda example: tokenizer.apply_chat_template(
           example,
           tokenize=False,
           add_generation_prompt=True,
        )


def clean_response(response):
    new_response = ''
    if isinstance(response, list):
        response = response[0]
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    response = [int(x) for x in new_response.split()]

    return response


def get_score(x, score_threshold):
    try:
        return int(x[0] >= score_threshold)
    except:
        return 0
    

def apply_chat_template(
    example, tokenizer, task
):
    messages = example["real"]
    example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    return example


def get_messages(prompt, tokenizer, tag='before', question=None, input=None, response=None, answer=None, judgment=None, error=None):
    if tag == 'before':
        content = prompt.format_map(dict(instruction=question, input=input))
    elif tag == 'judge':
        content = prompt.format_map(dict(instruction=question, input=input, response=response, answer=answer))
    elif tag == 'rectify':
        content = prompt.format_map(dict(instruction=question, input=input, response=response, answer=answer))
    elif tag == 'extract':
        content = prompt.format_map(dict(judgment=judgment))
    elif tag == 'error':
        content = prompt.format_map(dict(instruction=question, input=input, response=response, answer=answer))
    elif tag == 'error_rectify':
        content = prompt.format_map(dict(instruction=question, input=input, response=response, answer=answer, error=error))

    messages = [{'role': 'user', 'content': content}]

    return messages.map(apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "generation"},
        num_proc=8,
        remove_columns=None,
        desc="Applying chat template",
    )


def get_tokens(messages, model_name, question_total_token):
    instant_token = num_tokens_from_messages(messages, model_name)
    question_total_token += instant_token

    return instant_token, question_total_token


def remove_redundant(response):
    # Check and remove the first prefix
    if response.startswith("Here is the revised response:"):
        response = response[len("Here is the revised response:"):]
        response = response.strip()
    # Check and remove the second prefix
    if response.startswith("### Revised Response:"):
        response = response[len("### Revised Response:"):]
        response = response.strip()
    if response.startswith("<begin_of_response>"):
        response = response[len("<begin_of_response>"):]
        response = response.strip()
    if response.endswith("<end_of_response>"):
        response = response[:-len("<end_of_response>")]
        response = response.strip()
    
    return response

def main():
    start_time = datetime.now()
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    wandb.require("core")
    wandb.login(key=os.environ.get('WANDB_API_KEY') )
    wandb.init(project='LLM_Alignment_generate_data_'+args.model+'_'+args.dataset, entity=os.environ.get('WANDB_USER_NAME'), name=args.exp_name)

    # prepare model
    if 'gen' in args.task or args.task == 'all':
        args.model = get_model(args.model)
        args.max_tokens = max_tokens(args.model)
        ref_tokenizer = AutoTokenizer.from_pretrained(args.model)   
        ref_tokenizer.pad_token = ref_tokenizer.eos_token
        ref_tokenizer.model_max_length = args.max_tokens
        ref_chat_template = get_chat_template(args.model)
        if ref_chat_template is not None:
            ref_tokenizer.chat_template = ref_chat_template
        ref_model = LLM(model=args.model, gpu_memory_utilization=0.9, max_num_seqs=args.batch_size, max_model_len=args.max_tokens, tensor_parallel_size=args.world_size)
    
    if 'jud' in args.task or args.task == 'all':
        if args.is_judge_model_api:
            args.judge_model = get_model(args.judge_model)
        else:
            args.judge_model = get_model(args.judge_model)
            args.judge_max_tokens = max_tokens(args.judge_model)
            if args.judge_model == args.model:
                judge_tokenizer = ref_tokenizer
                judge_model = ref_model
            else:
                judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
                judge_tokenizer.pad_token = judge_tokenizer.eos_token
                judge_tokenizer.model_max_length = args.judge_max_tokens
                judge_chat_template = get_chat_template(args.judge_model)
                if judge_chat_template is not None:
                    judge_tokenizer.chat_template = judge_chat_template
                judge_model = LLM(model=args.judge_model, gpu_memory_utilization=0.9, max_num_seqs=args.batch_size, max_model_len=args.judge_max_tokens, swap_space=100, cpu_offload_gb=200, tensor_parallel_size=args.world_size)

    print(json.dumps(vars(args), indent=4))
    head, tail = os.path.split(args.exp_name)
    args.save_dir = os.path.join(args.save_path, head, tail)
    
    # prepare prompts
    data_info = []
    original_prompt = get_prompt(args.origin_prompt_file, args.origin_prompt_index)
    rectify_prompt = get_prompt(args.rectify_prompt_file, args.rectify_prompt_index)
    judge_prompt = get_prompt(args.judge_prompt_file, args.judge_prompt_index)
    extract_prompt = get_prompt(args.extract_prompt_file, args.extract_prompt_index)
    data_info.append(
        {
            'exp_name': args.exp_name,
            'gpu': args.gpu,
            'task': args.task,
            'model': args.model,
            'judge_model': args.judge_model,
            'is_judge_model_api': args.is_judge_model_api,
            'n_y_before': args.n_y_before,
            'n_y_rectify': args.n_y_rectify,
            'n_judge': args.n_judge,
            'dataset': args.dataset,
            'data_path': args.data_path,
            'data_frac': args.data_frac,
            'frac_len': args.frac_len,
            'batch_size': args.batch_size,
            'y_before_path': args.y_before_path,
            'y_rectify_path': args.y_rectify_path,
            'original_prompt': original_prompt,
            'original_prompt_index': args.origin_prompt_index,
            'rectify_prompt': rectify_prompt,
            'rectify_prompt_index': args.rectify_prompt_index,
            'judge_prompt': judge_prompt,
            'judge_prompt_index': args.judge_prompt_index,
            'extract_prompt': extract_prompt,
            'extract_prompt_index': args.extract_prompt_index,
            'y_before_threshold': args.y_before_threshold,
            'y_rectify_threshold': args.y_rectify_threshold,
        }
    )
    if args.given_data:
        data_frac = [[ {
            "question": "Rachel and Sara want to attend a beauty and modeling contest. They both want to buy new pairs of shoes and dresses. Sara buys a pair of shoes which costs $50 and a dress which costs $200. How much should Rachel budget if she wants to spend twice as much as what Sara spent on the pair of shoes and dress?",
            "answer": "The cost Rachel should budget for her pair of shoes is $50 * 2 = $<<50*2=100>>100.\nThe cost Rachel should budget for her dress is $200 * 2 = $<<200*2=400>>400.\nThe total Rachel should budget is $100 + $400 = $<<100+400=500>>500.\n#### 500",
            "input": None,
            "y_before": "Let's break it down step by step!\n\nSara spent $50 on shoes and $200 on a dress, so she spent a total of:\n\n$50 + $200 = $250\n\nRachel wants to spend twice as much as Sara, so she should budget:\n\n$250 x 2 = $500\n\nTherefore, Rachel should budget $500.",
            "y_before_score": 0,
            "y_rectify": "Here is the revised response:\n\nLet's break it down step by step!\n\nSara spent $50 on shoes and $200 on a dress, so she spent a total of:\n\n$50 + $200 = $250\n\nRachel wants to spend twice as much as Sara, so she should budget:\n\n$50 * 2 = $100 for the shoes\n$200 * 2 = $400 for the dress\n\nTherefore, Rachel should budget $100 + $400 = $500.",
            "y_rectify_score": 1,
            "error": None
        }]]
        extract_data_lambda = lambda example: {
            "question": example['question'].strip(), 
            "answer": example['answer'].strip(), 
            "input": example['input'].strip() if 'input' in example and example['input'] is not None else None,
            "y_before": example['y_before'].strip() if 'y_before' in example and example['y_before'] is not None else None,
            "y_before_score": example['y_before_score'] if 'y_before_score' in example and example['y_before_score'] is not None else None,
            "y_rectify": example['y_rectify'].strip() if 'y_rectify' in example and example['y_rectify'] else None,
            "y_rectify_score": example['y_rectify_score'] if 'y_rectify_score' in example and example['y_rectify_score'] is not None else None,
            "error": example['error'].strip() if 'error' in example and example['error'] is not None else None,
        }
    
        data_frac = list(map(lambda x: list(map(extract_data_lambda, x)), data_frac))
    else:
        data_frac = get_data(args)

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'data_info.json'), 'w') as json_file:
        json.dump(data_info, json_file, indent=4)
    print(f"data_info saved to {os.path.join(args.save_dir, 'data_info.json')}")
    # random.shuffle(data_frac)
    num_samples = min(3, len(data_frac))
    random_indices = random.sample(range(len(data_frac)), num_samples)
    # generate y_before
    if args.task == 'gen_y_before' or args.task == 'all':
        print('\n' + '*' * 50 + f' generate y_before ' + '*' * 50)
        y_before_input_data = data_frac
        print("y_before_input_data:")
        for i in random_indices:
            print(json.dumps(y_before_input_data[i], indent=4))
        y_before_messages = [[{'role': 'user', 'content': original_prompt.format_map(dict(instruction=piece['question'], input=piece['input']))} for piece in item] for item in y_before_input_data]
        print("y_before_messages:")
        for i in random_indices:
            print(json.dumps(y_before_messages[i], indent=4))
        
        ref_sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512, n=args.n_y_before)
        tokenizer = get_tokenizer(ref_tokenizer)
        y_before_messages_format = list(map(tokenizer, y_before_messages))
        total_y_before_messages_tokens = sum(map(lambda x: num_tokens_from_messages(x, args.model), y_before_messages_format))
        print("total_y_before_messages_tokens: ", total_y_before_messages_tokens)
        args.total_input_tokens += total_y_before_messages_tokens
        print("y_before_messages_format:")
        for i in random_indices:
            print(json.dumps(y_before_messages_format[i], indent=4))

        y_before_responses = [ 
            [response.text for response in responses.outputs]
            for responses in ref_model.generate(y_before_messages_format, ref_sampling_params)
        ]
        print("y_before_responses:")
        for i in random_indices:
            print(json.dumps(y_before_responses[i], indent=4))
        total_y_before_responses_tokens = sum(map(lambda x: sum(map(lambda y: num_tokens_from_messages(y, args.model), x)), y_before_responses))
        print("total_y_before_responses_tokens: ", total_y_before_responses_tokens)
        args.total_output_tokens += total_y_before_responses_tokens
        y_before_without_score_data = [
            [{**x[0], 'y_before': y_data} for y_data in y]
            for x, y in zip(y_before_input_data, y_before_responses)
        ]
        print("y_before_generated_data:")
        for i in random_indices:
            print(json.dumps(y_before_without_score_data[i], indent=4))
        y_before_path = os.path.join(args.save_dir, 'y_before_without_score.json')
        with open(y_before_path, 'w') as json_file:
            json.dump(y_before_without_score_data, json_file, indent=4)
        print('finish generating y_before')
        print(f"y_before saved to {y_before_path}")

    # judge y_before
    if args.task == 'jud_y_before' or args.task == 'all':
        print('\n' + '*' * 50 + f' judge y_before ' + '*' * 50)
        try:
            y_before_without_score_data
        except NameError:
            y_before_without_score_data = get_data(args)
        print("y_before_without_score_data:")
        for i in random_indices:
            print(json.dumps(y_before_without_score_data[i], indent=4))

        y_before_judge_messages = [[[{'role': 'user', 'content': judge_prompt.format_map(dict(instruction=piece['question'], input=piece['input'], response=piece['y_before'], answer=piece['answer']))}] for piece in item] for item in y_before_without_score_data]
        print("y_before_judge_messages:")
        for i in random_indices:
            print(json.dumps(y_before_judge_messages[i], indent=4))
        
        if args.is_judge_model_api:
            y_before_judge_messages_format = y_before_judge_messages
        else:
            tokenizer = get_tokenizer(judge_tokenizer)
            y_before_judge_messages_format = list(map(lambda x: list(map(tokenizer, x)), y_before_judge_messages))
        total_y_before_judge_messages_tokens = sum(map(lambda x: sum(map(lambda y: num_tokens_from_messages(y, args.judge_model), x)), y_before_judge_messages_format))
        print("total_y_before_judge_messages_tokens: ", total_y_before_judge_messages_tokens)
        args.total_input_tokens += total_y_before_judge_messages_tokens
        print("y_before_judge_messages_format:")
        for i in random_indices:
            print(json.dumps(y_before_judge_messages_format[i], indent=4))
        
        sub_len = len(y_before_judge_messages[0])
        y_before_judge_messages_format = [item for sublist in y_before_judge_messages_format for item in sublist]
        if args.is_judge_model_api:
            # use async and asyncio to speed up https://github.com/openai/openai-python
            y_before_judge_responses = []
            for message in y_before_judge_messages_format:
                response = run_llm(message, model_name=args.judge_model, temperature=0.7, top_p=0.95, n=args.n_judge, max_tokens=256)
                y_before_judge_responses.append(response)
            y_before_judge_responses_tokens = sum(
                num_tokens_from_messages(response, args.judge_model)
                for responses in y_before_judge_responses
                for response in responses
            )
            print("y_before_judge_responses_tokens: ", y_before_judge_responses_tokens)
            args.total_output_tokens += y_before_judge_responses_tokens
            print("y_before_judge_responses:")
            for i in random_indices:
                for j in range(sub_len):
                    print(json.dumps(y_before_judge_responses[sub_len*i+j], indent=4))
            
            # extract score
            y_before_judge_responses_concat = [
                ''.join(f"judgment{i}: {item}\n\n" for i, item in enumerate(response))
                for response in y_before_judge_responses
            ]
            y_before_extract_messages = [[{'role': 'user', 'content': extract_prompt.format_map(dict(judgment=response))}] for response in y_before_judge_responses_concat]
            y_before_extract_messages_tokens = sum(map(lambda x: num_tokens_from_messages(x, args.judge_model), y_before_extract_messages))
            print("y_before_extract_messages_tokens: ", y_before_extract_messages_tokens)
            print("y_before_extract_messages:")
            for i in random_indices:
                for j in range(sub_len):
                    print(json.dumps(y_before_extract_messages[sub_len*i+j], indent=4))

            args.total_input_tokens += y_before_extract_messages_tokens
            y_before_extract_responses = []
            for message in y_before_extract_messages:
                response = run_llm(message, model_name=args.judge_model, temperature=0.7, top_p=0.95, n=1, max_tokens=256)
                y_before_extract_responses.append(response)

        else:
            judge_sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256, n=args.n_judge)
            extract_sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256, n=1)
            y_before_judge_responses = [ 
                [response.text for response in responses.outputs]
                for responses in judge_model.generate(y_before_judge_messages_format, judge_sampling_params)
            ]
            y_before_judge_responses_tokens = sum(map(lambda x: sum(map(lambda y: num_tokens_from_messages(y, args.judge_model), x)), y_before_judge_responses))
            print("y_before_judge_responses_tokens: ", y_before_judge_responses_tokens)
            args.total_output_tokens += y_before_judge_responses_tokens
            print("y_before_judge_responses:")
            for i in random_indices:
                for j in range(sub_len):
                    print(json.dumps(y_before_judge_responses[sub_len*i+j], indent=4))
            
            # extract score
            y_before_judge_responses_concat = [
                ''.join(f"judgment{i}: {item}\n\n" for i, item in enumerate(response))
                for response in y_before_judge_responses
            ]
            y_before_extract_messages = [[[{'role': 'user', 'content': extract_prompt.format_map(dict(judgment=response))}]] for response in y_before_judge_responses_concat]
            print("y_before_extract_messages:")
            for i in random_indices:
                for j in range(sub_len):
                    print(json.dumps(y_before_extract_messages[sub_len*i+j], indent=4))
            y_before_extract_messages_format = list(map(lambda x: list(map(tokenizer, x)), y_before_extract_messages))
            y_before_extract_messages_tokens = sum(map(lambda x: sum(map(lambda y: num_tokens_from_messages(y, args.judge_model), x)), y_before_extract_messages_format))
            print("y_before_extract_messages_tokens: ", y_before_extract_messages_tokens)
            args.total_input_tokens += y_before_extract_messages_tokens
            print("y_before_extract_messages_format:")
            for i in random_indices:
                for j in range(sub_len):
                    print(json.dumps(y_before_extract_messages_format[sub_len*i+j], indent=4))

            y_before_extract_messages_format = [item for sublist in y_before_extract_messages_format for item in sublist]
            y_before_extract_responses = [ 
                [response.text for response in responses.outputs]
                for responses in judge_model.generate(y_before_extract_messages_format, extract_sampling_params)
            ]
        
        y_before_extract_responses_tokens = sum(
            num_tokens_from_messages(response, args.judge_model)
            for responses in y_before_extract_responses
            for response in responses
        )
        print("y_before_extract_responses_tokens: ", y_before_extract_responses_tokens)
        args.total_output_tokens += y_before_extract_responses_tokens
        print("y_before_extract_responses:")
        for i in random_indices:
            for j in range(args.n_y_before):
                print(json.dumps(y_before_extract_responses[args.n_y_before*i+j], indent=4))
        y_before_extract_responses_clean = list(map(clean_response, y_before_extract_responses))
        y_before_scores = list(map(lambda x: get_score(x, args.y_before_threshold), y_before_extract_responses_clean))
        y_before_scores_concat = [
            y_before_scores[i:i + args.n_y_before]
            for i in range(0, len(y_before_scores), args.n_y_before)
        ]
        print("y_before_scores_concat:")
        for i in random_indices:
            print(json.dumps(y_before_scores_concat[i], indent=4))
        y_before_with_score_data = [
            [{**x, 'y_before_score': y} for x, y in zip(x_item, y_item)]
            for x_item, y_item in zip(y_before_without_score_data, y_before_scores_concat)
        ]
        print("y_before_with_score_data:")
        for i in random_indices:
            print(json.dumps(y_before_with_score_data[i], indent=4))
        
        y_before_with_score_correct = []
        y_before_with_score_incorrect = []
        for item in y_before_with_score_data:
            if max([piece['y_before_score'] for piece in item]) >= 1:
                y_before_with_score_correct.append(item)
            else:
                y_before_with_score_incorrect.append(item)
        y_before_with_score_data = y_before_with_score_incorrect

        y_before_scores_path = os.path.join(args.save_dir, 'y_before_with_score.json')
        with open(y_before_scores_path, 'w') as json_file:
            json.dump(y_before_with_score_data, json_file, indent=4)
        print(f"y_before_scores saved to {y_before_scores_path}")
        y_before_with_score_correct_path = os.path.join(args.save_dir, 'y_before_with_score_correct.json')
        with open(y_before_with_score_correct_path, 'w') as json_file:
            json.dump(y_before_with_score_correct, json_file, indent=4)
        print(f"y_before_scores_correct saved to {y_before_with_score_correct_path}")

    # generate y_rectify
    if args.given_data:
        y_before_with_score_data = data_frac
    if args.task == 'gen_y_rectify' or args.task == 'all':
        print('\n' + '*' * 50 + f' generate y_rectify ' + '*' * 50)
        try:
            y_before_with_score_data
        except NameError:
            y_before_with_score_data = get_data(args)
        num_samples = min(3, len(y_before_with_score_data))
        random_indices = random.sample(range(len(y_before_with_score_data)), num_samples)
        print("y_before_with_score_data:")
        if len(y_before_with_score_data) == 0:
            print("No data to generate y_rectify")
            exit()

        for i in random_indices:
            print(json.dumps(y_before_with_score_data[i], indent=4))
        
        y_rectify_messages = [[[{'role': 'user', 'content': rectify_prompt.format_map(dict(instruction=piece['question'], input=piece['input'], response=piece['y_before'], answer=piece['answer']))}] for piece in item] for item in y_before_with_score_data]
        print("y_rectify_messages:")
        for i in random_indices:
            print(json.dumps(y_rectify_messages[i], indent=4))
        
        if args.is_rectify_model_api:
            y_rectify_messages_format = y_rectify_messages
        else:
            tokenizer = get_tokenizer(ref_tokenizer)
            y_rectify_messages_format = list(map(lambda x: list(map(tokenizer, x)), y_rectify_messages))

        total_y_rectify_messages_tokens = sum(map(lambda x: sum(map(lambda y: num_tokens_from_messages(y, args.judge_model), x)), y_rectify_messages_format))
        print("total_y_rectify_messages_tokens: ", total_y_rectify_messages_tokens)
        args.total_input_tokens += total_y_rectify_messages_tokens
        print("y_rectify_messages_format:")
        for i in random_indices:
            print(json.dumps(y_rectify_messages_format[i], indent=4))

        sub_len = len(y_rectify_messages[0])
        y_rectify_messages_format = [item for sublist in y_rectify_messages_format for item in sublist]
        if args.is_rectify_model_api:
            y_rectify_responses = []
            for message in y_rectify_messages_format:
                response = run_llm(message, model_name=args.judge_model, temperature=0.7, top_p=0.95, n=args.n_y_rectify, max_tokens=512)
                y_rectify_responses.append(response)
            total_y_rectify_responses_tokens = sum(
                num_tokens_from_messages(response, args.judge_model)
                for responses in y_rectify_responses
                for response in responses
            )
            print("y_rectify_responses_tokens: ", total_y_rectify_responses_tokens)
            args.total_output_tokens += total_y_rectify_responses_tokens
            print("y_rectify_responses:")
            for i in random_indices:
                for j in range(sub_len):
                    print(json.dumps(y_rectify_responses[sub_len*i+j], indent=4))
        else:
            ref_sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512, n=args.n_y_rectify)
            y_rectify_responses = [
                [response.text for response in responses.outputs]
                for responses in ref_model.generate(y_rectify_messages_format, ref_sampling_params)
            ]
            total_y_rectify_responses_tokens = sum(
                num_tokens_from_messages(response, args.model)
                for responses in y_rectify_responses
                for response in responses
            )
            print("total_y_rectify_responses_tokens: ", total_y_rectify_responses_tokens)
            args.total_output_tokens += total_y_rectify_responses_tokens
            print("y_rectify_responses:")
            for i in random_indices:
                for j in range(sub_len):
                    print(json.dumps(y_rectify_responses[sub_len*i+j], indent=4))

        # remove redundant prefixes
        y_rectify_responses = [[remove_redundant(response) for response in item] for item in y_rectify_responses]

        y_rectify_responses_concat = [
            y_rectify_responses[i:i + sub_len]
            for i in range(0, len(y_rectify_responses), sub_len)
        ]
        print("y_rectify_responses_concat:")
        for i in random_indices:
            # print(y_rectify_responses_concat[i])
            print(json.dumps(y_rectify_responses_concat[i], indent=4))
        
        y_rectify_without_score_data = [
            [{**x, 'y_rectify': y} for x, y_list in zip(x_item, y_item) for y in y_list]
            for x_item, y_item in zip(y_before_with_score_data, y_rectify_responses_concat)
        ]
        print("y_rectify_without_score_data:")
        for i in random_indices:
            # print(y_rectify_without_score_data[i])
            print(json.dumps(y_rectify_without_score_data[i], indent=4))
        y_rectify_path = os.path.join(args.save_dir, 'y_rectify_without_score.json')
        with open(y_rectify_path, 'w') as json_file:
            json.dump(y_rectify_without_score_data, json_file, indent=4)
        print(f"y_rectify saved to {y_rectify_path}")

    # judge y_rectify
    if args.task == 'jud_y_rectify' or args.task == 'all':
        print('\n' + '*' * 50 + f' judge y_rectify ' + '*' * 50)
        try:
            y_rectify_without_score_data
        except NameError:
            y_rectify_without_score_data = get_data(args)
        print("y_rectify_without_score_data:")
        for i in random_indices:
            # print(y_rectify_without_score_data[i])
            print(json.dumps(y_rectify_without_score_data[i], indent=4))

        y_rectify_judge_messages = [[[{'role': 'user', 'content': judge_prompt.format_map(dict(instruction=piece['question'], input=piece['input'], response=piece['y_rectify'], answer=piece['answer']))}] for piece in item] for item in y_rectify_without_score_data]
        print("y_rectify_judge_messages:")
        for i in random_indices:
            # print(y_rectify_judge_messages[i])
            print(json.dumps(y_rectify_judge_messages[i], indent=4))

        if args.is_judge_model_api:
            y_rectify_judge_messages_format = y_rectify_judge_messages
        else:
            tokenizer = get_tokenizer(judge_tokenizer)
            y_rectify_judge_messages_format = list(map(lambda x: list(map(tokenizer, x)), y_rectify_judge_messages))
        total_y_rectify_judge_messages_tokens = sum(map(lambda x: sum(map(lambda y: num_tokens_from_messages(y, args.judge_model), x)), y_rectify_judge_messages_format))
        print("total_y_rectify_judge_messages_tokens: ", total_y_rectify_judge_messages_tokens)
        args.total_input_tokens += total_y_rectify_judge_messages_tokens
        print("y_rectify_judge_messages_format:")
        for i in random_indices:
            # print(y_rectify_judge_messages_format[i])
            print(json.dumps(y_rectify_judge_messages_format[i], indent=4))

        sub_len = len(y_rectify_judge_messages[0])
        y_rectify_judge_messages_format = [item for sublist in y_rectify_judge_messages_format for item in sublist]
        if args.is_judge_model_api:
            y_rectify_judge_responses = []
            for message in y_rectify_judge_messages_format:
                response = run_llm(message, model_name=args.judge_model, temperature=0.7, top_p=0.95, n=args.n_judge, max_tokens=256)
                y_rectify_judge_responses.append(response)
            y_rectify_judge_responses_tokens = sum(
                num_tokens_from_messages(response, args.judge_model)
                for responses in y_rectify_judge_responses
                for response in responses
            )
            print("y_rectify_judge_responses_tokens: ", y_rectify_judge_responses_tokens)
            args.total_output_tokens += y_rectify_judge_responses_tokens
            print("y_rectify_judge_responses:")
            for i in random_indices:
                for j in range(sub_len):
                    # print(y_rectify_judge_responses[sub_len*i+j])
                    print(json.dumps(y_rectify_judge_responses[sub_len*i+j], indent=4))

            # extract score
            y_rectify_judge_responses_concat = [
                ''.join(f"judgment{i}: {item}\n\n" for i, item in enumerate(response))
                for response in y_rectify_judge_responses
            ]
            y_rectify_extract_messages = [[{'role': 'user', 'content': extract_prompt.format_map(dict(judgment=response))}] for response in y_rectify_judge_responses_concat]
            y_rectify_extract_messages_tokens = sum(map(lambda x: num_tokens_from_messages(x, args.judge_model), y_rectify_extract_messages))
            print("y_rectify_extract_messages_tokens: ", y_rectify_extract_messages_tokens)
            print("y_rectify_extract_messages:")
            for i in random_indices:
                for j in range(sub_len):
                    # print(y_rectify_extract_messages[sub_len*i+j])
                    print(json.dumps(y_rectify_extract_messages[sub_len*i+j], indent=4))

            args.total_input_tokens += y_rectify_extract_messages_tokens
            y_rectify_extract_responses = []
            for message in y_rectify_extract_messages:
                response = run_llm(message, model_name=args.judge_model, temperature=0.7, top_p=0.95, n=1, max_tokens=256)
                y_rectify_extract_responses.append(response)

        else:
            judge_sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256, n=args.n_judge)
            extract_sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=256, n=1)
            y_rectify_judge_responses = [ 
                [response.text for response in responses.outputs]
                for responses in judge_model.generate(y_rectify_judge_messages_format, judge_sampling_params)
            ]
            y_rectify_judge_responses_tokens = sum(map(lambda x: sum(map(lambda y: num_tokens_from_messages(y, args.judge_model), x)), y_rectify_judge_responses))
            print("y_rectify_judge_responses_tokens: ", y_rectify_judge_responses_tokens)
            args.total_output_tokens += y_rectify_judge_responses_tokens
            print("y_rectify_judge_responses:")
            for i in random_indices:
                for j in range(sub_len):
                    # print(y_rectify_judge_responses[sub_len*i+j])
                    print(json.dumps(y_rectify_judge_responses[sub_len*i+j], indent=4))
            
            # extract score
            y_rectify_judge_responses_concat = [
                ''.join(f"judgment{i}: {item}\n\n" for i, item in enumerate(response))
                for response in y_rectify_judge_responses
            ]
            y_rectify_extract_messages = [[[{'role': 'user', 'content': extract_prompt.format_map(dict(judgment=response))}]] for response in y_rectify_judge_responses_concat]
            print("y_rectify_extract_messages:")
            for i in random_indices:
                for j in range(sub_len):
                    # print(y_rectify_extract_messages[sub_len*i+j])
                    print(json.dumps(y_rectify_extract_messages[sub_len*i+j], indent=4))
            y_rectify_extract_messages_format = list(map(lambda x: list(map(tokenizer, x)), y_rectify_extract_messages))
            y_rectify_extract_messages_tokens = sum(map(lambda x: sum(map(lambda y: num_tokens_from_messages(y, args.judge_model), x)), y_rectify_extract_messages_format))
            print("y_rectify_extract_messages_tokens: ", y_rectify_extract_messages_tokens)
            args.total_input_tokens += y_rectify_extract_messages_tokens
            print("y_rectify_extract_messages_format:")
            for i in random_indices:
                for j in range(sub_len):
                    # print(y_rectify_extract_messages_format[sub_len*i+j])
                    print(json.dumps(y_rectify_extract_messages_format[sub_len*i+j], indent=4))

            y_rectify_extract_messages_format = [item for sublist in y_rectify_extract_messages_format for item in sublist]
            y_rectify_extract_responses = [ 
                [response.text for response in responses.outputs]
                for responses in judge_model.generate(y_rectify_extract_messages_format, extract_sampling_params)
            ]
        
        y_rectify_extract_responses_tokens = sum(
            num_tokens_from_messages(response, args.judge_model)
            for responses in y_rectify_extract_responses
            for response in responses
        )
        print("y_rectify_extract_responses_tokens: ", y_rectify_extract_responses_tokens)
        args.total_output_tokens += y_rectify_extract_responses_tokens
        print("y_rectify_extract_responses:")
        for i in random_indices:
            for j in range(sub_len):
                # print(y_rectify_extract_responses[sub_len*i+j])
                print(json.dumps(y_rectify_extract_responses[sub_len*i+j], indent=4))
        y_rectify_extract_responses_clean = list(map(clean_response, y_rectify_extract_responses))
        y_rectify_scores = list(map(lambda x: get_score(x, args.y_rectify_threshold), y_rectify_extract_responses_clean))
        y_rectify_scores_concat = [
            y_rectify_scores[i:i + sub_len]
            for i in range(0, len(y_rectify_scores), sub_len)
        ]
        print("y_rectify_scores_concat:")
        for i in random_indices:
            # print(y_rectify_scores_concat[i])
            print(json.dumps(y_rectify_scores_concat[i], indent=4))
        
        y_rectify_with_score_data = [
            [{**x, 'y_rectify_score': y} for x, y in zip(x_item, y_item)]
            for x_item, y_item in zip(y_rectify_without_score_data, y_rectify_scores_concat)
        ]
        print("y_rectify_with_score_data:")
        for i in random_indices:
            # print(y_rectify_with_score_data[i])
            print(json.dumps(y_rectify_with_score_data[i], indent=4))
        y_rectify_scores_path = os.path.join(args.save_dir, 'y_rectify_with_score.json')
        with open(y_rectify_scores_path, 'w') as json_file:
            json.dump(y_rectify_with_score_data, json_file, indent=4)
        print(f"y_rectify_scores saved to {y_rectify_scores_path}")

    if args.task == 'post_process' or args.task == 'all':
        print('\n' + '*' * 50 + f' post process ' + '*' * 50)
        try:
            y_rectify_with_score_data
        except NameError:
            y_rectify_with_score_data = get_data(args)
        final_data = y_rectify_with_score_data
        # summarize the results
        sub_len = len(final_data[0])
        total_scores = sub_len * len(final_data)
        incorrect_y_before_num = len(final_data)
        win_scores = sum([int(piece['y_rectify_score'] >= piece['y_before_score'] and piece['y_rectify_score']) if 'y_rectify_score' in piece and 'y_before_score' in piece and piece['y_rectify_score'] is not None and piece ['y_before_score'] is not None else 0 for item in final_data for piece in item])
        win_y_rectify_num = sum(max([piece['y_rectify_score'] if 'y_rectify_score' in piece and piece['y_rectify_score'] is not None else 0 for piece in item]) for item in final_data)
        rectify_total_scores = sum([piece['y_rectify_score'] if 'y_rectify_score' in piece and piece['y_rectify_score'] is not None else 0 for item in final_data for piece in item])
        before_total_scores = sum([piece['y_before_score'] if 'y_before_score' in piece and piece['y_before_score'] is not None else 0 for item in final_data for piece in item])

        final_rectify_data = []
        for item in final_data:
            tag = 0
            for piece in item:
                if piece['y_rectify_score'] is not None and piece['y_rectify_score'] ==1:
                    final_rectify_data.insert(0, [piece])
                    tag = 1
                    break
            if tag == 0:
                item[0]['y_rectify'] = item[0]['answer']
                item[0]['y_rectify_score'] = 1
                final_rectify_data.append([item[0]])

        try:
            y_before_with_score_correct
            y_before_with_score_correct_data = y_before_with_score_correct
        except NameError:
            if os.path.exists(os.path.join(args.save_dir, 'y_before_with_score_correct.json')):
                with open(os.path.join(args.save_dir, 'y_before_with_score_correct.json'), 'r') as json_file:
                    y_before_with_score_correct_data = json.load(json_file)
            else:
                y_before_with_score_correct_data = []

        for item in y_before_with_score_correct_data:
            for piece in item:
                piece['y_rectify'] = piece['y_before']
                piece['y_rectify_score'] = piece['y_before_score']
                # piece['y_rectify'] = piece['answer']
                # piece['y_rectify_score'] = 1
        final_data = final_rectify_data + y_before_with_score_correct_data
        with open(os.path.join(args.save_dir, 'final_data.json'), 'w') as json_file:
            json.dump(final_data, json_file, indent=4)
        print(f"final data saved to {os.path.join(args.save_dir, 'final_data.json')}")

        end_time = datetime.now()
        print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        elapsed_time = end_time - start_time
        total_seconds = elapsed_time.total_seconds()
        # Calculate hours, minutes, and seconds
        hours = total_seconds / 3600
        minutes = total_seconds / 60
        # Print the results
        print('Total Statistics:\n',
            f'Total data num: {len(final_data)}\n',
            f'incorrect y_before num: {incorrect_y_before_num}\n',
            f'win y_rectify num: {win_y_rectify_num}\n',
            f'Total scores: {total_scores}\n',
            f'win scores: {win_scores}\n',
            f'rectify total scores: {rectify_total_scores}\n',
            f'before total scores: {before_total_scores}\n',
            f'Total Tokens: {args.total_input_tokens + args.total_output_tokens}\n',
            f'Total input tokens: {args.total_input_tokens}\n',
            f'Total output tokens: {args.total_output_tokens}\n',
            f"Elapsed time: {elapsed_time}\n",
            f"Elapsed time in seconds: {total_seconds:.2f} seconds\n",
            f"Elapsed time in minutes: {minutes:.2f} minutes\n",
            f"Elapsed time in hours: {hours:.2f} hours\n",
        )

        data_info.append(
            {
                'total_data_num': len(final_data),
                'incorrect_y_before_num': incorrect_y_before_num,
                'win_y_rectify_num': win_y_rectify_num,
                'total_scores': total_scores,
                'win_scores': win_scores,
                'rectify_total_scores': rectify_total_scores,
                'before_total_scores': before_total_scores,
                'total_token': args.total_input_tokens + args.total_output_tokens,
                'total_input_tokens': args.total_input_tokens,
                'total_output_tokens': args.total_output_tokens,
                'elapsed hours': hours,
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            }
        )

        with open(os.path.join(args.save_dir, 'data_info.json'), 'w') as json_file:
            json.dump(data_info, json_file, indent=4)
        print(f"data info saved to {os.path.join(args.save_dir, 'data_info.json')}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()