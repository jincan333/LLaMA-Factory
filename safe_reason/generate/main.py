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

current_script_directory = os.path.dirname(os.path.abspath(__file__))
utils_directory = os.path.abspath(os.path.join(current_script_directory, '..'))
sys.path.append(utils_directory)
from utils.llm_api import OpenaiClient, TogetherClient, ClaudeClient


def parse_args():
    # general
    parser = argparse.ArgumentParser(description='LLM_Alignment')
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='LLM_Alignment')
    parser.add_argument('--save_path', type=str, default='chat_history/')
    parser.add_argument('--gpu', type=int, default=0)

    # model
    parser.add_argument('--model', type=str, default='llama-3-70b', choices=['gpt-4o', 'gpt-4o-mini', 'llama-3-8b', 'llama-3-70b', 'llama-2-7b', 'llama-2-13b', 'llama-2-70b', 'qwen-2-72b', 'llama-3.1-8b', 'llama-3.1-70b'])
    parser.add_argument('--judge_model', type=str, default='gpt-4o', choices=['gpt-4o', 'gpt-4o-mini', 'llama-3-8b', 'llama-3-70b', 'llama-2-7b', 'llama-2-13b', 'llama-2-70b', 'qwen-2-72b', 'llama-3.1-8b', 'llama-3.1-70b'])

    # dataset
    parser.add_argument('--dataset', type=str, default='gsm8k', choices=['gsm8k', 'MATH', 'AQuA', 'deepmind', 'mmlu', 'numglue', 'sat', 'simuleq', 'SVAMP'])
    parser.add_argument('--data_path', type=str, default='../data/math_eval_datasets/')
    parser.add_argument('--data_num', type=int, default=20)
    parser.add_argument('--data_index', type=int, default=-1)

    # prompt
    parser.add_argument('--origin_prompt_file', type=str, default='prompt/origin_manual.json', help='origin prompt file location')
    parser.add_argument('--origin_prompt_index_list', type=str, default='0')
    parser.add_argument('--rectify_prompt_file', type=str, default='prompt/rectify_manual.json', help='rectify prompt file location')
    parser.add_argument('--rectify_prompt_index_list', type=str, default='0')
    parser.add_argument('--judge_prompt_file', type=str, default='prompt/judge_manual.json', help='judge prompt file location')
    parser.add_argument('--judge_prompt_index_list', type=str, default='0')
    parser.add_argument('--extract_prompt_file', type=str, default='prompt/extract_manual.json', help='extract prompt file location')
    parser.add_argument('--extract_prompt_index_list', type=str, default='0')
    parser.add_argument('--is_answer', type=int, default=0)

    # count
    parser.add_argument('--chat_cnt', type=int, default=0)
    parser.add_argument('--instant_token', type=int, default=0)
    parser.add_argument('--total_token', type=int, default=0)

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
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def max_tokens(model):
    model = model.lower()
    if 'gpt-4' in model:
        return 128000
    elif any(x in model for x in ['gemma', 'llama-3']):
        return 8192
    elif any(x in model for x in ['qwen', 'mixtral']):
        return 32768
    else:
        raise ValueError(f"Model {model} not supported")


def run_llm(messages, model_name="gpt-4o-2024-05-13", temperature=0, batchsize=1, retry_attempts=3):
    response_list = []
    for attempt in range(retry_attempts):
        try:
            if len(response_list) > 0:
                response_list = []
            if 'gpt-4' in model_name.lower(): 
                agent = OpenaiClient()
                response = agent.chat(model=model_name, messages=messages, temperature=temperature, return_text=True, n=batchsize)
            elif 'claude' in model_name.lower():
                agent = ClaudeClient()
                response = agent.chat(model=model_name, messages=messages, temperature=temperature, return_text=True, n=batchsize)
            else:
                agent = TogetherClient()
                response = agent.chat(model=model_name, messages=messages, temperature=temperature, return_text=True, n=batchsize)
            return response
        
        except Exception as e:
            print(e)
            time.sleep(10 * attempt + 3)
            for _ in range(batchsize):
                response_list.append('error')

    return response_list


def get_prompt(prompt_path, prompt_index_list):
    with open(prompt_path, 'r', encoding='utf-8') as file:
        all_prompt = json.load(file)
    prompt_list = []
    prompt_index_list = [int(item) for item in prompt_index_list.split(',')]
    for i in prompt_index_list:
        prompt_list.append(all_prompt[i])

    return prompt_list
    

def get_model(model_name):
    if model_name == 'gpt-4o':
        model_name = 'gpt-4o-2024-05-13'
    elif model_name == 'gpt-4':
        model_name = 'gpt-4-turbo-2024-04-09'
    elif model_name == 'gpt-4o-mini':
        model_name = 'gpt-4o-mini-2024-07-18'
    elif model_name == 'llama-2-7b':
        model_name = 'meta-llama/Llama-2-7b-chat-hf'
    elif model_name == 'llama-2-13b':
        model_name = 'meta-llama/Llama-2-13b-chat-hf'
    elif model_name == 'llama-2-70b':
        model_name = 'meta-llama/Llama-2-70b-chat-hf'
    elif model_name == 'llama-3-8b':
        model_name = 'meta-llama/Llama-3-8B-chat-hf'
    elif model_name == 'llama-3-70b':
        model_name = 'meta-llama/Llama-3-70B-chat-hf'
    elif model_name == 'llama-3.1-8b':
        model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
    elif model_name == 'llama-3.1-70b':
        model_name = 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo'
    elif model_name == 'qwen-2-72b':
        model_name = 'Qwen/Qwen2-72B-Instruct'

    return model_name


def clean_response(response):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    response = [int(x) for x in new_response.split()]

    return response


def main():
    args = parse_args()
    wandb_username=os.environ.get('WANDB_USER_NAME')
    wandb_key=os.environ.get('WANDB_API_KEY')    
    wandb.login(key=wandb_key)
    wandb.init(project='LLM_Alignment_'+args.model+'_'+args.dataset, entity=wandb_username, name=args.exp_name)

    # prepare model
    args.model = get_model(args.model)
    args.judge_model = get_model(args.judge_model)

    print(json.dumps(vars(args), indent=4))
    start_time = datetime.now()
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # load data
    try:
        data_path = os.path.join(args.data_path, args.dataset, args.dataset+'.json')
        data_all = []
        with open(data_path, 'r') as f:
            data_all = json.load(f) 
    except:
        try:
            data_path = os.path.join(args.data_path, args.dataset, args.dataset+'.jsonl')
            with open(data_path, 'r') as f:
                for line in f:
                    data_all.append(json.loads(line))
        except:
            print('dataset not found')
            return

    # chat
    chat_history = []
    for i in range(args.data_num): 
        if args.data_index >= 0:
            i = args.data_index
        examples = data_all[i]
        # gsm8k: question + answer
        # MATH: problem + solution
        # mmlu: question + choice + answer
        if args.dataset == 'gsm8k':
            key1 = 'question'
            key2 = 'answer'
        elif args.dataset == 'MATH':
            key1 = 'question'
            key2 = 'solution'
        elif args.dataset == 'mmlu':
            key1 = 'question'
            key2 = 'choice'
            key3 = 'answer'
        question = examples[key1].strip()
        answer = examples[key2].strip()
        print('current question: ', question)
        print('current answer: ', answer)

        # obtain y_before
        ## original prompt
        original_prompt_list = get_prompt(args.origin_prompt_file, args.origin_prompt_index_list)
        original_prompt = original_prompt_list[0]
        if 'input' in examples:
            original_cotents = [
                original_prompt['prompt_input'].format_map(dict(instruction=instruction, input=input)) if input != "" \
                else original_prompt['prompt_no_input'].format_map(dict(instruction=instruction)) \
                for instruction, input in zip(question, examples['input']) 
            ]
        else:
            # original_contents = [
            #     original_prompt['prompt_no_input'].format_map(dict(instruction=instruction)) \
            #     for instruction in examples[key1]
            # ]
            original_cotents = [original_prompt['prompt_no_input'].format_map(dict(instruction=question))]
        # targets = [output for output in examples[key2]]
        targets = [answer]

        for j in range(len(original_cotents)):
            messages = [{'role': 'user', 'content': original_cotents[j]}]
            # messages += "Let\'s think step by step."
            print('original messages:\n', messages)
            args.instant_token = num_tokens_from_messages(messages, args.model)
            print('instant token', args.instant_token)
            args.total_token += args.instant_token
            args.chat_cnt += 1
            y_before = run_llm(messages, model_name=args.model, temperature=0)
            y_before = re.sub(r'response', 'answer', y_before, flags=re.IGNORECASE)
            print('y_before:\n', y_before)

        # obtain y_rectify
        ## rectify prompt
        rectify_prompt_list = get_prompt(args.rectify_prompt_file, args.rectify_prompt_index_list)
        rectify_prompt = rectify_prompt_list[0]
        if 'input' in examples:
            if args.is_answer:
                rectify_contents = [
                    rectify_prompt['prompt_input'].format_map(dict(instruction=instruction, input=input, response=y_before, answer=answer)) if input != "" \
                    else rectify_prompt['prompt_no_input'].format_map(dict(instruction=instruction, response=y_before, answer=answer)) \
                    for instruction, input in zip(question, examples['input'])
                ]
            else:
                rectify_contents = [
                    rectify_prompt['prompt_input'].format_map(dict(instruction=instruction, input=input, response=y_before)) if input != "" \
                    else rectify_prompt['prompt_no_input'].format_map(dict(instruction=instruction, response=y_before)) \
                    for instruction, input in zip(question, examples['input']) 
                ]
        else:
            if args.is_answer:
                rectify_contents = [rectify_prompt['prompt_no_input'].format_map(dict(instruction=question, response=y_before, answer=answer))]
            else:
                rectify_contents = [rectify_prompt['prompt_no_input'].format_map(dict(instruction=question, response=y_before))]

        for j in range(len(rectify_contents)):
            messages = [{'role': 'user', 'content': rectify_contents[j]}]
            print('rectify messages:\n', messages)
            args.instant_token = num_tokens_from_messages(messages, args.model)
            print('instant token', args.instant_token)
            args.total_token += args.instant_token
            args.chat_cnt += 1
            y_rectify = run_llm(messages, model_name=args.model, temperature=0)
            # y_rectify = answer
            print('y_rectify:\n', y_rectify)

        # judge y_before and y_rectify
        ## judge prompt
        judge_prompt_list = get_prompt(args.judge_prompt_file, args.judge_prompt_index_list)
        judge_prompt = judge_prompt_list[0]
        if 'input' in examples:
            judge_contents = [
                judge_prompt['prompt_input'].format_map(dict(instruction=question, input=input, response1=y_rectify, response2=y_before, answer=answer)) if input != "" \
                else judge_prompt['prompt_no_input'].format_map(dict(instruction=question, response1=y_rectify, response2=y_before, answer=answer)) \
                for instruction, input in zip(question, examples['input'])
            ]
        else:
            judge_contents = [
                judge_prompt['prompt_no_input'].format_map(dict(instruction=question, response1=y_rectify, response2=y_before, answer=answer))
            ]
        for j in range(len(judge_contents)):
            messages = [{'role': 'user', 'content': judge_contents[j]}]
            print('judge messages:\n', messages)
            args.instant_token = num_tokens_from_messages(messages, args.model)
            print('instant token', args.instant_token)
            args.total_token += args.instant_token
            args.chat_cnt += 1
            judge_response = run_llm(messages, model_name=args.judge_model, temperature=0)
            print('judge_response:\n', judge_response)

        # extract score
        ## extract prompt
        extract_prompt_list = get_prompt(args.extract_prompt_file, args.extract_prompt_index_list)
        extract_prompt = extract_prompt_list[0]
        extract_contents = [extract_prompt['prompt'].format_map(dict(judgement=judge_response))]
        for j in range(len(extract_contents)):
            messages = [{'role': 'user', 'content': extract_contents[j]}]
            print('extract messages:\n', messages)
            args.instant_token = num_tokens_from_messages(messages, args.model)
            print('instant token', args.instant_token)
            args.total_token += args.instant_token
            args.chat_cnt += 1
            extract_response = run_llm(messages, model_name=args.judge_model, temperature=0)
            print('extract_response:\n', extract_response)
            
            # clean extract_response
            cleaned_extract_response = clean_response(extract_response)
            if len(cleaned_extract_response) == 2:
                valid = 1
                # judge_score = cleaned_extract_response[0] if cleaned_extract_response[0] != 2 else -1
                y_rectify_score = cleaned_extract_response[0]
                y_before_score = cleaned_extract_response[1]
                if y_rectify_score > y_before_score:
                    judge_score = 1
                elif y_rectify_score == y_before_score:
                    judge_score = 0
                else:
                    judge_score = -1
            else:
                valid = 0
                judge_score = 0
                y_before_score = 0
                y_rectify_score = 0

        if len(chat_history) == 0:
            chat_history.append(
                {
                    'model': args.model,
                    'judge_model': args.judge_model,
                    'dataset': args.dataset,
                    'origin_prompt': original_prompt,
                    'rectify_prompt': rectify_prompt,
                    'judge_prompt': judge_prompt,
                    'extract_prompt': extract_prompt,
                }
            )
        chat_history.append(
            {
                'question': question,
                'input': examples['input'] if 'input' in examples else None,
                'answer': answer,
                'y_before': y_before,
                'y_rectify': y_rectify,
                'judge': judge_response,
                'extract': extract_response,
                'valid': valid,
                'judge_score': judge_score,
                'y_before_score': y_before_score,
                'y_rectify_score': y_rectify_score,
            }
        )
        if args.data_index >= 0:
            break

    # summarize the results
    total_num = len(chat_history) - 1
    valid_num = sum([item['valid'] if 'valid' in item else 0 for item in chat_history])
    scores = sum([item['judge_score'] if 'judge_score' in item else 0 for item in chat_history])
    rectify_scores = sum([item['y_rectify_score'] if 'y_rectify_score' in item else 0 for item in chat_history])
    before_scores = sum([item['y_before_score'] if 'y_before_score' in item else 0 for item in chat_history])

    end_time = datetime.now()
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    elapsed_time = end_time - start_time
    total_seconds = elapsed_time.total_seconds()
    # Calculate hours, minutes, and seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    # Print the results
    print('Total Statistics:\n',
        f'total_num: {total_num}\n',
        f'valid_num: {valid_num}\n',
        f'scores: {scores}\n',
        f'rectify_scores: {rectify_scores}\n',
        f'before_scores: {before_scores}\n',
        f'Total Tokens: {args.total_token}\n',
        f"Elapsed time: {elapsed_time}\n",
        f"Elapsed time in seconds: {total_seconds:.2f} seconds\n",
        f"Elapsed time in minutes: {minutes:.2f} minutes\n",
        f"Elapsed time in hours: {hours:.2f} hours\n",
    )

    chat_history.append(
        {
            'total_num': total_num,
            'valid_num': valid_num,
            'scores': scores,
            'rectify_scores': rectify_scores,
            'before_scores': before_scores,
            'chat_cnt': args.chat_cnt,
            'total_token': args.total_token,
            'elapsed seconds': total_seconds,
        }
    )
    # save chat_history
    filename = args.exp_name+'.json'
    with open(os.path.join(args.save_path, filename), 'w') as json_file:
        json.dump(chat_history, json_file, indent=4)
    print('chat history saved to', os.path.join(args.save_path, filename))


if __name__ == '__main__':
    main()