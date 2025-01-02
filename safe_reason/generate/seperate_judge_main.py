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
    parser.add_argument('--model', type=str, default='llama-3.1-8b', choices=['gpt-4o', 'gpt-4o-mini', 'llama-3-8b', 'llama-3-70b', 'llama-2-7b', 'llama-2-13b', 'llama-2-70b', 'qwen-2-72b', 'llama-3.1-8b', 'llama-3.1-70b', 'mistral-0.3-7b'])
    parser.add_argument('--judge_model', type=str, default='gpt-4o', choices=['gpt-4o', 'gpt-4o-mini', 'llama-3-8b', 'llama-3-70b', 'llama-2-7b', 'llama-2-13b', 'llama-2-70b', 'qwen-2-72b', 'llama-3.1-8b', 'llama-3.1-70b', 'mistral-0.3-7b'])

    # dataset
    parser.add_argument('--dataset', type=str, default='gsm8k', choices=['gsm8k', 'MATH', 'AQuA', 'deepmind', 'mmlu', 'numglue', 'sat', 'simuleq', 'SVAMP'])
    parser.add_argument('--data_path', type=str, default='../data/math_eval_datasets/')
    parser.add_argument('--data_num', type=int, default=10000)
    parser.add_argument('--data_index', type=int, default=1)

    # prompt
    parser.add_argument('--origin_prompt_file', type=str, default='prompt/origin_manual.json', help='origin prompt file location')
    parser.add_argument('--origin_prompt_index_list', type=str, default='0')
    parser.add_argument('--rectify_prompt_file', type=str, default='prompt/rectify_manual.json', help='rectify prompt file location')
    parser.add_argument('--rectify_prompt_index_list', type=str, default='17')
    parser.add_argument('--judge_prompt_file', type=str, default='prompt/judge_manual.json', help='judge prompt file location')
    parser.add_argument('--judge_prompt_index_list', type=str, default='6')
    parser.add_argument('--extract_prompt_file', type=str, default='prompt/extract_manual.json', help='extract prompt file location')
    parser.add_argument('--extract_prompt_index_list', type=str, default='1')
    parser.add_argument('--is_replace', type=int, default=0)
    parser.add_argument('--max_attempt', type=int, default=1)
    parser.add_argument('--score_threshold', type=int, default=1)
    parser.add_argument('--is_json', type=int, default=0)
    parser.add_argument('--is_find_error', type=int, default=1)
    parser.add_argument('--error_prompt_file', type=str, default='prompt/error_manual.json', help='find error prompt file location')
    parser.add_argument('--error_prompt_index_list', type=str, default='0')


    # count
    parser.add_argument('--question_chat_cnt', type=int, default=0)
    parser.add_argument('--total_chat_cnt', type=int, default=0)
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
    elif any(x in model for x in ['qwen', 'mistral']):
        return 32768
    else:
        raise ValueError(f"Model {model} not supported")


def run_llm(messages, model_name="gpt-4o-2024-05-13", temperature=0, batchsize=1, retry_attempts=3):
    for _ in range(retry_attempts):
        try:
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
            time.sleep(10)
            response = 'error'

    return response


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
    elif model_name == 'mistral-0.3-7b':
        model_name = 'mistralai/Mistral-7B-Instruct-v0.3'

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


def get_messages(prompt_path, prompt_index_list, tag='before', question=None, input=None, response=None, answer=None, judgement=None, error=None):
    prompt_list = get_prompt(prompt_path, prompt_index_list)
    prompt = prompt_list[0]
    if tag == 'before':
        content = prompt['prompt'].format_map(dict(instruction=question, input=input))
    elif tag == 'judge':
        content = prompt['prompt'].format_map(dict(instruction=question, input=input, response=response, answer=answer))
    elif tag == 'rectify':
        content = prompt['prompt'].format_map(dict(instruction=question, input=input, response=response, answer=answer))
    elif tag == 'extract':
        content = prompt['prompt'].format_map(dict(judgement=judgement))
    elif tag == 'error':
        content = prompt['prompt'].format_map(dict(instruction=question, input=input, response=response, answer=answer))
    elif tag == 'error_rectify':
        content = prompt['prompt'].format_map(dict(instruction=question, input=input, response=response, answer=answer, error=error))

    messages = [{'role': 'user', 'content': content}]

    return messages


def get_response(messages, model_name, args, temperature=0):
    args.instant_token = num_tokens_from_messages(messages, model_name)
    args.total_token += args.instant_token
    args.question_chat_cnt += 1
    response = run_llm(messages, model_name=model_name, temperature=temperature)

    return response


def get_tokens(messages, model_name, args):
    args.instant_token = num_tokens_from_messages(messages, model_name)
    args.total_token += args.instant_token

    return args.instant_token


def main():
    args = parse_args()
    wandb.login(key=os.environ.get('WANDB_API_KEY') )
    wandb.init(project='LLM_Alignment_'+args.model+'_'+args.dataset, entity=os.environ.get('WANDB_USER_NAME'), name=args.exp_name)

    # prepare model
    args.model = get_model(args.model)
    args.max_tokens = max_tokens(args.model)
    args.judge_model = get_model(args.judge_model)

    print(json.dumps(vars(args), indent=4))
    start_time = datetime.now()
    print(f"\nStart time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # load data
    if args.dataset == 'gsm8k':
        data_path = '../data/gsm8k/train.jsonl'
    elif args.dataset == 'MATH':
        data_path = '../data/MATH/train.jsonl'
    elif args.dataset == 'mmlu':
        data_path = '../data/mmlu/train.jsonl'
    try:
        data_all = []
        with open(data_path, 'r') as f:
            data_all = json.load(f) 
    except:
        try:
            with open(data_path, 'r') as f:
                for line in f:
                    data_all.append(json.loads(line))
        except:
            print('dataset not found')
            return

    # chat
    chat_history = []
    if args.data_num > len(data_all):
        repeat_count = args.data_num // len(data_all)
        remainder = args.data_num % len(data_all)
        data_all = data_all * repeat_count + data_all[:remainder]

    for i in range(args.data_num): 
        if args.data_index >= 0:
            i = args.data_index
        print('\n' + '*' * 50 + f' Question ID: {i} ' + '*' * 50)
        example = data_all[i]
        args.question_chat_cnt = 0
        # gsm8k: question + answer
        # MATH: problem + solution
        # mmlu: question + choice + answer
        if args.dataset == 'gsm8k':
            key1 = 'query'
            key2 = 'response'
        elif args.dataset == 'MATH':
            key1 = 'question'
            key2 = 'solution'
        elif args.dataset == 'mmlu':
            key1 = 'question'
            key2 = 'choice'
            key3 = 'answer'
        question = example[key1].strip()
        input = example['input'].strip() if 'input' in example else None
        answer = example[key2].strip()
        print('current question: ', question)
        print('current answer: ', answer)

        # obtain y_before
        ## original prompt
        # TODO temperature is an improtant hyperparameter
        # TODO can add repetition penalty to the LLM
        attempt = 1
        while True:
            messages = get_messages(args.origin_prompt_file, args.origin_prompt_index_list, tag='before', question=question)
            print('y_before messages:\n', messages)
            y_before = get_response(messages, args.model, args, temperature=0.2)
            y_before_messages_tokens = args.instant_token
            print('y_before messages tokens', y_before_messages_tokens)
            print('y_before response:\n', y_before)
            y_before_tokens = get_tokens(y_before, args.model, args)
            print('y_before tokens: ', y_before_tokens)
            if args.is_replace:
                y_before = re.sub(r'response', 'answer', y_before, flags=re.IGNORECASE)
            
            if y_before_tokens < args.max_tokens - 500:
                break
            else:
                if attempt >= args.max_attempt:
                    y_before = y_before[: args.max_tokens - 500]
                    break
                attempt += 1

        # judge y_before
        messages = get_messages(args.judge_prompt_file, args.judge_prompt_index_list, tag='judge', question=question, input=input, response=y_before, answer=answer)
        print('y_before judge messages:\n', messages)
        y_before_judge_response = get_response(messages, args.judge_model, args, temperature=0)
        y_before_judge_messages_tokens = args.instant_token
        print('y_before judge messages tokens', y_before_judge_messages_tokens)
        print('y_before judge response:\n', y_before_judge_response)
        y_before_judge_tokens = get_tokens(y_before_judge_response, args.judge_model, args)
        print('y_before judge tokens: ', y_before_judge_tokens)
        # extract the score of y_before
        ## extract prompt
        # attempt = 1
        # while True:
        messages = get_messages(args.extract_prompt_file, args.extract_prompt_index_list, tag='extract', judgement=y_before_judge_response)
        print('y_before extract messages:\n', messages)
        y_before_extract_response = get_response(messages, args.judge_model, args, temperature=0)
        y_before_extract_messages_tokens = args.instant_token
        print('y_before extract messages tokens', y_before_extract_messages_tokens)
        print('y_before extract response:\n', y_before_extract_response)
        y_before_extract_tokens = get_tokens(y_before_extract_response, args.judge_model, args)
        print('y_before extract tokens: ', y_before_extract_tokens)
        
        # clean extract_response
        cleaned_y_before_extract_response = clean_response(y_before_extract_response)
        if len(cleaned_y_before_extract_response) == 1:
            y_before_valid = 1
            y_before_score = cleaned_y_before_extract_response[0]
        else:
            y_before_valid = 0
            y_before_score = 0
            # attempt += 1
            # if attempt >= args.max_attempt:
            #     break

        attempt = 1
        while True:
            try:
                # obtain, judge, and extract y_rectify until correct
                # obtain y_rectify
                if args.is_find_error:
                    # first find errors, then revise
                    messages = get_messages(args.error_prompt_file, args.error_prompt_index_list, tag='error', question=question, input=input, response=y_before, answer=answer)
                    print('find error messages:\n', messages)
                    error = get_response(messages, args.model, args, temperature=0)
                    error_messages_tokens = args.instant_token
                    print('find error messages tokens', error_messages_tokens)
                    print('find error response:\n', error)
                    error_tokens = get_tokens(error, args.model, args)
                    print('find error tokens: ', error_tokens)

                    messages = get_messages(args.rectify_prompt_file, args.rectify_prompt_index_list, tag='error_rectify', question=question, input=input, response=y_before, answer=answer, error=error)
                    print('y_rectify messages:\n', messages)
                    y_rectify_whole = get_response(messages, args.model, args, temperature=0.95)
                    y_rectify_whole_messages_tokens = args.instant_token
                    print('y_rectify_whole messages tokens', y_rectify_whole_messages_tokens)
                    print('y_rectify_whole response:\n', y_rectify_whole)
                    y_rectify_whole_tokens = get_tokens(y_rectify_whole, args.model, args)
                    print('y_rectify_whole tokens: ', y_rectify_whole_tokens)
                else:
                    messages = get_messages(args.rectify_prompt_file, args.rectify_prompt_index_list, tag='rectify', question=question, input=input, response=y_before, answer=answer)
                    print('y_rectify messages:\n', messages)
                    y_rectify_whole = get_response(messages, args.model, args, temperature=0.95)
                    y_rectify_whole_messages_tokens = args.instant_token
                    print('y_rectify_whole messages tokens', y_rectify_whole_messages_tokens)
                    print('y_rectify_whole response:\n', y_rectify_whole)
                    y_rectify_whole_tokens = get_tokens(y_rectify_whole, args.model, args)
                    print('y_rectify_whole tokens: ', y_rectify_whole_tokens)

                if args.is_json == 1:
                    # extract response
                    response_json = json.loads(y_rectify_whole)
                    y_rectify = response_json.get('response', '')
                    print('y_rectify response:\n', y_rectify)
                else:
                    y_rectify = y_rectify_whole
            except Exception as e:
                print("There is an exception in finding the errors:\n", e)
                continue

            # judge y_rectify
            messages = get_messages(args.judge_prompt_file, args.judge_prompt_index_list, tag='judge', question=question, input=input, response=y_rectify, answer=answer)
            print('y_rectify judge messages:\n', messages)
            y_rectify_judge_response = get_response(messages, args.judge_model, args, temperature=0)
            y_rectify_judge_messages_tokens = args.instant_token
            print('y_rectify judge messages tokens', y_rectify_judge_messages_tokens)
            print('y_rectify judge response:\n', y_rectify_judge_response)
            y_rectify_judge_tokens = get_tokens(y_rectify_judge_response, args.judge_model, args)
            print('y_rectify judge tokens: ', y_rectify_judge_tokens)

            # extract y_rectify
            messages = get_messages(args.extract_prompt_file, args.extract_prompt_index_list, tag='extract', judgement=y_rectify_judge_response)
            print('y_rectify extract messages:\n', messages)
            y_rectify_extract_response = get_response(messages, args.judge_model, args, temperature=0)
            y_rectify_extract_messages_tokens = args.instant_token
            print('y_rectify extract messages tokens', y_rectify_extract_messages_tokens)
            print('y_rectify extract response:\n', y_rectify_extract_response)
            y_rectify_extract_tokens = get_tokens(y_rectify_extract_response, args.judge_model, args)
            print('y_rectify extract tokens: ', y_rectify_extract_tokens)
            
            cleaned_y_rectify_extract_response = clean_response(y_rectify_extract_response)
            if len(cleaned_y_rectify_extract_response) == 1:
                y_rectify_valid = 1
                y_rectify_score = cleaned_y_rectify_extract_response[0]
            else:
                y_rectify_valid = 0
                y_rectify_score = 0

            if (y_rectify_valid == 1 and y_rectify_score >= args.score_threshold) or attempt >= args.max_attempt:
                break
            else:
                attempt += 1

        if len(chat_history) == 0:
            chat_history.append(
                {
                    'model': args.model,
                    'judge_model': args.judge_model,
                    'dataset': args.dataset,
                    'origin_prompt': get_prompt(args.origin_prompt_file, args.origin_prompt_index_list),
                    'rectify_prompt': get_prompt(args.rectify_prompt_file, args.rectify_prompt_index_list),
                    'judge_prompt': get_prompt(args.judge_prompt_file, args.judge_prompt_index_list),
                    'extract_prompt': get_prompt(args.extract_prompt_file, args.extract_prompt_index_list),
                    'is_replace': args.is_replace,
                    'max_attempt': args.max_attempt,
                    'score_threshold': args.score_threshold,
                }
            )
        chat_history.append(
            {
                'id': i,
                'question': question,
                'input': input,
                'answer': answer,
                'question_chat_cnt': args.question_chat_cnt,
                'attempt': attempt,
                'y_before': y_before,
                'y_before_input_tokens': y_before_messages_tokens,
                'y_before_output_tokens': y_before_tokens,
                'y_before_judge': y_before_judge_response,
                'y_before_judge_input_tokens': y_before_judge_messages_tokens,
                'y_before_judge_output_tokens': y_before_judge_tokens,
                'y_before_extract': y_before_extract_response,
                'y_before_extract_input_tokens': y_before_extract_messages_tokens,
                'y_before_extract_output_tokens': y_before_extract_tokens,
                'y_before_valid': y_before_valid,
                'y_before_score': y_before_score,
                'y_rectify': y_rectify,
                'y_rectify_input_tokens': y_rectify_whole_messages_tokens,
                'y_rectify_output_tokens': y_rectify_whole_tokens,
                'y_rectify_judge': y_rectify_judge_response,
                'y_rectify_judge_input_tokens': y_rectify_judge_messages_tokens,
                'y_rectify_judge_output_tokens': y_rectify_judge_tokens,
                'y_rectify_extract': y_rectify_extract_response,
                'y_rectify_extract_input_tokens': y_rectify_extract_messages_tokens,
                'y_rectify_extract_output_tokens': y_rectify_extract_tokens,
                'y_rectify_valid': y_rectify_valid,
                'y_rectify_score': y_rectify_score,
            }
        )
        # save chat_history
        filename = args.exp_name+'.json'
        with open(os.path.join(args.save_path, filename), 'w') as json_file:
            json.dump(chat_history, json_file, indent=4)
        args.total_chat_cnt +=args.question_chat_cnt
        if args.data_index >= 0:
            break

    # summarize the results
    total_num = len(chat_history) - 1
    valid_num = sum([int(item['y_rectify_valid'] and item['y_before_valid']) if 'y_rectify_valid' in item else 0 for item in chat_history])
    win_scores = sum([int(item['y_rectify_score'] >= item['y_before_score']) if 'y_rectify_score' in item else 0 for item in chat_history])
    rectify_total_scores = sum([item['y_rectify_score'] if 'y_rectify_score' in item else 0 for item in chat_history])
    before_total_scores = sum([item['y_before_score'] if 'y_before_score' in item else 0 for item in chat_history])

    end_time = datetime.now()
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    elapsed_time = end_time - start_time
    total_seconds = elapsed_time.total_seconds()
    # Calculate hours, minutes, and seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    # Print the results
    print('Total Statistics:\n',
        f'total_question_num: {total_num}\n',
        f'valid_answer_num: {valid_num}\n',
        f'win_scores: {win_scores}\n',
        f'rectify_total_scores: {rectify_total_scores}\n',
        f'before_total_scores: {before_total_scores}\n',
        f'Total Tokens: {args.total_token}\n',
        f'Total Chat Cnt: {args.total_chat_cnt}\n',
        f"Elapsed time: {elapsed_time}\n",
        f"Elapsed time in seconds: {total_seconds:.2f} seconds\n",
        f"Elapsed time in minutes: {minutes:.2f} minutes\n",
        f"Elapsed time in hours: {hours:.2f} hours\n",
    )

    chat_history.append(
        {
            'total_question_num': total_num,
            'valid_answer_num': valid_num,
            'win_scores': win_scores,
            'rectify_total_scores': rectify_total_scores,
            'before_total_scores': before_total_scores,
            'total_chat_cnt': args.total_chat_cnt,
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