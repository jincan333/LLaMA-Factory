import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq
import logging
import os
import random
import wandb
from datasets import load_dataset
import re


def setup_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='post_process')
    parser.add_argument('--dataset', type=str, default='gsm8k')
    parser.add_argument('--data_path', type=str, default='data/P6_generate_data_llama-3-8b-instruct_gpt-4o_gsm8k')
    parser.add_argument('--save_path', type=str, default='data/P6_generate_data_llama-3-8b-instruct_gpt-4o_gsm8k/seed0')
    parser.add_argument('--test_num', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


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

    patterns = ['**Reasoning Process:**\n\n*', '**Reasoning Process:**\n\n', '**Reasoning Process:**\n', '**Reasoning Process:**', \
                'Reasoning Process:\n\n*', 'Reasoning Process:\n\n', 'Reasoning Process:\n', 'Reasoning Process:', \
                '[Reasoning Process]:\n\n*',  '[Reasoning Process]:\n\n', '[Reasoning Process]:\n', '[Reasoning Process]', \
                '**Reasoning Process**\n\n*', '**Reasoning Process**\n\n', '**Reasoning Process**\n', '**Reasoning Process**', \
                'Reasoning Process\n\n*', 'Reasoning Process\n\n', 'Reasoning Process\n', 'Reasoning Process', \
                '[Reasoning Process]\n\n*',  '[Reasoning Process]\n\n', '[Reasoning Process]\n', '[Reasoning Process]', \
                '**Verification:**\n\n*', '**Verification:**\n\n', '**Verification:**\n', '**Verification:**', \
                'Verification:\n\n*', 'Verification:\n\n', 'Verification:\n', 'Verification:', \
                '[Verification]:\n\n*', '[Verification]:\n\n', '[Verification]:\n', '[Verification]', \
                '**Verification**\n\n*', '**Verification**\n\n', '**Verification**\n', \
                '**Verification**', 'Verification\n\n*', 'Verification\n\n', 'Verification\n', 'Verification', \
                '[Verification]\n\n*', '[Verification]\n\n', '[Verification]\n', '[Verification]', \
                '**Final Answer:**\n\n*', '**Final Answer:**\n\n', '**Final Answer:**\n', '**Final Answer:**', \
                'Final Answer:\n\n*', 'Final Answer:\n\n', 'Final Answer:\n', 'Final Answer:', \
                '[Final Answer]:\n\n*', '[Final Answer]:\n\n', '[Final Answer]:\n', '[Final Answer]', \
                '**Final Answer**\n\n*', '**Final Answer**\n\n', '**Final Answer**\n', '**Final Answer**', \
                'Final Answer\n\n*', 'Final Answer\n\n', 'Final Answer\n', 'Final Answer', \
                '[Final Answer]\n\n*', '[Final Answer]\n\n', '[Final Answer]\n', '[Final Answer]']
    for pattern in patterns:
        response = re.sub(re.escape(pattern), '', response, flags=re.IGNORECASE).strip()

    return response


def load_and_process_data_ultrachat(dataset_name, split):
    try:
        dataset = load_dataset(dataset_name, split=split)
        reformatted_data = [{
            'generated': [message['messages'][0], {"role": "assistant", "content": ""}], 
            'real': [message['messages'][0], message['messages'][1]]
        } for message in dataset]
        return reformatted_data
    except Exception as e:
        logging.error(f"Error loading or processing dataset: {e}")
        return []


def load_and_process_data_tulu(dataset_name, input_split, test_split: float=0.1):
    try:
        dataset = load_dataset(dataset_name, split=input_split)
        dataset = dataset.train_test_split(test_size=test_split)
        reformatted_train_data = [{
            'generated': [message['messages'][0], {"role": "assistant", "content": ""}], 
            'real': [message['messages'][0], message['messages'][1]]
        } for message in dataset["train"]]
        reformatted_test_data = [{
            'generated': [message['messages'][0], {"role": "assistant", "content": ""}], 
            'real': [message['messages'][0], message['messages'][1]]
        } for message in dataset["test"]]
        return reformatted_train_data, reformatted_test_data
    except Exception as e:
        logging.error(f"Error loading or processing dataset: {e}")
        return []
    

def load_and_process_data_gsm8k(args):
    try:
        subfolders = [f.path for f in os.scandir(args.data_path) if f.is_dir() and 'frac' in f.path]
        print("Subfolders:\n", subfolders)
        total_data = []
        total_info = []
        for i, sub_path in enumerate(subfolders):
            with open(os.path.join(sub_path, 'final_data.json'), 'r') as f:
                data_frac = json.load(f)
            with open(os.path.join(sub_path, 'data_info.json'), 'r') as f:
                data_info = json.load(f)
            
            # statistic data info
            total_info.insert(2*i, data_info[0])
            total_info.insert(2*i+1, data_info[1])
            if i == 0:
                total_info.append(data_info[-1])
            else:
                total_info[-1]['total_data_num'] += data_info[-1]['total_data_num']
                total_info[-1]['incorrect_y_before_num'] += data_info[-1]['incorrect_y_before_num']
                total_info[-1]['win_y_rectify_num'] += data_info[-1]['win_y_rectify_num']
                total_info[-1]['total_scores'] += data_info[-1]['total_scores']
                total_info[-1]['win_scores'] += data_info[-1]['win_scores']
                total_info[-1]['rectify_total_scores'] += data_info[-1]['rectify_total_scores']
                total_info[-1]['before_total_scores'] += data_info[-1]['before_total_scores']
                total_info[-1]['total_token'] += data_info[-1]['total_token']
                total_info[-1]['total_input_tokens'] += data_info[-1]['total_input_tokens']
                total_info[-1]['total_output_tokens'] += data_info[-1]['total_output_tokens']
                total_info[-1]['elapsed hours'] += data_info[-1]['elapsed hours']
                total_info[-1]['start_time'] = min(total_info[-1]['start_time'], data_info[-1]['start_time'])
                total_info[-1]['end_time'] = max(total_info[-1]['end_time'], data_info[-1]['end_time'])

            # post process data
            prompt = data_info[0]['original_prompt']

            post_data = [{
                'generated': [{'role': 'user', 'content': prompt.format_map(dict(instruction=piece['question'], input=piece['input']))}, {"role": "assistant", "content": piece['y_before']}], 
                'real': [{'role': 'user', 'content': prompt.format_map(dict(instruction=piece['question'], input=piece['input']))}, {"role": "assistant", "content": piece['answer']}],
                'rectify': [{'role': 'user', 'content': prompt.format_map(dict(instruction=piece['question'], input=piece['input']))}, {"role": "assistant", "content": piece['y_rectify']}],
                'rectify_before_combine': [{'role': 'user', 'content': prompt.format_map(dict(instruction=piece['question'], input=piece['input']))}, {"role": "assistant", "content": piece['y_before']}] if piece['y_before_score'] == 1 else [{'role': 'user', 'content': prompt.format_map(dict(instruction=piece['question'], input=piece['input']))}, {"role": "assistant", "content": piece['y_rectify']}],
                'rectify_answer_combine': [{'role': 'user', 'content': prompt.format_map(dict(instruction=piece['question'], input=piece['input']))}, {"role": "assistant", "content": piece['answer']}] if piece['y_before_score'] == 1 else [{'role': 'user', 'content': prompt.format_map(dict(instruction=piece['question'], input=piece['input']))}, {"role": "assistant", "content": piece['y_rectify']}]
            } for item in data_frac for piece in item]
            total_data.extend(post_data)

        return total_data, total_info

    except Exception as e:
        logging.error(f"Error loading or processing dataset: {e}")
        return []


def save_to_json(data, path):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
    except IOError as e:
        logging.error(f"Error saving data to {path}: {e}")


def save_to_parquet(dataset, path):
    try:
        pq.write_table(dataset.data.table, path)
    except Exception as e:
        logging.error(f"Error saving data to {path}: {e}")


def main():
    args = setup_arg_parser()
    print(json.dumps(vars(args), indent=4))

    wandb.require('core')
    wandb.login(key=os.environ.get('WANDB_API_KEY') )
    wandb.init(project='LLM_Alignment_post_process_'+args.dataset, entity=os.environ.get('WANDB_USER_NAME'), name=args.exp_name)

    output_dir = Path(args.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == 'HuggingFaceH4/ultrachat_200k':
        train_data = load_and_process_data_ultrachat(args.dataset, 'train_sft')
        test_data = load_and_process_data_ultrachat(args.dataset, 'test_sft')
    elif "tulu-v2-sft-mixture" in args.dataset:
        train_data, test_data = load_and_process_data_tulu(args.dataset, 'train', test_split=0.1)
    elif args.dataset == 'gsm8k':
        total_data, data_info = load_and_process_data_gsm8k(args)
    else:
        raise ValueError(f"current {args.dataset} dataset is not supported")
    
    # remove redundant from y_rectify
    for item in total_data:
        item['rectify'][1]['content'] = remove_redundant(item['rectify'][1]['content'])

    test_num = min(args.test_num, len(total_data))
    test_data = random.sample(total_data, test_num)

    train_json_path = os.path.join(output_dir,'train.json')
    test_json_path = os.path.join(output_dir,'test.json')

    save_to_json(total_data, train_json_path)
    save_to_json(test_data, test_json_path)
    save_to_json(data_info, os.path.join(output_dir,'data_info.json'))
    save_to_json(total_data, os.path.join(output_dir,'total_data.json'))

    dataset_train = load_dataset('json', data_files=str(train_json_path), split='train')
    dataset_test = load_dataset('json', data_files=str(test_json_path), split='train')

    save_to_parquet(dataset_train, os.path.join(output_dir,'train_prefs-00000-of-00001.parquet'))
    save_to_parquet(dataset_test, os.path.join(output_dir,'test_prefs-00000-of-00001.parquet'))

    os.remove(train_json_path)
    os.remove(test_json_path)
    print("post process done")
    print("data location: ", output_dir)

if __name__ == "__main__":
    main()