import os
import time
import warnings
from typing import Union
import openai
from openai import OpenAI
from datasets import Dataset, concatenate_datasets
import multiprocessing
import datasets
import re
import argparse
import json
from strong_reject.load_datasets import load_strongreject
import specifications as specs
from generate import personalized_generate, extract_content, extract_content_first, get_model
from deepthought import DeepthoughtModel
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from strong_reject.jailbreaks import apply_jailbreaks_to_dataset


specifications = {
    1: specs.violent_crimes,
    2: specs.non_violent_crimes,
    3: specs.sex_related_crimes,
    4: specs.child_sexual_exploitation,
    5: specs.cbrne,
    6: specs.suicide_and_self_harm,
    7: specs.hate,
    8: specs.privacy,
    9: specs.intellectual_property,
    10: specs.disinformation_and_defamation,
    11: specs.elections,
    12: specs.sexual_content,
    13: specs.others
}

if __name__ == "__main__":  

    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--task', type=str, default='beavertails_build_train_dataset', choices=['beavertails_classification', 'beavertails_generate_cot_safe', 'beavertails_generate_cot_unsafe', 'beavertails_build_train_dataset', 'temp'])
        parser.add_argument('--model', type=str, default='llama-3-8b-instruct', help='base models are gpt-4o-mini-2024-07-18, gpt-4o-2024-11-20, llama-3-8b-instruct, gemma-2-9b-it, qwq-32b-preview, deepthought-8b, o1-2024-12-17')
        parser.add_argument('--judge_model', type=str, default='gpt-4o-2024-11-20', help='base models are gpt-4o-mini-2024-07-18, gpt-4o-2024-11-20, o1-2024-12-17')
        parser.add_argument('--jailbreak', type=str, default='pap_misrepresentation', help="none, pair, happy_to_help, wikipedia, distractors, prefix_injection, combination_2, pap_misrepresentation")
        parser.add_argument('--temperature', type=float, default=0, help='temperature for generation')
        parser.add_argument('--top_p', type=float, default=0.7, help='top_p for generation')
        parser.add_argument('--max_length', type=int, default=4096, help='max_length for generation')
        parser.add_argument('--num_samples', type=int, default=500, help='number of samples for generation')
        parser.add_argument('--ratio', type=int, default=0, help='ratio of safe samples for generation')
        parser.add_argument('--cot_rating', type=int, default=5, help='cot rating for generation')
        parser.add_argument('--final_response_rating', type=int, default=1, help='final response rating for generation')
        parser.add_argument('--category_unsafe_samples', type=int, default=100, help='number of samples for generation')
        parser.add_argument('--total_unsafe_samples', type=int, default=1000, help='number of samples for generation')
        args = parser.parse_args()
        return args

    args = parse_args()
    args.model, model_print_name = get_model(args.model)
    print(json.dumps(vars(args), indent=4))
    jailbreaks = list(args.jailbreak.split(','))

    if args.model != 'gpt-4o-mini-2024-07-18' and args.model != 'gpt-4o-2024-11-20' and args.model != 'o1-2024-12-17':
        tokenizer = AutoTokenizer.from_pretrained(args.model)   
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = args.max_length
        generate_model = LLM(model=args.model, max_num_seqs=64, tensor_parallel_size=1, max_model_len=args.max_length)
        sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_length)

    if args.task == 'strongreject_classification':
        print(f'generating classification for strongreject')
        dataset = load_strongreject()
        prompt_classification = specs.prompt_classification
        dataset = dataset.map(lambda x: {"prompt": prompt_classification.format(prompt=x['forbidden_prompt'])})
        responses_dataset = personalized_generate(dataset, system_prompt=None, models=[args.judge_model], use_local=False, decode_responses=False, temperature=0, max_tokens=512)
        pattern = re.compile(r'(?:final category number)', re.IGNORECASE | re.DOTALL)
        responses_dataset = responses_dataset.map(lambda x: {"final_response": extract_content(x['response'], pattern)})
        processed_dataset = responses_dataset.map(
            lambda x: {
                'number': int(match.group(0)) if (match := re.search(r"\d+", x['final_response'])) else None,
            },
            num_proc=32
        )
        processed_dataset = processed_dataset.remove_columns('model')
        records = [dict(row) for row in processed_dataset]
        with open('data/strongreject/classification.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)

        full_dataset = []
        for number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]:
            filtered_dataset = processed_dataset.filter(lambda x: x['number'] == number)
            filtered_dataset = filtered_dataset.shuffle(seed=1234)
            print('unsafe', number, len(filtered_dataset))
            cot_specification = specs.cot_specification
            filtered_dataset = filtered_dataset.map(lambda x: {"classification_cot_prompt": cot_specification.format(prompt="{prompt}", spec_category=specifications[x['number']])})
            full_dataset.append(filtered_dataset)
        full_dataset = concatenate_datasets(full_dataset)
        full_dataset = full_dataset.remove_columns(['prompt', 'response', 'final_response'])
        records = [dict(row) for row in full_dataset]
        with open(f'data/strongreject/classification.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)
        
    if args.task == 'beavertails_classification':
        print(f'generating classification for beavertails')
        dataset = datasets.load_dataset('PKU-Alignment/BeaverTails', split='30k_train')
        col_renames = {
            'prompt': 'forbidden_prompt',
            'category': 'category',
            'response': 'rejected',
            'is_safe': 'is_safe'
        }
        dataset = dataset.rename_columns(col_renames)
        safe_dataset = dataset.filter(lambda x: x['is_safe'] == True)
        records = [dict(row) for row in safe_dataset]
        with open('data/beavertails/safe.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)
        
        unsafe_dataset = dataset.filter(lambda x: x['is_safe'] == False)
        prompt_classification = specs.prompt_classification
        unsafe_prompt_dataset = unsafe_dataset.map(lambda x: {"prompt": prompt_classification.format(prompt=x['forbidden_prompt'])})
        responses_dataset = personalized_generate(unsafe_prompt_dataset, system_prompt=None, models=[args.judge_model], use_local=False, decode_responses=False, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_length)
        pattern = re.compile(r'(?:final category number)', re.IGNORECASE | re.DOTALL)
        responses_dataset = responses_dataset.map(lambda x: {"final_response": extract_content(x['response'], pattern)})
        processed_dataset = responses_dataset.map(
            lambda x: {
                'number': int(match.group(0)) if (match := re.search(r"\d+", x['final_response'])) else None,
            },
            num_proc=32
        )
        processed_dataset = processed_dataset.remove_columns('model')
        records = [dict(row) for row in processed_dataset]
        with open('cache/unsafe_gpt-4o-2024-11-20_classification.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)
        
        unsafe_dataset = datasets.load_dataset('json', data_files='cache/unsafe_gpt-4o-2024-11-20_classification.json', split='train')
        unsafe_dataset = unsafe_dataset.remove_columns(['response', 'final_response', 'prompt'])
        unmatch_unsafe_dataset = unsafe_dataset.filter(lambda x: x['number'] is None or x['number'] not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        match_unsafe_dataset = unsafe_dataset.filter(lambda x: x['number'] is not None and 1 <= x['number'] <= 12)
        records = [dict(row) for row in unmatch_unsafe_dataset]
        with open('data/beavertails/unsafe_unmatch.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)
        records = [dict(row) for row in match_unsafe_dataset]
        with open('data/beavertails/unsafe_match.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)

    if args.task == 'beavertails_generate_cot_unsafe':
        print(f'generating cot for beavertails unsafe')
        unsafe_dataset = datasets.load_dataset('json', data_files='data/beavertails/unsafe_match.json', split='train')
        
        for number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            filtered_dataset = unsafe_dataset.filter(lambda x: x['number'] == number)
            filtered_dataset = filtered_dataset.shuffle(seed=1234)
            filtered_dataset = filtered_dataset.select(range(min(len(filtered_dataset), args.num_samples)))
            print('unsafe', number, len(filtered_dataset))
            cot_specification = specs.cot_specification
            cot_prompt_unsafe_dataset = filtered_dataset.map(lambda x: {"cot_prompt": cot_specification.format(prompt=x['forbidden_prompt'], spec_category=specifications[x['number']])})
            if args.model == 'gpt-4o-mini-2024-07-18' or args.model == 'gpt-4o-2024-11-20' or args.model == 'o1-2024-12-17':
                cot_dataset = personalized_generate(cot_prompt_unsafe_dataset, [args.model], target_column="cot_prompt", use_local=False, decode_responses=False, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_length)
            else:
                cot_prompt_unsafe_dataset = cot_prompt_unsafe_dataset.map(lambda x: {"cot_prompt_formatted": tokenizer.apply_chat_template([{'role': 'user', 'content': x['cot_prompt']}], tokenize=False, add_generation_prompt=True)})
                responses_outputs = generate_model.generate(cot_prompt_unsafe_dataset['cot_prompt_formatted'], sampling_params)
                cot_dataset = cot_prompt_unsafe_dataset.map(lambda x, idx: {"response": responses_outputs[idx].outputs[0].text}, with_indices=True)

            # filter cot_dataset
            pattern = re.compile(r'(?:Analysis\s*(.*?)\s*Final Response)', re.IGNORECASE | re.DOTALL)
            cot_dataset = cot_dataset.map(lambda x: {"analysis": extract_content_first(x['response'], pattern)})
            pattern = re.compile(r'(?:Final Response)', re.IGNORECASE | re.DOTALL)
            cot_dataset = cot_dataset.map(lambda x: {"final_response": extract_content(x['response'], pattern)})
            cot_dataset = cot_dataset.remove_columns(['cot_prompt'])
            cot_dataset = cot_dataset.rename_column('response', 'cot_response')

            reward_judge = specs.reward_judge
            judge_dataset = cot_dataset.map(lambda x: {"judge_prompt": reward_judge.format(prompt=x['forbidden_prompt'], cot=x['analysis'], final_response=x['final_response'], spec_category=specifications[x['number']])})
            judge_dataset = personalized_generate(judge_dataset, system_prompt=None, target_column='judge_prompt', models=[args.judge_model], use_local=False, decode_responses=False, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_length)
            pattern = re.compile(r'(?:Chain of Thought Rating\s*(.*?)\s*Final Response Rating)', re.IGNORECASE | re.DOTALL)
            judge_dataset = judge_dataset.map(lambda x: {"cot_rating": extract_content_first(x['response'], pattern)})
            judge_dataset = judge_dataset.map(lambda x: {'cot_rating': int(match.group(0)) if (match := re.search(r"\d+", x['cot_rating'])) else None})
            pattern = re.compile(r'(?:Final Response Rating)', re.IGNORECASE | re.DOTALL)
            judge_dataset = judge_dataset.map(lambda x: {"final_response_rating": extract_content(x['response'], pattern)})
            judge_dataset = judge_dataset.map(lambda x: {'final_response_rating': int(match.group(0)) if (match := re.search(r"\d+", x['final_response_rating'])) else None})
            judge_dataset = judge_dataset.remove_columns(['model', 'judge_prompt', 'response'])
            col_renames = {
                'analysis': 'unsafe_analysis',
                'final_response': 'unsafe_final_response',
                'cot_rating': 'unsafe_cot_rating',
                'final_response_rating': 'unsafe_final_response_rating',
                'cot_response': 'unsafe_cot_response'
            }
            judge_dataset = judge_dataset.rename_columns(col_renames)
            records = [dict(row) for row in judge_dataset]
            with open(f'data/beavertails/{model_print_name}_unsafe_{number}_cot.json', 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=4)

            cot_specification_helpful = specs.cot_specification_helpful
            overall_helpful = specs.overall_helpful
            judge_dataset = apply_jailbreaks_to_dataset(judge_dataset, jailbreaks)
            if args.jailbreak == 'none':
                cot_prompt_unsafe_dataset = judge_dataset.map(lambda x: {"cot_prompt": cot_specification_helpful.format(prompt=x['forbidden_prompt'], spec_category=overall_helpful)})
            else:
                cot_prompt_unsafe_dataset = judge_dataset.map(lambda x: {"cot_prompt": cot_specification_helpful.format(prompt=x['jailbroken_prompt'], spec_category=overall_helpful)})

            if args.model == 'gpt-4o-mini-2024-07-18' or args.model == 'gpt-4o-2024-11-20' or args.model == 'o1-2024-12-17':
                cot_dataset = personalized_generate(cot_prompt_unsafe_dataset, [args.model], target_column="cot_prompt", use_local=False, decode_responses=False, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_length)
            else:
                cot_prompt_unsafe_dataset = cot_prompt_unsafe_dataset.map(lambda x: {"cot_prompt_formatted": tokenizer.apply_chat_template([{'role': 'user', 'content': x['cot_prompt']}], tokenize=False, add_generation_prompt=True)})
                responses_outputs = generate_model.generate(cot_prompt_unsafe_dataset['cot_prompt_formatted'], sampling_params)
                cot_dataset = cot_prompt_unsafe_dataset.map(lambda x, idx: {"response": responses_outputs[idx].outputs[0].text}, with_indices=True)

            pattern = re.compile(r'(?:Analysis\s*(.*?)\s*Final Response)', re.IGNORECASE | re.DOTALL)
            cot_dataset = cot_dataset.map(lambda x: {"analysis": extract_content_first(x['response'], pattern)})
            pattern = re.compile(r'(?:Final Response)', re.IGNORECASE | re.DOTALL)
            cot_dataset = cot_dataset.map(lambda x: {"final_response": extract_content(x['response'], pattern)})
            cot_dataset = cot_dataset.remove_columns(['cot_prompt'])
            cot_dataset = cot_dataset.rename_column('response', 'cot_response')
            records = [dict(row) for row in cot_dataset]
            with open(f'data/beavertails/{model_print_name}_unsafe_{number}_cot.json', 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=4)

            reward_judge_helpful = specs.reward_judge_helpful
            judge_dataset = cot_dataset.map(lambda x: {"judge_prompt": reward_judge_helpful.format(prompt=x['forbidden_prompt'], cot=x['analysis'], final_response=x['final_response'], spec_category=overall_helpful)})
            judge_dataset = personalized_generate(judge_dataset, system_prompt=None, target_column='judge_prompt', models=[args.judge_model], use_local=False, decode_responses=False, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_length)
            pattern = re.compile(r'(?:Chain of Thought Rating\s*(.*?)\s*Final Response Rating)', re.IGNORECASE | re.DOTALL)
            judge_dataset = judge_dataset.map(lambda x: {"cot_rating": extract_content_first(x['response'], pattern)})
            judge_dataset = judge_dataset.map(lambda x: {'cot_rating': int(match.group(0)) if (match := re.search(r"\d+", x['cot_rating'])) else None})
            pattern = re.compile(r'(?:Final Response Rating)', re.IGNORECASE | re.DOTALL)
            judge_dataset = judge_dataset.map(lambda x: {"final_response_rating": extract_content(x['response'], pattern)})
            judge_dataset = judge_dataset.map(lambda x: {'final_response_rating': int(match.group(0)) if (match := re.search(r"\d+", x['final_response_rating'])) else None})
            judge_dataset = judge_dataset.remove_columns(['judge_prompt', 'response', 'model'])
            col_renames = {
                'analysis': 'safe_analysis',
                'final_response': 'safe_final_response',
                'cot_rating': 'safe_cot_rating',
                'final_response_rating': 'safe_final_response_rating',
                'cot_response': 'safe_cot_response'
            }
            judge_dataset = judge_dataset.rename_columns(col_renames)
            records = [dict(row) for row in judge_dataset]
            with open(f'data/beavertails/{model_print_name}_unsafe_{number}_cot.json', 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=4)

    if args.task == 'beavertails_build_train_dataset':
        print(f'building cot dataset for beavertails')
        full_dataset = []
        print(f'cot_rating: {args.cot_rating}, final_response_rating: {args.final_response_rating}, class_num: {args.category_unsafe_samples}')
        for number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            class_dataset = datasets.load_dataset('json', data_files=f'cache/{model_print_name}_unsafe_{number}_cot.json', split='train')
            print(f'class_dataset {number} total: {len(class_dataset)}')
            class_dataset = class_dataset.filter(lambda x: x['unsafe_cot_rating'] is not None and x['unsafe_cot_rating'] >= args.cot_rating and x['unsafe_final_response_rating'] is not None and x['unsafe_final_response_rating'] <= args.final_response_rating)
            print(f'class_dataset {number} selected: {len(class_dataset)}')
            # if class_num > len(class_dataset):
            #     class_dataset = concatenate_datasets([class_dataset] * (class_num // len(class_dataset)), class_dataset.select(range(class_num % len(class_dataset))))
            # else:
            #     class_dataset = class_dataset.select(range(class_num))
            class_dataset = class_dataset.select(range(min(args.category_unsafe_samples, len(class_dataset))))
            class_dataset = class_dataset.remove_columns(['number'])
            full_dataset.extend(class_dataset)

        final_response_rating = 6 - args.final_response_rating
        safe_dataset = datasets.load_dataset('json', data_files=f'cache/{model_print_name}_safe_cot.json', split='train')
        safe_dataset = safe_dataset.filter(lambda x: x['safe_cot_rating'] is not None and x['safe_cot_rating'] >= args.cot_rating and x['safe_final_response_rating'] is not None and x['safe_final_response_rating'] >= final_response_rating)
        print(f'safe_dataset filtered: {len(safe_dataset)}')
        print(f'safe_dataset selected: {min(len(full_dataset) * args.ratio, len(safe_dataset))}')
        if len(safe_dataset) < len(full_dataset) * args.ratio:
            safe_dataset = concatenate_datasets([safe_dataset] * (len(full_dataset) * args.ratio // len(safe_dataset)), safe_dataset.select(range(len(full_dataset) * args.ratio % len(safe_dataset))))
        else:
            safe_dataset = safe_dataset.select(range(len(full_dataset) * args.ratio))
        safe_dataset = safe_dataset.select(range(min(len(full_dataset) * args.ratio, len(safe_dataset))))
        full_dataset.extend(safe_dataset)
        records = [dict(row) for row in full_dataset]
        with open(f'data/beavertails/{model_print_name}_full_cot_{args.cot_rating}_{args.final_response_rating}_{args.category_unsafe_samples}_{args.total_unsafe_samples}_{args.ratio}.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)

        full_dataset = datasets.load_dataset('json', data_files=f'data/beavertails/{model_print_name}_full_cot_{args.cot_rating}_{args.final_response_rating}_{args.category_unsafe_samples}_{args.total_unsafe_samples}_{args.ratio}.json', split='train')
        full_dataset = full_dataset.rename_columns({
            'safe_cot_response': 'helpful',
            'unsafe_cot_response': 'constrained'
        })
        unsafe_dataset = full_dataset.filter(lambda x: x['is_safe'] is not None and x['is_safe'] == False)
        unsafe_dataset = unsafe_dataset.shuffle(seed=1234)
        if len(unsafe_dataset) < args.total_unsafe_samples:
            repeated_datasets = [unsafe_dataset] * (args.total_unsafe_samples // len(unsafe_dataset))
            repeated_datasets.append(unsafe_dataset.select(range(args.total_unsafe_samples % len(unsafe_dataset))))
            unsafe_dataset = concatenate_datasets(repeated_datasets)
        else:
            unsafe_dataset = unsafe_dataset.select(range(args.total_unsafe_samples))
        safe_dataset = full_dataset.filter(lambda x: x['is_safe'] is not None and x['is_safe'] == True)
        
        dpo_unsafe_dataset = unsafe_dataset.map(lambda x: {"messages": [{"role": "user", "content": x['forbidden_prompt']}], "constrained_helpful_chosen": {"role": "assistant", "content": x['constrained']}, "constrained_answer_chosen": {"role": "assistant", "content": x['constrained']}, "constrained_refusal_chosen": {"role": "assistant", "content": x['constrained']}, "constrained_helpful_rejected": {"role": "assistant", "content": x['helpful']}, "constrained_answer_rejected": {"role": "assistant", "content": x['rejected']}, "constrained_refusal_rejected": {"role": "assistant", "content": "I am sorry, but I cannot comply with the request."}})
        dpo_unsafe_dataset = dpo_unsafe_dataset.remove_columns(['safe_analysis', 'safe_final_response', 'safe_cot_rating', 'safe_final_response_rating', 'unsafe_analysis', 'unsafe_final_response', 'unsafe_cot_rating', 'unsafe_final_response_rating', 'forbidden_prompt', 'category', 'constrained', 'helpful', 'rejected'])
        sft_unsafe_dataset = dpo_unsafe_dataset.map(lambda x: {"constrained_helpful": [x['messages'][0], x['constrained_helpful_chosen']], "constrained_answer": [x['messages'][0], x['constrained_answer_chosen']]})

        dpo_safe_dataset = safe_dataset.map(lambda x: {"messages": [{"role": "user", "content": x['forbidden_prompt']}], "constrained_helpful_chosen": {"role": "assistant", "content": x['helpful']}, "constrained_answer_chosen": {"role": "assistant", "content": x['rejected']}, "constrained_refusal_chosen": {"role": "assistant", "content": x['helpful']}, "constrained_helpful_rejected": {"role": "assistant", "content": x['constrained']}, "constrained_answer_rejected": {"role": "assistant", "content": x['constrained']}, "constrained_refusal_rejected": {"role": "assistant", "content": "I am sorry, but I cannot comply with the request."}})
        dpo_safe_dataset = dpo_safe_dataset.remove_columns(['safe_analysis', 'safe_final_response', 'safe_cot_rating', 'safe_final_response_rating', 'unsafe_analysis', 'unsafe_final_response', 'unsafe_cot_rating', 'unsafe_final_response_rating', 'forbidden_prompt', 'category', 'constrained', 'helpful', 'rejected'])
        sft_safe_dataset = dpo_safe_dataset.map(lambda x: {"constrained_helpful": [x['messages'][0], x['constrained_helpful_chosen']], "constrained_answer": [x['messages'][0], x['constrained_answer_chosen']]})

        dpo_full_dataset = concatenate_datasets([dpo_safe_dataset, dpo_unsafe_dataset])
        dpo_full_dataset = dpo_full_dataset.shuffle(seed=1234)
        sft_full_dataset = concatenate_datasets([sft_safe_dataset, sft_unsafe_dataset])
        sft_full_dataset = sft_full_dataset.remove_columns(['messages', 'constrained_helpful_chosen', 'constrained_answer_chosen', 'constrained_helpful_rejected', 'constrained_answer_rejected'])
        sft_full_dataset = sft_full_dataset.shuffle(seed=1234)
        
        # records = [dict(row) for row in sft_full_dataset]
        # with open(f'data/beavertails/sft_{model_print_name}_{cot_rating}_{final_response_rating}_{class_num}_{ratio}.json', 'w', encoding='utf-8') as f:
        #     json.dump(records, f, indent=4)
        # print(f'sft_full_dataset: {len(sft_full_dataset)}')
        records = [dict(row) for row in dpo_full_dataset]
        with open(f'data/beavertails/dpo_{model_print_name}_{args.cot_rating}_{args.final_response_rating}_{args.category_unsafe_samples}_{args.total_unsafe_samples}_{args.ratio}.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)
        print(f'dpo_full_dataset: {len(dpo_full_dataset)}')

        sft_dr_dataset = sft_full_dataset.map(lambda x: {"dr": [x['constrained_answer'][0], {"role": "assistant", "content": "I am sorry, but I cannot comply with the request."}]} if x['is_safe'] == False else {"dr": x['constrained_answer']})
        cot_instruction = specs.cot_instruction
        cot_prompt_unsafe_dataset = sft_dr_dataset.filter(lambda x: x['is_safe'] == False)
        cot_prompt_safe_dataset = sft_dr_dataset.filter(lambda x: x['is_safe'] == True)
        cot_prompt_safe_dataset = cot_prompt_safe_dataset.map(lambda x: {"sr": x['constrained_answer']})
        cot_prompt_unsafe_dataset = cot_prompt_unsafe_dataset.map(lambda x: {"cot_prompt": cot_instruction.format(prompt=x['constrained_answer'][0]['content'])})
        if args.model == 'gpt-4o-mini-2024-07-18' or args.model == 'gpt-4o-2024-11-20' or args.model == 'o1-2024-12-17':
            cot_prompt_unsafe_dataset = personalized_generate(cot_prompt_unsafe_dataset, system_prompt=None, target_column='cot_prompt', models=[args.model], use_local=False, decode_responses=False, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_length)
        else:
            cot_prompt_unsafe_dataset = cot_prompt_unsafe_dataset.map(lambda x: {"cot_prompt_formatted": tokenizer.apply_chat_template([{'role': 'user', 'content': x['cot_prompt']}], tokenize=False, add_generation_prompt=True)})
            responses_outputs = generate_model.generate(cot_prompt_unsafe_dataset['cot_prompt_formatted'], sampling_params)
            cot_prompt_unsafe_dataset = cot_prompt_unsafe_dataset.map(lambda x, idx: {"response": responses_outputs[idx].outputs[0].text}, with_indices=True)
        cot_prompt_unsafe_dataset = cot_prompt_unsafe_dataset.map(lambda x: {"sr": [x['constrained_answer'][0], {"role": "assistant", "content": x['response']}]})
        cot_prompt_unsafe_dataset = cot_prompt_unsafe_dataset.remove_columns(['cot_prompt', 'response'])
        sft_preliminary_dataset = concatenate_datasets([cot_prompt_safe_dataset, cot_prompt_unsafe_dataset])
        sft_preliminary_dataset = sft_preliminary_dataset.shuffle(seed=1234)
        print('sft_preliminary_dataset', len(sft_preliminary_dataset))
        records = [dict(row) for row in sft_preliminary_dataset]
        with open(f'data/beavertails/sft_preliminary_{model_print_name}_{args.cot_rating}_{args.final_response_rating}_{args.category_unsafe_samples}_{args.total_unsafe_samples}_{args.ratio}.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)

    if args.task == 'temp':
        dpo_full_dataset = datasets.load_dataset('json', data_files=f'data/beavertails/dpo_{model_print_name}_{args.cot_rating}_{args.final_response_rating}_{args.category_unsafe_samples}_{args.total_unsafe_samples}_{args.ratio}.json', split='train')
        dpo_full_dataset = dpo_full_dataset.select(range(250))
        print(f'dpo_full_dataset: {len(dpo_full_dataset)}')
        dpo_full_dataset = dpo_full_dataset.map(lambda x: {"constrained_refusal_chosen": {"role": "assistant", "content": x['constrained_helpful_chosen']['content']}, "constrained_refusal_rejected": {"role": "assistant", "content": "I am sorry, but I cannot comply with the request."}})
        records = [dict(row) for row in dpo_full_dataset]
        with open(f'data/beavertails/dpo_{model_print_name}_{args.cot_rating}_{args.final_response_rating}_{args.category_unsafe_samples}_{args.total_unsafe_samples}_{args.ratio}.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)
