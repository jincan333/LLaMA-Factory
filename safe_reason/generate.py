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


def extract_content(response, pattern):
    matches = list(pattern.finditer(response))
    if not matches:
        final_response = response
    else:
        final_response = re.sub(r'^\W+|\W+$', '', response[matches[-1].end():])
    return final_response


def extract_content_first(response, pattern):
    matches = list(pattern.finditer(response))
    if not matches:
        final_response = response
    else:
        final_response = re.sub(r'^\W+|\W+$', '', response[matches[-1].start(): matches[-1].end()])
    final_response = re.sub(r'^\W+|\W+$', '', final_response.strip('Analysis'))
    final_response = re.sub(r'^\W+|\W+$', '', final_response.strip('Final Response'))
    final_response = re.sub(r'^\W+|\W+$', '', final_response.strip('Chain of Thought Rating'))
    final_response = re.sub(r'^\W+|\W+$', '', final_response.strip('Final Response Rating'))
    return final_response


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
    elif model_name in ('qwq-32b-preview', 'Qwen/QwQ-32B-Preview'):
        model_name = 'Qwen/QwQ-32B-Preview'
    elif model_name in ('deepthought-8b', 'ruliad/deepthought-8b-llama-v0.01-alpha'):
        model_name = 'ruliad/deepthought-8b-llama-v0.01-alpha'
    else:
        model_name = os.path.join(project_path, 'safe_reason', model_name)
    model_print_name = re.sub(r'/', '-', model_name)

    return model_name, model_print_name


def generate(
    prompt: Union[str, list[str]],
    model: str,
    system_prompt: str = None,
    num_retries: int = 5,
    delay: int = 0,
    use_local: bool = False,
    **kwargs,
) -> str:
    """Call our local ChatCompletion endpoint (OpenAI-style)."""

    if num_retries == 0:
        msg = f"Failed to get response from local model {model} for prompt {prompt}"
        warnings.warn(msg)
        return ""

    if delay > 0:
        time.sleep(delay)

    # Build OpenAI-style "messages"
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})

    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    else:
        # prompt is a list of alternating user and assistant messages
        for i, content in enumerate(prompt):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": content})

    try:
        # Use openai.ChatCompletion
        if use_local:
            if 'llama' in model:
                client = OpenAI(api_key="any-string-works", base_url="http://localhost:8000/v1")
            elif 'deepthought' in model:
                client = OpenAI(api_key="any-string-works", base_url="http://localhost:8001/v1")
            elif 'gemma' in model:
                client = OpenAI(api_key="any-string-works", base_url="http://localhost:8002/v1")
            else:
                raise ValueError(f"Model {model} not supported")
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,         # e.g. temperature=0.7, top_p=0.95, etc.
            )
        else:
            with openai.OpenAI() as client:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs,
                )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return generate(
            prompt,
            model,
            system_prompt=system_prompt,
            num_retries=num_retries - 1,
            delay=2 * max(delay, 1),
            **kwargs,
        )


def personalized_generate(
    dataset: Dataset,
    models: list[str],
    target_column: str = "prompt",
    decode_responses: bool = True,
    **kwargs,
) -> Dataset:
    """Generate responses to a Hugging Face dataset of prompts.

    We just call our local openai-style endpoint for each prompt.
    """
    generated_datasets = []
    
    for model in models:
        generated_dataset = dataset.map(
            lambda x: {"response": generate(x[target_column], model, **kwargs)},
            num_proc=32
        )
        generated_dataset = generated_dataset.add_column("model", len(generated_dataset) * [model])
        generated_datasets.append(generated_dataset)

    dataset = concatenate_datasets(generated_datasets)
    if decode_responses:
        from strong_reject.jailbreaks import decode_dataset
        dataset = decode_dataset(dataset)

    return dataset


if __name__ == "__main__":  

    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--task', type=str, default='strongreject_classification', choices=['beavertails_classification', 'beavertails_generate_cot_safe', 'beavertails_generate_cot_unsafe', 'beavertails_build_cot_dataset', 'test'])
        parser.add_argument('--model', type=str, default='llama-3-8b-instruct', help='base models are gpt-4o-mini-2024-07-18, gpt-4o-2024-11-20, llama-3-8b-instruct, gemma-2-9b-it, qwq-32b-preview, deepthought-8b, o1-2024-12-17')
        args = parser.parse_args()
        return args

    args = parse_args()
    args.model, model_print_name = get_model(args.model)
    print(json.dumps(vars(args), indent=4))

    if args.task == 'strongreject_classification':
        print(f'generating classification for strongreject')
        # dataset = load_strongreject()
        # prompt_classification = specs.prompt_classification
        # dataset = dataset.map(lambda x: {"prompt": prompt_classification.format(prompt=x['forbidden_prompt'])})
        # responses_dataset = personalized_generate(dataset, system_prompt=None, models=['gpt-4o-2024-11-20'], use_local=False, decode_responses=False, temperature=0, max_tokens=512)
        # pattern = re.compile(r'(?:final category number)', re.IGNORECASE | re.DOTALL)
        # responses_dataset = responses_dataset.map(lambda x: {"final_response": extract_content(x['response'], pattern)})
        # processed_dataset = responses_dataset.map(
        #     lambda x: {
        #         'number': int(match.group(0)) if (match := re.search(r"\d+", x['final_response'])) else None,
        #     },
        #     num_proc=32
        # )
        # processed_dataset = processed_dataset.remove_columns('model')
        # records = [dict(row) for row in processed_dataset]
        # with open('data/strongreject/classification.json', 'w', encoding='utf-8') as f:
        #     json.dump(records, f, indent=4)

        processed_dataset = datasets.load_dataset('json', data_files='data/strongreject/classification.json', split='train')

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

        responses_dataset = personalized_generate(unsafe_prompt_dataset, system_prompt=None, models=['gpt-4o-2024-11-20'], use_local=False, decode_responses=False, temperature=0, top_p=1, max_tokens=512)
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
        with open('data/beavertails/unsafe_gpt-4o-2024-11-20_classification.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)

    if args.task == 'beavertails_generate_cot_safe':
        print(f'generating cot for beavertails safe')
        safe_dataset = datasets.load_dataset('json', data_files='data/beavertails/safe.json', split='train')
        safe_dataset = safe_dataset.shuffle(seed=1234)
        safe_dataset = safe_dataset.select(range(min(len(safe_dataset), 5000)))
        print('safe', len(safe_dataset))

        overall_helpful = specs.overall_helpful
        cot_specification_helpful = specs.cot_specification_helpful
        cot_prompt_safe_dataset = safe_dataset.map(lambda x: {"cot_prompt": cot_specification_helpful.format(prompt=x['forbidden_prompt'], spec_category=overall_helpful)})
        cot_dataset = personalized_generate(cot_prompt_safe_dataset, system_prompt=None, target_column='cot_prompt', models=[args.model], use_local=True, decode_responses=False, temperature=0, max_tokens=4096)
        pattern = re.compile(r'# Analysis\s*(.*?)\s*# Final Response', re.IGNORECASE | re.DOTALL)
        cot_dataset = cot_dataset.map(lambda x: {"analysis": extract_content_first(x['response'], pattern)})
        pattern = re.compile(r'(?:# Final Response)', re.IGNORECASE | re.DOTALL)
        cot_dataset = cot_dataset.map(lambda x: {"final_response": extract_content(x['response'], pattern)})
        cot_dataset = cot_dataset.remove_columns(['model', 'cot_prompt'])
        records = [dict(row) for row in cot_dataset]
        with open(f'data/beavertails/{model_print_name}_safe_cot.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)

        # filter cot_dataset
        cot_dataset = cot_dataset.rename_column('response', 'cot_response')
        reward_judge_helpful = specs.reward_judge_helpful
        judge_dataset = cot_dataset.map(lambda x: {"judge_prompt": reward_judge_helpful.format(prompt=x['forbidden_prompt'], cot=x['analysis'], final_response=x['final_response'], spec_category=overall_helpful)})
        judge_dataset = personalized_generate(judge_dataset, system_prompt=None, target_column='judge_prompt', models=['gpt-4o-2024-11-20'], use_local=False, decode_responses=False, temperature=0.95, top_p=0.7, max_tokens=4096)
        pattern = re.compile(r'(?:# Chain of Thought Rating\s*(.*?)\s*# Final Response Rating)', re.IGNORECASE | re.DOTALL)
        judge_dataset = judge_dataset.map(lambda x: {"cot_rating": extract_content_first(x['response'], pattern)})
        judge_dataset = judge_dataset.map(lambda x: {'cot_rating': int(match.group(0)) if (match := re.search(r"\d+", x['cot_rating'])) else None})
        pattern = re.compile(r'(?:# Final Response Rating)', re.IGNORECASE | re.DOTALL)
        judge_dataset = judge_dataset.map(lambda x: {"final_response_rating": extract_content(x['response'], pattern)})
        judge_dataset = judge_dataset.map(lambda x: {'final_response_rating': int(match.group(0)) if (match := re.search(r"\d+", x['final_response_rating'])) else None})
        judge_dataset = judge_dataset.remove_columns(['model', 'judge_prompt', 'cot_response', 'response'])
        col_renames = {
            'analysis': 'safe_analysis',
            'final_response': 'safe_final_response',
            'cot_rating': 'safe_cot_rating',
            'final_response_rating': 'safe_final_response_rating'
        }
        judge_dataset = judge_dataset.rename_columns(col_renames)
        records = [dict(row) for row in judge_dataset]
        with open(f'data/beavertails/{model_print_name}_safe_cot.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)


        overall = specs.overall
        cot_specification = specs.cot_specification
        cot_prompt_safe_dataset = judge_dataset.map(lambda x: {"cot_prompt": cot_specification.format(prompt=x['forbidden_prompt'], spec_category=overall)})
        cot_dataset = personalized_generate(cot_prompt_safe_dataset, system_prompt=None, target_column='cot_prompt', models=[args.model], use_local=True, decode_responses=False, temperature=0, max_tokens=4096)
        pattern = re.compile(r'# Analysis\s*(.*?)\s*# Final Response', re.IGNORECASE | re.DOTALL)
        cot_dataset = cot_dataset.map(lambda x: {"analysis": extract_content_first(x['response'], pattern)})
        pattern = re.compile(r'(?:# Final Response)', re.IGNORECASE | re.DOTALL)
        cot_dataset = cot_dataset.map(lambda x: {"final_response": extract_content(x['response'], pattern)})
        cot_dataset = cot_dataset.remove_columns(['model', 'cot_prompt'])
        records = [dict(row) for row in cot_dataset]
        with open(f'data/beavertails/{model_print_name}_safe_cot.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)

        # filter cot_dataset
        cot_dataset = cot_dataset.rename_column('response', 'cot_response')
        reward_judge = specs.reward_judge
        judge_dataset = cot_dataset.map(lambda x: {"judge_prompt": reward_judge.format(prompt=x['forbidden_prompt'], cot=x['analysis'], final_response=x['final_response'], spec_category=overall)})
        judge_dataset = personalized_generate(judge_dataset, system_prompt=None, target_column='judge_prompt', models=['gpt-4o-2024-11-20'], use_local=False, decode_responses=False, temperature=0.95, top_p=0.7, max_tokens=4096)
        pattern = re.compile(r'(?:# Chain of Thought Rating\s*(.*?)\s*# Final Response Rating)', re.IGNORECASE | re.DOTALL)
        judge_dataset = judge_dataset.map(lambda x: {"cot_rating": extract_content_first(x['response'], pattern)})
        judge_dataset = judge_dataset.map(lambda x: {'cot_rating': int(match.group(0)) if (match := re.search(r"\d+", x['cot_rating'])) else None})
        pattern = re.compile(r'(?:# Final Response Rating)', re.IGNORECASE | re.DOTALL)
        judge_dataset = judge_dataset.map(lambda x: {"final_response_rating": extract_content(x['response'], pattern)})
        judge_dataset = judge_dataset.map(lambda x: {'final_response_rating': int(match.group(0)) if (match := re.search(r"\d+", x['final_response_rating'])) else None})
        judge_dataset = judge_dataset.remove_columns(['model', 'judge_prompt', 'cot_response', 'response'])
        col_renames = {
            'analysis': 'unsafe_analysis',
            'final_response': 'unsafe_final_response',
            'cot_rating': 'unsafe_cot_rating',
            'final_response_rating': 'unsafe_final_response_rating'
        }
        judge_dataset = judge_dataset.rename_columns(col_renames)
        records = [dict(row) for row in judge_dataset]
        with open(f'data/beavertails/{model_print_name}_safe_cot.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)

    if args.task == 'beavertails_generate_cot_unsafe':
        print(f'generating cot for beavertails unsafe')
        # unsafe_dataset = datasets.load_dataset('json', data_files='cache/unsafe_gpt-4o-2024-11-20_classification.json', split='train')
        # unsafe_dataset = unsafe_dataset.remove_columns(['response', 'final_response', 'prompt'])
        # unmatch_unsafe_dataset = unsafe_dataset.filter(lambda x: x['number'] is None or x['number'] not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        # match_unsafe_dataset = unsafe_dataset.filter(lambda x: x['number'] is not None and 1 <= x['number'] <= 12)
        # records = [dict(row) for row in unmatch_unsafe_dataset]
        # with open('data/beavertails/unsafe_unmatch.json', 'w', encoding='utf-8') as f:
        #     json.dump(records, f, indent=4)
        # records = [dict(row) for row in match_unsafe_dataset]
        # with open('data/beavertails/unsafe_match.json', 'w', encoding='utf-8') as f:
        #     json.dump(records, f, indent=4)
        unsafe_dataset = datasets.load_dataset('json', data_files='data/beavertails/unsafe_match.json', split='train')
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
            12: specs.sexual_content
        }
        
        for number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            filtered_dataset = unsafe_dataset.filter(lambda x: x['number'] == number)
            filtered_dataset = filtered_dataset.shuffle(seed=1234)
            filtered_dataset = filtered_dataset.select(range(min(len(filtered_dataset), 500)))
            print('unsafe', number, len(filtered_dataset))
            cot_specification = specs.cot_specification
            cot_prompt_unsafe_dataset = filtered_dataset.map(lambda x: {"cot_prompt": cot_specification.format(prompt=x['forbidden_prompt'], spec_category=specifications[x['number']])})
            # cot_dataset = personalized_generate(cot_prompt_unsafe_dataset, system_prompt=None, target_column='cot_prompt', models=['gpt-4o-2024-11-20'], use_local=False, decode_responses=False, temperature=0.7, top_p=0.95, max_tokens=4096)
            # cot_dataset = personalized_generate(cot_prompt_unsafe_dataset, system_prompt=None, target_column='cot_prompt', models=['o1-2024-12-17'], use_local=False, decode_responses=False, max_completion_tokens=4096)
            cot_dataset = personalized_generate(cot_prompt_unsafe_dataset, system_prompt=None, target_column='cot_prompt', models=[args.model], use_local=True, decode_responses=False, temperature=0, max_tokens=4096)
            
            # filter cot_dataset
            pattern = re.compile(r'# Analysis\s*(.*?)\s*# Final Response', re.IGNORECASE | re.DOTALL)
            cot_dataset = cot_dataset.map(lambda x: {"analysis": extract_content_first(x['response'], pattern)})
            pattern = re.compile(r'(?:# Final Response)', re.IGNORECASE | re.DOTALL)
            cot_dataset = cot_dataset.map(lambda x: {"final_response": extract_content(x['response'], pattern)})
            cot_dataset = cot_dataset.remove_columns(['model', 'cot_prompt'])
            cot_dataset = cot_dataset.rename_column('response', 'cot_response')
            reward_judge = specs.reward_judge
            judge_dataset = cot_dataset.map(lambda x: {"judge_prompt": reward_judge.format(prompt=x['forbidden_prompt'], cot=x['analysis'], final_response=x['final_response'], spec_category=specifications[x['number']])})
            judge_dataset = personalized_generate(judge_dataset, system_prompt=None, target_column='judge_prompt', models=['gpt-4o-2024-11-20'], use_local=False, decode_responses=False, temperature=0.95, top_p=0.7, max_tokens=4096)
            pattern = re.compile(r'(?:# Chain of Thought Rating\s*(.*?)\s*# Final Response Rating)', re.IGNORECASE | re.DOTALL)
            judge_dataset = judge_dataset.map(lambda x: {"cot_rating": extract_content_first(x['response'], pattern)})
            judge_dataset = judge_dataset.map(lambda x: {'cot_rating': int(match.group(0)) if (match := re.search(r"\d+", x['cot_rating'])) else None})
            pattern = re.compile(r'(?:# Final Response Rating)', re.IGNORECASE | re.DOTALL)
            judge_dataset = judge_dataset.map(lambda x: {"final_response_rating": extract_content(x['response'], pattern)})
            judge_dataset = judge_dataset.map(lambda x: {'final_response_rating': int(match.group(0)) if (match := re.search(r"\d+", x['final_response_rating'])) else None})
            judge_dataset = judge_dataset.remove_columns(['model', 'judge_prompt', 'cot_response', 'response'])
            col_renames = {
                'analysis': 'unsafe_analysis',
                'final_response': 'unsafe_final_response',
                'cot_rating': 'unsafe_cot_rating',
                'final_response_rating': 'unsafe_final_response_rating'
            }
            judge_dataset = judge_dataset.rename_columns(col_renames)
            records = [dict(row) for row in judge_dataset]
            with open(f'data/beavertails/{model_print_name}_unsafe_{number}_cot.json', 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=4)

            cot_specification_helpful = specs.cot_specification_helpful
            overall_helpful = specs.overall_helpful
            cot_prompt_unsafe_dataset = judge_dataset.map(lambda x: {"cot_prompt": cot_specification_helpful.format(prompt=x['forbidden_prompt'], spec_category=overall_helpful)})
            cot_dataset = personalized_generate(cot_prompt_unsafe_dataset, system_prompt=None, target_column='cot_prompt', models=[args.model], use_local=True, decode_responses=False, temperature=0, max_tokens=4096)
            pattern = re.compile(r'# Analysis\s*(.*?)\s*# Final Response', re.IGNORECASE | re.DOTALL)
            cot_dataset = cot_dataset.map(lambda x: {"analysis": extract_content_first(x['response'], pattern)})
            pattern = re.compile(r'(?:# Final Response)', re.IGNORECASE | re.DOTALL)
            cot_dataset = cot_dataset.map(lambda x: {"final_response": extract_content(x['response'], pattern)})
            cot_dataset = cot_dataset.remove_columns(['model', 'cot_prompt'])
            cot_dataset = cot_dataset.rename_column('response', 'cot_response')
            records = [dict(row) for row in cot_dataset]
            with open(f'data/beavertails/{model_print_name}_unsafe_{number}_cot.json', 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=4)

            reward_judge_helpful = specs.reward_judge_helpful
            judge_dataset = cot_dataset.map(lambda x: {"judge_prompt": reward_judge_helpful.format(prompt=x['forbidden_prompt'], cot=x['analysis'], final_response=x['final_response'], spec_category=overall_helpful)})
            judge_dataset = personalized_generate(judge_dataset, system_prompt=None, target_column='judge_prompt', models=['gpt-4o-2024-11-20'], use_local=False, decode_responses=False, temperature=0.95, top_p=0.7, max_tokens=4096)
            pattern = re.compile(r'(?:# Chain of Thought Rating\s*(.*?)\s*# Final Response Rating)', re.IGNORECASE | re.DOTALL)
            judge_dataset = judge_dataset.map(lambda x: {"cot_rating": extract_content_first(x['response'], pattern)})
            judge_dataset = judge_dataset.map(lambda x: {'cot_rating': int(match.group(0)) if (match := re.search(r"\d+", x['cot_rating'])) else None})
            pattern = re.compile(r'(?:# Final Response Rating)', re.IGNORECASE | re.DOTALL)
            judge_dataset = judge_dataset.map(lambda x: {"final_response_rating": extract_content(x['response'], pattern)})
            judge_dataset = judge_dataset.map(lambda x: {'final_response_rating': int(match.group(0)) if (match := re.search(r"\d+", x['final_response_rating'])) else None})
            judge_dataset = judge_dataset.remove_columns(['model', 'judge_prompt', 'cot_response', 'response'])
            col_renames = {
                'analysis': 'safe_analysis',
                'final_response': 'safe_final_response',
                'cot_rating': 'safe_cot_rating',
                'final_response_rating': 'safe_final_response_rating'
            }
            judge_dataset = judge_dataset.rename_columns(col_renames)
            records = [dict(row) for row in judge_dataset]
            with open(f'data/beavertails/{model_print_name}_unsafe_{number}_cot.json', 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=4)

    if args.task == 'beavertails_build_train_dataset':
        print(f'building cot dataset for beavertails')
        cot_ratings = [4]
        final_response_ratings = [2]
        class_nums = [50]
        ratio = 4
        for cot_rating in cot_ratings:
            for final_response_rating in final_response_ratings:
                for class_num in class_nums:
                    full_dataset = []
                    print(f'cot_rating: {cot_rating}, final_response_rating: {final_response_rating}, class_num: {class_num}')
                    for number in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                        class_dataset = datasets.load_dataset('json', data_files=f'cache/{model_print_name}_unsafe_{number}_cot.json', split='train')
                        print(f'class_dataset {number} total: {len(class_dataset)}')
                        class_dataset = class_dataset.filter(lambda x: x['unsafe_cot_rating'] is not None and x['unsafe_cot_rating'] >= cot_rating and x['unsafe_final_response_rating'] is not None and x['unsafe_final_response_rating'] <= final_response_rating)
                        print(f'class_dataset {number} selected: {min(class_num, len(class_dataset))}')
                        class_dataset = class_dataset.select(range(min(class_num, len(class_dataset))))
                        class_dataset = class_dataset.remove_columns(['number'])
                        full_dataset.extend(class_dataset)

                final_response_rating = 6 - final_response_rating
                safe_dataset = datasets.load_dataset('json', data_files=f'cache/{model_print_name}_safe_cot.json', split='train')
                print(f'safe_dataset total: {len(safe_dataset)}')
                safe_dataset = safe_dataset.filter(lambda x: x['safe_cot_rating'] is not None and x['safe_cot_rating'] >= cot_rating and x['safe_final_response_rating'] is not None and x['safe_final_response_rating'] >= final_response_rating)
                print(f'safe_dataset selected: {min(len(full_dataset) * ratio, len(safe_dataset))}')
                safe_dataset = safe_dataset.select(range(min(len(full_dataset) * ratio, len(safe_dataset))))
                full_dataset.extend(safe_dataset)
                records = [dict(row) for row in full_dataset]
                with open(f'data/beavertails/{model_print_name}_full_cot_{cot_rating}_{final_response_rating}_{class_num}_{ratio}.json', 'w', encoding='utf-8') as f:
                    json.dump(records, f, indent=4)

                full_dataset = datasets.load_dataset('json', data_files=f'data/beavertails/{model_print_name}_full_cot_{cot_rating}_{final_response_rating}_{class_num}_{ratio}.json', split='train')
                full_dataset = full_dataset.map(lambda x: {"helpful": "### Analysis\n\n{}\n\n### Final Response\n{}".format(x['safe_analysis'], x['safe_final_response'])})
                full_dataset = full_dataset.map(lambda x: {"constrained": "### Analysis\n\n{}\n\n### Final Response\n{}".format(x['unsafe_analysis'], x['unsafe_final_response'])})
                unsafe_dataset = full_dataset.filter(lambda x: x['is_safe'] is not None and x['is_safe'] == False)
                safe_dataset = full_dataset.filter(lambda x: x['is_safe'] is not None and x['is_safe'] == True)
                
                dpo_unsafe_dataset = unsafe_dataset.map(lambda x: {"messages": [{"role": "user", "content": x['forbidden_prompt']}], "constrained_helpful_chosen": {"role": "assistant", "content": x['constrained']}, "constrained_answer_chosen": {"role": "assistant", "content": x['constrained']}, "constrained_helpful_rejected": {"role": "assistant", "content": x['helpful']}, "constrained_answer_rejected": {"role": "assistant", "content": x['rejected']}})
                dpo_unsafe_dataset = dpo_unsafe_dataset.remove_columns(['safe_analysis', 'safe_final_response', 'safe_cot_rating', 'safe_final_response_rating', 'unsafe_analysis', 'unsafe_final_response', 'unsafe_cot_rating', 'unsafe_final_response_rating', 'forbidden_prompt', 'category', 'constrained', 'helpful', 'rejected'])
                sft_unsafe_dataset = dpo_unsafe_dataset.map(lambda x: {"constrained_helpful": [x['messages'][0], x['constrained_helpful_chosen']], "constrained_answer": [x['messages'][0], x['constrained_answer_chosen']]})

                dpo_safe_dataset = safe_dataset.map(lambda x: {"messages": [{"role": "user", "content": x['forbidden_prompt']}], "constrained_helpful_chosen": {"role": "assistant", "content": x['helpful']}, "constrained_answer_chosen": {"role": "assistant", "content": x['rejected']}, "constrained_helpful_rejected": {"role": "assistant", "content": x['constrained']}, "constrained_answer_rejected": {"role": "assistant", "content": x['constrained']}})
                dpo_safe_dataset = dpo_safe_dataset.remove_columns(['safe_analysis', 'safe_final_response', 'safe_cot_rating', 'safe_final_response_rating', 'unsafe_analysis', 'unsafe_final_response', 'unsafe_cot_rating', 'unsafe_final_response_rating', 'forbidden_prompt', 'category', 'constrained', 'helpful', 'rejected'])
                sft_safe_dataset = dpo_safe_dataset.map(lambda x: {"constrained_helpful": [x['messages'][0], x['constrained_helpful_chosen']], "constrained_answer": [x['messages'][0], x['constrained_answer_chosen']]})

                dpo_full_dataset = concatenate_datasets([dpo_safe_dataset, dpo_unsafe_dataset])
                dpo_full_dataset = dpo_full_dataset.shuffle(seed=1234)
                sft_full_dataset = concatenate_datasets([sft_safe_dataset, sft_unsafe_dataset])
                sft_full_dataset = sft_full_dataset.remove_columns(['messages', 'constrained_helpful_chosen', 'constrained_answer_chosen', 'constrained_helpful_rejected', 'constrained_answer_rejected'])
                sft_full_dataset = sft_full_dataset.shuffle(seed=1234)
                
                records = [dict(row) for row in sft_full_dataset]
                with open(f'data/beavertails/sft_{model_print_name}_{cot_rating}_{final_response_rating}_{class_num}_{ratio}.json', 'w', encoding='utf-8') as f:
                    json.dump(records, f, indent=4)
                records = [dict(row) for row in dpo_full_dataset]
                with open(f'data/beavertails/dpo_{model_print_name}_{cot_rating}_{final_response_rating}_{class_num}_{ratio}.json', 'w', encoding='utf-8') as f:
                    json.dump(records, f, indent=4)
                print(f'sft_full_dataset: {len(sft_full_dataset)}')
                print(f'dpo_full_dataset: {len(dpo_full_dataset)}')

    if args.task == 'test':
        print(f'testing')
        safe_dataset = datasets.load_dataset('json', data_files='cache/meta-llama-Meta-Llama-3-8B-Instruct_safe_cot.json', split='train')
        print(len(safe_dataset))
        unsafe_dataset = datasets.load_dataset('PKU-Alignment/BeaverTails', split='30k_train')
        unsafe_dataset = unsafe_dataset.filter(lambda x: x['is_safe'] == False)
        safe_dataset = safe_dataset.filter(lambda x: x['forbidden_prompt'] not in unsafe_dataset['prompt'])
        print(len(safe_dataset))
        records = [dict(row) for row in safe_dataset]
        with open('cache/meta-llama-Meta-Llama-3-8B-Instruct_safe_cot_filtered.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)
