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

import specifications as specs


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
            if model == 'ruliad/deepthought-8b-llama-v0.01-alpha':
                client = OpenAI(api_key="any-string-works", base_url="http://localhost:8001/v1")
            else:
                client = OpenAI(api_key="any-string-works", base_url="http://localhost:8000/v1")
            response = client.chat.completions.create(
                model=model,      # e.g. "your_local_model" or anything you've set
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
        parser.add_argument('--task', type=str, default='beavertails_cot', choices=['beavertails_classification', 'beavertails_cot'])
        args = parser.parse_args()
        return args

    args = parse_args()
    print(json.dumps(vars(args), indent=4))

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
        with open('data/beavertails/beavertails_safe.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)
        
        unsafe_dataset = dataset.filter(lambda x: x['is_safe'] == False)
        prompt_classification = specs.prompt_classification
        unsafe_prompt_dataset = unsafe_dataset.map(lambda x: {"prompt": prompt_classification.format(prompt=x['forbidden_prompt'])})
        # unsafe_prompt_dataset = unsafe_prompt_dataset.select(range(20))

        responses_dataset = personalized_generate(unsafe_prompt_dataset, system_prompt=None, models=['gpt-4o-2024-11-20'], use_local=False, decode_responses=False, temperature=0.7, top_p=0.95, max_tokens=128)
        processed_dataset = responses_dataset.map(
            lambda x: {
                'number': int(match.group(1)) if (match := re.match(r'(\d+)\.\s*(.+)', x['response'])) else None,
                'raw_category': match.group(2).strip() if match else None
            },
            num_proc=32
        )
        processed_dataset = processed_dataset.remove_columns('model')
        records = [dict(row) for row in processed_dataset]
        with open('data/beavertails/beavertails_unsafe_gpt-4o-2024-11-20_classification.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)

        classifications = {
            1: 'Violent Crimes',
            2: 'Non-Violent Crimes',
            3: 'Sex-Related Crimes',
            4: 'Child Sexual Exploitation',
            5: 'Indiscriminate Weapons, Chemical, Biological, Radiological, Nuclear, and high yield Explosives (CBRNE)',
            6: 'Suicide & Self-Harm',
            7: 'Hate',
            8: 'Privacy',
            9: 'Intellectual Property',
            10: 'Defamation',
            11: 'Elections',
            12: 'Sexual Content',
            13: 'Other'
        }

    if args.task == 'beavertails_cot':
        print(f'generating cot for beavertails')
        with open('data/beavertails/beavertails_safe.json', 'r', encoding='utf-8') as f:
            data = json.load(f)  # if it's a single JSON object/array
        print(len(data))  # Or see if it prints your data
        safe_dataset = datasets.load_dataset('json', data_files='data/beavertails/beavertails_safe.json', split='train')
        unsafe_dataset = datasets.load_dataset('json', data_files='data/beavertails/beavertails_unsafe.json', split='train')
        unmatch_unsafe_dataset = unsafe_dataset.filter(lambda x: x['number'] is None or x['number'] not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        records = [dict(row) for row in unmatch_unsafe_dataset]
        with open('data/beavertails/beavertails_unsafe_unmatch.json', 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=4)
        unsafe_dataset = unsafe_dataset.filter(lambda x: x['number'] is not None and 1 <= x['number'] <= 12)

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
            10: specs.defamation,
            11: specs.elections,
            12: specs.sexual_content
        }
        cot_generation = specs.cot_generation
        unsafe_dataset = unsafe_dataset.filter(lambda x: x['number'] == 1)

        cot_prompt_unsafe_dataset = unsafe_dataset.map(lambda x: {"cot_prompt": cot_generation.format(prompt=x['forbidden_prompt'], spec_category=specifications[x['number']])})
        # cot_dataset = personalized_generate(cot_prompt_unsafe_dataset, system_prompt=None, target_column='cot_prompt', models=['gpt-4o-2024-11-20'], use_local=False, decode_responses=False, temperature=0.7, top_p=0.95, max_tokens=2048)
        # cot_dataset = personalized_generate(cot_prompt_unsafe_dataset, system_prompt=None, target_column='cot_prompt', models=['o1-2024-12-17'], use_local=False, decode_responses=False, max_completion_tokens=2048)
        # cot_dataset = personalized_generate(cot_prompt_unsafe_dataset, system_prompt=None, target_column='cot_prompt', models=['meta-llama/Meta-Llama-3-8B-Instruct'], use_local=True, decode_responses=False, max_completion_tokens=4096)
        cot_dataset = personalized_generate(cot_prompt_unsafe_dataset, system_prompt=None, target_column='cot_prompt', models=['ruliad/deepthought-8b-llama-v0.01-alpha'], use_local=True, decode_responses=False, max_completion_tokens=4096)

        col_renames = {
            'prompt': 'forbidden_prompt',
            'category': 'category',
            'response': 'rejected',
            'is_safe': 'is_safe'
        }
        dataset = dataset.rename_columns(col_renames)
