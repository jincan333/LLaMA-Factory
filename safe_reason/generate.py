import os
import time
import warnings
from typing import Union
from openai import OpenAI
from datasets import Dataset, concatenate_datasets
import multiprocessing

# ---------------------------------------------------------------------
# Instead of referencing "real" OpenAI or other vendors,
# we point openai to our local endpoint.
# No special keys needed, but we do need to set openai.api_key
# to pass the internal openai python library checks
# ---------------------------------------------------------------------
client = OpenAI(api_key="any-string-works", base_url="http://localhost:8000/v1")

def generate(
    prompt: Union[str, list[str]],
    model: str,
    system_prompt: str = None,
    num_retries: int = 5,
    delay: int = 0,
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
        response = client.chat.completions.create(
            model=model,      # e.g. "your_local_model" or anything you've set
            messages=messages,
            **kwargs,         # e.g. temperature=0.7, top_p=0.95, etc.
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
            num_proc=multiprocessing.cpu_count(),
        )
        generated_dataset = generated_dataset.add_column("model", len(generated_dataset) * [model])
        generated_datasets.append(generated_dataset)

    dataset = concatenate_datasets(generated_datasets)
    if decode_responses:
        from strong_reject.jailbreaks import decode_dataset
        dataset = decode_dataset(dataset)

    return dataset