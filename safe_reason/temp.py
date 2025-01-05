import os
import multiprocessing
from functools import partial
from datasets import Dataset, concatenate_datasets
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from strong_reject.jailbreaks import decode_dataset


def max_tokens(model):
    model = model.lower()
    if any(x in model for x in ['gemma', 'llama']):
        return 8192
    elif any(x in model for x in ['qwen', 'mistral']):
        return 32768
    else:
        raise ValueError(f"Model {model} not supported")


def generate(prompt, model, system_prompt=None, num_retries=3, delay=0, **kwargs):
    # Example single-call generator that uses vLLM on exactly one GPU per process
    from time import sleep
    import warnings

    if num_retries == 0:
        msg = f"Failed to get response from model {model} for prompt {prompt}"
        warnings.warn(msg)
        return ""

    if delay > 0:
        sleep(delay)

    # Build messages
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})

    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    else:
        for i, content in enumerate(prompt):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": content})

    try:
        max_length = max_tokens(model)
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.model_max_length = max_length

        # Use just one GPU per process (tensor_parallel_size=1)
        llm = LLM(model=model, max_num_seqs=1, tensor_parallel_size=1)
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=max_length,
        )

        messages_formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        return llm.generate(messages_formatted, sampling_params)[0].outputs[0].text

    except Exception as e:
        print(f"Error: {e}")
        return generate(
            prompt,
            model,
            system_prompt=system_prompt,
            num_retries=num_retries - 1,
            delay=max(2 * delay, 1),
            **kwargs,
        )


def _assign_gpu_and_generate(example, index, model, prompt_col, num_proc, **kwargs):
    # Determine which worker this is. index can be large, so mod by #GPUs.
    gpu_id = index % num_proc  # use 4 GPUs (0..3)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Now call your generate function
    prompt = example[prompt_col]
    response = generate(prompt, model, **kwargs)
    return {"response": response}


def personalized_generate(
    dataset: Dataset,
    models: list[str],
    target_column: str = "prompt",
    decode_responses: bool = True,
    num_proc: int = 4,
    **kwargs,
) -> Dataset:
    from datasets import concatenate_datasets

    generated_datasets = []
    for model in models:
        # We partial‑apply the map function so we can pass extra parameters easily
        map_fn = partial(_assign_gpu_and_generate, model=model, prompt_col=target_column, num_proc=num_proc, **kwargs)

        # Spawn map with n processes.  with_indices=True -> pass the (row, index).
        generated_dataset = dataset.map(
            map_fn,
            with_indices=True,
            num_proc=num_proc,
        )

        generated_dataset = generated_dataset.add_column("model", [model] * len(generated_dataset))
        generated_datasets.append(generated_dataset)

    # Concatenate all model generations
    final_dataset = concatenate_datasets(generated_datasets)

    if decode_responses:
        # Example: decode for jailbreaking / filtering
        from .jailbreaks import decode_dataset
        final_dataset = decode_dataset(final_dataset)

    return final_dataset