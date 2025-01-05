# import os
# import multiprocessing
# from functools import partial
# from datasets import Dataset, concatenate_datasets
# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer
# from strong_reject.jailbreaks import decode_dataset


# def max_tokens(model):
#     model = model.lower()
#     if any(x in model for x in ['gemma', 'llama']):
#         return 8192
#     elif any(x in model for x in ['qwen', 'mistral']):
#         return 32768
#     else:
#         raise ValueError(f"Model {model} not supported")


# def generate(prompt, model, system_prompt=None, num_retries=3, delay=0, **kwargs):
#     # Example single-call generator that uses vLLM on exactly one GPU per process
#     from time import sleep
#     import warnings

#     if num_retries == 0:
#         msg = f"Failed to get response from model {model} for prompt {prompt}"
#         warnings.warn(msg)
#         return ""

#     if delay > 0:
#         sleep(delay)

#     # Build messages
#     messages = []
#     if system_prompt is not None:
#         messages.append({"role": "system", "content": system_prompt})

#     if isinstance(prompt, str):
#         messages.append({"role": "user", "content": prompt})
#     else:
#         for i, content in enumerate(prompt):
#             role = "user" if i % 2 == 0 else "assistant"
#             messages.append({"role": role, "content": content})

#     try:
#         max_length = max_tokens(model)
#         tokenizer = AutoTokenizer.from_pretrained(model)
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.model_max_length = max_length

#         # Use just one GPU per process (tensor_parallel_size=1)
#         llm = LLM(model=model, max_num_seqs=1, tensor_parallel_size=1)
#         sampling_params = SamplingParams(
#             temperature=0.7,
#             top_p=0.95,
#             max_tokens=max_length,
#         )

#         messages_formatted = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#         )

#         return llm.generate(messages_formatted, sampling_params)[0].outputs[0].text

#     except Exception as e:
#         print(f"Error: {e}")
#         return generate(
#             prompt,
#             model,
#             system_prompt=system_prompt,
#             num_retries=num_retries - 1,
#             delay=max(2 * delay, 1),
#             **kwargs,
#         )


# def _assign_gpu_and_generate(example, index, model, prompt_col, num_proc, **kwargs):
#     # Determine which worker this is. index can be large, so mod by #GPUs.
#     gpu_id = index % num_proc  # use 4 GPUs (0..3)
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

#     # Now call your generate function
#     prompt = example[prompt_col]
#     response = generate(prompt, model, **kwargs)
#     return {"response": response}


# def personalized_generate(
#     dataset: Dataset,
#     models: list[str],
#     target_column: str = "prompt",
#     decode_responses: bool = True,
#     num_proc: int = 4,
#     **kwargs,
# ) -> Dataset:
#     from datasets import concatenate_datasets

#     generated_datasets = []
#     for model in models:
#         # We partial‑apply the map function so we can pass extra parameters easily
#         map_fn = partial(_assign_gpu_and_generate, model=model, prompt_col=target_column, num_proc=num_proc, **kwargs)

#         # Spawn map with n processes.  with_indices=True -> pass the (row, index).
#         generated_dataset = dataset.map(
#             map_fn,
#             with_indices=True,
#             num_proc=num_proc,
#         )

#         generated_dataset = generated_dataset.add_column("model", [model] * len(generated_dataset))
#         generated_datasets.append(generated_dataset)

#     # Concatenate all model generations
#     final_dataset = concatenate_datasets(generated_datasets)

#     if decode_responses:
#         # Example: decode for jailbreaking / filtering
#         from .jailbreaks import decode_dataset
#         final_dataset = decode_dataset(final_dataset)

#     return final_dataset






# import os
# import multiprocessing
# from functools import partial
# from datasets import Dataset, concatenate_datasets
# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer

# # ---------------------------------------------------------------------
# # Global caches (per process). Once loaded, we reuse the same objects.
# # Each Python process has its own memory, so we won't collide across workers.
# # ---------------------------------------------------------------------
# LLM_CACHE = {}
# TOKENIZER_CACHE = {}

# def max_tokens(model):
#     model = model.lower()
#     if any(x in model for x in ['gemma', 'llama']):
#         return 8192
#     elif any(x in model for x in ['qwen', 'mistral']):
#         return 32768
#     else:
#         raise ValueError(f"Model {model} not supported")


# def get_llm_and_tokenizer(model):
#     """
#     For the current process, return the (LLM, tokenizer) for the given model.
#     If not yet cached, load them once and store them in LLM_CACHE/TOKENIZER_CACHE.
#     """
#     if model in LLM_CACHE and model in TOKENIZER_CACHE:
#         return LLM_CACHE[model], TOKENIZER_CACHE[model]

#     # Load model/tokenizer once, store in globals
#     max_length = max_tokens(model)
#     tokenizer = AutoTokenizer.from_pretrained(model)
#     tokenizer.pad_token = tokenizer.eos_token
#     tokenizer.model_max_length = max_length

#     # Use just one GPU (tensor_parallel_size=1)
#     llm = LLM(model=model, max_num_seqs=1, tensor_parallel_size=1)

#     LLM_CACHE[model] = llm
#     TOKENIZER_CACHE[model] = tokenizer
#     return llm, tokenizer


# def build_message_list(prompt, system_prompt=None):
#     """
#     Convert either a string or a list of alternating user/assistant messages
#     into a single list of role/content dicts.
#     """
#     messages = []
#     if system_prompt is not None:
#         messages.append({"role": "system", "content": system_prompt})
#     if isinstance(prompt, str):
#         messages.append({"role": "user", "content": prompt})
#     else:
#         for i, content in enumerate(prompt):
#             role = "user" if i % 2 == 0 else "assistant"
#             messages.append({"role": role, "content": content})
#     return messages


# def generate(prompt, model, system_prompt=None, num_retries=3, delay=0, **kwargs):
#     """
#     Generate a response from vLLM using the cached (LLM, tokenizer) objects in this process.
#     """
#     from time import sleep
#     import warnings

#     if num_retries == 0:
#         msg = f"Failed to get response from model {model} for prompt {prompt}"
#         warnings.warn(msg)
#         return ""

#     if delay > 0:
#         sleep(delay)

#     # Build messages
#     messages = build_message_list(prompt, system_prompt)

#     try:
#         # Retrieve cached LLM & tokenizer (loads once per process)
#         llm, tokenizer = get_llm_and_tokenizer(model)

#         # Prepare sampling params
#         max_length = tokenizer.model_max_length
#         sampling_params = SamplingParams(
#             temperature=0.7,
#             top_p=0.95,
#             max_tokens=max_length,
#         )

#         # Create a single “prompt” out of your entire conversation
#         messages_formatted = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#         )

#         # vLLM: if older versions, result may be a list of strings
#         # If newer versions (≥0.1.0), result is a list of GenerateOutput objects
#         outputs = llm.generate([messages_formatted], sampling_params)
#         if isinstance(outputs, list):
#             # On older vLLM versions: outputs is a list of strings
#             return outputs[0]
#         else:
#             # On newer vLLM versions: outputs is a list of GenerateOutput objects
#             return outputs[0].outputs[0].text

#     except Exception as e:
#         print(f"Error: {e}")
#         # Retry with exponential backoff
#         return generate(
#             prompt,
#             model,
#             system_prompt=system_prompt,
#             num_retries=num_retries - 1,
#             delay=max(2 * delay, 1),
#             **kwargs,
#         )


# def _assign_gpu_and_generate(example, index, model, prompt_col, **kwargs):
#     """
#     This is the function invoked by Dataset.map(..., with_indices=True).
#     Each process will call this repeatedly for its chunk of rows.
#     Each process sets CUDA_VISIBLE_DEVICES based on index % <num GPUs>.
#     Then calls generate(...) which uses the cached LLM & tokenizer.
#     """
#     # For 4 GPUs in total:
#     gpu_id = index % 4
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

#     prompt = example[prompt_col]
#     response = generate(prompt, model, **kwargs)
#     return {"response": response}


# def personalized_generate(
#     dataset: Dataset,
#     models: list[str],
#     target_column: str = "prompt",
#     decode_responses: bool = True,
#     num_proc: int = 4,
#     **kwargs,
# ) -> Dataset:
#     """
#     For each model, run the generation in parallel (num_proc processes),
#     but load the model once per process rather than once per row.
#     """
#     from datasets import concatenate_datasets

#     generated_datasets = []
#     for model in models:
#         # Use partial() to pass the model name & other args into the map function
#         map_fn = partial(_assign_gpu_and_generate, model=model, prompt_col=target_column, **kwargs)

#         # with_indices=True => map_fn will receive (example, index)
#         generated_dataset = dataset.map(
#             map_fn,
#             with_indices=True,
#             num_proc=num_proc,
#         )

#         # Mark the "model" column
#         generated_dataset = generated_dataset.add_column("model", [model] * len(generated_dataset))
#         generated_datasets.append(generated_dataset)

#     # Concatenate all per-model results
#     final_dataset = concatenate_datasets(generated_datasets)

#     if decode_responses:
#         # Postprocessing step, e.g., remove disallowed content or parse text
#         from .jailbreaks import decode_dataset
#         final_dataset = decode_dataset(final_dataset)

#     return final_dataset













import os
import warnings
import time
from typing import Union, List

from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------
# Global references, so we only load the model/tokenizer once.
# ---------------------------------------------------------------------
GLOBAL_LLM = None
GLOBAL_TOKENIZER = None

def max_tokens(model_name: str) -> int:
    model_name = model_name.lower()
    if any(x in model_name for x in ['gemma', 'llama']):
        return 8192
    elif any(x in model_name for x in ['qwen', 'mistral']):
        return 32768
    else:
        raise ValueError(f"Model {model_name} not supported")


def init_llm(model_name: str):
    """
    Initialize vLLM & tokenizer once, storing them in GLOBAL_LLM / GLOBAL_TOKENIZER.
    If already initialized, do nothing.
    """
    global GLOBAL_LLM, GLOBAL_TOKENIZER
    if GLOBAL_LLM is not None and GLOBAL_TOKENIZER is not None:
        return  # already loaded

    max_length = max_tokens(model_name)
    print(f"[init_llm] Loading model '{model_name}' with max_tokens={max_length} ...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_length

    # Build vLLM with all visible GPUs (tensor_parallel_size automatically uses them).
    llm = LLM(
        model=model_name,
        max_num_seqs=128,        # up to 128 sequences at once, adjust as you like
        tensor_parallel_size=0,  # 0 = use all visible GPUs in parallel if you want
    )
    GLOBAL_TOKENIZER = tokenizer
    GLOBAL_LLM = llm
    print("[init_llm] Model & tokenizer loaded successfully.")


def build_messages(prompt: Union[str, List[str]], system_prompt: str = None):
    """
    Turn either a single string or a list of user/assistant strings
    into a list[dict(role, content)] for the chat format.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # If prompt is a single string:
    if isinstance(prompt, str):
        messages.append({"role": "user", "content": prompt})
    else:
        # If it's a list of alternating user/assistant messages
        for i, content in enumerate(prompt):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": content})
    return messages


def generate_batch(
    prompts: List[Union[str, List[str]]],
    model_name: str,
    system_prompt: str = None,
    num_retries: int = 3,
    delay: float = 0.0,
    **kwargs,
) -> List[str]:
    """
    Generate for a batch of prompts in ONE call to vLLM.
    Re-try on exception with the same "num_retries" logic you had before.
    """
    global GLOBAL_LLM, GLOBAL_TOKENIZER

    if num_retries == 0:
        warnings.warn(f"Failed to get response from model {model_name} for these prompts: {prompts}")
        # Return empty strings for all prompts
        return [""] * len(prompts)

    if delay > 0:
        time.sleep(delay)

    try:
        # Ensure we have a single LLM & tokenizer loaded once
        init_llm(model_name)

        # Build the list of chat-formatted strings for each prompt
        # We'll pass them all at once to `llm.generate`
        batch_messages = []
        for prompt in prompts:
            msg_list = build_messages(prompt, system_prompt)
            # vLLM usually wants a single prompt string or an already-templated chat
            # We can just flatten the roles into a single text or use `apply_chat_template`.
            # For simplicity, let's just build a text from roles:
            text_str = "\n".join(f"[{m['role']}] {m['content']}" for m in msg_list)
            batch_messages.append(text_str)

        # Prepare sampling params
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=GLOBAL_TOKENIZER.model_max_length,
        )

        # Actually generate for the entire batch in one call
        outputs = GLOBAL_LLM.generate(batch_messages, sampling_params)

        # On vLLM >= 0.1.0, "outputs" is a list of GenerateOutput objects, each has .outputs
        # On older vLLM, it may be just a list of strings.
        results = []
        if hasattr(outputs[0], "outputs"):
            # Newer style
            for out in outputs:
                results.append(out.outputs[0].text if out.outputs else "")
        else:
            # Older style
            results = outputs

        return results

    except Exception as e:
        print(f"[generate_batch] Error: {e}")
        # Exponential backoff delay
        next_delay = max(2 * delay, 1.0)
        return generate_batch(
            prompts,
            model_name,
            system_prompt=system_prompt,
            num_retries=num_retries - 1,
            delay=next_delay,
            **kwargs,
        )


def personalized_generate(
    dataset: Dataset,
    model_name: str,
    target_column: str = "prompt",
    decode_responses: bool = True,
    batch_size: int = 16,
    num_proc: int = 1,
    **kwargs,
) -> Dataset:
    """
    1) Build exactly one vLLM model on the visible GPUs.
    2) For each batch of data, call generate_batch(...) once for that entire batch.
    3) Return the final dataset with responses.
    """
    from datasets import concatenate_datasets

    # Ensure LLM is loaded once at the start (optional; or let the first call do it).
    init_llm(model_name)

    def _batched_map_fn(batch):
        # "batch" is a dict of lists. We want the list of prompts:
        prompts = batch[target_column]

        # generate responses in a single call
        responses = generate_batch(
            prompts,
            model_name=model_name,
            **kwargs,  # e.g. system_prompt, custom sampling
        )
        return {"response": responses}

    # We'll do a single "map" call over the dataset. 
    # Instead of a loop over each model, 
    # we'll assume you want to load just one model_name as you said.
    mapped_dataset = dataset.map(
        _batched_map_fn,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
    )

    # You can add a "model" column if you like
    mapped_dataset = mapped_dataset.add_column("model", [model_name] * len(mapped_dataset))

    if decode_responses:
        # Post-processing if you have a "decode_dataset" function
        from .jailbreaks import decode_dataset
        mapped_dataset = decode_dataset(mapped_dataset)

    return mapped_dataset