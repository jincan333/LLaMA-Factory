
model_name_or_path = "ruliad/deepthought-8b-llama-v0.01-alpha"

import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # Disable oneDNN optimizations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from vllm import LLM, SamplingParams
import warnings


warnings.filterwarnings("ignore", message="A NumPy version >=")
logging.basicConfig(level=logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


# Check if Flash Attention is available
try:
    import flash_attn  # noqa: F401
    flash_attn_exists = True
except ImportError:
    flash_attn_exists = False


# Define the DeepthoughtModel class
class DeepthoughtModel:
    def __init__(self, model_name_or_path=model_name_or_path):
        print(f"Loading model: {model_name_or_path}")
        self.model = LLM(model=model_name_or_path, max_num_seqs=64, tensor_parallel_size=1, max_model_len=4096)
        self.sampling_params = SamplingParams(temperature=0, max_tokens=4096)

    # Helper method to generate the initial prompt
    def _get_initial_prompt(
        self, query: str, system_message: str = None
    ) -> str:
        '''Helper method to generate the initial prompt format.'''
        if system_message is None:
            system_message = '''You are a superintelligent AI system, capable of comprehensive reasoning. When provided with <reasoning>, you must provide your logical reasoning chain to solve the user query. Be verbose with your outputs.'''

        return f'''<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>reasoning
<reasoning>
[
  {{
    "step": 1,
    "type": "problem_understanding",
    "thought": "'''

    # Method to generate reasoning given the prompt
    def generate_reasoning(self, query: str, system_message: str = None) -> dict:
        print('Generating reasoning...')

        # Get and print prompt
        prompt = self._get_initial_prompt(query, system_message)
        print(prompt, end='')

        try:
            generated_text = self.model.generate(prompt, self.sampling_params)[0].outputs[0].text
            return {
                'raw_output': generated_text,
                'success': True,
                'error': None,
                'initial_prompt': prompt,
            }

        except Exception as e:
            logging.error(f'Error during generation: {e}')
            return {
                'raw_output': query,
                'success': False,
                'error': str(e),
                'initial_prompt': None,
            }

    # Method to generate the final output
    def generate_final_output(self, reasoning_output: dict) -> dict:

        # Get the reasoning text and create the full prompt for the final output
        reasoning_text = reasoning_output['raw_output'].replace(reasoning_output['initial_prompt'], '')
        full_prompt = f'''{reasoning_text}<|im_end|>
<|im_start|>assistant
'''
        print('Generating final response...')
        try:
            generated_text = self.model.generate(full_prompt, self.sampling_params)[0].outputs[0].text
            return {'final_output': generated_text, 'success': True, 'error': None}

        except Exception as e:
            logging.error(f'Error during final generation: {e}')
            return {'final_output': None, 'success': False, 'error': str(e)}


def main():
    model = DeepthoughtModel()

    # Test queries
    queries = [
        "We want you to tell us the answer to life, the universe and everything. We'd really like an answer, something simple.",
        "Who are you?"
    ]

    # Process each query at a time (because we are streaming)
    for query in queries:
        print(f'\nProcessing query: {query}')
        print('='*50)

        # Reasoning
        reasoning_result = model.generate_reasoning(query)
        if not reasoning_result['success']:
            print(f'\nError in reasoning: {reasoning_result["error"]}')
            print('='*50)
            continue

        print('-'*50)

        # Final output
        final_result = model.generate_final_output(reasoning_result)
        if not final_result['success']:
            print(f'\nError in final generation: {final_result["error"]}')
        else:
            print(f'\nFinal output: {final_result["final_output"]}')

        print('='*50)

if __name__ == '__main__':
    main()