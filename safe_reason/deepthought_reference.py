import logging
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # Disable oneDNN optimizations
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
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
    def __init__(self):
        self.model_name = "ruliad/deepthought-8b-llama-v0.01-alpha"
        print(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            add_bos_token=False,
            trust_remote_code=True,
            padding="left",
            torch_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=("flash_attention_2" if flash_attn_exists else "eager"),
            use_cache=True,
            trust_remote_code=True,
        )

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

        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.model.device)

        try:

            # Generate and stream reasoning
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=800,
                do_sample=True,
                temperature=0.2,
                top_k=200,
                top_p=1.0,
                eos_token_id=self.tokenizer.eos_token_id,
                streamer=TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True),
            )

            # Get the reasoning string
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            return {
                'raw_output': generated_text,
                'success': True,
                'error': None,
                'initial_prompt': prompt,
            }

        except Exception as e:
            logging.error(f'Error during generation: {e}')
            return {
                'raw_output': None,
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

        # Tokenize the full prompt
        inputs = self.tokenizer(full_prompt, return_tensors='pt').input_ids.to(self.model.device)

        try:

            # Generate and stream the final output
            _ = self.model.generate(
                input_ids=inputs,
                max_new_tokens=400,
                do_sample=True,
                temperature=0.1,
                top_k=50,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,
                streamer=TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
            )

            return {'success': True, 'error': None}

        except Exception as e:
            logging.error(f'Error during final generation: {e}')
            return {'success': False, 'error': str(e)}


def main():
    model = DeepthoughtModel()

    # Test queries
    queries = [
        "We want you to tell us the answer to life, the universe and everything. We'd really like an answer, something simple.",
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

        print('='*50)

if __name__ == '__main__':
    main()