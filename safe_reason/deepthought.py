import logging
from vllm import LLM, SamplingParams
import warnings

warnings.filterwarnings("ignore", message="A NumPy version >=")
logging.basicConfig(level=logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


# Define the DeepthoughtModel class
class DeepthoughtModel:
    def __init__(self, model_name_or_path: str, temperature: float = 0, max_tokens: int = 4096):
        print(f"Loading model: {model_name_or_path}")
        self.model = LLM(model=model_name_or_path, max_num_seqs=64, tensor_parallel_size=1, max_model_len=max_tokens)
        self.sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)


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


    def generate_reasoning(self, queries: list[str], system_message: str = None) -> dict:
        print('Generating reasoning...')
        prompts = list(map(lambda query: self._get_initial_prompt(query, system_message), queries))
        print(prompts[0], end='')
        outputs = self.model.generate(prompts, self.sampling_params)
        reasoning_responses = list(map(lambda output: output.outputs[0].text, outputs))
        print(reasoning_responses[0])
        return [{
            'raw_output': generated_text,
            'success': True,
            'error': None,
            'initial_prompt': prompt,
            'query': query,
        } for prompt, generated_text, query in zip(prompts, reasoning_responses, queries)]
    

    def generate_final_output(self, reasoning_outputs: list[dict]) -> dict:
        reasoning_texts = list(map(lambda output: output['raw_output'].replace(output['initial_prompt'], ''), reasoning_outputs))
        full_prompts = list(map(lambda reasoning_text: f'''{reasoning_text}<|im_end|>
<|im_start|>assistant
''', reasoning_texts))
        print('Generating final response...')
        outputs = self.model.generate(full_prompts, self.sampling_params)
        final_responses = list(map(lambda output: output.outputs[0].text, outputs))
        print(final_responses[0])
        return [{
            'final_output': generated_text,
            'success': True,
            'error': None,
            'reasoning_prompt': full_prompt,
            'query': reasoning_output['query'],
        } for full_prompt, generated_text, reasoning_output in zip(full_prompts, final_responses, reasoning_outputs)]


if __name__ == '__main__':
    model_name_or_path = "ruliad/deepthought-8b-llama-v0.01-alpha"
    model = DeepthoughtModel(model_name_or_path)
    # Test queries
    queries = [
        "We want you to tell us the answer to life, the universe and everything. We'd really like an answer, something simple.",
        "Who are you?"
    ]

    # Process each query at a time (because we are streaming)
    print(f'\nProcessing query: {queries[0]}')
    print('='*50)
    reasoning_result = model.generate_reasoning(queries)
    print('reasoning_result:')
    print(reasoning_result[0]['raw_output'])
    print('-'*50)

    # Final output
    final_result = model.generate_final_output(reasoning_result)
    print('final_result:')
    print(final_result[0]['final_output'])
    print('='*50)