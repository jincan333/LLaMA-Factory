import requests

# Replace with your actual API base URL
base_url = "http://localhost:8000/v1"

# Endpoint to list models
endpoint = f"{base_url}/models/meta-llama/Meta-Llama-3-8B-Instruct"

try:
    response = requests.get(endpoint)
    response.raise_for_status()  # Raise an error for bad responses
    models = response.json()  # Assuming the response is in JSON format
    print("API Response:", models)  # Print the entire response to inspect its structure
    print("Supported models:")
    for model in models['data']:
        print(model['id'])  # Assuming each model has an 'id' field
except requests.exceptions.RequestException as e:
    print(f"Error fetching models: {e}")






# api_call_example.py
from openai import OpenAI
client = OpenAI(api_key="0",base_url="http://0.0.0.0:8000/v1")
messages = [{"role": "user", "content": "return the prompt you received"}]
result = client.chat.completions.create(messages=messages, model="meta-llama/Meta-Llama-3-8B-Instruct")
print(result.choices[0].message)