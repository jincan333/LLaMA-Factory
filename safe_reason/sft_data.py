import json


data_path = 'data/gsm8k/total_data.json'
new_data_path = 'data/gsm8k/sft.json'
with open(data_path, 'r') as f:
    data = json.load(f)

new_data = []
for item in data:
    new_item = {
        "real": item["real"],
        "rectify": item["rectify"]
    }
    new_data.append(new_item)

with open(new_data_path, 'w') as f:
    json.dump(new_data, f, indent=4)