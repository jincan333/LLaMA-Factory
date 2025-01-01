import json


data_path = 'data/gsm8k/total_data.json'
new_data_path = 'data/gsm8k/dpo.json'
with open(data_path, 'r') as f:
    data = json.load(f)

new_data = []
for item in data:
    new_item = {
        "messages": [
            item["real"][0]
        ],
        "generated": item["generated"][1],
        "real": item["real"][1],
        "rectify": item["rectify"][1]
    }
    new_data.append(new_item)

with open(new_data_path, 'w') as f:
    json.dump(new_data, f, indent=4)