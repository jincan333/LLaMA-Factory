# # BeaverTails
# from datasets import load_dataset

# # Load the whole dataset
# dataset = load_dataset('PKU-Alignment/BeaverTails')

# # Load only the round 0 dataset
# round0_30k_dataset = load_dataset('PKU-Alignment/BeaverTails', data_dir='round0/30k')

# # Load the training dataset
# train_dataset = load_dataset('PKU-Alignment/BeaverTails', split='30k_train')
# test_dataset = load_dataset('PKU-Alignment/BeaverTails', split='30k_test')

# import json
# import os

# # Directory to save the JSON files
# output_dir = "data/beavertails"
# os.makedirs(output_dir, exist_ok=True)

# # Save each split of the dataset as a separate JSON file
# for split in round0_30k_dataset.keys():
#     json_path = os.path.join(output_dir, f"{split}.json")
#     # Save the dataset split as JSON
#     round0_30k_dataset[split].to_json(json_path, orient="records", lines=False, indent=4)
#     print(f"Saved {split} split to {json_path}")



# # HEx-PHI
# from datasets import load_dataset
# import pandas as pd
# import os

# # Categories and corresponding file names
# categories = {
#     "Category_1_Illegal_Activity": "category_1.csv",
#     "Category_3_Hate_Harass_Violence": "category_3.csv",
#     "Category_4_Malware": "category_4.csv",
#     "Category_5_Physical_Harm": "category_5.csv",
#     "Category_6_Economic_Harm": "category_6.csv",
#     "Category_7_Fraud_Deception": "category_7.csv",
#     "Category_8_Adult_Content": "category_8.csv",
#     "Category_9_Political_Campaigning": "category_9.csv",
#     "Category_10_Privacy_Violation_Activity": "category_10.csv",
#     "Category_11_Tailored_Financial_Advice": "category_11.csv"
# }

# # Load and process each category
# data_all = {category: [] for category in categories}

# for category, file_name in categories.items():
#     data_file = load_dataset('LLM-Tuning-Safety/HEx-PHI', data_files=file_name)
#     # Append the column name (features.keys()[0]) and content
#     col_name = list(data_file['train'].features.keys())[0]
#     data_all[category].append(col_name)
#     data_all[category].extend([list(row.values())[0] for row in data_file['train']])

# # Convert to a DataFrame and reshape it
# df = pd.DataFrame(data_all)
# df = df.melt(var_name="Category", value_name="Content")

# # Ensure the output directory exists and save the DataFrame
# os.makedirs('data/hex_phi', exist_ok=True)
# df.to_csv('data/hex_phi/test.csv', index=False)

# print("Data saved successfully!")

# path = "/research/cbim/vast/cj574/.cache/huggingface/hub/datasets--LLM-Tuning-Safety--HEx-PHI/snapshots/83128b46334b80cc567bd7a2caf7af11c5b0bab7"
# file1 = pd.read_csv(os.path.join(path, 'category_1.csv'))
# file3 = pd.read_csv(os.path.join(path, 'category_3.csv'))






# # StrongREJECT

# from strong_reject.load_datasets import load_strongreject_small, load_strongreject

# # load the small version of the StrongREJECT dataset
# forbidden_prompt_dataset = load_strongreject()

# import json
# import os

# # Directory to save the JSON files
# output_dir = "data/strongreject"
# os.makedirs(output_dir, exist_ok=True)
# json_path = os.path.join(output_dir, "test.json")
# forbidden_prompt_dataset.to_json(json_path, orient="records", lines=False, indent=4)
    
# # Save the dataset split as JSON
# with open(json_path, "w") as f:
#     json.dump(forbidden_prompt_dataset.to_dict(), f, indent=4)
#     print(f"Saved strongreject to {json_path}")





# # AdvBench

# from datasets import load_dataset

# ds = load_dataset("walledai/AdvBench")
# ds['train'].to_json("data/advbench/test.json", orient="records", lines=False, indent=4)






# # Xstest

# from datasets import load_dataset

# ds = load_dataset("walledai/XSTest")
# ds['test'].to_json("data/xstest/test.json", orient="records", lines=False, indent=4)