# dataset_utils.py
import os
import json

def load_local_dataset(file_path):
    if not os.path.exists(file_path):
        print(f"[WARNING] {file_path} not found. Creating dummy dataset...")
        dummy_data = [{"input": "def add(a,b):", "output": " return a+b"}]
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(dummy_data, f, ensure_ascii=False)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def prepare_dataset(train_source="raw_data/train.json", valid_source="raw_data/valid.json"):
    print("Loading datasets...")
    train_data = load_local_dataset(train_source)
    valid_data = load_local_dataset(valid_source)
    
    # Optionally, save temporary JSON files for your LoRA pipeline
    train_file = "processed_train.json"
    valid_file = "processed_valid.json"
    
    with open(train_file, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False)
    with open(valid_file, "w", encoding="utf-8") as f:
        json.dump(valid_data, f, ensure_ascii=False)
    
    print(f"Datasets ready: {train_file}, {valid_file}")
    return train_file, valid_file