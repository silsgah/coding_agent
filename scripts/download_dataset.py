import os
import json
import hashlib
from datasets import load_dataset

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "data"
RAW_FILE = os.path.join(DATA_DIR, "raw_dataset.json")
HASH_FILE = os.path.join(DATA_DIR, ".dataset_hash")
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
VALID_FILE = os.path.join(DATA_DIR, "valid.jsonl")
DATASET_NAME = "sahil2801/CodeAlpaca-20k"
TRAIN_SPLIT_RATIO = 0.9

os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------
# Helper functions
# -----------------------------
def compute_hash(text: bytes) -> str:
    return hashlib.md5(text).hexdigest()

def normalize_text(text: str) -> str:
    """Normalize whitespace, line endings, and trim."""
    return " ".join(text.strip().split())

# -----------------------------
# Load dataset from HuggingFace
# -----------------------------
print("Downloading dataset...")
dataset = load_dataset(DATASET_NAME)

# Convert to JSON bytes for hashing
dataset_json_bytes = json.dumps(dataset["train"], ensure_ascii=False).encode("utf-8")
new_hash = compute_hash(dataset_json_bytes)

old_hash = None
if os.path.exists(HASH_FILE):
    with open(HASH_FILE, "r") as f:
        old_hash = f.read().strip()

# -----------------------------
# Check if download/processing is needed
# -----------------------------
if new_hash == old_hash and os.path.exists(TRAIN_FILE) and os.path.exists(VALID_FILE):
    print("Dataset already downloaded and normalized. Skipping.")
else:
    print("Processing dataset...")

    # Save raw dataset
    with open(RAW_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(dataset["train"], ensure_ascii=False))

    # Split train/valid
    split_index = int(len(dataset["train"]) * TRAIN_SPLIT_RATIO)
    with open(TRAIN_FILE, "w", encoding="utf-8") as train_f, \
         open(VALID_FILE, "w", encoding="utf-8") as valid_f:

        for i, item in enumerate(dataset["train"]):
            prompt = item["instruction"]
            if item.get("input"):
                prompt += "\n" + item["input"]
            completion = item["output"]

            # Normalize
            record = {
                "prompt": normalize_text(prompt),
                "completion": normalize_text(completion)
            }

            # Write to file
            if i < split_index:
                train_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                valid_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Update hash
    with open(HASH_FILE, "w") as f:
        f.write(new_hash)

    print("Dataset processed and saved:")
    print(f" - {TRAIN_FILE}")
    print(f" - {VALID_FILE}")