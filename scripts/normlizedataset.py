import json
from transformers import AutoTokenizer
from pathlib import Path

# ─── CONFIG ──────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # replace with your base model
DATA_DIR = Path("./data")                  # folder containing train.jsonl, valid.jsonl
FILES = ["train.jsonl", "valid.jsonl"]    # dataset files to fix
# ────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
def normalize_text(text: str) -> str:
    """Normalize whitespace and strip trailing newlines"""
    return text.replace("\r\n", "\n").strip()

def fix_dataset_file(file_path: Path):
    print(f"Processing {file_path}...")
    fixed_lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            prompt = normalize_text(data["prompt"])
            completion = normalize_text(data.get("completion", ""))

            # tokenize
            prompt_tokens = tokenizer(prompt, add_special_tokens=False).input_ids
            full_tokens = tokenizer(prompt + completion, add_special_tokens=False).input_ids

            # verify mismatch
            if full_tokens[:len(prompt_tokens)] != prompt_tokens:
                # fix by trimming leading whitespace in completion
                completion = completion.lstrip()
                full_tokens = tokenizer(prompt + completion, add_special_tokens=False).input_ids
                if full_tokens[:len(prompt_tokens)] != prompt_tokens:
                    print("⚠️ Still mismatch after trimming: ", data)
            data["prompt"] = prompt
            data["completion"] = completion
            fixed_lines.append(json.dumps(data, ensure_ascii=False))

    # save to a new file to avoid overwriting original
    out_path = file_path.with_name(file_path.stem + "_fixed.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(fixed_lines))
    print(f"Saved fixed dataset to {out_path}\n")

# run for all files
for file_name in FILES:
    file_path = DATA_DIR / file_name
    if file_path.exists():
        fix_dataset_file(file_path)
    else:
        print(f"File not found: {file_path}")