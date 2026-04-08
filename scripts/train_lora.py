"""
LoRA / QLoRA fine-tuning script for the coding agent.

Fixes vs. the original:
  - formatting_func combines prompt + completion (the model now trains on answers)
  - Explicit model save at the end
  - eval_strategy enabled
  - Consistent model name from config
  - Correct data paths pointing to data/ directory
"""

import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# ── Allow running from the repo root: `python -m scripts.train_lora` ──
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config.settings import load_settings  # noqa: E402

# ─────────────────────────────────────────────────────────
# 1️⃣  Config
# ─────────────────────────────────────────────────────────

settings = load_settings()
MODEL_NAME = settings.model.name
LORA_CFG = settings.model.lora
CHECKPOINT_DIR = LORA_CFG.checkpoint_path

# GPU memory safety
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Model: {MODEL_NAME}")

# ─────────────────────────────────────────────────────────
# 2️⃣  Tokenizer
# ─────────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# ─────────────────────────────────────────────────────────
# 3️⃣  4-bit Quantization
# ─────────────────────────────────────────────────────────

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# ─────────────────────────────────────────────────────────
# 4️⃣  Load Model
# ─────────────────────────────────────────────────────────

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
)

model = prepare_model_for_kbit_training(model)

# ─────────────────────────────────────────────────────────
# 5️⃣  LoRA Configuration (from central config)
# ─────────────────────────────────────────────────────────

lora_config = LoraConfig(
    r=LORA_CFG.r,
    lora_alpha=LORA_CFG.alpha,
    target_modules=LORA_CFG.target_modules,
    lora_dropout=LORA_CFG.dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

# ─────────────────────────────────────────────────────────
# 6️⃣  Load Dataset
# ─────────────────────────────────────────────────────────

DATA_DIR = REPO_ROOT / "data"
TRAIN_FILE = str(DATA_DIR / "train.jsonl")
VALID_FILE = str(DATA_DIR / "valid.jsonl")

# Fall back to _fixed variants if they exist
if (DATA_DIR / "train_fixed.jsonl").exists():
    TRAIN_FILE = str(DATA_DIR / "train_fixed.jsonl")
    VALID_FILE = str(DATA_DIR / "valid_fixed.jsonl")
    print("Using _fixed dataset variants")

dataset = load_dataset("json", data_files={
    "train": TRAIN_FILE,
    "validation": VALID_FILE,
})

print(f"Train samples: {len(dataset['train'])}")
print(f"Valid samples: {len(dataset['validation'])}")

# ─────────────────────────────────────────────────────────
# 7️⃣  Formatting Function
#     ⚠️  This is the critical fix: the old code only passed
#     dataset_text_field="prompt", so the model never saw the
#     completion/answer during training!
# ─────────────────────────────────────────────────────────

def formatting_func(example):
    """Combine prompt + completion into a single training string."""
    prompt = example.get("prompt", "")
    completion = example.get("completion", "")
    return f"### Instruction:\n{prompt}\n\n### Response:\n{completion}"

# ─────────────────────────────────────────────────────────
# 8️⃣  Training Arguments
# ─────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=200,
    save_total_limit=5,
    eval_strategy="steps",
    eval_steps=200,
    fp16=torch.cuda.is_available(),  # consistent with model dtype
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    report_to="none",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

# ─────────────────────────────────────────────────────────
# 9️⃣  Trainer
# ─────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=lora_config,
    formatting_func=formatting_func,
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
    packing=True,
)

# ─────────────────────────────────────────────────────────
# 🔟  Train + Save
# ─────────────────────────────────────────────────────────

print("Starting training…")
trainer.train()

# Save the final adapter weights
final_path = Path(CHECKPOINT_DIR)
final_path.mkdir(parents=True, exist_ok=True)
trainer.save_model(str(final_path))
tokenizer.save_pretrained(str(final_path))
print(f"Training complete — adapter saved to {final_path}")