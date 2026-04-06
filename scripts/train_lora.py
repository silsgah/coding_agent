# train_modern_llm.py

import os
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training
)

from trl import SFTTrainer

from dataset_utils import prepare_dataset


# -----------------------------
# 1️⃣ GPU Memory Safety
# -----------------------------

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# -----------------------------
# 2️⃣ Dataset Preparation
# -----------------------------

train_source = "raw_data/train.json"
valid_source = "raw_data/valid.json"

train_file, valid_file = prepare_dataset(train_source, valid_source)


# -----------------------------
# 3️⃣ Model + Tokenizer
# -----------------------------

MODEL_NAME = "Qwen/Qwen2-7B"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

tokenizer.pad_token = tokenizer.eos_token


# -----------------------------
# 4️⃣ 4-bit Quantization
# -----------------------------

bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_use_double_quant=True,

    bnb_4bit_quant_type="nf4"
)


# -----------------------------
# 5️⃣ Load Model
# -----------------------------

model = AutoModelForCausalLM.from_pretrained(

    MODEL_NAME,

    quantization_config=bnb_config,

    device_map="auto",

    torch_dtype=torch.float16,

    attn_implementation="flash_attention_2"
)

model = prepare_model_for_kbit_training(model)


# -----------------------------
# 6️⃣ LoRA Configuration
# -----------------------------

lora_config = LoraConfig(

    r=16,

    lora_alpha=32,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)


# -----------------------------
# 7️⃣ Load Dataset
# -----------------------------

data_files = {
    "train": train_file,
    "validation": valid_file
}

dataset = load_dataset("json", data_files=data_files)


# -----------------------------
# 8️⃣ Training Arguments
# -----------------------------

training_args = TrainingArguments(

    output_dir="lora_checkpoints",

    per_device_train_batch_size=1,

    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    num_train_epochs=1,

    logging_steps=10,

    save_steps=200,

    save_total_limit=5,

    bf16=torch.cuda.is_available(),

    gradient_checkpointing=True,

    optim="paged_adamw_32bit",

    report_to="none"
)


# -----------------------------
# 9️⃣ Trainer (Modern)
# -----------------------------

trainer = SFTTrainer(

    model=model,

    train_dataset=dataset["train"],

    eval_dataset=dataset["validation"],

    peft_config=lora_config,

    dataset_text_field="prompt",

    max_seq_length=1024,

    tokenizer=tokenizer,

    args=training_args,

    packing=True
)


# -----------------------------
# 🔟 Start Training
# -----------------------------

trainer.train()

print("Training completed!")