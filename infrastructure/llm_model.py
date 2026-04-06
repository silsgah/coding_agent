import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

class LLMModel:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.device = cfg["model"]["device"]
        self.max_new_tokens = cfg["model"]["max_new_tokens"]
        model_name = cfg["model"]["name"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(self.device)

        # LoRA
        lora_cfg = cfg["model"]["lora"]
        lora_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["alpha"],
            target_modules=lora_cfg["target_modules"]
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.eval()

    def generate(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)