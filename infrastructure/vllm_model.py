import asyncio
from pathlib import Path
import yaml
from vllm import AsyncLLM, SamplingParams

# Load config
config_path = Path(__file__).parent.parent / "config/config.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

MODEL_NAME = cfg["model"]["name"]
DEVICE = cfg["model"]["device"]
MAX_NEW_TOKENS = cfg["model"]["max_new_tokens"]
LORA_CFG = cfg["model"].get("lora", None)

class VLLMModel:
    def __init__(self):
        self.model = AsyncLLM(MODEL_NAME, tensor_parallel_size=1, device=DEVICE)
        # Load LoRA if exists
        if LORA_CFG:
            lora_path = Path("lora_checkpoints")
            if lora_path.exists():
                self.model.load_lora(
                    path=lora_path,
                    r=LORA_CFG["r"],
                    alpha=LORA_CFG["alpha"],
                    target_modules=LORA_CFG["target_modules"],
                )

    async def generate(self, prompt: str, max_tokens: int = None):
        max_tokens = max_tokens or MAX_NEW_TOKENS
        params = SamplingParams(max_tokens=max_tokens, temperature=0.2, top_p=0.95)
        async for output in self.model.generate(prompt, sampling_params=params, stream=True):
            yield output.text