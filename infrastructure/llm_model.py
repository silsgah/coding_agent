"""
HuggingFace Transformers inference backend.

Loads the base model + trained LoRA adapter (if checkpoint exists).
Supports async streaming via TextIteratorStreamer.
"""

import logging
from pathlib import Path
from threading import Thread
from typing import AsyncIterator

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from config.settings import Settings

logger = logging.getLogger(__name__)


class HFModel:
    """HuggingFace Transformers inference backend with LoRA support."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._model = None
        self._tokenizer = None
        self._ready = False

    # ── Lifecycle ─────────────────────────────────────────

    def load(self) -> None:
        """Load model and tokenizer. Call explicitly — never at import time."""
        cfg = self._settings.model
        logger.info("Loading model: %s on %s", cfg.name, cfg.device)

        self._tokenizer = AutoTokenizer.from_pretrained(cfg.name, use_fast=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Use 4-bit quantization when running on GPU to reduce VRAM
        load_kwargs: dict = {"torch_dtype": torch.float16}
        if cfg.device == "cuda" and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = "auto"
        else:
            logger.warning("CUDA not available — loading model on CPU (slow)")
            load_kwargs["device_map"] = "cpu"

        self._model = AutoModelForCausalLM.from_pretrained(
            cfg.name, **load_kwargs
        )

        # ── Load TRAINED LoRA adapter (not a fresh one!) ──
        lora_path = Path(cfg.lora.checkpoint_path)
        if lora_path.exists() and any(lora_path.iterdir()):
            logger.info("Loading trained LoRA adapter from %s", lora_path)
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, str(lora_path))
        else:
            logger.warning(
                "No LoRA checkpoint at %s — using base model only", lora_path
            )

        self._model.eval()
        self._ready = True
        logger.info("Model loaded successfully")

    def is_ready(self) -> bool:
        return self._ready

    # ── Inference ─────────────────────────────────────────

    async def generate(
        self, prompt: str, max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        """Stream generated text token-by-token."""
        if not self._ready:
            raise RuntimeError("Model not loaded. Call load() first.")

        from transformers import TextIteratorStreamer

        max_tokens = max_tokens or self._settings.model.max_new_tokens
        inputs = self._tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "streamer": streamer,
            "do_sample": True,
            "temperature": 0.2,
            "top_p": 0.95,
        }

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        for text in streamer:
            yield text

        thread.join()