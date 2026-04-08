"""
vLLM async inference backend.

Uses AsyncLLMEngine with LoRARequest for adapter loading at generation time.
vLLM is an optional dependency — fails gracefully if not installed.
"""

import logging
import uuid
from pathlib import Path
from typing import AsyncIterator

from config.settings import Settings

logger = logging.getLogger(__name__)


class VLLMModel:
    """vLLM-based high-throughput async inference backend."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._engine = None
        self._ready = False

    # ── Lifecycle ─────────────────────────────────────────

    def load(self) -> None:
        """Initialise the vLLM async engine. Call explicitly."""
        try:
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm import AsyncLLMEngine
        except ImportError:
            raise ImportError(
                "vLLM is not installed. Install with:  pip install vllm\n"
                "Or switch to the 'transformers' backend in config.yaml."
            )

        cfg = self._settings.model
        logger.info("Initialising vLLM engine: %s", cfg.name)

        engine_args = AsyncEngineArgs(
            model=cfg.name,
            tensor_parallel_size=1,
            dtype="float16",
            enable_lora=True,
            max_lora_rank=max(cfg.lora.r * 2, 64),
            gpu_memory_utilization=0.85,
        )

        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._ready = True
        logger.info("vLLM engine ready")

    def is_ready(self) -> bool:
        return self._ready

    # ── Inference ─────────────────────────────────────────

    async def generate(
        self, prompt: str, max_tokens: int | None = None
    ) -> AsyncIterator[str]:
        """Stream generated text chunks using the vLLM engine."""
        if not self._ready:
            raise RuntimeError("vLLM engine not loaded. Call load() first.")

        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        cfg = self._settings.model
        max_tokens = max_tokens or cfg.max_new_tokens

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.2,
            top_p=0.95,
        )

        # ── Attach trained LoRA adapter if checkpoint exists ──
        lora_request = None
        lora_path = Path(cfg.lora.checkpoint_path)
        if lora_path.exists() and any(lora_path.iterdir()):
            lora_request = LoRARequest(
                lora_name="coding_lora",
                lora_int_id=1,
                lora_path=str(lora_path.resolve()),
            )
            logger.debug("Attaching LoRA adapter from %s", lora_path)

        request_id = str(uuid.uuid4())
        prev_text_len = 0

        async for request_output in self._engine.generate(
            prompt, params, request_id, lora_request=lora_request
        ):
            current_text = request_output.outputs[0].text
            new_text = current_text[prev_text_len:]
            prev_text_len = len(current_text)
            if new_text:
                yield new_text