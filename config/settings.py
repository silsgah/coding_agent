"""
Centralized, validated configuration.

Loads from config.yaml with environment variable overrides.
Env vars use the AGENT_ prefix (e.g. AGENT_MODEL_NAME, AGENT_DEVICE).
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Sub-models ────────────────────────────────────────────

class LoRASettings(BaseModel):
    r: int = 16
    alpha: int = 32
    target_modules: list[str] = Field(
        default=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    dropout: float = 0.05
    checkpoint_path: str = "lora_checkpoints"


class ModelSettings(BaseModel):
    name: str = "Qwen/Qwen2-7B"
    device: str = "cuda"
    max_new_tokens: int = 512
    backend: str = "transformers"  # "transformers" or "vllm"
    lora: LoRASettings = LoRASettings()


class RepoSettings(BaseModel):
    paths: list[str] = ["./my_project"]
    file_types: list[str] = [".py", ".ipynb"]


class AgentSettings(BaseModel):
    max_steps: int = 6
    enable_tests: bool = True
    max_context_files: int = 5
    max_prompt_length: int = 4096


class SessionSettings(BaseModel):
    session_dir: str = ".mini-coding-agent/sessions"
    max_history_chars: int = 12000
    max_tool_output: int = 4000
    max_depth: int = 1  # subagent recursion limit


class APISettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]


# ── Root settings ─────────────────────────────────────────

class Settings(BaseModel):
    model: ModelSettings = ModelSettings()
    repo: RepoSettings = RepoSettings()
    agent: AgentSettings = AgentSettings()
    session: SessionSettings = SessionSettings()
    api: APISettings = APISettings()


# ── Loader ────────────────────────────────────────────────

def load_settings(config_path: Optional[str] = None) -> Settings:
    """
    Load settings from YAML config file, then apply env-var overrides.

    Priority: env vars > config.yaml > defaults
    """
    if config_path is None:
        config_path = os.environ.get(
            "AGENT_CONFIG_PATH",
            str(Path(__file__).parent / "config.yaml"),
        )

    config_data: dict = {}
    cfg_file = Path(config_path)
    if cfg_file.exists():
        with open(cfg_file) as f:
            config_data = yaml.safe_load(f) or {}
        logger.info("Loaded config from %s", cfg_file)
    else:
        logger.warning("Config file %s not found, using defaults", cfg_file)

    # ── Env-var overrides (flat, prefixed with AGENT_) ──
    _env_override(config_data, "model", "name", "AGENT_MODEL_NAME")
    _env_override(config_data, "model", "device", "AGENT_DEVICE")
    _env_override(config_data, "model", "backend", "AGENT_BACKEND")
    _env_override(config_data, "model", "max_new_tokens", "AGENT_MAX_NEW_TOKENS", cast=int)
    _env_override(config_data, "agent", "max_steps", "AGENT_MAX_STEPS", cast=int)
    _env_override(config_data, "session", "max_depth", "AGENT_MAX_DEPTH", cast=int)

    return Settings(**config_data)


def _env_override(
    data: dict, section: str, key: str, env_key: str, cast=str
) -> None:
    """Apply a single env-var override into the nested config dict."""
    value = os.environ.get(env_key)
    if value is not None:
        data.setdefault(section, {})[key] = cast(value)


@lru_cache
def get_settings() -> Settings:
    """Cached singleton – safe to call from anywhere."""
    return load_settings()
