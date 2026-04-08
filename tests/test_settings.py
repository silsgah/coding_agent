"""Tests for config/settings.py — validates config loading and env overrides."""

import os
from pathlib import Path
from unittest.mock import patch

from config.settings import load_settings, Settings


def test_default_settings():
    """Loading with no config file should return valid defaults."""
    settings = load_settings(config_path="/nonexistent/path.yaml")
    assert isinstance(settings, Settings)
    assert settings.model.name == "Qwen/Qwen2-7B"
    assert settings.model.device == "cuda"
    assert settings.model.max_new_tokens == 512
    assert settings.model.backend == "transformers"
    assert settings.model.lora.r == 16
    assert settings.api.port == 8000


def test_load_from_yaml(tmp_path):
    """Loading from a YAML file should populate settings correctly."""
    config = tmp_path / "config.yaml"
    config.write_text(
        "model:\n"
        "  name: test-model\n"
        "  device: cpu\n"
        "  max_new_tokens: 256\n"
        "api:\n"
        "  port: 9000\n"
    )
    settings = load_settings(config_path=str(config))
    assert settings.model.name == "test-model"
    assert settings.model.device == "cpu"
    assert settings.model.max_new_tokens == 256
    assert settings.api.port == 9000


def test_env_var_overrides(tmp_path):
    """Env vars should override YAML values."""
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  name: yaml-model\n  device: cuda\n")

    with patch.dict(os.environ, {
        "AGENT_MODEL_NAME": "env-model",
        "AGENT_DEVICE": "cpu",
    }):
        settings = load_settings(config_path=str(config))

    assert settings.model.name == "env-model"
    assert settings.model.device == "cpu"


def test_lora_defaults():
    """LoRA settings should have sensible defaults."""
    settings = load_settings(config_path="/nonexistent/path.yaml")
    lora = settings.model.lora
    assert lora.r == 16
    assert lora.alpha == 32
    assert "q_proj" in lora.target_modules
    assert lora.checkpoint_path == "lora_checkpoints"
