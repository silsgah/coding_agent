# Mini Coding Agent

A UV-style modular coding agent using FastAPI and LLMs.  

Features:
- Custom coding agent with repo context awareness
- Config-driven model selection (CodeLlama or other)
- LoRA / PEFT fine-tuning support
- FastAPI + Uvicorn interface
- Tools for reading/writing/searching code, running tests

## Setup

```bash
uv add torch transformers peft fastapi uvicorn pyyaml
uv run uvicorn interface.api:app --reload