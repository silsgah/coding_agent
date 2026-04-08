.PHONY: install dev lint format test train serve infer clean

# ── Setup ─────────────────────────────────────────────────

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

# ── Quality ───────────────────────────────────────────────

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy --ignore-missing-imports .

test:
	pytest -v

# ── Run ───────────────────────────────────────────────────

serve:
	python main.py

serve-dev:
	uvicorn interface.api:app --reload --host 0.0.0.0 --port 8000

# ── Training ──────────────────────────────────────────────

download-data:
	python scripts/download_dataset.py

train:
	python scripts/train_lora.py

infer:
	python -m scripts.infer --interactive

# ── Docker ────────────────────────────────────────────────

docker-build:
	docker build -t coding-agent .

docker-run:
	docker run --gpus all -p 8000:8000 coding-agent

# ── Clean ─────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -f processed_train.json processed_valid.json
