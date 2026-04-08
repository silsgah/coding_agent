# ─────────────────────────────────────────────────────────
# Multi-stage Dockerfile for Mini Coding Agent
# ─────────────────────────────────────────────────────────
# Build:  docker build -t coding-agent .
# Run:    docker run --gpus all -p 8000:8000 coding-agent
# ─────────────────────────────────────────────────────────

FROM python:3.12-slim AS base

WORKDIR /app

# System deps (git is needed for apply_patch tool)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (cache-friendly)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source
COPY . .

# ─────────────────────────────────────────────────────────
# Production image
# ─────────────────────────────────────────────────────────
FROM base AS production

ENV AGENT_DEVICE=cuda
ENV AGENT_BACKEND=transformers

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()"

CMD ["python", "main.py"]
