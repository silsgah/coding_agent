"""
Standalone inference script — test the model outside the API server.

Usage:
    python -m scripts.infer "Write a function to reverse a linked list"
    python -m scripts.infer --interactive     # REPL mode
"""

import sys
import asyncio
import logging
from pathlib import Path

# Allow running from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config.settings import load_settings  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def create_model(settings):
    """Create and load the appropriate LLM backend."""
    backend = settings.model.backend

    if backend == "vllm":
        from infrastructure.vllm_model import VLLMModel
        model = VLLMModel(settings)
    else:
        from infrastructure.llm_model import HFModel
        model = HFModel(settings)

    logger.info("Loading %s backend …", backend)
    model.load()
    logger.info("Model ready")
    return model


async def run_inference(model, prompt: str) -> str:
    """Run inference and print streamed output. Returns the full text."""
    print("\n--- Generated Code ---\n")
    chunks = []
    async for chunk in model.generate(prompt):
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    print("\n\n--- End ---\n")
    return "".join(chunks)


async def interactive_mode(model):
    """Interactive REPL loop."""
    print("\n🤖 Mini Coding Agent — Interactive Mode")
    print("   Type your coding task, or 'quit' to exit.\n")

    while True:
        try:
            task = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not task or task.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        prompt = (
            f"### Instruction:\n{task}\n\n### Response:\n"
        )
        await run_inference(model, prompt)


async def main():
    settings = load_settings()
    model = create_model(settings)

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        await interactive_mode(model)
    elif len(sys.argv) > 1:
        task = " ".join(sys.argv[1:])
        prompt = f"### Instruction:\n{task}\n\n### Response:\n"
        await run_inference(model, prompt)
    else:
        print("Usage:")
        print('  python -m scripts.infer "Write a Python function …"')
        print("  python -m scripts.infer --interactive")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
