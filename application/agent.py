from infrastructure.vllm_model import VLLMModel
from infrastructure.tools import list_files, read_file, run_linter, run_tests
from pathlib import Path
import yaml

# Load repo config
config_path = Path(__file__).parent.parent / "config/config.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

class CodingAgent:
    def __init__(self, llm: VLLMModel):
        self.llm = llm

    async def generate_code(self, description: str):
        # Gather repo context
        files = list_files(cfg["repo"]["paths"], cfg["repo"]["file_types"])
        repo_context = "\n\n".join([read_file(f) for f in files[:5]])

        # Build prompt
        prompt = f"# Task:\n{description}\n\n# Repo Context:\n{repo_context}\n\n# Code Suggestion:\n"

        # Generate code
        code_chunks = []
        async for chunk in self.llm.generate(prompt):
            code_chunks.append(chunk)
        generated_code = "".join(code_chunks)

        # Run linter and tests
        lint_code, lint_out, lint_err = run_linter()
        test_code, test_out, test_err = run_tests()

        return {
            "code": generated_code,
            "lint": {"returncode": lint_code, "stdout": lint_out, "stderr": lint_err},
            "tests": {"returncode": test_code, "stdout": test_out, "stderr": test_err}
        }