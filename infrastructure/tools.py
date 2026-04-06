import subprocess
from pathlib import Path
from typing import List

def list_files(repo_paths: List[str], file_types: List[str]):
    files = []
    for path in repo_paths:
        p = Path(path)
        if not p.exists():
            continue
        for ft in file_types:
            files.extend(list(p.rglob(f"*{ft}")))
    return files

def read_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def apply_patch(patch_content: str):
    result = subprocess.run(["git", "apply"], input=patch_content.encode(), capture_output=True)
    return result.returncode, result.stdout.decode(), result.stderr.decode()

def run_tests(test_command: str = "pytest"):
    result = subprocess.run(test_command.split(), capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def run_linter(linter_command: str = "ruff ."):
    result = subprocess.run(linter_command.split(), capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr