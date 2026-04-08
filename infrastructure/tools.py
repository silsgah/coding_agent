"""
Tool functions for the coding agent.

All filesystem operations are sandboxed to configured repo paths.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# ── Maximum file size to read (prevent OOM on huge files) ──
MAX_FILE_SIZE_BYTES = 512 * 1024  # 512 KB


def _is_safe_path(file_path: Path, allowed_roots: List[str]) -> bool:
    """Check that file_path is inside one of the allowed root directories."""
    resolved = file_path.resolve()
    for root in allowed_roots:
        root_resolved = Path(root).resolve()
        if resolved == root_resolved or root_resolved in resolved.parents:
            return True
    return False


def list_files(
    repo_paths: List[str],
    file_types: List[str],
    max_files: int = 100,
) -> List[Path]:
    """Recursively list files matching the given extensions in repo paths."""
    files: list[Path] = []
    for path_str in repo_paths:
        p = Path(path_str)
        if not p.exists():
            logger.warning("Repo path does not exist: %s", p)
            continue
        for ft in file_types:
            for f in p.rglob(f"*{ft}"):
                if f.is_file():
                    files.append(f)
                    if len(files) >= max_files:
                        return files
    return files


def read_file(file_path: str, allowed_roots: List[str] | None = None) -> str:
    """
    Read a file's contents with safety checks.

    Args:
        file_path: Path to the file.
        allowed_roots: If provided, the file must be inside one of these dirs.

    Returns:
        File contents as a string, or an error message.
    """
    fp = Path(file_path)

    if allowed_roots and not _is_safe_path(fp, allowed_roots):
        msg = f"Access denied: {file_path} is outside allowed repo paths"
        logger.warning(msg)
        return f"# ERROR: {msg}"

    if not fp.exists():
        return f"# ERROR: File not found: {file_path}"

    if fp.stat().st_size > MAX_FILE_SIZE_BYTES:
        return f"# ERROR: File too large ({fp.stat().st_size} bytes): {file_path}"

    try:
        return fp.read_text(encoding="utf-8")
    except Exception as e:
        logger.error("Failed to read %s: %s", file_path, e)
        return f"# ERROR: Could not read {file_path}: {e}"


def apply_patch(patch_content: str, cwd: str = ".") -> tuple[int, str, str]:
    """Apply a git patch. Returns (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            ["git", "apply"],
            input=patch_content.encode(),
            capture_output=True,
            cwd=cwd,
            timeout=30,
        )
        return result.returncode, result.stdout.decode(), result.stderr.decode()
    except subprocess.TimeoutExpired:
        return -1, "", "Patch timed out after 30s"
    except Exception as e:
        return -1, "", str(e)


def run_tests(test_command: str = "pytest", cwd: str = ".") -> tuple[int, str, str]:
    """Run tests. Returns (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            test_command.split(),
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=120,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Tests timed out after 120s"
    except FileNotFoundError:
        return -1, "", f"Command not found: {test_command}"
    except Exception as e:
        return -1, "", str(e)


def run_linter(linter_command: str = "ruff check .", cwd: str = ".") -> tuple[int, str, str]:
    """Run linter. Returns (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            linter_command.split(),
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=60,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"Linter timed out after 60s"
    except FileNotFoundError:
        return -1, "", f"Command not found: {linter_command}"
    except Exception as e:
        return -1, "", str(e)


# ── New tools for interactive agent loop ────────────────────


def search_files(
    pattern: str,
    search_path: str = ".",
    allowed_roots: List[str] | None = None,
    max_results: int = 200,
) -> str:
    """
    Search for a pattern in files using ripgrep (or fallback).

    Args:
        pattern: Search string.
        search_path: Directory or file to search within.
        allowed_roots: If provided, search_path must be inside one of these.
        max_results: Maximum number of matching lines to return.

    Returns:
        Matching lines as a string.
    """
    import shutil

    sp = Path(search_path).resolve()

    if allowed_roots and not _is_safe_path(sp, allowed_roots):
        return f"error: access denied — {search_path} is outside allowed paths"

    # Prefer ripgrep for speed
    if shutil.which("rg"):
        try:
            result = subprocess.run(
                ["rg", "-n", "--smart-case", "--max-count", str(max_results), pattern, str(sp)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout.strip() or result.stderr.strip() or "(no matches)"
        except subprocess.TimeoutExpired:
            return "error: search timed out after 30s"

    # Fallback: simple substring search
    matches: list[str] = []
    files = [sp] if sp.is_file() else [
        f for f in sp.rglob("*")
        if f.is_file() and f.stat().st_size < MAX_FILE_SIZE_BYTES
    ]
    for fp in files:
        try:
            for num, line in enumerate(fp.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if pattern.lower() in line.lower():
                    matches.append(f"{fp}:{num}:{line}")
                    if len(matches) >= max_results:
                        return "\n".join(matches)
        except Exception:
            continue
    return "\n".join(matches) or "(no matches)"


def write_file_tool(
    file_path: str,
    content: str,
    allowed_roots: List[str] | None = None,
) -> str:
    """
    Write content to a file (sandboxed).

    Args:
        file_path: Path to write to.
        content: File contents.
        allowed_roots: If provided, file must be inside one of these dirs.

    Returns:
        Success message or error string.
    """
    fp = Path(file_path)

    if allowed_roots and not _is_safe_path(fp, allowed_roots):
        msg = f"Access denied: {file_path} is outside allowed repo paths"
        logger.warning(msg)
        return f"error: {msg}"

    if fp.exists() and fp.is_dir():
        return f"error: {file_path} is a directory"

    try:
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
        return f"wrote {file_path} ({len(content)} chars)"
    except Exception as e:
        logger.error("Failed to write %s: %s", file_path, e)
        return f"error: could not write {file_path}: {e}"


def patch_file_tool(
    file_path: str,
    old_text: str,
    new_text: str,
    allowed_roots: List[str] | None = None,
) -> str:
    """
    Replace one exact occurrence of old_text with new_text in a file (sandboxed).

    Args:
        file_path: Path to the file.
        old_text: Exact text to find (must occur exactly once).
        new_text: Replacement text.
        allowed_roots: If provided, file must be inside one of these dirs.

    Returns:
        Success message or error string.
    """
    fp = Path(file_path)

    if allowed_roots and not _is_safe_path(fp, allowed_roots):
        msg = f"Access denied: {file_path} is outside allowed repo paths"
        logger.warning(msg)
        return f"error: {msg}"

    if not fp.is_file():
        return f"error: file not found: {file_path}"

    try:
        text = fp.read_text(encoding="utf-8")
        count = text.count(old_text)
        if count == 0:
            return f"error: old_text not found in {file_path}"
        if count > 1:
            return f"error: old_text occurs {count} times (must be exactly 1)"
        fp.write_text(text.replace(old_text, new_text, 1), encoding="utf-8")
        return f"patched {file_path}"
    except Exception as e:
        logger.error("Failed to patch %s: %s", file_path, e)
        return f"error: could not patch {file_path}: {e}"