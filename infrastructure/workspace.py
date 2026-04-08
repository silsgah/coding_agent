"""
Workspace context — gathers live repo facts for prompt injection.

Adapted from Raschka's WorkspaceContext (Component 1).
Collects git branch, status, recent commits, and project docs
so the agent starts each turn with situational awareness.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DOC_NAMES = ("AGENTS.md", "README.md", "pyproject.toml", "package.json")
MAX_DOC_CHARS = 1200

IGNORED_PATH_NAMES = {
    ".git", ".mini-coding-agent", "__pycache__",
    ".pytest_cache", ".ruff_cache", ".venv", "venv",
    "node_modules", ".mypy_cache",
}


def _git(args: list[str], cwd: Path, fallback: str = "") -> str:
    """Run a git command with timeout, returning fallback on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip() or fallback
    except Exception:
        return fallback


def _clip_doc(text: str, limit: int = MAX_DOC_CHARS) -> str:
    """Truncate a document for prompt injection."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


class WorkspaceContext:
    """
    Snapshot of live repo state, used to build the stable prompt prefix.

    This information rarely changes during a session, so it can be
    cached and reused across turns (Raschka's prompt cache reuse).
    """

    def __init__(
        self,
        cwd: str,
        repo_root: str,
        branch: str,
        default_branch: str,
        status: str,
        recent_commits: list[str],
        project_docs: dict[str, str],
    ):
        self.cwd = cwd
        self.repo_root = repo_root
        self.branch = branch
        self.default_branch = default_branch
        self.status = status
        self.recent_commits = recent_commits
        self.project_docs = project_docs

    @classmethod
    def build(cls, cwd: Optional[str] = None) -> "WorkspaceContext":
        """Gather workspace facts from the filesystem and git."""
        cwd_path = Path(cwd or ".").resolve()

        repo_root = Path(
            _git(["rev-parse", "--show-toplevel"], cwd_path, str(cwd_path))
        ).resolve()

        # Collect project docs from repo root and cwd
        docs: dict[str, str] = {}
        for base in (repo_root, cwd_path):
            for name in DOC_NAMES:
                path = base / name
                if not path.exists():
                    continue
                key = str(path.relative_to(repo_root))
                if key in docs:
                    continue
                try:
                    docs[key] = _clip_doc(
                        path.read_text(encoding="utf-8", errors="replace")
                    )
                except Exception as e:
                    logger.warning("Could not read %s: %s", path, e)

        # Parse recent commits
        raw_commits = _git(["log", "--oneline", "-5"], cwd_path)
        commits = [line for line in raw_commits.splitlines() if line]

        # Git status (clipped)
        status_raw = _git(["status", "--short"], cwd_path, "clean") or "clean"

        # Default branch
        default_ref = _git(
            ["symbolic-ref", "--short", "refs/remotes/origin/HEAD"],
            cwd_path,
            "origin/main",
        ) or "origin/main"
        default_branch = default_ref.removeprefix("origin/")

        return cls(
            cwd=str(cwd_path),
            repo_root=str(repo_root),
            branch=_git(["branch", "--show-current"], cwd_path, "-") or "-",
            default_branch=default_branch,
            status=_clip_doc(status_raw, 1500),
            recent_commits=commits,
            project_docs=docs,
        )

    def text(self) -> str:
        """Render workspace context as a prompt-injectable string."""
        commits = "\n".join(f"  - {c}" for c in self.recent_commits) or "  - none"
        docs = "\n".join(
            f"  - {path}\n{snippet}" for path, snippet in self.project_docs.items()
        ) or "  - none"
        return "\n".join([
            "Workspace:",
            f"  cwd: {self.cwd}",
            f"  repo_root: {self.repo_root}",
            f"  branch: {self.branch}",
            f"  default_branch: {self.default_branch}",
            "  status:",
            f"    {self.status}",
            "  recent_commits:",
            commits,
            "  project_docs:",
            docs,
        ])
