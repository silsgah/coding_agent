"""Domain entities — pure data structures with no infrastructure dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Task:
    """A coding task submitted by the user."""

    description: str
    repo_context: str = ""


@dataclass
class LintResult:
    returncode: int = -1
    stdout: str = ""
    stderr: str = ""


@dataclass
class TestRunResult:
    returncode: int = -1
    stdout: str = ""
    stderr: str = ""


@dataclass
class GenerationResult:
    """The full result returned to the caller after code generation."""

    code: str = ""
    lint: Optional[LintResult] = None
    tests: Optional[TestRunResult] = None
    error: Optional[str] = None