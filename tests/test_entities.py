"""Tests for domain/entities.py."""

from domain.entities import Task, GenerationResult, LintResult, TestRunResult


def test_task_defaults():
    t = Task(description="Do something")
    assert t.description == "Do something"
    assert t.repo_context == ""


def test_generation_result_success():
    r = GenerationResult(
        code="def add(a, b): return a + b",
        lint=LintResult(returncode=0, stdout="ok", stderr=""),
        tests=TestRunResult(returncode=0, stdout="passed", stderr=""),
    )
    assert r.error is None
    assert r.code == "def add(a, b): return a + b"
    assert r.lint.returncode == 0
    assert r.tests.returncode == 0


def test_generation_result_error():
    r = GenerationResult(error="Model not loaded")
    assert r.code == ""
    assert r.error == "Model not loaded"
