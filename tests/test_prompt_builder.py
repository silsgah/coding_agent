"""Tests for application/prompt_builder.py."""

from application.prompt_builder import PromptBuilder


def test_build_with_context():
    """Prompt should include task, context, and code sections."""
    builder = PromptBuilder()
    prompt = builder.build("Sort a list", repo_context="def foo(): pass")
    assert "### Task" in prompt
    assert "Sort a list" in prompt
    assert "### Repository Context" in prompt
    assert "def foo(): pass" in prompt
    assert "### Code" in prompt


def test_build_without_context():
    """Prompt without context should omit the context section."""
    builder = PromptBuilder()
    prompt = builder.build("Sort a list")
    assert "### Task" in prompt
    assert "Sort a list" in prompt
    assert "### Repository Context" not in prompt
    assert "### Code" in prompt


def test_build_includes_system_prefix():
    """The system prefix should be at the start of every prompt."""
    builder = PromptBuilder()
    prompt = builder.build("anything")
    assert prompt.startswith(PromptBuilder.SYSTEM_PREFIX)
