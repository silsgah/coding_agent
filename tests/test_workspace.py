"""Tests for infrastructure/workspace.py — workspace context gathering."""

from infrastructure.workspace import WorkspaceContext, _clip_doc


def test_clip_doc_short():
    """Short docs should pass through."""
    assert _clip_doc("hello", 100) == "hello"


def test_clip_doc_long():
    """Long docs should be truncated."""
    result = _clip_doc("a" * 2000, 100)
    assert len(result) < 2000
    assert "truncated" in result


def test_workspace_build(tmp_path):
    """WorkspaceContext.build should return a valid context."""
    # Create a minimal repo-like structure
    (tmp_path / "README.md").write_text("# Test Project\nHello world")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"')

    ctx = WorkspaceContext.build(str(tmp_path))
    assert ctx.cwd == str(tmp_path)
    assert ctx.repo_root  # Should have some value


def test_workspace_text(tmp_path):
    """text() should return a formatted string."""
    (tmp_path / "README.md").write_text("# Hello")

    ctx = WorkspaceContext.build(str(tmp_path))
    text = ctx.text()

    assert "Workspace:" in text
    assert "cwd:" in text
    assert "branch:" in text


def test_workspace_context_manual():
    """Manual construction should work."""
    ctx = WorkspaceContext(
        cwd="/tmp/test",
        repo_root="/tmp/test",
        branch="main",
        default_branch="main",
        status="clean",
        recent_commits=["abc123 initial commit"],
        project_docs={"README.md": "# Hello"},
    )
    text = ctx.text()
    assert "main" in text
    assert "initial commit" in text
    assert "README.md" in text
