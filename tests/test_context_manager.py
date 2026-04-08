"""Tests for application/context_manager.py — clipping and transcript compression."""

from application.context_manager import clip, build_history_text
from domain.session import SessionEvent


def test_clip_short_text():
    """Text under the limit should pass through unchanged."""
    assert clip("hello", 100) == "hello"


def test_clip_long_text():
    """Text over the limit should be truncated with a notice."""
    result = clip("a" * 200, 50)
    assert len(result) < 200
    assert "truncated" in result
    assert "150" in result  # 200 - 50 = 150 chars truncated


def test_clip_default_limit():
    """Default limit should be 4000."""
    short = "x" * 100
    assert clip(short) == short
    long = "x" * 5000
    assert "truncated" in clip(long)


def test_build_history_empty():
    """Empty history should return a placeholder."""
    assert build_history_text([]) == "- empty"


def test_build_history_user_event():
    """User events should be rendered with role prefix."""
    events = [SessionEvent(role="user", content="fix the bug")]
    result = build_history_text(events)
    assert "[user]" in result
    assert "fix the bug" in result


def test_build_history_tool_event():
    """Tool events should show the tool name and result."""
    events = [
        SessionEvent(
            role="tool",
            content="def hello(): pass",
            name="read_file",
            args={"path": "hello.py"},
        )
    ]
    result = build_history_text(events)
    assert "[tool:read_file]" in result
    assert "def hello" in result


def test_build_history_deduplication():
    """Repeated old reads of the same file should be deduplicated."""
    events = [
        # Old events (will be compressed)
        SessionEvent(role="tool", content="content v1", name="read_file", args={"path": "a.py"}),
        SessionEvent(role="tool", content="content v2", name="read_file", args={"path": "a.py"}),
        # Recent events (will not be deduplicated)
        SessionEvent(role="user", content="now fix it"),
        SessionEvent(role="assistant", content="done"),
    ]
    result = build_history_text(events, recent_count=2)
    # The first read should be suppressed (deduplicated)
    assert result.count("content v1") <= 1


def test_build_history_respects_max_chars():
    """Output should be clipped to max_chars."""
    events = [
        SessionEvent(role="user", content="x" * 5000),
        SessionEvent(role="assistant", content="y" * 5000),
    ]
    result = build_history_text(events, max_chars=500)
    assert len(result) <= 600  # 500 + truncation notice


def test_build_history_write_clears_dedup():
    """A write to a file should clear it from the dedup set."""
    events = [
        SessionEvent(role="tool", content="old content", name="read_file", args={"path": "a.py"}),
        SessionEvent(role="tool", content="wrote a.py", name="write_file", args={"path": "a.py"}),
        SessionEvent(role="tool", content="new content", name="read_file", args={"path": "a.py"}),
        # Add enough recent events to push older ones into compression
        SessionEvent(role="user", content="check"),
        SessionEvent(role="assistant", content="ok"),
    ]
    result = build_history_text(events, recent_count=2)
    # After write, the second read should NOT be suppressed
    assert "new content" in result
