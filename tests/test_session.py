"""Tests for domain/session.py and infrastructure/session_store.py."""

import json
from domain.session import Session, SessionEvent, SessionMemory
from infrastructure.session_store import SessionStore


# ── SessionMemory tests ───────────────────────────────────

def test_memory_remember_file():
    """Files should be tracked with LRU eviction."""
    mem = SessionMemory()
    for i in range(10):
        mem.remember_file(f"file_{i}.py", limit=3)
    assert len(mem.files) == 3
    assert mem.files[-1] == "file_9.py"


def test_memory_remember_file_dedup():
    """Re-adding a file should move it to the end."""
    mem = SessionMemory()
    mem.remember_file("a.py")
    mem.remember_file("b.py")
    mem.remember_file("a.py")
    assert mem.files == ["b.py", "a.py"]


def test_memory_remember_note():
    """Notes should be tracked with LRU eviction."""
    mem = SessionMemory()
    for i in range(8):
        mem.remember_note(f"note {i}", limit=3)
    assert len(mem.notes) == 3
    assert mem.notes[-1] == "note 7"


def test_memory_empty_note_ignored():
    """Empty notes should not be added."""
    mem = SessionMemory()
    mem.remember_note("")
    assert len(mem.notes) == 0


def test_memory_text():
    """Memory text should be a readable string."""
    mem = SessionMemory(task="fix tests", files=["a.py"], notes=["read a.py"])
    text = mem.text()
    assert "fix tests" in text
    assert "a.py" in text
    assert "read a.py" in text


def test_memory_roundtrip():
    """Memory should survive to_dict/from_dict."""
    mem = SessionMemory(task="build feature", files=["x.py"], notes=["started"])
    restored = SessionMemory.from_dict(mem.to_dict())
    assert restored.task == mem.task
    assert restored.files == mem.files
    assert restored.notes == mem.notes


# ── SessionEvent tests ────────────────────────────────────

def test_event_roundtrip():
    """Events should survive to_dict/from_dict."""
    event = SessionEvent(
        role="tool",
        content="result text",
        name="read_file",
        args={"path": "test.py"},
    )
    restored = SessionEvent.from_dict(event.to_dict())
    assert restored.role == event.role
    assert restored.content == event.content
    assert restored.name == event.name
    assert restored.args == event.args


def test_event_user():
    """User events should not have tool-specific fields."""
    event = SessionEvent(role="user", content="hello")
    d = event.to_dict()
    assert "name" not in d
    assert "args" not in d


# ── Session tests ─────────────────────────────────────────

def test_session_record():
    """Recording an event should append to history."""
    session = Session()
    session.record(SessionEvent(role="user", content="hello"))
    assert len(session.history) == 1
    assert session.history[0].content == "hello"


def test_session_reset():
    """Reset should clear history and memory."""
    session = Session()
    session.record(SessionEvent(role="user", content="hello"))
    session.memory.task = "test"
    session.reset()
    assert len(session.history) == 0
    assert session.memory.task == ""


def test_session_roundtrip():
    """Sessions should survive to_dict/from_dict."""
    session = Session(workspace_root="/tmp/test")
    session.record(SessionEvent(role="user", content="fix bug"))
    session.memory.task = "fix bug"

    restored = Session.from_dict(session.to_dict())
    assert restored.id == session.id
    assert restored.workspace_root == session.workspace_root
    assert len(restored.history) == 1
    assert restored.memory.task == "fix bug"


# ── SessionStore tests ────────────────────────────────────

def test_store_save_and_load(tmp_path):
    """Sessions should persist to disk and load back."""
    store = SessionStore(tmp_path / "sessions")
    session = Session(workspace_root="/tmp/repo")
    session.record(SessionEvent(role="user", content="hello"))

    store.save(session)
    loaded = store.load(session.id)

    assert loaded.id == session.id
    assert len(loaded.history) == 1
    assert loaded.history[0].content == "hello"


def test_store_latest(tmp_path):
    """latest() should return the most recently saved session."""
    store = SessionStore(tmp_path / "sessions")

    s1 = Session()
    s1.record(SessionEvent(role="user", content="first"))
    store.save(s1)

    s2 = Session()
    s2.record(SessionEvent(role="user", content="second"))
    store.save(s2)

    assert store.latest() == s2.id


def test_store_list_sessions(tmp_path):
    """list_sessions should return all IDs."""
    store = SessionStore(tmp_path / "sessions")

    for _ in range(3):
        s = Session()
        store.save(s)

    sessions = store.list_sessions()
    assert len(sessions) == 3


def test_store_load_missing(tmp_path):
    """Loading a missing session should raise FileNotFoundError."""
    store = SessionStore(tmp_path / "sessions")
    try:
        store.load("nonexistent")
        assert False, "Should have raised"
    except FileNotFoundError:
        pass
