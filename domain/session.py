"""
Domain entities for session management — pure data structures.

Provides working memory, event logging, and session state for
multi-turn agent interactions (Raschka Component 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import uuid


def _now() -> str:
    """UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionMemory:
    """
    Distilled working memory — small, actively maintained summary
    of what matters across turns.

    Unlike the full transcript (which grows forever), working memory
    is compacted and pruned each turn.
    """

    task: str = ""
    files: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def remember_file(self, path: str, limit: int = 8) -> None:
        """Track a file in working memory, with LRU eviction."""
        path = str(path)
        if path in self.files:
            self.files.remove(path)
        self.files.append(path)
        del self.files[:-limit]

    def remember_note(self, note: str, limit: int = 5) -> None:
        """Add a note to working memory, with LRU eviction."""
        if not note:
            return
        if note in self.notes:
            self.notes.remove(note)
        self.notes.append(note)
        del self.notes[:-limit]

    def to_dict(self) -> dict:
        return {"task": self.task, "files": list(self.files), "notes": list(self.notes)}

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMemory":
        return cls(
            task=data.get("task", ""),
            files=list(data.get("files", [])),
            notes=list(data.get("notes", [])),
        )

    def text(self) -> str:
        """Render working memory as a prompt-injectable string."""
        notes = "\n".join(f"- {n}" for n in self.notes) or "- none"
        return "\n".join([
            "Memory:",
            f"- task: {self.task or '-'}",
            f"- files: {', '.join(self.files) or '-'}",
            "- notes:",
            notes,
        ])


@dataclass
class SessionEvent:
    """A single event in the session transcript."""

    role: str  # "user", "assistant", or "tool"
    content: str
    created_at: str = field(default_factory=_now)
    # Tool-specific fields (only set when role == "tool")
    name: Optional[str] = None
    args: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }
        if self.name is not None:
            d["name"] = self.name
        if self.args is not None:
            d["args"] = self.args
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "SessionEvent":
        return cls(
            role=data["role"],
            content=data["content"],
            created_at=data.get("created_at", _now()),
            name=data.get("name"),
            args=data.get("args"),
        )


@dataclass
class Session:
    """
    Full session state — the durable record of an agent interaction.

    Contains the complete transcript (append-only) and the compacted
    working memory (actively pruned).
    """

    id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
    created_at: str = field(default_factory=_now)
    workspace_root: str = ""
    history: list[SessionEvent] = field(default_factory=list)
    memory: SessionMemory = field(default_factory=SessionMemory)

    def record(self, event: SessionEvent) -> None:
        """Append an event to the transcript."""
        self.history.append(event)

    def reset(self) -> None:
        """Clear history and memory for a fresh start."""
        self.history.clear()
        self.memory = SessionMemory()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "workspace_root": self.workspace_root,
            "history": [e.to_dict() for e in self.history],
            "memory": self.memory.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(
            id=data["id"],
            created_at=data.get("created_at", _now()),
            workspace_root=data.get("workspace_root", ""),
            history=[SessionEvent.from_dict(e) for e in data.get("history", [])],
            memory=SessionMemory.from_dict(data.get("memory", {})),
        )
