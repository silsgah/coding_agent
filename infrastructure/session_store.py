"""
Session store — file-backed JSON persistence for agent sessions.

Adapted from Raschka's SessionStore (Component 5).
Sessions are stored as individual JSON files for easy inspection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from domain.session import Session

logger = logging.getLogger(__name__)


class SessionStore:
    """
    Persist and load sessions as JSON files.

    Directory layout:
        {root}/
            20260408-190000-a1b2c3.json
            20260408-191500-d4e5f6.json
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        logger.debug("Session store initialized at %s", self.root)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def save(self, session: Session) -> Path:
        """Save session state to disk. Returns the file path."""
        path = self._path(session.id)
        path.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")
        logger.debug("Session saved: %s", path)
        return path

    def load(self, session_id: str) -> Session:
        """Load a session from disk by ID."""
        path = self._path(session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        logger.debug("Session loaded: %s", session_id)
        return Session.from_dict(data)

    def latest(self) -> Optional[str]:
        """Return the ID of the most recently modified session, or None."""
        files = sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime)
        return files[-1].stem if files else None

    def list_sessions(self) -> list[str]:
        """Return all session IDs, sorted by modification time (newest first)."""
        files = sorted(self.root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [f.stem for f in files]
