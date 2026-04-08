"""
Context reduction and output management — Raschka Component 4.

Manages context window budget through clipping, transcript compression,
and deduplication. As Raschka says: "A lot of apparent model quality
is really context quality."
"""

from domain.session import SessionEvent


def clip(text: str, limit: int = 4000) -> str:
    """
    Truncate text to a character limit, appending a truncation notice.

    This is the most basic context reduction tool — prevents any single
    tool output or file content from consuming the entire context budget.
    """
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def build_history_text(
    events: list[SessionEvent],
    max_chars: int = 12000,
    recent_count: int = 6,
) -> str:
    """
    Build a compressed transcript for prompt injection.

    Implements three context reduction strategies from Raschka:

    1. Recency bias — recent events get more space, older events are
       compressed more aggressively.
    2. Deduplication — if the same file was read multiple times, only
       keep the most recent read (unless it was modified in between).
    3. Clipping — each event is truncated to a per-event budget.

    Args:
        events: Full session history (list of SessionEvent).
        max_chars: Total character budget for the transcript.
        recent_count: Number of recent events treated at "full resolution."

    Returns:
        Compressed transcript as a string.
    """
    if not events:
        return "- empty"

    lines: list[str] = []
    seen_reads: set[str] = set()
    recent_start = max(0, len(events) - recent_count)

    for index, event in enumerate(events):
        is_recent = index >= recent_start

        # Deduplication: if a file was written/patched, clear it from
        # seen reads so the next read is not suppressed
        if event.role == "tool" and event.name in ("write_file", "patch_file"):
            path = str((event.args or {}).get("path", ""))
            seen_reads.discard(path)

        # Deduplication: suppress repeated file reads for older events
        if event.role == "tool" and event.name == "read_file" and not is_recent:
            path = str((event.args or {}).get("path", ""))
            if path in seen_reads:
                continue
            seen_reads.add(path)

        # Render with recency-appropriate limits
        if event.role == "tool":
            limit = 900 if is_recent else 180
            args_str = ""
            if event.args:
                import json
                args_str = f" {json.dumps(event.args, sort_keys=True)}"
            lines.append(f"[tool:{event.name}]{args_str}")
            lines.append(clip(event.content, limit))
        else:
            limit = 900 if is_recent else 220
            lines.append(f"[{event.role}] {clip(event.content, limit)}")

    return clip("\n".join(lines), max_chars)
