"""
Coding Agent — orchestrates prompt building, LLM generation,
and optional lint/test feedback.

Depends on the LLMPort protocol (not a concrete backend).

Supports two modes:
  1. Single-shot generation: generate_code() / generate_code_stream()
     (original functionality — unchanged)
  2. Interactive agent loop: ask() with tool use, session memory,
     and bounded subagent delegation (Raschka Components 3-6)
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Optional

from domain.ports import LLMPort
from domain.entities import Task, GenerationResult, LintResult, TestRunResult
from domain.session import Session, SessionEvent, SessionMemory
from application.prompt_builder import PromptBuilder
from application.context_manager import clip, build_history_text
from infrastructure.tools import (
    list_files, read_file, run_linter, run_tests,
    search_files, write_file_tool, patch_file_tool,
)
from config.settings import Settings

logger = logging.getLogger(__name__)


# ── Tool registry for the interactive agent loop ──────────

TOOL_REGISTRY = {
    "list_files": {
        "description": "List files in the workspace.",
        "schema": {"path": "str='.'"},
        "risky": False,
    },
    "read_file": {
        "description": "Read a file by path.",
        "schema": {"path": "str", "start": "int=1", "end": "int=200"},
        "risky": False,
    },
    "search": {
        "description": "Search for a pattern in workspace files.",
        "schema": {"pattern": "str", "path": "str='.'"},
        "risky": False,
    },
    "run_shell": {
        "description": "Run a shell command in the repo root.",
        "schema": {"command": "str", "timeout": "int=20"},
        "risky": True,
    },
    "write_file": {
        "description": "Write content to a file.",
        "schema": {"path": "str", "content": "str"},
        "risky": True,
    },
    "patch_file": {
        "description": "Replace one exact text block in a file.",
        "schema": {"path": "str", "old_text": "str", "new_text": "str"},
        "risky": True,
    },
}


def _build_tool_descriptions() -> str:
    """Render tool descriptions for prompt injection."""
    lines = []
    for name, info in TOOL_REGISTRY.items():
        fields = ", ".join(f"{k}: {v}" for k, v in info["schema"].items())
        risk = "approval required" if info["risky"] else "safe"
        lines.append(f"- {name}({fields}) [{risk}] {info['description']}")
    return "\n".join(lines)


class CodingAgent:
    """
    The main agent that ties together:
      1. Repo context gathering
      2. Prompt construction
      3. LLM generation (streamed)
      4. Optional lint + test feedback
      5. Interactive tool-use loop with session memory (Raschka)
    """

    def __init__(
        self,
        llm: LLMPort,
        settings: Settings,
        session: Optional[Session] = None,
        workspace_text: str = "",
        read_only: bool = False,
        depth: int = 0,
        max_depth: int = 1,
    ):
        self.llm = llm
        self.settings = settings
        self.prompt_builder = PromptBuilder()
        self.session = session or Session()
        self.workspace_text = workspace_text
        self.read_only = read_only
        self.depth = depth
        self.max_depth = max_depth
        self.tool_descriptions = _build_tool_descriptions()

    # ── Streaming generation (for the SSE / StreamingResponse path) ──

    async def generate_code_stream(self, description: str) -> AsyncIterator[str]:
        """Yield code chunks as they are generated."""
        prompt = self._build_prompt(description)
        logger.info("Generating code for: %.80s…", description)

        async for chunk in self.llm.generate(prompt):
            yield chunk

    # ── Full generation (returns a complete result with lint + tests) ──

    async def generate_code(self, description: str) -> GenerationResult:
        """Generate code, then run linter and tests. Returns a structured result."""
        try:
            prompt = self._build_prompt(description)
            logger.info("Generating code for: %.80s…", description)

            chunks: list[str] = []
            async for chunk in self.llm.generate(prompt):
                chunks.append(chunk)
            generated_code = "".join(chunks)

            # ── Optional lint + test feedback ──
            lint_result = None
            test_result = None

            if self.settings.agent.enable_tests:
                try:
                    rc, out, err = run_linter()
                    lint_result = LintResult(returncode=rc, stdout=out, stderr=err)
                except Exception as e:
                    logger.warning("Linter failed: %s", e)
                    lint_result = LintResult(returncode=-1, stderr=str(e))

                try:
                    rc, out, err = run_tests()
                    test_result = TestRunResult(returncode=rc, stdout=out, stderr=err)
                except Exception as e:
                    logger.warning("Tests failed: %s", e)
                    test_result = TestRunResult(returncode=-1, stderr=str(e))

            return GenerationResult(
                code=generated_code,
                lint=lint_result,
                tests=test_result,
            )

        except Exception as e:
            logger.exception("Code generation failed")
            return GenerationResult(error=str(e))

    # ── Interactive agent loop (Raschka Components 3-6) ──────

    async def ask(self, user_message: str) -> str:
        """
        Interactive agent loop — multi-turn with tool use and memory.

        This is the core agentic behavior from Raschka's framework:
        1. Record the user message in the session transcript
        2. Build a prompt with workspace, memory, and history
        3. Generate a response from the LLM
        4. Parse for tool calls or final answers
        5. If tool call: execute, record result, loop back to step 2
        6. If final answer: record and return

        The loop runs up to max_steps times before forcing a stop.
        """
        memory = self.session.memory
        max_steps = self.settings.agent.max_steps

        # Set the task if this is the first message
        if not memory.task:
            memory.task = clip(user_message.strip(), 300)

        # Record user message
        self.session.record(SessionEvent(role="user", content=user_message))

        tool_steps = 0

        while tool_steps < max_steps:
            # Build the full prompt with all context
            prompt = self.prompt_builder.build_agent_prompt(
                user_message=user_message,
                workspace_text=self.workspace_text,
                memory_text=memory.text(),
                history_text=build_history_text(
                    self.session.history,
                    max_chars=self.settings.session.max_history_chars,
                ),
                tool_descriptions=self.tool_descriptions,
            )

            # Generate response
            chunks: list[str] = []
            async for chunk in self.llm.generate(prompt):
                chunks.append(chunk)
            raw = "".join(chunks)

            # Parse the response
            parsed = self._parse_response(raw)

            if parsed["type"] == "tool":
                tool_steps += 1
                name = parsed["name"]
                args = parsed["args"]

                # Execute the tool
                result = self._run_tool(name, args)

                # Record in transcript + update memory
                self.session.record(SessionEvent(
                    role="tool",
                    content=result,
                    name=name,
                    args=args,
                ))
                self._update_memory(name, args, result)
                logger.info("Tool [%s] step %d/%d", name, tool_steps, max_steps)
                continue

            # Final answer
            final = parsed.get("content", raw).strip()
            self.session.record(SessionEvent(role="assistant", content=final))
            memory.remember_note(clip(final, 220))
            return final

        # Hit step limit
        final = f"Stopped after {max_steps} tool steps without a final answer."
        self.session.record(SessionEvent(role="assistant", content=final))
        return final

    # ── Tool execution ────────────────────────────────────────

    def _run_tool(self, name: str, args: dict) -> str:
        """Execute a tool by name with args. Returns the result string."""
        allowed_roots = self.settings.repo.paths

        try:
            if name == "list_files":
                path = args.get("path", ".")
                files = list_files([path], self.settings.repo.file_types)
                return "\n".join(str(f) for f in files) or "(empty)"

            elif name == "read_file":
                path = args.get("path", "")
                return read_file(path, allowed_roots=allowed_roots)

            elif name == "search":
                pattern = args.get("pattern", "")
                path = args.get("path", ".")
                return search_files(pattern, path, allowed_roots=allowed_roots)

            elif name == "run_shell":
                if self.read_only:
                    return "error: shell commands not allowed in read-only mode"
                command = args.get("command", "")
                timeout = int(args.get("timeout", 20))
                import subprocess
                result = subprocess.run(
                    command, shell=True,
                    capture_output=True, text=True,
                    timeout=min(timeout, 120),
                )
                output = f"exit_code: {result.returncode}\n"
                output += f"stdout:\n{result.stdout.strip() or '(empty)'}\n"
                output += f"stderr:\n{result.stderr.strip() or '(empty)'}"
                return clip(output, self.settings.session.max_tool_output)

            elif name == "write_file":
                if self.read_only:
                    return "error: file writes not allowed in read-only mode"
                path = args.get("path", "")
                content = args.get("content", "")
                return write_file_tool(path, content, allowed_roots=allowed_roots)

            elif name == "patch_file":
                if self.read_only:
                    return "error: file patches not allowed in read-only mode"
                path = args.get("path", "")
                old_text = args.get("old_text", "")
                new_text = args.get("new_text", "")
                return patch_file_tool(path, old_text, new_text, allowed_roots=allowed_roots)

            elif name == "delegate":
                return self._tool_delegate(args)

            else:
                return f"error: unknown tool '{name}'"

        except Exception as e:
            logger.error("Tool %s failed: %s", name, e)
            return f"error: tool {name} failed: {e}"

    # ── Subagent delegation (Raschka Component 6) ─────────────

    def _tool_delegate(self, args: dict) -> str:
        """
        Spawn a bounded, read-only child agent to investigate a subtask.

        The child inherits the model and workspace context but:
        - Cannot write files or run shell commands (read_only=True)
        - Has a limited step budget
        - Cannot spawn further children (depth check)
        """
        if self.depth >= self.max_depth:
            return "error: delegation depth limit reached"

        task = args.get("task", "").strip()
        if not task:
            return "error: delegate task must not be empty"

        max_child_steps = min(int(args.get("max_steps", 3)), 3)

        child = CodingAgent(
            llm=self.llm,
            settings=self.settings,
            workspace_text=self.workspace_text,
            read_only=True,
            depth=self.depth + 1,
            max_depth=self.max_depth,
        )

        # Give the child some context from the parent
        child.session.memory.task = task
        child.session.memory.remember_note(
            clip(build_history_text(self.session.history), 300)
        )

        # Override max_steps for the child
        import copy
        child_settings = copy.deepcopy(self.settings)
        child_settings.agent.max_steps = max_child_steps

        logger.info("Delegating to child agent: %.80s…", task)

        # Since ask() is async, we need to run it — but we're already
        # in an async context, so we can await directly
        # The caller (ask) is already async, so this works
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # We're already in a loop, create a task
            result = loop.run_until_complete(child.ask(task))
        except RuntimeError:
            # Fallback: create a new loop
            result = asyncio.run(child.ask(task))

        return f"delegate_result:\n{result}"

    # ── Memory updates ────────────────────────────────────────

    def _update_memory(self, name: str, args: dict, result: str) -> None:
        """Update working memory based on tool usage."""
        memory = self.session.memory
        path = args.get("path")

        if name in {"read_file", "write_file", "patch_file"} and path:
            memory.remember_file(str(path))

        note = f"{name}: {clip(result.replace(chr(10), ' '), 220)}"
        memory.remember_note(note)

    # ── Response parsing ──────────────────────────────────────

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """
        Parse an LLM response for tool calls or final answers.

        Looks for JSON with "tool" key (tool call) or "final" key (answer).
        Falls back to treating the entire response as a final answer.
        """
        raw = raw.strip()

        # Try to extract JSON from the response
        try:
            # Look for JSON block in the response
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                candidate = raw[json_start:json_end]
                parsed = json.loads(candidate)

                if isinstance(parsed, dict):
                    # Tool call: {"tool": "name", "args": {...}}
                    if "tool" in parsed:
                        return {
                            "type": "tool",
                            "name": str(parsed["tool"]),
                            "args": parsed.get("args", {}),
                        }
                    # Final answer: {"final": "answer text"}
                    if "final" in parsed:
                        return {
                            "type": "final",
                            "content": str(parsed["final"]),
                        }
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: treat entire response as final answer
        return {"type": "final", "content": raw}

    # ── Private helpers (original) ────────────────────────────

    def _build_prompt(self, description: str) -> str:
        """Gather repo context and build the full prompt."""
        cfg = self.settings
        repo_context = ""

        try:
            files = list_files(cfg.repo.paths, cfg.repo.file_types)
            context_files = files[: cfg.agent.max_context_files]

            if context_files:
                snippets = [
                    read_file(str(f), allowed_roots=cfg.repo.paths)
                    for f in context_files
                ]
                repo_context = "\n\n".join(snippets)
        except Exception as e:
            logger.warning("Could not gather repo context: %s", e)

        prompt = self.prompt_builder.build(description, repo_context)

        # Guard against excessively long prompts
        max_len = cfg.agent.max_prompt_length
        if len(prompt) > max_len:
            logger.warning(
                "Prompt truncated from %d to %d chars", len(prompt), max_len
            )
            prompt = prompt[:max_len]

        return prompt