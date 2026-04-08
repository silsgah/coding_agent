"""
Prompt builder — constructs well-structured prompts for code generation.

Keeps prompt construction separate from agent logic so it can be
tested and swapped independently.

Supports two modes:
  1. Simple mode (original): build(description, repo_context)
  2. Full agent mode (Raschka): build_agent_prompt(...) with workspace,
     memory, history, and tool descriptions.
"""


class PromptBuilder:
    """Builds structured prompts for the coding agent."""

    SYSTEM_PREFIX = (
        "You are an expert coding assistant. "
        "Given the task description and repository context below, "
        "produce clean, well-documented code.\n\n"
    )

    AGENT_SYSTEM_PREFIX = (
        "You are a coding agent with access to tools for reading files, "
        "writing code, searching, and running commands.\n\n"
    )

    AGENT_RULES = "\n".join([
        "Rules:",
        "- Use tools to inspect the workspace before making changes.",
        "- Return exactly one tool call or one final answer per response.",
        '- Tool calls: {"tool": "tool_name", "args": {...}}',
        '- Final answers: {"final": "your answer"}',
        "- Never invent tool results.",
        "- Keep answers concise and concrete.",
        "- Before writing tests, read the implementation first.",
        "- New files should be complete and runnable, including imports.",
        "- Do not repeat the same tool call if it did not help.",
    ])

    def build(self, description: str, repo_context: str = "") -> str:
        """
        Build a simple prompt from a task description and optional repo context.

        This is the original method — used by generate_code() and
        generate_code_stream() for single-shot generation.

        Args:
            description: What the user wants the code to do.
            repo_context: Relevant source files for grounding.

        Returns:
            A formatted prompt string.
        """
        parts = [self.SYSTEM_PREFIX]

        parts.append(f"### Task\n{description}\n")

        if repo_context.strip():
            parts.append(f"### Repository Context\n{repo_context}\n")

        parts.append("### Code\n")

        return "\n".join(parts)

    def build_agent_prompt(
        self,
        user_message: str,
        workspace_text: str = "",
        memory_text: str = "",
        history_text: str = "",
        tool_descriptions: str = "",
    ) -> str:
        """
        Build a full agent prompt with workspace, memory, history, and tools.

        Used by the interactive agent loop (ask()) for multi-turn,
        tool-using sessions. Implements Raschka's Component 2 (prompt
        shape and cache reuse).

        The prompt is structured in two zones:
          - Stable prefix: system instructions, rules, tools, workspace
            (rarely changes — cacheable)
          - Dynamic suffix: memory, history, current request
            (changes each turn)

        Args:
            user_message: The current user request.
            workspace_text: Workspace context (git, docs).
            memory_text: Working memory summary.
            history_text: Compressed conversation transcript.
            tool_descriptions: Available tools and their schemas.

        Returns:
            A formatted prompt string.
        """
        # ── Stable prefix (cacheable across turns) ──
        stable_parts = [self.AGENT_SYSTEM_PREFIX, self.AGENT_RULES]

        if tool_descriptions:
            stable_parts.append(f"\nAvailable tools:\n{tool_descriptions}")

        if workspace_text:
            stable_parts.append(f"\n{workspace_text}")

        # ── Dynamic suffix (changes each turn) ──
        dynamic_parts = []

        if memory_text:
            dynamic_parts.append(memory_text)

        if history_text:
            dynamic_parts.append(f"Transcript:\n{history_text}")

        dynamic_parts.append(f"Current request:\n{user_message}")

        return "\n\n".join(stable_parts + dynamic_parts)