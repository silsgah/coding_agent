class PromptBuilder:
    def build_prompt(self, task):
        prompt = f"# Task:\n{task.description}\n"
        if task.repo_context:
            prompt += f"\n# Repo context:\n{task.repo_context}\n"
        prompt += "\n# Code Suggestion:\n"
        return prompt