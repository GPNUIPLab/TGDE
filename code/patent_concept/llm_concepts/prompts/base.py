"""Base prompt template protocol for concept extraction."""

from typing import Protocol


class PromptTemplate(Protocol):
    """Interface for concept extraction prompt templates."""

    system_prompt: str
    user_template: str
    few_shot_examples: list[dict]

    def format_user(self, abstract: str) -> str:
        """Format the user message with the given abstract."""
        ...

    def format_few_shot(self) -> list[dict]:
        """Return few-shot demonstration messages."""
        ...


def build_messages(
    template: PromptTemplate,
    abstract: str,
    include_few_shot: bool = True,
) -> list[dict]:
    """Build the full message list for an API call.

    Returns a list of dicts with 'role' and 'content' keys,
    compatible with both OpenAI and Anthropic APIs.
    """
    messages = [{"role": "system", "content": template.system_prompt}]

    if include_few_shot and template.few_shot_examples:
        messages.extend(template.format_few_shot())

    messages.append({"role": "user", "content": template.format_user(abstract)})
    return messages
