"""Per-user agent session state.

A session owns the conversation memory, system prompt, and optional model
preference. Keeping session state separate from the agent loop makes the
runtime easier to test and naturally supports multiple concurrent users.
"""

from dataclasses import dataclass, field

from core.memory import ConversationMemory


@dataclass
class AgentSession:
    """State for one user conversation."""

    user_id: str
    memory: ConversationMemory = field(default_factory=ConversationMemory)
    model_name: str | None = None

    def set_system_prompt(self, prompt: str) -> None:
        self.memory.set_system_prompt(prompt)

    def set_model(self, model_name: str | None) -> None:
        self.model_name = model_name

    def clear(self) -> None:
        self.memory.clear()

    def add_turn(self, user_message: str, assistant_message: str, metadata: dict | None = None) -> None:
        self.memory.add_turn(user_message, assistant_message, metadata)
