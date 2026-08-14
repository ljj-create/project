"""Unified tool manager used by the agent loop.

Tools are currently implemented as skills, but the agent loop should not know
about that detail. This class exposes the minimal tool contract required by
the runtime: definitions, execution, and result formatting.
"""

from loguru import logger

from core.prompt import PromptBuilder
from skills.base import SkillResult
from skills.skill_registry import SkillRegistry


class ToolManager:
    """Adapter around SkillRegistry for the agent runtime."""

    def __init__(self, registry: SkillRegistry | None = None):
        self.registry = registry or SkillRegistry()

    def has_tools(self) -> bool:
        return len(self.registry) > 0

    def get_tool_definitions(self) -> list[dict]:
        if not self.has_tools():
            return []
        return self.registry.get_tool_definitions()

    def list_tools(self) -> list[dict]:
        return self.registry.list_skills()

    async def execute(self, name: str, args: dict | None = None) -> SkillResult:
        return await self.registry.execute(name, **(args or {}))

    def format_result(self, name: str, result: SkillResult) -> str:
        content = getattr(result, "content", str(result))
        return PromptBuilder.build_tool_result(name, content)

    def __bool__(self) -> bool:
        return self.has_tools()
