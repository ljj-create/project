"""CSAgent facade.

This module keeps the public CSAgent API used by the bot/scripts while the
execution details live in AgentLoop, AgentSession, ToolManager and
ConversationMemory. The split follows the Codex / Claude Code separation of
session state, tool runtime, and agent loop.
"""

from typing import AsyncIterator

from loguru import logger

from core.loop import AgentLoop
from core.memory import ConversationMemory
from core.prompt import PromptBuilder
from core.session import AgentSession
from core.tools import ToolManager
from llm.router import LLMRouter


class CSAgent:
    """Unified CS-Agent entry point.

    Responsibilities are intentionally narrow:
    1. Create and isolate per-user sessions.
    2. Resolve the model name.
    3. Delegate execution to AgentLoop.
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        retriever=None,
        skill_registry=None,
        model_name: str | None = None,
    ):
        self.llm_router = llm_router
        self.retriever = retriever
        self.skill_registry = skill_registry
        self.default_model = model_name

        self.tool_manager = ToolManager(skill_registry)
        self.loop = AgentLoop(
            llm_router=llm_router,
            tool_manager=self.tool_manager,
            retriever=retriever,
        )
        self.system_prompt = PromptBuilder.build_system_prompt()
        self._sessions: dict[str, AgentSession] = {}

        logger.info("CS-Agent initialized with unified AgentLoop")

    def get_session(self, user_id: str) -> AgentSession:
        """Return or create the session for a user."""
        if user_id not in self._sessions:
            session = AgentSession(user_id=user_id)
            session.set_system_prompt(self.system_prompt)
            session.set_model(self.default_model)
            self._sessions[user_id] = session
        return self._sessions[user_id]

    def _get_memory(self, user_id: str) -> ConversationMemory:
        return self.get_session(user_id).memory

    async def chat(
        self,
        user_message: str,
        user_id: str = "default",
        model_name: str | None = None,
        use_rag: bool = True,
    ) -> str:
        session = self.get_session(user_id)
        model = model_name or session.model_name or self.default_model
        session.set_model(model)

        result = await self.loop.run(
            memory=session.memory,
            user_message=user_message,
            model_name=model,
            use_rag=use_rag,
        )

        logger.info(
            f"chat complete | user: {user_id} | model: {model} | "
            f"input tokens: {result.usage.get('prompt_tokens', '?')} | "
            f"output tokens: {result.usage.get('completion_tokens', '?')} | "
            f"steps: {len(result.steps)}"
        )
        return result.content

    async def chat_stream(
        self,
        user_message: str,
        user_id: str = "default",
        model_name: str | None = None,
    ) -> AsyncIterator[str]:
        session = self.get_session(user_id)
        model = model_name or session.model_name or self.default_model
        session.set_model(model)

        async for chunk in self.loop.run_stream(
            memory=session.memory,
            user_message=user_message,
            model_name=model,
        ):
            yield chunk

    def clear_memory(self, user_id: str = "default"):
        session = self._sessions.get(user_id)
        if session:
            session.clear()

    def get_memory(self, user_id: str = "default") -> ConversationMemory:
        return self._get_memory(user_id)
