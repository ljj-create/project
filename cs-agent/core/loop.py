"""Provider-agnostic agent execution loop.

This is the unified runtime used by CSAgent. It follows the common
Codex/Claude Code shape:

    user message -> context -> model -> tool calls -> model -> answer

The loop does not know which provider or tool implementation is active; those
details are injected as LLMRouter and ToolManager adapters.
"""

from typing import AsyncIterator

from loguru import logger

from core.memory import ConversationMemory
from core.prompt import PromptBuilder
from core.tools import ToolManager
from core.types import AgentResult, AgentStep, AgentStatus
from llm.base import LLMMessage, MessageRole
from llm.router import LLMRouter


class AgentLoop:
    """Unified agent execution loop."""

    def __init__(
        self,
        llm_router: LLMRouter,
        tool_manager: ToolManager | None = None,
        retriever=None,
        max_tool_rounds: int = 4,
    ):
        self.llm_router = llm_router
        self.tool_manager = tool_manager
        self.retriever = retriever
        self.max_tool_rounds = max_tool_rounds

    async def run(
        self,
        memory: ConversationMemory,
        user_message: str,
        model_name: str,
        use_rag: bool = True,
    ) -> AgentResult:
        """Run one agent turn and persist the completed turn in memory."""
        messages = memory.get_messages()
        rag_context = await self._retrieve_context(user_message, use_rag)
        messages.append(
            LLMMessage(
                role=MessageRole.USER,
                content=self._build_user_content(user_message, rag_context),
            )
        )

        steps: list[AgentStep] = []
        tools = self.tool_manager.get_tool_definitions() if self.tool_manager else []

        for round_index in range(self.max_tool_rounds):
            llm = self.llm_router.get_llm(model_name)
            response = await self._safe_chat(llm, messages, tools)
            if response is None:
                return AgentResult.error("LLM call failed")

            if response.tool_calls and self.tool_manager:
                messages.append(
                    LLMMessage(
                        role=MessageRole.ASSISTANT,
                        content=response.content or "",
                        tool_calls=response.tool_calls,
                    )
                )

                for tool_call in response.tool_calls:
                    step = await self._execute_tool(messages, tool_call)
                    steps.append(step)

                continue

            memory.add_turn(user_message, response.content)
            return AgentResult.success(
                response.content,
                model=response.model,
                usage=response.usage,
                steps=steps,
            )

        # If we exhausted tool rounds, request a final answer without tools.
        llm = self.llm_router.get_llm(model_name)
        response = await self._safe_chat(llm, messages, [])
        if response is None:
            return AgentResult.error("LLM final call failed", steps=steps)

        memory.add_turn(user_message, response.content)
        return AgentResult.success(
            response.content,
            model=response.model,
            usage=response.usage,
            steps=steps,
        )

    async def run_stream(
        self,
        memory: ConversationMemory,
        user_message: str,
        model_name: str,
    ) -> AsyncIterator[str]:
        """Run a streaming turn and persist the completed answer."""
        messages = memory.get_messages()
        messages.append(LLMMessage(role=MessageRole.USER, content=user_message))

        llm = self.llm_router.get_llm(model_name)
        full_response = ""
        try:
            async for chunk in llm.chat_stream(messages):
                full_response += chunk
                yield chunk
        except Exception as exc:
            logger.error(f"Streaming LLM call failed: {exc}")
            yield f"\n[error: {exc}]"

            return

        memory.add_turn(user_message, full_response)

    async def _retrieve_context(self, user_message: str, use_rag: bool) -> str:
        if not use_rag or not self.retriever:
            return ""

        try:
            documents = await self.retriever.retrieve(user_message)
            if documents:
                logger.info(f"RAG retrieved {len(documents)} document chunks")
                return PromptBuilder.build_rag_context(documents)
        except Exception as exc:
            logger.warning(f"RAG retrieval failed: {exc}")

        return ""

    @staticmethod
    def _build_user_content(user_message: str, rag_context: str) -> str:
        if not rag_context:
            return user_message
        return f"{rag_context}\n\nUser question: {user_message}"



    async def _safe_chat(self, llm, messages: list[LLMMessage], tools: list[dict]):
        try:
            return await llm.chat(messages, tools=tools or None)
        except Exception as exc:
            logger.error(f"LLM call failed: {exc}")
            return None

    async def _execute_tool(self, messages: list[LLMMessage], tool_call) -> AgentStep:
        step = AgentStep(name=tool_call.name, args=tool_call.arguments)
        try:
            result = await self.tool_manager.execute(tool_call.name, tool_call.arguments)
            step.status = AgentStatus.SUCCESS if result.is_success else AgentStatus.ERROR
            step.output = result.content
            tool_content = self.tool_manager.format_result(tool_call.name, result)
        except Exception as exc:
            step.status = AgentStatus.ERROR
            step.output = str(exc)
            tool_content = PromptBuilder.build_tool_result(
                tool_call.name,
                f"Tool execution failed: {exc}",
            )

        messages.append(
            LLMMessage(
                role=MessageRole.TOOL,
                content=tool_content,
                tool_call_id=tool_call.id,
                name=tool_call.name,
            )
        )
        return step
