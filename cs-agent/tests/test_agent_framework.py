"""
Unified Agent framework tests.

These tests cover the new layered architecture:
    AgentSession -> AgentLoop -> ToolManager -> SkillRegistry
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.base import LLMMessage, LLMResponse, MessageRole, ToolCall
from core.loop import AgentLoop
from core.memory import ConversationMemory
from core.session import AgentSession
from core.tools import ToolManager
from core.types import AgentStatus
from skills.base import SkillResult


class FakeLLM:
    """Returns a predefined response sequence."""

    def __init__(self, responses):
        self.responses = list(responses)

    async def chat(self, messages, temperature=0.7, max_tokens=4096, tools=None):
        return self.responses.pop(0)

    async def chat_stream(self, messages, temperature=0.7, max_tokens=4096):
        for chunk in self.responses.pop(0).content:
            yield chunk


class FakeRouter:
    def __init__(self, llm):
        self.llm = llm

    def get_llm(self, model_name=None):
        return self.llm


class FakeRegistry:
    """Small registry fake that only exercises the ToolManager contract."""

    def __init__(self):
        self.executed = []

    def __len__(self):
        return 1

    def get_tool_definitions(self):
        return [{
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }]

    def list_skills(self):
        return [{"name": "echo", "description": "Echo tool"}]

    async def execute(self, name, **kwargs):
        self.executed.append((name, kwargs))
        return SkillResult.success("ok")


class TestAgentSession:
    def test_session_state_is_isolated(self):
        session = AgentSession(user_id="u1")
        session.set_system_prompt("sys")
        session.set_model("fake/model")
        session.add_turn("q", "a")

        messages = session.memory.get_messages()
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[1].content == "q"
        assert messages[2].content == "a"
        assert session.model_name == "fake/model"

    def test_clear_keeps_system_prompt(self):
        session = AgentSession(user_id="u1")
        session.set_system_prompt("sys")
        session.add_turn("q", "a")
        session.clear()
        assert len(session.memory.turns) == 0
        assert session.memory.system_prompt == "sys"


class TestToolManager:
    def test_tool_definitions_and_execution(self):
        registry = FakeRegistry()
        manager = ToolManager(registry)

        assert manager.has_tools()
        assert len(manager.get_tool_definitions()) == 1
        assert manager.list_tools()[0]["name"] == "echo"

        result = asyncio.run(manager.execute("echo", {"value": 1}))
        assert result.is_success
        assert registry.executed == [("echo", {"value": 1})]
        assert "echo" in manager.format_result("echo", result)


class TestAgentLoop:
    def test_plain_answer(self):
        memory = ConversationMemory()
        memory.set_system_prompt("sys")
        llm = FakeLLM([LLMResponse(content="answer", model="fake", usage={})])
        loop = AgentLoop(
            llm_router=FakeRouter(llm),
            tool_manager=ToolManager(FakeRegistry()),
            retriever=None,
            max_tool_rounds=2,
        )

        result = asyncio.run(loop.run(memory, "hi", "fake", use_rag=False))
        assert result.status == AgentStatus.SUCCESS
        assert result.content == "answer"
        assert len(memory.turns) == 1

    def test_tool_call_then_final_answer(self):
        memory = ConversationMemory()
        registry = FakeRegistry()
        tool_call = ToolCall(id="call_1", name="echo", arguments={"value": 1})
        llm = FakeLLM([
            LLMResponse(content="", model="fake", tool_calls=[tool_call]),
            LLMResponse(content="done", model="fake"),
        ])
        loop = AgentLoop(
            llm_router=FakeRouter(llm),
            tool_manager=ToolManager(registry),
            retriever=None,
            max_tool_rounds=2,
        )

        result = asyncio.run(loop.run(memory, "hi", "fake", use_rag=False))
        assert result.status == AgentStatus.SUCCESS
        assert result.content == "done"
        assert len(result.steps) == 1
        assert result.steps[0].name == "echo"
        assert registry.executed == [("echo", {"value": 1})]

    def test_tool_call_message_serialization(self):
        tool_call = ToolCall(id="call_1", name="echo", arguments={"value": 1})
        message = LLMMessage(role=MessageRole.ASSISTANT, content="", tool_calls=[tool_call])
        payload = message.to_dict()
        assert payload["tool_calls"][0]["function"]["arguments"] == '{"value": 1}'


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
