"""
CS-Agent 核心模块测试
"""
import pytest
import asyncio

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.base import LLMMessage, LLMResponse, MessageRole
from core.memory import ConversationMemory
from core.prompt import PromptBuilder


class TestConversationMemory:
    """对话记忆测试"""

    def test_add_turn(self):
        memory = ConversationMemory(max_turns=5)
        memory.set_system_prompt("你是 CS 助手")
        memory.add_turn("你好", "你好！有什么可以帮你的？")

        messages = memory.get_messages()
        assert len(messages) == 3  # system + user + assistant
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[1].role == MessageRole.USER
        assert messages[2].role == MessageRole.ASSISTANT

    def test_max_turns_compression(self):
        memory = ConversationMemory(max_turns=3)
        memory.set_system_prompt("test")

        for i in range(5):
            memory.add_turn(f"问题{i}", f"回答{i}")

        assert len(memory.turns) <= 3

    def test_clear(self):
        memory = ConversationMemory()
        memory.add_turn("q", "a")
        memory.clear()
        assert len(memory.turns) == 0

    def test_export_import(self):
        memory = ConversationMemory()
        memory.add_turn("你好", "你好！")

        exported = memory.export_history()
        assert len(exported) == 1
        assert exported[0]["user"] == "你好"

        new_memory = ConversationMemory()
        new_memory.import_history(exported)
        assert len(new_memory.turns) == 1


class TestPromptBuilder:
    """Prompt 构建器测试"""

    def test_build_system_prompt(self):
        prompt = PromptBuilder.build_system_prompt()
        assert "CS-Agent" in prompt
        assert "硕士" in prompt

    def test_build_rag_context(self):
        docs = [
            {"content": "Transformer 是一种模型", "metadata": {"source": "ml.md"}},
            {"content": "BERT 基于 Transformer", "metadata": {"source": "bert.md"}},
        ]
        context = PromptBuilder.build_rag_context(docs)
        assert "Transformer" in context
        assert "ml.md" in context

    def test_build_tool_result(self):
        result = PromptBuilder.build_tool_result("code_executor", "Hello World")
        assert "code_executor" in result
        assert "Hello World" in result

    def test_empty_rag_context(self):
        context = PromptBuilder.build_rag_context([])
        assert context == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
