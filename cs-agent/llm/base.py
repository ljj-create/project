"""
LLM 抽象基类 — 所有模型适配器继承此类
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator
from enum import Enum
import json


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class LLMMessage:
    """统一的消息格式"""
    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list["ToolCall"] | None = None

    def to_dict(self) -> dict:
        d = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = [tc.to_openai_dict() for tc in self.tool_calls]
        return d


@dataclass
class ToolCall:
    """工具调用"""
    id: str
    name: str
    arguments: dict

    def to_openai_dict(self) -> dict:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass
class LLMResponse:
    """统一的响应格式"""
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


class BaseLLM(ABC):
    """
    LLM 抽象基类

    所有模型适配器（OpenAI、Qwen、DeepSeek、Ollama）都继承此类，
    实现统一的 chat / chat_stream 接口。
    """

    def __init__(self, model_id: str, api_key: str = "", base_url: str = ""):
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """
        同步对话（非流式）

        Args:
            messages: 对话历史
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            tools: 工具定义列表（Function Calling 格式）

        Returns:
            LLMResponse 统一响应
        """
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        """
        流式对话 — 逐 chunk 返回文本

        Args:
            messages: 对话历史
            temperature: 温度参数
            max_tokens: 最大生成 token 数

        Yields:
            文本片段
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_id})"
