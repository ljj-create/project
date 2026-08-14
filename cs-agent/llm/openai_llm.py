"""
OpenAI 模型适配器 — 兼容所有 OpenAI API 格式的服务
"""
import json
from typing import AsyncIterator

from openai import AsyncOpenAI
from loguru import logger

from .base import BaseLLM, LLMMessage, LLMResponse, ToolCall


class OpenAILLM(BaseLLM):
    """
    OpenAI API 适配器

    同时兼容:
    - OpenAI 官方 API
    - 任何 OpenAI 兼容接口（如 vLLM、LiteLLM proxy）
    """

    def __init__(self, model_id: str, api_key: str, base_url: str = "https://api.openai.com/v1"):
        super().__init__(model_id, api_key, base_url)
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=30.0)

    async def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        try:
            kwargs = {
                "model": self.model_id,
                "messages": [m.to_dict() for m in messages],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = await self.client.chat.completions.create(**kwargs)
            if not response.choices:
                raise ValueError("API 返回了空的 choices 列表")
            choice = response.choices[0]

            # 解析工具调用
            tool_calls = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=arguments,
                    ))

            usage_data = response.usage
            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                usage={
                    "prompt_tokens": usage_data.prompt_tokens if usage_data else 0,
                    "completion_tokens": usage_data.completion_tokens if usage_data else 0,
                    "total_tokens": usage_data.total_tokens if usage_data else 0,
                },
                tool_calls=tool_calls,
                finish_reason=choice.finish_reason,
            )
        except Exception as e:
            logger.error(f"OpenAI API 调用失败: {e}")
            raise

    async def chat_stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        try:
            stream = await self.client.chat.completions.create(
                model=self.model_id,
                messages=[m.to_dict() for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI 流式 API 调用失败: {e}")
            raise
