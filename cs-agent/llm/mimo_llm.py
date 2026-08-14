"""
Mimo 模型适配器 — 小米大模型，使用 Anthropic API 格式
"""
from typing import AsyncIterator
import httpx
from loguru import logger

from .base import BaseLLM, LLMMessage, LLMResponse, MessageRole


class MimoLLM(BaseLLM):
    """
    Mimo (小米) 适配器

    使用 Anthropic Messages API 格式:
    - POST /v1/messages
    - 请求/响应格式与 Claude API 一致
    """

    def __init__(self, model_id: str, api_key: str, base_url: str = "https://token-plan-cn.xiaomimimo.com/anthropic"):
        super().__init__(model_id, api_key, base_url)
        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    async def close(self):
        """关闭 httpx 客户端"""
        await self.client.aclose()

    async def chat(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        try:
            # 转换为 Anthropic 格式
            system_prompt, anthropic_messages = self._convert_messages(messages)

            payload = {
                "model": self.model_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": anthropic_messages,
            }
            if system_prompt:
                payload["system"] = system_prompt

            resp = await self.client.post("/v1/messages", json=payload)
            resp.raise_for_status()
            data = resp.json()

            # 解析响应
            content = ""
            for block in data.get("content", []):
                if block.get("type") == "text":
                    content += block.get("text", "")

            usage = data.get("usage", {})
            return LLMResponse(
                content=content,
                model=data.get("model", self.model_id),
                usage={
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                },
                finish_reason=data.get("stop_reason", "end_turn"),
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Mimo API HTTP 错误: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Mimo API 调用失败: {e}")
            raise

    async def chat_stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        try:
            system_prompt, anthropic_messages = self._convert_messages(messages)

            payload = {
                "model": self.model_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": anthropic_messages,
                "stream": True,
            }
            if system_prompt:
                payload["system"] = system_prompt

            async with self.client.stream("POST", "/v1/messages", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        import json
                        data = json.loads(line[6:])
                        if data.get("type") == "content_block_delta":
                            text = data.get("delta", {}).get("text", "")
                            if text:
                                yield text
        except Exception as e:
            logger.error(f"Mimo 流式 API 调用失败: {e}")
            raise

    def _convert_messages(self, messages: list[LLMMessage]) -> tuple[str, list[dict]]:
        """
        将统一格式转换为 Anthropic 格式

        Returns:
            (system_prompt, messages) 元组
        """
        system_prompt = ""
        anthropic_messages = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_prompt = msg.content
            elif msg.role == MessageRole.USER:
                anthropic_messages.append({"role": "user", "content": msg.content})
            elif msg.role == MessageRole.ASSISTANT:
                anthropic_messages.append({"role": "assistant", "content": msg.content})

        # Anthropic 要求第一条消息必须是 user 角色
        if anthropic_messages and anthropic_messages[0]["role"] != "user":
            anthropic_messages.insert(0, {"role": "user", "content": "请开始对话。"})

        # 合并连续的同角色消息
        merged = []
        for msg in anthropic_messages:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append(msg)

        return system_prompt, merged
