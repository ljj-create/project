"""
Ollama 本地模型适配器
"""
from typing import AsyncIterator
import httpx
from loguru import logger

from .base import BaseLLM, LLMMessage, LLMResponse


class OllamaLLM(BaseLLM):
    """
    Ollama 本地模型适配器

    通过 Ollama REST API 调用本地运行的模型。
    无需 API Key，适合离线使用。
    """

    def __init__(self, model_id: str, base_url: str = "http://localhost:11434"):
        super().__init__(model_id, api_key="", base_url=base_url)
        self.client = httpx.AsyncClient(base_url=base_url, timeout=120.0)

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
            payload = {
                "model": self.model_id,
                "messages": [m.to_dict() for m in messages],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            if tools:
                payload["tools"] = tools

            resp = await self.client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

            return LLMResponse(
                content=data.get("message", {}).get("content", ""),
                model=self.model_id,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                },
            )
        except Exception as e:
            logger.error(f"Ollama API 调用失败: {e}")
            raise

    async def chat_stream(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        try:
            payload = {
                "model": self.model_id,
                "messages": [m.to_dict() for m in messages],
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            }
            async with self.client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if data.get("done"):
                            break
        except Exception as e:
            logger.error(f"Ollama 流式 API 调用失败: {e}")
            raise
