"""
DeepSeek 模型适配器 — 基于 OpenAI 兼容接口
"""
from .openai_llm import OpenAILLM


class DeepSeekLLM(OpenAILLM):
    """
    DeepSeek 适配器

    DeepSeek API 完全兼容 OpenAI 格式，直接复用 OpenAILLM。
    """

    def __init__(self, model_id: str, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        super().__init__(model_id, api_key, base_url)
