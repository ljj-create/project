"""
通义千问模型适配器 — 基于 OpenAI 兼容接口
"""
from .openai_llm import OpenAILLM


class QwenLLM(OpenAILLM):
    """
    通义千问适配器

    阿里云 DashScope 提供 OpenAI 兼容接口，直接复用 OpenAILLM。
    """

    def __init__(self, model_id: str, api_key: str, base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        super().__init__(model_id, api_key, base_url)
