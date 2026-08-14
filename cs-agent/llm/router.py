"""
LLM 路由器 — 根据任务类型和用户偏好选择模型
"""
from pathlib import Path
import yaml
from loguru import logger

from .base import BaseLLM
from .openai_llm import OpenAILLM
from .qwen_llm import QwenLLM
from .deepseek_llm import DeepSeekLLM
from .ollama_llm import OllamaLLM
from .mimo_llm import MimoLLM
from config.settings import settings, BASE_DIR


# Provider → 适配器类映射
PROVIDER_CLASSES = {
    "openai": OpenAILLM,
    "qwen": QwenLLM,
    "deepseek": DeepSeekLLM,
    "ollama": OllamaLLM,
    "mimo": MimoLLM,
    "opencode-go": OpenAILLM,
}

# Provider → API Key 配置映射
PROVIDER_KEYS = {
    "openai": ("openai_api_key", "openai_base_url"),
    "qwen": ("qwen_api_key", "qwen_base_url"),
    "deepseek": ("deepseek_api_key", "deepseek_base_url"),
    "ollama": ("", "ollama_base_url"),
    "mimo": ("mimo_api_key", "mimo_base_url"),
    "opencode-go": ("opencode_api_key", "opencode_base_url"),
}


class LLMRouter:
    """
    模型路由器

    功能:
    1. 根据 provider/model_id 格式的模型名创建 LLM 实例
    2. 根据任务类型（coding/research/general）自动选择最佳模型
    3. 缓存已创建的 LLM 实例
    """

    def __init__(self):
        self._instances: dict[str, BaseLLM] = {}
        self._routing: dict[str, str] = {}
        self._model_defs: dict[str, dict] = {}
        self._load_config()

    def _load_config(self):
        """加载 models.yaml 中的路由规则"""
        config_path = BASE_DIR / "config" / "models.yaml"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
                self._routing = config.get("routing", {})
                self._model_defs = config.get("models", {})
            logger.info(f"加载模型路由配置: {self._routing}")

    def get_llm(self, model_name: str | None = None, task_type: str = "general") -> BaseLLM:
        """
        获取 LLM 实例

        Args:
            model_name: 模型名，格式为 "provider/model_id"，如 "openai/gpt-4o"
                       若为 None，则根据 task_type 自动选择
            task_type: 任务类型，用于自动路由

        Returns:
            BaseLLM 实例
        """
        if model_name is None:
            model_name = self._routing.get(task_type, settings.default_model)
            logger.debug(f"自动路由 task_type={task_type} → {model_name}")

        if model_name in self._instances:
            return self._instances[model_name]

        llm = self._create_llm(model_name)
        self._instances[model_name] = llm
        return llm

    def _create_llm(self, model_name: str) -> BaseLLM:
        """根据 provider/model_id 创建 LLM 实例"""
        parts = model_name.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"模型名格式错误: '{model_name}'，应为 'provider/model_id'，如 'openai/gpt-4o'")

        provider, model_id = parts

        if provider not in PROVIDER_CLASSES:
            raise ValueError(f"不支持的 provider: '{provider}'，可选: {list(PROVIDER_CLASSES.keys())}")

        cls = PROVIDER_CLASSES[provider]
        key_attr, url_attr = PROVIDER_KEYS[provider]

        api_key = getattr(settings, key_attr, "") if key_attr else ""
        base_url = getattr(settings, url_attr, "") if url_attr else ""

        if provider != "ollama" and not api_key:
            logger.warning(f"模型 {model_name} 的 API Key 未配置")

        if provider == "ollama":
            llm = cls(model_id=model_id, base_url=base_url)
        else:
            llm = cls(model_id=model_id, api_key=api_key, base_url=base_url)

        logger.info(f"创建 LLM 实例: {llm}")
        return llm

    def list_models(self) -> list[str]:
        """列出所有可用模型"""
        models = []
        for provider in PROVIDER_CLASSES:
            key_attr, _ = PROVIDER_KEYS[provider]
            has_key = not key_attr or bool(getattr(settings, key_attr, ""))
            if has_key:
                for name, info in self._model_defs.items():
                    if info.get("provider") == provider:
                        models.append(name)
        return models
