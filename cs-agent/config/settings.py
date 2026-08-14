"""
全局配置 — 从 .env 文件加载
"""
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """应用配置，自动从环境变量 / .env 文件读取"""

    # --- LLM ---
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"

    opencode_api_key: str = ""
    opencode_base_url: str = "https://opencode.ai/zen/go/v1"

    qwen_api_key: str = ""
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    ollama_base_url: str = "http://localhost:11434"

    mimo_api_key: str = ""
    mimo_base_url: str = "https://token-plan-cn.xiaomimimo.com/anthropic"

    default_model: str = "opencode-go/deepseek-v4-flash"

    # --- 飞书 ---
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_verification_token: str = ""
    feishu_encrypt_key: str = ""

    # --- 知识库 ---
    knowledge_db_path: str = str(BASE_DIR / "storage" / "chromadb")
    embedding_model: str = "shibing624/text2vec-base-chinese"
    chunk_size: int = 500
    chunk_overlap: int = 50

    # --- 代码执行 ---
    code_execution_timeout: int = 30
    docker_image: str = "python:3.11-slim"

    # --- 服务 ---
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    model_config = {"env_file": str(BASE_DIR / ".env"), "env_file_encoding": "utf-8"}


settings = Settings()
