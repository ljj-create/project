"""
CS-Agent 本地部署脚本

用法:
    python scripts/run.py              # 启动 Web 服务
    python scripts/run.py --init-db    # 初始化数据库
"""
import sys
import asyncio
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from config.settings import settings


def setup_logging():
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
    )
    logger.add(
        "storage/logs/cs_agent_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
    )


def create_app():
    """创建 FastAPI 应用"""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    from llm.router import LLMRouter
    from knowledge.vectorstore import VectorStore
    from knowledge.retriever import HybridRetriever
    from skills.skill_registry import SkillRegistry
    from skills.code_executor import CodeExecutorSkill
    from skills.paper_search import PaperSearchSkill
    from skills.file_qa import FileQASkill
    from skills.web_search import WebSearchSkill
    from core.agent import CSAgent
    from storage.database import Database

    app = FastAPI(title="CS-Agent", description="CS 硕士生智能体", version="1.0.0")

    # 初始化组件
    llm_router = LLMRouter()
    skill_registry = SkillRegistry()
    skill_registry.register(CodeExecutorSkill())
    skill_registry.register(PaperSearchSkill())
    skill_registry.register(FileQASkill())
    skill_registry.register(WebSearchSkill())

    # 知识库（可选，可能未初始化）
    try:
        vector_store = VectorStore()
        retriever = HybridRetriever(vector_store)
    except Exception as e:
        logger.warning(f"知识库初始化失败（RAG 功能不可用）: {e}")
        retriever = None

    agent = CSAgent(
        llm_router=llm_router,
        retriever=retriever,
        skill_registry=skill_registry,
        model_name=settings.default_model,
    )

    db = Database()

    @app.on_event("startup")
    async def startup():
        await db.init()
        logger.info("CS-Agent 启动完成")

    @app.post("/api/chat")
    async def chat(request: Request):
        """直接对话 API（用于调试）"""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(content={"error": "请求体不是有效的 JSON"}, status_code=400)

        user_id = body.get("user_id", "api_user")
        message = body.get("message", "")
        model = body.get("model")

        if not message:
            return JSONResponse(content={"error": "message 不能为空"}, status_code=400)

        try:
            response = await agent.chat(
                user_message=message,
                user_id=user_id,
                model_name=model,
            )
            return JSONResponse(content={"response": response})
        except Exception as e:
            logger.error(f"对话失败: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)

    @app.get("/api/models")
    async def list_models():
        """列出可用模型"""
        models = llm_router.list_models()
        return JSONResponse(content={"models": models})

    @app.get("/api/skills")
    async def list_skills():
        """列出可用工具"""
        skills = skill_registry.list_skills()
        return JSONResponse(content={"skills": skills})

    @app.get("/api/health")
    async def health():
        """健康检查"""
        return JSONResponse(content={"status": "ok", "version": "1.0.0"})

    return app


async def init_db():
    """初始化数据库"""
    from storage.database import Database
    db = Database()
    await db.init()
    logger.info("数据库初始化完成")


def main():
    parser = argparse.ArgumentParser(description="CS-Agent 启动脚本")
    parser.add_argument("--web-only", action="store_true", help="仅启动 Web 服务")
    parser.add_argument("--init-db", action="store_true", help="初始化数据库")
    parser.add_argument("--host", default=settings.host, help="监听地址")
    parser.add_argument("--port", type=int, default=settings.port, help="监听端口")
    args = parser.parse_args()

    setup_logging()

    if args.init_db:
        asyncio.run(init_db())
        return

    logger.info(f"启动 CS-Agent | {args.host}:{args.port}")

    app = create_app()

    import uvicorn
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
