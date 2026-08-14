"""
SQLite 数据库 — 对话历史持久化存储
"""
import aiosqlite
from pathlib import Path
from loguru import logger

from config.settings import settings, BASE_DIR

DB_PATH = BASE_DIR / "storage" / "cs_agent.db"


class Database:
    """
    SQLite 异步数据库

    存储:
    - 对话历史
    - 用户配置
    - 文件记录
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or str(DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    async def init(self):
        """初始化数据库表"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    model TEXT,
                    tokens_used INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_configs (
                    user_id TEXT PRIMARY KEY,
                    preferred_model TEXT,
                    extra_config TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS file_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_type TEXT,
                    file_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_user
                ON conversations(user_id, created_at)
            """)
            await db.commit()
        logger.info(f"数据库初始化完成: {self.db_path}")

    async def save_message(self, user_id: str, role: str, content: str, model: str = "", tokens: int = 0):
        """保存一条消息"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO conversations (user_id, role, content, model, tokens_used) VALUES (?, ?, ?, ?, ?)",
                (user_id, role, content, model, tokens),
            )
            await db.commit()

    async def get_history(self, user_id: str, limit: int = 50) -> list[dict]:
        """获取用户对话历史"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM conversations WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in reversed(rows)]

    async def get_user_config(self, user_id: str) -> dict:
        """获取用户配置"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM user_configs WHERE user_id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()
            return dict(row) if row else {}

    async def save_user_config(self, user_id: str, config: dict):
        """保存用户配置"""
        import json
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO user_configs (user_id, preferred_model, extra_config)
                   VALUES (?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                   preferred_model = excluded.preferred_model,
                   extra_config = excluded.extra_config,
                   updated_at = CURRENT_TIMESTAMP""",
                (user_id, config.get("model", ""), json.dumps(config)),
            )
            await db.commit()

    async def save_file_record(self, user_id: str, file_name: str, file_path: str, file_type: str = "", file_size: int = 0):
        """保存文件上传记录"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO file_records (user_id, file_name, file_path, file_type, file_size) VALUES (?, ?, ?, ?, ?)",
                (user_id, file_name, file_path, file_type, file_size),
            )
            await db.commit()

    async def get_stats(self, user_id: str) -> dict:
        """获取用户使用统计"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) as msg_count, SUM(tokens_used) as total_tokens FROM conversations WHERE user_id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()
            return {
                "message_count": row[0] if row else 0,
                "total_tokens": row[1] if row and row[1] else 0,
            }
