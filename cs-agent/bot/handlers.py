"""
消息路由处理器 — 根据消息类型分发到不同处理逻辑
"""
import re
from pathlib import Path
from loguru import logger

from core.agent import CSAgent
from bot.card_builder import CardBuilder


class MessageHandler:
    """
    消息路由处理器

    功能:
    - 解析用户消息（文本/文件/命令）
    - 路由到对应的处理逻辑
    - 格式化回复内容
    """

    # 命令模式
    CMD_PATTERN = re.compile(r"^/(\w+)\s*(.*)", re.DOTALL)

    def __init__(self, agent: CSAgent):
        self.agent = agent
        self.card_builder = CardBuilder()

        # 用户当前选择的模型
        self._user_models: dict[str, str] = {}

    async def handle_text(self, user_id: str, text: str) -> dict:
        """
        处理文本消息

        Returns:
            飞书消息卡片 dict
        """
        # 检查是否是命令
        match = self.CMD_PATTERN.match(text.strip())
        if match:
            cmd = match.group(1).lower()
            args = match.group(2).strip()
            return await self._handle_command(user_id, cmd, args)

        # 普通对话
        try:
            model = self._user_models.get(user_id)
            response = await self.agent.chat(
                user_message=text,
                user_id=user_id,
                model_name=model,
            )
            return self.card_builder.text_card("💬 CS-Agent", response, "blue")
        except Exception as e:
            logger.error(f"对话处理失败: {e}")
            return self.card_builder.error_card(f"处理失败: {str(e)}")

    async def handle_file(self, user_id: str, file_path: str, file_name: str) -> dict:
        """
        处理文件消息

        用户上传文件后，解析文件内容并提示用户提问。
        """
        try:
            from knowledge.loader import DocumentLoader
            loader = DocumentLoader()
            doc = loader.load_file(file_path)

            preview = doc.content[:1500]
            if len(doc.content) > 1500:
                preview += "\n\n... (内容已截断)"

            content = (
                f"**文件**: {file_name}\n"
                f"**类型**: {doc.metadata.get('type', '未知')}\n\n"
                f"**内容预览**:\n\n{preview}\n\n"
                f"💡 你可以针对文件内容提问，例如：\n"
                f"- 总结这个文件的主要内容\n"
                f"- 解释第3段的代码逻辑\n"
                f"- 这个算法的时间复杂度是多少？"
            )
            return self.card_builder.text_card(f"📄 文件已解析: {file_name}", content, "green")
        except Exception as e:
            logger.error(f"文件处理失败: {e}")
            return self.card_builder.error_card(f"文件解析失败: {str(e)}")

    async def _handle_command(self, user_id: str, cmd: str, args: str) -> dict:
        """处理命令"""
        if cmd == "help":
            return self.card_builder.help_card()

        elif cmd == "clear":
            self.agent.clear_memory(user_id)
            return self.card_builder.text_card("✅ 已清空", "对话历史已清空。", "green")

        elif cmd == "model":
            if not args:
                current = self._user_models.get(user_id, "默认模型")
                return self.card_builder.text_card(
                    "🔧 当前模型",
                    f"当前使用: `{current}`\n\n使用 `/model <模型名>` 切换，`/models` 查看可用列表。",
                )
            available_models = self.agent.llm_router.list_models()
            if args not in available_models:
                return self.card_builder.error_card(
                    f"模型 `{args}` 不可用。\n\n使用 `/models` 查看可用模型。"
                )
            self._user_models[user_id] = args
            return self.card_builder.text_card("✅ 模型已切换", f"已切换到: `{args}`", "green")

        elif cmd == "models":
            models = self.agent.llm_router.list_models()
            if not models:
                return self.card_builder.text_card(
                    "📋 可用模型",
                    "当前没有已配置 API Key 的模型。\n\n请在 `.env` 中配置模型凭据后重试。",
                )
            models_text = "\n".join(f"- `{m}`" for m in models)
            return self.card_builder.text_card(
                "📋 可用模型",
                f"{models_text}\n\n使用 `/model <模型名>` 切换。",
            )

        elif cmd == "paper":
            if not args:
                return self.card_builder.error_card("请提供搜索关键词，例如: `/paper Transformer attention`")
            if not self.agent.skill_registry:
                return self.card_builder.error_card("工具系统未初始化")
            result = await self.agent.skill_registry.execute("paper_search", query=args)
            return self.card_builder.text_card("📚 论文检索", result.content, "purple")

        elif cmd == "code":
            if not args:
                return self.card_builder.error_card("请提供代码，例如: `/code print('Hello')`")
            if not self.agent.skill_registry:
                return self.card_builder.error_card("工具系统未初始化")
            result = await self.agent.skill_registry.execute("code_executor", code=args)
            return self.card_builder.text_card("💻 代码执行", result.content, "green" if result.is_success else "red")

        else:
            return self.card_builder.error_card(f"未知命令: `/{cmd}`\n\n使用 `/help` 查看可用命令。")
