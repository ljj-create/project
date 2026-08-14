"""
飞书消息卡片构建器 — 构建结构化的消息卡片
"""


class CardBuilder:
    """
    飞书消息卡片构建器

    支持构建:
    - 文本消息卡片
    - 代码执行结果卡片
    - 论文搜索结果卡片
    - 错误提示卡片
    """

    @staticmethod
    def text_card(title: str, content: str, color: str = "blue") -> dict:
        """构建文本消息卡片"""
        return {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": color,
            },
            "elements": [
                {"tag": "markdown", "content": content},
            ],
        }

    @staticmethod
    def code_result_card(code: str, output: str, language: str = "python", success: bool = True) -> dict:
        """构建代码执行结果卡片"""
        color = "green" if success else "red"
        status = "✅ 执行成功" if success else "❌ 执行失败"

        content = f"**语言**: {language}\n**状态**: {status}\n\n"
        content += f"**代码**:\n```{language}\n{code}\n```\n\n"
        content += f"**输出**:\n```\n{output}\n```"

        return {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": "代码执行结果"},
                "template": color,
            },
            "elements": [
                {"tag": "markdown", "content": content},
            ],
        }

    @staticmethod
    def paper_card(papers: list[dict]) -> dict:
        """构建论文搜索结果卡片"""
        elements = []
        for i, paper in enumerate(papers[:5], 1):
            authors = paper.get("authors", [])
            authors_str = ", ".join(authors[:3])
            if len(authors) > 3:
                authors_str += " 等"

            content = (
                f"**{i}. {paper.get('title', '')}**\n"
                f"- 作者: {authors_str}\n"
                f"- 年份: {paper.get('year', '未知')}\n"
                f"- 来源: {paper.get('source', '')}\n"
                f"- [查看论文]({paper.get('url', '')})"
            )
            elements.append({"tag": "markdown", "content": content})
            elements.append({"tag": "hr"})

        return {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": "📚 论文检索结果"},
                "template": "purple",
            },
            "elements": elements,
        }

    @staticmethod
    def error_card(error_message: str) -> dict:
        """构建错误提示卡片"""
        return {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": "⚠️ 错误"},
                "template": "red",
            },
            "elements": [
                {"tag": "markdown", "content": error_message},
            ],
        }

    @staticmethod
    def help_card() -> dict:
        """构建帮助信息卡片"""
        content = """## 🤖 CS-Agent 使用指南

**基本对话**
直接发送消息即可开始对话。

**命令**
- `/help` — 显示帮助信息
- `/clear` — 清空对话历史
- `/model <模型名>` — 切换模型
- `/models` — 列出可用模型
- `/paper <关键词>` — 搜索论文
- `/code <代码>` — 执行代码
- `/upload` — 上传文件后提问

**支持的模型**
- `opencode-go/deepseek-v4-flash` - OpenCode Go
- `openai/gpt-4o` — GPT-4o
- `qwen/qwen-plus` — 通义千问
- `deepseek/deepseek-chat` — DeepSeek
- `ollama/qwen2.5:7b` — 本地模型

**支持的文件格式**
PDF、Word、Markdown、代码文件（.py/.cpp/.java 等）"""

        return {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": "🤖 CS-Agent 帮助"},
                "template": "blue",
            },
            "elements": [
                {"tag": "markdown", "content": content},
            ],
        }
