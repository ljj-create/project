"""
网络搜索 Skill — 通过 DuckDuckGo 搜索互联网
"""
from loguru import logger

from skills.base import BaseSkill, SkillResult


class WebSearchSkill(BaseSkill):
    """
    网络搜索工具

    通过 DuckDuckGo 搜索互联网，获取最新信息。
    无需 API Key，免费使用。
    """

    name = "web_search"
    description = "搜索互联网获取最新信息。当需要查找最新新闻、技术文档、教程、实时数据时使用。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词",
            },
            "max_results": {
                "type": "integer",
                "description": "最大返回结果数，默认 5",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs) -> SkillResult:
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)

        if not query.strip():
            return SkillResult.error("搜索关键词不能为空")

        logger.info(f"网络搜索 | 查询: {query}")

        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))

            if not results:
                return SkillResult.success("未找到相关结果，请尝试更换关键词。")

            # 格式化输出
            parts = [f"## 网络搜索结果（共 {len(results)} 条）\n"]
            for i, r in enumerate(results, 1):
                parts.append(f"### {i}. {r.get('title', '')}\n")
                parts.append(f"- **链接**: {r.get('href', '')}\n")
                parts.append(f"- **摘要**: {r.get('body', '')}\n")

            result_text = "\n".join(parts)
            return SkillResult.success(result_text, results_count=len(results))

        except ImportError:
            return SkillResult.error("duckduckgo-search 未安装，请运行: pip install duckduckgo-search")
        except Exception as e:
            logger.error(f"网络搜索失败: {e}")
            return SkillResult.error(f"搜索失败: {str(e)}")
