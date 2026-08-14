"""
论文检索 Skill — 搜索 arXiv 和 Semantic Scholar 论文
"""
import asyncio
from dataclasses import dataclass
from loguru import logger

from skills.base import BaseSkill, SkillResult


@dataclass
class Paper:
    """论文信息"""
    title: str
    authors: list[str]
    abstract: str
    year: int | str
    url: str
    source: str  # "arxiv" or "semantic_scholar"
    venue: str = ""
    citation_count: int = 0


class PaperSearchSkill(BaseSkill):
    """
    论文检索工具

    支持:
    - arXiv 论文搜索（预印本，更新快）
    - Semantic Scholar 搜索（有引用数据）
    - 返回标题、作者、摘要、链接
    """

    name = "paper_search"
    description = "搜索学术论文。当用户需要查找论文、了解研究方向、获取文献综述时使用。支持关键词、作者名、论文标题搜索。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词或查询",
            },
            "max_results": {
                "type": "integer",
                "description": "最大返回数量，默认 5",
                "default": 5,
            },
            "source": {
                "type": "string",
                "enum": ["arxiv", "semantic_scholar", "both"],
                "description": "搜索来源，默认 both",
                "default": "both",
            },
        },
        "required": ["query"],
    }

    async def execute(self, **kwargs) -> SkillResult:
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)
        source = kwargs.get("source", "both")

        if not query.strip():
            return SkillResult.error("搜索关键词不能为空")

        logger.info(f"论文检索 | 查询: {query} | 来源: {source}")

        papers = []

        if source in ("arxiv", "both"):
            arxiv_papers = await self._search_arxiv(query, max_results)
            papers.extend(arxiv_papers)

        if source in ("semantic_scholar", "both"):
            ss_papers = await self._search_semantic_scholar(query, max_results)
            papers.extend(ss_papers)

        if not papers:
            return SkillResult.success("未找到相关论文，请尝试更换关键词。")

        # 格式化输出
        result_text = self._format_results(papers, max_results)
        return SkillResult.success(result_text, papers_count=len(papers))

    async def _search_arxiv(self, query: str, max_results: int) -> list[Paper]:
        """搜索 arXiv"""
        try:
            import arxiv
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            papers = []
            for result in client.results(search):
                papers.append(Paper(
                    title=result.title,
                    authors=[str(a) for a in result.authors],
                    abstract=result.summary[:500],
                    year=result.published.year,
                    url=result.entry_id,
                    source="arxiv",
                    venue=result.primary_category,
                ))
            return papers
        except Exception as e:
            logger.warning(f"arXiv 搜索失败: {e}")
            return []

    async def _search_semantic_scholar(self, query: str, max_results: int) -> list[Paper]:
        """搜索 Semantic Scholar"""
        try:
            import httpx
            url = "https://api.semanticscholar.org/graph/v1/paper/search"
            params = {
                "query": query,
                "limit": max_results,
                "fields": "title,authors,abstract,year,url,citationCount,venue",
            }
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            papers = []
            for item in data.get("data", []):
                papers.append(Paper(
                    title=item.get("title", ""),
                    authors=[a.get("name", "") for a in item.get("authors", [])],
                    abstract=(item.get("abstract") or "")[:500],
                    year=item.get("year", "未知"),
                    url=item.get("url", ""),
                    source="semantic_scholar",
                    venue=item.get("venue", ""),
                    citation_count=item.get("citationCount", 0),
                ))
            return papers
        except Exception as e:
            logger.warning(f"Semantic Scholar 搜索失败: {e}")
            return []

    def _format_results(self, papers: list[Paper], max_results: int) -> str:
        """格式化论文搜索结果"""
        papers = papers[:max_results]
        parts = [f"## 论文检索结果（共 {len(papers)} 篇）\n"]

        for i, paper in enumerate(papers, 1):
            authors_str = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_str += " 等"

            part = f"### {i}. {paper.title}\n"
            part += f"- **作者**: {authors_str}\n"
            part += f"- **年份**: {paper.year}\n"
            if paper.venue:
                part += f"- **来源**: {paper.venue}\n"
            if paper.citation_count:
                part += f"- **引用数**: {paper.citation_count}\n"
            part += f"- **链接**: {paper.url}\n"
            part += f"- **摘要**: {paper.abstract}\n"
            parts.append(part)

        return "\n".join(parts)
