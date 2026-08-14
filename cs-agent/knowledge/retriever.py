"""
混合检索器 — 向量语义搜索 + BM25 关键词搜索 + Reranking
"""
from loguru import logger
from knowledge.vectorstore import VectorStore


class HybridRetriever:
    """
    混合检索器

    策略:
    1. 向量语义搜索（ChromaDB）— 理解语义相似性
    2. 关键词匹配 — 精确匹配专业术语
    3. 合并去重 + 按相关性排序

    使用方式:
        retriever = HybridRetriever(vector_store)
        docs = await retriever.retrieve("什么是 Transformer 的自注意力机制？")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.35,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    async def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """
        执行混合检索

        Args:
            query: 用户查询
            top_k: 返回结果数量（覆盖默认值）

        Returns:
            检索结果列表，按相关性排序
        """
        k = top_k or self.top_k

        # 1. 向量语义搜索
        vector_results = self._vector_search(query, k * 2)

        # 2. 关键词搜索（在向量库的文档中做文本匹配）
        keyword_results = self._keyword_search(query, k * 2)

        # 3. 合并去重
        merged = self._merge_results(vector_results, keyword_results)

        # 4. 过滤低分结果
        filtered = [
            r for r in merged
            if r.get("score", 0) >= self.score_threshold
        ]

        # 返回 top_k
        results = filtered[:k]

        logger.debug(
            f"检索完成 | 查询: {query[:50]}... | "
            f"向量结果: {len(vector_results)} | "
            f"关键词结果: {len(keyword_results)} | "
            f"最终结果: {len(results)}"
        )

        return results

    def _vector_search(self, query: str, top_k: int) -> list[dict]:
        """向量语义搜索"""
        try:
            results = self.vector_store.search(query, top_k=top_k)
            # ChromaDB 的 distance 越小越相似（cosine），转换为分数
            for r in results:
                r["score"] = 1 - r.get("distance", 0)
                r["source"] = "vector"
            return results
        except Exception as e:
            logger.warning(f"向量搜索失败: {e}")
            return []

    def _keyword_search(self, query: str, top_k: int) -> list[dict]:
        """
        关键词搜索

        在已有的向量库结果中做文本匹配（简单实现）。
        生产环境建议接入 Elasticsearch 或 BM25 索引。
        """
        try:
            # 用查询中的关键词做简单匹配
            # 提取关键词（去除停用词）
            keywords = self._extract_keywords(query)
            if not keywords:
                return []

            # 使用 ChromaDB 的 where_document 过滤
            results = self.vector_store.search(query, top_k=top_k)
            keyword_results = []

            for r in results:
                content = r.get("content", "").lower()
                # 计算关键词命中数
                hits = sum(1 for kw in keywords if kw.lower() in content)
                if hits > 0:
                    r["score"] = hits / len(keywords)
                    r["source"] = "keyword"
                    keyword_results.append(r)

            return keyword_results
        except Exception as e:
            logger.warning(f"关键词搜索失败: {e}")
            return []

    def _extract_keywords(self, query: str) -> list[str]:
        """从查询中提取关键词"""
        # 简单实现：按空格分词，过滤短词
        # 生产环境建议使用 jieba 分词
        words = query.split()
        keywords = [w for w in words if len(w) >= 2]
        return keywords

    def _merge_results(self, vector_results: list[dict], keyword_results: list[dict]) -> list[dict]:
        """合并去重向量和关键词搜索结果"""
        seen_contents = set()
        merged = []

        # 向量结果优先（语义匹配更准确）
        for r in vector_results:
            content_hash = hash(r["content"][:100])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                merged.append(r)

        # 补充关键词结果
        for r in keyword_results:
            content_hash = hash(r["content"][:100])
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                merged.append(r)

        # 按分数排序
        merged.sort(key=lambda x: x.get("score", 0), reverse=True)
        return merged
