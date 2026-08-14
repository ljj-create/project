"""
向量存储 — 基于 ChromaDB 的文档嵌入与检索
"""
import sys
from pathlib import Path


def _preload_msvc_runtime():
    """预加载系统新版 MSVC 运行库，避免被 Anaconda 旧版遮蔽导致 torch 加载失败。"""
    if sys.platform != "win32":
        return
    import ctypes
    import os

    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    for name in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        try:
            ctypes.WinDLL(os.path.join(system32, name))
        except OSError:
            pass


_preload_msvc_runtime()

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from knowledge.splitter import TextChunk
from config.settings import settings


class VectorStore:
    """
    ChromaDB 向量存储

    功能:
    - 将文本块嵌入并存储到 ChromaDB
    - 支持向量语义搜索
    - 支持按元数据过滤
    """

    def __init__(
        self,
        collection_name: str = "cs_knowledge",
        db_path: str | None = None,
        embedding_model: str | None = None,
    ):
        self.collection_name = collection_name
        self.db_path = db_path or settings.knowledge_db_path
        self.embedding_model_name = embedding_model or settings.embedding_model

        # 初始化 ChromaDB
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # 获取或创建 collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # 延迟加载 embedding 模型
        self._embedding_model = None

        logger.info(
            f"向量库初始化 | collection: {self.collection_name} | "
            f"已有文档数: {self.collection.count()}"
        )

    @property
    def embedding_model(self):
        """延迟加载 embedding 模型"""
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer

            model_ref = self._resolve_model_ref()
            logger.info(f"加载 embedding 模型: {model_ref}")
            try:
                self._embedding_model = SentenceTransformer(model_ref)
            except Exception as e:
                logger.error(f"加载 embedding 模型失败: {e}")
                raise
        return self._embedding_model

    def _resolve_model_ref(self) -> str:
        """
        解析 embedding 模型引用。

        优先返回本地 HF 缓存路径（完全离线，避免在无外网/弱网时
        反复重试导致启动缓慢）；未缓存时返回模型名以便在线下载。
        """
        import os

        try:
            from huggingface_hub import snapshot_download

            path = snapshot_download(self.embedding_model_name, local_files_only=True)
            logger.info(f"使用本地缓存的 embedding 模型: {path}")
            return path
        except Exception:
            # 未命中本地缓存，回退为在线加载
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            return self.embedding_model_name

    def add_chunks(self, chunks: list[TextChunk], batch_size: int = 100):
        """
        批量添加文本块到向量库

        Args:
            chunks: 文本块列表
            batch_size: 每批处理的数量
        """
        if not chunks:
            return

        total = len(chunks)
        logger.info(f"开始添加 {total} 个文本块到向量库")

        for i in range(0, total, batch_size):
            batch = chunks[i:i + batch_size]

            # 使用「来源 + 块序号」生成稳定 ID；重复导入时用 upsert 更新而非报错
            ids = [
                f"{c.metadata.get('source', 'unknown')}:{c.index}"
                for c in batch
            ]
            documents = [c.content for c in batch]
            metadatas = [c.metadata for c in batch]

            # 使用 sentence-transformers 生成嵌入
            embeddings = self.embedding_model.encode(documents).tolist()

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            logger.debug(f"已添加 {min(i + batch_size, total)}/{total} 个文本块")

        logger.info(f"完成添加 {total} 个文本块，当前库中共 {self.collection.count()} 条")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[dict]:
        """
        向量语义搜索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            filter_metadata: 元数据过滤条件

        Returns:
            检索结果列表，每个结果包含 content、metadata、distance
        """
        # 生成查询嵌入
        query_embedding = self.embedding_model.encode([query]).tolist()

        kwargs = {
            "query_embeddings": query_embedding,
            "n_results": top_k,
        }
        if filter_metadata:
            kwargs["where"] = filter_metadata

        results = self.collection.query(**kwargs)

        # 整理结果
        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else 0,
            })

        return search_results

    def delete_by_source(self, source: str):
        """按来源删除文档"""
        self.collection.delete(where={"source": source})
        logger.info(f"已删除来源为 {source} 的文档")

    def clear(self):
        """清空向量库"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("向量库已清空")

    def count(self) -> int:
        """返回文档总数"""
        return self.collection.count()
