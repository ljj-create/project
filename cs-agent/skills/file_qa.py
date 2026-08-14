"""
文件解析与问答 Skill — 上传文件后基于内容进行问答
"""
import tempfile
import hashlib
from pathlib import Path
from loguru import logger

from skills.base import BaseSkill, SkillResult
from knowledge.loader import DocumentLoader
from knowledge.splitter import DocumentSplitter
from knowledge.vectorstore import VectorStore


class FileQASkill(BaseSkill):
    """
    文件解析与问答工具

    功能:
    1. 解析上传的文件（PDF、Word、代码、Markdown）
    2. 将内容分块嵌入到临时向量库
    3. 基于文件内容回答用户问题

    使用场景:
    - 用户上传论文 PDF，要求解读
    - 用户上传代码文件，要求分析
    - 用户上传文档，要求总结
    """

    name = "file_qa"
    description = "解析上传的文件并基于内容回答问题。支持 PDF、Word、Markdown、代码文件。当用户上传文件并提问时使用。"
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件路径",
            },
            "question": {
                "type": "string",
                "description": "关于文件内容的问题",
            },
        },
        "required": ["file_path", "question"],
    }

    def __init__(self):
        self.loader = DocumentLoader()
        self.splitter = DocumentSplitter()
        # 使用独立的 collection 存储用户上传的文件
        self._file_stores: dict[str, VectorStore] = {}

    async def execute(self, **kwargs) -> SkillResult:
        file_path = kwargs.get("file_path", "")
        question = kwargs.get("question", "")

        if not file_path:
            return SkillResult.error("文件路径不能为空")
        if not Path(file_path).exists():
            return SkillResult.error(f"文件不存在: {file_path}")

        logger.info(f"文件问答 | 文件: {file_path} | 问题: {question[:50]}...")

        try:
            # 1. 加载文件
            document = self.loader.load_file(file_path)

            # 2. 分块
            chunks = self.splitter.split(document)
            if not chunks:
                return SkillResult.error("文件内容为空或无法解析")

            # 3. 为该文件创建独立的向量存储
            file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
            collection_name = f"file_{file_hash}"

            vector_store = VectorStore(collection_name=collection_name)
            # 同一文件重复上传时先清空旧内容，避免新旧片段混在一起
            vector_store.clear()
            vector_store.add_chunks(chunks)

            # 4. 检索相关内容
            results = vector_store.search(question, top_k=5)

            if not results:
                return SkillResult.success(
                    f"文件已解析（{len(chunks)} 个段落），但未找到与问题直接相关的内容。\n\n"
                    f"文件概要:\n{document.content[:1000]}..."
                )

            # 5. 组织检索结果
            context_parts = []
            for r in results:
                context_parts.append(r["content"])

            context = "\n\n---\n\n".join(context_parts)

            result_text = (
                f"## 文件解析结果\n\n"
                f"**文件**: {Path(file_path).name}\n"
                f"**类型**: {document.metadata.get('type', '未知')}\n"
                f"**段落数**: {len(chunks)}\n\n"
                f"### 与问题相关的内容\n\n{context}\n\n"
                f"请基于以上内容回答用户的问题。"
            )

            return SkillResult.success(
                result_text,
                file_path=file_path,
                chunks_count=len(chunks),
                relevant_count=len(results),
            )

        except Exception as e:
            logger.error(f"文件解析失败: {e}")
            return SkillResult.error(f"文件解析失败: {str(e)}")
