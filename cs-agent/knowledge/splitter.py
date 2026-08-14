"""
文档分块器 — 将长文档切分为适合嵌入的小块
"""
from dataclasses import dataclass
from knowledge.loader import Document
from config.settings import settings


@dataclass
class TextChunk:
    """文本块"""
    content: str
    metadata: dict
    index: int  # 在原文档中的序号


class DocumentSplitter:
    """
    文档分块器

    策略:
    - 按段落/换行符分割
    - 保证每个 chunk 不超过 chunk_size 字符
    - chunk 之间有 chunk_overlap 字符的重叠，避免语义断裂
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def split(self, document: Document) -> list[TextChunk]:
        """
        将文档分块

        Args:
            document: 文档对象

        Returns:
            文本块列表
        """
        text = document.content
        if not text.strip():
            return []

        # 先按段落分割
        paragraphs = self._split_paragraphs(text)

        # 再合并小段落、切分大段落
        chunks = self._merge_and_split(paragraphs)

        # 构建 TextChunk 对象
        result = []
        for i, chunk_text in enumerate(chunks):
            if chunk_text.strip():
                result.append(TextChunk(
                    content=chunk_text.strip(),
                    metadata={**document.metadata, "chunk_index": i},
                    index=i,
                ))

        return result

    def split_batch(self, documents: list[Document]) -> list[TextChunk]:
        """批量分块"""
        all_chunks = []
        for doc in documents:
            chunks = self.split(doc)
            all_chunks.extend(chunks)
        return all_chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """按段落分割"""
        # 优先按双换行分割（段落边界）
        paragraphs = text.split("\n\n")
        # 过滤空段落
        return [p for p in paragraphs if p.strip()]

    def _merge_and_split(self, paragraphs: list[str]) -> list[str]:
        """合并小段落、切分大段落"""
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # 如果当前 chunk 加上新段落不超限，合并
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                current_chunk = f"{current_chunk}\n\n{para}" if current_chunk else para
            else:
                # 保存当前 chunk
                if current_chunk:
                    chunks.append(current_chunk)

                # 如果段落本身超限，需要进一步切分
                if len(para) > self.chunk_size:
                    sub_chunks = self._split_long_text(para)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        # 添加重叠
        if self.chunk_overlap > 0:
            chunks = self._add_overlap(chunks)

        return chunks

    def _split_long_text(self, text: str) -> list[str]:
        """切分超长文本（按句子/换行）"""
        chunks = []
        # 按换行符切分
        lines = text.split("\n")
        current = ""

        for line in lines:
            if len(current) + len(line) + 1 <= self.chunk_size:
                current = f"{current}\n{line}" if current else line
            else:
                if current:
                    chunks.append(current)
                # 如果单行也超限，按字符硬切
                if len(line) > self.chunk_size:
                    for i in range(0, len(line), self.chunk_size - self.chunk_overlap):
                        chunks.append(line[i:i + self.chunk_size])
                    current = ""
                else:
                    current = line

        if current:
            chunks.append(current)
        return chunks

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """为相邻 chunk 添加重叠"""
        if len(chunks) <= 1:
            return chunks

        result = [chunks[0]]
        for i in range(1, len(chunks)):
            # 取上一个 chunk 的末尾作为当前 chunk 的开头
            prev_tail = chunks[i - 1][-self.chunk_overlap:]
            result.append(f"{prev_tail}\n\n{chunks[i]}")
        return result
