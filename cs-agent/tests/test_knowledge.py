"""
知识库系统测试
"""
import pytest
from pathlib import Path
import tempfile

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge.loader import DocumentLoader, Document
from knowledge.splitter import DocumentSplitter, TextChunk


class TestDocumentLoader:
    """文档加载器测试"""

    def test_load_markdown(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# 测试\n\n这是一个测试文档。")
            f.flush()
            doc = DocumentLoader.load_file(f.name)
            assert "测试" in doc.content
            assert doc.metadata["type"] == "markdown"

    def test_load_python(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write("print('hello')")
            f.flush()
            doc = DocumentLoader.load_file(f.name)
            assert "print" in doc.content
            assert doc.metadata["language"] == "python"

    def test_unsupported_format(self):
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            with pytest.raises(ValueError, match="不支持"):
                DocumentLoader.load_file(f.name)


class TestDocumentSplitter:
    """文档分块器测试"""

    def test_short_document(self):
        doc = Document(content="短文档", metadata={}, source="test")
        splitter = DocumentSplitter(chunk_size=100, chunk_overlap=10)
        chunks = splitter.split(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "短文档"

    def test_long_document(self):
        content = "\n\n".join([f"段落{i}: " + "x" * 100 for i in range(10)])
        doc = Document(content=content, metadata={}, source="test")
        splitter = DocumentSplitter(chunk_size=200, chunk_overlap=20)
        chunks = splitter.split(doc)
        assert len(chunks) > 1

    def test_empty_document(self):
        doc = Document(content="", metadata={}, source="test")
        splitter = DocumentSplitter()
        chunks = splitter.split(doc)
        assert len(chunks) == 0

    def test_batch_split(self):
        docs = [
            Document(content=f"文档{i}内容", metadata={}, source=f"test{i}")
            for i in range(3)
        ]
        splitter = DocumentSplitter()
        chunks = splitter.split_batch(docs)
        assert len(chunks) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
