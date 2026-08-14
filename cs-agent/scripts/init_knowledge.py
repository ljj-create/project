"""
初始化知识库 — 批量导入知识文档到 ChromaDB

用法:
    python scripts/init_knowledge.py [--clear]

参数:
    --clear  清空现有知识库后重新导入
"""
import sys
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from knowledge.loader import DocumentLoader
from knowledge.splitter import DocumentSplitter
from knowledge.vectorstore import VectorStore
from config.settings import settings, BASE_DIR


def main():
    parser = argparse.ArgumentParser(description="初始化 CS-Agent 知识库")
    parser.add_argument("--clear", action="store_true", help="清空现有知识库")
    args = parser.parse_args()

    docs_dir = BASE_DIR / "knowledge" / "docs"
    if not docs_dir.exists():
        logger.error(f"知识文档目录不存在: {docs_dir}")
        logger.info("请先在 knowledge/docs/ 下添加知识文档（.md / .pdf / .txt）")
        return

    # 初始化组件
    loader = DocumentLoader()
    splitter = DocumentSplitter()
    vector_store = VectorStore()

    if args.clear:
        logger.info("清空现有知识库...")
        vector_store.clear()

    # 加载所有文档
    logger.info(f"扫描知识文档目录: {docs_dir}")
    documents = loader.load_directory(docs_dir)

    if not documents:
        logger.warning("未找到任何知识文档")
        logger.info("""
请在以下目录添加知识文档:
  knowledge/docs/algorithms/   — 算法与数据结构
  knowledge/docs/os/           — 操作系统
  knowledge/docs/networks/     — 计算机网络
  knowledge/docs/database/     — 数据库
  knowledge/docs/ml/           — 机器学习
  knowledge/docs/research/     — 科研方法论
  knowledge/docs/engineering/  — 工程实践
""")
        return

    # 分块
    logger.info(f"开始分块（chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap}）...")
    chunks = splitter.split_batch(documents)
    logger.info(f"共生成 {len(chunks)} 个文本块")

    # 导入向量库
    logger.info("开始导入向量库...")
    vector_store.add_chunks(chunks)

    logger.info(f"知识库初始化完成！共 {vector_store.count()} 条文档")


if __name__ == "__main__":
    main()
