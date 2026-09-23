"""构建知识库向量索引。

    .\\venv\\Scripts\\python.exe scripts\\build_knowledge_base.py
    .\\venv\\Scripts\\python.exe scripts\\build_knowledge_base.py --force   # 强制重建

语料没变时直接复用磁盘上的索引，几秒就跑完。
"""

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.services.vector_rag_service import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    VectorRAGService,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="构建知识库向量索引")
    parser.add_argument("--force", action="store_true", help="忽略缓存，强制重建")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="向量模型名")
    parser.add_argument("--query", default=None, help="可选：建完索引后试检索这条查询")
    args = parser.parse_args()

    service = VectorRAGService(model_name=args.model, verbose=True)

    started = time.perf_counter()
    count = service.build(force=args.force)
    elapsed = time.perf_counter() - started

    if count == 0:
        print("知识库是空的，没有建立索引。往 data/knowledge_base/notes/ 里放点 .md 文件吧。")
        return 1

    print(f"\n完成：{count} 个文本块，用时 {elapsed:.1f} 秒")
    print(f"向量模型：{service.backend_name}")
    if service.degraded_reason:
        print(f"注意：已降级到 TF-IDF。原因：{service.degraded_reason}")

    if args.query:
        print(f"\n试检索：{args.query}")
        for index, hit in enumerate(service.retrieve(args.query, top_k=3), start=1):
            heading = hit["heading"] or "(无小节标题)"
            print(f"  [{index}] 相似度 {hit['score']:.4f} | {hit['source']} > {heading}")
            snippet = hit["content"].replace("\n", " ")[:120]
            print(f"      {snippet}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
