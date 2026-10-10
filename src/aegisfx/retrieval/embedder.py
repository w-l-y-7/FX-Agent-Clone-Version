"""向量化：直接复用复现代码里那套（`BAAI/bge-small-zh-v1.5` + TF-IDF 兜底）。

复用而不是重写，是因为那个实现里有两处踩过坑的细节：模型懒加载
（导入阶段不碰网络）、以及 transformers 5.x 把维度方法改了名要两个都试。
TF-IDF 兜底也不是摆设——模型下不下来的时候它真能出结果。
"""

from __future__ import annotations

from typing import Optional, Sequence

from src.services.vector_rag_service import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbedder,
    TfidfEmbedder,
)

__all__ = ["build_embedder", "DEFAULT_EMBEDDING_MODEL"]


def build_embedder(corpus: Sequence[str], *, prefer: str = DEFAULT_EMBEDDING_MODEL):
    """优先用语义模型；加载不了就退回 TF-IDF。

    返回 `(embedder, note)`，`note` 说明发生了什么，界面和日志里要显示出来——
    悄悄换个 embedder 会让检索结果变化却看不出原因。
    """
    try:
        return SentenceTransformerEmbedder(prefer), ""
    except Exception as exc:
        note = f"语义模型 {prefer} 不可用（{type(exc).__name__}），已退回 TF-IDF"
        return TfidfEmbedder(corpus), note
