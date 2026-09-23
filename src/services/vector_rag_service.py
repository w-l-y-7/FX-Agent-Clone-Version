"""`BaseRAG` 的真实实现：本地向量检索。

检索链路：查询文本 → 向量化 → 在 FAISS 里找最相似的文本块 → 返回带出处的结果。
用 `IndexFlatIP`（精确内积检索）而不是近似索引，是因为知识库只有几百个块，
暴力比对是毫秒级的，引入近似索引只会增加出错的地方。

三层降级，任何一层出问题都不会让程序崩：

1. 正常情况：bge-small-zh 向量化 + FAISS 检索
2. 模型下载不下来 / 依赖不兼容：退回 TF-IDF 字符 n-gram（中文没有空格分词，
   `char_wb` 的 1-3 元组在短文本检索上表现相当能打，这是真兜底不是摆设）
3. 知识库为空：返回空列表并打印警告，DA 会据此把所有特征判为"证据不足"
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

# 这一行必须在 sentence_transformers 之前，它负责把下载地址指到国内镜像
from ..utils.hf_env import configure_hf_endpoint  # noqa: F401
from ..core.abstractions.base_rag import BaseRAG
from ..core.knowledge_base import (
    DEFAULT_KB_ROOT,
    corpus_fingerprint,
    load_documents,
)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_DIR = _PROJECT_ROOT / "data" / "knowledge_base" / "index"

_MANIFEST_NAME = "manifest.json"
_INDEX_NAME = "index.faiss"
_CHUNKS_NAME = "chunks.json"

_TFIDF_FALLBACK_DIM = 256


@runtime_checkable
class Embedder(Protocol):
    """只要能给出固定维度、且已做 L2 归一化的向量即可。"""

    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """返回 shape 为 (len(texts), dim) 且每行已 L2 归一化的矩阵。"""
        ...


class SentenceTransformerEmbedder:
    """基于 sentence-transformers 的中文向量模型。

    模型是**懒加载**的：只有真正要建索引或检索时才去下载，模块导入阶段
    不碰网络。首次运行约需下载 95MB（走镜像站，几十秒）。
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        configure_hf_endpoint()
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)
        # transformers 5.x 把这个方法改了名，两个都试一下
        dimension = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        self.dim = int(dimension())

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class TfidfEmbedder:
    """零下载的兜底方案：字符 n-gram TF-IDF + SVD 降维。

    中文没有天然的词边界，`char_wb` 的 1-3 元组能同时兼顾单字、词和短语，
    在几百条的短文本检索上效果足够。它不是"玩具兜底"——真跑起来能出结果。
    """

    def __init__(self, corpus: Sequence[str], dim: int = _TFIDF_FALLBACK_DIM):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import make_pipeline

        self.name = "tfidf-char-ngram"
        self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3))
        matrix = self._vectorizer.fit_transform(corpus)

        # 语料块数太少时 SVD 的维度不能超过样本数，否则会报错
        effective_dim = max(2, min(dim, matrix.shape[0] - 1, matrix.shape[1] - 1))
        self._svd = (
            make_pipeline(TruncatedSVD(n_components=effective_dim, random_state=0))
            if matrix.shape[0] > 2 and matrix.shape[1] > 2
            else None
        )
        self.dim = int(self._svd.named_steps["truncatedsvd"].n_components) if self._svd else 0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self._svd is None:
            return np.zeros((len(texts), 0), dtype=np.float32)
        vectors = self._svd.fit_transform(self._vectorizer.transform(texts)).astype(np.float32)
        return _l2_normalize(vectors)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class VectorRAGService(BaseRAG):
    """本地向量检索服务。检索结果里带上出处，方便追溯 DA 的判断依据。"""

    def __init__(
        self,
        kb_root: Optional[Path] = None,
        index_dir: Optional[Path] = None,
        *,
        embedder: Optional[Embedder] = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        top_k_default: int = 3,
        verbose: bool = False,
    ):
        self._kb_root = Path(kb_root) if kb_root else DEFAULT_KB_ROOT
        self._index_dir = Path(index_dir) if index_dir else DEFAULT_INDEX_DIR
        self._model_name = model_name
        self._top_k_default = top_k_default
        self._verbose = verbose

        self._embedder: Optional[Embedder] = embedder
        self._index = None
        self._chunks: List[Dict[str, Any]] = []
        self._degraded_reason: Optional[str] = None

    # ---------- 对外接口 ----------

    @property
    def is_ready(self) -> bool:
        return self._index is not None and bool(self._chunks)

    @property
    def document_count(self) -> int:
        return len(self._chunks)

    @property
    def backend_name(self) -> str:
        if self._embedder is not None:
            return self._embedder.name
        return "(未初始化)"

    @property
    def degraded_reason(self) -> Optional[str]:
        """为 None 表示用的是语义向量；有值表示已降级到 TF-IDF。"""
        return self._degraded_reason

    def build(self, *, force: bool = False) -> int:
        """建索引。语料没变就直接复用磁盘上的，返回文本块数量。

        知识库为空时返回 0，不报错——DA 会据此走"证据不足"的降级路径。
        """
        chunks = load_documents(self._kb_root, verbose=self._verbose)
        if not chunks:
            print(f"知识库为空（{self._kb_root}），检索将返回空结果。")
            return 0

        fingerprint = corpus_fingerprint(chunks)

        if not force and self._load_from_disk(fingerprint):
            return len(self._chunks)

        self._chunks = chunks
        self._embedder = self._make_embedder(chunks)
        if self._embedder is None or self._embedder.dim == 0:
            print("向量化失败，检索将返回空结果。")
            return 0

        vectors = self._embedder.encode([chunk["content"] for chunk in chunks])
        self._index = _build_faiss_index(vectors)
        self._save_to_disk(fingerprint)

        if self._verbose:
            print(f"知识库索引已建好：{len(chunks)} 个文本块，"
                  f"向量模型 {self._embedder.name}（{self._embedder.dim} 维）")
        return len(chunks)

    def retrieve(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """检索最相关的文本块。

        **这个方法保证不抛异常。** 知识库空、索引没建、模型加载失败、
        查询本身出问题——任何一种情况都返回空列表，让上层去走降级逻辑。
        原因是它位于 DA 的循环里，一次检索失败不该让整轮特征筛选崩掉。
        """
        k = int(top_k if top_k is not None else self._top_k_default)
        if k <= 0:
            return []

        if self._index is None:
            return []

        try:
            import faiss

            vector = self._embedder.encode([query])
            if vector.shape[1] != self._index.d:
                return []

            scores, indices = self._index.search(np.ascontiguousarray(vector), k)
        except Exception as exc:  # noqa: BLE001 — 检索失败必须降级，不能中断 DA
            print(f"检索失败，本轮返回空结果：{type(exc).__name__}: {exc}")
            return []

        results: List[Dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index >= len(self._chunks):
                continue
            chunk = self._chunks[index]
            results.append({
                "source": chunk["source"],
                "heading": chunk.get("heading"),
                "content": chunk["content"],
                "score": round(float(score), 4),
            })
        return results

    # ---------- 内部实现 ----------

    def _make_embedder(self, chunks: List[Dict[str, Any]]) -> Optional[Embedder]:
        if self._embedder is not None:
            return self._embedder

        try:
            embedder = SentenceTransformerEmbedder(self._model_name)
            self._degraded_reason = None
            return embedder
        except Exception as exc:  # noqa: BLE001 — 下载或加载失败就降级
            self._degraded_reason = f"{type(exc).__name__}: {exc}"
            print(
                f"语义向量模型 {self._model_name} 加载失败，改用 TF-IDF 兜底。"
                f"\n原因：{self._degraded_reason}"
            )
            corpus = [chunk["content"] for chunk in chunks]
            try:
                return TfidfEmbedder(corpus)
            except Exception as fallback_exc:  # noqa: BLE001
                print(f"TF-IDF 兜底也失败了：{fallback_exc}")
                return None

    def _manifest_path(self) -> Path:
        return self._index_dir / _MANIFEST_NAME

    def _load_from_disk(self, fingerprint: str) -> bool:
        manifest_path = self._manifest_path()
        if not manifest_path.exists():
            return False

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False

        # 语料变了、或者换了向量模型，旧索引就作废
        if manifest.get("fingerprint") != fingerprint:
            if self._verbose:
                print("知识库内容有变动，重建索引。")
            return False

        try:
            self._chunks = json.loads(
                (self._index_dir / _CHUNKS_NAME).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return False

        self._embedder = self._make_embedder(self._chunks)
        if self._embedder is None or self._embedder.dim != manifest.get("dim"):
            return False

        try:
            self._index = _read_faiss_index(self._index_dir / _INDEX_NAME)
        except Exception:  # noqa: BLE001
            return False

        return True

    def _save_to_disk(self, fingerprint: str) -> None:
        try:
            self._index_dir.mkdir(parents=True, exist_ok=True)
            _write_faiss_index(self._index, self._index_dir / _INDEX_NAME)
            (self._index_dir / _CHUNKS_NAME).write_text(
                json.dumps(self._chunks, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self._manifest_path().write_text(
                json.dumps({
                    "fingerprint": fingerprint,
                    "embedder": self._embedder.name,
                    "dim": self._embedder.dim,
                    "chunks": len(self._chunks),
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001 — 存不下来只是下次要重建，不该中断
            print(f"索引落盘失败（不影响本次使用）：{type(exc).__name__}: {exc}")

    def reset(self) -> None:
        """删掉磁盘上的索引，下次 build 会重建。"""
        self._index = None
        self._chunks = []
        if self._index_dir.exists():
            shutil.rmtree(self._index_dir)


def _build_faiss_index(vectors: np.ndarray):
    import faiss

    matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


# faiss 自带的 `write_index` / `read_index` 在 Windows 上走的是 ANSI 文件接口，
# 路径里只要有中文（本机是 `D:\git仓库\...`）就会 fopen 失败，而且失败得很安静。
# 改成先序列化成字节、再交给 Python 的文件接口写，就完全绕开了这个限制。
def _write_faiss_index(index, path: Path) -> None:
    import faiss

    path.write_bytes(faiss.serialize_index(index).tobytes())


def _read_faiss_index(path: Path):
    import faiss

    return faiss.deserialize_index(np.frombuffer(path.read_bytes(), dtype=np.uint8))
