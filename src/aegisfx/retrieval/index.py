"""事件向量索引：建、存、读、查。

## 为什么要自己写一遍检索，而不是直接调 FAISS

**As-of 截断要求「在候选集上取最近邻」，而 FAISS 的 `search(k)` 是在全库上取的。**
常见做法是先取一个很大的 k 再过滤，但那样截断后的候选不足时结果会静默变少——
这正是本项目最怕的静默错误。

本项目的库只有几百条，所以在**掩码后的子集**上直接算内积：结果和
`IndexFlatIP` 完全一致（都是精确内积），而且支持任意候选集。

FAISS 索引仍然会建、会落盘——它是「可重复建立的索引」这个交付物，
也是以后事件库变大时的扩展路径。检索时用它验证一遍：掩码为全集时两者结果必须一致。

索引的序列化沿用复现代码里的实现，那里有一处 Windows 专用的绕行
（faiss 自带的 `write_index` 在 Windows 上走 ANSI 文件接口，非 ASCII 路径会挂），
不重写第二遍。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.services.vector_rag_service import (  # noqa: E402
    _build_faiss_index,
    _read_faiss_index,
    _write_faiss_index,
)

from ..paths import DATA_DIR
from ..schemas.historical import HistoricalEvent

INDEX_DIR = DATA_DIR / "retrieval"
_INDEX_NAME = "events.faiss"
_ORDER_NAME = "event_order.json"
_MANIFEST_NAME = "manifest.json"


@dataclass
class Hit:
    event_id: str
    score: float
    rank: int


def _fingerprint(events: Sequence[HistoricalEvent], embedder_name: str) -> str:
    digest = hashlib.sha1()
    digest.update(embedder_name.encode("utf-8"))
    for event in sorted(events, key=lambda item: item.event_id):
        digest.update(event.event_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(event.text.encode("utf-8"))
        digest.update(b"\x01")
    return digest.hexdigest()


class EventIndex:
    """437 条事件的内积索引，支持 as-of 候选集。"""

    def __init__(self, events: Sequence[HistoricalEvent], embedder, note: str = ""):
        self.events = list(events)
        self.embedder = embedder
        self.note = note
        self._ids: List[str] = [event.event_id for event in self.events]
        self._position: Dict[str, int] = {event_id: i for i, event_id in enumerate(self._ids)}
        self._vectors: Optional[np.ndarray] = None

    # ---------------------------------------------------------------- 构建

    @property
    def vectors(self) -> np.ndarray:
        if self._vectors is None:
            self._vectors = self.embedder.encode([event.text for event in self.events])
        return self._vectors

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.events, self.embedder.name)

    def build(self, *, force: bool = False) -> bool:
        """建索引并落盘。指纹没变就跳过。返回是否真的重建了。"""
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        manifest_path = INDEX_DIR / _MANIFEST_NAME
        fingerprint = self.fingerprint

        if not force and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("fingerprint") == fingerprint:
                return False

        vectors = self.vectors
        index = _build_faiss_index(vectors)
        _write_faiss_index(index, INDEX_DIR / _INDEX_NAME)
        (INDEX_DIR / _ORDER_NAME).write_text(
            json.dumps(self._ids, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_path.write_text(json.dumps({
            "fingerprint": fingerprint,
            "embedder": self.embedder.name,
            "dim": int(vectors.shape[1]),
            "events": len(self._ids),
            "note": self.note,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return True

    def load_vectors_from_disk(self) -> bool:
        """从落盘的 FAISS 索引里读回向量，省掉重新编码。"""
        path = INDEX_DIR / _INDEX_NAME
        order_path = INDEX_DIR / _ORDER_NAME
        if not path.exists() or not order_path.exists():
            return False
        try:
            order = json.loads(order_path.read_text(encoding="utf-8"))
            if order != self._ids:
                return False
            index = _read_faiss_index(path)
            self._vectors = np.vstack([
                index.reconstruct(i) for i in range(index.ntotal)
            ]).astype(np.float32)
            return self._vectors.shape[0] == len(self._ids)
        except Exception:
            return False

    # ---------------------------------------------------------------- 检索

    def search(
        self,
        query_vector: np.ndarray,
        *,
        allowed_ids: Optional[Sequence[str]] = None,
        top_k: int = 10,
    ) -> List[Hit]:
        """在候选集内取内积最大的 top_k。

        `allowed_ids=None` 表示全库。传空列表会返回空结果（而不是退化成全库）——
        这一点很重要：as-of 池为空时必须真的检不到东西。
        """
        vectors = self.vectors
        if allowed_ids is None:
            positions = np.arange(len(self._ids))
        else:
            positions = np.array(
                [self._position[event_id] for event_id in allowed_ids
                 if event_id in self._position],
                dtype=int,
            )
        if positions.size == 0:
            return []

        scores = vectors[positions] @ query_vector.astype(np.float32)
        take = min(top_k, len(positions))
        order = np.argsort(-scores)[:take]
        return [
            Hit(event_id=self._ids[positions[i]], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order)
        ]

    def vector_for(self, event_id: str) -> np.ndarray:
        return self.vectors[self._position[event_id]]


def load_manifest() -> Optional[dict]:
    path = INDEX_DIR / _MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
