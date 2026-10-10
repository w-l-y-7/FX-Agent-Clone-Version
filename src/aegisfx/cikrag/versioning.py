"""知识版本、人工反馈、增量更新。

## 增量更新不是「每来一条就全库重训」

方案 §9.1 把知识增量和预测效用增量分成两条：

- **知识增量**：新文档进来 → 去重 → 抽观点 → 判关系 → **只重算受影响的特征**；
- **预测效用增量**：只有新汇率结果**已成熟**才更新，且同一结果不能重复更新。

旧知识**不删除**，只改状态（`active` / `superseded` / `out_of_scope` / `invalid`）。
删掉了就没法回答「当时为什么这么判」。

## 人工反馈的边界

方案 §10.3 写死了人工**不得**：自由打分、改权重、改预测值、跳过样本外验证。
所以 `apply_feedback` 的入参里**根本没有预测值这一项**——
不是靠文档约定，是接口上就没有这个口子。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from .relations import HUMAN_EXTRACTION_ERROR, HUMAN_RELATION_CHOICES
from .schema import Claim, ClaimEdge, ClaimRelation, ClaimStatus


@dataclass
class KnowledgeVersion:
    knowledge_version: str
    claim_count: int
    active_claim_count: int
    conflict_edges: int
    updated_features: List[str] = field(default_factory=list)
    added_sources: List[str] = field(default_factory=list)
    superseded_claims: List[str] = field(default_factory=list)
    feature_set_version: str = ""
    forecast_model_version: str = ""
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "knowledge_version": self.knowledge_version,
            "claim_count": self.claim_count,
            "active_claim_count": self.active_claim_count,
            "conflict_edges": self.conflict_edges,
            "updated_features": self.updated_features,
            "added_sources": self.added_sources,
            "superseded_claims": self.superseded_claims,
            "feature_set_version": self.feature_set_version,
            "forecast_model_version": self.forecast_model_version,
            "created_at": self.created_at,
        }


def version_id(claims: Sequence[Claim]) -> str:
    """由**内容**算版本号，同样的知识得到同样的版本。"""
    digest = hashlib.sha1()
    for claim in sorted(claims, key=lambda c: c.claim_id):
        payload = (claim.model_dump(mode="json") if hasattr(claim, "model_dump")
                   else claim.dict())
        digest.update(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                 default=str).encode("utf-8"))
        digest.update(b"\x01")
    return f"kv_{digest.hexdigest()[:12]}"


def make_version(
    claims: Sequence[Claim],
    *,
    edges: Sequence[ClaimEdge] = (),
    updated_features: Sequence[str] = (),
    added_sources: Sequence[str] = (),
    feature_set_version: str = "",
    forecast_model_version: str = "",
) -> KnowledgeVersion:
    active = [c for c in claims if c.status is ClaimStatus.ACTIVE]
    superseded = [c.claim_id for c in claims
                  if c.status is ClaimStatus.SUPERSEDED]
    return KnowledgeVersion(
        knowledge_version=version_id(claims),
        claim_count=len(claims),
        active_claim_count=len(active),
        conflict_edges=sum(1 for e in edges
                           if e.relation is ClaimRelation.CONTRADICTS),
        updated_features=list(updated_features),
        added_sources=list(added_sources),
        superseded_claims=superseded,
        feature_set_version=feature_set_version,
        forecast_model_version=forecast_model_version,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# 人工反馈
# ---------------------------------------------------------------------------


class FeedbackError(ValueError):
    """人工反馈不合法——选项不在允许集合里，或者缺了必需的理由与原文。"""


@dataclass
class FeedbackRecord:
    feedback_id: str
    source_claim_id: str
    target_claim_id: str
    choice: str
    relation: Optional[ClaimRelation]
    reason: str
    quoted_span: str
    applied: bool = True


class FeedbackLog:
    """人工反馈的落地记录。**同一条反馈不重复应用。**"""

    def __init__(self) -> None:
        self.records: List[FeedbackRecord] = []
        self._seen: Dict[str, str] = {}

    @staticmethod
    def _key(source: str, target: str, choice: str) -> str:
        payload = f"{source}|{target}|{choice}".encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:16]

    def submit(self, source: str, target: str, choice: str, *,
               reason: str, quoted_span: str) -> FeedbackRecord:
        """登记一条反馈。

        `reason` 与 `quoted_span` 都是**必填**：没有理由和原文引用的人工判断，
        和机器瞎猜没有区别，事后也无法审计。
        """
        choice = choice.strip().upper()
        if choice != HUMAN_EXTRACTION_ERROR and choice not in HUMAN_RELATION_CHOICES:
            raise FeedbackError(
                f"选项 {choice!r} 不在允许集合 "
                f"{sorted(HUMAN_RELATION_CHOICES) + [HUMAN_EXTRACTION_ERROR]} 内")
        if not reason.strip():
            raise FeedbackError("必须填写判断理由")
        if not quoted_span.strip():
            raise FeedbackError("必须引用原文片段")

        key = self._key(source, target, choice)
        if key in self._seen:
            return FeedbackRecord(feedback_id=self._seen[key],
                                  source_claim_id=source, target_claim_id=target,
                                  choice=choice, relation=None, reason=reason,
                                  quoted_span=quoted_span, applied=False)

        self._seen[key] = key
        record = FeedbackRecord(
            feedback_id=key, source_claim_id=source, target_claim_id=target,
            choice=choice,
            relation=HUMAN_RELATION_CHOICES.get(choice),
            reason=reason.strip(), quoted_span=quoted_span.strip())
        self.records.append(record)
        return record

    def apply(self, records: Sequence[Claim]) -> List[Claim]:
        """把人工反馈落到观点上。

        **只能改观点之间的关系和状态，改不了任何预测值**——
        这个函数的入参里就没有预测这一项。
        """
        by_id = {c.claim_id: c for c in records}
        for feedback in self.records:
            if feedback.choice == HUMAN_EXTRACTION_ERROR:
                target = by_id.get(feedback.source_claim_id)
                if target is not None:
                    target.status = ClaimStatus.INVALID
                continue
            target = by_id.get(feedback.target_claim_id)
            if target is not None and feedback.relation is ClaimRelation.SUPERSEDES:
                # 被替代的知识不删除，只改状态，审计记录保留
                target.status = ClaimStatus.SUPERSEDED
        return list(records)


# ---------------------------------------------------------------------------
# 增量更新
# ---------------------------------------------------------------------------


def affected_features(
    new_claims: Sequence[Claim],
    existing: Sequence[Claim],
) -> List[str]:
    """所有新增有效观点涉及的特征都要重算，包括库里已经存在的特征。"""
    return sorted({c.feature_id for c in new_claims
                   if c.status is ClaimStatus.ACTIVE})


def incremental_update(
    existing: Sequence[Claim],
    incoming: Sequence[Claim],
    *,
    duplicate_cluster_lookup: Optional[Dict[str, str]] = None,
) -> tuple:
    """增量并入新观点。返回 `(全部观点, 受影响的特征)`。

    等价性要求：**增量更新的结果必须和「把所有观点一起重跑一遍」完全一致。**
    这条有测试守着——增量逻辑最容易出的错就是「慢慢跑偏了，
    但每一步看起来都对」。
    """
    duplicate_cluster_lookup = duplicate_cluster_lookup or {}
    merged: List[Claim] = list(existing)
    seen_text = {c.claim_text.strip(): c.duplicate_cluster_id for c in existing}

    added: List[Claim] = []
    for claim in incoming:
        text = claim.claim_text.strip()
        if text in seen_text:
            # 转载：不新增有效证据，标成 invalid 并归入既有簇
            claim.duplicate_cluster_id = seen_text[text]
            claim.status = ClaimStatus.INVALID
            merged.append(claim)
            continue
        if claim.claim_id not in duplicate_cluster_lookup:
            claim.duplicate_cluster_id = (claim.duplicate_cluster_id
                                          or f"cluster_{len(seen_text) + 1:03d}")
        seen_text[text] = claim.duplicate_cluster_id
        added.append(claim)
        merged.append(claim)

    return merged, affected_features(added, existing)


def feature_set_hash(features: Sequence[str]) -> str:
    return hashlib.sha1("|".join(sorted(features)).encode("utf-8")).hexdigest()[:12]


def dump_version(version: KnowledgeVersion, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.append(version.to_dict())
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2),
                    encoding="utf-8")
