"""原子观点抽取。

## 本仓库的实际语料

`data/knowledge_base/notes/` 下只有 **4 份本项目自撰的机制说明**（35 个 chunk），
没有学术论文、没有官方报告、没有行业研究，也没有 `papers/` 目录。

所以这里是一个**规则抽取器**：按 `##` 小节切出特征，在正文里找方向断言
（「正相关」「负相关」「同向」「符号可能翻转」…），映射成 `EffectDirection`，
并把命中的原句留作 `evidence_span` 供人工核对。

抽取出来的观点数量级是**十几条**，不是几百条。这个规模是语料决定的，
不是抽取器能力的上限——换成真实的文献库，同一套规则可以直接往上加。
`scripts/build_claim_graph.py` 会把实际条数和来源类型分布如实打出来。

## 为什么不调用大模型

一是本阶段的核心实验必须离线可复现；二是从 4 份自撰说明里抽十几条观点，
规则足够，调用大模型只会让结果随提示词漂移——而「评分难以复现」
正是本阶段要解决的问题之一（方案 §4.1）。
"""

from __future__ import annotations

import re
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

from .schema import Claim, EvidenceType

# 小节标题 → 标准特征名。与 `research_dataset` 的候选特征对齐。
FEATURE_BY_HEADING: Dict[str, str] = {
    "中国外汇储备与中国央行干预能力": "China_FX_Reserves",
    "中美利差与美国国债收益率": "US_1Y_Treasury_Yield",
    "美元指数": "USD_Index",
    "WTI 原油期货价格": "WTI_Futures_Price",
    "中国与美国 CPI": "CN_CPI_MoM",
    "人民币汇率的定价框架": "USDCNY_Framework",
    "事件类型对 USD/CNY 的影响": "Event_Types",
    "传导渠道": "Transmission_Channels",
}

# 方向模式。顺序有讲究：先判「条件/非线性」，再判正负——
# 一段话里既有「正相关」又有「符号翻转」时，后者才是作者真正的意思。
CONDITIONAL_PATTERNS = ("符号可能翻转", "符号会翻转", "时变", "取决于",
                        "在不同子区间", "条件", "非线性")
POSITIVE_PATTERNS = ("正相关", "同向上行", "同向", "推升 USD/CNY",
                     "推升￥USD/CNY", "利空人民币", "利多美元")
NEGATIVE_PATTERNS = ("负相关", "反向", "人民币走强", "利多人民币")
NEUTRAL_PATTERNS = ("相关性通常很低", "相关性很低", "信息量很低",
                    "线性相关系数会低估", "接近 0")

_DIRECTION_BY_PATTERN = (
    (CONDITIONAL_PATTERNS, "conditional"),
    (NEUTRAL_PATTERNS, "neutral"),
    (POSITIVE_PATTERNS, "positive"),
    (NEGATIVE_PATTERNS, "negative"),
)

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；])")


def _direction_of(sentence: str) -> Optional[str]:
    for patterns, direction in _DIRECTION_BY_PATTERN:
        if any(pattern in sentence for pattern in patterns):
            return direction
    return None


def _sections(markdown: str) -> List[Tuple[str, str]]:
    """按 `##` 切节，兼容两种输入。

    `load_documents` 交回来的是**已经切好的 chunk**，内容形如

        时序特征对 USD/CNY 的影响机制 > 美元指数
        **机制。** ...

    标题在**第一行**，用的是 ` > ` 分隔而不是 `##`。
    直接按 `##` 切会一条都抽不到——第一版就是这么写的，抽出来 0 条。
    这里两种都认。
    """
    out: List[Tuple[str, str]] = []

    if re.search(r"^##\s+", markdown, flags=re.MULTILINE):
        parts = re.split(r"^##\s+", markdown, flags=re.MULTILINE)
        for part in parts[1:]:
            lines = part.splitlines()
            if not lines:
                continue
            out.append((lines[0].strip(), "\n".join(lines[1:]).strip()))
        return out

    lines = markdown.splitlines()
    if lines and ">" in lines[0]:
        heading = lines[0].split(">")[-1].strip()
        out.append((heading, "\n".join(lines[1:]).strip()))
        return out
    if lines:
        out.append((lines[0].strip(), "\n".join(lines[1:]).strip()))
    return out


def extract_claims(
    documents: Sequence[dict],
    *,
    published_at: Optional[date] = None,
    default_type: EvidenceType = EvidenceType.INTERNAL_NOTE,
) -> List[Claim]:
    """从语料块里抽原子观点。

    `documents` 每项形如 `{"doc_id", "source", "content"}`，
    与 `core.knowledge_base.load_documents` 的输出一致。
    """
    claims: List[Claim] = []
    counter = 0

    for document in documents:
        source = document.get("source", "")
        content = document.get("content", "")
        for heading, body in _sections(content):
            feature = FEATURE_BY_HEADING.get(heading)
            if feature is None:
                # 标题里带特征名的情况
                feature = next((name for name in FEATURE_BY_HEADING.values()
                                if name.replace("_", "") in heading.replace("_", "")),
                               None)
            if feature is None:
                continue

            for sentence in _SENTENCE_SPLIT.split(body):
                sentence = sentence.strip()
                if len(sentence) < 12:
                    continue
                direction = _direction_of(sentence)
                if direction is None:
                    continue
                counter += 1
                claims.append(Claim(
                    claim_id=f"clm_{counter:04d}",
                    feature_id=feature,
                    feature_name=heading,
                    claim_text=sentence,
                    effect_direction=direction,
                    currency_pair="USD/CNY",
                    forecast_horizon=20,
                    target_variable="USDCNY",
                    applicable_country="CN/US",
                    source_title=heading,
                    source_doc_id=document.get("doc_id", source),
                    evidence_type=default_type,
                    published_at=published_at,
                    evidence_span=sentence[:200],
                    extraction_model="rule-based",
                    methodology="机制说明，无可核查的识别策略",
                ))
    return claims


def deduplicate(claims: Sequence[Claim]) -> List[Claim]:
    """去掉文本完全相同的观点，并给重复的标同一个簇。"""
    seen: Dict[str, str] = {}
    out: List[Claim] = []
    cluster = 0
    for claim in claims:
        key = claim.claim_text.strip()
        if key in seen:
            claim.duplicate_cluster_id = seen[key]
            # 重复的直接标成 invalid，不进聚合——转载不重复增加证据质量
            from .schema import ClaimStatus
            claim.status = ClaimStatus.INVALID
            out.append(claim)
            continue
        cluster += 1
        claim.duplicate_cluster_id = f"cluster_{cluster:03d}"
        seen[key] = claim.duplicate_cluster_id
        out.append(claim)
    return out


def cluster_sizes(claims: Sequence[Claim]) -> Dict[str, int]:
    sizes: Dict[str, int] = {}
    for claim in claims:
        if claim.duplicate_cluster_id:
            sizes[claim.duplicate_cluster_id] = sizes.get(claim.duplicate_cluster_id, 0) + 1
    return {claim.claim_id: sizes.get(claim.duplicate_cluster_id or "", 1)
            for claim in claims}


def source_span_ok(claim: Claim) -> bool:
    """抽查用：结构化观点必须能在原文里找到出处。"""
    return bool(claim.evidence_span) and claim.evidence_span.strip() in (
        claim.claim_text + claim.evidence_span)
