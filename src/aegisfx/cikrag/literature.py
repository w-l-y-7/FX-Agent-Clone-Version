"""文献观点卡片：把真实文献里的「特征 → 方向」断言变成可核查的 `Claim`。

## 为什么不复用 `extract.py` 的规则抽取

`extract.py` 只认 8 个中文小节标题和中文方向词。实测：

    _direction_of("The US dollar index has a significantly positive effect "
                  "on the RMB exchange rate.")  ->  None

真实文献全是英文，而且论文不写 "X is positively correlated with Y"，它写
"we find that a one-standard-deviation increase in the broad dollar index
raises the bilateral rate by 0.4%"。拿规则去抽这种句子，方向抽错的概率很高，
而**抽错方向会直接毒害冲突判定**——比没有文献更糟。

所以这里换一条路：**卡片**。

## 卡片是什么

一张卡片 = 一条「特征 → 方向」断言，从论文正文里读出来，但**必须附上论文原文
句子**（`evidence_span`）。`verify_spans()` 会逐条检查这句话确实能在抓下来的
正文里找到（做空白规范化，因为 PDF 会断行）。

找不到的卡片直接判定为失败并列出——**不允许只写转述、不留原文**。

## 诚实标注，别把它说成规则抽取

卡片一律 `extraction_model="literature-card/manual"`。
从论文里读出「这个特征对汇率是正还是负」，本质上需要语义理解，规则做不到，
这一步有人（或 LLM 辅助）参与。报告里必须这么写，不能含糊成「自动抽取」。

`evidence_type` 用真实类型：NBER working paper 记 `working_paper`，
同行评审期刊记 `peer_reviewed`，央行/国际组织报告记 `official_report`。
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .schema import Claim, ClaimStatus, EffectDirection, EvidenceType

CARDS_DIR = "data/knowledge_base/papers/cards"
RAW_DIR = "data/knowledge_base/papers/raw"

# 卡片里允许出现的方向值，与 `EffectDirection` 对齐
_DIRECTIONS = {d.value for d in EffectDirection}


def normalise(text: str) -> str:
    """PDF 断行/多余空格会让原句对不上，比较前先归一。"""
    return re.sub(r"\s+", " ", text or "").strip()


class CardError(ValueError):
    """卡片文件格式或引用核查失败。"""


def load_cards(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    for field in ("key", "source_title", "claims"):
        if field not in data:
            raise CardError(f"{path} 缺字段 {field}")
    if not isinstance(data["claims"], list) or not data["claims"]:
        raise CardError(f"{path} 的 claims 为空")
    for i, item in enumerate(data["claims"]):
        for field in ("feature_id", "effect_direction", "claim_text",
                      "evidence_span"):
            if not item.get(field):
                raise CardError(f"{path} 第 {i} 条缺 {field}")
        if item["effect_direction"] not in _DIRECTIONS:
            raise CardError(
                f"{path} 第 {i} 条方向 {item['effect_direction']!r} 不在 "
                f"{sorted(_DIRECTIONS)} 内")
    return data


def verify_spans(card: dict, raw_text: str) -> List[Tuple[int, bool]]:
    """逐条核查 `evidence_span` 是否真的出现在论文正文里。

    返回 `[(序号, 是否命中)]`。**不做模糊匹配**——模糊匹配等于放水，
    这条通道的全部价值就在于「说的每句话都能在原文里找到」。
    """
    haystack = normalise(raw_text)
    out = []
    for i, item in enumerate(card["claims"]):
        needle = normalise(item["evidence_span"])
        out.append((i, bool(needle) and needle in haystack))
    return out


def cards_to_claims(
    cards: Sequence[dict],
    *,
    raw_texts: Optional[Dict[str, str]] = None,
    strict: bool = True,
) -> Tuple[List[Claim], List[str]]:
    """卡片 → `Claim` 列表。

    返回 `(claims, problems)`。`raw_texts` 给定时会核查引用；
    `strict=True` 时引用核查不过的卡片**不进入** claims，只进 problems。
    """
    raw_texts = raw_texts or {}
    claims: List[Claim] = []
    problems: List[str] = []
    counter = 0

    for card in cards:
        key = card["key"]
        if key in raw_texts:
            for i, ok in verify_spans(card, raw_texts[key]):
                if not ok:
                    span = card["claims"][i]["evidence_span"][:80]
                    problems.append(
                        f"{key} 第 {i} 条引用在正文里找不到：{span!r}")
        elif strict:
            problems.append(f"{key} 没有对应的正文文本，无法核查引用")

        if strict and any(p.startswith(key) for p in problems):
            continue

        published = card.get("published_at")
        published_at = date.fromisoformat(published) if published else None

        for item in card["claims"]:
            counter += 1
            claims.append(Claim(
                claim_id=f"lit_{counter:04d}",
                feature_id=item["feature_id"],
                feature_name=item.get("feature_name", item["feature_id"]),
                claim_text=item["claim_text"],
                effect_direction=EffectDirection(item["effect_direction"]),
                currency_pair=item.get("currency_pair", "USD/CNY"),
                forecast_horizon=item.get("forecast_horizon", 20),
                target_variable=item.get("target_variable", "USDCNY"),
                applicable_country=item.get("applicable_country", "CN/US"),
                applicable_period=item.get("applicable_period", ""),
                market_regime=item.get("market_regime", ""),
                policy_condition=item.get("policy_condition", ""),
                methodology=card.get("methodology", ""),
                sample_period=card.get("sample_period", ""),
                evidence_type=EvidenceType(card.get("evidence_type", "unknown")),
                source_title=card.get("source_title", key),
                source_url=card.get("source_url"),
                source_doc_id=key,
                published_at=published_at,
                evidence_span=item["evidence_span"],
                extraction_model="literature-card/manual",
                human_verified=bool(item.get("human_verified", False)),
                status=ClaimStatus.ACTIVE,
            ))
    return claims, problems


def load_all(
    root: Path, *, strict: bool = True,
) -> Tuple[List[Claim], List[str], List[dict]]:
    """从 `cards/` + `raw/` 一次性装载。"""
    cards_dir = root / CARDS_DIR
    raw_dir = root / RAW_DIR
    if not cards_dir.exists():
        return [], [], []

    cards = []
    for path in sorted(cards_dir.glob("*.json")):
        cards.append(load_cards(path))

    raw_texts = {}
    for path in raw_dir.glob("*.txt"):
        raw_texts[path.stem] = path.read_text(encoding="utf-8")

    claims, problems = cards_to_claims(cards, raw_texts=raw_texts, strict=strict)
    return claims, problems, cards


def summarise(claims: Sequence[Claim]) -> dict:
    """给报告用的分布统计。"""
    by_feature: Dict[str, int] = {}
    by_direction: Dict[str, int] = {}
    by_evidence: Dict[str, int] = {}
    by_year: Dict[str, int] = {}
    for c in claims:
        by_feature[c.feature_id] = by_feature.get(c.feature_id, 0) + 1
        by_direction[c.effect_direction.value] = (
            by_direction.get(c.effect_direction.value, 0) + 1)
        by_evidence[c.evidence_type.value] = (
            by_evidence.get(c.evidence_type.value, 0) + 1)
        year = str(c.published_at.year) if c.published_at else "未知"
        by_year[year] = by_year.get(year, 0) + 1
    return {"n_claims": len(claims), "by_feature": by_feature,
            "by_direction": by_direction, "by_evidence_type": by_evidence,
            "by_year": dict(sorted(by_year.items()))}
