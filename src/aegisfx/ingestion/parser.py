"""消息 → 结构化事件。

两条路：

- **大模型**（主路）——按方案 §4.1，把消息转成固定 JSON 字段；
- **规则**（兜底）——关键词匹配。没有密钥、接口超时、模型返回不合法的枚举值时
  用它顶上，保证流水线永远不断。

两条路都会做**枚举校验**：模型说的事件类型不在词表里就退回 `other`，
不能让它自由发挥。`event_stage` 只能靠模型推断，所以只用于展示。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

from ..schemas.events import (
    EVENT_TYPES,
    INITIATING_SIDES,
    OTHER_EVENT_TYPE,
    SENTIMENTS,
)

# --------------------------------------------------------------------------
# 规则兜底
# --------------------------------------------------------------------------

# 关键词按动作分组，**打分决定归属**，不是"命中第一个就返回"。
#
# 之所以改成打分：真实新闻一句话里往往同时命中好几组（"对华301调查项下征税建议
# 提起磋商请求" 同时含 磋商 / 调查 / 征税）。原先按固定顺序取第一个命中，
# 在 445 条带标签语料上只有 54.8% 的一致率。改成「关键词越长权重越高」之后
# 明显改善——`争端解决`(4) + `调查`(2) 会压过 `磋商`(2)，落回 litigation。
_ACTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "fx_manipulator": ("汇率操纵国", "汇率操纵"),
    "dialogue": ("对话", "磋商", "谈判", "会谈", "通话", "会晤", "会见",
                 "研讨会", "经贸磋商"),
    # 正反两种语序都要收：「提高关税」和「关税提高」在中文新闻里同样常见，
    # 只收前者的话「部分商品关税提高至 50%」会漏掉。
    "tariff_increase": ("反倾销税", "反补贴税", "加征关税", "征收关税", "提高关税",
                        "上调关税", "关税提高", "关税上调", "关税增加", "全面征税",
                        "加征", "征税", "加税"),
    "tariff_relief": ("关税排除", "排除清单", "取消关税", "降低关税", "放宽关税",
                      "关税降低", "关税下调", "关税减免", "关税豁免",
                      "豁免", "减免", "暂缓"),
    "restrict_firms": ("实体清单", "出口管制", "不可靠实体", "制裁", "禁令", "断供",
                       "黑名单", "禁止", "限制", "审查", "列入"),
    "ease_restrictions": ("移出", "移除", "解除", "放宽限制", "取消限制", "解禁", "恢复"),
    "litigation": ("争端解决", "301调查", "232调查", "反倾销", "反补贴", "终裁",
                   "初裁", "诉讼", "起诉", "申诉", "调查"),
    # 不收裸的"进口"——"对自中国进口的钢铁全面征税" 会被它误判成采购
    "increase_purchases": ("增加进口", "扩大进口", "采购", "购买"),
    "criticize": ("指责", "批评", "谴责", "抨击", "表示不满"),
}

# 打分相同时的裁决顺序：越靠前越优先。
# 放在前面的都是"更具体的动作"，比如征税压过采购、限制压过调查。
_ACTION_PRIORITY: Tuple[str, ...] = (
    "fx_manipulator", "tariff_increase", "tariff_relief", "restrict_firms",
    "ease_restrictions", "litigation", "criticize", "dialogue",
    "increase_purchases",
)

# 打到这一步说明「光靠词频打分不够」：语料里大量放宽关税的新闻，正文却是
# 「公布了对 2000 亿中国商品**加征关税**的**排除清单**」——`加征关税` 得分压过
# `排除清单`，方向就反了。
#
# 这些组合词一旦全部出现，说明动作是往缓和方向走，**直接判定为放宽，不再比分数**。
# 每个元组里的词必须全部命中才算数，`("取消", "关税")` 这种写法是为了避开
# 「取消谈判」之类的误伤。
_RELIEF_OVERRIDE: Tuple[Tuple[str, ...], ...] = (
    ("排除清单",), ("关税排除",), ("暂停加征",), ("暂不加征",), ("暂不征收",),
    ("不再加征",), ("不加征",), ("关税豁免",), ("豁免",),
    ("推迟", "关税"), ("取消", "关税"), ("下调", "关税"),
)

_US_MARKERS = ("美国", "美方", "特朗普", "拜登", "白宫", "USTR", "美联储")
_CN_MARKERS = ("中国", "中方", "国务院", "商务部", "人民银行", "央行", "海关总署")

_NEGATIVE_ACTIONS = frozenset(
    {"tariff_increase", "restrict_firms", "litigation", "criticize", "designate"}
)
_POSITIVE_ACTIONS = frozenset(
    {"tariff_relief", "ease_restrictions", "dialogue", "increase_purchases", "remove"}
)

# (发起方, 动作) -> 事件类型
_TYPE_TABLE = {
    ("US", "tariff_increase"): "us_tariff_increase",
    ("US", "tariff_relief"): "us_tariff_relief",
    ("US", "restrict_firms"): "us_restrict_cn_firms",
    ("US", "ease_restrictions"): "us_ease_restrictions_cn_firms",
    ("US", "litigation"): "us_litigation_cn",
    ("US", "criticize"): "us_criticize_cn",
    ("CN", "tariff_increase"): "cn_tariff_increase",
    ("CN", "tariff_relief"): "cn_tariff_relief",
    ("CN", "restrict_firms"): "cn_restrict_us_firms",
    ("CN", "ease_restrictions"): "cn_ease_restrictions_us_firms",
    ("CN", "litigation"): "cn_litigation_us",
    ("CN", "criticize"): "cn_criticize_us",
    ("CN", "increase_purchases"): "cn_increase_purchases",
}

_REMOVE_WORDS = ("剔除", "移出", "移除", "取消")


@dataclass
class Draft:
    """两条路共同的产出。"""

    event_type: str
    initiating_side: str
    sentiment: str
    event_stage: Optional[str]
    method: str
    notes: str = ""


def _first_position(text: str, markers: Tuple[str, ...]) -> Optional[int]:
    positions = [text.find(m) for m in markers if m in text]
    return min(positions) if positions else None


def _detect_action(text: str) -> Optional[str]:
    """按「命中关键词的总长度」给每个动作打分，取最高分。

    用长度当权重等于用**具体程度**当权重：`争端解决` 比 `磋商` 更能说明这是
    一场官司而不是一次会谈。
    """
    scores = {
        action: sum(len(keyword) for keyword in keywords if keyword in text)
        for action, keywords in _ACTION_KEYWORDS.items()
    }
    best = max(scores.values(), default=0)
    if best == 0:
        return None
    for action in _ACTION_PRIORITY:
        if scores.get(action) == best:
            return action
    return None


def _detect_side(text: str) -> str:
    """谁先出现在句子里，谁就是发起方。

    "美国对中国加征关税" → US；"中国对美国加征关税" → CN。
    这是个粗糙但足够稳的启发式，中文新闻的主语习惯把施动者放在前面。
    """
    us_at = _first_position(text, _US_MARKERS)
    cn_at = _first_position(text, _CN_MARKERS)
    if us_at is None and cn_at is None:
        return "UNKNOWN"
    if cn_at is None:
        return "US"
    if us_at is None:
        return "CN"
    return "US" if us_at < cn_at else "CN"


def parse_with_rules(text: str) -> Draft:
    action = None
    if any(all(word in text for word in combo) for combo in _RELIEF_OVERRIDE):
        action = "tariff_relief"
    if action is None:
        action = _detect_action(text)
    if action is None:
        return Draft(OTHER_EVENT_TYPE, "UNKNOWN", "unknown", None, "rule",
                     "没有命中任何动作关键词")

    if action == "dialogue":
        # 只有「中美对话」是双向事件，其余按发起方分
        return Draft("bilateral_dialogue", "BOTH", "positive", None, "rule", "")

    side = _detect_side(text)
    if side == "UNKNOWN":
        return Draft(OTHER_EVENT_TYPE, "UNKNOWN", "unknown", None, "rule",
                     "认不出发起方")

    if action == "fx_manipulator":
        event_type = (
            "us_remove_cn_fx_manipulator"
            if any(word in text for word in _REMOVE_WORDS)
            else "us_designate_cn_fx_manipulator"
        )
        sentiment = "positive" if event_type == "us_remove_cn_fx_manipulator" else "negative"
        return Draft(event_type, "US", sentiment, None, "rule", "")

    event_type = _TYPE_TABLE.get((side, action))
    if event_type is None:
        return Draft(OTHER_EVENT_TYPE, side, "unknown", None, "rule",
                     f"{side} 不适用动作 {action}")

    if action in _NEGATIVE_ACTIONS:
        sentiment = "negative"
    elif action in _POSITIVE_ACTIONS:
        sentiment = "positive"
    else:
        sentiment = "unknown"
    return Draft(event_type, side, sentiment, None, "rule", "")


# --------------------------------------------------------------------------
# 大模型主路
# --------------------------------------------------------------------------

_SYSTEM_PROMPT = """你是金融政策事件解析器。把用户给的中美经贸政策消息转成结构化字段。

要求：
- event_type 只能从下面这张表里选，选不出就填 other，不要自己造词：
{table}
- initiating_side：CN（中方发起）/ US（美方发起）/ BOTH（双方共同，如对话）/ UNKNOWN
- sentiment：positive（缓和、合作）/ negative（摩擦、升级）/ unknown
- event_stage：消息处于哪个阶段，从 announcement（宣布）/ implementation（实施）/
  negotiation（谈判中）/ retaliation（反制）/ other 里选。这一项没有权威来源，只能靠文本推断。
- 只输出字段，不要解释。"""


def _build_prompt(text: str) -> str:
    types = "\n".join(f"  - {name}" for name in EVENT_TYPES)
    return _SYSTEM_PROMPT.format(table=types) + f"\n\n消息：\n{text}"


def parse_with_llm(text: str, timeout_note: str = "") -> Draft:
    """调 DeepSeek 解析。任何异常都由调用方兜住，这里不吞。"""
    from pydantic import BaseModel, Field

    from src.services.deepseek_llm_service import DeepSeekLLMService

    class _Draft(BaseModel):
        event_type: str = Field(description="事件类型，取自给定词表，选不出填 other")
        initiating_side: str = Field(description="CN / US / BOTH / UNKNOWN")
        sentiment: str = Field(description="positive / negative / unknown")
        event_stage: str = Field(description="announcement / implementation / negotiation / retaliation / other")

    import os

    service = DeepSeekLLMService(api_key=os.environ["DEEPSEEK_API_KEY"])
    raw = service.invoke_structured(_build_prompt(text), _Draft)

    notes = []
    event_type = raw.event_type.strip()
    if event_type not in EVENT_TYPES and event_type != OTHER_EVENT_TYPE:
        notes.append(f"模型给出词表外的事件类型 {event_type!r}，已归入 other")
        event_type = OTHER_EVENT_TYPE

    side = raw.initiating_side.strip().upper()
    if side not in INITIATING_SIDES:
        notes.append(f"发起方 {side!r} 不在枚举内，已置为 UNKNOWN")
        side = "UNKNOWN"

    sentiment = raw.sentiment.strip().lower()
    if sentiment not in SENTIMENTS:
        notes.append(f"情感 {sentiment!r} 不在枚举内，已置为 unknown")
        sentiment = "unknown"

    stage = raw.event_stage.strip() or None
    return Draft(event_type, side, sentiment, stage, "llm", "；".join(notes) + timeout_note)


# --------------------------------------------------------------------------


def parse_event(text: str, *, use_llm: bool = True) -> Draft:
    """解析一条消息。大模型优先，失败或不可用时退回规则。"""
    text = (text or "").strip()
    if not text:
        return Draft(OTHER_EVENT_TYPE, "UNKNOWN", "unknown", None, "failed", "消息为空")

    if use_llm:
        import os

        if not os.getenv("DEEPSEEK_API_KEY"):
            draft = parse_with_rules(text)
            draft.notes = (draft.notes + "；未配置 DEEPSEEK_API_KEY，已退回规则解析").strip("；")
            return draft
        try:
            return parse_with_llm(text)
        except Exception as exc:  # 网络、额度、格式错误都在这里兜住
            draft = parse_with_rules(text)
            draft.notes = (draft.notes + f"；大模型解析失败（{type(exc).__name__}），已退回规则").strip("；")
            return draft

    return parse_with_rules(text)


def summarise(text: str, limit: int = 60) -> str:
    """给界面用的短标题。"""
    flat = re.sub(r"\s+", " ", (text or "").strip())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"
