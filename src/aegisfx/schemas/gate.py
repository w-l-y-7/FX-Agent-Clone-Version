"""执行闸门。

设计要点有两条，都是刻意的：

1. **没有「跳过」这一档。** 检查项只有 PASS / FAIL / NOT_IMPLEMENTED 三种状态，
   而 `allow_execution` 要求全部 PASS。所以只要有任何一项还没实现，闸门就是关的。
   第 1 周执行层整体没写，于是系统**必然**输出 BLOCKED——这是对的行为，
   不是缺陷。

2. **输出是清单，不是一个布尔值。** 只给 `False` 没法审计，也没法在界面上
   解释「为什么不让下单」。每条检查都带名字、状态、说明。
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


class GateCheck(BaseModel):
    name: str
    status: CheckStatus
    detail: str

    @property
    def passed(self) -> bool:
        return self.status is CheckStatus.PASS


class GateDecision(BaseModel):
    allow_execution: bool
    checks: List[GateCheck]
    blocked_reasons: List[str] = Field(default_factory=list)

    @classmethod
    def from_checks(cls, checks: List[GateCheck]) -> "GateDecision":
        blocked = [f"{c.name}：{c.detail}" for c in checks if not c.passed]
        return cls(
            allow_execution=not blocked,
            checks=checks,
            blocked_reasons=blocked,
        )

    @property
    def mode(self) -> str:
        """给界面用的一句话。"""
        return "ALLOW" if self.allow_execution else "BLOCKED"

    def render(self) -> str:
        """纯文本渲染，CLI 和演示页共用同一套措辞。"""
        width = max((len(c.name) for c in self.checks), default=0)
        lines = [f"执行状态：{self.mode}"]
        for check in self.checks:
            mark = {CheckStatus.PASS: "✓", CheckStatus.FAIL: "✗",
                    CheckStatus.NOT_IMPLEMENTED: "—"}[check.status]
            lines.append(f"  {check.name:<{width}}  {mark}  {check.detail}")
        return "\n".join(lines)
