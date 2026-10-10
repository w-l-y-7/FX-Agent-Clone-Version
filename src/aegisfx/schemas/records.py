"""落盘记录：一条事件从投递到闸门判定的完整快照。

三个对象放在一个信封里写进 `events.jsonl`，这样读一行就拿到全貌，
不用在几个文件之间按 id 拼。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from .events import ParsedEvent, Signal
from .gate import GateDecision


class EventRecord(BaseModel):
    event: ParsedEvent
    signal: Signal
    gate: GateDecision

    @property
    def recorded_at(self) -> datetime:
        return self.signal.signal_generated_at
