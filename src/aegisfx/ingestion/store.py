"""JSONL 存储：只追加，不修改。

选 JSONL 而不是数据库的理由很实际——投递记录是要拿给人看的审计线索，
纯文本能直接 `git diff`、能直接打开读，坏了一行也只丢一行。
数据量在几百条这个级别，不需要 SQLite。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def append_jsonl(path: Path, record: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" 是刻意的：Windows 默认会写成 \r\n，和 Linux 上的产物不一致
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(record.model_dump_json() + "\n")


def append_many_jsonl(path: Path, records: Iterable[BaseModel]) -> None:
    for record in records:
        append_jsonl(path, record)


def read_jsonl(path: Path, model: Type[T], *, skip_bad: bool = True) -> List[T]:
    """读回 JSONL。

    默认跳过坏行而不是整个报错——审计日志里出现一行读不懂的内容，
    不应该让整个界面打不开。跳过了多少条，调用方可以从长度差看出来。
    """
    if not path.exists():
        return []

    records: List[T] = []
    # utf-8-sig：自己写出来的文件没有 BOM，但用户拿记事本改过一版之后就会有
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(model.model_validate_json(line))
            except ValidationError:
                if not skip_bad:
                    raise
    return records
