"""监视目录入口：把 `.json` 丢进 `data/inbox/`，系统自己读走。

这是三个投递入口里最适合做半自动流水线的一个——写个爬虫、或者手工导出，
只要产出 JSON 落进目录就行，不需要对接任何 API。

处理完的文件会挪到 `processed/`（成功）或 `failed/`（解析不了），
原文件不留，避免重复投递。
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ..paths import INBOX_DIR, INBOX_FAILED, INBOX_PROCESSED, ensure_dirs
from .service import IngestResult, submit_raw_event

# 投递文件里允许出现的字段。多出来的字段会被忽略而不是报错——
# 上游爬虫多带几个字段是常事。
_ALLOWED_KEYS = {
    "text",
    "source_url",
    "source_name",
    "source_published_at",
    "verified",
    "use_llm",
}


def _coerce_published(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"source_published_at 不是合法的 ISO 时间：{value!r}") from exc


def _load_payloads(path: Path) -> List[Dict[str, Any]]:
    # `utf-8-sig` 而不是 `utf-8`：Windows 上 PowerShell 的 `Out-File -Encoding utf8`
    # 和记事本的「UTF-8」都会写 BOM，用 `utf-8` 读会在第一个字符就 JSONDecodeError。
    # 这个编码对没有 BOM 的文件一样能读。
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return data
    raise ValueError("投递文件必须是对象或对象数组")


def _ingest_payload(payload: Dict[str, Any], use_llm: bool) -> IngestResult:
    if not isinstance(payload, dict):
        raise ValueError(f"投递项必须是对象，收到 {type(payload).__name__}")
    if not str(payload.get("text", "")).strip():
        raise ValueError("投递项缺少 text 字段")

    kwargs = {key: value for key, value in payload.items() if key in _ALLOWED_KEYS}
    kwargs["text"] = str(kwargs["text"])
    kwargs["source_published_at"] = _coerce_published(payload.get("source_published_at"))
    kwargs["ingested_by"] = "inbox"
    # 文件里显式写了 use_llm 就以文件为准，否则听调用方的
    kwargs.setdefault("use_llm", use_llm)
    return submit_raw_event(**kwargs)  # type: ignore[arg-type]


def process_inbox(use_llm: bool = True) -> List[IngestResult]:
    """读走 `data/inbox/` 下所有 `.json`。返回成功处理的条目。

    单个文件里可以写 `"use_llm": false` 覆盖这个参数。
    """
    ensure_dirs()
    results: List[IngestResult] = []

    for path in sorted(INBOX_DIR.glob("*.json")):
        try:
            payloads = _load_payloads(path)
            for payload in payloads:
                results.append(_ingest_payload(payload, use_llm))
        except Exception as exc:
            INBOX_FAILED.mkdir(parents=True, exist_ok=True)
            note = INBOX_FAILED / f"{path.stem}.error.txt"
            note.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            shutil.move(str(path), str(INBOX_FAILED / path.name))
            continue

        shutil.move(str(path), str(INBOX_PROCESSED / path.name))

    return results
