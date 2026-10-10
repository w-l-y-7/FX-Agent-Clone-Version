"""投递一条政策消息，或者处理 inbox 目录。

    # 粘一条消息（带来源链接，会被判为已核验）
    .\\venv\\Scripts\\python.exe scripts\\ingest_event.py --text "美国宣布对部分中国商品加征25%关税" --url "https://example.com/news/123"

    # 不带链接——研究层照常跑，执行闸门会因 source_verified 关掉
    .\\venv\\Scripts\\python.exe scripts\\ingest_event.py --text "中美双方在华盛顿举行经贸磋商"

    # 从文件读（一条消息一个文件）
    .\\venv\\Scripts\\python.exe scripts\\ingest_event.py --file news.txt

    # 读走 data/inbox/ 下所有 .json
    .\\venv\\Scripts\\python.exe scripts\\ingest_event.py --process-inbox

    # 看最近投递
    .\\venv\\Scripts\\python.exe scripts\\ingest_event.py --list 5

    # 人工 Kill Switch
    .\\venv\\Scripts\\python.exe scripts\\ingest_event.py --kill-switch on

`--no-llm` 强制走规则解析（不花 API 调用，用来核对规则表本身写得对不对）。
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.aegisfx.ingestion import (  # noqa: E402
    IngestResult,
    list_events,
    process_inbox,
    submit_raw_event,
)
from src.aegisfx.ingestion.service import set_kill_switch  # noqa: E402
from src.aegisfx.paths import load_env  # noqa: E402


def _parse_time(value: str | None):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise SystemExit(f"发布时间格式不对：{value!r}，要 ISO 8601，例如 2026-09-26T14:30:00+08:00")
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _print_result(result: IngestResult, as_json: bool) -> None:
    if as_json:
        print(json.dumps({
            "raw_id": result.raw.raw_id,
            "event_id": result.event.event_id,
            **result.event.model_dump(mode="json"),
            "gate": result.gate.model_dump(mode="json"),
        }, ensure_ascii=False, indent=2))
        return

    event = result.event
    print()
    print(f"事件 ID      {event.event_id}")
    print(f"事件类型     {event.event_type}"
          + (f"（{event.parse_notes}）" if event.parse_notes else ""))
    print(f"发起方       {event.initiating_side} → 目标 {event.target_side}")
    print(f"情感         {event.sentiment}")
    print(f"解析方式     {event.parse_method}")
    print(f"发布时间     {event.source_published_at.isoformat()}"
          + ("（投递人未填，用接收时间占位）" if result.raw.source_published_at_inferred else ""))
    print(f"标签状态     {event.abret_status}")
    print(f"研究层       方向 {result.signal.direction}"
          f"（{result.signal.status}：{result.signal.notes}）")
    print()
    print(result.gate.render())


def main() -> int:
    parser = argparse.ArgumentParser(description="AegisFX-Agent 消息投递")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="消息原文")
    source.add_argument("--file", type=Path, help="从文件读消息原文")
    source.add_argument("--process-inbox", action="store_true",
                        help="处理 data/inbox/ 下所有 .json")
    source.add_argument("--list", type=int, metavar="N", help="列出最近 N 条投递")
    source.add_argument("--kill-switch", choices=["on", "off"], help="人工叫停开关")

    parser.add_argument("--url", help="来源链接；给了它默认判为已核验")
    parser.add_argument("--source-name", help="来源名称")
    parser.add_argument("--published-at", help="消息发布时间，ISO 8601")
    parser.add_argument("--verified", dest="verified", action="store_true", default=None,
                        help="强制标为已核验")
    parser.add_argument("--unverified", dest="verified", action="store_false",
                        help="强制标为未核验")
    parser.add_argument("--no-llm", action="store_true", help="只用规则解析，不调大模型")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")

    args = parser.parse_args()
    load_env()

    if args.kill_switch:
        engaged = set_kill_switch(args.kill_switch == "on")
        print(f"Kill Switch：{'已按下（禁止下单）' if engaged else '已复位'}")
        return 0

    if args.list is not None:
        records = list_events(args.list)
        if not records:
            print("还没有任何投递记录。")
            return 0
        for record in records:
            event = record.event
            print(f"{record.recorded_at:%Y-%m-%d %H:%M}  "
                  f"{event.event_type:<32} {event.initiating_side:<8} "
                  f"{record.gate.mode:<8} {event.event_id}")
        return 0

    if args.process_inbox:
        results = process_inbox(use_llm=not args.no_llm)
        if not results:
            print("inbox 里没有待处理的消息。")
            return 0
        for result in results:
            _print_result(result, args.json)
        print(f"\n共处理 {len(results)} 条。")
        return 0

    text = args.text
    if args.file:
        text = args.file.read_text(encoding="utf-8")
    if not text or not text.strip():
        parser.error("要给 --text 或 --file，或者用 --process-inbox / --list / --kill-switch")

    result = submit_raw_event(
        text.strip(),
        source_url=args.url,
        source_name=args.source_name,
        source_published_at=_parse_time(args.published_at),
        ingested_by="cli",
        verified=args.verified,
        use_llm=not args.no_llm,
    )
    _print_result(result, args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
