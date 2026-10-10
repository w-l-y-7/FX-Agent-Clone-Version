# 人工投递目录

把 `.json` 丢进这个目录，跑一次 `scripts/ingest_event.py --process-inbox`
（或者在演示页点「扫描并处理 inbox」），系统就会读走。

处理完的文件会被挪走：成功的进 `processed/`，解析不了的进 `failed/`
（同目录下留一个 `.error.txt` 说明原因）。所以同一个文件不会被重复投递。

## 文件格式

一个文件可以放**一条**或**一批**：

```json
{
  "text": "美国商务部宣布将 12 家中国企业列入实体清单",
  "source_url": "https://example.com/news/123",
  "source_name": "路透社",
  "source_published_at": "2026-09-25T09:00:00+08:00"
}
```

```json
[
  {"text": "第一条消息", "source_url": "https://example.com/1"},
  {"text": "第二条消息", "source_name": "路透社"}
]
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `text` | 是 | 消息原文 |
| `source_url` | 否 | **给了它默认判为已核验**，执行闸门才可能放行 |
| `source_name` | 否 | 来源名称 |
| `source_published_at` | 否 | ISO 8601。不填就用接收时间占位，并标记为推测值，**不计入延迟统计** |
| `verified` | 否 | 显式指定是否已核验，覆盖「看有没有链接」的默认规则 |
| `use_llm` | 否 | 单独指定这条走不走大模型，覆盖命令行的 `--no-llm` |

## 两个提醒

**发布时间尽量填。** 系统记录四个时间戳（发布 / 发现 / 出信号 / 下单），
延迟指标是靠它们算的。人工补录的消息如果不填发布时间，这段滞后就没法从
真实延迟里区分出来。

**文件编码。** 程序用 `utf-8-sig` 读，带 BOM 也能正确处理——PowerShell 的
`Out-File -Encoding utf8` 和 Windows 记事本存出来的都是带 BOM 的，不用特意改。
