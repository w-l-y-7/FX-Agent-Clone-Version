"""把抓下来的文献导入 Zotero，并挂上 PDF 链接。

    .\\venv\\Scripts\\python.exe scripts\\import_to_zotero.py --check
    .\\venv\\Scripts\\python.exe scripts\\import_to_zotero.py --import
    .\\venv\\Scripts\\python.exe scripts\\import_to_zotero.py --clean

## Zotero 本地 API 的写入有三道关，都踩过了

1. 缺 `Zotero-Server-ID` 头       -> 428
2. Server-ID 填得不对              -> 412（正确值只在 GET 的响应头里，prefs.js 没有）
3. 缺 API key                      -> 401

第 3 关要 `POST /api/local/authorize`，**Zotero 会弹窗让人点「允许」**，
这一步没法自动化，必须用户手动点一次。拿到的 key 存在
`data/knowledge_base/zotero_credentials.json`（已加进 .gitignore，不会提交）。

## 附件用 linked_file，不是 imported_file

`imported_file`（文件复制进 Zotero storage）上传时卡在 `MD5 hash not provided`，
换 `Content-MD5` 头也不认。所以改用 `linked_file`：**Zotero 只记路径，
PDF 留在仓库里**。

代价说清楚：**仓库一旦移动或改名，Zotero 里的附件链接就断了**。
好处是文件跟着仓库走，不占 Zotero 存储。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

PAPER_DIR = _PROJECT_ROOT / "data" / "knowledge_base" / "papers"
MANIFEST = PAPER_DIR / "manifest.json"
CREDS = _PROJECT_ROOT / "data" / "knowledge_base" / "zotero_credentials.json"
IMPORT_LOG = PAPER_DIR / "zotero_import.json"

BASE = "http://127.0.0.1:23119/api/users/0"
COLLECTION_NAME = "AegisFX 汇率文献"

# NBER 首页那些不是作者的行
_NOT_AUTHOR = re.compile(
    r"(NBER|Working Paper|University|Department|Institute|Bureau|"
    r"@|http|www\.|\d{5}|No\.\s*\d)", re.IGNORECASE)


def discover_server_id() -> str:
    """server ID 不在 prefs.js 里，只能从响应头拿。"""
    r = requests.get(f"{BASE}/items?limit=1", timeout=10)
    sid = r.headers.get("Zotero-Server-ID")
    if not sid:
        raise SystemExit("拿不到 Zotero-Server-ID——Zotero 可能没开或本地 API 未启用")
    return sid


def load_api_key() -> str:
    if not CREDS.exists():
        raise SystemExit(
            f"没有 {CREDS}。先跑一次授权：\n"
            f"  POST http://127.0.0.1:23119/api/local/authorize\n"
            f"  body: {{\"appName\": \"AegisFX-Literature\"}}\n"
            f"（Zotero 会弹窗，点允许；把返回的 key 存进该文件）")
    return json.loads(CREDS.read_text(encoding="utf-8"))["api_key"]


def headers() -> dict:
    return {"Content-Type": "application/json",
            "Zotero-Server-ID": discover_server_id(),
            "Zotero-API-Key": load_api_key()}


# ---------------------------------------------------------------------------
# 从正文里抠出摘要与作者
# ---------------------------------------------------------------------------


def extract_abstract(text: str) -> str:
    m = re.search(r"\bABSTRACT\b|\bAbstract\b", text)
    if not m:
        return ""
    body = text[m.end():m.end() + 2500]
    # 摘要在作者机构块之前结束，那里通常出现邮箱或 "Working Paper"
    cut = re.search(r"@|Working Paper No|NBER Working Paper", body)
    if cut:
        body = body[:cut.start()]
    return re.sub(r"\s+", " ", body).strip()[:1800]


def _is_person_name(line: str) -> bool:
    """2-4 个词、每词首字母大写、不含机构词。"""
    if not (3 < len(line) < 45) or _NOT_AUTHOR.search(line):
        return False
    words = [w for w in line.split() if w]
    if not (2 <= len(words) <= 4):
        return False
    return all(w[0].isupper() and w.replace(".", "").replace("-", "").isalpha()
               for w in words)


# NBER 的 PDF 开头是固定框：系列名 → 大写标题 → 作者行 → "Working Paper NNNNN"。
# 用这个框定位作者，比"标题后面找"可靠——标题在正文里还会再出现一次，
# 那种写法会命中后面那次，正好跳过作者块。
_NBER_FRAME = re.compile(
    r"NBER WORKING PAPER SERIES\s*(?P<body>.*?)\s*Working Paper\s+\d+", re.S)


def extract_authors_nber(text: str) -> list:
    m = _NBER_FRAME.search(text)
    if not m:
        return []
    out = []
    for line in m.group("body").splitlines():
        line = line.strip()
        if not line or line.isupper():      # 空行 / 大写标题
            continue
        if _is_person_name(line):
            out.append(line)
    return out[:8]


def extract_authors_arxiv(aid: str) -> list:
    """走 arXiv 官方 API——正文里的作者名带角标（Xua,b），正则切不干净。"""
    try:
        r = requests.get("https://export.arxiv.org/api/query",
                         params={"id_list": aid, "max_results": 1}, timeout=30)
        if r.status_code != 200:
            return []
        return [n.strip() for n in
                re.findall(r"<author>\s*<name>(.*?)</name>", r.text)]
    except Exception:                                   # noqa: BLE001
        return []


def authors_for(record: dict, raw_text: str) -> list:
    if record["source"] == "nber":
        return extract_authors_nber(raw_text)
    aid = record["url"].rstrip("/").split("/")[-1]
    return extract_authors_arxiv(aid)


def creators_of(record: dict, raw_text: str) -> list:
    return [{"creatorType": "author", "firstName": split_name(a)[0],
             "lastName": split_name(a)[1]} for a in authors_for(record, raw_text)]


def split_name(full: str) -> tuple:
    parts = full.split()
    if len(parts) == 1:
        return "", parts[0]
    return " ".join(parts[:-1]), parts[-1]


# ---------------------------------------------------------------------------
# Zotero 操作
# ---------------------------------------------------------------------------


def find_collection(name: str) -> str:
    r = requests.get(f"{BASE}/collections", headers=headers(), timeout=15)
    for col in r.json():
        if col["data"]["name"] == name:
            return col["data"]["key"]
    r = requests.post(f"{BASE}/collections", headers=headers(),
                      json=[{"name": name}], timeout=20)
    r.raise_for_status()
    return r.json()["successful"]["0"]["data"]["key"]


def build_item(entry: dict, record: dict, raw_text: str,
               collection_key: str) -> dict:
    title = record["title"]
    creators = creators_of(record, raw_text)

    if record["source"] == "nber":
        item = {
            "itemType": "report",
            "title": title,
            "creators": creators,
            "date": str(record["year"]),
            "institution": "National Bureau of Economic Research",
            "reportNumber": entry.get("wid", ""),
            "reportType": "NBER Working Paper",
        }
    else:
        item = {
            "itemType": "preprint",
            "title": title,
            "creators": creators,
            "date": str(record["year"]),
            "repository": "arXiv",
        }

    item.update({
        "url": record["url"],
        "abstractNote": extract_abstract(raw_text),
        "extra": f"AegisFX | feature: {record.get('feature', '')} | "
                 f"{record.get('why', '')}",
        "collections": [collection_key],
        "tags": [{"tag": "AegisFX"}, {"tag": record.get("feature", "unknown")}],
    })
    return item


def cmd_check() -> int:
    try:
        sid = discover_server_id()
    except Exception as exc:                           # noqa: BLE001
        print(f"连不上 Zotero：{exc}")
        return 1
    print(f"Zotero 可达，Server-ID = {sid}")
    try:
        key = load_api_key()
        print(f"API key 已就位（{key[:8]}...）")
    except SystemExit as exc:
        print(exc)
        return 1
    if not MANIFEST.exists():
        print("还没有 manifest.json，先跑 fetch_literature.py --fetch")
        return 1
    papers = json.loads(MANIFEST.read_text(encoding="utf-8"))["papers"]
    ready = [k for k, v in papers.items() if v.get("status") == "ok"]
    print(f"清单里 {len(ready)} 篇可导入")
    missing = [k for k in ready
               if not (PAPER_DIR / "pdf" / f"{k}.pdf").exists()]
    if missing:
        print(f"缺 PDF 的：{missing}")
    return 0


def cmd_import(only: list) -> int:
    papers = json.loads(MANIFEST.read_text(encoding="utf-8"))["papers"]
    h = headers()
    ckey = find_collection(COLLECTION_NAME)
    print(f"collection「{COLLECTION_NAME}」= {ckey}\n")

    log = (json.loads(IMPORT_LOG.read_text(encoding="utf-8"))
           if IMPORT_LOG.exists() else {"collection_key": ckey, "items": {}})
    log["collection_key"] = ckey

    from scripts.fetch_literature import CATALOGUE
    by_key = {e["key"]: e for e in CATALOGUE}

    ok = fail = 0
    for key, record in papers.items():
        if only and key not in only:
            continue
        if record.get("status") != "ok":
            continue
        if key in log["items"]:
            print(f"  [跳过] {key} 已导入")
            continue
        pdf = PAPER_DIR / "pdf" / f"{key}.pdf"
        raw = PAPER_DIR / "raw" / f"{key}.txt"
        if not pdf.exists() or not raw.exists():
            print(f"  [缺件] {key}")
            fail += 1
            continue

        text = raw.read_text(encoding="utf-8")
        item = build_item(by_key[key], record, text, ckey)
        r = requests.post(f"{BASE}/items", headers=h, json=[item], timeout=30)
        if r.status_code not in (200, 201) or not r.json().get("successful"):
            print(f"  [失败] {key}: {r.status_code} {r.text[:120]}")
            fail += 1
            continue
        ikey = r.json()["successful"]["0"]["data"]["key"]

        att = {"itemType": "attachment", "linkMode": "linked_file",
               "parentItem": ikey, "title": "Full Text PDF",
               "path": str(pdf.resolve()), "contentType": "application/pdf"}
        ra = requests.post(f"{BASE}/items", headers=h, json=[att], timeout=30)
        akey = (ra.json()["successful"]["0"]["data"]["key"]
                if ra.status_code in (200, 201) and ra.json().get("successful")
                else None)

        log["items"][key] = {"item_key": ikey, "attachment_key": akey,
                             "title": record["title"]}
        print(f"  [ok]   {key:<22} item={ikey} attach={akey}")
        ok += 1

    IMPORT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(f"\n导入 {ok} 篇，失败 {fail} 篇")
    return 0 if fail == 0 else 1


def cmd_fix_authors() -> int:
    """给已导入但没抓到作者的条目回填 creators。"""
    if not IMPORT_LOG.exists():
        print("没有导入记录")
        return 1
    log = json.loads(IMPORT_LOG.read_text(encoding="utf-8"))
    papers = json.loads(MANIFEST.read_text(encoding="utf-8"))["papers"]
    h = headers()
    fixed = skipped = 0
    for key, rec in log.get("items", {}).items():
        ikey, record = rec.get("item_key"), papers.get(key)
        raw = PAPER_DIR / "raw" / f"{key}.txt"
        if not (ikey and record and raw.exists()):
            continue
        creators = creators_of(record, raw.read_text(encoding="utf-8"))
        if not creators:
            print(f"  [没抽到] {key}")
            continue
        g = requests.get(f"{BASE}/items/{ikey}", headers=h, timeout=15)
        if g.status_code != 200 or g.json()["data"].get("creators"):
            skipped += 1
            continue
        ver = g.json()["version"]
        r = requests.patch(
            f"{BASE}/items/{ikey}",
            headers=dict(h, **{"If-Unmodified-Since-Version": str(ver)}),
            json={"creators": creators}, timeout=20)
        names = ", ".join(c["lastName"] for c in creators[:3])
        print(f"  [ok] {key:<22} {names}  ({r.status_code})")
        fixed += 1
    print(f"\n补了 {fixed} 条，跳过 {skipped} 条")
    return 0


def cmd_clean() -> int:
    if not IMPORT_LOG.exists():
        print("没有导入记录，无需清理")
        return 0
    log = json.loads(IMPORT_LOG.read_text(encoding="utf-8"))
    h = headers()
    n = 0
    for key, rec in log.get("items", {}).items():
        # 删除要带版本号，否则 428
        for kind, ikey in (("attachment", rec.get("attachment_key")),
                           ("item", rec.get("item_key"))):
            if not ikey:
                continue
            g = requests.get(f"{BASE}/items/{ikey}", headers=h, timeout=15)
            if g.status_code != 200:
                continue
            ver = g.json()["version"]
            d = requests.delete(f"{BASE}/items/{ikey}", headers=dict(
                h, **{"If-Unmodified-Since-Version": str(ver)}), timeout=15)
            print(f"  删 {kind} {ikey}: {d.status_code}")
            n += 1
    print(f"共删除 {n} 个对象（collection 保留）")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="把文献导入 Zotero")
    parser.add_argument("--check", action="store_true", help="检查连通性")
    parser.add_argument("--import", dest="do_import", action="store_true",
                        help="执行导入")
    parser.add_argument("--clean", action="store_true", help="删除本工具导入的条目")
    parser.add_argument("--fix-authors", action="store_true",
                        help="给已导入条目回填作者")
    parser.add_argument("--only", nargs="*", default=[], help="只导入指定 key")
    args = parser.parse_args()

    if args.clean:
        return cmd_clean()
    if args.fix_authors:
        return cmd_fix_authors()
    if args.do_import:
        return cmd_import(args.only)
    return cmd_check()


if __name__ == "__main__":
    raise SystemExit(main())
