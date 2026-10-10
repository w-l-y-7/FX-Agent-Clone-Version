"""从开放获取来源抓取文献，转成纯文本放进知识库。

    .\\venv\\Scripts\\python.exe scripts\\fetch_literature.py --list
    .\\venv\\Scripts\\python.exe scripts\\fetch_literature.py --fetch
    .\\venv\\Scripts\\python.exe scripts\\fetch_literature.py --fetch --only nber_w27654

## 为什么要有这个脚本

知识库原来只有 4 份本项目自撰的机制说明，没有一篇真实文献。方案 §12 要求的
「≥120 条观点、覆盖学术/官方/行业来源」因此建不出来。

这个脚本只做一件事：**把公开免费的论文正文拉到本地**。
它不做观点抽取——抽取是 `src/aegisfx/cikrag/literature.py` 的事。

## 来源都必须是合法免费全文

- NBER working paper：作者自存版，官网直接提供 PDF，无需订阅
- arXiv：全部开放获取

**没有**任何付费墙绕过。抓不到的（BIS 在本机网络下连接被重置）就不抓，
在清单里标 `unreachable`，不假装成功。

## 产物

    data/knowledge_base/papers/raw/<key>.txt     ← 转好的正文
    data/knowledge_base/papers/manifest.json     ← 每篇的来源、URL、日期、字符数
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

PAPER_DIR = _PROJECT_ROOT / "data" / "knowledge_base" / "papers"
RAW_DIR = PAPER_DIR / "raw"
PDF_DIR = PAPER_DIR / "pdf"
MANIFEST = PAPER_DIR / "manifest.json"

# ---------------------------------------------------------------------------
# 清单：每篇都注明来源与为什么选它（对应哪个特征）
# ---------------------------------------------------------------------------
# feature 只是**选题理由**，不是抽取结果。真实方向由 literature.py 从正文判定。
CATALOGUE = [
    # ---- 关税 / 贸易摩擦 → 汇率（对应 Event_Types）----
    dict(key="nber_w27654", source="nber", wid="w27654",
         title="To What Extent Are Tariffs Offset By Exchange Rates?",
         year=2020, feature="Event_Types",
         why="关税冲击有多少被汇率抵消——直接回答「贸易摩擦事件如何影响汇率」"),
    dict(key="nber_w26396", source="nber", wid="w26396",
         title="Tariff Passthrough at the Border and at the Store: Evidence from US Trade Policy",
         year=2019, feature="Event_Types",
         why="关税的价格传递：进口商吸收还是转嫁给汇率"),
    dict(key="nber_w17593", source="nber", wid="w17593",
         title="The External Impact of China's Exchange Rate Policy: Evidence from Firm Level Data",
         year=2011, feature="Event_Types",
         why="人民币重估对企业估值的影响，含 2005/2010 两次汇改的意外公告"),

    # ---- 中国汇率框架 / 中间价（对应 USDCNY_Framework）----
    dict(key="nber_w14673", source="nber", wid="w14673",
         title="China's Current Account and Exchange Rate",
         year=2009, feature="USDCNY_Framework",
         why="中国经常账户与汇率的长期关系"),
    dict(key="nber_w12551", source="nber", wid="w12551",
         title="Are Currency Appreciations Contractionary in China?",
         year=2006, feature="USDCNY_Framework",
         why="人民币升值是否紧缩——升值方向效应的经典争论"),
    dict(key="nber_w17074", source="nber", wid="w17074",
         title="Exchange Rates in Emerging Countries: Eleven Empirical Regularities",
         year=2011, feature="USDCNY_Framework",
         why="新兴市场汇率的 11 条经验规律，覆盖汇率制度与实际汇率"),

    # ---- 外汇储备 / 干预（对应 China_FX_Reserves）----
    dict(key="nber_w34888", source="nber", wid="w34888",
         title="Drivers of Dollar Share in Foreign Exchange Reserves",
         year=2026, feature="China_FX_Reserves",
         why="储备币种份额的驱动因素"),
    dict(key="nber_w20386", source="nber", wid="w20386",
         title="International Reserves Before and After the Global Crisis: Is There No End to Hoarding?",
         year=2014, feature="China_FX_Reserves",
         why="危机前后储备囤积动机的变化"),

    # ---- 资本流动 / 利差（对应 US_1Y_Treasury_Yield）----
    dict(key="nber_w30887", source="nber", wid="w30887",
         title="International Capital Flow Pressures and Global Factors",
         year=2023, feature="US_1Y_Treasury_Yield",
         why="全球因子对跨境资本流的驱动，利差传导的上游"),
    dict(key="nber_w32877", source="nber", wid="w32877",
         title="Capital Controls on Outflows: New Evidence and a Theoretical Framework",
         year=2024, feature="US_1Y_Treasury_Yield",
         why="资本流出管制——「条件不同则符号不同」的典型场景"),

    # ---- 美元指数（对应 USD_Index）----
    dict(key="nber_w34626", source="nber", wid="w34626",
         title="The Ins & Outs of Chinese Monetary Policy Transmission",
         year=2026, feature="USD_Index",
         why="中国货币政策传导，含外部货币环境渠道"),

    # ---- 油价 / 大宗（对应 WTI_Futures_Price）----
    dict(key="arxiv_1602.03271", source="arxiv", aid="1602.03271",
         title="A study of co-movements between oil price, stock index and exchange rate under a cross-bicorrelation test",
         year=2016, feature="WTI_Futures_Price",
         why="油价、股指与汇率的联动"),

    # ---- 人民币在岸/离岸（对应 Transmission_Channels）----
    dict(key="arxiv_1803.09432", source="arxiv", aid="1803.09432",
         title="Time-dependent lead-lag relationship between the onshore and offshore Renminbi exchange rates",
         year=2018, feature="Transmission_Channels",
         why="在岸与离岸人民币的领先滞后——传导渠道的直接证据"),
    dict(key="arxiv_2605.25392", source="arxiv", aid="2605.25392",
         title="One Currency, Two Forward Prices: The Onshore-Offshore Renminbi Puzzle",
         year=2026, feature="Transmission_Channels",
         why="同一货币两个远期价格——在岸离岸分割的最新证据"),

    # ---- 储备（对应 China_FX_Reserves）----
    dict(key="arxiv_2206.13751", source="arxiv", aid="2206.13751",
         title="Estimating the Currency Composition of Foreign Exchange Reserves",
         year=2022, feature="China_FX_Reserves",
         why="储备币种构成的估计方法"),

    # ---- 利差 / 平价条件（对应 US_1Y_Treasury_Yield）----
    dict(key="arxiv_1303.4314", source="arxiv", aid="1303.4314",
         title="Reinvestigating the Uncovered Interest Rate Parity Puzzle via Analysis of Multivariate Tail Dependence",
         year=2013, feature="US_1Y_Treasury_Yield",
         why="无抛补利率平价之谜——利差与汇率关系的核心争论"),
    dict(key="arxiv_2605.20137", source="arxiv", aid="2605.20137",
         title="A Three-Variable Benchmark for Post-GFC Covered Interest Parity Deviations",
         year=2026, feature="US_1Y_Treasury_Yield",
         why="危机后抛补利率平价偏离的基准"),

    # ---- 资本管制（对应 Event_Types）----
    dict(key="arxiv_2505.23025", source="arxiv", aid="2505.23025",
         title="Learning to Regulate: A New Event-Level Dataset of Capital Control Measures",
         year=2025, feature="Event_Types",
         why="资本管制措施的事件级数据集"),
    dict(key="arxiv_2506.09168", source="arxiv", aid="2506.09168",
         title="Export proceeds repatriation policies: A shield against exchange rate volatility",
         year=2025, feature="Event_Types",
         why="出口收汇回流政策作为汇率波动的缓冲"),

    # ---- 汇率预测（方法对照）----
    dict(key="arxiv_2410.19241", source="arxiv", aid="2410.19241",
         title="Enhancing Exchange Rate Forecasting with Explainable Deep Learning Models",
         year=2024, feature="USDCNY_Framework",
         why="可解释深度学习做汇率预测，特征重要性的外部对照"),
    dict(key="arxiv_2606.15058", source="arxiv", aid="2606.15058",
         title="Machine Learning and the Random Walk Puzzle: Forecasting the CAD/USD Exchange Rate",
         year=2026, feature="USDCNY_Framework",
         why="机器学习能否打败随机游走——汇率可预测性的基准争论"),
]

USER_AGENT = "AegisFX-literature/0.1 (academic reproduction; contact: local)"


def _url_of(entry: dict) -> str:
    if entry["source"] == "nber":
        wid = entry["wid"]
        return f"https://www.nber.org/system/files/working_papers/{wid}/{wid}.pdf"
    if entry["source"] == "arxiv":
        return f"https://arxiv.org/pdf/{entry['aid']}"
    raise ValueError(entry["source"])


def _download(url: str, dest: Path, timeout: int = 90) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read()
    dest.write_bytes(payload)
    return len(payload)


# PDF 里的连字（ff / fi / fl / ffi / ffl）常常没有 Unicode 映射，
# pypdf 会把它们吐成字体原始码位的控制字符。实测 NBER 的 PDF：
#
#     "imposed new tari\x1bs of 15.1 percent"   ->  tariffs
#     "We \x1crst present a simple"             ->  We first present
#     "in terms of in\x1dation"                 ->  inflation
#     "the coe\x1ecient"                        ->  coefficient
#
# 不还原的话，卡片里抄的原文句子在正文里根本搜不到，引用核查会全部失败。
# 这是 Adobe 标准连字编码顺序。
_LIGATURES = {
    "\x1b": "ff", "\x1c": "fi", "\x1d": "fl",
    "\x1e": "ffi", "\x1f": "ffl",
    # 少数 PDF 把连字吐成真正的 Unicode 连字字符（U+FB00 起）而不是控制字符
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
}

# 其余控制字符（除换行/制表）一律删掉——它们在正文里没有意义，
# 留着只会让引用核查莫名其妙地失败
_CONTROL_JUNK = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1a\x7f]")


def _pdf_to_text(pdf_path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:                       # noqa: BLE001
            pages.append(f"[page failed: {exc}]")
    text = "\n".join(pages)
    for raw, ligature in _LIGATURES.items():
        text = text.replace(raw, ligature)
    text = _CONTROL_JUNK.sub("", text)
    text = text.replace("­", "")                  # soft hyphen
    text = re.sub(r"-\n(?=[a-z])", "", text)           # 跨行连字符
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"fetched_at": None, "papers": {}}


def save_manifest(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_list() -> int:
    manifest = load_manifest()
    done = manifest.get("papers", {})
    print(f"{'key':<22}{'year':<6}{'status':<14}title")
    print("-" * 100)
    for entry in CATALOGUE:
        status = done.get(entry["key"], {}).get("status", "未抓取")
        print(f"{entry['key']:<22}{entry['year']:<6}{status:<14}{entry['title'][:58]}")
    print(f"\n共 {len(CATALOGUE)} 篇；已就绪 "
          f"{sum(1 for e in CATALOGUE if done.get(e['key'], {}).get('status') == 'ok')} 篇")
    return 0


def cmd_fetch(only: list) -> int:
    manifest = load_manifest()
    manifest["fetched_at"] = date.today().isoformat()
    papers = manifest.setdefault("papers", {})
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    # PDF 本体留着——Zotero 附件要指向真实文件，不能转完就删
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    failures = []
    for entry in CATALOGUE:
        if only and entry["key"] not in only:
            continue
        url = _url_of(entry)
        record = dict(source=entry["source"], title=entry["title"],
                      year=entry["year"], url=url, feature=entry["feature"],
                      why=entry["why"])
        pdf_path = PDF_DIR / f"{entry['key']}.pdf"
        try:
            size = _download(url, pdf_path)
            text = _pdf_to_text(pdf_path)
            if len(text) < 2000:
                raise ValueError(f"正文只抽到 {len(text)} 字符，疑似扫描版或解析失败")
            out = RAW_DIR / f"{entry['key']}.txt"
            out.write_text(text, encoding="utf-8")
            record.update(status="ok", pdf_bytes=size, text_chars=len(text),
                          text_words=len(text.split()),
                          pdf_path=str(pdf_path),
                          fetched_at=date.today().isoformat())
            print(f"  [ok]   {entry['key']:<22} {size/1024:7.0f} KB -> "
                  f"{len(text):>7} 字符")
        except Exception as exc:                       # noqa: BLE001
            record.update(status="failed", error=str(exc)[:200],
                          fetched_at=date.today().isoformat())
            failures.append(entry["key"])
            print(f"  [fail] {entry['key']:<22} {str(exc)[:70]}")
        papers[entry["key"]] = record
        time.sleep(1.0)                                 # 对站点客气一点

    save_manifest(manifest)
    print(f"\n完成：成功 {len(CATALOGUE) - len(failures)} / 失败 {len(failures)}")
    if failures:
        print(f"失败的：{', '.join(failures)}")
    return 1 if failures else 0


def repair_existing() -> int:
    """就地修复已抓文本里的连字控制字符，不必重新下载。

    `_pdf_to_text` 修好之后新抓的本来就是干净的；这个命令是给
    修复之前已经落盘的那批用的。幂等，跑几遍都一样。
    """
    fixed = 0
    for path in sorted(RAW_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        new = text
        for raw, ligature in _LIGATURES.items():
            new = new.replace(raw, ligature)
        new = _CONTROL_JUNK.sub("", new)
        if new != text:
            path.write_text(new, encoding="utf-8")
            fixed += 1
            print(f"  修复 {path.name}：还原 {len(text) - len(new)} 个连字")
    print(f"共修复 {fixed} 篇")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="抓取开放获取文献")
    parser.add_argument("--list", action="store_true", help="只看清单与状态")
    parser.add_argument("--fetch", action="store_true", help="执行抓取")
    parser.add_argument("--repair", action="store_true",
                        help="就地修复已抓文本里的连字")
    parser.add_argument("--only", nargs="*", default=[], help="只抓指定的 key")
    args = parser.parse_args()

    if args.repair:
        return repair_existing()
    if args.list or not (args.fetch or args.only):
        return cmd_list()
    return cmd_fetch(args.only)


if __name__ == "__main__":
    raise SystemExit(main())
