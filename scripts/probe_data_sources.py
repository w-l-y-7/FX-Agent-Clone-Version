"""数据源接入可行性探测（第 1 周交付物之一）。

方案里把「分钟级实时接入」标成了唯一可裁剪项，但裁剪的前提是**先知道能接到什么**。
这个脚本把候选源挨个试一遍，输出一张「能用 / 不能用 / 为什么」的表，
别等到第 7 周才发现首选源根本连不上。

    .\\venv\\Scripts\\python.exe scripts\\probe_data_sources.py
    .\\venv\\Scripts\\python.exe scripts\\probe_data_sources.py --skip-network   # 只查依赖

要联网。结果写到 `reports/aegisfx_data_source_probe.json`。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.aegisfx.paths import load_env  # noqa: E402

_REPORT = _PROJECT_ROOT / "reports" / "aegisfx_data_source_probe.json"
_TIMEOUT = 15

results: list[dict] = []


def record(name, ok, detail, extra=None):
    results.append({"source": name, "ok": ok, "detail": detail, **(extra or {})})
    mark = "可用  " if ok else "不可用"
    print(f"  [{mark}] {name}")
    print(f"           {detail}")


# --------------------------------------------------------------------------


def probe_dependencies():
    print("\n[1] 依赖")
    for module in ("akshare", "requests", "pandas"):
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "?")
            record(f"依赖 {module}", True, f"已安装，版本 {version}")
        except ImportError:
            record(f"依赖 {module}", False, "未安装")


def probe_akshare_catalogue():
    """先看看这个版本的 akshare 到底有哪些外汇接口，再逐个试。

    比写死一个函数名稳：akshare 的接口名换过好几轮，
    硬编码的名字在新版本里很可能已经不存在了。
    """
    print("\n[2] akshare 外汇接口清点")
    try:
        import akshare as ak
    except ImportError:
        record("akshare 接口清点", False, "akshare 未安装")
        return

    names = [n for n in dir(ak)
             if any(k in n.lower() for k in ("currency", "forex", "fx_", "_fx"))]
    record("akshare 接口清点", bool(names),
           f"找到 {len(names)} 个疑似外汇接口", {"candidates": sorted(names)[:40]})


def probe_akshare_usdcnh():
    """试几个最可能给 USD/CNH 日频的接口。"""
    print("\n[3] akshare 取 USD/CNH 日频")
    try:
        import akshare as ak
    except ImportError:
        record("akshare USD/CNH 日频", False, "akshare 未安装")
        return

    attempts = [
        ("currency_boc_sina", lambda: ak.currency_boc_sina(
            symbol="美元", start_date="20240101", end_date="20240131")),
        ("fx_spot_quote", lambda: ak.fx_spot_quote()),
        ("currency_hist", lambda: ak.currency_hist(
            symbol="USDCNH", start_date="20240101", end_date="20240131")),
    ]
    for name, call in attempts:
        try:
            frame = call()
            shape = getattr(frame, "shape", None)
            record(f"akshare.{name}", True,
                   f"返回 {shape}，列名 {list(getattr(frame, 'columns', []))[:8]}",
                   {"function": name})
            return
        except Exception as exc:
            record(f"akshare.{name}", False, f"{type(exc).__name__}: {str(exc)[:110]}")


def probe_fred():
    """FRED 的 DEXCHUS：日频在岸人民币，不用密钥。"""
    print("\n[4] FRED DEXCHUS（日频 USD/CNY 核验源）")
    try:
        import requests
    except ImportError:
        record("FRED DEXCHUS", False, "requests 未安装")
        return

    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXCHUS"
    try:
        response = requests.get(url, timeout=_TIMEOUT)
        response.raise_for_status()
        lines = [line for line in response.text.strip().splitlines() if line]
        record("FRED DEXCHUS", True,
               f"HTTP {response.status_code}，{len(lines) - 1} 行，"
               f"首行 {lines[0]}，末行 {lines[-1]}",
               {"first_row": lines[1] if len(lines) > 1 else None,
                "last_row": lines[-1]})
    except Exception as exc:
        record("FRED DEXCHUS", False, f"{type(exc).__name__}: {str(exc)[:110]}")


def probe_oanda():
    """OANDA Practice：先看有没有凭证，有的话探一下 instruments 接口。

    **只探 Practice，不碰 Live。**
    """
    print("\n[5] OANDA Practice")
    token = os.getenv("OANDA_API_KEY") or os.getenv("OANDA_TOKEN")
    if not token:
        record("OANDA Practice", False,
               "未配置 OANDA_API_KEY。需要先注册 Practice 账户；"
               "注册前请确认所在地区能否开通、是否覆盖 USD/CNH、数据授权范围",
               {"needs": "OANDA_API_KEY"})
        return

    try:
        import requests
    except ImportError:
        record("OANDA Practice", False, "requests 未安装")
        return

    host = os.getenv("OANDA_HOST", "https://api-fxpractice.oanda.com")
    try:
        response = requests.get(
            f"{host}/v3/accounts",
            headers={"Authorization": f"Bearer {token}"},
            timeout=_TIMEOUT,
        )
        if response.status_code == 200:
            accounts = response.json().get("accounts", [])
            record("OANDA Practice", True,
                   f"HTTP 200，{len(accounts)} 个账户，host={host}",
                   {"accounts": accounts})
        else:
            record("OANDA Practice", False,
                   f"HTTP {response.status_code}：{response.text[:110]}")
    except Exception as exc:
        record("OANDA Practice", False, f"{type(exc).__name__}: {str(exc)[:110]}")


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="数据源接入可行性探测")
    parser.add_argument("--skip-network", action="store_true",
                        help="只查依赖，不联网")
    args = parser.parse_args()

    load_env()

    print("=" * 78)
    print("AegisFX-Agent 数据源可行性探测")
    print("=" * 78)

    probe_dependencies()
    if not args.skip_network:
        probe_akshare_catalogue()
        probe_akshare_usdcnh()
        probe_fred()
        probe_oanda()

    usable = [r for r in results if r["ok"]]
    print("\n" + "=" * 78)
    print(f"结论：{len(usable)}/{len(results)} 项可用")
    print("=" * 78)

    print("\n日频研究线（必需）")
    daily_ok = any(r["ok"] and "FRED" in r["source"] for r in results)
    print(f"  核验源 FRED DEXCHUS   {'可用' if daily_ok else '不可用'}")
    print("  主源 Data.xlsx        可用（仓库自带，2017-01-03 → 2024-12-31）")

    print("\n模拟执行线（第 7 周，可裁剪）")
    oanda = next((r for r in results if r["source"] == "OANDA Practice"), None)
    if oanda and oanda["ok"]:
        print("  OANDA Practice        可用")
    else:
        print("  OANDA Practice        不可用 —— 走降级路线：")
        print("                        akshare 日频 CNH + 人工投递 data/inbox/")
        print("                        执行闸门与风险 Agent 的评测不受影响（在历史留出集上做）")

    _REPORT.parent.mkdir(parents=True, exist_ok=True)
    _REPORT.write_text(json.dumps({
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {_REPORT.relative_to(_PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
