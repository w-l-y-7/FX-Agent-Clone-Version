"""行情层。

**这里只做一件事：把两套行情分清楚，绝不拼接。**

- `paper_market()` —— 论文那份 `Data.xlsx`，2017-01-03 → 2024-12-31。
  事件库、市场模型、全部评测都用它。
- `fred_usdcny()` —— FRED 的 `DEXCHUS`，1981 到今天。**只用于给 2024-12-31
  之后的新事件补标签**，不进历史评测。

为什么不能拼：两条序列在 1863 个重叠交易日上，**价格水平相关性 0.9994、
波动率几乎一样（0.2867% vs 0.2804%），但日收益相关性只有 0.746**。
日期是对齐的（平移 ±1 天相关性掉到 0.24 / 0.01），所以差异来自定盘口径——
FRED 记的是美联储 H.10 的午盘买入价，`Data.xlsx` 那列是另一套。

对事件研究来说日收益才是原料，0.746 的相关性意味着拼接会在 20 日窗口里
累积出百分之零点几的伪异常收益，和真实信号（20 日均值 +0.20%）同量级。
所以宁可承认「2025 年以后用另一套口径」，也不把两条序列缝起来。
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

from ..paths import DATA_DIR
from ..paths import DAILY_DATA as PAPER_DATA_PATH

_FRED_CACHE = DATA_DIR / "fred_DEXCHUS.csv"
_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXCHUS"
_FRED_MAX_AGE_DAYS = 7

# 市场状态向量（方案 §4.3 的四个变量），全部能从 Data.xlsx 算出来。
MARKET_FEATURES: tuple[str, ...] = (
    "rate_spread_1y",
    "usd_index",
    "fx_vol_20d",
    "fx_reserve_chg_3m",
)


def paper_market() -> pd.DataFrame:
    """论文那份日频表，另加市场状态与市场模型要用的派生列。

    返回列：原始宏观列 + `rate_spread_1y` / `fx_vol_20d` / `fx_reserve_chg_3m`
    + `ret_fx` / `ret_dxy` / `d_spread`。
    """
    # 复现代码里的读表逻辑照用（它已经处理了日期序列号、列名错拼、月度值填 0）
    from src.core.research_dataset import load_daily_frame

    frame = load_daily_frame()

    frame["rate_spread_1y"] = (
        frame["CN_1Y_GovBond_Yield"] - frame["US_1Y_Treasury_Yield"]
    )
    frame["usd_index"] = frame["USD_Index"]
    frame["fx_vol_20d"] = frame["USDCNY"].pct_change().rolling(20).std()
    frame["fx_reserve_chg_3m"] = frame["China_FX_Reserves"].pct_change(63)

    # 市场模型的原料：汇率收益、美元指数收益、利差变化
    frame["ret_fx"] = frame["USDCNY"].pct_change()
    frame["ret_dxy"] = frame["USD_Index"].pct_change()
    frame["d_spread"] = frame["rate_spread_1y"].diff()
    return frame


def _fred_cache_is_fresh() -> bool:
    if not _FRED_CACHE.exists():
        return False
    age = date.today() - date.fromtimestamp(_FRED_CACHE.stat().st_mtime)
    return age <= timedelta(days=_FRED_MAX_AGE_DAYS)


def fred_usdcny(*, refresh: bool = False) -> pd.Series:
    """FRED 的日频 USD/CNY。

    结果缓存到 `data/aegisfx/fred_DEXCHUS.csv`，默认 7 天内不重新下载——
    没网也能跑，这对一个要反复跑的评测脚本很重要。

    拉不到且没有缓存时返回空 Series，调用方自己判断。
    """
    if refresh or not _fred_cache_is_fresh():
        try:
            import requests

            response = requests.get(_FRED_URL, timeout=30)
            response.raise_for_status()
            import io

            raw = pd.read_csv(io.StringIO(response.text))
            raw.columns = ["date", "value"]
            raw["date"] = pd.to_datetime(raw["date"])
            raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
            raw = raw.dropna().set_index("date")["value"]

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            raw.to_csv(_FRED_CACHE, encoding="utf-8")
        except Exception:
            # 拉不到就用旧缓存；连旧缓存都没有就返回空
            pass

    if not _FRED_CACHE.exists():
        return pd.Series(dtype="float64")

    cached = pd.read_csv(_FRED_CACHE, index_col=0)["value"]
    cached.index = pd.to_datetime(cached.index)
    return cached.sort_index()


def extended_usdcny() -> pd.Series:
    """`Data.xlsx` 结束之后的那一段，来自 FRED。空的话说明没拉到。"""
    series = fred_usdcny()
    if series.empty:
        return series
    return series.loc[series.index > paper_market().index[-1]]


@lru_cache(maxsize=1)
def _paper_index() -> pd.DatetimeIndex:
    return paper_market().index


def market_state_on(day: pd.Timestamp) -> Optional[dict]:
    """取某个交易日的市场状态。**用事件日前一交易日**的值，避免用到当天信息。

    取不到就返回 None（比如日期早于数据起点）。
    """
    frame = paper_market()
    position = frame.index.searchsorted(pd.Timestamp(day), side="left")
    if position == 0 or position >= len(frame) + 1:
        return None
    position = min(position, len(frame)) - 1
    row = frame.iloc[position]
    return {name: float(row[name]) for name in MARKET_FEATURES}


def describe_sources() -> str:
    """给界面/日志用的一段话，说清楚用的是哪套行情。"""
    frame = paper_market()
    extension = extended_usdcny()
    lines = [
        f"论文数据（Data.xlsx）  {frame.index[0].date()} → {frame.index[-1].date()}"
        f"　{len(frame)} 个交易日",
        f"FRED 扩展段           {len(extension)} 个交易日"
        + (f"，{extension.index[0].date()} → {extension.index[-1].date()}"
           if len(extension) else "（未取到）"),
        "两段不拼接：日收益相关性仅 0.746，拼起来会引入同量级的伪异常收益。",
    ]
    return "\n".join(lines)
