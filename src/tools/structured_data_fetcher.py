from datetime import date, timedelta
from typing import Any, Dict

import akshare as ak
import pandas as pd

from ..core.abstractions.base_tool import BaseTool

# 中国银行汇率接口按中文货币名查询
_CNY_NAMES: Dict[str, str] = {
    "USD": "美元",
    "EUR": "欧元",
    "JPY": "日元",
    "GBP": "英镑",
    "HKD": "港币",
    "AUD": "澳大利亚元",
    "CAD": "加拿大元",
    "CHF": "瑞士法郎",
    "SGD": "新加坡元",
}

# 中行中间价以 100 外币为单位报价，取人民币直盘时要还原成 1 单位
_QUOTE_UNITS = 100.0


class StructuredDataFetcher(BaseTool):
    """Fetches historical exchange rates via the Bank of China rates (AKShare)."""

    name = "structured_data_fetcher"
    description = (
        "Fetches historical exchange-rate data for a currency pair, e.g. EUR/USD."
    )

    def execute(self, **kwargs: Any) -> Dict[str, Any]:
        symbol = kwargs.get("symbol", "EUR/USD").upper()
        days = int(kwargs.get("days", 30))

        try:
            base, quote = symbol.split("/")
        except ValueError:
            raise ValueError(f"Symbol must look like 'EUR/USD', got: {symbol}")

        end = date.today()
        start = end - timedelta(days=days)

        base_series = self._cny_rate(base, start, end)
        if quote == "CNY":
            series = base_series / _QUOTE_UNITS
        else:
            series = base_series / self._cny_rate(quote, start, end)

        # 中行只在交易日发布牌价，周末与节假日的行要去掉
        series = series.dropna().sort_index()

        return {
            "source": "akshare:currency_boc_sina",
            "symbol": symbol,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "observations": len(series),
            "series": [
                {"date": index.date().isoformat(), "close": round(float(value), 4)}
                for index, value in series.items()
            ],
        }

    @staticmethod
    def _cny_rate(currency: str, start: date, end: date) -> pd.Series:
        """1 单位 `currency` 兑人民币的中间价序列。"""
        name = _CNY_NAMES.get(currency)
        if name is None:
            raise ValueError(
                f"Unsupported currency '{currency}'. Supported: {', '.join(sorted(_CNY_NAMES))}"
            )

        frame = ak.currency_boc_sina(
            symbol=name,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        frame = frame.dropna(subset=["央行中间价"])
        return pd.Series(
            frame["央行中间价"].astype(float).values,
            index=pd.to_datetime(frame["日期"]),
        )
