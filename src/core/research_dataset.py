"""论文实验的数据层：把 `src/legacy_models/Data.xlsx` 读成干净的分析表。

这个 Excel 是整篇论文的实验基础，两个 sheet 是**上下游关系**：

- `Sheet1`（445 行）——原始中美贸易摩擦新闻语料，是 PA1 的产出。
  每条含日期、一句话摘要、事件类型（中文，多标签用 `/` 分隔）、来源、情感。
- `Data`（1944 行 × 35 列）——日频建模表，是 PA2 的产出。
  目标列 `USDCNY`，加上 7 个宏观驱动变量、25 个事件类哑变量、1 个未使用的贸易差额。

原文件有四个坑，都在这里就地修好（调用方拿到的一定是干净数据）：

1. `Date` 是 Excel 序列号整数（42738…45657），不是日期；
2. 列名有多余空格和拼写错误（`China_ FX_Reserves`、`Purchasses`、`Enteprises`），
   且 `US Tariff increases on China` 与其他列大小写不一致；
3. 月度宏观变量（外汇储备、中美 CPI）在日频表里被填成**字面的 0**，
   全列只有约 3% 是非零。直接当 0 用会把数据毁掉；
4. `Sino-US trade balance` 的量级（最大约 420 万）与其他列完全不同，明显是另一套
   单位；原表也没标注口径，所以不猜。论文 Table 4 的特征集里同样没有它，
   因此本模块只把它作为附加列保留，不放进 `TIME_SERIES_FEATURES`。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_DATA_SHEET = "Data"
_NEWS_SHEET = "Sheet1"

# Excel 的日期序列号以 1899-12-30 为原点（不是 1900-01-01，那是 Excel 自己的闰年 bug）
_EXCEL_EPOCH = "1899-12-30"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXCEL_PATH = _PROJECT_ROOT / "src" / "legacy_models" / "Data.xlsx"

# 非零值占比低于这个数，就认为该列是「低频数据被填了 0」而不是「真的有 0」。
# 实测：本表里只有外汇储备、中美 CPI、贸易差额四列低于它（各约 3%），
# 而 WTI 油价（min = -37.63）、美国 1 年期国债收益率（min = 0.04）都远高于它，
# 不会被误伤——这两个 0 附近的真实值一旦被当成缺失数据抹掉，后果比不填还糟。
_LOW_FREQUENCY_NONZERO_FRACTION = 0.2

TARGET = "USDCNY"

# 论文 Table 4 的 7 个宏观/金融驱动变量（按论文命名）。
# 注意：本表里的 `Sino-US trade balance` 不在论文特征集内，故意不列入。
TIME_SERIES_FEATURES: List[str] = [
    "USD_Index",
    "CN_1Y_GovBond_Yield",
    "US_1Y_Treasury_Yield",
    "China_FX_Reserves",
    "CN_CPI_MoM",
    "US_CPI_MoM",
    "WTI_Futures_Price",
]

# 论文 Table 5 的事件特征：16 个逐事件哑变量 + 3 个来源 + 6 个情感聚合 = 25 个。
# 全部是 0/1 指示变量（「当天有没有发生过这类事件」），不是发生次数——
# 各列求和正好等于论文 Table 5 的 sample size 列
# （Negative events 248、Positive events 181、US restrictions 160 …）。
EVENT_TYPE_FEATURES: List[str] = [
    "China_Increasing_Purchases_of_US_Goods",
    "China_Restrictions_on_US_Enterprises",
    "China_Reducing_Restrictions_on_US_Enterprises",
    "China_Tariff_Increases_on_the_US",
    "China_Litigation_or_Investigation_against_the_US",
    "China_Tariff_Relief_Policies_for_the_US",
    "China_Criticizing_the_US",
    "Sino_US_Dialogue",
    "US_Restrictions_on_Chinese_Enterprises",
    "US_Reducing_Restrictions_on_Chinese_Enterprises",
    "US_Tariff_Increases_on_China",
    "US_Litigation_or_Investigation_against_China",
    "US_Tariff_Relief_Policies_for_China",
    "US_Removing_China_from_Currency_Manipulator_List",
    "US_Designating_China_as_Currency_Manipulator",
    "US_Criticizing_China",
]

EVENT_SOURCE_FEATURES: List[str] = ["China", "Sino_US", "US"]

EVENT_SENTIMENT_FEATURES: List[str] = [
    "Positive_Events",
    "Negative_Events",
    "Positive_Events_from_China",
    "Negative_Events_from_China",
    "Positive_Events_from_the_US",
    "Negative_Events_from_the_US",
]

EVENT_FEATURES: List[str] = (
    EVENT_TYPE_FEATURES + EVENT_SOURCE_FEATURES + EVENT_SENTIMENT_FEATURES
)

# 论文没用作特征、但表里有的列，保留下来方便自己看
EXTRA_COLUMNS: List[str] = ["Sino_US_Trade_Balance"]

# 原表列名 -> 本模块的规范列名。改名的目的是去空格、修拼写、统一大小写，
# 让列名可以直接当 Python 变量用；一一对应关系没有做任何顺手牵羊的改动。
COLUMN_ALIASES: Dict[str, str] = {
    # 改两类：① 真有毛病的（多余空格、拼写错误、大小写不一致、含连字符）；
    # ② 宏观列改用论文 Table 4 的规范名，这样 DA 的打分结果能直接和论文
    #    Table 6/8 对照（那两张表用的就是 `US_1Y_Treasury_Yield` 这类名字）。
    "China_ FX_Reserves": "China_FX_Reserves",
    "CN_Treasury_Yield_1Y": "CN_1Y_GovBond_Yield",
    "US_Treasury_Yield_1Y": "US_1Y_Treasury_Yield",
    "CN_CPI": "CN_CPI_MoM",
    "US_CPI": "US_CPI_MoM",
    "Sino-US trade balance": "Sino_US_Trade_Balance",
    "Sino-US": "Sino_US",
    # 事件列：`Purchasses` / `Enteprises` 是拼写错误，逐列改写为规范名
    "China Increasing Purchasses of US Goods": "China_Increasing_Purchases_of_US_Goods",
    "China Restrictions on US Enteprises": "China_Restrictions_on_US_Enterprises",
    "China Reducing Restrictions on US Enterprises":
        "China_Reducing_Restrictions_on_US_Enterprises",
    "China Tariff Increases on the US": "China_Tariff_Increases_on_the_US",
    "China Litigation or Investigation against the US":
        "China_Litigation_or_Investigation_against_the_US",
    "China Tariff Relief Policies for the US": "China_Tariff_Relief_Policies_for_the_US",
    "China Criticizing the US": "China_Criticizing_the_US",
    "Sino-US Dialogue": "Sino_US_Dialogue",
    "US Restrictions on Chinese Enterprises": "US_Restrictions_on_Chinese_Enterprises",
    "US Reducing Restrictions on Chinese Enterprises":
        "US_Reducing_Restrictions_on_Chinese_Enterprises",
    "US Tariff increases on China": "US_Tariff_Increases_on_China",
    "US Litigation or Investigation against China":
        "US_Litigation_or_Investigation_against_China",
    "US Tariff Relief Policies for China": "US_Tariff_Relief_Policies_for_China",
    "US Removing China from Currency Manipulator List":
        "US_Removing_China_from_Currency_Manipulator_List",
    "US Designating China as Currency Manipulator":
        "US_Designating_China_as_Currency_Manipulator",
    "US Criticizing China": "US_Criticizing_China",
    "Positive Events": "Positive_Events",
    "Negative Events": "Negative_Events",
    "Positive Events from China": "Positive_Events_from_China",
    "Negative Events from China": "Negative_Events_from_China",
    "Positive Events from the US": "Positive_Events_from_the_US",
    "Negative Events from the US": "Negative_Events_from_the_US",
}

# 中文事件类型 -> 规范英文列名。Sheet1 的 `事件类型` 与 Data 的 16 个哑变量一一对应，
# 这张表是两边对得上的唯一依据，改动前请先核对 Sheet1 的取值。
EVENT_TYPE_TRANSLATION: Dict[str, str] = {
    "中国加大对美国采购": "China_Increasing_Purchases_of_US_Goods",
    "中国对美国企业的限制": "China_Restrictions_on_US_Enterprises",
    "中国对美国企业降低限制": "China_Reducing_Restrictions_on_US_Enterprises",
    "中国对美国加征关税": "China_Tariff_Increases_on_the_US",
    "中国对美国发起诉讼或调查": "China_Litigation_or_Investigation_against_the_US",
    "中国对美国放宽关税政策": "China_Tariff_Relief_Policies_for_the_US",
    "中国指责美国": "China_Criticizing_the_US",
    "中美对话": "Sino_US_Dialogue",
    "美国对中国企业的限制": "US_Restrictions_on_Chinese_Enterprises",
    "美国对中国企业降低限制": "US_Reducing_Restrictions_on_Chinese_Enterprises",
    "美国对中国加征关税": "US_Tariff_Increases_on_China",
    "美国对中国发起诉讼或调查": "US_Litigation_or_Investigation_against_China",
    "美国对中国放宽关税政策": "US_Tariff_Relief_Policies_for_China",
    "美国将中国从汇率操纵名单剔除": "US_Removing_China_from_Currency_Manipulator_List",
    "美国将中国列为汇率操纵国": "US_Designating_China_as_Currency_Manipulator",
    "美国指责中国": "US_Criticizing_China",
}

# 来源 -> 规范列名
SOURCE_TRANSLATION: Dict[str, str] = {"中": "China", "中美": "Sino_US", "美": "US"}

# 情感 -> 规范列名
SENTIMENT_TRANSLATION: Dict[str, str] = {"正面": "Positive_Events", "负面": "Negative_Events"}


def excel_serial_to_datetime(series: pd.Series) -> pd.Series:
    """把 Excel 日期序列号还原成 pandas 时间戳。"""
    return pd.to_datetime(series, unit="D", origin=_EXCEL_EPOCH)


def _clean_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    renames = {old: COLUMN_ALIASES.get(old, old) for old in frame.columns}
    return frame.rename(columns=renames)


def _looks_low_frequency(series: pd.Series) -> bool:
    """判断一列是不是「低频数据被填成 0」了。"""
    values = series.dropna()
    if values.empty:
        return False
    return float((values != 0).mean()) < _LOW_FREQUENCY_NONZERO_FRACTION


def _repair_low_frequency(series: pd.Series, method: str) -> Tuple[pd.Series, int]:
    """先把字面的 0 还原成缺失，再按 method 补齐。返回 (新序列, 被补的个数)。"""
    if not _looks_low_frequency(series):
        return series, 0

    repaired = series.mask(series == 0.0)
    missing = int(repaired.isna().sum())
    if method == "ffill":
        # 外汇储备是存量指标，某天没公布就沿用上一次的值，比插值更贴近现实
        repaired = repaired.ffill().bfill()
    elif method == "interp":
        # CPI 是月度变化率，两个公布值之间线性过渡比较合理
        repaired = repaired.interpolate(method="linear", limit_direction="both")
    else:
        raise ValueError(f"不认识的补值方式：{method}")
    return repaired, missing


def load_daily_frame(
    excel_path: Optional[Path] = None,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """读 `Data` sheet，返回以日期为索引、列名已规范的日频表。

    行数、日期范围、目标列取值都有断言兜底——将来换数据源时如果对不上会立刻报错，
    而不是悄悄跑出一个看起来正常、其实错位的结果。
    """
    path = Path(excel_path) if excel_path else DEFAULT_EXCEL_PATH
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")

    raw = pd.read_excel(path, sheet_name=_DATA_SHEET)
    frame = _clean_column_names(raw)

    frame["Date"] = excel_serial_to_datetime(frame["Date"])
    frame = frame.sort_values("Date").set_index("Date")

    # 按论文 §4.4 的补值方案：外汇储备前向填充，中美 CPI 线性插值
    imputed: Dict[str, int] = {}
    for column, method in (("China_FX_Reserves", "ffill"), ("CN_CPI_MoM", "interp"),
                           ("US_CPI_MoM", "interp")):
        frame[column], filled = _repair_low_frequency(frame[column], method)
        if filled:
            imputed[column] = filled

    if verbose:
        print(f"读入 {path.name} / {_DATA_SHEET}：{frame.shape[0]} 行 × "
              f"{frame.shape[1]} 列，{frame.index[0].date()} → {frame.index[-1].date()}")
        for column, filled in imputed.items():
            print(f"  {column}：{filled} 个空缺已补（原表用字面 0 占位）")

    _assert_expected_shape(frame)

    # 事件类列必须是 0/1，出现别的值说明列名映射错了
    offending = [
        column for column in EVENT_FEATURES
        if not set(frame[column].dropna().unique()) <= {0, 1}
    ]
    if offending:
        raise ValueError(f"这些事件列不是 0/1 哑变量，列名映射可能有误：{offending}")

    return frame


def _assert_expected_shape(frame: pd.DataFrame) -> None:
    expected_rows = 1944
    if len(frame) != expected_rows:
        raise ValueError(f"预期 {expected_rows} 个交易日，实际 {len(frame)} 行。")

    first, last = frame.index[0].date().isoformat(), frame.index[-1].date().isoformat()
    if (first, last) != ("2017-01-03", "2024-12-31"):
        raise ValueError(f"预期区间 2017-01-03 → 2024-12-31，实际 {first} → {last}。")

    required = [TARGET] + TIME_SERIES_FEATURES + EVENT_FEATURES
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"缺少必要的列：{missing}")

    blanks = {column: int(frame[column].isna().sum())
              for column in required if frame[column].isna().any()}
    if blanks:
        raise ValueError(f"这些列补值后仍有空缺：{blanks}")


def _split_slash(value: Any) -> List[str]:
    """把 `负面/负面` 这类多标签拆成列表。"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [part.strip() for part in str(value).split("/") if part.strip()]


def load_news_corpus(excel_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """读 `Sheet1`，返回新闻语料列表。

    每个元素形如::

        {"date": Timestamp, "content": str, "event_types": [英文列名, ...],
         "sources": [英文列名, ...], "sentiments": [英文列名, ...]}

    关于 `Sheet1` 和 `Data` 能不能对上的问题，实测结论是**能对上，而且相当准**：
    把新闻日期映射到「之后最近的交易日」再按天折叠，25 个事件列里 7 列与
    `Data` sheet 完全一致，平均交并比 0.948，多数列的差异在 1~4 天以内
    （`Negative_Events` 248 vs 248、`US` 254 vs 254、`Sino_US` 115 vs 115）。
    复现过程见 `src/core/event_aggregation.py`，跑 `scripts/verify_pa2_aggregation.py`
    可以自己看对照表。

    剩下的差异来自 `Sheet1` 是**子集**：`Data` 那一侧的语料比这 445 条更多，
    所以是 `Data` 有、`Sheet1` 没有的情况为主，反向的很少。
    """
    path = Path(excel_path) if excel_path else DEFAULT_EXCEL_PATH
    raw = pd.read_excel(path, sheet_name=_NEWS_SHEET)

    corpus: List[Dict[str, Any]] = []
    for record in raw.to_dict("records"):
        corpus.append({
            "date": pd.Timestamp(record["日期"]),
            "content": str(record["文章内容"]).strip(),
            "event_types": [
                EVENT_TYPE_TRANSLATION[name]
                for name in _split_slash(record["事件类型"])
                if name in EVENT_TYPE_TRANSLATION
            ],
            "sources": [
                SOURCE_TRANSLATION[name]
                for name in _split_slash(record["来源"])
                if name in SOURCE_TRANSLATION
            ],
            "sentiments": [
                SENTIMENT_TRANSLATION[name]
                for name in _split_slash(record["事件情感"])
                if name in SENTIMENT_TRANSLATION
            ],
        })
    return corpus


_TIME_SERIES_DETAILS: Dict[str, str] = {
    "USD_Index": "美元指数：美元相对一篮子货币的强弱，走强通常对应 USD/CNY 上行",
    "CN_1Y_GovBond_Yield": "中国 1 年期国债收益率：中美利差的一端，中国利率上行通常吸引资本流入、人民币走强",
    "US_1Y_Treasury_Yield": "美国 1 年期国债收益率：利差的另一端，美国利率上行通常推升美元",
    "China_FX_Reserves": "中国外汇储备：央行干预能力和资本流动压力的代理变量，规模下降常伴随贬值压力。月度公布值，日频表中前向填充",
    "CN_CPI_MoM": "中国 CPI：论文 Table 4 标注为环比，但本表该列均值 1.54、区间 -0.8~4.5，是同比量级而非环比量级。命名沿用论文，口径存疑",
    "US_CPI_MoM": "美国 CPI：论文同样标注为环比，本表均值 3.22、最高 6.6，也是同比量级。通胀走高强化加息预期，对美元偏正面",
    "WTI_Futures_Price": "WTI 原油期货价格：大宗商品与通胀预期的代理，也反映全球需求",
}

_EVENT_TYPE_DETAILS: Dict[str, str] = {
    "China_Increasing_Purchases_of_US_Goods": "中国加大对美采购：缓和贸易摩擦，通常利多人民币",
    "China_Restrictions_on_US_Enterprises": "中国限制美国企业：中方反制，通常利空人民币",
    "China_Reducing_Restrictions_on_US_Enterprises": "中国放宽对美国企业的限制：摩擦缓和信号",
    "China_Tariff_Increases_on_the_US": "中国对美国加征关税：中方反制，通常利空人民币",
    "China_Litigation_or_Investigation_against_the_US": "中国对美国发起诉讼或调查：摩擦升级",
    "China_Tariff_Relief_Policies_for_the_US": "中国对美国放宽关税：谈判缓和信号",
    "China_Criticizing_the_US": "中国指责美国：口头交锋，对汇率的实质影响通常有限",
    "Sino_US_Dialogue": "中美对话：沟通渠道通畅，通常降低不确定性",
    "US_Restrictions_on_Chinese_Enterprises": "美国限制中国企业：美方施压，通常利多美元、利空人民币",
    "US_Reducing_Restrictions_on_Chinese_Enterprises": "美国放宽对中国企业的限制：缓和信号",
    "US_Tariff_Increases_on_China": "美国对中国加征关税：最典型的摩擦升级事件，通常推动人民币贬值",
    "US_Litigation_or_Investigation_against_China": "美国对中国发起诉讼或调查：摩擦升级",
    "US_Tariff_Relief_Policies_for_China": "美国对中国放宽关税：缓和信号",
    "US_Removing_China_from_Currency_Manipulator_List": "美国将中国从汇率操纵名单剔除：样本内只发生过 1 次，缓和信号",
    "US_Designating_China_as_Currency_Manipulator": "美国将中国列为汇率操纵国：样本内只发生过 1 次，摩擦升级到顶点",
    "US_Criticizing_China": "美国指责中国：口头交锋",
}

_EVENT_SOURCE_DETAILS: Dict[str, str] = {
    "China": "当天是否有中方发起的事件",
    "Sino_US": "当天是否有中美共同参与的事件（通常是对话）",
    "US": "当天是否有美方发起的事件",
}

_EVENT_SENTIMENT_DETAILS: Dict[str, str] = {
    "Positive_Events": "当天是否有正面事件",
    "Negative_Events": "当天是否有负面事件，论文中重要性排名第一的特征",
    "Positive_Events_from_China": "当天是否有中方发起的正面事件",
    "Negative_Events_from_China": "当天是否有中方发起的负面事件",
    "Positive_Events_from_the_US": "当天是否有美方发起的正面事件",
    "Negative_Events_from_the_US": "当天是否有美方发起的负面事件",
}


def candidate_features() -> Dict[str, str]:
    """DA（决策智能体）的候选特征清单：特征名 -> 中文说明。

    这是论文实验的完整候选集，共 32 个（7 个宏观 + 25 个事件）。
    """
    catalog = dict(_TIME_SERIES_DETAILS)
    catalog.update(_EVENT_TYPE_DETAILS)
    catalog.update(_EVENT_SOURCE_DETAILS)
    catalog.update(_EVENT_SENTIMENT_DETAILS)

    missing = [name for name in TIME_SERIES_FEATURES + EVENT_FEATURES
               if name not in catalog]
    if missing:
        raise ValueError(f"候选特征清单缺少说明文字：{missing}")
    return catalog


def paper_selected_features() -> List[str]:
    """论文 Table 8 里 DA 最终选中的 4 个特征，用来做对照基准（归一化后得分均 ≥ 80）。"""
    return ["Negative_Events", "China_FX_Reserves", "WTI_Futures_Price",
            "US_1Y_Treasury_Yield"]


def paper_rejected_features() -> List[str]:
    """论文 Table 6 里明确给出得分、但没被选中的特征（得分 8，垫底）。"""
    return ["US_Removing_China_from_Currency_Manipulator_List"]


def ablation_feature_set(name: str) -> List[str]:
    """论文 Table 7 消融实验的三套特征配置。

    - `"pa1"`        —— 只有 7 个宏观金融变量
    - `"pa1_pa2"`    —— 再加上 25 个事件哑变量
    - `"pa1_pa2_da"` —— 再加上 DA 在 Table 8 里最终选中的那几个

    论文 Table 7 的核心结论就是这三行之间的差距（Transformer 的 RMSE
    从 0.0936 → 0.0737 → 0.0566），提升全部来自 PA2 和 DA。
    """
    if name == "pa1":
        return list(TIME_SERIES_FEATURES)
    if name == "pa1_pa2":
        return list(TIME_SERIES_FEATURES) + list(EVENT_FEATURES)
    if name == "pa1_pa2_da":
        # 去重但保持顺序：DA 选的 4 个里有两个本来就在宏观变量里
        return list(dict.fromkeys(list(TIME_SERIES_FEATURES) + paper_selected_features()))
    raise ValueError(
        f"不认识的特征集名称：{name}（可选 pa1 / pa1_pa2 / pa1_pa2_da）"
    )


def traditional_feature_engineering() -> List[str]:
    """论文 Table 8 右列「traditional feature engineering」选出的 4 个特征。

    它等于「§4.4 的相关性筛选结果 + XGBoost 重要度均值过滤的结果」：
    时序侧是 `{USD_Index, CN_1Y_GovBond_Yield}`，事件侧是
    `{Positive_Events, Negative_Events}`。Table 9 拿它和 DA 的选择做对照。

    **这个函数返回的是论文报的答案，不是算出来的结果**——写死在这里是为了
    方便其它地方引用。真要跑这条基线、要看算出来的和论文差多少，用
    `src/core/traditional_baseline.py`（跑 `scripts/verify_traditional_baseline.py`），
    实测这四个特征全部能算出来。
    """
    return ["USD_Index", "CN_1Y_GovBond_Yield", "Positive_Events", "Negative_Events"]


def drop_correlated_features(
    frame: pd.DataFrame,
    columns: List[str],
    target: str = TARGET,
    *,
    threshold: float = 0.7,
    min_abs_correlation: float = 0.3,
) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    """论文 §4.4 的时间序列特征筛选：先扔掉与目标几乎不相关的，再对剩下的做
    两两去重（相关系数超过 threshold 时只留下与目标相关性更高的那个）。

    返回 (留下的特征, [(被丢的, 留下的, 相关系数), ...])。

    **这一套筛的是「传统特征工程」基线，不是 DA 的输入。** 论文 §4.4 只给出
    这一步的结果 `{USD_Index, CN_1Y_GovBond_Yield}`，而它恰好就是 Table 8
    右列「traditional feature engineering」里的两个时序特征。DA 那边是另一条
    路径——它把包括 `China_FX_Reserves`、`WTI_Futures_Price`、
    `US_1Y_Treasury_Yield` 在内的全部候选特征都评了一遍（见 Table 6），
    所以 DA 的候选集不能提前用这个函数砍。
    """
    if not columns:
        return [], []

    correlations = frame[columns + [target]].corr(method="pearson").abs()
    target_corr = correlations[target]

    # 第一关：与目标相关性太弱的直接出局。少了这一步，结果会与论文对不上——
    # 本表里 China_FX_Reserves / CN_CPI_MoM / US_CPI_MoM / WTI 与 USD/CNY 的
    # 相关性都在 0.11 以内，光靠 0.7 去重是筛不掉它们的。
    candidates = [name for name in columns if target_corr[name] >= min_abs_correlation]

    kept: List[str] = []
    dropped: List[Tuple[str, str, float]] = []
    for column in sorted(candidates, key=lambda name: target_corr[name], reverse=True):
        clash = next(
            (other for other in kept if correlations.loc[column, other] > threshold),
            None,
        )
        if clash is None:
            kept.append(column)
        else:
            dropped.append((column, clash, float(correlations.loc[column, clash])))
    return kept, dropped


def standardize(
    frame: pd.DataFrame,
    columns: List[str],
    *,
    fit_index: Optional[pd.Index] = None,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """标准化指定的列。

    `fit_index` 用来指定「只在这些行上算均值和标准差」。默认在全表上算，
    但做训练/测试划分时**应当传入训练段的索引**——否则测试集的分布会泄漏进
    标准化参数，指标会虚高。这是时间序列建模里最容易踩、也最难发现的坑。
    """
    scaler = StandardScaler()
    fit_rows = frame.loc[frame.index.intersection(fit_index)] if fit_index is not None else frame
    scaler.fit(fit_rows[columns])

    out = frame.copy()
    out[columns] = scaler.transform(frame[columns])
    return out, scaler
