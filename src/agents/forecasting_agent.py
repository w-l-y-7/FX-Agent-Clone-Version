"""预测智能体（FA）。

职责有两块：

1. **把 DA 选出来的特征送进预测模型**，拿到下一个交易日的汇率预测；
2. **把模型的技术结论翻译成人话**——调参选了什么模型、SHAP 认为哪些特征在
   推动预测、以及最要紧的一条：这个模型有没有跑赢「假设价格不变」的朴素基准。

第 2 块对应 README 里 FA 的第三条职责（"uses an LLM to translate these technical
insights into accessible, natural language reports"）。

传了 `llm_service` 才会做第 2 块；不传就只打印结构化结果。
"""

from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

from ..core.abstractions.base_agent import BaseAgent
from ..core.abstractions.base_forecasting import BaseForecasting
from ..core.abstractions.base_llm import BaseLLM
from ..core.feature_engineering import (
    build_feature_frame,
    default_features,
    feature_catalog,
)
from ..core.research_dataset import candidate_features
from ..core.workflow_state import AppState

# 用户问的是"下周"，一周按 5 个交易日算
_HORIZON_DAYS = 5

# 给 LLM 看多少个特征就不往下带了。SHAP 会给出全部特征，
# 但一份人读的报告说前几名就够了。
_TOP_FEATURES_IN_REPORT = 5


class ForecastInterpretation(BaseModel):
    """把模型的技术结论翻译成一段人话。"""

    interpretation: str = Field(
        description=(
            "一段中文解读，3~5 句话。要说明：调参选中的是哪个模型、"
            "哪些特征对本次预测影响最大、方向是什么，以及模型有没有跑赢"
            "「假设价格不变」的朴素基准。只能依据给定的数字，不要补充新的事实。"
        )
    )


_INTERPRETATION_PROMPT = """你是一个汇率预测项目的分析师。
预测智能体刚跑完，技术结论如下：

【模型】{model}
【预测对象】{symbol}，{horizon} 个交易日后的收盘价
【预测结果】最新收盘 {last_close}（{origin_date}）→ 预测 {predicted_close}
（{change_pct:+.4f}%）

【调参】{optimization}
【回测】MAE {mae}，RMSE {rmse}；朴素基准（假设价格不变）MAE {naive_mae}
【是否跑赢朴素基准】{beats}

【特征贡献（SHAP，按影响从大到小）】
{explanation}

请写一段**中文**解读，3~5 句话：
1. 这次用的是哪个模型；如果上面写了跑过 Optuna，就说清最终选中了哪个模型；
2. 哪些特征对这次预测影响最大、方向是推高还是压低；如果没算出 SHAP，如实说明；
3. 明确说明模型有没有跑赢朴素基准。如果没有跑赢，要如实指出这次预测
   不具备实际指导意义，不要用委婉措辞掩盖。

**只依据上面的数字来写，不要引入任何新的事实或数字。**"""


class ForecastingAgent(BaseAgent):
    def __init__(
        self,
        forecasting_service: BaseForecasting,
        llm_service: Optional[BaseLLM] = None,
        horizon: int = _HORIZON_DAYS,
    ):
        self._forecasting_service = forecasting_service
        self._llm_service = llm_service
        self._horizon = horizon

    def run(self, state: AppState) -> AppState:
        print("--- Forecasting Agent Running ---")

        structured = (state.perception_data or {}).get("structured")
        if not structured:
            raise ValueError("感知数据缺失，无法进行预测。")

        symbol = structured["symbol"]
        requested = state.features_for_forecasting or []

        # 两条线：论文那条（数据源自带特征表）和技术指标那条。
        # 特征清单跟着走不同的目录，所以这里要先判断走的是哪条。
        research = structured.get("catalog") == "research"
        catalog = candidate_features() if research else feature_catalog(symbol)
        horizon = int(structured.get("horizon", self._horizon))

        # 决策智能体是从清单里挑的，但模型输出不可全信，这里按实际情况过滤一遍
        features = [name for name in requested if name in catalog]
        dropped = [name for name in requested if name not in catalog]
        if dropped:
            print(f"本地数据算不出来的特征已丢弃：{dropped}")
        if not features:
            features = _fallback_features(research, symbol)
            print(f"选中的特征全都无法计算，退回备用特征集：{features}")

        frame = (
            _frame_from_records(structured["frame"])
            if research
            else build_feature_frame(structured["series"], symbol, horizon=horizon)
        )

        result = self._forecasting_service.predict(
            frame, {"features": features, "horizon": horizon, "symbol": symbol}
        )
        result["features_used"] = features
        result["unavailable_features"] = dropped

        _print_forecast(result)

        if self._llm_service is not None:
            result["interpretation"] = self._write_interpretation(result)
            print(f"\n【模型解读】{result['interpretation']}")

        state.forecast_result = result
        state.next_step = "END"
        return state

    def _write_interpretation(self, result: dict) -> str:
        optimization = result.get("optimization")
        if optimization:
            optimization_text = (
                f"{optimization['n_trials']} 组超参里选中最优的 "
                f"{optimization['best_params'].get('family')}"
                f"（验证段 RMSE {optimization['validation_rmse']}）"
            )
        else:
            optimization_text = "本服务用的是固定超参，没有跑 Optuna 搜索"

        explanation = result.get("explanation") or []
        if explanation:
            explanation_text = "\n".join(
                f"- {item['feature']}：平均绝对贡献 {item['mean_abs_shap']}，"
                f"{item.get('direction', '')}"
                for item in explanation[:_TOP_FEATURES_IN_REPORT]
            )
        else:
            explanation_text = "（这次没有算出 SHAP 贡献）"

        evaluation = result["evaluation"]
        prompt = _INTERPRETATION_PROMPT.format(
            model=result["model"],
            symbol=result["symbol"],
            horizon=result["horizon_trading_days"],
            last_close=result["last_close"],
            origin_date=result["origin_date"],
            predicted_close=result["predicted_close"],
            change_pct=result["predicted_change_pct"],
            optimization=optimization_text,
            mae=evaluation["mae"],
            rmse=evaluation["rmse"],
            naive_mae=evaluation["naive_mae"],
            beats="是" if result["beats_naive_baseline"] else "**没有**，模型还不如朴素基准",
            explanation=explanation_text,
        )
        parsed = self._llm_service.invoke_structured(
            prompt, schema=ForecastInterpretation, config={"model": "forecasting_llm"}
        )
        return parsed.interpretation


def _fallback_features(research: bool, symbol: str) -> list:
    if research:
        from ..core.research_dataset import paper_selected_features

        return paper_selected_features()
    return default_features(symbol)


def _frame_from_records(records: list) -> pd.DataFrame:
    """把数据工具给的记录列表还原成特征矩阵。"""
    frame = pd.DataFrame(records)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values("date").reset_index(drop=True)


def _print_forecast(result: dict) -> None:
    evaluation = result["evaluation"]
    print(f"{result['symbol']} 最新收盘 {result['last_close']}（{result['origin_date']}）")
    print(
        f"预测 {result['horizon_trading_days']} 个交易日后"
        f"（约 {result['forecast_date_estimate']}）：{result['predicted_close']} "
        f"（{result['predicted_change_pct']:+.4f}%）"
    )
    print(
        f"回测：MAE {evaluation['mae']} / RMSE {evaluation['rmse']} "
        f"vs 朴素基准 MAE {evaluation['naive_mae']}"
        f"（训练 {evaluation['train_rows']} 行 / 测试 {evaluation['test_rows']} 行）"
    )

    # 调参详情、SHAP 贡献明细和 LLM 解读都放在 main.py 结尾的总结里，
    # 这里只报一行状态，免得同一份内容在终端出现两次。
    optimization = result.get("optimization")
    if optimization:
        print(
            f"调参：{optimization['n_trials']} 组超参，选中 "
            f"{optimization['best_params'].get('family')}"
            f"（耗时 {optimization['seconds']} 秒）"
        )

    if result["beats_naive_baseline"]:
        print(f"模型优于朴素基准，置信度 {result['confidence']}")
    else:
        print(
            f"警告：模型没有跑赢「假设价格不变」的朴素基准，"
            f"该预测只能当作参考，不具备实际指导意义（置信度 {result['confidence']}）"
        )
