"""FX-Agents 的主入口。

跑法：

    .\\venv\\Scripts\\python.exe main.py

默认走「论文那条线」：用 `Data.xlsx` 里 2017-2024 年的 USD/CNY 及其宏观、
中美贸易事件特征，让决策智能体（DA）基于知识库证据挑特征，再用 Ridge 回归
预测下一个交易日的收盘价。

完整的运行记录（含 DA 每一轮检索到的证据出处）会写到 reports/last_run.json。
终端只打印一份人看得懂的摘要——整个 state 有近两千行特征数据，全打出来没法看。
"""

import argparse
import json
import os
import sys

# 必须在任何模型库之前导入：它负责把 HuggingFace 的下载地址指到国内镜像
import src.utils.hf_env  # noqa: F401  isort:skip

from pathlib import Path

from dotenv import load_dotenv

from src.core.da_engine import (
    DEFAULT_THRESHOLD,
    EvidenceBasedSelector,
)
from src.core.graph import create_workflow
from src.core.workflow_state import AppState

from src.services.deepseek_llm_service import DeepSeekLLMService
from src.services.optimized_forecasting_service import OptimizedForecastingService
from src.services.sklearn_forecasting_service import SklearnForecastingService
from src.services.tool_service import ToolService
from src.services.vector_rag_service import VectorRAGService

from src.agents.decision_agent import DecisionAgent
from src.agents.forecasting_agent import ForecastingAgent
from src.agents.perception_agent import PerceptionAgent
from src.agents.planning_agent import PlanningAgent

from src.tools.research_data_fetcher import ResearchDataFetcher
from src.tools.structured_data_fetcher import StructuredDataFetcher
from src.tools.unstructured_data_scraper import UnstructuredDataScraper

_PROJECT_ROOT = Path(__file__).resolve().parent
_STATE_DUMP = _PROJECT_ROOT / "reports" / "last_run.json"
_DA_CACHE = _PROJECT_ROOT / ".cache" / "da"

# 默认问题。走论文那条线时，问的是下一个交易日——和论文的实验协议一致。
_DEFAULT_REQUEST = "Forecast the USD/CNY exchange rate for the next trading day."


def build_workflow(
    *,
    use_research_data: bool = True,
    limit: int = 0,
    use_cache: bool = True,
    normalization: str = "theoretical",
    threshold: float = DEFAULT_THRESHOLD,
    optimize: bool = False,
    trials: int = 30,
):
    """装配所有服务、智能体和工具，返回可运行的 LangGraph 应用。"""
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise RuntimeError(
            "没找到 DEEPSEEK_API_KEY。请在项目根目录的 .env 文件里填上你的密钥。"
        )

    llm_service = DeepSeekLLMService(api_key=api_key)

    # 真实的知识库检索。知识库空的时候 retrieve() 会返回空列表而不是报错，
    # DA 会据此把所有特征判为"证据不足"——这是设计好的降级路径。
    rag_service = VectorRAGService(verbose=True)
    rag_service.build()

    # 两个预测服务共用 core/backtest.py 里的回测口径，MAE 可以直接比：
    # 默认是固定超参的 Ridge（秒级出结果），--optimize 换成 Optuna 调参 + SHAP 解释。
    forecasting_service = (
        OptimizedForecastingService(n_trials=trials)
        if optimize
        else SklearnForecastingService()
    )

    tool_service = ToolService()
    if use_research_data:
        tool_service.register_tool(ResearchDataFetcher())
    else:
        # 演示那条线：AKShare 取行情 + Firecrawl 抓新闻，预测技术指标。
        # 保留它是为了不破坏原作者的演示路径，但知识库里没有技术指标相关的
        # 证据，DA 的评审在这条线上基本是走过场。
        tool_service.register_tool(StructuredDataFetcher())
        tool_service.register_tool(UnstructuredDataScraper())

    # 缓存必须显式传进来。不传的话每次跑 main.py 都要重新调几百次 API，
    # 一轮二十分钟，改一行代码就要重付一次这个代价。
    selector = EvidenceBasedSelector(
        rag_service=rag_service,
        llm_service=llm_service,
        top_k=3,
        threshold=threshold,
        normalization=normalization,
        cache_dir=_DA_CACHE if use_cache else None,
    )

    perception_agent = (
        PerceptionAgent(
            tool_service=tool_service,
            structured_tool="research_data_fetcher",
            unstructured_tool=None,
        )
        if use_research_data
        else PerceptionAgent(tool_service=tool_service)
    )

    agents = {
        "planning_agent": PlanningAgent(llm_service=llm_service).run,
        "perception_agent": perception_agent.run,
        "decision_agent": DecisionAgent(
            rag_service=rag_service,
            llm_service=llm_service,
            selector=selector,
            limit=limit,
        ).run,
        "forecasting_agent": ForecastingAgent(
            forecasting_service=forecasting_service,
            # 让 FA 把调参结论和 SHAP 贡献翻译成一段中文解读（README 里
            # 「Ensures Interpretability」那条）。
            llm_service=llm_service,
        ).run,
    }
    return create_workflow(agent_map=agents, entry_point="planning_agent")


def _print_summary(state: dict) -> None:
    print("\n" + "=" * 70)
    print("运行结束")
    print("=" * 70)

    report = state.get("feature_selection_report") or {}
    selected = state.get("features_for_forecasting") or []
    print(f"\n【DA 选出的特征】共 {len(selected)} 个：{selected}")
    if report.get("note"):
        print(f"  评审口径：{report['note']}")

    rows = sorted(
        report.get("evaluations", []),
        key=lambda item: item.get("normalized_score", 0),
        reverse=True,
    )
    if rows:
        print(f"\n  得分前 5：")
        for item in rows[:5]:
            critique = item.get("critique") or {}
            print(
                f"    {item['normalized_score']:>5.1f}  {item['feature']}"
                f"（相关 {critique.get('relevance')} / 支持 {critique.get('supportiveness')}"
                f" / 实用 {critique.get('utility')}）"
            )
        no_evidence = [item for item in rows if not item.get("critique")]
        if no_evidence:
            print(f"  另有 {len(no_evidence)} 个特征因检索不到证据而淘汰。")

    print(f"\n【市场评述】\n{state.get('market_commentary')}")

    forecast = state.get("forecast_result") or {}
    if forecast:
        evaluation = forecast.get("evaluation", {})
        print(f"\n【预测结果】")
        print(f"  {forecast.get('symbol')} 最新收盘 {forecast.get('last_close')}"
              f"（{forecast.get('origin_date')}）")
        print(f"  预测 {forecast.get('forecast_date_estimate')}："
              f"{forecast.get('predicted_close')}"
              f"（{forecast.get('predicted_change_pct'):+.4f}%）")
        print(f"  回测 MAE {evaluation.get('mae')} / RMSE {evaluation.get('rmse')} "
              f"vs 朴素基准 MAE {evaluation.get('naive_mae')}，"
              f"置信度 {forecast.get('confidence')}")
        if not forecast.get("beats_naive_baseline"):
            print("  注意：模型没跑赢「假设价格不变」的朴素基准，这个预测仅供参考。")

        optimization = forecast.get("optimization")
        if optimization:
            print(f"\n【Optuna 调参】{optimization['n_trials']} 组超参，"
                  f"耗时 {optimization['seconds']} 秒，选中：")
            print(f"  {optimization['best_params']}")
            print(f"  各模型族试了多少次：{optimization['trials_per_family']}")

        explanation = forecast.get("explanation") or []
        if explanation:
            print(f"\n【特征贡献（SHAP，前 5）】")
            for item in explanation[:5]:
                print(f"  {item['mean_abs_shap']:.6f}  {item['feature']}"
                      f"（{item.get('direction', '')}）")

        if forecast.get("interpretation"):
            print(f"\n【模型解读】{forecast['interpretation']}")


def _dump_state(state: dict) -> None:
    """把完整 state 写盘。里面近两千行特征数据，只适合落文件不适合打屏。"""
    _STATE_DUMP.parent.mkdir(parents=True, exist_ok=True)
    try:
        _STATE_DUMP.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n完整运行记录已写入：{_STATE_DUMP}")
    except (TypeError, OSError) as exc:
        print(f"\n运行记录落盘失败（不影响结果）：{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FX-Agents 多智能体汇率预测流程",
        epilog=(
            "第一次跑建议加 --limit 8 先跑通（约 3 分钟）；"
            "确认没问题再去掉它跑完整的 32 个候选特征。"
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="只评审前 N 个候选特征，0 表示全部（32 个）。第一次跑建议先用 8。",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="不使用 .cache/da 里的缓存。默认会用，改代码重跑时能省掉全部 API 调用。",
    )
    parser.add_argument(
        "--normalization", choices=["theoretical", "batch"], default="theoretical",
        help="theoretical=除以理论满分（能复现论文 Table 6）；batch=论文公式(2)的字面写法。",
    )
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"特征采纳阈值（0-100），默认 {DEFAULT_THRESHOLD}，与论文一致。",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="改用 AKShare/Firecrawl 实时数据线（预测技术指标）。默认走论文数据线。",
    )
    parser.add_argument(
        "--optimize", action="store_true",
        help="预测改用「Optuna 调参 + SHAP 解释」那套服务（默认是固定超参的 Ridge）。",
    )
    parser.add_argument(
        "--trials", type=int, default=30,
        help="--optimize 时 Optuna 搜索的超参组数，默认 30。调大更慢但可能更好。",
    )
    args = parser.parse_args()

    app = build_workflow(
        use_research_data=not args.live,
        limit=args.limit,
        use_cache=not args.no_cache,
        normalization=args.normalization,
        threshold=args.threshold,
        optimize=args.optimize,
        trials=args.trials,
    )
    final_state = app.invoke(AppState(user_request=_DEFAULT_REQUEST))
    _print_summary(final_state)
    _dump_state(final_state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
