"""决策智能体（DA）。

这个智能体的职责是**从候选特征里挑出值得送进预测模型的那些**，方法是
「检索证据 → 自评打分 → 迭代精炼」三步循环，实现在 `core/da_engine.py`。

本文件刻意写得很薄：选特征的逻辑全在 `EvidenceBasedSelector` 里，这里只负责
准备候选清单、调用它、把结果写进 state，最后让 LLM 把评审理由组织成一段
给用户看的市场评述。
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.abstractions.base_agent import BaseAgent
from ..core.abstractions.base_llm import BaseLLM
from ..core.abstractions.base_rag import BaseRAG
from ..core.da_engine import (
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_THRESHOLD,
    DEFAULT_WEIGHTS,
    EvidenceBasedSelector,
    FeatureSelectionReport,
)
from ..core.feature_engineering import default_features, feature_catalog
from ..core.research_dataset import candidate_features
from ..core.workflow_state import AppState

# 送进预测模型的特征上限。选太多会过拟合，也会拖慢训练。
_MAX_FEATURES = 12

# 候选特征数量上限。0 表示不限制（跑完整候选集）。
# 每个特征最多要调 3 次 LLM，候选越多越慢，这是 DA 唯一的耗时开关。
_DEFAULT_LIMIT = 0


class MarketCommentary(BaseModel):
    """把 DA 的评审结论组织成人话。"""

    market_commentary: str = Field(
        description=(
            "一段中文市场评述。要说明：哪些特征被选中、依据是什么、"
            "哪些被淘汰、为什么。只能引用给定的评审结论，不要补充新的事实。"
        )
    )


_COMMENTARY_PROMPT = """你是一个汇率预测项目的分析师。

决策智能体刚刚对候选特征做了基于证据的评审，结论如下：

候选集来源：{catalog_name}
归一化方式与阈值：{note}
最终选中（{selected_count} 个）：{selected}
被淘汰（{rejected_count} 个）：{rejected}

得分为正的评审明细（按分数从高到低）：
{details}

请写一段**中文**市场评述，向用户解释这次特征筛选：
1. 哪些特征入选，各自依据了哪些证据（可以点出证据出处）；
2. 哪些特征被淘汰，主要原因是什么；
3. 如果出现了"知识库里检索不到证据"的情况，如实说明这会让该特征的结论不牢靠。

**只依据上面的评审结论来写，不要引入任何新的事实、数字或文献。**"""


class DecisionAgent(BaseAgent):
    def __init__(
        self,
        rag_service: BaseRAG,
        llm_service: BaseLLM,
        *,
        selector: Optional[EvidenceBasedSelector] = None,
        catalog: Optional[Dict[str, str]] = None,
        max_features: int = _MAX_FEATURES,
        limit: int = _DEFAULT_LIMIT,
    ):
        self._rag_service = rag_service
        self._llm_service = llm_service
        self._selector = selector or EvidenceBasedSelector(
            rag_service=rag_service,
            llm_service=llm_service,
            top_k=3,
            max_attempts=DEFAULT_MAX_ATTEMPTS,
            weights=DEFAULT_WEIGHTS,
            threshold=DEFAULT_THRESHOLD,
            normalization="theoretical",
        )
        self._catalog = catalog
        self._max_features = max_features
        self._limit = limit

    def run(self, state: AppState) -> AppState:
        print("--- Decision Agent Running ---")

        catalog, catalog_name = self._resolve_catalog(state)
        if not catalog:
            raise ValueError("候选特征清单为空，无法进行特征筛选。")

        report = self._selector.select(catalog, limit=self._limit, progress=True)
        self._print_report(report)

        selected = report.selected_features[: self._max_features]
        if not selected:
            # 一个都没达标时不能让流程断掉：知识库可能是空的，或者阈值太严。
            # 退回论文 Table 8 那套 / 默认技术指标，同时把情况说清楚。
            selected = self._fallback_features(state)
            print(f"没有任何特征达到阈值 {report.normalized_threshold}，"
                  f"退回备用特征集：{selected}")

        state.features_for_forecasting = selected
        state.feature_selection_report = report.model_dump()
        state.market_commentary = self._write_commentary(catalog_name, report, selected)
        state.next_step = "forecasting_agent"

        print(f"市场评述：{state.market_commentary}")
        return state

    # ---------- 内部实现 ----------

    def _resolve_catalog(self, state: AppState):
        """决定候选特征集用哪一套。

        优先级：调用方显式传入 > 感知数据自己声明的 > 按 symbol 推断。

        感知数据带 `catalog: "research"` 时用论文那一套特征
        （7 个宏观 + 25 个事件），这时知识库里的中美贸易机制说明才检索得上。
        否则退回到本地技术指标——但技术指标（均线、RSI）是检索不出什么
        有用证据的，那种组合下 DA 的评审基本等于走个过场。
        """
        if self._catalog:
            return self._catalog, "调用方指定"

        structured = (state.perception_data or {}).get("structured") or {}
        if structured.get("catalog") == "research":
            return (
                candidate_features(),
                "论文候选集（USD/CNY，7 个宏观 + 25 个事件）",
            )

        symbol = structured.get("symbol")
        if symbol:
            return feature_catalog(symbol), f"本地技术指标（{symbol}）"

        return candidate_features(), "论文候选集（USD/CNY，7 个宏观 + 25 个事件）"

    def _fallback_features(self, state: AppState) -> List[str]:
        from ..core.research_dataset import paper_selected_features

        structured = (state.perception_data or {}).get("structured") or {}
        if structured.get("catalog") != "research" and structured.get("symbol"):
            return default_features(structured["symbol"])

        # 论文那条线上，用 Table 8 的结果兜底
        return paper_selected_features()

    @staticmethod
    def _print_report(report: FeatureSelectionReport) -> None:
        print(f"\n特征评审结果（{report.note}）：")
        print(f"{'特征':<50s} {'相关':>5s} {'支持':>5s} {'实用':>5s} {'得分':>6s}  结论")
        for row in report.as_table():
            if row["relevance"] is None:
                print(f"{row['feature']:<50s} {'—':>5s} {'—':>5s} {'—':>5s} "
                      f"{row['score']:>6.1f}  淘汰（{row['note']}）")
                continue
            print(
                f"{row['feature']:<50s} {row['relevance']:>5.1f} "
                f"{row['supportiveness']:>5.1f} {row['utility']:>5.1f} "
                f"{row['score']:>6.1f}  {'采纳' if row['selected'] else '淘汰'}"
            )
        print(f"\n共采纳 {len(report.selected_features)} 个，"
              f"淘汰 {len(report.rejected_features)} 个。\n")

    def _write_commentary(
        self,
        catalog_name: str,
        report: FeatureSelectionReport,
        selected: List[str],
    ) -> str:
        details = []
        for item in sorted(
            report.evaluations, key=lambda entry: entry.normalized_score, reverse=True
        ):
            if item.critique is None:
                details.append(f"- {item.feature}：未检索到证据（{item.note}）")
                continue
            sources = sorted({doc.get("source", "?") for doc in item.evidence})
            details.append(
                f"- {item.feature}（{item.normalized_score:.1f} 分，"
                f"{item.attempts} 轮）\n"
                f"  相关性 {item.critique.relevance}：{item.critique.relevance_reason}\n"
                f"  支持度 {item.critique.supportiveness}：{item.critique.supportiveness_reason}\n"
                f"  实用性 {item.critique.utility}：{item.critique.utility_reason}\n"
                f"  证据出处：{', '.join(sources)}"
            )

        prompt = _COMMENTARY_PROMPT.format(
            catalog_name=catalog_name,
            note=report.note,
            selected_count=len(selected),
            selected="、".join(selected) or "（无）",
            rejected_count=len(report.rejected_features),
            rejected="、".join(report.rejected_features) or "（无）",
            details="\n".join(details),
        )
        result = self._llm_service.invoke_structured(
            prompt, schema=MarketCommentary, config={"model": "decision_llm"}
        )
        return result.market_commentary
