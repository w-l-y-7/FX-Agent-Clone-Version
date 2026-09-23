from typing import Any, Dict, Optional

from ..core.abstractions.base_agent import BaseAgent
from ..core.workflow_state import AppState
from ..services.tool_service import ToolService

# 预测模型要算均线、波动率这类需要回看窗口的特征，还要留出足够的训练样本，
# 所以历史区间得按年拉，不能只取最近两个月。
_HISTORY_DAYS = 1095


class PerceptionAgent(BaseAgent):
    """感知智能体：负责把数据取回来。

    用哪个工具是**注入进来**的，不写死。因为项目里有两条数据线：

    - 论文那条线用 `research_data_fetcher`，取 `Data.xlsx` 里的 USD/CNY 及
      其宏观、事件特征，不需要联网，也不需要抓新闻；
    - 演示那条线用 `structured_data_fetcher`（AKShare 实时行情）配
      `unstructured_data_scraper`（Firecrawl 抓新闻）。

    早先这里把工具名硬编码在 `run()` 里，换成论文那条线就会因为找不到工具而报错。
    """

    def __init__(
        self,
        tool_service: ToolService,
        history_days: int = _HISTORY_DAYS,
        *,
        structured_tool: str = "structured_data_fetcher",
        unstructured_tool: Optional[str] = "unstructured_data_scraper",
        structured_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._tool_service = tool_service
        self._history_days = history_days
        self._structured_tool = structured_tool
        self._unstructured_tool = unstructured_tool
        self._structured_kwargs = structured_kwargs

    def run(self, state: AppState) -> AppState:
        print("--- Perception Agent Running ---")

        plan = state.plan
        if not plan or "plan" not in plan:
            raise ValueError("Plan is missing or invalid.")

        structured_data = self._tool_service.execute_tool(
            self._structured_tool,
            **(self._structured_kwargs or {"days": self._history_days}),
        )

        if self._unstructured_tool:
            unstructured_data = self._tool_service.execute_tool(
                self._unstructured_tool, query=state.user_request
            )
        else:
            # 论文那条线不需要抓新闻：事件特征已经在数据集里了，
            # 而 DA 的证据来自知识库检索，不是来自当天的新闻。
            unstructured_data = {
                "articles": [],
                "note": "本次未启用新闻抓取（事件特征已包含在数据集中）",
            }

        state.perception_data = {
            "structured": structured_data,
            "unstructured": unstructured_data,
        }
        state.next_step = "decision_agent"

        print(
            f"Structured: {structured_data['observations']} rows of "
            f"{structured_data['symbol']} ({structured_data['start']} → "
            f"{structured_data['end']})"
        )
        print(f"Unstructured: {len(unstructured_data['articles'])} articles scraped")
        return state
