"""决策智能体（DA）的核心：基于证据的特征筛选。

论文里 DA 走三步，本模块按同样的三步实现：

1. **证据检索** —— 把候选特征转写成一个关于「它如何影响 USD/CNY」的自然语言
   问题，去知识库里做向量检索。
2. **自评打分** —— 让 LLM 就检索到的证据，给三个维度各打 0-5 分：相关性
   Relevance、支持度 Supportiveness、实用性 Utility。
3. **迭代精炼** —— 分数不达标**不直接丢弃**，而是先反思"证据为什么不够"、
   换个更聚焦的问题重新检索，最多重试 3 次（论文 Fig. 3 的 `attempt: 3`）。

## 一条关键的设计原则

**LLM 只负责打三个 0-5 分和写理由，加权、归一化、比阈值全部由 Python 算。**

论文公式 (1) 的 `S_raw = Σ W_G · S_G` 是个确定性算式，把它交给 LLM 去算，
就等于请一个会算错的语言模型来做算术题——分数会飘。放在 Python 里算，
每个中间量都能打印出来核对，这也是整个项目里最经得起追问的一段逻辑。

## 关于归一化方式的取舍

论文公式 (2) 白纸黑字写的是用「所有候选特征得分的最大最小值」做归一化
（批归一化），但论文 Table 6 里那五行的分数——
100 / 92 / 90 / 80 / 8——**只有按理论范围 0~5 归一化才算得出来**：

    Negative events      R=5.0 S=5.0 U=5.0 → raw 5.0 → 100×5.0/5 = 100 ✓
    China_FX_Reserves    R=5.0 S=4.0 U=5.0 → raw 4.6 → 100×4.6/5 =  92 ✓
    WTI_Futures_Price    R=4.5 S=4.0 U=5.0 → raw 4.5 → 100×4.5/5 =  90 ✓
    US_1Y_Treasury_Yield R=5.0 S=3.0 U=4.5 → raw 4.0 → 100×4.0/5 =  80 ✓
    US Removal ...       R=0   S=0   U=1.0 → raw 0.4 → 100×0.4/5 =   8 ✓

换成批归一化会是 91 / 89 / 78 / 0（5.0 归一化到 100、0.4 归一化到 0），
五行的分数全都对不上。所以本模块**默认用 `theoretical` 模式**，因为它才是
论文实际做出那张表时用的算法；`batch` 模式也实现了，想复现论文的字面描述
或者想看两种模式的差别时可以切过去。
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, Field

from .abstractions.base_llm import BaseLLM
from .abstractions.base_rag import BaseRAG

# 论文公式 (3) 的权重：相关性 0.2，支持度 0.4，实用性 0.4
DEFAULT_WEIGHTS: Dict[str, float] = {
    "relevance": 0.2,
    "supportiveness": 0.4,
    "utility": 0.4,
}

# 三个维度都在 0-5 区间，权重之和为 1，所以 S_raw 的理论范围是 [0, 5]
RAW_SCORE_MAX = 5.0

# 论文正文：「selects features with composite scores meeting or exceeding a threshold of 80」
DEFAULT_THRESHOLD = 80.0

# 论文 Fig. 3 的流程框里写的是 `attempt: 3`
DEFAULT_MAX_ATTEMPTS = 3

# 每条证据喂给 LLM 时截断到多少字，避免一个长文本块把提示词撑爆
_EVIDENCE_CHARS = 700


# --------------------------------------------------------------------------
# 结构化输出用的 schema
# --------------------------------------------------------------------------

class FeatureQuery(BaseModel):
    """第 1 步：把特征名转写成检索问题。"""

    query: str = Field(
        description=(
            "A natural-language question, in Chinese, asking how this feature "
            "affects the USD/CNY exchange rate through a concrete mechanism."
        )
    )


class EvidenceCritique(BaseModel):
    """第 2 步：就检索到的证据打三个维度的分，并给出理由。"""

    relevance: float = Field(
        ge=0, le=5,
        description="0-5. How directly the retrieved evidence discusses this feature's link to USD/CNY.",
    )
    relevance_reason: str = Field(description="理由，中文，一到两句。")

    supportiveness: float = Field(
        ge=0, le=5,
        description="0-5. How strongly the evidence supports the claim that this feature helps prediction.",
    )
    supportiveness_reason: str = Field(description="理由，中文，一到两句。")

    utility: float = Field(
        ge=0, le=5,
        description="0-5. How likely this feature is to improve forecasting accuracy.",
    )
    utility_reason: str = Field(description="理由，中文，一到两句。")


class RefinedQuery(BaseModel):
    """第 3 步：反思证据为何不足，并改写问题。"""

    reflection: str = Field(description="上一轮证据不足的原因，中文。")
    query: str = Field(description="改写后的、更聚焦的检索问题，中文。")


class FeatureEvaluation(BaseModel):
    """单个特征的完整评审记录，含每一轮的证据出处，便于事后追溯。"""

    feature: str
    description: str
    attempts: int = 0
    queries: List[str] = Field(default_factory=list)
    critique: Optional[EvidenceCritique] = None
    raw_score: float = 0.0
    normalized_score: float = 0.0
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    selected: bool = False
    note: str = ""


class FeatureSelectionReport(BaseModel):
    """DA 一轮筛选的完整结果。"""

    evaluations: List[FeatureEvaluation] = Field(default_factory=list)
    selected_features: List[str] = Field(default_factory=list)
    rejected_features: List[str] = Field(default_factory=list)
    weights: Dict[str, float] = Field(default_factory=dict)
    normalization: str = "theoretical"
    raw_threshold: float = 0.0
    normalized_threshold: float = DEFAULT_THRESHOLD
    evidence_available: bool = True
    note: str = ""

    def as_table(self) -> List[Dict[str, Any]]:
        """按论文 Table 6 的样子整理成表格行，方便和论文对照。"""
        rows = []
        for item in sorted(
            self.evaluations, key=lambda entry: entry.normalized_score, reverse=True
        ):
            critique = item.critique
            rows.append({
                "feature": item.feature,
                "relevance": critique.relevance if critique else None,
                "supportiveness": critique.supportiveness if critique else None,
                "utility": critique.utility if critique else None,
                "score": round(item.normalized_score, 1),
                "selected": item.selected,
                "attempts": item.attempts,
                "note": item.note,
            })
        return rows


# --------------------------------------------------------------------------
# 打分与归一化：纯函数，没有 IO，可以单独测
# --------------------------------------------------------------------------

def weighted_raw_score(
    critique: EvidenceCritique, weights: Dict[str, float] = None
) -> float:
    """论文公式 (1)：S_raw = Σ W_G · S_G。"""
    w = weights or DEFAULT_WEIGHTS
    return float(
        w["relevance"] * critique.relevance
        + w["supportiveness"] * critique.supportiveness
        + w["utility"] * critique.utility
    )


def normalize_scores(
    raw_scores: Sequence[float],
    *,
    mode: str = "theoretical",
    raw_scale: float = RAW_SCORE_MAX,
) -> List[float]:
    """把原始得分映射到 [0, 100]。

    - `theoretical`：除以理论满分（5.0）。分数只取决于特征自己，与同批其他
      特征无关，重跑一次结果不变，也是能复现论文 Table 6 的那种算法。
    - `batch`：论文公式 (2) 的字面写法，用同批候选特征的最大最小值。
      副作用是**永远会有一个特征得 100、一个得 0**，于是"阈值 80"实际上
      变成了"取前百分之多少"，而且往候选集里加一个特征会改变所有特征的分数。
    """
    if not len(raw_scores):
        return []

    if mode == "theoretical":
        return [max(0.0, min(100.0, 100.0 * value / raw_scale)) for value in raw_scores]

    if mode == "batch":
        lowest, highest = min(raw_scores), max(raw_scores)
        span = highest - lowest
        if span <= 0:
            # 所有特征得分一样时，公式 (2) 的分母是 0。这里按"全部满分"处理，
            # 因为此时没有相对高低可言，把它们全判成不该选反而更武断。
            return [100.0 for _ in raw_scores]
        return [100.0 * (value - lowest) / span for value in raw_scores]

    raise ValueError(f"不认识的归一化方式：{mode}（可选 theoretical / batch）")


def raw_threshold(
    normalized_threshold: float = DEFAULT_THRESHOLD,
    *,
    mode: str = "theoretical",
    raw_scale: float = RAW_SCORE_MAX,
) -> float:
    """把归一化阈值换算回原始分阈值。

    提前换算的原因：循环里每评完一个特征就要判断"够不够"，而批归一化要等
    所有特征都评完才知道最大最小值。用理论模式时这个换算是确定的
    （阈值 80 ⇔ 原始分 ≥ 4.0），所以循环可以边评边判、达标即停，省掉大量
    LLM 调用。批归一化模式没法提前换算，只能先全部评完再统一算——这也是
    它除了"复现字面描述"之外没什么优势的原因之一。
    """
    if mode == "theoretical":
        return normalized_threshold / 100.0 * raw_scale
    return float("-inf")


# --------------------------------------------------------------------------
# 缓存：DA 是网络延迟瓶颈，不是算力瓶颈，缓存是唯一能让迭代变得可忍受的手段
# --------------------------------------------------------------------------

class _DiskCache:
    """把 LLM 的返回按内容哈希存到磁盘。命中时零 API 调用、零等待。

    缓存键里还带着**模型名**（`cache_namespace`）。不带的话，把 `DEEPSEEK_MODEL`
    从 `deepseek-chat` 换成 `deepseek-reasoner` 之后，DA 会直接命中上一个模型留下
    的分数——看起来"换模型没影响"，实际上根本没调新模型。这种静默复用是这个项目
    最不想要的那类错误。
    """

    def __init__(self, directory: Optional[Path]):
        self._directory = Path(directory) if directory else None
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Optional[Path]:
        return self._directory / f"{key}.json" if self._directory else None

    @staticmethod
    def make_key(*parts: Any) -> str:
        payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[dict]:
        path = self._path(key)
        if path is None or not path.exists():
            self.misses += 1
            return None
        try:
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.misses += 1
            return None

    def set(self, key: str, value: dict) -> None:
        path = self._path(key)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass


# --------------------------------------------------------------------------
# 提示词
# --------------------------------------------------------------------------

_QUERY_PROMPT = """你是一个汇率预测项目的特征评审员。

请把下面这个候选特征，转写成一个**用来在知识库里检索证据**的中文问题。
问题要问到具体的传导机制（利率平价、资本流动、央行干预、风险溢价等），
不要只泛泛地问"这个特征重不重要"。

候选特征：{feature}
特征说明：{description}

只输出一个问题。"""

_CRITIQUE_PROMPT = """你是一个汇率预测项目的特征评审员。请**只依据下面给出的证据**来评分。

候选特征：{feature}
特征说明：{description}
检索用的问题：{query}

检索到的证据：
{evidence}

请就三个维度各打 0-5 分，并给出中文理由：
1. 相关性 Relevance：证据是否**直接讨论**了该特征与 USD/CNY 汇率的联系。
2. 支持度 Supportiveness：证据在多大程度上支撑"该特征对预测汇率有用"这一论断。
3. 实用性 Utility：该特征是否可能对模型的预测精度有实际贡献。

**评分纪律（务必遵守）：**
- 你要评的是「**这些证据够不够**」，而不是「这个特征在现实世界里重不重要」。
- 如果检索到的证据没有讨论该特征的作用机制，**相关性必须给 0-1 分**，
  哪怕凭你自己的知识知道这个特征在现实里很重要。
- 证据互相矛盾、或只提到该特征却没说它怎么影响汇率，都算支持度低。
- 实用性可以略高于相关性，因为一个机制上说得通的变量，即使这次证据弱，
  也可能对模型有贡献。"""

_REFLECT_PROMPT = """你是一个汇率预测项目的特征评审员。

候选特征「{feature}」（说明：{description}）在第 {attempt} 轮评审中没达到阈值。

上一轮用的检索问题：{query}
上一轮的评分：相关性 {relevance}、支持度 {supportiveness}、实用性 {utility}
上一轮的理由：相关性——{relevance_reason}；支持度——{supportiveness_reason}；实用性——{utility_reason}

先分析证据不足的原因，通常是下面几种之一：
- 问题太宽泛，没问到具体机制；
- 问题问的机制方向不对（比如问的是"是否重要"，而知识库讲的是"通过什么渠道起作用"）；
- 知识库里确实没有这方面的内容。

然后写一个**更聚焦**的新问题重新检索。
如果判断知识库里根本没有相关内容，就把新问题写成对缺失内容最接近的追问，
并在反思里说明这一点。"""


def format_evidence(documents: Sequence[Dict[str, Any]]) -> str:
    """把检索结果拼成给 LLM 看的证据块，带出处。"""
    blocks = []
    for index, document in enumerate(documents, start=1):
        heading = document.get("heading") or "（无小节标题）"
        content = str(document.get("content", ""))[:_EVIDENCE_CHARS]
        blocks.append(
            f"[证据 {index}] 出处：{document.get('source', '未知')} > {heading}"
            f"（相似度 {document.get('score', 0)}）\n{content}"
        )
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# 引擎主体
# --------------------------------------------------------------------------

class EvidenceBasedSelector:
    """论文 DA 的三步循环。"""

    def __init__(
        self,
        rag_service: BaseRAG,
        llm_service: BaseLLM,
        *,
        top_k: int = 3,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        weights: Dict[str, float] = None,
        threshold: float = DEFAULT_THRESHOLD,
        normalization: str = "theoretical",
        cache_dir: Optional[Path] = None,
        llm_config: Optional[Dict[str, Any]] = None,
    ):
        self._rag = rag_service
        self._llm = llm_service
        self._top_k = int(top_k)
        self._max_attempts = max(1, int(max_attempts))
        self._weights = dict(weights or DEFAULT_WEIGHTS)
        self._threshold = float(threshold)
        self._normalization = normalization
        self._llm_config = llm_config or {"model": "decision_llm"}
        # 默认模型（`deepseek-chat`）的命名空间是空串，走的是原来的键格式，
        # 项目里已有的缓存仍然命中；换模型才会另起一套键。
        self._cache_namespace = str(getattr(llm_service, "cache_namespace", "") or "")
        self._cache = _DiskCache(cache_dir)

    @property
    def cache_stats(self) -> Dict[str, int]:
        return {"hits": self._cache.hits, "misses": self._cache.misses}

    def _knowledge_base_available(self) -> bool:
        """知识库里有没有东西可检索。

        查不到这个信息时一律当作"有"——宁可多花一次调用，也不要因为 RAG 实现
        没提供 `document_count` 就把所有特征误判成证据不足。
        """
        count = getattr(self._rag, "document_count", None)
        return True if count is None else count > 0

    def _progress_score(self, evaluation: FeatureEvaluation) -> str:
        """打印每个特征时的即时得分。

        理论归一化模式下每个特征的分数是当场就确定的，可以直接显示；
        批归一化要等所有特征评完才知道最大最小值，这时只能显示原始分——
        显示一个假的归一化分数比不显示更糟。
        """
        if evaluation.critique is None:
            return "无证据"
        if self._normalization == "theoretical":
            shown = normalize_scores([evaluation.raw_score], mode="theoretical")[0]
            return f"得分 {shown:.1f}"
        return f"原始分 {evaluation.raw_score:.2f}"

    def select(
        self,
        catalog: Dict[str, str],
        *,
        limit: int = 0,
        progress: bool = True,
    ) -> FeatureSelectionReport:
        """对候选特征逐个评估，返回完整报告。

        `limit` 大于 0 时只评前 N 个特征，用于快速试跑提示词。
        """
        features = list(catalog)
        if limit and limit > 0:
            features = features[:limit]

        if not self._knowledge_base_available():
            # 知识库是空的：所有特征都判为"证据不足"，直接返回。
            # 不先问一句再发现没证据——那会给每个特征白烧一次 API 调用。
            print("知识库为空，没有可检索的证据。所有候选特征都会被判为证据不足。")
            print("请先跑：.\\venv\\Scripts\\python.exe scripts\\build_knowledge_base.py")
            evaluations = [
                FeatureEvaluation(
                    feature=feature,
                    description=catalog[feature],
                    note="知识库为空，检索不到任何证据",
                )
                for feature in features
            ]
            return FeatureSelectionReport(
                evaluations=evaluations,
                selected_features=[],
                rejected_features=[item.feature for item in evaluations],
                weights=dict(self._weights),
                normalization=self._normalization,
                normalized_threshold=self._threshold,
                evidence_available=False,
                note="知识库为空，本次评审没有依据",
            )

        raw_cutoff = raw_threshold(self._threshold, mode=self._normalization)
        evaluations: List[FeatureEvaluation] = []

        for index, feature in enumerate(features, start=1):
            if progress:
                print(f"[{index}/{len(features)}] 评估 {feature} …", end="", flush=True)
            evaluation = self.evaluate_one(feature, catalog[feature])
            evaluations.append(evaluation)
            if progress:
                print(f" {self._progress_score(evaluation)}"
                      f"（{evaluation.attempts} 轮）")

        scores = normalize_scores(
            [item.raw_score for item in evaluations], mode=self._normalization
        )
        for item, score in zip(evaluations, scores):
            item.normalized_score = round(score, 1)
            item.selected = score >= self._threshold

        selected = [item.feature for item in evaluations if item.selected]
        rejected = [item.feature for item in evaluations if not item.selected]

        return FeatureSelectionReport(
            evaluations=evaluations,
            selected_features=selected,
            rejected_features=rejected,
            weights=dict(self._weights),
            normalization=self._normalization,
            raw_threshold=round(raw_cutoff, 3) if raw_cutoff != float("-inf") else 0.0,
            normalized_threshold=self._threshold,
            evidence_available=any(item.evidence for item in evaluations),
            note=(
                f"归一化方式 {self._normalization}，阈值 {self._threshold}"
                f"（折算到原始分为 {raw_cutoff:.2f}）"
                if raw_cutoff != float("-inf")
                else f"归一化方式 {self._normalization}，阈值 {self._threshold}"
            ),
        )

    def evaluate_one(self, feature: str, description: str) -> FeatureEvaluation:
        """评估单个特征。这是三步循环的完整实现。"""
        evaluation = FeatureEvaluation(feature=feature, description=description)
        query = self._generate_query(feature, description)
        evaluation.queries.append(query)

        for attempt in range(1, self._max_attempts + 1):
            evaluation.attempts = attempt
            documents = self._rag.retrieve(query, self._top_k)

            if not documents:
                evaluation.note = "知识库里检索不到任何证据"
                return evaluation

            evaluation.evidence = documents
            critique = self._critique(feature, description, query, documents)
            evaluation.critique = critique
            evaluation.raw_score = weighted_raw_score(critique, self._weights)

            cutoff = raw_threshold(self._threshold, mode=self._normalization)
            # 批归一化模式下 cutoff 是负无穷，这里先停下、由 select() 统一判定
            if cutoff == float("-inf") or evaluation.raw_score >= cutoff:
                break
            if attempt >= self._max_attempts:
                evaluation.note = f"重试 {attempt} 轮仍未达标"
                break

            query = self._reflect_and_refine(
                feature, description, critique, query, attempt
            )
            evaluation.queries.append(query)

        return evaluation

    # ---------- 三个步骤各自对应的 LLM 调用 ----------

    def _generate_query(self, feature: str, description: str) -> str:
        prompt = _QUERY_PROMPT.format(feature=feature, description=description)
        result = self._structured(prompt, FeatureQuery, "query", feature, description)
        query = result.get("query") or feature
        return str(query).strip()

    def _critique(
        self,
        feature: str,
        description: str,
        query: str,
        documents: Sequence[Dict[str, Any]],
    ) -> EvidenceCritique:
        evidence = format_evidence(documents)
        prompt = _CRITIQUE_PROMPT.format(
            feature=feature, description=description, query=query, evidence=evidence
        )
        payload = self._structured(
            prompt, EvidenceCritique, "critique", feature, query
        )
        return EvidenceCritique(**payload)

    def _reflect_and_refine(
        self,
        feature: str,
        description: str,
        critique: EvidenceCritique,
        query: str,
        attempt: int,
    ) -> str:
        prompt = _REFLECT_PROMPT.format(
            feature=feature,
            description=description,
            attempt=attempt,
            query=query,
            relevance=critique.relevance,
            supportiveness=critique.supportiveness,
            utility=critique.utility,
            relevance_reason=critique.relevance_reason,
            supportiveness_reason=critique.supportiveness_reason,
            utility_reason=critique.utility_reason,
        )
        payload = self._structured(
            prompt, RefinedQuery, "reflect", feature, query, attempt
        )
        refined = payload.get("query") or query
        return str(refined).strip()

    def _structured(
        self, prompt: str, schema: type, *cache_parts: Any
    ) -> Dict[str, Any]:
        """带磁盘缓存的结构化调用。缓存键包含 schema 名与提示词原文，
        改了提示词就会自动失效，不会拿着旧结果骗自己。"""
        # 命名空间为空时**不往键里加元素**，而不是加一个空串——加空串会改变
        # 序列化结果，把默认模型那批已有缓存全部作废。
        parts = (schema.__name__, prompt) + cache_parts
        if self._cache_namespace:
            parts = (self._cache_namespace,) + parts
        key = _DiskCache.make_key(*parts)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        result = self._llm.invoke_structured(prompt, schema=schema, config=self._llm_config)
        payload = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        self._cache.set(key, payload)
        return payload
