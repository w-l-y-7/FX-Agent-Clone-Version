# FX-Agents 复现 + AegisFX-Agent 扩展

本仓库是论文 **《A Novel Exchange Rate Forecasting Paradigm Based on Multi-Agent
Collaboration and Multimodal Big Data-Driven Methods》**（Di Han 等，*Big Data Mining
and Analytics* 9(4), 2026, pp. 1009–1025）公开代码
<https://github.com/Kon-Kwok/FX-Agent> 的克隆，并在其上做了两件事：

| 线 | 内容 | 代码位置 |
|---|---|---|
| **线一：论文复现** | 把论文的四智能体框架重构出来、逐项核对、跑出结果 | `main.py`、`src/core`、`src/agents`、`src/services`、`src/legacy_models` |
| **线二：AegisFX-Agent** | 在 DA 的检索层上另起一条独立研究线：冲突感知 RAG + 共形预测 + 执行闸门 | `src/aegisfx/`、`app/`、`scripts/` |

> ### 先说清楚定位
>
> **本项目不是论文数值结果的严格复现。** 论文公开了数据和框架，但没有公开跑实验的
> 那套脚本（详见第三节）。本仓库是按论文正文 + 从作者 git 历史
> 恢复的实现重新搭的**可运行重构**，并在统一的**时间切分、as-of 检索、异常收益
> 评测协议**下开展独立的方法扩展。
>
> 实验代号（全文统一）：
>
> | 代号 | 含义 |
> |---|---|
> | `B0` | 原仓库的可运行重构，**仅说明项目起点，不声称复现论文数值** |
> | `E1-semantic` | 纯语义检索基线 |
> | `E2-fixed-reranker` | 固定权重金融重排序 |
> | `E3-adaptive-conflict-rag` | 语义约束的自适应冲突感知重排序 |
> | `E4` | 冲突感知 + 共形区间 + 执行闸门（预测值与 E3 相同，只差执不执行） |

![FX-Agents 框架](./figure1.png)
*图 1：FX-Agents 的多智能体协作流程（论文原图）。*

---

## 一、快速开始

```powershell
# 1. 虚拟环境与依赖
python -m venv venv
.\venv\Scripts\pip.exe install -r requirements.txt

# 2. 密钥：复制 .env.example 为 .env，填上 DEEPSEEK_API_KEY
#    想按论文口径用 R1，加一行 DEEPSEEK_MODEL="deepseek-reasoner"
copy .env.example .env

# 3. 建知识库索引（第一次要下载约 95MB 的向量模型）
.\venv\Scripts\python.exe scripts\build_knowledge_base.py

# 4. 跑论文那条线。先用 --limit 8 试水，约 3 分钟
.\venv\Scripts\python.exe main.py --limit 8
```

**线二不需要密钥就能自检**（几秒到几十秒，不联网）：

```powershell
.\venv\Scripts\python.exe scripts\verify_all.py      # 七个套件一次性跑完
.\venv\Scripts\streamlit.exe run app\Home.py         # 演示界面
```

`verify_all.py` 可用 `--skip store retrieval app stage4 stage5 stage6 risk` 跳过指定套件。

> 线二的运行数据（`data/aegisfx/`）**不进版本库**，新克隆的仓库里没有。要用界面
> 或跑评测，先按第四节「怎么跑」建一次事件库与索引。

---

## 二、目录结构

```
main.py                      线一入口：LangGraph 编排的四智能体流程
app/                         线二演示界面（Streamlit，4 页）
  ├── Home.py                    启动入口，注册 4 个页面
  ├── ui.py                      三页共用的渲染函数（闸门清单、信号、事件卡片）
  └── pages/                     event_signal / evidence / risk / labeling
src/
├── core/                   线一核心逻辑（不依赖网络，纯计算）
├── agents/                 线一四个智能体
├── services/               线一对外能力（LLM、向量 RAG、预测服务）
├── legacy_models/          论文 Table 7 的全部五个模型 + Data.xlsx
└── aegisfx/                ★ 线二新增算法包（与线一解耦，可自由改）
    ├── schemas/                数据载体：RawEvent / ParsedEvent / Signal / 闸门 / 视图
    ├── ingestion/              人机混合输入通道（投递 → 解析 → 信号 → 闸门）
    ├── data/                   行情、异常收益、事件库、标签成熟时间协议
    ├── retrieval/              E1/E2/E3 三路检索 + 指标 + 查询集
    ├── conflict/               双冲突度量、按交易日聚合、方向回退链
    ├── conformal/              CP0 滚动共形 / ACI / CP1（默认关闭）
    ├── routing/                E4 路由器（特征、门控、时间验证、融合）
    ├── risk/                   执行风险闸门与历史回测
    └── cikrag/                 第 6 阶段：冲突感知增量知识 RAG
data/
├── knowledge_base/        线一 DA 检索的证据（notes/ 是项目自行整理的机制说明）
├── inbox/                 线二人工投递目录（丢 .json 进去即可）
└── aegisfx/               线二运行时状态（**不进版本库**）
author_original_code/        从作者 git 历史恢复的原始实现，见其 README
scripts/                     独立运行的脚本（见第五节）
reports/                     全部运行结果（**进版本库**，见第七节）
docs/                        中文文档（见第六节）
```

---

## 三、线一：论文复现（B0）

**一句话结论：框架部分已完整重构并能端到端跑通；实验部分没能还原。**
根本原因是公开仓库被精简过——作者真正的实现（Optuna 调参、FAISS 向量检索、
RNN 与 LSTM+Attention 的网络结构）在 2025-07-16 的提交 `a838298` 里被删掉了，
已从 git 历史恢复到 `author_original_code/`；**仍然缺的是把这些零件串起来、
跑出论文那次实验的脚本**。所以 Table 7 那组精度无法复现。

完整对照、逐项结论与待确认问题清单见
[`docs/进度记录.md`](docs/进度记录.md)，使用说明与每个模块在做什么见
[`docs/运行指南.md`](docs/运行指南.md)，恢复出来的原始实现见
[`author_original_code/README.md`](author_original_code/README.md)。

### 怎么跑

```powershell
# 主流程（DA 特征筛选 + Ridge 预测）
.\venv\Scripts\python.exe main.py --limit 8        # 只评前 8 个候选特征，试水
.\venv\Scripts\python.exe main.py                  # 完整 32 个候选特征
.\venv\Scripts\python.exe main.py --optimize       # 换用 Optuna 调参 + SHAP 解释
.\venv\Scripts\python.exe main.py --live           # 原作者的实时数据线（AKShare + Firecrawl）

# 不需要密钥、不联网的验证脚本（几秒钟，适合直接看结果）
.\venv\Scripts\python.exe scripts\verify_pa2_aggregation.py        # PA2 事件聚合，逐列对照
.\venv\Scripts\python.exe scripts\verify_traditional_baseline.py   # §4.4 传统基线，对照 Table 8

# 只跑 DA 的特征筛选、不跑预测（**需要密钥**；结果缓存在 .cache/da/）
.\venv\Scripts\python.exe scripts\run_da_selection.py

# 论文 Table 7 的五个深度模型（注释里是本机 CPU 实测耗时）
.\venv\Scripts\python.exe src\legacy_models\Timesnet.py            # 约 1 分钟
.\venv\Scripts\python.exe src\legacy_models\RNN.py                 # 约 3 分钟
.\venv\Scripts\python.exe src\legacy_models\LSTM_Attention.py      # 约 3 分钟
.\venv\Scripts\python.exe src\legacy_models\Transformer.py         # 约 4 分钟
.\venv\Scripts\python.exe src\legacy_models\TFT.py                 # 约 25~30 分钟

# 按作者的搜索空间调深度模型超参
.\venv\Scripts\python.exe scripts\optimize_deep_model.py --trials 10
```

`main.py` 的其余开关（`--no-cache` / `--normalization` / `--threshold` / `--trials`）
见 [`docs/运行指南.md`](docs/运行指南.md) 第三节。

### 能对上论文的部分

- `Data.xlsx` 里 25 个事件列的求和，与论文 Table 5 的 sample size 列**逐列吻合**；
  2017-02-28 那一行与 Table 4 **逐位吻合**；
- PA2 论文说过滤后剩 **429 条**——实测正是「有事件发生的交易日数」；
- 传统基线（§4.4）：时序线算出的 `{USD_Index, CN_1Y_GovBond_Yield}` 与论文 Table 8
  **完全一致**，事件线论文的 2 个特征正是算出来的前两名；
- DA 特征筛选 32 个候选与论文 Table 8 的 4 个特征**交集 3 个**，排序结构对得上。

### 跑出来的数（HORIZON=20，论文 Table 7 的 `FA(χ)+PA1+PA2+DA` 列）

| 模型 | 本机 RMSE | 本机 MAPE | 论文 RMSE | 论文 MAPE |
|---|---|---|---|---|
| Transformer | 0.3238 | 3.8719% | 0.0566 | 0.6532% |
| LSTM + Attention | 0.2667 | 3.4820% | 0.0599 | 0.7205% |
| RNN | 0.2506 | 3.2873% | 0.0650 | 0.8121% |
| TFT | 0.3721 | 4.9051% | 0.0330 | 0.3427% |
| TimesNet | 0.2035 | 2.6077% | 0.0436 | 0.4874% |
| **朴素基准（价格不变）** | **0.0813** | **0.8438%** | — | — |

**五个模型都没跑赢朴素基准**，而论文里它们分别赢 30%~60%。这是**定性差别**，
不是"差一点"。另外 **TFT 在本机是最差的，而论文里它最好**，这一处最可疑、
属于待查项而非结论。原因与三处待确认的公式/表格不一致，见
[`docs/进度记录.md`](docs/进度记录.md) 第二节。

---

## 四、线二：AegisFX-Agent

把论文 DA 的「检索—自评—精炼」继续推进：**当检索到的知识互相冲突、或根本没有
相关知识时，系统应该怎么表态。** 方案见
[`docs/AegisFX-Agent方案.md`](docs/AegisFX-Agent方案.md)，做到哪了、每步实测
什么见 [`docs/AegisFX-Agent进度.md`](docs/AegisFX-Agent进度.md)。

**研究问题（RQ1）**：普通语义检索、固定金融重排序和语义约束自适应重排序，
在**主题相关性**与**离线结果有用性**之间表现出怎样的权衡？

### 三条数据线，不要混

| 线 | 数据 | 用途 |
|---|---|---|
| 研究线 | `Data.xlsx` 日频 USD/CNY，2017–2024 | 复现、RAG 研究、异常收益、1/5/20 日预测 |
| 摄入线 | 人工投递 + CLI + inbox 目录 | 把最新消息读进来 |
| 模拟线 | USD/CNH 分钟级 | 滑点、点差、延迟、paper trading（**可裁剪**） |

**事件库底料**：论文语料 445 条新闻，按「同一交易日 + 同一事件类型 + 同一发起方」
合并为 **437 条**，时间切分为检索池 2017–2022（**281 条**）与评测集 2023–2024
（**156 条**）。检索质量评测用 20 条查询（按事件族配额抽取）。

### 人机混合输入通道（三个入口，一个队列）

```powershell
# 入口 1：Streamlit 界面上的输入框（跑 app 后直接用）
# 入口 2：CLI
.\venv\Scripts\python.exe scripts\ingest_event.py --text "美国宣布对华加征关税" --url "https://example.com/1"
.\venv\Scripts\python.exe scripts\ingest_event.py --list 10
# 入口 3：把 .json 丢进 data/inbox/，再扫一次
.\venv\Scripts\python.exe scripts\ingest_event.py --process-inbox
```

三个入口都写进同一个 `RawEvent` 队列，之后的处理路径完全一致。
投递文件格式见 [`data/inbox/README.md`](data/inbox/README.md)。

**未验证消息（没给来源链接）照常跑研究层，但执行层自动 `BLOCKED`**——这条设计
让"人能随时喂消息"和"不能拿小道消息下单"同时成立。

### 算法要点

- **异常收益**：$h$ 日尺度市场模型（DXY + 利差），估计窗口 250 天。逐日估在本数据上
  不成立（人民币日频不跟美元指数走），1 日尺度模型几乎不起作用——这是事实，报告里
  必须把 1 日和 5/20 日分开讲；
- **As-of 检索**：按 `source_published_at` 截断，越界直接抛 `AsOfViolation`（硬断言）；
  收益标签另有成熟时间协议 `label_available_at = t + h + 1`，防标签泄漏；
- **双冲突度量**：方向分歧度 CI + 幅度离散度 D，两项交叉才能把「噪声」和「真冲突」分开；
- **共形预测**：Walk-Forward 滚动共形，输出 80% / 90% 两档，**显式报告实际覆盖率**；
- **执行闸门**：18 项确定性检查，输出**清单而不是布尔值**，方便事后审计；
- **E4 路由**：$\hat r_{E4} = (1-g)\hat r_{E1} + g \hat r_{E2}$，$g$ 由逻辑门控学习，
  purged expanding-window 时间验证。

**风险 Agent 全部走确定性规则，不调用大模型**；只允许 `research / backtest / shadow /
paper` 四种模式，**仓库默认不得进入任何真实交易模式**。

### 怎么跑

```powershell
# 界面（含内部标注工具页）
.\venv\Scripts\streamlit.exe run app\Home.py

# 全量自检：事件库 → 检索层 → 界面 → 第 4/5/6 阶段 → 风险 Agent
.\venv\Scripts\python.exe scripts\verify_all.py

# 事件库与检索
.\venv\Scripts\python.exe scripts\build_event_store.py
.\venv\Scripts\python.exe scripts\build_retrieval_index.py
.\venv\Scripts\python.exe scripts\audit_event_corpus.py
.\venv\Scripts\python.exe scripts\probe_data_sources.py

# 三路检索评测（含配对 bootstrap）
.\venv\Scripts\python.exe scripts\evaluate_retrieval.py
# 需要人工标注时：先用 run_retrieval.py 产出工作表，再用 app 的「标注工具」页填，
# 最后 migrate_labels.py 迁移结构

# 第 5 阶段：伪查询 → 冲突信号 → 路由 → 冻结配置 → 评测
.\venv\Scripts\python.exe scripts\build_pseudo_queries.py
.\venv\Scripts\python.exe scripts\evaluate_conflict_signal.py
.\venv\Scripts\python.exe scripts\train_stage5_router.py
.\venv\Scripts\python.exe scripts\evaluate_stage5.py --split dev
.\venv\Scripts\python.exe scripts\lock_stage5_config.py

# 风险闸门：在 2023 开发集冻结阈值并回测
.\venv\Scripts\python.exe scripts\evaluate_risk_gate.py --split dev --calibrate

# 第 6 阶段：CIK-RAG
.\venv\Scripts\python.exe scripts\evaluate_cognitive_limits.py
.\venv\Scripts\python.exe scripts\run_cikrag.py
```

> **最终测试集只跑一次。** 2024 准留出的评测命令是
> `scripts\evaluate_stage5.py --split final --confirm-locked`，
> `verify_all.py` **不会**自动重跑它。人工标注相关脚本
> （`migrate_labels.py`、`app/pages/labeling.py`）见第 4 阶段说明。

### 实测结果（**负面结果居多，如实保留**）

**三路检索**（as-of 候选池，Top-10，18 条查询，`offline_outcome_utility` NDCG@5）：

| 方法 | 均值 | 变化 | 配对 bootstrap 95% 区间 |
|---|---:|---|---|
| E1-semantic | 0.4585 | — | — |
| E2-fixed-reranker | 0.5011 | +0.0426 | [−0.0709, +0.1616]（跨 0） |
| E3-adaptive-conflict-rag | 0.4364 | −0.0221 | [−0.1061, +0.0483]（跨 0） |

→ **三个方法在离线结果有用性上分不出高下。** 而且 E3 确实在工作（多数查询 α 取到
上限），**活跃但没帮上忙**，比"没生效"是更强的负面结论。

**主题相关性口径给出相反结论**：大模型标注下 E1 0.879 vs E2 0.699。**两套口径都要报**
——只报前者会得出"重排序有害"，只报后者会得出"重排序有效"。另：**大模型当第二标注员
已被判定不可靠并放弃**（给分近乎均分，对同族同源候选给出不同分）。

**冲突有效性：不成立。** 四条判据在 5 日和 20 日**都只过 1/4**，相关接近 0、
bootstrap 区间全跨 0、符号逐年翻转。→ **CP1（冲突尺度化共形）在代码里固化成自动降级，
不许启用。**

**共形区间没有跑赢最简单的对照。** h=5 上固定历史分位既更接近目标覆盖、区间又更窄；
所有方法都**明显过度覆盖**（20 日 CP0 的 80% 区间实际覆盖 89.6%）。

**E4 路由器**：OOF（2020–2022，141 条）g 均值 0.194，没有门控塌缩——路由器自己学到
「金融重排序不太可信」。

**2024 准留出（首次运行，配置不再调整）**：

| 期限 | 方法 | MAE | 方向准确率 | E4 − E1 的 MAE 差（95% 区间） |
|---|---|---:|---:|---|
| 5 日 | E1 | 0.00299 | 55.6% | — |
| 5 日 | E4 | 0.00294 | 56.8% | −0.00005 [−0.00019, +0.00011] |
| 20 日 | E1 | 0.00640 | 52.6% | — |
| 20 日 | E4 | 0.00628 | 50.0% | −0.00012 [−0.00030, +0.00010] |

→ 两个期限 MAE 都略有改善，但**区间都跨 0，不能声称 E4 稳定优于 E1**。

**执行闸门最终回测**：

| 项目 | 全部执行 | 风险闸门 |
|---|---:|---:|
| 交易数 | 78 | **1** |
| 漂移调整累计 PnL | **+1.00%** | **−0.28%** |
| 漂移调整最大回撤 | 8.89% | 0.28% |

→ **闸门显著减少交易与回撤，但没有改善漂移调整收益；它拦下了大量利润，
唯一放行的一笔仍是事件相关亏损。执行闸门无效。** 这是负面结果，与第 5 阶段一致：
点预测改善不显著、区间偏宽，系统没有足够依据自动交易。

**CIK-RAG（第 6 阶段）**：受控认知边界压力测试中，选择性规则 **8/8** 符合预期、
**0 次不安全自动采纳**，而"只看平均增益正负"的基线是 3/8、出现 5 次不安全自动采纳。
但真实仓库知识库只有 4 份内部说明，32 个候选中 **29 个仅供研究、2 个转人工、
1 个自动拒绝、0 个自动采纳**——**这是证据不足下的保守结果，不应包装成预测性能提升**。
推导与论文写法见 [`docs/CIK-RAG认知边界与增量学习.md`](docs/CIK-RAG认知边界与增量学习.md)。

### 必须承认的局限

- 研究设计已受到部分 2023–2024 结果启发（第 3、4 阶段都看过那批查询），
  所以 2024 这一集只能叫**准留出**，真正的外部确认要等 2025–2026 的新事件；
- 历史语料没有逐笔 Bid/Ask、实时 API 健康与订单幂等状态，**完整 paper gate 仍保持
  `BLOCKED`**，页面上的回测只隔离信号时可知的研究规则；
- **方向准确率必须与多数类基准（异常收益口径 55.6%）并列报告**，否则数字没有意义；
- 事件成簇、20 日窗口大量重叠，**标签不独立**，一律用 block bootstrap 报区间。

---

## 五、脚本一览

全部脚本在 `scripts/`。**除标注「需要密钥」的几个外，其余都不联网、不需要密钥**
（向量模型读本地缓存，拿不到会自动退到 TF-IDF）。每个脚本的完整参数用 `--help` 查。

| 脚本 | 作用 | 密钥 |
|---|---|---|
| **线一：复现** | | |
| `build_knowledge_base.py` | 建 DA 用的知识库向量索引（首次下载约 95MB 模型） | — |
| `run_da_selection.py` | 只跑 DA 的特征筛选，不跑预测 | **需要** |
| `verify_pa2_aggregation.py` | PA2 新闻→日频事件聚合，与论文数据逐列对照 | — |
| `verify_traditional_baseline.py` | 论文 §4.4 传统特征工程基线，对照 Table 8 | — |
| `optimize_deep_model.py` | 按作者的搜索空间用 Optuna 调深度模型超参 | — |
| **线二：事件库与检索** | | |
| `probe_data_sources.py` | 数据源接入可行性探测（FRED / akshare / OANDA） | 联网 |
| `audit_event_corpus.py` | 事件语料审计；`--llm-sample N` 才真的调 API | `--llm-sample` |
| `build_event_store.py` | 建历史事件库（437 条，含异常收益与市场状态） | — |
| `verify_event_store.py` | 事件库健康检查（11 项） | — |
| `build_retrieval_index.py` | 建事件向量索引与 20 条查询集 | — |
| `run_retrieval.py` | 跑检索、产出人工标注工作表；`--llm-label` 加大模型第二标注 | `--llm-label` |
| `migrate_labels.py` | 把标注表迁移到第 4 阶段结构（幂等） | — |
| `evaluate_retrieval.py` | 三路检索评测（主题相关性 + 离线结果有用性 + 配对 bootstrap） | — |
| `ingest_event.py` | 在线投递入口：`--text` / `--file` / `--process-inbox` / `--list` / `--kill-switch` | 默认需要，`--no-llm` 可免 |
| **线二：第 5 阶段（路由 / 冲突 / 共形）** | | |
| `build_pseudo_queries.py` | 构造 walk-forward 伪查询训练集 | — |
| `train_stage5_router.py` | 训练 E4 路由器，产时间折 OOF 预测 | — |
| `evaluate_conflict_signal.py` | 冲突指标有效性检验（四条判据，决定 CP1 开不开） | — |
| `lock_stage5_config.py` | 冻结第 5 阶段配置并算 config / data hash | — |
| `evaluate_stage5.py` | 点预测与共形区间评测（`--split dev` / `final`） | — |
| `evaluate_risk_gate.py` | 风险闸门冻结阈值（`--calibrate`）与历史回测 | — |
| **线二：第 6 阶段（CIK-RAG）** | | |
| `run_cikrag.py` | CIK-RAG 主实验：知识侧抽观点 + 预测侧样本外消融 | — |
| `evaluate_cognitive_limits.py` | 8 情形认知边界压力测试 | — |
| **自检与测试** | | |
| `verify_all.py` | **总入口**：一条命令跑完下面全部套件，`--skip` 可跳过 | — |
| `verify_retrieval.py` / `verify_app.py` | 检索层健康检查（13 项）/ 界面冒烟测试 | — |
| `test_aegisfx.py` / `test_stage5.py` / `test_stage6.py` / `test_risk.py` | 第 4/5/6 阶段与风险 Agent 的测试套件 | — |

---

## 六、文档索引

| 文档 | 内容 |
|---|---|
| [`docs/运行指南.md`](docs/运行指南.md) | **线一使用手册**：怎么用、每个模块在做什么、哪里和论文对不上以及为什么 |
| [`docs/进度记录.md`](docs/进度记录.md) | **线一结论**：复现到哪一步、交付边界、待确认问题清单、给作者的问题清单 |
| [`docs/AegisFX-Agent方案.md`](docs/AegisFX-Agent方案.md) | **线二方案**：E0–E4 定义、冲突度量、共形预测、执行闸门、八周计划、术语口径 |
| [`docs/AegisFX-Agent进度.md`](docs/AegisFX-Agent进度.md) | **线二进度**：第 1–6 阶段逐步实测结果、每个数字的来源脚本、踩过的坑 |
| [`docs/CIK-RAG认知边界与增量学习.md`](docs/CIK-RAG认知边界与增量学习.md) | CIK-RAG 的完整推导、金融例子、增量算法、实验边界与论文写法 |
| [`author_original_code/README.md`](author_original_code/README.md) | 从作者 git 历史恢复的原始实现：每个文件是真实现还是模拟桩、从哪个提交恢复 |
| [`data/knowledge_base/README.md`](data/knowledge_base/README.md) | DA 检索用的知识库怎么加料、为什么 `notes/` 是项目自行整理的 |
| [`data/inbox/README.md`](data/inbox/README.md) | 人工投递的 JSON 格式与字段说明 |

---

## 七、运行产物在哪里看

**`reports/` 下的运行结果是要提交进版本库的**（它们是"做了什么"最直接的证据），
不用自己跑就能看到：

| 文件 | 内容 |
|---|---|
| `reports/da_selection.json` | 线一 DA 的完整评审记录：32 个特征的分数、理由、证据出处 |
| `reports/pa2_aggregation.json` | PA2 事件聚合的逐列对照结果 |
| `reports/traditional_baseline.json` | 传统基线（相关系数、XGBoost 重要度、两种口径对比） |
| `reports/main_pipeline*.log` | 完整流程与 `--optimize` 的运行输出 |
| `reports/*_run.log`、`reports/*_predictions.png` | 五个深度模型的原始日志与预测曲线 |
| `reports/aegisfx_corpus_audit.json` | 事件语料审计：两条解析路径的一致率 |
| `reports/aegisfx_data_source_probe.json` | 数据源可行性探测结论 |
| `reports/aegisfx_event_store.json` | 事件库健康检查 |
| `reports/aegisfx_retrieval*.json` | 三路检索评测（含双目标口径） |
| `reports/stage5/` | 第 5 阶段：冻结配置、伪查询、路由训练、开发集与最终评测 |
| `reports/stage6/` | 第 6 阶段：CIK-RAG 主实验与认知边界压力测试 |
| `reports/risk_gate/` | 风险闸门：冻结阈值、开发集与最终回测 |

**不进版本库的产物**（能重新生成、体积大，或属运行时状态）：
`.cache/`（DA 调用缓存）、`data/knowledge_base/index/`（向量索引）、
`data/aegisfx/`（投递记录、事件库、检索索引、标签、路由数据）、
`data/inbox/` 的投递内容、`reports/last_run.json`。

---

## 八、边界与诚实性

这个项目里有几处刻意"不好看"但必须保留的地方，写在
[`docs/运行指南.md`](docs/运行指南.md) 第十一节，这里只列结论：

1. **模型常常跑不赢朴素基准**，程序会如实打印这个对比，不掩饰；
2. **知识库里的机制说明是项目自行整理的**，不是学术引用——避免拿论文自己的结论
   去论证论文自己的特征选择（循环论证）；
3. **数据口径存疑处会标出来**（如 `CN_CPI_MoM` 的量级看着像同比），
   处理方式是沿用论文命名 + 注明存疑，不替原作者改口径；
4. **原代码里"看着像 bug 其实是设计"的地方保留原样**，只加注释说明简化在哪；
5. **选不到特征、检索不到证据、算不出 SHAP、模型打不过基准**，每一种都走降级路径
   并如实说明，不会静默吞掉。

**线二的负面结果同样是结论**：三路检索分不出高下、冲突与误差无关、共形没跑赢固定
分位、执行闸门没有改善收益。这些都原样保留在文档和报告里，没有为了让结果好看而
调整基线或评测协议。
