# FX-Agents 复现工作

本仓库是论文 **《A Novel Exchange Rate Forecasting Paradigm Based on Multi-Agent
Collaboration and Multimodal Big Data-Driven Methods》**（Di Han 等，*Big Data
Mining and Analytics* 9(4), 2026, pp. 1009–1025）公开代码
<https://github.com/Kon-Kwok/FX-Agent> 的克隆，并在此基础上完成了复现工作。
论文 PDF 在仓库根目录。

![FX-Agents 框架](./figure1.png)
*图 1：FX-Agents 的多智能体协作流程（论文原图）。本仓库复现的对象就是这条流水线。*

---

## 一句话结论

**框架部分已完整重构并能端到端跑通；实验部分没能还原。**

原因很具体：论文公开了数据和框架，但**没有公开跑实验的那套脚本**——公开仓库
现在的 `main` 分支是被精简过的，模型文件只剩造随机数据的模板；作者真正写过的东西
（Optuna 调参、FAISS 向量检索、RNN 与 LSTM+Attention 的网络结构）在 2025-07-16 的
提交 `a838298` 里被删掉了，**已从 git 历史恢复并保存在 `author_original_code/`**。

仍然缺的是把那些零件串起来、真正跑出论文那次实验的脚本。所以 Table 7 那组精度
无法复现，本仓库是按论文正文 + 从历史恢复的实现重新搭的一套。

**请不要把本仓库当作"论文实验的复现"来读**，它是"按论文正文做的一个合理实现"。

---

## 做了什么：逐项对照

| 论文里的东西 | 状态 | 验证方式 |
|---|---|---|
| 四智能体流程 PA1 → PA2 → DA → FA | ✅ 完整实现，端到端跑通 | `python main.py` |
| DA 的证据三步循环（检索 / 自评 / 迭代精炼） | ✅ 实现，含 RAG 与磁盘缓存 | `scripts/run_da_selection.py` |
| PA2：新闻 → 日频事件特征聚合 | ✅ 与论文口径一致 | `scripts/verify_pa2_aggregation.py` |
| FA：Optuna 调参 + SHAP 解释 + LLM 解读 | ✅ 两条路都实现了：深度模型（作者的搜索空间）/ 表格模型（快，带 SHAP） | `scripts/optimize_deep_model.py`、`python main.py --optimize` |
| **Table 7 的全部 5 个模型** | ✅ 全部实现并跑出结果；**RNN 与 LSTM+Attention 的网络结构取自作者被删除的原始实现** | 见下文「模型结果」 |
| **§4.4 传统特征工程基线** | ✅ 实现，Table 8 的 4 个特征全部算得出来 | `scripts/verify_traditional_baseline.py` |
| Table 7 的精度 | ❌ 复现不出 | 见下文「为什么复现不出」 |
| PA1 实采 1030 条事件 | ⚠️ 改为读现成的 `Data.xlsx` | 历史新闻源无法重新采集 |
| 论文的知识库（学术文献 + 官方报告） | ⚠️ 换成 4 份自行整理的机制说明 | 说明写在 `data/knowledge_base/README.md` |

---

## 能对上论文的部分

这些是跑出来的实测结果，不是照抄的：

**数据层**
- `Data.xlsx` 里 25 个事件列的求和，与论文 Table 5 的 sample size 列**逐列吻合**
  （248 / 181 / 215 / 160 / 114 / 45 / 39 / 34 / 1 / 1）
- 2017-02-28 那一行与论文 Table 4 **逐位吻合**
- 论文 §4.2 说"落在非交易日的事件顺延到之后最近的交易日"——用三种口径实测，
  `next` 口径最优（25 列平均交并比 0.948）

**PA2 事件聚合**
- 论文说 PA2 过滤后剩 **429 条**。实测：`Data` 的 25 个事件列按行求或，
  **恰好 429 天**不为零——429 是"有事件发生的交易日数"，不是新闻条数
- `Sheet1` 的 445 条新闻按同样规则折叠，覆盖 **426 个**交易日，只差 3 天

**DA 特征筛选**
- 32 个候选特征全部评完，与论文 Table 8 的 4 个特征**交集 3 个**
- 排序结构对得上：论文 Table 6 排第一的 `Negative events`，本次也是第一；
  论文垫底（8 分）的 `US removal from currency manipulator list`，本次也是最后一名

**传统特征工程基线（论文 §4.4 的对照组）**
- 时序线（Pearson 相关 + 0.7 去重）：`{USD_Index, CN_1Y_GovBond_Yield}`，
  与论文 Table 8 **完全一致**
- 事件线（XGBoost 重要度 + 均值阈值）：论文的 2 个特征正是算出来的**前两名**，
  且 Table 8 的 4 个特征**全部**出现在算出的集合里

---

## 模型结果（HORIZON=20，论文 Table 7 的 `FA(χ)+PA1+PA2+DA` 列）

| 模型 | 本机 RMSE | 本机 MAPE | 论文 RMSE | 论文 MAPE |
|---|---|---|---|---|
| Transformer | 0.3238 | 3.8719% | 0.0566 | 0.6532% |
| LSTM + Attention | 0.2667 | 3.4820% | 0.0599 | 0.7205% |
| RNN | 0.2506 | 3.2873% | 0.0650 | 0.8121% |
| TFT | 0.3721 | 4.9051% | 0.0330 | 0.3427% |
| TimesNet | 0.2035 | 2.6077% | 0.0436 | 0.4874% |
| **朴素基准（价格不变）** | **0.0813** | **0.8438%** | — | — |

**五个模型都没跑赢朴素基准**，而论文里它们分别赢 30%~60%。差距的来源见下。

---

## 为什么复现不出 Table 7

**公开仓库是"被精简过"的版本。** 现在 `main` 分支上的 `src/legacy_models/` 里，
三个模型文件的驱动代码都只造随机数据：

```python
sample_data = {
    'feature1': np.random.uniform(low=0, high=100, size=200),
    'feature2': np.random.uniform(low=50, high=150, size=200),
    'target_variable': np.sin(np.linspace(0, 10, 200)) * 50 + np.random.normal(0, 5, 200)
}
FEATURES = ['feature1', 'feature2']   # 占位
TARGET = 'target_variable'            # 占位
```

**但作者真正写过的东西在 git 历史里还在。** 2025-07-16 的提交 `a838298` 把
`src/models/`、`src/rag/`、`src/utils/` 三个目录删掉了，被删的内容包括：

| 被删的文件 | 内容 | 本项目怎么处理的 |
|---|---|---|
| `src/utils/hyperparameter_optimizer.py` | **真的 Optuna**：搜索 `learning_rate` / `hidden_dim` / `num_layers` / `dropout`，**`batch_size = 32`**，每 trial 50 轮 | 按它实现了 `src/core/deep_optimization.py` |
| `src/rag/knowledge_base_handler.py` | **真的向量 RAG**：SentenceTransformer + FAISS | 作为参考（本项目用自己的三层降级实现） |
| `src/models/RNN.py`、`LSTM_Attention.py` | **真的网络结构**（驱动代码是模板） | 本项目的这两个模型已按作者结构对齐 |
| `src/utils/event_extractor.py`、`model_explainer.py` | **模拟桩**（返回随机事件 / 假的 SHAP 值） | 本项目另做了真实现 |

原件全部保存在 `author_original_code/`，逐个标注了来源提交，没有改动一个字符。

**所以仍然缺的是**：把上面这些零件串起来、真正跑出论文那次实验的脚本。
论文只交代了 4 个核心超参数（隐藏维度 64、dropout 0.1、学习率 0.001、300 轮），
其余——**优化器、学习率调度、早停、预测步长 HORIZON、切分比例、目标变量口径、
随机种子**——都没有写。**`batch_size = 32` 是唯一从作者代码里捡回来的训练参数。**

---

## 核对时发现的三处待确认问题

1. **公式 (2) 与 Table 6 对不上。** 公式写的是最大最小值归一化，但 Table 6 的
   分数（100 / 92 / 90 / 80 / 8）只有按理论满分 5.0 归一化才算得出来。
2. **公式 (4) 与 Fig. 6 对不上。** 公式写的是"特征当分裂变量的次数"，但按它算，
   Fig. 6 排第 3 的 `Negative_Events` 会掉到第 20 名；换成 XGBoost 默认的 `gain`
   才对得上。
3. **§4.4「信息重叠」那一步的剔除规则**只有一句描述，没有可执行判据。

另外 **HORIZON 论文没有写明**。可以反推确认的是：论文的 RMSE 与 MAPE 之比约 8.5，
而随机游走下该比值 ≈ 1.25 × 价格 ≈ 8.6，所以目标变量是**人民币计价的价格水平**。

这三条加上缺失的训练参数，都写在 `docs/进度记录.md` 的问题清单里。

---

## 怎么跑

```powershell
# 1. 建虚拟环境并装依赖
python -m venv venv
.\venv\Scripts\pip.exe install -r requirements.txt

# 2. 配置密钥：复制 .env.example 为 .env，填上 DEEPSEEK_API_KEY
#    想按论文口径用 R1，加一行 DEEPSEEK_MODEL="deepseek-reasoner"

# 3. 建知识库索引（第一次要下载约 95MB 的向量模型）
.\venv\Scripts\python.exe scripts\build_knowledge_base.py

# 4. 跑完整流程（先用 --limit 8 试水，约 3 分钟）
.\venv\Scripts\python.exe main.py --limit 8
.\venv\Scripts\python.exe main.py              # 完整 32 个候选特征
.\venv\Scripts\python.exe main.py --optimize   # 换用 Optuna 调参 + SHAP 解释
```

**两个不需要密钥、不联网的验证脚本**（几秒钟，适合直接看结果）：

```powershell
.\venv\Scripts\python.exe scripts\verify_pa2_aggregation.py        # PA2 事件聚合，逐列对照
.\venv\Scripts\python.exe scripts\verify_traditional_baseline.py   # §4.4 传统基线，对照 Table 8
```

只跑 DA 的特征筛选（不跑预测）也可以。它**需要密钥**：第一次跑约 15 分钟、
142 次 API 调用；之后结果会缓存在本机 `.cache/da/`（该目录不进版本库），
同一批特征重跑只花几秒钟、不再产生调用。

```powershell
.\venv\Scripts\python.exe scripts\run_da_selection.py
```

**跑五个深度模型**（`--` 之后的耗时是本机 CPU 实测）：

```powershell
.\venv\Scripts\python.exe src\legacy_models\Timesnet.py         # 约 1 分钟
.\venv\Scripts\python.exe src\legacy_models\RNN.py              # 约 3 分钟
.\venv\Scripts\python.exe src\legacy_models\LSTM_Attention.py   # 约 3 分钟
.\venv\Scripts\python.exe src\legacy_models\Transformer.py      # 约 4 分钟
.\venv\Scripts\python.exe src\legacy_models\TFT.py              # 约 25~30 分钟
```

**按作者的搜索空间调深度模型超参**（论文 §3.3.4 的 FA，10 组约 2 分钟）：

```powershell
.\venv\Scripts\python.exe scripts\optimize_deep_model.py --trials 10
```

---

## 目录结构

```
src/
├── core/                    核心逻辑（不依赖网络，纯计算）
│   ├── research_dataset.py      读 Data.xlsx，修掉原表的四个坑
│   ├── event_aggregation.py     ★ PA2 的复现：新闻 → 日频事件哑变量
│   ├── traditional_baseline.py  ★ 论文 §4.4 的传统特征工程基线
│   ├── da_engine.py             ★ DA 的三步循环 + 打分归一化
│   ├── backtest.py              ★ 两个预测服务共用的回测口径
│   ├── model_optimization.py    ★ FA 的 Optuna 调参 + SHAP 解释
│   ├── sequence_dataset.py      给深度模型准备序列数据（防信息泄漏）
│   └── feature_engineering.py   技术指标特征（另一条线用）
├── services/                对外的能力（会调网络/模型）
│   ├── vector_rag_service.py           向量检索，三层降级
│   ├── deepseek_llm_service.py         大模型调用
│   ├── sklearn_forecasting_service.py  Ridge，固定超参
│   └── optimized_forecasting_service.py ★ Optuna 调参 + SHAP
├── agents/                  四个智能体
├── tools/                   数据获取工具
└── legacy_models/           论文 Table 7 的全部五个模型
    ├── Transformer.py / Timesnet.py / TFT.py    已补全，能跑
    └── RNN.py / LSTM_Attention.py               ★ 结构取自作者的原始实现

data/knowledge_base/notes/   ★ DA 检索用的机制说明（项目自行整理）
scripts/                     独立跑的脚本
docs/                        中文补充文档，见下
```

标 ★ 的是原公开代码里**没有**、本次补上的部分。

---

## 进一步阅读

| 文档 | 内容 |
|---|---|
| [`docs/运行指南.md`](docs/运行指南.md) | 怎么用、每个模块在做什么、哪里和论文对不上以及为什么 |
| [`docs/进度记录.md`](docs/进度记录.md) | 复现到哪一步了、交付边界、待确认问题清单 |
| [`author_original_code/README.md`](author_original_code/README.md) | **从作者 git 历史恢复的原始实现**：每个文件是真实现还是模拟桩、从哪个提交恢复的 |

### 运行结果在哪里看

**`reports/` 里的运行结果都在版本库里**，不用自己跑就能看到：

| 文件 | 内容 |
|---|---|
| `reports/da_selection.json` | DA 的完整评审记录：32 个特征的分数、理由、检索到的证据出处 |
| `reports/pa2_aggregation.json` | PA2 事件聚合的逐列对照结果 |
| `reports/traditional_baseline.json` | 传统基线结果（相关系数、XGBoost 重要度、两种口径对比） |
| `reports/main_pipeline.log` | 完整流程 `main.py` 的运行输出 |
| `reports/main_pipeline_optimize.log` | `main.py --optimize` 的运行输出（含 Optuna 与 SHAP） |
| `reports/*_run.log` | 五个深度模型各自的原始运行日志 |
| `reports/*_predictions.png` | 五个模型预测 vs 实际的曲线图 |

另外三个目录**不进版本库**（能重新生成、体积也大）：`.cache/`（DA 调用缓存）、
`data/knowledge_base/index/`（向量索引）、`reports/last_run.json`（单次流程的完整
状态，3.3MB）。
