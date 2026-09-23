# 作者原始实现（从 git 历史恢复）

这个目录里是**论文作者的代码**，不是本项目写的。放在这里是为了留个凭据。

## 它们从哪来

公开仓库 <https://github.com/Kon-Kwok/FX-Agent> 的当前版本已经被精简过：原来的
`src/models/`、`src/rag/`、`src/utils/` 三个目录在 2025-07-16 的提交
**`a838298`（"feat: Implement Temporal Fusion Transformer (TFT) model..."）** 里被
删掉了，只留下 `src/legacy_models/` 那几个模板。

这些文件是从**被删除前**的提交 `c3bccc1` 里恢复的：

```bash
git show c3bccc1:src/utils/hyperparameter_optimizer.py
git show c3bccc1:src/rag/knowledge_base_handler.py
...
```

恢复出来的内容**一个字符都没有改动**，原路径 `src/` 前缀换成了本目录的
`models/` / `rag/` / `utils/`，避免和项目自己的 `src/` 混在一起。

## 里面有什么——真实现和模拟桩要分开算

### 真实实现（可以当参考）

| 文件 | 内容 | 对本复现的意义 |
|---|---|---|
| `utils/hyperparameter_optimizer.py` | **真的 Optuna**：`create_study(direction="minimize")`，搜索 `learning_rate`(1e-4~1e-2)、`hidden_dim`(32~128)、`num_layers`(1~3)、`dropout`(0.1~0.5)；**`batch_size = 32`**、每 trial 50 轮 | 这是论文 §3.3.4"FA 用 Optuna"的实现，也**给出了批次大小**——那一直是文档里列为"论文没交代"的参数 |
| `rag/knowledge_base_handler.py` | **真的向量 RAG**：`SentenceTransformer('all-MiniLM-L6-v2')` + `faiss.IndexFlatL2`，含 build / save / load / search | 对应论文 §3.3.3 的"Vectorized retrieval"，作者的实现和论文描述一致 |
| `rag/retriever.py` | 检索封装，接上面那个 | DA 的证据检索 |
| `models/RNN.py` | RNN 的**网络结构是真的**：`nn.RNN` 直接吃原始特征、`nonlinearity='relu'`、**没有输入投影层** | 本项目的 `src/legacy_models/RNN.py` 已按这个结构对齐 |
| `models/LSTM_Attention.py` | LSTM + Attention 的**网络结构是真的**：注意力是 `Linear(hidden, hidden)` → `tanh` → 与一个**可学习的 attention vector** 做点积 → softmax 加权 | 同上，项目里的实现已对齐 |
| `models/model_factory.py` | 按名字造模型，供 Optuna 调参用（支持 `LSTM` / `Transformer`） | 说明论文调的是**深度模型**，不是表格模型 |
| `utils/data_processor.py` | 一个能跑的通用预处理类（序列切分 + MinMaxScaler），但用的是 yfinance 的 EURUSD 示例数据，特征不存在时会**自动生成随机列** | 是工具，不是论文实验的预处理 |

### 模拟桩（作者自己标注为 simulation）

| 文件 | 内容 |
|---|---|
| `utils/event_extractor.py` | 返回**随机假事件**，函数注释写明"this simulation provides a placeholder" |
| `utils/model_explainer.py` | `time.sleep(3)` 之后返回**假的特征重要度**，注释写明"Simulates the process of model explanation using a tool like SHAP" |

这两个**不是 SHAP / LLM 的真实调用**。本项目里的 SHAP 与事件抽取是在别处真正实现的
（`src/core/model_optimization.py`、`src/core/event_aggregation.py`）。

## 仍然没有的东西

**接 `Data.xlsx`、真正跑出论文 Table 7 那次实验的代码，从来没有公开过。**
上面这些是实现零件，但没有把它们串起来跑论文那组实验的脚本——论文报的
批次之外（优化器、HORIZON、切分比例、随机种子）仍然无从得知。

详见 `docs/进度记录.md` 的「交付边界」一节。
