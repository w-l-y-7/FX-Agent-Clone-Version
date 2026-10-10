"""AegisFX-Agent：冲突感知 RAG + 共形预测 + 执行风险闸门。

这个包是**复现工作之外的新增部分**，与 `src/core`、`src/agents` 里那套
FX-Agents 复现代码没有耦合。边界写在这里，方便日后分清哪些是复现、哪些是新做的：

- `src/core`、`src/agents`、`src/services`、`src/legacy_models` —— 论文复现，不动；
- `src/aegisfx` —— 本项目新增，可以自由改。

方案见 `docs/AegisFX-Agent方案.md`。
"""
