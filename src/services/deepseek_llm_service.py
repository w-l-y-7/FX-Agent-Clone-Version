import os
from typing import Any, Dict, Type

from pydantic import BaseModel

from langchain_openai import ChatOpenAI

from ..core.abstractions.base_llm import BaseLLM

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# 论文 §4.1 写明底座模型是 DeepSeek-R1（671B）。但这里默认用 `deepseek-chat`，
# 原因不是偷懒，是 API 能力的硬约束：R1 的思考模式**不支持 `tool_choice`**。
# 实测（两个模型各发一次结构化请求）：
#
#   deepseek-chat     + function_calling → OK
#   deepseek-reasoner + function_calling → 400 "Thinking mode does not support
#                                              this tool_choice"
#   deepseek-reasoner + json_mode        → OK
#   deepseek-chat     + json_mode        → OK
#
# 所以按模型能力选结构化输出的实现方式：思考型模型走 `json_mode`，其余走
# 更严格的 `function_calling`。想在论文口径下跑，在 `.env` 里设
# `DEEPSEEK_MODEL=deepseek-reasoner` 即可，代码会自动切过去。
DEFAULT_MODEL = "deepseek-chat"

# 思考型（reasoning）模型，不支持 tool_choice，只能用 json_mode
_REASONING_MODELS = frozenset({"deepseek-reasoner", "deepseek-r1"})


class DeepSeekLLMService(BaseLLM):
    """A real LLM service backed by DeepSeek's OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        model: str = None,
        temperature: float = 0.3,
    ):
        # 模型名可以来自构造参数，也可以来自环境变量，都没给就用默认值
        self._model = model or os.getenv("DEEPSEEK_MODEL") or DEFAULT_MODEL
        self._structured_method = (
            "json_mode" if self._model in _REASONING_MODELS else "function_calling"
        )
        self._client = ChatOpenAI(
            model=self._model,
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL,
            temperature=temperature,
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def cache_namespace(self) -> str:
        """给 DA 的磁盘缓存用的命名空间（见 `da_engine._DiskCache` 的说明）。

        默认模型返回空串——项目里已有的那批缓存是它跑出来的，换一个键就等于
        让它们全部作废，那是实打实花过 API 调用换来的。
        """
        return "" if self._model == DEFAULT_MODEL else self._model

    def invoke(self, prompt: str, config: Dict[str, Any] = None) -> str:
        response = self._client.invoke(prompt)
        return response.content

    def invoke_structured(
        self, prompt: str, schema: Type[BaseModel], config: Dict[str, Any] = None
    ) -> BaseModel:
        # DeepSeek 拒绝默认的 `response_format` json_schema 模式，所以不能用
        # langchain 的默认实现。可用的两条路见文件开头：function_calling（更严格，
        # 但思考型模型用不了）和 json_mode（所有模型都支持）。
        return self._client.with_structured_output(
            schema, method=self._structured_method
        ).invoke(prompt)
