"""把 HuggingFace 的下载地址指到国内镜像。

`huggingface.co` 在部分网络环境下连不上（本机实测是 50 秒超时），而 `hf-mirror.com`
可用。这个设置必须**在导入 transformers / sentence_transformers 之前**生效，
所以本模块靠「导入即执行」来生效，而不是提供一个需要调用者记得调用的函数。

用法：在程序的**第一个 import** 位置写上

    import src.utils.hf_env  # noqa: F401

之后随便 import 什么模型库都已经走镜像了。
"""

import os

DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"


def configure_hf_endpoint(endpoint: str | None = None, *, force: bool = False) -> str:
    """设置 HF_ENDPOINT 环境变量，返回最终生效的值。

    参数 endpoint 为空时，依次尝试环境变量 `FX_HF_ENDPOINT`、内置镜像地址。
    force=False 时不会覆盖用户已经手动设好的 HF_ENDPOINT。
    """
    resolved = endpoint or os.environ.get("FX_HF_ENDPOINT") or DEFAULT_HF_ENDPOINT
    if force or not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = resolved
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # 缓存目录落在含中文的用户目录下时，huggingface_hub 建不了符号链接，
    # 会刷一大段警告。缓存照常能用，只是多占点磁盘，所以直接静音。
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return os.environ["HF_ENDPOINT"]


configure_hf_endpoint()
