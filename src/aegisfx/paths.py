"""集中管理本项目新建的文件路径，别在别处硬编码相对路径。"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 本项目自己的数据目录，和复现用的 data/knowledge_base/ 分开
DATA_DIR = PROJECT_ROOT / "data" / "aegisfx"

# 投递进来的原始消息，只追加不修改
RAW_EVENTS = DATA_DIR / "raw_events.jsonl"
# 解析后的事件 + 信号 + 闸门判定，只追加不修改
EVENTS = DATA_DIR / "events.jsonl"

# 人工投递的监视目录：把 .json 丢进来就会被读走
INBOX_DIR = PROJECT_ROOT / "data" / "inbox"
INBOX_PROCESSED = INBOX_DIR / "processed"
INBOX_FAILED = INBOX_DIR / "failed"

# Kill Switch：文件存在即代表人工叫停一切下单
KILL_SWITCH = DATA_DIR / "KILL_SWITCH"

# 日频行情。第 1 周只用来取「数据水位」，第 2 周起接异常收益计算。
DAILY_DATA = PROJECT_ROOT / "src" / "legacy_models" / "Data.xlsx"


def ensure_dirs() -> None:
    """建好本项目需要的目录，重复调用无副作用。"""
    for directory in (DATA_DIR, INBOX_DIR, INBOX_PROCESSED, INBOX_FAILED):
        directory.mkdir(parents=True, exist_ok=True)


def load_env() -> None:
    """读项目根目录的 `.env`。**只在入口脚本里调用**，库模块不自动调。

    显式给路径而不是靠 `load_dotenv()` 自己往上找：Streamlit 的工作目录
    不一定是项目根目录，靠默认行为会静默读不到密钥。
    """
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
