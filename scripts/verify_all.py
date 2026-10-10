"""一条命令跑完全部检查。

    .\\venv\\Scripts\\python.exe scripts\\verify_all.py

依次跑：事件库 11 项 → 检索层 13 项 → 界面 16 项 → 第 4 阶段 70 项。
**不需要密钥，也不联网**（向量模型读本地缓存，拿不到会自动退到 TF-IDF）。

想跳过某一块用 `--skip`，例如 `--skip app` 只跑逻辑层。
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 使用启动当前脚本的解释器。仓库移动或路径含非 ASCII 字符时，硬编码
# ``venv/Scripts/python.exe`` 会让 Windows 的 venv launcher 找回旧路径而失败。
# 从已激活的环境运行时 ``sys.executable`` 仍然就是该环境的 Python。
PYTHON = Path(sys.executable)

SUITES = [
    ("事件库", "scripts/verify_event_store.py"),
    ("检索层", "scripts/verify_retrieval.py"),
    ("界面", "scripts/verify_app.py"),
    ("第 4 阶段", "scripts/test_aegisfx.py"),
    ("第 5 阶段", "scripts/test_stage5.py"),
    ("第 6 阶段", "scripts/test_stage6.py"),
    ("风险 Agent", "scripts/test_risk.py"),
]

SKIP_KEYS = {
    "store": "事件库",
    "retrieval": "检索层",
    "app": "界面",
    "stage4": "第 4 阶段",
    "stage5": "第 5 阶段",
    "stage6": "第 6 阶段",
    "risk": "风险 Agent",
}


def run(script: str) -> tuple:
    started = time.time()
    env = os.environ.copy()
    # 允许通过包装器加载项目 site-packages 后再运行本脚本；子进程要继承同一
    # 依赖搜索路径，否则父进程能 import、各测试套件却找不到依赖。
    inherited = [entry for entry in sys.path if entry]
    previous = env.get("PYTHONPATH")
    if previous:
        inherited.append(previous)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(inherited))
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(
        [str(PYTHON), "-X", "utf8", str(_PROJECT_ROOT / script)],
        cwd=str(_PROJECT_ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode, output, time.time() - started


def summarise(output: str) -> str:
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("通过 ") and "失败" in stripped:
            return stripped
    return "（没找到结果行）"


def main() -> int:
    parser = argparse.ArgumentParser(description="一条命令跑完全部检查")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=sorted(SKIP_KEYS), help="跳过指定套件")
    args = parser.parse_args()

    skip_names = {SKIP_KEYS[key] for key in args.skip}

    print("=" * 78)
    print("AegisFX-Agent 全量检查")
    print("=" * 78)

    failures = []
    total_seconds = 0.0
    for name, script in SUITES:
        if name in skip_names:
            print(f"\n[{name}] 已跳过")
            continue
        print(f"\n[{name}] {script}")
        code, output, seconds = run(script)
        total_seconds += seconds
        print(f"  {summarise(output)}　（{seconds:.1f} 秒）")
        if code != 0:
            failures.append((name, script, output))

    print("\n" + "=" * 78)
    if failures:
        print(f"{len(failures)} 个套件失败：")
        for name, script, output in failures:
            print(f"\n--- {name}（{script}）---")
            tail = [line for line in output.splitlines() if line.strip()][-20:]
            print("\n".join(tail))
    else:
        print("全部套件通过。")
    print(f"总耗时 {total_seconds:.1f} 秒")
    print("=" * 78)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
