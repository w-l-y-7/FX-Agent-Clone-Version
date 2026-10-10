"""三页演示界面的冒烟测试。

用 Streamlit 自带的 `AppTest` 把脚本真跑一遍——不用起浏览器、不用装额外依赖，
能抓到导入错误、运行时异常、以及「填完表单点提交之后页面炸了」这类问题。

    .\\venv\\Scripts\\python.exe scripts\\verify_app.py

第 1 周的三页里只有投递是真的，所以这里也真的走一遍投递流程（用规则解析，
不花 API 调用），确认结果能渲染出来。
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

_APP = _PROJECT_ROOT / "app"
PASSED, FAILED = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASSED if condition else FAILED).append(name)
    print(f"  [{'通过' if condition else '失败'}] {name}" + (f" —— {detail}" if detail else ""))


def run_page(path: Path, label: str) -> AppTest:
    print(f"\n{label}  ({path.relative_to(_PROJECT_ROOT)})")
    app = AppTest.from_file(str(path), default_timeout=120)
    app.run()
    if app.exception:
        check(f"{label} 无异常", False, str(app.exception[0].value)[:200])
    else:
        check(f"{label} 无异常", True)
    return app


def main() -> int:
    print("=" * 78)
    print("AegisFX-Agent 界面冒烟测试")
    print("=" * 78)

    # ---- 入口（会按 st.navigation 加载默认页）----
    run_page(_APP / "Home.py", "入口 Home")

    # ---- 页 1：事件与信号 ----
    page = run_page(_APP / "pages" / "event_signal.py", "页 1 事件与信号")
    if not page.exception:
        check("页 1 有投递按钮", len(page.button) > 0,
              f"找到 {len(page.button)} 个按钮")
        check("页 1 有消息输入框", len(page.text_area) > 0)

        # 真的投一条，走规则解析
        if page.text_area and page.button:
            page.text_area[0].set_value(
                "美国贸易代表办公室宣布，将对中国进口的电动汽车加征关税至100%")
            if len(page.checkbox):
                page.checkbox[0].set_value(False)  # 关掉大模型，测试不花调用
            page.button[0].click().run()
            check("页 1 投递后无异常", not page.exception,
                  str(page.exception[0].value)[:200] if page.exception else "")

            # 闸门走 st.warning / st.success，研究层走 st.info，不是 markdown——
            # 一开始按 markdown 找，全部落空，是两个断言写错了而不是页面没渲染
            banner = " ".join(b.value for b in list(page.warning) + list(page.success))
            check("页 1 渲染出执行闸门", "执行状态" in banner, banner[:80])

            infos = " ".join(i.value for i in page.info)
            check("页 1 渲染出研究层输出", "研究层状态" in infos)

            metric_values = [str(m.value) for m in page.metric]
            check("页 1 解析出事件类型", "美国对中国加征关税" in metric_values,
                  f"指标值：{metric_values[:4]}")
            check("页 1 记录进审计日志",
                  any("evt_" in e.label for e in page.expander))

    # ---- 页 2：历史证据 ----
    page = run_page(_APP / "pages" / "evidence.py", "页 2 历史证据")
    if not page.exception:
        headings = " ".join([s.value for s in page.subheader]
                            + [m.value for m in page.markdown])
        check("页 2 讲了 as-of 协议", "As-of" in headings)

    # ---- 页 3：风险与模拟交易 ----
    page = run_page(_APP / "pages" / "risk.py", "页 3 风险与模拟交易")
    if not page.exception:
        labels = [b.label for b in page.button]
        check("页 3 有 Kill Switch 按钮", any("Kill Switch" in l for l in labels),
              f"按钮：{labels}")
        text = " ".join(str(x.value) for x in list(page.subheader) + list(page.caption))
        check("页 3 展示风险闸门回测或运行指引", "风险" in text or "闸门" in text)

    # ---- 标注工具（内部页）----
    page = run_page(_APP / "pages" / "labeling.py", "标注工具")
    if not page.exception:
        # 「当前事件」是 subheader，不是 markdown——第一版又按 markdown 找了一遍
        headings = " ".join(s.value for s in page.subheader)
        check("标注页加载出当前事件与候选", "当前事件" in headings)
        # 每条候选两个打分控件：机制相关性 + 总体相关性。
        # 不数控件个数（AppTest 对某些控件的暴露不稳定），改成看两套选项都在不在
        options = set()
        for widget in page.radio:
            for option in getattr(widget, "options", []) or []:
                options.add(str(option))
        check("标注页同时渲染机制相关性与总体相关性两套选项",
              any("政策传导机制" in o for o in options)
              and any("高度相关" in o for o in options),
              f"{len(page.radio)} 个控件，{len(options)} 种选项")
        check("标注页显示主评测范围进度",
              any("主评测" in str(c.value) for c in page.caption)
              or any("主评测" in str(m.value) for m in page.markdown))

    print("\n" + "=" * 78)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for name in FAILED:
            print(f"  失败：{name}")
    print("=" * 78)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
