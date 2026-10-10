"""冲突指标到底能不能预测误差（第 5 阶段第九节）。

    .\\venv\\Scripts\\python.exe scripts\\evaluate_conflict_signal.py

用 2020–2022 的 **out-of-fold** 预测，**不是训练内拟合误差**。
拿拟合误差做这个检验会把结论做得虚高——模型本来就是照着那些点拟合的。

判据（方案里写死的四条，缺一条就不算成立）：

1. 误差关系方向在多数时间折上一致；
2. 高冲突组误差高于低冲突组；
3. block bootstrap 区间基本稳定（不跨 0）；
4. 结果不是完全由某一个事件族驱动。

**不成立就保留负结果，并且不许把冲突指标拿去给共形区间定尺度。**
这条比得到一个好看的结论重要。
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

DATA_DIR = _PROJECT_ROOT / "data" / "aegisfx" / "routing"
REPORT_DIR = _PROJECT_ROOT / "reports" / "stage5"

HORIZONS = (5, 20)
BOOTSTRAP_SEED = 20260929
BOOTSTRAP_RESAMPLES = 5000

# 主集合（方案第九节）：成对离散度取代 MAD 型
CONFLICT_METRICS = ("directional_conflict", "neutral_mass",
                    "pairwise_dispersion_norm", "effective_sample_size")


def load_oof(horizon: int) -> pd.DataFrame:
    path = DATA_DIR / f"oof_predictions_h{horizon}.csv"
    if not path.exists():
        raise SystemExit(f"找不到 {path}。先跑 scripts/train_stage5_router.py")
    return pd.read_csv(path, encoding="utf-8-sig")


def block_bootstrap_spearman(values: np.ndarray, errors: np.ndarray,
                             months: np.ndarray, *, resamples=BOOTSTRAP_RESAMPLES,
                             seed=BOOTSTRAP_SEED) -> dict:
    """按**月份**分块的 bootstrap。不要在候选行级随机抽样。"""
    groups = defaultdict(list)
    for index, month in enumerate(months):
        groups[month].append(index)
    keys = list(groups)
    if len(keys) < 3:
        return {"n_blocks": len(keys), "low": None, "high": None}

    rng = np.random.default_rng(seed)
    rhos = []
    for _ in range(resamples):
        picked = rng.choice(len(keys), size=len(keys), replace=True)
        index = np.concatenate([groups[keys[i]] for i in picked])
        if len(set(values[index])) < 2 or len(set(errors[index])) < 2:
            continue
        rho, _ = stats.spearmanr(values[index], errors[index])
        if not np.isnan(rho):
            rhos.append(rho)
    if not rhos:
        return {"n_blocks": len(keys), "low": None, "high": None}
    return {
        "n_blocks": len(keys),
        "mean": float(np.mean(rhos)),
        "low": float(np.quantile(rhos, 0.025)),
        "high": float(np.quantile(rhos, 0.975)),
        "share_positive": float(np.mean(np.array(rhos) > 0)),
    }


def analyse_horizon(horizon: int, verbose=True) -> dict:
    frame = load_oof(horizon)
    frame = frame.dropna(subset=["abs_error_e1", "realized_abret"]).copy()
    frame["month"] = frame["query_date"].str[:7]
    errors = frame["abs_error_e1"].to_numpy()   # 冲突指标来自 E1 的诊断
    months = frame["month"].to_numpy()

    result = {
        "horizon": horizon,
        "n": len(frame),
        "n_months": int(frame["month"].nunique()),
        "n_missing_directional_conflict": int(frame["directional_conflict"].isna().sum()),
        "univariate": {},
        "quartiles": {},
        "fold_consistency": {},
        "by_family": {},
    }

    if verbose:
        print(f"\n{'=' * 74}\n期限 {horizon} 日　OOF 样本 {len(frame)} 条，"
              f"{result['n_months']} 个月份块\n{'=' * 74}")
        print(f"方向分歧度缺失（无非中性证据）{result['n_missing_directional_conflict']} 条")

    # ---- 1. Spearman 相关 ----
    if verbose:
        print(f"\n{'指标':<28}{'Spearman ρ':>12}{'p':>10}{'块bootstrap 95%区间':>26}")
        print("-" * 76)
    for metric in CONFLICT_METRICS:
        values = frame[metric].to_numpy(dtype=float)
        mask = ~np.isnan(values)
        if mask.sum() < 10 or len(set(values[mask])) < 2:
            if verbose:
                print(f"{metric:<28}{'样本不足':>12}")
            continue
        rho, p_value = stats.spearmanr(values[mask], errors[mask])
        boot = block_bootstrap_spearman(values[mask], errors[mask], months[mask])
        result["univariate"][metric] = {
            "spearman": float(rho), "p_value": float(p_value),
            "n_used": int(mask.sum()), "bootstrap": boot,
        }
        interval = (f"[{boot['low']:+.3f}, {boot['high']:+.3f}]"
                    if boot["low"] is not None else "—")
        if verbose:
            print(f"{metric:<28}{rho:>12.4f}{p_value:>10.4f}{interval:>26}")

    # ---- 2. 四分位组 ----
    if verbose:
        print(f"\n按指标四分位分组的平均绝对误差")
    for metric in CONFLICT_METRICS:
        values = frame[metric].to_numpy(dtype=float)
        mask = ~np.isnan(values)
        if mask.sum() < 20 or len(set(values[mask])) < 4:
            continue
        try:
            quartile = pd.qcut(values[mask], 4, labels=False, duplicates="drop")
        except ValueError:
            continue
        grouped = {}
        for level in sorted(set(quartile)):
            selected = errors[mask][quartile == level]
            grouped[int(level)] = {
                "n": int(len(selected)),
                "mean_abs_error": float(np.mean(selected)),
            }
        result["quartiles"][metric] = grouped
        if verbose:
            display = "　".join(f"Q{k}={v['mean_abs_error']:.5f}" for k, v in grouped.items())
            spread = (grouped[max(grouped)]["mean_abs_error"]
                      - grouped[min(grouped)]["mean_abs_error"])
            print(f"  {metric:<28} {display}　最高−最低 = {spread:+.5f}")

    # ---- 3. 时间折一致性 ----
    if verbose:
        print("\n误差关系的方向在三个验证年上是否一致")
    for metric in CONFLICT_METRICS:
        directions = []
        per_year = {}
        for year in sorted(frame["fold_valid_year"].unique()):
            subset = frame[frame["fold_valid_year"] == year]
            values = subset[metric].to_numpy(dtype=float)
            subset_errors = subset["abs_error_e1"].to_numpy(dtype=float)
            mask = ~np.isnan(values)
            if mask.sum() < 8 or len(set(values[mask])) < 2:
                continue
            rho, _ = stats.spearmanr(values[mask], subset_errors[mask])
            if not np.isnan(rho):
                directions.append(np.sign(rho))
                per_year[int(year)] = float(rho)
        if not directions:
            continue
        consistent = abs(sum(directions)) == len(directions)
        result["fold_consistency"][metric] = {
            "per_year": per_year, "consistent": bool(consistent),
            "n_years": len(directions),
        }
        if verbose:
            detail = "　".join(f"{y}:{v:+.3f}" for y, v in per_year.items())
            print(f"  {metric:<28} {detail}　一致：{'是' if consistent else '否'}")

    # ---- 4. 按事件族 ----
    if verbose:
        print("\n按事件族重复分析（看结论是不是只由一个族驱动）")
    for metric in ("directional_conflict", "pairwise_dispersion_norm"):
        families = {}
        for family, subset in frame.groupby("event_family"):
            values = subset[metric].to_numpy(dtype=float)
            subset_errors = subset["abs_error_e1"].to_numpy(dtype=float)
            mask = ~np.isnan(values)
            if mask.sum() < 10 or len(set(values[mask])) < 2:
                continue
            rho, _ = stats.spearmanr(values[mask], subset_errors[mask])
            families[family] = {"n": int(mask.sum()), "spearman": float(rho)}
        result["by_family"][metric] = families
        if verbose:
            for family, item in sorted(families.items()):
                print(f"  {metric:<28} {family:<22} n={item['n']:>3}  ρ={item['spearman']:+.3f}")

    return result


def verdict(results: dict) -> dict:
    """按方案里的四条判据给结论。**缺一条就不算成立。**"""
    checks = {}
    for horizon, item in results.items():
        passed = {}
        # 判据 1：方向在多数折一致
        consistency = item["fold_consistency"]
        passed["方向在多数折一致"] = any(
            v["consistent"] for v in consistency.values())
        # 判据 2：高冲突组误差高于低冲突组
        higher = False
        for metric, groups in item["quartiles"].items():
            if len(groups) < 4:
                continue
            top = groups[max(groups)]["mean_abs_error"]
            bottom = groups[min(groups)]["mean_abs_error"]
            if top > bottom:
                higher = True
        passed["高冲突组误差高于低冲突组"] = higher
        # 判据 3：bootstrap 区间不跨 0
        stable = False
        for metric, stat in item["univariate"].items():
            boot = stat.get("bootstrap", {})
            if boot.get("low") is not None and boot["low"] > 0:
                stable = True
        passed["bootstrap 区间不跨 0"] = stable
        # 判据 4：不是单一家族驱动
        multi_family = False
        for metric, families in item["by_family"].items():
            positive = [f for f, v in families.items() if v["spearman"] > 0]
            if len(families) >= 3 and len(positive) >= 2:
                multi_family = True
        passed["不是单一家族驱动"] = multi_family

        checks[horizon] = {
            "criteria": passed,
            "all_passed": all(passed.values()),
            "n_passed": sum(passed.values()),
        }
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="冲突指标与误差的关系检验")
    parser.parse_args()

    print("=" * 74)
    print("冲突指标是否预测误差（用 2020–2022 的 out-of-fold 预测）")
    print("=" * 74)
    print("注意：不是训练内拟合误差。用拟合误差做这个检验会把结论做得虚高。")

    results = {horizon: analyse_horizon(horizon) for horizon in HORIZONS}
    checks = verdict(results)

    print(f"\n{'=' * 74}\n四条判据\n{'=' * 74}")
    for horizon, item in checks.items():
        print(f"\n期限 {horizon} 日：通过 {item['n_passed']}/4")
        for name, ok in item["criteria"].items():
            print(f"  [{'通过' if ok else '不通过'}] {name}")

    overall = all(item["all_passed"] for item in checks.values())
    print(f"\n{'=' * 74}")
    if overall:
        print("结论：四个期限都满足四条判据 → **允许启用 CP1（冲突尺度化共形）**")
    else:
        print("结论：判据未全部满足 → **不许启用 CP1**，")
        print("      共形区间只用 CP0（滚动共形），并把这份负结果如实写进报告。")
    print("=" * 74)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "conflict_signal.json"
    path.write_text(json.dumps({
        "horizons": {str(k): v for k, v in results.items()},
        "checks": {str(k): v for k, v in checks.items()},
        "cp1_allowed": bool(overall),
        "bootstrap": {"seed": BOOTSTRAP_SEED, "resamples": BOOTSTRAP_RESAMPLES,
                      "unit": "月份分块"},
        "caveat": "使用 2020–2022 out-of-fold 预测，不是训练内拟合误差。",
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n报告已写入 {path.relative_to(_PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
