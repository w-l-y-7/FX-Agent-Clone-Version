"""第 5 阶段点预测与共形区间评测。

    # 开发集（2023）——随便跑
    .\\venv\\Scripts\\python.exe scripts\\evaluate_stage5.py --split dev

    # 准留出集（2024）——必须显式确认，而且只能跑一次
    .\\venv\\Scripts\\python.exe scripts\\evaluate_stage5.py --split final --confirm-locked

## 两个隔离

**开发集 2023**：可以反复跑，用来做模型诊断和参数冻结。

**准留出集 2024**：要减掉原 20 条人工查询里属于 2024 的那几条——那些已经被
看过，不能再算完全未见。而且必须带 `--confirm-locked`，且需要 `frozen_config.json`
存在。跑完会写 `final_run_manifest.json`，记录配置 hash、数据 hash、样本 ID、
输出 hash。**后续重跑只用于验证复现，不允许改了配置再覆盖第一次的结果。**

## 必须承认的一件事

研究设计已经受到部分 2023–2024 结果的启发（第 3、4 阶段都看过那批查询），
所以 2024 这一集只能叫**准留出**。真正的外部确认要等 2025–2026 的新事件。
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.aegisfx.conflict import analyse, realized_abret  # noqa: E402
from src.aegisfx.conformal import ScaledConformal, rolling_conformal  # noqa: E402
from src.aegisfx.conformal.rolling import _summarise  # noqa: E402
from src.aegisfx.data.event_store import load_event_store  # noqa: E402
from src.aegisfx.data.maturity import MaturityCalendar  # noqa: E402
from src.aegisfx.data.market import paper_market  # noqa: E402
from src.aegisfx.retrieval.embedder import build_embedder  # noqa: E402
from src.aegisfx.retrieval.index import EventIndex  # noqa: E402
from src.aegisfx.retrieval.queries import load_or_build  # noqa: E402
from src.aegisfx.retrieval.retriever import (  # noqa: E402
    MODE_ADAPTIVE,
    MODE_FINANCIAL,
    MODE_SEMANTIC,
    as_of_pool,
    retrieve_detailed,
)
from src.aegisfx.routing.features import FeatureBuilder  # noqa: E402
from src.aegisfx.routing.gate import GateModel  # noqa: E402
from src.aegisfx.schemas.views import RetrievalEventView  # noqa: E402

REPORT_DIR = _PROJECT_ROOT / "reports" / "stage5"
CONFIG_PATH = REPORT_DIR / "frozen_config.json"
DATA_DIR = _PROJECT_ROOT / "data" / "aegisfx" / "routing"

BOOTSTRAP_SEED = 20260929
BOOTSTRAP_RESAMPLES = 5000
EVIDENCE_TOP_K = 10
# 评测期的多数类方向基准（第 2 周实测），作为方向准确率的地板
MAJORITY_BASELINE = 0.556


def _hash(payload) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        .encode("utf-8")).hexdigest()[:16]


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"找不到 {CONFIG_PATH}。**必须先在 2023 开发集上冻结配置**：\n"
            "  python scripts/lock_stage5_config.py")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    stored = config.get("config_hash")
    payload = {k: v for k, v in config.items()
               if k not in ("locked_at", "config_hash")}
    if _hash(payload) != stored:
        raise SystemExit(
            f"配置 hash 对不上（存的 {stored}，算出来 {_hash(payload)}）。"
            "配置文件被改过——要么改回去，要么重新跑 lock_stage5_config.py。")
    return config


def _split_events(events, split: str, calendar: MaturityCalendar):
    """按切分挑事件。final 要剔掉已被看过的 2024 查询。"""
    if split == "dev":
        picked = [e for e in events if pd.Timestamp(e.date).year == 2023]
        excluded = []
    else:
        seen = {q.event_id for q in load_or_build(events)
                if q.date.startswith("2024")}
        picked = [e for e in events
                  if pd.Timestamp(e.date).year == 2024 and e.event_id not in seen]
        excluded = sorted(seen)
    picked.sort(key=lambda e: e.date)
    return picked, excluded


def _month_blocks(dates: List[str]) -> np.ndarray:
    return np.array([d[:7] for d in dates])


def block_bootstrap_mean(values: np.ndarray, months: np.ndarray, *,
                         resamples: int = BOOTSTRAP_RESAMPLES,
                         seed: int = BOOTSTRAP_SEED) -> dict:
    """按月份分块的 bootstrap 均值区间。"""
    groups: Dict[str, List[int]] = {}
    for index, month in enumerate(months):
        groups.setdefault(month, []).append(index)
    keys = list(groups)
    if len(keys) < 3 or len(values) == 0:
        return {"n_blocks": len(keys), "low": None, "high": None}
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for i in range(resamples):
        picked = rng.choice(len(keys), size=len(keys), replace=True)
        index = np.concatenate([groups[keys[j]] for j in picked])
        means[i] = values[index].mean()
    return {
        "n_blocks": len(keys),
        "mean": float(values.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "share_favourable": float((means < 0).mean()),
        "seed": seed, "resamples": resamples, "unit": "月份分块",
    }


def evaluate_horizon(split, horizon, config, events, index, calendar, builder,
                     verbose=True) -> dict:
    picked, excluded = _split_events(events, split, calendar)
    item = config["horizons"][str(horizon)]
    gate = (GateModel.from_config(item["model_config"])
            if item.get("model_config") else None)

    rows = []
    for event in picked:
        realized = realized_abret(event, horizon)
        if realized is None:
            continue
        pool = as_of_pool(events, event, horizon=horizon, calendar=calendar)
        if len(pool) < 5:
            continue

        results = {}
        for mode in (MODE_SEMANTIC, MODE_FINANCIAL, MODE_ADAPTIVE):
            outcome = retrieve_detailed(event, events, index, mode=mode,
                                        top_k=EVIDENCE_TOP_K, pool=pool,
                                        horizon=horizon)
            results[mode] = (analyse(event, list(outcome.evidence), pool,
                                     horizon=horizon, retrieval_mode=mode),
                             outcome)

        r1, outcome1 = results[MODE_SEMANTIC]
        r2, _ = results[MODE_FINANCIAL]
        r3, _ = results[MODE_ADAPTIVE]
        if r1.predicted_abnormal_return is None:
            continue

        e4_prediction, gate_value = None, None
        if gate is not None and horizon in (5, 20):
            view = RetrievalEventView.from_event(event, include_realized=False)
            features = builder.build(
                view,
                same_family_ratio=outcome1.same_family_ratio,
                semantic_top=[h.components.get("semantic", 0.0)
                              for h in outcome1.evidence],
                mature_count=len(pool),
                conflict_e1=r1.conflict.model_dump(),
                conflict_e2=r2.conflict.model_dump())
            kept = item["features"]
            vector = np.array([[features[name] for name in kept]])
            gate_value = float(gate.gate(vector)[0])
            e4_prediction = float(gate.blend(
                vector, [r1.predicted_abnormal_return],
                [r2.predicted_abnormal_return])[0])

        rows.append({
            "event_id": event.event_id,
            "query_date": str(event.date)[:10],
            "event_family": event.event_family,
            "realized": realized,
            "raw_return": getattr(event, f"ret_{horizon}d", None),
            "e1": r1.predicted_abnormal_return,
            "e2": r2.predicted_abnormal_return,
            "e3": r3.predicted_abnormal_return,
            "e4": e4_prediction,
            "gate": gate_value,
            "directional_conflict": r1.conflict.directional_conflict,
            "neutral_mass": r1.conflict.neutral_mass,
            "pairwise_dispersion_norm": r1.conflict.pairwise_dispersion_norm,
            "effective_sample_size": r1.conflict.effective_sample_size,
            "available_at": calendar.available_index(event.date, horizon),
        })

    if not rows:
        return {"horizon": horizon, "n": 0, "note": "这个切分上没有可用样本"}

    frame = pd.DataFrame(rows)
    months = _month_blocks(frame["query_date"].tolist())
    realized = frame["realized"].to_numpy()

    metrics = {}
    for name in ("e1", "e2", "e3", "e4"):
        values = frame[name].to_numpy(dtype=float)
        mask = ~np.isnan(values)
        if mask.sum() == 0:
            continue
        error = np.abs(values[mask] - realized[mask])
        direction = np.sign(values[mask]) == np.sign(realized[mask])
        metrics[name] = {
            "n": int(mask.sum()),
            "mae": float(np.mean(error)),
            "median_ae": float(np.median(error)),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "direction_accuracy": float(np.mean(direction)),
            "mae_bootstrap": block_bootstrap_mean(error, months[mask]),
        }

    # 与最强基线的配对比较。**最强基线在冻结配置时就由时间验证定好了**，
    # 不是看着开发集或留出集挑的——否则就是把评测集当成了选型集。
    strongest = str(item.get("strongest_baseline", "E1")).lower()
    comparisons = {"strongest_baseline": strongest,
                   "baseline_fold_mean_mae": item.get("baseline_fold_mean_mae")}
    if "e4" in metrics and strongest in metrics:
        mask = ~np.isnan(frame["e4"].to_numpy(dtype=float))
        delta = (np.abs(frame["e4"].to_numpy()[mask] - realized[mask])
                 - np.abs(frame[strongest].to_numpy()[mask] - realized[mask]))
        comparisons["e4_minus_strongest"] = {
            "strongest_baseline": strongest,
            "mean_delta_mae": float(delta.mean()),
            "bootstrap": block_bootstrap_mean(delta, months[mask]),
        }

    # ---- 共形区间 ----
    order = frame.sort_values("query_date")
    predictions = order["e1"].tolist()
    oof_rows = _load_oof(horizon)
    if oof_rows:
        predictions = [float(r["e1_prediction"]) for r in oof_rows] + predictions
        realized_seq = [float(r["realized_abret"]) for r in oof_rows] + \
            order["realized"].tolist()
        available = [calendar.available_index(r["query_date"], horizon) or 10 ** 9
                     for r in oof_rows] + order["available_at"].tolist()
        n_oof = len(oof_rows)
    else:
        realized_seq = order["realized"].tolist()
        available = order["available_at"].tolist()
        n_oof = 0

    cp = {}
    # 对照：固定历史分位区间 —— 用 OOF 残差的分位数定一个固定宽度，之后不再更新。
    # 没有这个对照，「共形有没有用」就无从谈起（它可能只是和一个很宽的固定区间一样好）
    cp["固定历史分位"] = _fixed_quantile_baseline(
        order, n_oof, predictions, realized_seq, available,
        item["conformal"]["targets"], split)
    cp0 = rolling_conformal(predictions, realized_seq, available,
                            targets=config["horizons"][str(horizon)]["conformal"]["targets"],
                            window=item["conformal"]["window"])
    cp["CP0"] = _tail_summary(cp0, n_oof, split)
    gamma = item["conformal"]["gamma"]
    if gamma:
        aci = rolling_conformal(predictions, realized_seq, available,
                                targets=item["conformal"]["targets"],
                                window=item["conformal"]["window"], gamma=gamma)
        cp["CP0+ACI"] = _tail_summary(aci, n_oof, split)
    scaled = ScaledConformal.from_signal_report()
    cp1 = scaled.interval(predictions, realized_seq, available,
                          [{} for _ in predictions],
                          targets=item["conformal"]["targets"],
                          window=item["conformal"]["window"])
    cp["CP1"] = _tail_summary(cp1, n_oof, split)
    cp["CP1"]["degrade_reason"] = scaled.reason

    # 给下一阶段的执行闸门保留逐事件、可审计的输入。这里使用已经算好的
    # CP0 80% 区间，不在风险回测脚本里重新拟合或偷看标签。
    cp0_entries = cp0["per_query"].get(0.8, [])[n_oof:]
    observations = []
    for (_, row), interval in zip(order.iterrows(), cp0_entries):
        observations.append({
            "event_id": row["event_id"],
            "query_date": row["query_date"],
            "event_family": row["event_family"],
            "realized_abret": float(row["realized"]),
            "raw_return": (None if pd.isna(row["raw_return"])
                           else float(row["raw_return"])),
            "prediction": (None if pd.isna(row["e4"]) else float(row["e4"])),
            "e1_prediction": float(row["e1"]),
            "interval_80": ([float(interval.lower), float(interval.upper)]
                            if interval.lower is not None else None),
            "interval_width": (None if interval.width is None
                               else float(interval.width)),
            "n_calibration": int(interval.n_calibration),
            "directional_conflict": (None if pd.isna(row["directional_conflict"])
                                     else float(row["directional_conflict"])),
            "neutral_mass": float(row["neutral_mass"]),
            "pairwise_dispersion_norm": (
                None if pd.isna(row["pairwise_dispersion_norm"])
                else float(row["pairwise_dispersion_norm"])),
            "effective_sample_size": float(row["effective_sample_size"]),
        })

    if verbose:
        print(f"\n{'=' * 78}\n期限 {horizon} 日　切分 {split}　样本 {len(frame)} 条\n{'=' * 78}")
        print(f"{'方法':<8}{'n':>5}{'MAE':>12}{'中位AE':>12}{'RMSE':>12}{'方向准确率':>12}")
        for name, value in metrics.items():
            print(f"{name.upper():<8}{value['n']:>5}{value['mae']:>12.5f}"
                  f"{value['median_ae']:>12.5f}{value['rmse']:>12.5f}"
                  f"{value['direction_accuracy']:>11.1%}")
        print(f"{'多数类基准':<8}{'':>5}{'':>12}{'':>12}{'':>12}{MAJORITY_BASELINE:>11.1%}")
        if "e4_minus_strongest" in comparisons:
            item2 = comparisons["e4_minus_strongest"]
            boot = item2["bootstrap"]
            print(f"\nE4 − 最强基线({strongest.upper()}) 的配对 MAE 差："
                  f"{item2['mean_delta_mae']:+.5f}"
                  f"　95% [{boot['low']:+.5f}, {boot['high']:+.5f}]"
                  if boot["low"] is not None else "")
        print(f"\n{'共形方法':<12}{'目标':>6}{'覆盖':>10}{'均宽':>12}{'中位宽':>12}{'n区间':>8}")
        for name, summary in cp.items():
            for target, value in summary.get("targets", {}).items():
                coverage = value["coverage"]
                mean_width = value["mean_width"]
                median_width = value["median_width"]
                print(f"{name:<12}{target:>6.0%}"
                      f"{(f'{coverage:.1%}' if coverage is not None else '—'):>10}"
                      f"{(f'{mean_width:.5f}' if mean_width is not None else '—'):>12}"
                      f"{(f'{median_width:.5f}' if median_width is not None else '—'):>12}"
                      f"{value['n_with_interval']:>8}")
        if cp["CP1"].get("degrade_reason"):
            print(f"\nCP1：{cp['CP1']['degrade_reason']}")

    return {
        "horizon": horizon, "split": split, "n": len(frame),
        "n_oof_prefix": n_oof,
        "excluded_seen_queries": excluded,
        "majority_baseline": MAJORITY_BASELINE,
        "metrics": metrics, "comparisons": comparisons, "conformal": cp,
        "gate_mean": float(frame["gate"].mean()) if frame["gate"].notna().any() else None,
        "gate_std": float(frame["gate"].std()) if frame["gate"].notna().any() else None,
        "sample_ids": frame["event_id"].tolist(),
        "observations": observations,
    }


def _fixed_quantile_baseline(order, n_prefix, predictions, realized_seq,
                             available, targets, split) -> dict:
    """固定历史分位区间：一度用 OOF 残差定宽度，之后不再更新。

    这是共形方法必须打败的对照。它足够简单，但在一段平稳期里可能比滚动共形还好
    ——毕竟滚动窗口会跟着最近的噪声来回摆。
    """
    if n_prefix < 10:
        return {"targets": {}, "method": "固定历史分位（样本不足，无法建）",
                "degrade_reason": "OOF 残差不足 10 条"}
    from src.aegisfx.conformal.rolling import ConformalResult, conformal_quantile

    prefix_scores = [abs(realized_seq[i] - predictions[i]) for i in range(n_prefix)]
    entries = {target: [] for target in targets}
    for index in range(n_prefix, len(predictions)):
        for target in targets:
            q = conformal_quantile(prefix_scores, 1.0 - target)
            point = predictions[index]
            value = realized_seq[index]
            if np.isinf(q):
                entries[target].append(ConformalResult(
                    target, None, None, None, None, n_prefix))
                continue
            lower, upper = point - q, point + q
            entries[target].append(ConformalResult(
                target, float(lower), float(upper), float(2 * q), float(q),
                n_prefix, covered=bool(lower <= value <= upper), realized=value))
    return {"targets": {target: _summarise(items, target)
                        for target, items in entries.items()},
            "method": "固定历史分位"}


def _tail_summary(result: dict, n_prefix: int, split: str) -> dict:
    """只报告切分那一段的覆盖率——OOF 前缀只是给校准池预热用的。"""
    targets = {}
    for target, entries in result["per_query"].items():
        targets[target] = _summarise(entries[n_prefix:], target)
    return {"targets": targets, "method": result.get("method"),
            "degrade_reason": result.get("degrade_reason", "")}


def _load_oof(horizon: int):
    path = DATA_DIR / f"oof_predictions_h{horizon}.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: r["query_date"])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="第 5 阶段评测")
    parser.add_argument("--split", choices=["dev", "final"], default="dev")
    parser.add_argument("--confirm-locked", action="store_true",
                        help="确认配置已冻结，才允许跑 final")
    parser.add_argument("--horizons", nargs="*", type=int, default=[5, 20])
    args = parser.parse_args()

    if args.split == "final" and not args.confirm_locked:
        print("拒绝运行：final 切分必须显式确认配置已冻结。")
        print("  python scripts/evaluate_stage5.py --split final --confirm-locked")
        return 2

    config = _load_config()
    manifest_path = REPORT_DIR / "final_run_manifest.json"
    if args.split == "final" and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("config_hash") != config["config_hash"]:
            print("拒绝覆盖：已有一次 final 运行，且配置 hash 不同。")
            print(f"  已有 {previous.get('config_hash')}，当前 {config['config_hash']}")
            print("  要重跑请先说明为什么改配置，并手动移走 manifest。")
            return 3

    events = load_event_store()
    calendar = MaturityCalendar.from_market()
    embedder, note = build_embedder([e.text for e in events])
    if note:
        print(f"⚠ {note}")
    index = EventIndex(events, embedder, note=note)
    if not index.load_vectors_from_disk():
        index.build()
    builder = FeatureBuilder(paper_market())

    print("=" * 78)
    print(f"第 5 阶段评测　切分 = {args.split}")
    print("=" * 78)
    print(f"配置 hash {config['config_hash']}")
    if args.split == "final":
        print("⚠ 准留出测试：设计受过 2023–2024 结果启发，不是完全未见样本")

    report = {"split": args.split, "config_hash": config["config_hash"],
              "horizons": {}}
    for horizon in args.horizons:
        report["horizons"][horizon] = evaluate_horizon(
            args.split, horizon, config, events, index, calendar, builder)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"evaluation_{args.split}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                        encoding="utf-8")

    if args.split == "final":
        manifest = {
            "config_hash": config["config_hash"],
            "data_hashes": config.get("data_hashes"),
            "report_hash": _hash(report),
            "split": "final",
            "horizons": list(report["horizons"]),
            "sample_ids": {str(h): v.get("sample_ids", [])
                           for h, v in report["horizons"].items()},
            "note": "首次运行的记录。重跑只用于验证复现，不得改配置后覆盖。",
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"\n最终运行清单已写入 {manifest_path.relative_to(_PROJECT_ROOT)}")

    print(f"报告已写入 {out_path.relative_to(_PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
