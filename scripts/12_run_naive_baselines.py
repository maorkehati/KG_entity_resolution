"""Run naive and lexical baselines (B0–B4) on processed WDC pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import available_baselines, make_baseline
from src.config import DEFAULT_WDC_VARIANT_ID, OUTPUTS_DIR, PROJECT_ROOT
from src.data_loading import load_processed_variant, processed_variant_dir
from src.evaluation import (
    apply_threshold,
    choose_threshold,
    compute_all_metrics,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run naive/lexical baselines.")
    parser.add_argument("--variant", default=DEFAULT_WDC_VARIANT_ID)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=available_baselines(),
    )
    parser.add_argument("--calibration-split", default="valid")
    parser.add_argument("--eval-splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument(
        "--threshold-objective",
        default="f1",
        choices=[
            "f1",
            "precision",
            "recall",
            "balanced_accuracy",
            "precision_at_min_recall",
            "recall_at_min_precision",
        ],
    )
    parser.add_argument("--min-precision", type=float, default=None)
    parser.add_argument("--min-recall", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _outcome(label: int, pred: int) -> str:
    if label == 1 and pred == 1:
        return "TP"
    if label == 0 and pred == 1:
        return "FP"
    if label == 0 and pred == 0:
        return "TN"
    return "FN"


def _build_pred_df(
    source: pd.DataFrame,
    scores: np.ndarray,
    split: str,
    baseline_name: str,
    threshold: float | str,
    preds: np.ndarray,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "split": split,
            "baseline": baseline_name,
            "pair_id": source["pair_id"].values if "pair_id" in source.columns else None,
            "left_id": source["left_id"].values if "left_id" in source.columns else None,
            "right_id": source["right_id"].values if "right_id" in source.columns else None,
            "label": source["label"].astype(int).values,
            "baseline_score": scores,
            "baseline_pred": preds.astype(int),
            "threshold": threshold,
        }
    )
    out["outcome"] = [
        _outcome(int(l), int(p)) for l, p in zip(out["label"], out["baseline_pred"])
    ]
    for col in (
        "left_title",
        "right_title",
        "left_brand",
        "right_brand",
        "left_price",
        "right_price",
        "pair_text",
    ):
        if col in source.columns:
            out[col] = source[col].values
    return out


def _select_threshold(
    baseline_name: str,
    cal_df: pd.DataFrame,
    cal_scores: np.ndarray,
    objective: str,
    min_precision: float | None,
    min_recall: float | None,
) -> tuple[float | str, dict]:
    y_true = cal_df["label"].astype(int).values
    if baseline_name == "always_negative":
        return "fixed_all_negative", {
            "threshold": "fixed_all_negative",
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "fp": 0,
            "fn": int((y_true == 1).sum()),
        }
    thr, cal_metrics, _ = choose_threshold(
        y_true,
        cal_scores,
        objective=objective,
        min_precision=min_precision,
        min_recall=min_recall,
    )
    return thr, cal_metrics


def main() -> int:
    args = parse_args()

    if not processed_variant_dir(args.variant).exists():
        print("Processed data not found. Run: python scripts/03_prepare_variant.py", file=sys.stderr)
        return 1

    table_dir = OUTPUTS_DIR / "tables" / "baselines" / args.variant
    pred_dir = OUTPUTS_DIR / "predictions" / args.variant / "baselines"
    metrics_path = table_dir / "baseline_metrics.csv"
    if metrics_path.exists() and not args.overwrite:
        print(f"Metrics exist at {metrics_path}. Use --overwrite.", file=sys.stderr)
        return 1

    table_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_processed_variant(args.variant)
    split_dfs = {
        "train": dataset.train.df,
        "valid": dataset.valid.df,
        "test": dataset.test.df,
    }

    for b in args.baselines:
        if b not in available_baselines():
            print(f"Unknown baseline: {b}", file=sys.stderr)
            return 1

    if args.calibration_split not in split_dfs:
        print(f"Unknown calibration split: {args.calibration_split}", file=sys.stderr)
        return 1

    metric_rows: list[dict] = []
    threshold_rows: list[dict] = []

    for baseline_name in args.baselines:
        scorer = make_baseline(baseline_name)
        scorer.fit(split_dfs["train"])

        scores_by_split: dict[str, np.ndarray] = {}
        for split in args.eval_splits:
            if split not in split_dfs:
                continue
            scores_by_split[split] = scorer.score(split_dfs[split])

        cal_scores = scores_by_split[args.calibration_split]
        thr, cal_diag = _select_threshold(
            baseline_name,
            split_dfs[args.calibration_split],
            cal_scores,
            args.threshold_objective,
            args.min_precision,
            args.min_recall,
        )

        threshold_rows.append(
            {
                "variant_id": args.variant,
                "baseline": baseline_name,
                "calibration_split": args.calibration_split,
                "threshold": thr,
                "threshold_objective": args.threshold_objective,
                "calibration_precision": cal_diag.get("precision", np.nan),
                "calibration_recall": cal_diag.get("recall", np.nan),
                "calibration_f1": cal_diag.get("f1", np.nan),
                "calibration_fp": cal_diag.get("fp", np.nan),
                "calibration_fn": cal_diag.get("fn", np.nan),
            }
        )

        for split in args.eval_splits:
            if split not in scores_by_split:
                continue
            df = split_dfs[split]
            scores = scores_by_split[split]
            y_true = df["label"].astype(int).values

            if baseline_name == "always_negative":
                preds = np.zeros(len(df), dtype=int)
                thr_val = "fixed_all_negative"
                metrics = compute_all_metrics(y_true, scores, threshold=1.0)
                metrics["threshold"] = thr_val
            else:
                thr_val = float(thr)
                preds = apply_threshold(scores, thr_val)
                metrics = compute_all_metrics(y_true, scores, threshold=thr_val)

            metrics_row = {
                "variant_id": args.variant,
                "baseline": baseline_name,
                "split": split,
                "threshold": thr_val,
                "threshold_objective": args.threshold_objective,
                **metrics,
            }
            metric_rows.append(metrics_row)

            pred_df = _build_pred_df(df, scores, split, baseline_name, thr_val, preds)
            pred_path = pred_dir / f"{baseline_name}_{split}_predictions.csv"
            pred_df.to_csv(pred_path, index=False)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(metrics_path, index=False)
    pd.DataFrame(threshold_rows).to_csv(table_dir / "baseline_thresholds.csv", index=False)

    test_summary = metrics_df[metrics_df["split"] == "test"].sort_values(
        "f1", ascending=False
    )
    summary_cols = [
        "baseline",
        "threshold",
        "precision",
        "recall",
        "f1",
        "fp",
        "fn",
        "auroc",
        "auprc",
    ]
    test_summary = test_summary[[c for c in summary_cols if c in test_summary.columns]]
    test_summary.to_csv(table_dir / "baseline_test_summary.csv", index=False)

    metadata = {
        "variant_id": args.variant,
        "baselines": args.baselines,
        "calibration_split": args.calibration_split,
        "eval_splits": args.eval_splits,
        "threshold_objective": args.threshold_objective,
        "notes": "Naive and lexical baselines for WDC entity resolution.",
    }
    save_json(metadata, table_dir / "baseline_metadata.json")

    print("NAIVE BASELINE RESULTS\n")
    print(f"Calibration split: {args.calibration_split}")
    print(f"Threshold objective: {args.threshold_objective}\n")
    header = f"{'Baseline':<28} {'Split':<6} {'Threshold':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FP':>6}"
    print(header)
    print("-" * len(header))
    for _, row in metrics_df.iterrows():
        thr_s = str(row["threshold"])[:12]
        print(
            f"{row['baseline']:<28} {row['split']:<6} {thr_s:<12} "
            f"{row['precision']:10.4f} {row['recall']:8.4f} {row['f1']:8.4f} {int(row['fp']):6d}"
        )

    print(f"\nSaved metrics:\n  {metrics_path.relative_to(PROJECT_ROOT)}")
    print(
        f"Saved test summary:\n  "
        f"{(table_dir / 'baseline_test_summary.csv').relative_to(PROJECT_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
