"""Stage 2: apply threshold to raw neural predictions and evaluate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_WDC_VARIANT_ID, OUTPUTS_DIR, PROJECT_ROOT
from src.evaluation import (
    add_thresholded_predictions,
    choose_threshold,
    compute_all_metrics,
    make_error_analysis,
    save_json,
    save_metrics_table,
    threshold_to_str,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate raw neural predictions with a decision threshold."
    )
    parser.add_argument("--variant", default=DEFAULT_WDC_VARIANT_ID)
    parser.add_argument("--run-name", default="neural_logreg")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--calibration-split", default=None)
    parser.add_argument(
        "--objective",
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
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=["train", "valid", "test"],
    )
    parser.add_argument("--score-column", default="neural_score")
    parser.add_argument("--max-error-examples", type=int, default=50)
    return parser.parse_args()


def _pred_dir(variant_id: str, run_name: str) -> Path:
    return OUTPUTS_DIR / "predictions" / variant_id / run_name


def _table_dir(variant_id: str, run_name: str) -> Path:
    return OUTPUTS_DIR / "tables" / "neural_scorer" / variant_id / run_name


def _load_raw_predictions(
    pred_dir: Path,
    split: str,
    score_column: str,
) -> pd.DataFrame:
    path = pred_dir / f"raw_{split}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Raw predictions not found: {path}\n"
            "Run scripts/06_train_neural_scorer.py first."
        )
    df = pd.read_csv(path)
    if score_column not in df.columns:
        raise ValueError(
            f"Score column {score_column!r} missing in {path}. "
            f"Columns: {list(df.columns)}"
        )
    if "label" not in df.columns:
        raise ValueError(f"label column missing in {path}")
    return df


def _print_metrics_table(rows: list[dict]) -> None:
    header = (
        f"{'Split':<8} {'Precision':>10} {'Recall':>8} {'F1':>8} "
        f"{'AUROC':>8} {'AUPRC':>8} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['split']:<8} "
            f"{row['precision']:10.4f} "
            f"{row['recall']:8.4f} "
            f"{row['f1']:8.4f} "
            f"{row.get('auroc', float('nan')):8.4f} "
            f"{row.get('auprc', float('nan')):8.4f} "
            f"{row['tp']:5d} "
            f"{row['fp']:5d} "
            f"{row['tn']:5d} "
            f"{row['fn']:5d}"
        )


def main() -> int:
    args = parse_args()
    pred_dir = _pred_dir(args.variant, args.run_name)
    table_dir = _table_dir(args.variant, args.run_name)
    table_dir.mkdir(parents=True, exist_ok=True)

    if args.threshold is None and args.calibration_split is None:
        print(
            "Provide either --threshold or --calibration-split.",
            file=sys.stderr,
        )
        return 1

    sweep_df: pd.DataFrame | None = None
    calibration_metrics: dict | None = None

    if args.threshold is not None:
        threshold = float(args.threshold)
        threshold_source = "fixed"
        calibration_split = None
    else:
        cal_df = _load_raw_predictions(
            pred_dir, args.calibration_split, args.score_column
        )
        threshold, calibration_metrics, sweep_df = choose_threshold(
            cal_df["label"].astype(int).values,
            cal_df[args.score_column].values,
            objective=args.objective,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
        )
        threshold_source = "calibration_split"
        calibration_split = args.calibration_split
        sweep_path = table_dir / f"threshold_sweep_{calibration_split}.csv"
        sweep_df.to_csv(sweep_path, index=False)

    thr_str = threshold_to_str(threshold)

    print("VALIDATING NEURAL PREDICTIONS\n")
    print(f"Variant: {args.variant}")
    print(f"Run name: {args.run_name}")
    print(f"Threshold: {threshold:.3f}")
    print(f"Threshold source: {threshold_source}")
    if calibration_split:
        print(f"Calibration split: {calibration_split}")
        print(f"Objective: {args.objective}")

    metric_rows: list[dict] = []
    thresholded_frames: list[pd.DataFrame] = []

    for split in args.eval_splits:
        raw_df = _load_raw_predictions(pred_dir, split, args.score_column)
        y_true = raw_df["label"].astype(int).values
        y_score = raw_df[args.score_column].values

        metrics = compute_all_metrics(y_true, y_score, threshold)
        metrics["split"] = split
        metric_rows.append(metrics)

        thr_df = add_thresholded_predictions(raw_df, threshold, args.score_column)
        out_path = pred_dir / f"thresholded_{split}_predictions_thr_{thr_str}.csv"
        thr_df.to_csv(out_path, index=False)
        thresholded_frames.append(thr_df)

    print()
    _print_metrics_table(metric_rows)

    metrics_path = table_dir / f"metrics_thr_{thr_str}.csv"
    save_metrics_table(metric_rows, metrics_path)

    score_only_rows = []
    for row in metric_rows:
        score_only_rows.append(
            {k: row[k] for k in row if k in (
                "split", "auroc", "auprc", "average_precision", "log_loss",
                "brier_score", "score_min", "score_max", "score_mean", "score_std",
            )}
        )
    score_metrics_path = table_dir / f"score_metrics_thr_{thr_str}.csv"
    save_metrics_table(score_only_rows, score_metrics_path)

    selection = {
        "variant_id": args.variant,
        "run_name": args.run_name,
        "threshold_source": threshold_source,
        "threshold": threshold,
        "calibration_split": calibration_split,
        "objective": args.objective if threshold_source == "calibration_split" else None,
        "min_precision": args.min_precision,
        "min_recall": args.min_recall,
        "best_calibration_metrics": calibration_metrics,
        "eval_splits": args.eval_splits,
    }
    selection_path = table_dir / "threshold_selection.json"
    save_json(selection, selection_path)

    all_thr = pd.concat(thresholded_frames, ignore_index=True)
    errors = make_error_analysis(all_thr, max_per_group=args.max_error_examples)
    error_path = table_dir / f"error_analysis_thr_{thr_str}.csv"
    errors.to_csv(error_path, index=False)

    print(f"\nSaved metrics:\n  {metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved score metrics:\n  {score_metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved threshold selection:\n  {selection_path.relative_to(PROJECT_ROOT)}")
    print("Saved thresholded predictions:")
    for split in args.eval_splits:
        p = pred_dir / f"thresholded_{split}_predictions_thr_{thr_str}.csv"
        print(f"  {split}: {p.relative_to(PROJECT_ROOT)}")
    print(f"Saved error analysis:\n  {error_path.relative_to(PROJECT_ROOT)}")
    if sweep_df is not None:
        print(
            f"Saved threshold sweep:\n  "
            f"{(table_dir / f'threshold_sweep_{calibration_split}.csv').relative_to(PROJECT_ROOT)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
