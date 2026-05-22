"""Lexical title-similarity threshold baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_WDC_VARIANT_ID, OUTPUTS_DIR
from src.data_loading import load_processed_variant, processed_variant_dir
from src.evaluation import (
    build_prediction_dataframe,
    choose_threshold,
    compute_all_metrics,
)


def title_similarity_scores(df: pd.DataFrame) -> np.ndarray:
    scores = []
    for _, row in df.iterrows():
        left = row.get("left_title", "")
        right = row.get("right_title", "")
        if pd.isna(left) or pd.isna(right):
            scores.append(0.0)
        else:
            scores.append(fuzz.token_set_ratio(str(left), str(right)) / 100.0)
    return np.asarray(scores, dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lexical title-similarity baseline.")
    parser.add_argument("--variant", default=DEFAULT_WDC_VARIANT_ID)
    parser.add_argument("--threshold-metric", default="f1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    variant_id = args.variant

    if not processed_variant_dir(variant_id).exists():
        print(
            "Processed data not found. Run:\n"
            "  python scripts/03_prepare_variant.py",
            file=sys.stderr,
        )
        return 1

    dataset = load_processed_variant(variant_id)
    splits = {
        "train": dataset.train.df,
        "valid": dataset.valid.df,
        "test": dataset.test.df,
    }

    valid_scores = title_similarity_scores(splits["valid"])
    threshold, diagnostics, _ = choose_threshold(
        splits["valid"]["label"].astype(int).values,
        valid_scores,
        objective=args.threshold_metric
        if args.threshold_metric in ("f1", "precision", "recall", "balanced_accuracy")
        else "f1",
    )

    print(f"\nLEXICAL BASELINE (title token_set_ratio >= threshold)")
    print(f"Selected threshold: {threshold:.2f}")
    print(f"Validation diagnostics: {diagnostics}")

    metric_rows = []
    for split_name, df in splits.items():
        scores = title_similarity_scores(df)
        preds = (scores >= threshold).astype(int)
        metrics = compute_all_metrics(df["label"].values, scores, threshold=threshold)
        metrics["split"] = split_name
        metric_rows.append(metrics)

        if split_name == "test":
            pred_df = build_prediction_dataframe(df, scores, split_name, score_column="lexical_score")
            pred_df["lexical_pred"] = preds
            out_dir = OUTPUTS_DIR / "predictions" / variant_id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "lexical_test_predictions.csv"
            pred_df.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")

    table_dir = OUTPUTS_DIR / "tables" / "lexical_baseline" / variant_id
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(table_dir / "metrics.csv", index=False)
    (table_dir / "threshold_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )

    print(f"\n{'Split':<8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUROC':>8}")
    for row in metric_rows:
        print(
            f"{row['split']:<8} {row['precision']:10.4f} "
            f"{row['recall']:8.4f} {row['f1']:8.4f} {row['auroc']:8.4f}"
        )
    print(f"\nSaved metrics: {table_dir / 'metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
