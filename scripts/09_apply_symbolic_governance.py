"""Apply symbolic governance to raw neural predictions."""

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

from src.config import DEFAULT_WDC_VARIANT_ID, OUTPUTS_DIR, PROJECT_ROOT
from src.decision import DecisionConfig, apply_governed_decision, compute_governance_diagnostics
from src.evaluation import (
    compute_binary_metrics_from_predictions,
    save_json,
    save_metrics_table,
    threshold_to_str,
)
from src.symbolic import (
    ProductConstraintChecker,
    SymbolicConfig,
    explode_constraint_counts,
    lists_to_json_columns,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply symbolic governance to neural predictions.")
    parser.add_argument("--variant", default=DEFAULT_WDC_VARIANT_ID)
    parser.add_argument("--run-name", default="neural_logreg")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--threshold-selection-json", default=None)
    parser.add_argument(
        "--decision-mode",
        default="invalid_blocks",
        choices=["invalid_blocks", "strict_valid_or_high_confidence"],
    )
    parser.add_argument("--tau-high", type=float, default=0.90)
    parser.add_argument(
        "--uncertain-action",
        default="flag",
        choices=["accept", "flag", "reject"],
    )
    parser.add_argument("--eval-splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--score-column", default="neural_score")
    parser.add_argument("--price-conflict-as-invalid", action="store_true")
    parser.add_argument("--color-conflict-as-invalid", action="store_true")
    parser.add_argument("--bundle-conflict-as-invalid", action="store_true")
    parser.add_argument("--category-conflict-as-invalid", action="store_true")
    return parser.parse_args()


def _resolve_threshold(args: argparse.Namespace) -> float:
    if args.threshold is not None:
        return float(args.threshold)
    if args.threshold_selection_json:
        path = Path(args.threshold_selection_json)
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data["threshold"])
    default_path = (
        OUTPUTS_DIR
        / "tables"
        / "neural_scorer"
        / args.variant
        / args.run_name
        / "threshold_selection.json"
    )
    if default_path.exists():
        data = json.loads(default_path.read_text(encoding="utf-8"))
        return float(data["threshold"])
    raise ValueError(
        "Provide --threshold or --threshold-selection-json, or run validation first."
    )


def _load_raw_predictions(pred_dir: Path, split: str) -> pd.DataFrame:
    path = pred_dir / f"raw_{split}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw predictions: {path}")
    return pd.read_csv(path)


def make_governed_error_analysis(df: pd.DataFrame, max_per_group: int = 30) -> pd.DataFrame:
    work = df.copy()
    neural_pred = (work["neural_score"] >= work["threshold"]).astype(int)

    def _take(mask: pd.Series, sort_col: str, ascending: bool, group: str) -> pd.DataFrame:
        sub = work[mask].sort_values(sort_col, ascending=ascending).head(max_per_group)
        if sub.empty:
            return sub
        out = sub.copy()
        out["analysis_group"] = group
        return out

    frames = [
        _take(
            (work["label"] == 0) & (neural_pred == 1) & (work["symbolic_status"] == "invalid"),
            "neural_score",
            False,
            "neural_fp_blocked_by_symbolic_invalid",
        ),
        _take(
            (work["label"] == 0) & (work["governed_pred"] == 1),
            "neural_score",
            False,
            "governed_false_positive",
        ),
        _take(
            (work["label"] == 1) & (neural_pred == 1) & (work["governed_pred"] == 0)
            & (work["symbolic_status"] == "invalid"),
            "neural_score",
            False,
            "true_positive_blocked_by_symbolic_invalid",
        ),
        _take(
            (work["symbolic_status"] == "uncertain") & (work["neural_score"] >= work["tau_high"]),
            "neural_score",
            False,
            "uncertain_high_score",
        ),
        _take(
            (work["symbolic_status"] == "valid") & (work["final_decision"] == "accept"),
            "neural_score",
            False,
            "valid_high_score_accepted",
        ),
        _take(
            (work["symbolic_status"] == "invalid") & (work["neural_score"] >= work["threshold"]),
            "neural_score",
            False,
            "invalid_high_score_rejected",
        ),
    ]

    cols = [
        "split", "analysis_group", "pair_id", "label", "neural_score",
        "symbolic_status", "violated_constraints", "positive_evidence",
        "uncertain_reasons", "final_decision", "governed_pred", "decision_reason",
        "left_title", "right_title", "left_brand", "right_brand",
        "left_price", "right_price",
    ]
    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    return combined[[c for c in cols if c in combined.columns]]


def main() -> int:
    args = parse_args()
    try:
        threshold = _resolve_threshold(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    pred_dir = OUTPUTS_DIR / "predictions" / args.variant / args.run_name
    table_dir = OUTPUTS_DIR / "tables" / "symbolic_governance" / args.variant / args.run_name
    table_dir.mkdir(parents=True, exist_ok=True)

    sym_config = SymbolicConfig(
        price_conflict_as_invalid=args.price_conflict_as_invalid,
        color_conflict_as_invalid=args.color_conflict_as_invalid,
        bundle_conflict_as_invalid=args.bundle_conflict_as_invalid,
        category_conflict_as_invalid=args.category_conflict_as_invalid,
    )
    checker = ProductConstraintChecker(sym_config)
    decision_config = DecisionConfig(
        decision_mode=args.decision_mode,
        threshold=threshold,
        tau_high=args.tau_high,
        uncertain_action=args.uncertain_action,
    )

    thr_str = threshold_to_str(threshold)
    mode_slug = args.decision_mode

    print("APPLYING SYMBOLIC GOVERNANCE\n")
    print(f"Variant: {args.variant}")
    print(f"Run name: {args.run_name}")
    print(f"Threshold: {threshold:.3f}")
    print(f"Decision mode: {args.decision_mode}")
    if args.decision_mode == "strict_valid_or_high_confidence":
        print(f"Tau high: {args.tau_high}")
        print(f"Uncertain action: {args.uncertain_action}")

    metric_rows: list[dict] = []
    status_frames: list[dict] = []
    constraint_frames: list[pd.DataFrame] = []
    governed_frames: list[pd.DataFrame] = []
    all_governed: list[pd.DataFrame] = []

    for split in args.eval_splits:
        raw_df = _load_raw_predictions(pred_dir, split)
        raw_df["split"] = split

        sym_df = checker.check_dataframe(raw_df)
        merged = raw_df.merge(sym_df, on="pair_id", how="left")
        governed = apply_governed_decision(
            merged, decision_config, score_column=args.score_column
        )
        governed_frames.append(governed)
        all_governed.append(governed)

        out_path = pred_dir / f"governed_{split}_{mode_slug}_thr_{thr_str}.csv"
        lists_to_json_columns(governed).to_csv(out_path, index=False)

        y_true = governed["label"].astype(int).values
        y_score = governed[args.score_column].values
        neural_pred = (y_score >= threshold).astype(int)

        for predictor, pred_col in (
            ("neural_only", neural_pred),
            ("governed", governed["governed_pred"].values),
        ):
            m = compute_binary_metrics_from_predictions(
                y_true, pred_col, y_score=y_score, threshold=threshold
            )
            m["split"] = split
            m["predictor"] = predictor
            m["decision_mode"] = mode_slug
            metric_rows.append(m)

        status_frames.append(
            {
                "split": split,
                "valid": int((governed["symbolic_status"] == "valid").sum()),
                "invalid": int((governed["symbolic_status"] == "invalid").sum()),
                "uncertain": int((governed["symbolic_status"] == "uncertain").sum()),
                "total": len(governed),
            }
        )

        for col, kind in (
            ("violated_constraints", "violation"),
            ("positive_evidence", "positive_evidence"),
            ("uncertain_reasons", "uncertain_reason"),
        ):
            constraint_frames.append(
                explode_constraint_counts(governed, col, split, kind)
            )

    metrics_path = table_dir / f"governed_metrics_{mode_slug}_thr_{thr_str}.csv"
    save_metrics_table(metric_rows, metrics_path)

    pd.DataFrame(status_frames).to_csv(
        table_dir / f"symbolic_status_counts_{mode_slug}_thr_{thr_str}.csv",
        index=False,
    )
    if constraint_frames:
        pd.concat(constraint_frames, ignore_index=True).to_csv(
            table_dir / f"constraint_trigger_counts_{mode_slug}_thr_{thr_str}.csv",
            index=False,
        )

    full_governed = pd.concat(all_governed, ignore_index=True)
    diag = compute_governance_diagnostics(full_governed)
    diag.update(
        {
            "variant_id": args.variant,
            "run_name": args.run_name,
            "threshold": threshold,
            "decision_mode": args.decision_mode,
            "tau_high": args.tau_high,
            "uncertain_action": args.uncertain_action,
        }
    )
    diag_path = table_dir / f"governance_diagnostics_{mode_slug}_thr_{thr_str}.json"
    save_json(diag, diag_path)

    error_df = make_governed_error_analysis(full_governed)
    error_path = table_dir / f"governed_error_analysis_{mode_slug}_thr_{thr_str}.csv"
    error_df.to_csv(error_path, index=False)

    print("\nSplit metrics (governed):")
    gov_metrics = [r for r in metric_rows if r["predictor"] == "governed"]
    header = f"{'Split':<8} {'Precision':>10} {'Recall':>8} {'F1':>8} {'FP':>6} {'FN':>6}"
    print(header)
    print("-" * len(header))
    for row in gov_metrics:
        print(
            f"{row['split']:<8} {row['precision']:10.4f} {row['recall']:8.4f} "
            f"{row['f1']:8.4f} {row['fp']:6d} {row['fn']:6d}"
        )

    print("\nNeural-only vs governed (test):")
    test_rows = [r for r in metric_rows if r["split"] == "test"]
    for row in test_rows:
        print(
            f"  {row['predictor']:<12} P={row['precision']:.4f} R={row['recall']:.4f} "
            f"F1={row['f1']:.4f} FP={row['fp']}"
        )

    print(f"\nInvalid blocks (high neural score): {diag.get('invalid_block_count', 0)}")
    print(f"Saved governed predictions under: {pred_dir.relative_to(PROJECT_ROOT)}")
    print(f"Saved metrics: {metrics_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved diagnostics: {diag_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved error analysis: {error_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
