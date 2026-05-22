"""Iterative symbolically guided hard-negative reweighting for the pairwise scorer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_WDC_VARIANT_ID, OUTPUTS_DIR
from src.data_loading import load_processed_variant
from src.decision import DecisionConfig, apply_governed_decision
from src.evaluation import compute_binary_metrics_from_predictions, save_json
from src.features import PairFeatureConfig
from src.hard_negative_mining import (
    identify_symbolic_hard_negatives,
    parse_list_column,
    summarize_hard_negatives,
    update_sample_weights,
)
from src.scorer import PairwiseMatchScorer, ScorerConfig
from src.symbolic import ProductConstraintChecker, lists_to_json_columns, make_symbolic_config

EXAMPLE_COLS = [
    "iteration",
    "pair_id",
    "label",
    "previous_neural_score",
    "violated_constraints",
    "left_title",
    "right_title",
    "left_brand",
    "right_brand",
    "left_price",
    "right_price",
]

METRIC_COLS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "tp",
    "fp",
    "tn",
    "fn",
    "positive_rate_pred",
    "auroc",
    "auprc",
    "average_precision",
    "brier_score",
    "log_loss",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterative symbolic hard-negative training experiment."
    )
    parser.add_argument("--variant", default=DEFAULT_WDC_VARIANT_ID)
    parser.add_argument("--base-run-name", default="neural_logreg")
    parser.add_argument(
        "--experiment-name",
        default="symbolic_hard_negative_training",
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--score-threshold", type=float, default=0.66)
    parser.add_argument("--hard-negative-weight", type=float, default=3.0)
    parser.add_argument("--max-weight", type=float, default=10.0)
    parser.add_argument(
        "--constraint-mode",
        default="strong_only",
        choices=["strong_only", "all_conservative"],
    )
    parser.add_argument(
        "--symbolic-profile",
        default="conservative",
        choices=["conservative", "moderate"],
    )
    parser.add_argument("--threshold", type=float, default=0.66)
    parser.add_argument("--selection-split", default="valid")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-examples", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _merge_symbolic(df: pd.DataFrame, symbolic: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sym = symbolic.drop(columns=["pair_id"], errors="ignore")
    for col in sym.columns:
        out[col] = sym[col].values
    return out


def _evaluate_split(
    df: pd.DataFrame,
    decision_type: str,
    threshold: float,
) -> dict[str, Any]:
    if decision_type == "neural_only":
        preds = (df["neural_score"] >= threshold).astype(int).values
        scores = df["neural_score"].astype(float).values
    else:
        governed = apply_governed_decision(
            df,
            DecisionConfig(decision_mode="invalid_blocks", threshold=threshold),
        )
        preds = governed["governed_pred"].astype(int).values
        scores = df["neural_score"].astype(float).values
    y_true = df["label"].astype(int).values
    return compute_binary_metrics_from_predictions(
        y_true, preds, scores, threshold
    )


def _save_predictions(
    df: pd.DataFrame,
    iteration: int,
    split: str,
    threshold: float,
    path: Path,
) -> None:
    work = apply_governed_decision(
        df,
        DecisionConfig(decision_mode="invalid_blocks", threshold=threshold),
    )
    out = pd.DataFrame(
        {
            "iteration": iteration,
            "split": split,
            "pair_id": work.get("pair_id"),
            "label": work["label"],
            "neural_score": work["neural_score"],
            "neural_pred": (work["neural_score"] >= threshold).astype(int),
            "symbolic_status": work.get("symbolic_status"),
            "violated_constraints": work.get("violated_constraints"),
            "governed_pred": work["governed_pred"],
            "left_title": work.get("left_title"),
            "right_title": work.get("right_title"),
            "left_brand": work.get("left_brand"),
            "right_brand": work.get("right_brand"),
            "left_price": work.get("left_price"),
            "right_price": work.get("right_price"),
        }
    )
    if "violated_constraints" in out.columns:
        out = lists_to_json_columns(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _save_figure(fig: plt.Figure, png: Path, pdf: Path) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_iteration_curves(metrics: pd.DataFrame, fig_dir: Path) -> None:
    test = metrics[metrics["split"] == "test"].copy()
    if test.empty:
        return

    iterations = sorted(test["iteration"].unique())

    # Plot 1: test F1
    fig, ax = plt.subplots(figsize=(7, 5))
    for dtype, label, color in [
        ("neural_only", "Neural-only", "#4d4d4d"),
        ("hard_governed_invalid_blocks", "Hard governed", "#1f77b4"),
    ]:
        sub = test[test["decision_type"] == dtype].sort_values("iteration")
        ax.plot(sub["iteration"], sub["f1"], marker="o", label=label, color=color)
    ax.set_xticks(iterations)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Test F1")
    ax.set_title("Symbolic hard-negative training: test F1 over iterations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(
        fig,
        fig_dir / "test_f1_over_iterations.png",
        fig_dir / "test_f1_over_iterations.pdf",
    )

    # Plot 2: precision/recall
    fig, ax = plt.subplots(figsize=(7, 5))
    for dtype, label, color, col in [
        ("neural_only", "Neural precision", "#4d4d4d", "precision"),
        ("neural_only", "Neural recall", "#888888", "recall"),
        ("hard_governed_invalid_blocks", "Governed precision", "#1f77b4", "precision"),
        ("hard_governed_invalid_blocks", "Governed recall", "#6baed6", "recall"),
    ]:
        sub = test[test["decision_type"] == dtype].sort_values("iteration")
        ls = "-" if "precision" in label else "--"
        ax.plot(sub["iteration"], sub[col], marker="o", label=label, color=color, linestyle=ls)
    ax.set_xticks(iterations)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Metric value")
    ax.set_title("Precision/recall trade-off over hard-negative iterations")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(
        fig,
        fig_dir / "test_precision_recall_over_iterations.png",
        fig_dir / "test_precision_recall_over_iterations.pdf",
    )

    # Plot 3: FP
    fig, ax = plt.subplots(figsize=(7, 5))
    for dtype, label, color in [
        ("neural_only", "Neural-only", "#d9534f"),
        ("hard_governed_invalid_blocks", "Hard governed", "#1f77b4"),
    ]:
        sub = test[test["decision_type"] == dtype].sort_values("iteration")
        ax.plot(sub["iteration"], sub["fp"], marker="o", label=label, color=color)
    ax.set_xticks(iterations)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("False positives")
    ax.set_title("False accepted merges over hard-negative iterations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_figure(
        fig,
        fig_dir / "test_fp_over_iterations.png",
        fig_dir / "test_fp_over_iterations.pdf",
    )


def plot_mining_curve(mining: pd.DataFrame, fig_dir: Path) -> None:
    sub = mining[mining["iteration"] > 0].sort_values("iteration")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(sub["iteration"], sub["num_hard_negatives"], color="#5cb85c", edgecolor="black")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Hard negatives mined from training")
    ax.set_title("Symbolically mined hard negatives per iteration")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_figure(
        fig,
        fig_dir / "mined_hard_negatives_over_iterations.png",
        fig_dir / "mined_hard_negatives_over_iterations.pdf",
    )


def print_summary(
    metrics: pd.DataFrame,
    mining: pd.DataFrame,
    *,
    threshold: float,
    score_threshold: float,
    constraint_mode: str,
    table_dir: Path,
    fig_dir: Path,
) -> None:
    print("SYMBOLIC HARD-NEGATIVE TRAINING COMPLETE")
    print()
    print(f"Fixed decision threshold: {threshold}")
    print(f"Hard-negative mining threshold: {score_threshold}")
    print(f"Constraint mode: {constraint_mode}")
    print()
    print("Test metrics by iteration:")
    print(
        "Iteration | Neural F1 | Neural Prec | Neural Rec | Neural FP | "
        "Gov F1 | Gov Prec | Gov Rec | Gov FP"
    )
    test = metrics[metrics["split"] == "test"].sort_values("iteration")
    for it in sorted(test["iteration"].unique()):
        n = test[(test["iteration"] == it) & (test["decision_type"] == "neural_only")]
        g = test[
            (test["iteration"] == it)
            & (test["decision_type"] == "hard_governed_invalid_blocks")
        ]
        if n.empty or g.empty:
            continue
        nr, gr = n.iloc[0], g.iloc[0]
        print(
            f"{int(it):9d} | {nr['f1']:9.4f} | {nr['precision']:11.4f} | "
            f"{nr['recall']:10.4f} | {int(nr['fp']):9d} | "
            f"{gr['f1']:6.4f} | {gr['precision']:8.4f} | {gr['recall']:7.4f} | {int(gr['fp']):6d}"
        )
    print()
    print("Mined hard negatives:")
    print("Iteration | Count | Mean score | Top constraints")
    msub = mining[mining["iteration"] > 0].sort_values("iteration")
    for _, row in msub.iterrows():
        counts = json.loads(row["constraint_counts_json"])
        top = ", ".join(
            f"{k}({v})" for k, v in sorted(counts.items(), key=lambda x: -x[1])[:3]
        )
        print(
            f"{int(row['iteration']):9d} | {int(row['num_hard_negatives']):5d} | "
            f"{row['mean_hard_negative_score']:10.4f} | {top or 'n/a'}"
        )
    print()
    print("Saved:")
    print(f"  {table_dir / 'iteration_metrics.csv'}")
    print(f"  {table_dir / 'mining_summary.csv'}")
    print(f"  {fig_dir}/")


def main() -> int:
    args = parse_args()
    if args.iterations < 0:
        print("--iterations must be >= 0", file=sys.stderr)
        return 1

    variant_id = args.variant
    experiment = args.experiment_name
    table_dir = OUTPUTS_DIR / "tables" / "hard_negative_training" / variant_id / experiment
    pred_dir = OUTPUTS_DIR / "predictions" / variant_id / experiment
    model_dir = OUTPUTS_DIR / "models" / variant_id / experiment
    fig_dir = OUTPUTS_DIR / "figures" / "hard_negative_training" / variant_id / experiment
    meta_path = table_dir / "metadata.json"

    if meta_path.exists() and not args.overwrite:
        print("Outputs exist; use --overwrite to regenerate.", file=sys.stderr)
        metrics = pd.read_csv(table_dir / "iteration_metrics.csv")
        mining = pd.read_csv(table_dir / "mining_summary.csv")
        print_summary(
            metrics,
            mining,
            threshold=args.threshold,
            score_threshold=args.score_threshold,
            constraint_mode=args.constraint_mode,
            table_dir=table_dir,
            fig_dir=fig_dir,
        )
        return 0

    dataset = load_processed_variant(variant_id)
    splits = {
        "train": dataset.train.df,
        "valid": dataset.valid.df,
        "test": dataset.test.df,
    }
    checker = ProductConstraintChecker(make_symbolic_config(args.symbolic_profile))
    eval_splits = ["train", "valid", "test"]

    metric_rows: list[dict[str, Any]] = []
    mining_rows: list[dict[str, Any]] = []
    example_rows: list[pd.DataFrame] = []

    sample_weight: np.ndarray | None = None
    max_iter = args.iterations  # 0..iterations inclusive

    for iteration in range(max_iter + 1):
        print(f"\n=== Iteration {iteration} ===")

        scorer = PairwiseMatchScorer(
            config=ScorerConfig(
                variant_id=variant_id,
                model_type="logreg",
                class_weight=None if sample_weight is not None else "balanced",
                random_state=args.random_state,
            ),
            feature_config=PairFeatureConfig(),
        )
        if sample_weight is not None:
            scorer.fit(splits["train"], sample_weight=sample_weight)
        else:
            scorer.fit(splits["train"])

        model_path = model_dir / f"iteration_{iteration}_model.joblib"
        model_dir.mkdir(parents=True, exist_ok=True)
        scorer.save(model_path)
        print(f"Saved model: {model_path}")

        scored_by_split: dict[str, pd.DataFrame] = {}
        for split_name, split_df in splits.items():
            sym = checker.check_dataframe(split_df)
            merged = _merge_symbolic(split_df, sym)
            merged["neural_score"] = scorer.predict_proba(split_df)
            scored_by_split[split_name] = merged

            pred_path = pred_dir / f"iteration_{iteration}_{split_name}_predictions.csv"
            _save_predictions(merged, iteration, split_name, args.threshold, pred_path)

            for decision_type in ("neural_only", "hard_governed_invalid_blocks"):
                m = _evaluate_split(merged, decision_type, args.threshold)
                metric_rows.append(
                    {
                        "variant_id": variant_id,
                        "experiment_name": experiment,
                        "iteration": iteration,
                        "split": split_name,
                        "decision_type": decision_type,
                        "threshold": args.threshold,
                        "constraint_mode": args.constraint_mode,
                        "score_threshold": args.score_threshold,
                        "hard_negative_weight": args.hard_negative_weight,
                        **{k: m[k] for k in METRIC_COLS},
                    }
                )

        if iteration < max_iter:
            train_scored = scored_by_split["train"]
            hn_mask = identify_symbolic_hard_negatives(
                train_scored,
                score_threshold=args.score_threshold,
                constraint_mode=args.constraint_mode,
            ).values

            summary = summarize_hard_negatives(train_scored, hn_mask)
            sample_weight = update_sample_weights(
                sample_weight,
                hn_mask,
                hard_negative_weight=args.hard_negative_weight,
                cumulative=True,
                max_weight=args.max_weight,
            )

            mining_rows.append(
                {
                    "variant_id": variant_id,
                    "experiment_name": experiment,
                    "iteration": iteration + 1,
                    "num_hard_negatives": summary["num_hard_negatives"],
                    "hard_negative_rate": summary["hard_negative_rate"],
                    "mean_hard_negative_score": summary["mean_score_hard_negatives"],
                    "min_hard_negative_score": summary["min_score_hard_negatives"],
                    "max_hard_negative_score": summary["max_score_hard_negatives"],
                    "mean_sample_weight": float(sample_weight.mean()),
                    "max_sample_weight": float(sample_weight.max()),
                    "constraint_counts_json": json.dumps(summary["constraint_counts"]),
                }
            )

            examples = train_scored.loc[hn_mask].copy()
            if not examples.empty:
                examples = examples.sort_values("neural_score", ascending=False).head(
                    args.max_examples
                )
                ex = pd.DataFrame(
                    {
                        "iteration": iteration + 1,
                        "pair_id": examples["pair_id"],
                        "label": examples["label"],
                        "previous_neural_score": examples["neural_score"],
                        "violated_constraints": examples["violated_constraints"].map(
                            lambda x: json.dumps(parse_list_column(x))
                        ),
                        "left_title": examples.get("left_title"),
                        "right_title": examples.get("right_title"),
                        "left_brand": examples.get("left_brand"),
                        "right_brand": examples.get("right_brand"),
                        "left_price": examples.get("left_price"),
                        "right_price": examples.get("right_price"),
                    }
                )
                example_rows.append(ex)

            print(
                f"Mined {summary['num_hard_negatives']} hard negatives "
                f"for iteration {iteration + 1} "
                f"(rate={summary['hard_negative_rate']:.4f})"
            )

    metrics_df = pd.DataFrame(metric_rows)
    mining_df = pd.DataFrame(mining_rows)
    examples_df = (
        pd.concat(example_rows, ignore_index=True)
        if example_rows
        else pd.DataFrame(columns=EXAMPLE_COLS)
    )

    table_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(table_dir / "iteration_metrics.csv", index=False)
    mining_df.to_csv(table_dir / "mining_summary.csv", index=False)
    examples_df.to_csv(table_dir / "mined_hard_negative_examples.csv", index=False)

    metadata = {
        "variant_id": variant_id,
        "experiment_name": experiment,
        "base_run_name": args.base_run_name,
        "iterations": args.iterations,
        "score_threshold": args.score_threshold,
        "hard_negative_weight": args.hard_negative_weight,
        "max_weight": args.max_weight,
        "constraint_mode": args.constraint_mode,
        "symbolic_profile": args.symbolic_profile,
        "threshold": args.threshold,
        "random_state": args.random_state,
        "notes": "Iterative symbolic hard-negative reweighting experiment.",
    }
    save_json(metadata, meta_path)

    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_iteration_curves(metrics_df, fig_dir)
    plot_mining_curve(mining_df, fig_dir)

    print_summary(
        metrics_df,
        mining_df,
        threshold=args.threshold,
        score_threshold=args.score_threshold,
        constraint_mode=args.constraint_mode,
        table_dir=table_dir,
        fig_dir=fig_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
