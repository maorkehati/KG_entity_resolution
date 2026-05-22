"""Train and compare structured neural scorers vs existing logistic scorer."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
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
from src.evaluation import choose_threshold, compute_all_metrics, save_json
from src.structured_features import (
    StructuredFeatureConfig,
    build_structured_feature_groups,
    group_dims,
)
from src.symbolic import ProductConstraintChecker, make_symbolic_config
from src.torch_models import FieldAttentionMatcher, StructuredTransformerMatcher
from src.torch_training import TorchTrainConfig, predict_torch_matcher, train_torch_matcher

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

MODEL_LABELS = {
    "existing_scorer": "Existing scorer",
    "field_attention": "Field attention",
    "structured_transformer": "Structured transformer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train structured neural matchers.")
    parser.add_argument("--variant", default=DEFAULT_WDC_VARIANT_ID)
    parser.add_argument("--run-name", default="neural_logreg")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["field_attention", "structured_transformer"],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold-objective", default="f1")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _save_figure(fig: plt.Figure, png: Path, pdf: Path) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def _metrics_row(
    variant_id: str,
    model_name: str,
    split: str,
    threshold: float,
    objective: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, Any]:
    m = compute_all_metrics(y_true, y_score, threshold)
    return {
        "variant_id": variant_id,
        "model_name": model_name,
        "split": split,
        "threshold": threshold,
        "threshold_objective": objective,
        **{k: m[k] for k in METRIC_COLS},
    }


def _save_predictions(
    df: pd.DataFrame,
    scores: np.ndarray,
    model_name: str,
    split: str,
    threshold: float,
    path: Path,
) -> None:
    preds = (scores >= threshold).astype(int)
    out = pd.DataFrame(
        {
            "model_name": model_name,
            "split": split,
            "pair_id": df.get("pair_id"),
            "label": df["label"],
            "score": scores,
            "pred": preds,
            "threshold": threshold,
            "left_title": df.get("left_title"),
            "right_title": df.get("right_title"),
            "left_brand": df.get("left_brand"),
            "right_brand": df.get("right_brand"),
            "left_price": df.get("left_price"),
            "right_price": df.get("right_price"),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def _load_existing_scorer(
    pred_dir: Path,
    splits: dict[str, pd.DataFrame],
    variant_id: str,
    objective: str,
) -> tuple[list[dict[str, Any]], dict[str, float], dict[str, np.ndarray]] | None:
    valid_path = pred_dir / "raw_valid_predictions.csv"
    if not valid_path.exists():
        return None
    scores_by_split: dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        if split not in splits:
            continue
        path = pred_dir / f"raw_{split}_predictions.csv"
        if not path.exists():
            warnings.warn(f"Missing {path}; skipping existing scorer on {split}.")
            continue
        raw = pd.read_csv(path)
        scores_by_split[split] = raw["neural_score"].astype(float).values
    if "valid" not in scores_by_split:
        return None
    thr, _, _ = choose_threshold(
        splits["valid"]["label"].astype(int).values,
        scores_by_split["valid"],
        objective=objective,
    )
    metric_rows: list[dict[str, Any]] = []
    for split in scores_by_split:
        metric_rows.append(
            _metrics_row(
                variant_id,
                "existing_scorer",
                split,
                thr,
                objective,
                splits[split]["label"].astype(int).values,
                scores_by_split[split],
            )
        )
    return metric_rows, {"selected": thr}, scores_by_split


def _governed_metrics(
    df: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    checker: ProductConstraintChecker,
) -> dict[str, Any]:
    sym = checker.check_dataframe(df)
    merged = df.copy()
    for col in sym.columns:
        if col != "pair_id":
            merged[col] = sym[col].values
    merged["neural_score"] = scores
    governed = apply_governed_decision(
        merged,
        DecisionConfig(decision_mode="invalid_blocks", threshold=threshold),
    )
    y_true = df["label"].astype(int).values
    y_pred = governed["governed_pred"].astype(int).values
    from src.evaluation import compute_binary_metrics_from_predictions

    return compute_binary_metrics_from_predictions(
        y_true, y_pred, scores, threshold
    )


def plot_comparisons(
    comparison: pd.DataFrame,
    fig_dir: Path,
) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    names = [MODEL_LABELS.get(m, m) for m in comparison["model_name"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    ax.bar(x, comparison["test_f1"], color=["#9e9e9e", "#5cb85c", "#1f77b4"][: len(names)])
    for i, (_, row) in enumerate(comparison.iterrows()):
        ax.text(
            i,
            row["test_f1"] + 0.01,
            f"{row['test_f1']:.3f}\nP={row['test_precision']:.2f} R={row['test_recall']:.2f}",
            ha="center",
            fontsize=7,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Test F1")
    ax.set_title("Structured neural scorer comparison: test F1")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_figure(
        fig,
        fig_dir / "structured_model_test_f1_comparison.png",
        fig_dir / "structured_model_test_f1_comparison.pdf",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.35
    ax.bar(x - w / 2, comparison["test_precision"], w, label="Precision")
    ax.bar(x + w / 2, comparison["test_recall"], w, label="Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_title("Structured neural scorer comparison: test precision and recall")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_figure(
        fig,
        fig_dir / "structured_model_precision_recall_test.png",
        fig_dir / "structured_model_precision_recall_test.pdf",
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, comparison["test_fp"], color="#d9534f")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("False positives")
    ax.set_title("False positives by neural scorer architecture")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_figure(
        fig,
        fig_dir / "structured_model_fp_test.png",
        fig_dir / "structured_model_fp_test.pdf",
    )


def plot_training_curve(history: pd.DataFrame, model_name: str, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train loss")
    ax.plot(history["epoch"], history["valid_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training curve: {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    stem = fig_dir / f"{model_name}_training_curve"
    _save_figure(fig, stem.with_suffix(".png"), stem.with_suffix(".pdf"))


def plot_attention(attn_df: pd.DataFrame, fig_dir: Path) -> None:
    sub = attn_df[attn_df["group"] == "all"].sort_values("mean_attention", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(sub["field"], sub["mean_attention"], yerr=sub["std_attention"], capsize=3)
    ax.set_ylabel("Mean attention weight")
    ax.set_title("Field-level attention weights on test pairs")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_figure(
        fig,
        fig_dir / "field_attention_average_weights_test.png",
        fig_dir / "field_attention_average_weights_test.pdf",
    )


def main() -> int:
    args = parse_args()
    variant_id = args.variant
    table_dir = OUTPUTS_DIR / "tables" / "structured_models" / variant_id
    pred_dir = OUTPUTS_DIR / "predictions" / variant_id / "structured_models"
    model_dir = OUTPUTS_DIR / "models" / variant_id / "structured_models"
    fig_dir = OUTPUTS_DIR / "figures" / "structured_models" / variant_id
    meta_path = table_dir / "metadata.json"

    if meta_path.exists() and not args.overwrite:
        print("Outputs exist; use --overwrite to regenerate.", file=sys.stderr)
        return 0

    table_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_processed_variant(variant_id)
    splits = {
        "train": dataset.train.df,
        "valid": dataset.valid.df,
        "test": dataset.test.df,
    }
    feat_cfg = StructuredFeatureConfig()
    groups = {
        split: build_structured_feature_groups(df, feat_cfg)
        for split, df in splits.items()
    }
    y = {split: df["label"].astype(int).values for split, df in splits.items()}
    dims = group_dims(groups["train"])

    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    governed_rows: list[dict[str, Any]] = []
    selected_thresholds: dict[str, float] = {}
    scores_cache: dict[str, dict[str, np.ndarray]] = {}

    existing_pred_dir = OUTPUTS_DIR / "predictions" / variant_id / args.run_name
    existing = _load_existing_scorer(
        existing_pred_dir, splits, variant_id, args.threshold_objective
    )
    if existing is None:
        warnings.warn(
            "Existing scorer raw predictions not found; skipping existing_scorer comparison."
        )
    else:
        ex_metrics, ex_thr, ex_scores = existing
        metric_rows.extend(ex_metrics)
        selected_thresholds["existing_scorer"] = ex_thr["selected"]
        scores_cache["existing_scorer"] = ex_scores
        valid_m = next(r for r in ex_metrics if r["split"] == "valid")
        test_m = next(r for r in ex_metrics if r["split"] == "test")
        comparison_rows.append(
            {
                "model_name": "existing_scorer",
                "validation_threshold": ex_thr["selected"],
                "validation_precision": valid_m["precision"],
                "validation_recall": valid_m["recall"],
                "validation_f1": valid_m["f1"],
                "validation_fp": valid_m["fp"],
                "test_precision": test_m["precision"],
                "test_recall": test_m["recall"],
                "test_f1": test_m["f1"],
                "test_fp": test_m["fp"],
                "test_fn": test_m["fn"],
                "auroc": test_m["auroc"],
                "auprc": test_m["auprc"],
            }
        )

    train_cfg = TorchTrainConfig(
        model_type="",
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        random_state=args.random_state,
        device=args.device,
    )
    checker = ProductConstraintChecker(make_symbolic_config("conservative"))

    import torch

    for model_name in args.models:
        print(f"\n=== Training {model_name} ===")
        if model_name == "field_attention":
            model = FieldAttentionMatcher(
                dims,
                hidden_dim=args.hidden_dim,
                field_dim=train_cfg.field_dim,
                dropout=args.dropout,
            )
            train_cfg.model_type = "field_attention"
        elif model_name == "structured_transformer":
            model = StructuredTransformerMatcher(
                dims,
                model_dim=train_cfg.model_dim,
                num_heads=train_cfg.num_heads,
                num_layers=train_cfg.num_layers,
                ff_dim=train_cfg.ff_dim,
                dropout=args.dropout,
                hidden_dim=args.hidden_dim,
            )
            train_cfg.model_type = "structured_transformer"
        else:
            warnings.warn(f"Unknown model {model_name}; skip.")
            continue

        model, train_out = train_torch_matcher(
            model,
            groups["train"],
            y["train"],
            groups["valid"],
            y["valid"],
            train_cfg,
        )
        history_df = pd.DataFrame(train_out["history"])
        history_df.to_csv(table_dir / f"{model_name}_training_history.csv", index=False)
        plot_training_curve(history_df, model_name, fig_dir)

        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": model.state_dict(), "group_dims": dims, "model_name": model_name},
            model_dir / f"{model_name}.pt",
        )

        split_scores: dict[str, np.ndarray] = {}
        test_attention = None
        for split in ("train", "valid", "test"):
            if model_name == "field_attention" and split == "test":
                scores, attn = predict_torch_matcher(
                    model,
                    groups[split],
                    batch_size=train_cfg.batch_size,
                    device=train_cfg.device,
                    return_attention=True,
                )
                test_attention = attn
            else:
                scores = predict_torch_matcher(
                    model,
                    groups[split],
                    batch_size=train_cfg.batch_size,
                    device=train_cfg.device,
                )
            split_scores[split] = scores

        thr, _, _ = choose_threshold(
            y["valid"],
            split_scores["valid"],
            objective=args.threshold_objective,
        )
        selected_thresholds[model_name] = thr
        scores_cache[model_name] = split_scores

        for split in ("train", "valid", "test"):
            metric_rows.append(
                _metrics_row(
                    variant_id,
                    model_name,
                    split,
                    thr,
                    args.threshold_objective,
                    y[split],
                    split_scores[split],
                )
            )
            _save_predictions(
                splits[split],
                split_scores[split],
                model_name,
                split,
                thr,
                pred_dir / f"{model_name}_{split}_predictions.csv",
            )

        valid_m = _metrics_row(
            variant_id, model_name, "valid", thr, args.threshold_objective, y["valid"], split_scores["valid"]
        )
        test_m = _metrics_row(
            variant_id, model_name, "test", thr, args.threshold_objective, y["test"], split_scores["test"]
        )
        comparison_rows.append(
            {
                "model_name": model_name,
                "validation_threshold": thr,
                "validation_precision": valid_m["precision"],
                "validation_recall": valid_m["recall"],
                "validation_f1": valid_m["f1"],
                "validation_fp": valid_m["fp"],
                "test_precision": test_m["precision"],
                "test_recall": test_m["recall"],
                "test_f1": test_m["f1"],
                "test_fp": test_m["fp"],
                "test_fn": test_m["fn"],
                "auroc": test_m["auroc"],
                "auprc": test_m["auprc"],
            }
        )

        if model_name == "field_attention" and test_attention is not None:
            field_names = model.field_names
            attn_rows = []
            for fi, fname in enumerate(field_names):
                vals = test_attention[:, fi]
                attn_rows.append(
                    {
                        "field": fname,
                        "group": "all",
                        "mean_attention": float(vals.mean()),
                        "std_attention": float(vals.std()),
                    }
                )
            preds = (split_scores["test"] >= thr).astype(int)
            labels = y["test"]
            for outcome, mask_fn in [
                ("TP", lambda l, p: (l == 1) & (p == 1)),
                ("FP", lambda l, p: (l == 0) & (p == 1)),
                ("TN", lambda l, p: (l == 0) & (p == 0)),
                ("FN", lambda l, p: (l == 1) & (p == 0)),
            ]:
                mask = mask_fn(labels, preds)
                if not mask.any():
                    continue
                for fi, fname in enumerate(field_names):
                    vals = test_attention[mask, fi]
                    attn_rows.append(
                        {
                            "field": fname,
                            "group": outcome,
                            "mean_attention": float(vals.mean()),
                            "std_attention": float(vals.std()),
                        }
                    )
            attn_df = pd.DataFrame(attn_rows)
            attn_df.to_csv(
                table_dir / "field_attention_average_attention.csv", index=False
            )
            plot_attention(attn_df, fig_dir)

        for mode_label, neural_only in [
            (f"{model_name}_neural_only", True),
            (f"{model_name}_governed", False),
        ]:
            thr_m = selected_thresholds[model_name]
            scores = split_scores["test"]
            if neural_only:
                from src.evaluation import compute_binary_metrics_from_predictions

                m = compute_binary_metrics_from_predictions(
                    y["test"], (scores >= thr_m).astype(int), scores, thr_m
                )
            else:
                m = _governed_metrics(splits["test"], scores, thr_m, checker)
            governed_rows.append({"model_name": mode_label, **{k: m[k] for k in METRIC_COLS if k in m}})

    if "existing_scorer" in scores_cache:
        thr_e = selected_thresholds["existing_scorer"]
        sc = scores_cache["existing_scorer"]["test"]
        from src.evaluation import compute_binary_metrics_from_predictions

        m_n = compute_binary_metrics_from_predictions(
            y["test"], (sc >= thr_e).astype(int), sc, thr_e
        )
        governed_rows.append(
            {"model_name": "existing_scorer_neural_only", **{k: m_n[k] for k in METRIC_COLS if k in m_n}}
        )
        m_g = _governed_metrics(splits["test"], sc, thr_e, checker)
        governed_rows.append(
            {"model_name": "existing_scorer_governed", **{k: m_g[k] for k in METRIC_COLS if k in m_g}}
        )

    metrics_df = pd.DataFrame(metric_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    governed_df = pd.DataFrame(governed_rows)

    table_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(table_dir / "structured_model_metrics.csv", index=False)
    comparison_df.to_csv(table_dir / "structured_model_test_comparison.csv", index=False)
    governed_df.to_csv(table_dir / "structured_model_governed_comparison.csv", index=False)

    save_json(
        {
            "variant_id": variant_id,
            "run_name": args.run_name,
            "models": args.models,
            "threshold_objective": args.threshold_objective,
            "selected_thresholds": selected_thresholds,
            "group_dims": dims,
            "notes": "Structured field attention vs transformer-inspired matcher.",
        },
        meta_path,
    )

    if not comparison_df.empty:
        plot_comparisons(comparison_df, fig_dir)

    print("\nSTRUCTURED MODEL TRAINING COMPLETE\n")
    print(f"Threshold objective: {args.threshold_objective}\n")
    print("Test comparison (validation-selected threshold):")
    for _, row in comparison_df.iterrows():
        print(
            f"  {MODEL_LABELS.get(row['model_name'], row['model_name']):22} "
            f"F1={row['test_f1']:.4f} P={row['test_precision']:.4f} "
            f"R={row['test_recall']:.4f} FP={int(row['test_fp'])} "
            f"thr={row['validation_threshold']:.3f}"
        )
    print(f"\nSaved tables: {table_dir}")
    print(f"Saved figures: {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
