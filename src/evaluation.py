"""Evaluation utilities for pairwise entity resolution (threshold-independent)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.scorer import PairwiseMatchScorer

_EPS = 1e-7


def threshold_to_str(threshold: float) -> str:
    return f"{threshold:.3f}".replace(".", "p")


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Convert continuous match scores into binary predictions (1 iff score >= threshold)."""
    return (np.asarray(scores, dtype=float) >= threshold).astype(int)


def compute_score_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    """Threshold-independent metrics from raw neural scores."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    metrics: dict[str, Any] = {
        "score_min": float(np.min(y_score)) if len(y_score) else np.nan,
        "score_max": float(np.max(y_score)) if len(y_score) else np.nan,
        "score_mean": float(np.mean(y_score)) if len(y_score) else np.nan,
        "score_std": float(np.std(y_score)) if len(y_score) else np.nan,
    }

    if len(np.unique(y_true)) < 2:
        metrics["auroc"] = np.nan
        metrics["auprc"] = np.nan
        metrics["average_precision"] = np.nan
    else:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["auroc"] = np.nan
        try:
            ap = float(average_precision_score(y_true, y_score))
            metrics["auprc"] = ap
            metrics["average_precision"] = ap
        except ValueError:
            metrics["auprc"] = np.nan
            metrics["average_precision"] = np.nan

    clipped = np.clip(y_score, _EPS, 1.0 - _EPS)
    try:
        metrics["log_loss"] = float(log_loss(y_true, clipped))
    except ValueError:
        metrics["log_loss"] = np.nan
    try:
        metrics["brier_score"] = float(brier_score_loss(y_true, y_score))
    except ValueError:
        metrics["brier_score"] = np.nan

    return metrics


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Classification metrics after applying a decision threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = apply_threshold(y_score, threshold)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "support": int(len(y_true)),
        "positives_true": int((y_true == 1).sum()),
        "negatives_true": int((y_true == 0).sum()),
        "positives_pred": int((y_pred == 1).sum()),
        "negatives_pred": int((y_pred == 0).sum()),
        "positive_rate_true": float(y_true.mean()) if len(y_true) else np.nan,
        "positive_rate_pred": float(y_pred.mean()) if len(y_pred) else np.nan,
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Merge threshold-independent and thresholded metrics."""
    out = compute_score_metrics(y_true, y_score)
    out.update(compute_threshold_metrics(y_true, y_score, threshold))
    return out


def threshold_sweep(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """One row of threshold metrics per candidate threshold."""
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)
    rows = []
    for thr in thresholds:
        row = compute_threshold_metrics(y_true, y_score, float(thr))
        rows.append(row)
    return pd.DataFrame(rows)


def choose_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    objective: str = "f1",
    min_precision: float | None = None,
    min_recall: float | None = None,
) -> tuple[float, dict[str, Any], pd.DataFrame]:
    """Choose threshold on validation scores."""
    sweep = threshold_sweep(y_true, y_score)
    if sweep.empty:
        return 0.5, {}, sweep

    candidates = sweep.copy()

    if objective == "precision_at_min_recall":
        if min_recall is None:
            raise ValueError("min_recall is required for precision_at_min_recall")
        feasible = candidates[candidates["recall"] >= min_recall]
        candidates = feasible if not feasible.empty else sweep
        candidates = candidates.sort_values(
            ["precision", "recall"], ascending=[False, False]
        )
    elif objective == "recall_at_min_precision":
        if min_precision is None:
            raise ValueError("min_precision is required for recall_at_min_precision")
        feasible = candidates[candidates["precision"] >= min_precision]
        if feasible.empty:
            # No threshold meets the constraint: pick highest precision (safest for ER).
            candidates = sweep.sort_values(
                ["precision", "recall"], ascending=[False, False]
            )
            best_row = candidates.iloc[0]
            best_threshold = float(best_row["threshold"])
            best_metrics = compute_all_metrics(y_true, y_score, best_threshold)
            best_metrics["objective"] = objective
            best_metrics["constraint_met"] = False
            best_metrics["requested_min_precision"] = min_precision
            return best_threshold, best_metrics, sweep
        candidates = feasible.sort_values(
            ["recall", "precision"], ascending=[False, False]
        )
    elif objective == "f1":
        candidates = candidates.sort_values("f1", ascending=False)
    elif objective == "precision":
        candidates = candidates.sort_values(
            ["precision", "recall"], ascending=[False, False]
        )
    elif objective == "recall":
        candidates = candidates.sort_values(
            ["recall", "precision"], ascending=[False, False]
        )
    elif objective == "balanced_accuracy":
        candidates = candidates.sort_values("balanced_accuracy", ascending=False)
    else:
        raise ValueError(f"Unsupported objective: {objective}")

    best_row = candidates.iloc[0]
    best_threshold = float(best_row["threshold"])
    best_metrics = compute_all_metrics(y_true, y_score, best_threshold)
    best_metrics["objective"] = objective
    best_metrics["constraint_met"] = True
    return best_threshold, best_metrics, sweep


def build_prediction_dataframe(
    source_df: pd.DataFrame,
    scores: np.ndarray,
    split: str,
    score_column: str = "neural_score",
) -> pd.DataFrame:
    """Build rich raw-prediction dataframe without thresholding."""
    optional_cols = [
        "pair_id",
        "left_id",
        "right_id",
        "label",
        "left_title",
        "right_title",
        "left_brand",
        "right_brand",
        "left_price",
        "right_price",
        "left_description",
        "right_description",
        "pair_text",
    ]
    data: dict[str, Any] = {"split": split, score_column: scores}
    for col in optional_cols:
        if col in source_df.columns:
            data[col] = source_df[col].values
    if "label" in data:
        data["label"] = source_df["label"].astype(int).values
    return pd.DataFrame(data)


def add_thresholded_predictions(
    pred_df: pd.DataFrame,
    threshold: float,
    score_column: str = "neural_score",
) -> pd.DataFrame:
    """Add threshold, neural_pred, and outcome (TP/FP/TN/FN)."""
    out = pred_df.copy()
    out["threshold"] = float(threshold)
    out["neural_pred"] = apply_threshold(out[score_column].values, threshold)

    def _outcome(row: pd.Series) -> str:
        label = int(row["label"])
        pred = int(row["neural_pred"])
        if label == 1 and pred == 1:
            return "TP"
        if label == 0 and pred == 1:
            return "FP"
        if label == 0 and pred == 0:
            return "TN"
        return "FN"

    out["outcome"] = out.apply(_outcome, axis=1)
    return out


def make_error_analysis(
    thresholded_df: pd.DataFrame,
    max_per_group: int = 50,
    score_column: str = "neural_score",
) -> pd.DataFrame:
    """Curated error and borderline examples for reporting."""
    work = thresholded_df.copy()
    thr = float(work["threshold"].iloc[0]) if "threshold" in work.columns else 0.5
    work["score_margin"] = (work[score_column] - thr).abs()

    def _take(mask: pd.Series, sort_col: str, ascending: bool, group: str) -> pd.DataFrame:
        sub = work[mask].sort_values(sort_col, ascending=ascending).head(max_per_group)
        if sub.empty:
            return sub
        out = sub.copy()
        out["error_group"] = group
        return out

    frames = [
        _take(
            (work["outcome"] == "FP"),
            score_column,
            False,
            "false_positive_high_score",
        ),
        _take(
            (work["outcome"] == "FN"),
            score_column,
            True,
            "false_negative_low_score",
        ),
        _take(
            (work["outcome"] == "FP"),
            "score_margin",
            True,
            "borderline_false_positive",
        ),
        _take(
            (work["outcome"] == "FN"),
            "score_margin",
            True,
            "borderline_false_negative",
        ),
        _take(
            (work["outcome"] == "TP"),
            score_column,
            False,
            "true_positive_high_score",
        ),
        _take(
            (work["outcome"] == "TN"),
            score_column,
            True,
            "true_negative_low_score",
        ),
    ]

    cols = [
        "split",
        "error_group",
        "pair_id",
        "label",
        "neural_score",
        "neural_pred",
        "outcome",
        "left_title",
        "right_title",
        "left_brand",
        "right_brand",
        "left_price",
        "right_price",
    ]
    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    available = [c for c in cols if c in combined.columns]
    return combined[available]


def save_json(obj: dict[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        raise TypeError(f"Object of type {type(o)} is not JSON serializable")

    path.write_text(
        json.dumps(obj, indent=2, default=_default),
        encoding="utf-8",
    )


def save_metrics_table(metrics_by_split: list[dict[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics_by_split).to_csv(path, index=False)


def summarize_feature_importance(
    scorer: PairwiseMatchScorer,
    top_k: int | None = 50,
) -> pd.DataFrame:
    """Logistic regression coefficients mapped to feature names."""
    if scorer.model_ is None:
        raise RuntimeError("Scorer model is not fitted.")

    names = scorer.feature_extractor_.get_feature_names_out()
    coefs = scorer.model_.coef_.ravel()
    rows = []
    for name, coef in zip(names, coefs):
        rows.append(
            {
                "feature": str(name),
                "coefficient": float(coef),
                "abs_coefficient": float(abs(coef)),
                "direction": "positive_match" if coef > 0 else "negative_match",
            }
        )
    importance = pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False)
    if top_k is not None:
        importance = importance.head(top_k)
    return importance.reset_index(drop=True)


def compute_binary_metrics_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Classification metrics from binary predictions; optional score ranking metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    metrics: dict[str, Any] = {
        "threshold": float(threshold) if threshold is not None else np.nan,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "support": int(len(y_true)),
        "positives_true": int((y_true == 1).sum()),
        "negatives_true": int((y_true == 0).sum()),
        "positives_pred": int((y_pred == 1).sum()),
        "negatives_pred": int((y_pred == 0).sum()),
        "positive_rate_true": float(y_true.mean()) if len(y_true) else np.nan,
        "positive_rate_pred": float(y_pred.mean()) if len(y_pred) else np.nan,
    }

    if y_score is not None:
        metrics.update(compute_score_metrics(y_true, y_score))

    return metrics


# Backward-compatible aliases
def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    metrics = compute_all_metrics(y_true, y_score, threshold)
    return metrics


def predictions_dataframe(
    df: pd.DataFrame,
    y_score: np.ndarray,
    y_pred: np.ndarray,
    split: str,
) -> pd.DataFrame:
    raw = build_prediction_dataframe(df, y_score, split)
    raw["neural_pred"] = y_pred
    return raw


def error_analysis_table(
    pred_df: pd.DataFrame,
    max_examples: int = 50,
    threshold: float | None = None,
) -> pd.DataFrame:
    thr = threshold if threshold is not None else 0.5
    work = add_thresholded_predictions(pred_df, thr)
    return make_error_analysis(work, max_per_group=max_examples)
