"""Plotting helpers for decision-configuration sweep results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_STYLES: dict[str, dict[str, Any]] = {
    "neural_only": {
        "color": "#4d4d4d",
        "marker": "o",
        "linestyle": "-",
        "label": "Neural-only",
    },
    "neuro_symbolic_invalid_blocks": {
        "color": "#1f77b4",
        "marker": "s",
        "linestyle": "-",
        "label": "Invalid-blocks",
    },
    "neuro_symbolic_strict_two_threshold": {
        "color": "#ff7f0e",
        "marker": "^",
        "linestyle": "--",
        "label": "Strict two-threshold",
    },
}

PROFILE_MARKERS = {
    "conservative": "o",
    "moderate": "D",
    "none": "o",
}

PROFILE_LINESTYLES = {
    "conservative": "-",
    "moderate": ":",
    "none": "-",
}

NUMERIC_COLS = [
    "threshold",
    "tau_high",
    "precision",
    "recall",
    "f1",
    "fp",
    "fn",
    "fp_reduction",
    "fp_reduction_rate",
    "recall_loss",
    "recall_loss_rate",
]


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_sweep_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "neurosymbolic_enabled" in df.columns:
        df["neurosymbolic_enabled"] = df["neurosymbolic_enabled"].map(
            lambda x: str(x).lower() in ("true", "1", "yes")
        )
    return df


def method_label(method: str) -> str:
    return METHOD_STYLES.get(method, {}).get("label", method)


def profile_label(profile: str) -> str:
    if profile == "none" or pd.isna(profile):
        return "neural"
    return str(profile)


def config_display_label(row: pd.Series) -> str:
    parts = [method_label(str(row.get("method", "")))]
    if row.get("symbolic_profile") not in (None, "none", np.nan) and not pd.isna(
        row.get("symbolic_profile")
    ):
        parts.append(str(row["symbolic_profile"]))
    parts.append(f"thr={row['threshold']:.2f}")
    if not pd.isna(row.get("tau_high")):
        parts.append(f"tau={row['tau_high']:.2f}")
    return " | ".join(parts)


def add_reference_lines(ax, recall_loss: float = 0.25, fp_reduction: float = 0.25) -> None:
    ax.axhline(0, color="#cccccc", linewidth=0.8, zorder=0)
    ax.axvline(0, color="#cccccc", linewidth=0.8, zorder=0)


def savefig(fig, path: Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def governance_score_row(row: pd.Series) -> float:
    f1 = float(row["f1"]) if not pd.isna(row["f1"]) else 0.0
    fp_red = float(row["fp_reduction_rate"]) if not pd.isna(row["fp_reduction_rate"]) else 0.0
    rec_loss = float(row["recall_loss_rate"]) if not pd.isna(row["recall_loss_rate"]) else 0.0
    return f1 + 0.25 * fp_red - 0.25 * rec_loss


def compute_governance_scores(df: pd.DataFrame) -> pd.Series:
    return df.apply(governance_score_row, axis=1)


def find_highlight_configs(df: pd.DataFrame) -> dict[str, pd.Series | None]:
    """Return named highlight rows for annotations."""
    highlights: dict[str, pd.Series | None] = {}

    if df.empty:
        return highlights

    highlights["best_f1"] = df.loc[df["f1"].idxmax()]

    recall_ok = df[df["recall"] >= 0.50]
    highlights["best_precision_recall50"] = (
        recall_ok.loc[recall_ok["precision"].idxmax()] if not recall_ok.empty else None
    )

    neuro = df[df["method"] != "neural_only"].copy()
    neuro = neuro[neuro["recall_loss_rate"].notna()]
    low_loss = neuro[neuro["recall_loss_rate"] <= 0.25]
    highlights["best_fp_reduction"] = (
        low_loss.loc[low_loss["fp_reduction"].idxmax()] if not low_loss.empty else None
    )

    for thr, key in ((0.66, "neural_thr066"), (0.97, "neural_thr097")):
        match = df[(df["method"] == "neural_only") & (df["threshold"] == thr)]
        highlights[key] = match.iloc[0] if not match.empty else None

    scored = df.copy()
    scored["_gov_score"] = compute_governance_scores(scored)
    highlights["best_governance_score"] = scored.loc[scored["_gov_score"].idxmax()]

    return highlights


def get_style(method: str, profile: str) -> dict[str, Any]:
    base = METHOD_STYLES.get(method, {"color": "#888888", "marker": "o", "linestyle": "-", "label": method})
    prof = profile_label(profile) if profile not in (None, np.nan) else "none"
    marker = PROFILE_MARKERS.get(str(prof), base["marker"])
    return {
        "color": base["color"],
        "marker": marker,
        "linestyle": PROFILE_LINESTYLES.get(str(prof), base["linestyle"]),
        "label": base["label"],
    }


def annotate_point(ax, row: pd.Series, text: str, offset: tuple[float, float] = (5, 5)) -> None:
    ax.annotate(
        text,
        (row["recall"], row["precision"]) if "precision" in row else (row["recall_loss_rate"], row["fp_reduction_rate"]),
        xytext=offset,
        textcoords="offset points",
        fontsize=8,
        alpha=0.9,
    )


def select_top_configs(df: pd.DataFrame, split: str, top_k: int) -> pd.DataFrame:
    sub = df[df["split"] == split].copy()
    sub["governance_score"] = compute_governance_scores(sub)
    sub = sub.sort_values("governance_score", ascending=False).head(top_k)
    sub["rank"] = range(1, len(sub) + 1)
    cols = [
        "split",
        "rank",
        "config_id",
        "method",
        "symbolic_profile",
        "threshold",
        "tau_high",
        "precision",
        "recall",
        "f1",
        "fp",
        "fn",
        "fp_reduction_rate",
        "recall_loss_rate",
        "governance_score",
    ]
    return sub[[c for c in cols if c in sub.columns]]
