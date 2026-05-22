"""Symbolically guided hard-negative identification and sample reweighting."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.symbolic import parse_symbolic_list

STRONG_ONLY_CONSTRAINTS = frozenset(
    {
        "model_token_conflict",
        "quantity_conflict",
    }
)

ALL_CONSERVATIVE_CONSTRAINTS = frozenset(
    {
        "model_token_conflict",
        "quantity_conflict",
        "variant_modifier_conflict",
        "brand_conflict",
        "accessory_main_product_conflict",
    }
)


@dataclass(frozen=True)
class HardNegativeMiningConfig:
    score_threshold: float = 0.66
    hard_negative_weight: float = 3.0
    cumulative_weights: bool = True
    max_weight: float = 10.0
    constraint_mode: str = "strong_only"
    symbolic_profile: str = "conservative"


@dataclass
class HardNegativeMiningResult:
    train_df_with_scores: pd.DataFrame
    sample_weight: np.ndarray
    mining_summary: dict[str, Any]


def parse_list_column(x: Any) -> list[str]:
    """Parse list columns robustly (lists, JSON, empty, NaN)."""
    return parse_symbolic_list(x)


def get_constraints_for_mode(mode: str) -> set[str]:
    if mode == "strong_only":
        return set(STRONG_ONLY_CONSTRAINTS)
    if mode == "all_conservative":
        return set(ALL_CONSERVATIVE_CONSTRAINTS)
    raise ValueError(
        f"Unknown constraint_mode={mode!r}. Use 'strong_only' or 'all_conservative'."
    )


def identify_symbolic_hard_negatives(
    scored_train_df: pd.DataFrame,
    score_column: str = "neural_score",
    label_column: str = "label",
    constraints_column: str = "violated_constraints",
    score_threshold: float = 0.66,
    constraint_mode: str = "strong_only",
) -> pd.Series:
    """
    Boolean mask: label=0, score>=threshold, violated_constraints hits strong set.
    """
    allowed = get_constraints_for_mode(constraint_mode)
    labels = scored_train_df[label_column].astype(int)
    scores = scored_train_df[score_column].astype(float)
    violations = scored_train_df[constraints_column].map(parse_list_column)

    def _hits(row_violations: list[str]) -> bool:
        return bool(set(row_violations) & allowed)

    constraint_hit = violations.map(_hits)
    return (labels == 0) & (scores >= score_threshold) & constraint_hit


def update_sample_weights(
    previous_weights: np.ndarray | None,
    hard_negative_mask: np.ndarray,
    hard_negative_weight: float = 3.0,
    cumulative: bool = True,
    max_weight: float = 10.0,
) -> np.ndarray:
    """Update training sample weights from hard-negative mask."""
    mask = np.asarray(hard_negative_mask, dtype=bool)
    n = len(mask)
    if previous_weights is None:
        weights = np.ones(n, dtype=float)
    else:
        weights = np.asarray(previous_weights, dtype=float).copy()
        if weights.shape[0] != n:
            raise ValueError(
                f"previous_weights length {weights.shape[0]} != mask length {n}"
            )

    if cumulative:
        weights[mask] = np.minimum(weights[mask] * hard_negative_weight, max_weight)
    else:
        weights[:] = 1.0
        weights[mask] = min(hard_negative_weight, max_weight)

    return weights


def summarize_hard_negatives(
    train_df: pd.DataFrame,
    hard_negative_mask: np.ndarray,
    constraints_column: str = "violated_constraints",
    score_column: str = "neural_score",
) -> dict[str, Any]:
    """Summary statistics for mined hard negatives."""
    mask = np.asarray(hard_negative_mask, dtype=bool)
    n = len(train_df)
    hn = train_df.loc[mask]
    scores = hn[score_column].astype(float) if len(hn) else pd.Series(dtype=float)

    constraint_counts: Counter[str] = Counter()
    allowed_all = ALL_CONSERVATIVE_CONSTRAINTS | STRONG_ONLY_CONSTRAINTS
    for val in hn.get(constraints_column, pd.Series(dtype=object)):
        for c in parse_list_column(val):
            if c in allowed_all:
                constraint_counts[c] += 1

    return {
        "num_train": int(n),
        "num_hard_negatives": int(mask.sum()),
        "hard_negative_rate": float(mask.sum() / n) if n else 0.0,
        "mean_score_hard_negatives": float(scores.mean()) if len(scores) else np.nan,
        "min_score_hard_negatives": float(scores.min()) if len(scores) else np.nan,
        "max_score_hard_negatives": float(scores.max()) if len(scores) else np.nan,
        "constraint_counts": dict(constraint_counts),
    }
