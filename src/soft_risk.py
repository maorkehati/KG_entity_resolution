"""Graded symbolic risk scores and soft governance decision rules."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import pandas as pd

from src.symbolic import parse_symbolic_list


@dataclass(frozen=True)
class SymbolicRiskWeights:
    brand_conflict: float = 1.0
    model_token_conflict: float = 1.0
    quantity_conflict: float = 1.0
    variant_modifier_conflict: float = 0.8
    accessory_main_product_conflict: float = 1.0

    bundle_or_kit_conflict: float = 0.4
    category_keyword_conflict: float = 0.4
    same_currency_price_conflict: float = 0.2
    color_conflict: float = 0.2


@dataclass(frozen=True)
class SoftRiskDecisionConfig:
    mode: str
    threshold: float
    lambda_risk: float = 0.0
    rho: float = 1.0
    risk_normalization: str = "sum_weights"


SUPPORTED_MODES = (
    "neural_only",
    "hard_invalid_blocks",
    "soft_score_penalty",
    "soft_risk_gate",
)


def parse_list_column(x: Any) -> list[str]:
    """Parse list columns robustly (lists, JSON, empty, NaN)."""
    return parse_symbolic_list(x)


def _weight_map(weights: SymbolicRiskWeights) -> dict[str, float]:
    return {f.name: getattr(weights, f.name) for f in fields(weights)}


def _normalization_denominator(weights: SymbolicRiskWeights, normalize: str) -> float:
    wmap = _weight_map(weights)
    if normalize == "none":
        return 1.0
    if normalize in ("sum_weights", "max_weighted_possible"):
        return float(sum(abs(v) for v in wmap.values())) or 1.0
    raise ValueError(f"Unknown normalize: {normalize!r}")


def _risk_for_violations(
    violated: list[str],
    weights: SymbolicRiskWeights,
    normalize: str,
) -> tuple[float, float, list[str]]:
    wmap = _weight_map(weights)
    terms: list[str] = []
    raw = 0.0
    for name in violated:
        w = wmap.get(name, 0.0)
        if w > 0:
            raw += w
            terms.append(name)
    denom = _normalization_denominator(weights, normalize)
    normalized = min(1.0, max(0.0, raw / denom))
    return raw, normalized, terms


def compute_symbolic_risk(
    symbolic_df: pd.DataFrame,
    weights: SymbolicRiskWeights,
    constraints_column: str = "violated_constraints",
    normalize: str = "sum_weights",
) -> pd.DataFrame:
    """Add symbolic_risk_raw, symbolic_risk, risk_terms, num_risk_terms."""
    out = symbolic_df.copy()
    raw_vals: list[float] = []
    norm_vals: list[float] = []
    term_lists: list[list[str]] = []
    for _, row in out.iterrows():
        violated = parse_list_column(row.get(constraints_column))
        raw, norm, terms = _risk_for_violations(violated, weights, normalize)
        raw_vals.append(raw)
        norm_vals.append(norm)
        term_lists.append(terms)
    out["symbolic_risk_raw"] = raw_vals
    out["symbolic_risk"] = norm_vals
    out["risk_terms"] = [json.dumps(t) for t in term_lists]
    out["num_risk_terms"] = [len(t) for t in term_lists]
    return out


def _decide_row(
    neural_score: float,
    symbolic_status: str,
    symbolic_risk: float,
    config: SoftRiskDecisionConfig,
) -> tuple[float, str, int, str]:
    mode = config.mode
    thr = config.threshold
    lam = config.lambda_risk
    rho = config.rho

    if mode == "neural_only":
        governed_score = neural_score
        if neural_score >= thr:
            return governed_score, "accept", 1, "neural_score_above_threshold"
        return governed_score, "reject", 0, "neural_score_below_threshold"

    if mode == "hard_invalid_blocks":
        governed_score = neural_score
        if symbolic_status == "invalid":
            return governed_score, "reject", 0, "symbolic_invalid_blocks"
        if neural_score >= thr:
            return governed_score, "accept", 1, "score_above_threshold_not_invalid"
        return governed_score, "reject", 0, "score_below_threshold"

    if mode == "soft_score_penalty":
        governed_score = neural_score - lam * symbolic_risk
        if governed_score >= thr:
            return governed_score, "accept", 1, "penalized_score_above_threshold"
        return governed_score, "reject", 0, "penalized_score_below_threshold"

    if mode == "soft_risk_gate":
        governed_score = neural_score
        if neural_score >= thr and symbolic_risk <= rho:
            return governed_score, "accept", 1, "score_and_risk_within_bounds"
        if neural_score < thr:
            return governed_score, "reject", 0, "neural_score_below_threshold"
        return governed_score, "reject", 0, "symbolic_risk_above_rho"

    raise ValueError(f"Unsupported mode: {mode!r}")


def apply_soft_risk_decision(
    df: pd.DataFrame,
    config: SoftRiskDecisionConfig,
    score_column: str = "neural_score",
    risk_column: str = "symbolic_risk",
    symbolic_status_column: str = "symbolic_status",
) -> pd.DataFrame:
    """Apply soft-risk governance and add governed columns."""
    if config.mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode: {config.mode!r}")

    out = df.copy()
    governed_scores: list[float] = []
    final_decisions: list[str] = []
    governed_preds: list[int] = []
    reasons: list[str] = []

    for _, row in out.iterrows():
        gs, fd, gp, reason = _decide_row(
            float(row[score_column]),
            str(row.get(symbolic_status_column, "uncertain")),
            float(row.get(risk_column, 0.0)),
            config,
        )
        governed_scores.append(gs)
        final_decisions.append(fd)
        governed_preds.append(gp)
        reasons.append(reason)

    out["governed_score"] = governed_scores
    out["final_decision"] = final_decisions
    out["governed_pred"] = governed_preds
    out["decision_reason"] = reasons
    out["threshold"] = config.threshold
    out["lambda_risk"] = config.lambda_risk
    out["rho"] = config.rho
    out["soft_risk_mode"] = config.mode
    return out


def make_config_id(config: SoftRiskDecisionConfig) -> str:
    parts = [f"mode={config.mode}", f"thr={config.threshold:g}"]
    if config.mode == "soft_score_penalty":
        parts.append(f"lam={config.lambda_risk:g}")
    if config.mode == "soft_risk_gate":
        parts.append(f"rho={config.rho:g}")
    return "|".join(parts)


def weights_as_dict(weights: SymbolicRiskWeights) -> dict[str, float]:
    return asdict(weights)
