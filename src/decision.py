"""Governed merge decisions combining neural scores and symbolic constraints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DecisionConfig:
    decision_mode: str = "invalid_blocks"
    threshold: float = 0.5
    tau_high: float = 0.9
    uncertain_action: str = "flag"


def apply_governed_decision(
    df: pd.DataFrame,
    config: DecisionConfig,
    score_column: str = "neural_score",
    symbolic_status_column: str = "symbolic_status",
) -> pd.DataFrame:
    """Apply neuro-symbolic decision rules and add governed columns."""
    out = df.copy()
    out["threshold"] = config.threshold
    out["tau_high"] = config.tau_high
    out["decision_mode"] = config.decision_mode

    final_decisions: list[str] = []
    governed_preds: list[int] = []
    reasons: list[str] = []

    for _, row in out.iterrows():
        score = float(row[score_column])
        status = str(row[symbolic_status_column])
        fd, gp, reason = _decide_one(
            score, status, config.decision_mode, config.threshold,
            config.tau_high, config.uncertain_action,
        )
        final_decisions.append(fd)
        governed_preds.append(gp)
        reasons.append(reason)

    out["final_decision"] = final_decisions
    out["governed_pred"] = governed_preds
    out["decision_reason"] = reasons
    out["outcome_governed"] = out.apply(_governed_outcome, axis=1)
    return out


def _decide_one(
    neural_score: float,
    symbolic_status: str,
    decision_mode: str,
    threshold: float,
    tau_high: float,
    uncertain_action: str,
) -> tuple[str, int, str]:
    if decision_mode == "invalid_blocks":
        if symbolic_status == "invalid":
            return "reject", 0, "symbolic_invalid_blocks"
        if neural_score >= threshold:
            return "accept", 1, "score_above_threshold_not_invalid"
        return "reject", 0, "score_below_threshold"

    if decision_mode == "strict_valid_or_high_confidence":
        if symbolic_status == "invalid":
            return "reject", 0, "symbolic_invalid_blocks"
        if symbolic_status == "valid" and neural_score >= threshold:
            return "accept", 1, "valid_and_score_above_threshold"
        if symbolic_status == "uncertain" and neural_score >= tau_high:
            if uncertain_action == "accept":
                return "accept", 1, "uncertain_high_score_accepted"
            if uncertain_action == "flag":
                return "flag", 0, "uncertain_high_score_flagged"
            if uncertain_action == "reject":
                return "reject", 0, "uncertain_high_score_rejected"
            raise ValueError(f"Unsupported uncertain_action: {uncertain_action}")
        return "reject", 0, "decision_rule_reject"

    raise ValueError(f"Unsupported decision_mode: {decision_mode}")


def _governed_outcome(row: pd.Series) -> str:
    label = int(row["label"])
    pred = int(row["governed_pred"])
    final = str(row["final_decision"])

    if final == "flag":
        return "FLAG_POSITIVE" if label == 1 else "FLAG_NEGATIVE"
    if label == 1 and pred == 1:
        return "TP"
    if label == 0 and pred == 1:
        return "FP"
    if label == 0 and pred == 0:
        return "TN"
    return "FN"


def compute_governance_diagnostics(df: pd.DataFrame) -> dict:
    """Summary statistics for governed decisions."""
    total = len(df)
    accepted = int((df["final_decision"] == "accept").sum())
    rejected = int((df["final_decision"] == "reject").sum())
    flagged = int((df["final_decision"] == "flag").sum())

    invalid_block = df[
        (df["symbolic_status"] == "invalid")
        & (df["neural_score"] >= df["threshold"])
    ]
    return {
        "total": total,
        "accepted": accepted,
        "rejected": rejected,
        "flagged": flagged,
        "accept_rate": accepted / total if total else 0.0,
        "reject_rate": rejected / total if total else 0.0,
        "flag_rate": flagged / total if total else 0.0,
        "symbolic_valid_count": int((df["symbolic_status"] == "valid").sum()),
        "symbolic_invalid_count": int((df["symbolic_status"] == "invalid").sum()),
        "symbolic_uncertain_count": int((df["symbolic_status"] == "uncertain").sum()),
        "invalid_block_count": len(invalid_block),
        "invalid_block_positive_labels": int((invalid_block["label"] == 1).sum()),
        "invalid_block_negative_labels": int((invalid_block["label"] == 0).sum()),
    }
