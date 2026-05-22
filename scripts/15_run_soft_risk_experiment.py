"""Soft symbolic risk governance experiment (validation-selected, test-reported)."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_WDC_VARIANT_ID, OUTPUTS_DIR
from src.evaluation import compute_binary_metrics_from_predictions, save_json
from src.soft_risk import (
    SoftRiskDecisionConfig,
    SymbolicRiskWeights,
    apply_soft_risk_decision,
    compute_symbolic_risk,
    make_config_id,
    weights_as_dict,
)
from src.symbolic import ProductConstraintChecker, make_symbolic_config, parse_symbolic_list

METHOD_FAMILIES = (
    "neural_only",
    "hard_invalid_blocks",
    "soft_score_penalty",
    "soft_risk_gate",
)

METHOD_LABELS = {
    "neural_only": "Neural-only",
    "hard_invalid_blocks": "Hard veto",
    "soft_score_penalty": "Soft score penalty",
    "soft_risk_gate": "Soft risk gate",
}

EXAMPLE_COLS = [
    "split",
    "example_group",
    "pair_id",
    "label",
    "neural_score",
    "symbolic_risk",
    "governed_score",
    "mode",
    "threshold",
    "lambda_risk",
    "rho",
    "symbolic_status",
    "violated_constraints",
    "risk_terms",
    "left_title",
    "right_title",
    "left_brand",
    "right_brand",
    "left_price",
    "right_price",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Soft symbolic risk experiment.")
    parser.add_argument("--variant", default=DEFAULT_WDC_VARIANT_ID)
    parser.add_argument("--run-name", default="neural_logreg")
    parser.add_argument(
        "--symbolic-profile",
        default="conservative",
        choices=["conservative", "moderate"],
    )
    parser.add_argument("--calibration-split", default="valid")
    parser.add_argument("--test-split", default="test")
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.50, 0.60, 0.66, 0.70, 0.80, 0.90],
    )
    parser.add_argument(
        "--lambda-values",
        nargs="+",
        type=float,
        default=[0.00, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00],
    )
    parser.add_argument(
        "--rho-values",
        nargs="+",
        type=float,
        default=[0.00, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00],
    )
    parser.add_argument(
        "--selection-objective",
        default="f1",
        choices=["f1", "precision", "recall", "balanced_accuracy"],
    )
    parser.add_argument("--max-examples", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_raw(pred_dir: Path, split: str) -> pd.DataFrame:
    path = pred_dir / f"raw_{split}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    return pd.read_csv(path)


def _merge_symbolic(raw: pd.DataFrame, symbolic: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    sym = symbolic.drop(columns=["pair_id"], errors="ignore")
    for col in sym.columns:
        out[col] = sym[col].values
    return out


def build_candidate_configs(
    thresholds: list[float],
    lambda_values: list[float],
    rho_values: list[float],
) -> list[SoftRiskDecisionConfig]:
    configs: list[SoftRiskDecisionConfig] = []
    for thr in thresholds:
        configs.append(SoftRiskDecisionConfig(mode="neural_only", threshold=thr))
        configs.append(SoftRiskDecisionConfig(mode="hard_invalid_blocks", threshold=thr))
    for thr, lam in product(thresholds, lambda_values):
        if lam == 0.0:
            continue
        configs.append(
            SoftRiskDecisionConfig(
                mode="soft_score_penalty",
                threshold=thr,
                lambda_risk=lam,
            )
        )
    for thr, rho in product(thresholds, rho_values):
        configs.append(
            SoftRiskDecisionConfig(
                mode="soft_risk_gate",
                threshold=thr,
                rho=rho,
            )
        )
    return configs


def _risk_diagnostics(df: pd.DataFrame) -> dict[str, float]:
    risk = df["symbolic_risk"].astype(float)
    accepted = df["governed_pred"].astype(int) == 1
    rejected = ~accepted
    nonzero = int((risk > 0).sum())
    n = len(df)
    return {
        "mean_symbolic_risk": float(risk.mean()) if n else 0.0,
        "median_symbolic_risk": float(risk.median()) if n else 0.0,
        "mean_risk_accepted": float(risk[accepted].mean()) if accepted.any() else 0.0,
        "mean_risk_rejected": float(risk[rejected].mean()) if rejected.any() else 0.0,
        "num_nonzero_risk": nonzero,
        "nonzero_risk_rate": nonzero / n if n else 0.0,
    }


def evaluate_config(
    base_df: pd.DataFrame,
    config: SoftRiskDecisionConfig,
    *,
    variant_id: str,
    run_name: str,
    split: str,
    profile: str,
) -> dict[str, Any]:
    decided = apply_soft_risk_decision(base_df, config)
    y_true = decided["label"].astype(int).values
    y_pred = decided["governed_pred"].astype(int).values
    if config.mode == "soft_score_penalty":
        y_score = decided["governed_score"].astype(float).values
    else:
        y_score = decided["neural_score"].astype(float).values

    metrics = compute_binary_metrics_from_predictions(
        y_true, y_pred, y_score, config.threshold
    )
    metrics.update(_risk_diagnostics(decided))
    metrics.update(
        {
            "variant_id": variant_id,
            "run_name": run_name,
            "split": split,
            "symbolic_profile": profile,
            "config_id": make_config_id(config),
            "mode": config.mode,
            "threshold": config.threshold,
            "lambda_risk": config.lambda_risk,
            "rho": config.rho if config.mode == "soft_risk_gate" else np.nan,
            "risk_normalization": config.risk_normalization,
        }
    )
    return metrics


def _select_best(
    valid_df: pd.DataFrame,
    family: str,
    objective: str,
) -> pd.Series:
    sub = valid_df[valid_df["mode"] == family].copy()
    if sub.empty:
        raise ValueError(f"No validation configs for family {family}")
    sub["_thr_dist"] = (sub["threshold"] - 0.66).abs()
    if family == "soft_score_penalty":
        sub["_tie_simple"] = sub["lambda_risk"]
    elif family == "soft_risk_gate":
        sub["_tie_simple"] = -sub["rho"]
    else:
        sub["_tie_simple"] = sub["threshold"]
    sub = sub.sort_values(
        [objective, "precision", "fp", "recall", "_tie_simple", "_thr_dist"],
        ascending=[False, False, True, False, True, True],
        kind="mergesort",
    )
    return sub.iloc[0]


def _config_from_row(row: pd.Series) -> SoftRiskDecisionConfig:
    rho = float(row["rho"]) if row["mode"] == "soft_risk_gate" and not pd.isna(row["rho"]) else 1.0
    return SoftRiskDecisionConfig(
        mode=str(row["mode"]),
        threshold=float(row["threshold"]),
        lambda_risk=float(row["lambda_risk"]) if not pd.isna(row["lambda_risk"]) else 0.0,
        rho=rho,
        risk_normalization=str(row.get("risk_normalization", "sum_weights")),
    )


def collect_examples(
    base: pd.DataFrame,
    selected: dict[str, SoftRiskDecisionConfig],
    split: str,
    max_examples: int,
) -> pd.DataFrame:
    neural_cfg = selected["neural_only"]
    hard_cfg = selected["hard_invalid_blocks"]
    penalty_cfg = selected["soft_score_penalty"]
    gate_cfg = selected["soft_risk_gate"]

    neural = apply_soft_risk_decision(base, neural_cfg)
    hard = apply_soft_risk_decision(base, hard_cfg)
    penalty = apply_soft_risk_decision(base, penalty_cfg)
    gate = apply_soft_risk_decision(base, gate_cfg)

    rows: list[pd.DataFrame] = []

    def _pack(
        df: pd.DataFrame,
        cfg: SoftRiskDecisionConfig,
        mask: pd.Series,
        group: str,
    ) -> pd.DataFrame | None:
        sub = df[mask].sort_values("neural_score", ascending=False).head(max_examples)
        if sub.empty:
            return None
        out = sub.copy()
        out["example_group"] = group
        out["mode"] = cfg.mode
        out["threshold"] = cfg.threshold
        out["lambda_risk"] = cfg.lambda_risk
        out["rho"] = cfg.rho
        out["violated_constraints"] = out["violated_constraints"].map(
            lambda x: json.dumps(parse_symbolic_list(x))
        )
        return out

    # Soft penalty fixed hard veto error: hard reject, penalty accept, label=0
    m = (
        (hard["governed_pred"] == 0)
        & (penalty["governed_pred"] == 1)
        & (penalty["label"] == 0)
    )
    p = _pack(penalty, penalty_cfg, m, "soft_penalty_fixed_hard_veto_error")
    if p is not None:
        rows.append(p)

    m = (
        (neural["governed_pred"] == 0)
        & (penalty["governed_pred"] == 1)
        & (penalty["label"] == 0)
    )
    p = _pack(penalty, penalty_cfg, m, "soft_penalty_new_false_positive")
    if p is not None:
        rows.append(p)

    m = (
        (hard["governed_pred"] == 0)
        & (gate["governed_pred"] == 1)
        & (gate["label"] == 0)
    )
    p = _pack(gate, gate_cfg, m, "risk_gate_fixed_hard_veto_error")
    if p is not None:
        rows.append(p)

    m = (
        (hard["governed_pred"] == 1)
        & (gate["governed_pred"] == 0)
        & (gate["label"] == 1)
    )
    p = _pack(gate, gate_cfg, m, "risk_gate_new_false_negative")
    if p is not None:
        rows.append(p)

    m = (base["symbolic_risk"] >= 0.5) & (neural["governed_pred"] == 1)
    p = _pack(neural, neural_cfg, m, "high_risk_accepted_by_neural")
    if p is not None:
        rows.append(p)

    m = (base["symbolic_risk"] >= 0.3) & (penalty["governed_pred"] == 0) & (neural["governed_pred"] == 1)
    p = _pack(penalty, penalty_cfg, m, "high_risk_rejected_by_soft")
    if p is not None:
        rows.append(p)

    m = (
        (hard["governed_pred"] == 0)
        & (penalty["governed_pred"] == 1)
        & (penalty["label"] == 1)
    )
    p = _pack(penalty, penalty_cfg, m, "tp_recovered_by_soft_penalty")
    if p is not None:
        rows.append(p)

    if not rows:
        return pd.DataFrame(columns=EXAMPLE_COLS)
    out = pd.concat(rows, ignore_index=True)
    out["split"] = split
    return out[[c for c in EXAMPLE_COLS if c in out.columns]]


def _save_figure(fig: plt.Figure, png: Path, pdf: Path) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_f1_comparison(selected: pd.DataFrame, fig_dir: Path, test_split: str) -> None:
    sub = selected[selected["test_split"] == test_split].copy()
    sub["_order"] = sub["method_family"].map(
        {m: i for i, m in enumerate(METHOD_FAMILIES)}
    )
    sub = sub.sort_values("_order")
    names = [METHOD_LABELS.get(m, m) for m in sub["method_family"]]
    f1s = sub["test_f1"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    ax.bar(x, f1s, color=["#9e9e9e", "#4a90d9", "#5cb85c", "#f0ad4e"][: len(names)])
    for bar, val, (_, row) in zip(ax.patches, f1s, sub.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}\nP={row['test_precision']:.2f} R={row['test_recall']:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Test F1")
    ax.set_ylim(0, min(1.0, max(f1s) * 1.2 + 0.05) if f1s else 1.0)
    ax.set_title("Soft symbolic risk experiment: test F1 after validation selection")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    stem = fig_dir / "soft_risk_test_f1_comparison"
    _save_figure(fig, stem.with_suffix(".png"), stem.with_suffix(".pdf"))


def plot_pr_scatter(
    grid_test: pd.DataFrame,
    selected: pd.DataFrame,
    fig_dir: Path,
    test_split: str,
) -> None:
    sub = grid_test[grid_test["split"] == test_split]
    fig, ax = plt.subplots(figsize=(7, 6))
    markers = {
        "neural_only": "o",
        "hard_invalid_blocks": "s",
        "soft_score_penalty": "^",
        "soft_risk_gate": "D",
    }
    colors = {
        "neural_only": "#4d4d4d",
        "hard_invalid_blocks": "#1f77b4",
        "soft_score_penalty": "#2ca02c",
        "soft_risk_gate": "#ff7f0e",
    }
    for mode, group in sub.groupby("mode"):
        ax.scatter(
            group["recall"],
            group["precision"],
            c=colors.get(mode, "#888"),
            marker=markers.get(mode, "o"),
            alpha=0.35,
            s=30,
            label=METHOD_LABELS.get(mode, mode),
        )
    sel = selected[selected["test_split"] == test_split]
    for _, row in sel.iterrows():
        ax.scatter(
            row["test_recall"],
            row["test_precision"],
            c=colors.get(row["method_family"], "#000"),
            marker=markers.get(row["method_family"], "*"),
            s=180,
            edgecolors="black",
            linewidths=1.2,
            zorder=5,
        )
        ax.annotate(
            row["method_family"].replace("_", " "),
            (row["test_recall"], row["test_precision"]),
            fontsize=7,
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Soft symbolic risk: precision-recall trade-off on test")
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    stem = fig_dir / "soft_risk_precision_recall_test"
    _save_figure(fig, stem.with_suffix(".png"), stem.with_suffix(".pdf"))


def plot_lambda_curve(
    grid_test: pd.DataFrame,
    selected: pd.DataFrame,
    thresholds: list[float],
    fig_dir: Path,
    test_split: str,
) -> None:
    sub = grid_test[
        (grid_test["split"] == test_split) & (grid_test["mode"] == "soft_score_penalty")
    ]
    sel_row = selected[selected["method_family"] == "soft_score_penalty"]
    thr = 0.66 if 0.66 in thresholds else (
        float(sel_row.iloc[0]["threshold"]) if not sel_row.empty else float(thresholds[0])
    )
    curve = sub[np.isclose(sub["threshold"], thr)].sort_values("lambda_risk")
    if curve.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(curve["lambda_risk"], curve["f1"], marker="o", label="F1")
    ax.plot(curve["lambda_risk"], curve["precision"], marker="s", linestyle="--", label="Precision")
    ax.plot(curve["lambda_risk"], curve["recall"], marker="^", linestyle=":", label="Recall")
    if not sel_row.empty:
        ax.axvline(float(sel_row.iloc[0]["lambda_risk"]), color="gray", linestyle="--", alpha=0.6)
    ax.set_xlabel("lambda_risk")
    ax.set_ylabel("Metric value")
    ax.set_title(f"Effect of symbolic risk penalty strength on test performance (τ={thr:g})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    stem = fig_dir / "soft_risk_lambda_curve_test"
    _save_figure(fig, stem.with_suffix(".png"), stem.with_suffix(".pdf"))


def save_split_predictions(
    base: pd.DataFrame,
    selected: dict[str, SoftRiskDecisionConfig],
    path: Path,
) -> None:
    out = base.copy()
    for family, cfg in selected.items():
        decided = apply_soft_risk_decision(base, cfg)
        out[f"governed_score__{family}"] = decided["governed_score"]
        out[f"governed_pred__{family}"] = decided["governed_pred"]
        out[f"decision_reason__{family}"] = decided["decision_reason"]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def print_summary(selected: pd.DataFrame, objective: str) -> tuple[str, str]:
    print("SOFT SYMBOLIC RISK EXPERIMENT COMPLETE")
    print()
    print(f"Selected by validation {objective}:")
    print()
    header = (
        f"{'Method':<22} {'Val F1':>7} {'Test F1':>8} {'Test Prec':>10} "
        f"{'Test Rec':>9} {'Test FP':>8}  Config"
    )
    print(header)
    best_soft_f1 = -1.0
    best_soft_mode = ""
    hard_f1 = np.nan
    for _, row in selected.iterrows():
        fam = row["method_family"]
        cfg = row["config_id"]
        print(
            f"{METHOD_LABELS.get(fam, fam):<22} "
            f"{row['validation_f1']:7.4f} {row['test_f1']:8.4f} "
            f"{row['test_precision']:10.4f} {row['test_recall']:9.4f} "
            f"{int(row['test_fp']):8d}  {cfg}"
        )
        if fam == "hard_invalid_blocks":
            hard_f1 = float(row["test_f1"])
        if fam in ("soft_score_penalty", "soft_risk_gate"):
            if float(row["test_f1"]) > best_soft_f1:
                best_soft_f1 = float(row["test_f1"])
                best_soft_mode = fam
    print()
    improved = best_soft_f1 > hard_f1 if not np.isnan(hard_f1) else False
    verdict = "improved" if improved else "did not improve"
    print("Conclusion:")
    print(f"  - soft risk {verdict} over hard veto on test F1")
    if best_soft_mode:
        print(f"  - best soft mode was {best_soft_mode} (test F1={best_soft_f1:.4f})")
    return verdict, best_soft_mode


def main() -> None:
    args = parse_args()
    pred_dir = OUTPUTS_DIR / "predictions" / args.variant / args.run_name
    table_dir = OUTPUTS_DIR / "tables" / "soft_risk" / args.variant / args.run_name
    fig_dir = OUTPUTS_DIR / "figures" / "soft_risk" / args.variant / args.run_name
    meta_path = table_dir / "soft_risk_metadata.json"

    if meta_path.exists() and not args.overwrite:
        print("Outputs exist; use --overwrite to regenerate.")
        selected = pd.read_csv(table_dir / "soft_risk_selected_configs.csv")
        print_summary(selected, args.selection_objective)
        return

    weights = SymbolicRiskWeights()
    checker = ProductConstraintChecker(make_symbolic_config(args.symbolic_profile))
    configs = build_candidate_configs(
        args.thresholds, args.lambda_values, args.rho_values
    )

    splits = [args.calibration_split, args.test_split]
    grid_rows: list[dict[str, Any]] = []
    base_by_split: dict[str, pd.DataFrame] = {}

    for split in splits:
        raw = _load_raw(pred_dir, split)
        symbolic = checker.check_dataframe(raw)
        merged = _merge_symbolic(raw, symbolic)
        merged = compute_symbolic_risk(merged, weights, normalize="sum_weights")
        base_by_split[split] = merged

        for cfg in configs:
            grid_rows.append(
                evaluate_config(
                    merged,
                    cfg,
                    variant_id=args.variant,
                    run_name=args.run_name,
                    split=split,
                    profile=args.symbolic_profile,
                )
            )

    grid_df = pd.DataFrame(grid_rows)
    valid_df = grid_df[grid_df["split"] == args.calibration_split]
    test_df = grid_df[grid_df["split"] == args.test_split]

    selected_rows: list[dict[str, Any]] = []
    selected_configs: dict[str, SoftRiskDecisionConfig] = {}
    obj = args.selection_objective

    for family in METHOD_FAMILIES:
        best_valid = _select_best(valid_df, family, obj)
        cfg = _config_from_row(best_valid)
        selected_configs[family] = cfg
        test_match = test_df[test_df["config_id"] == best_valid["config_id"]]
        if test_match.empty:
            raise ValueError(f"Missing test row for {best_valid['config_id']}")
        test_row = test_match.iloc[0]
        selected_rows.append(
            {
                "method_family": family,
                "selection_split": args.calibration_split,
                "test_split": args.test_split,
                "selected_by": obj,
                "config_id": best_valid["config_id"],
                "mode": best_valid["mode"],
                "threshold": best_valid["threshold"],
                "lambda_risk": best_valid["lambda_risk"],
                "rho": best_valid["rho"],
                "validation_precision": best_valid["precision"],
                "validation_recall": best_valid["recall"],
                "validation_f1": best_valid["f1"],
                "validation_fp": best_valid["fp"],
                "validation_fn": best_valid["fn"],
                "test_precision": test_row["precision"],
                "test_recall": test_row["recall"],
                "test_f1": test_row["f1"],
                "test_fp": test_row["fp"],
                "test_fn": test_row["fn"],
                "test_accuracy": test_row["accuracy"],
                "test_balanced_accuracy": test_row["balanced_accuracy"],
            }
        )

    selected_df = pd.DataFrame(selected_rows)

    examples = collect_examples(
        base_by_split[args.test_split],
        selected_configs,
        args.test_split,
        args.max_examples,
    )

    table_dir.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(table_dir / "soft_risk_grid_results.csv", index=False)
    selected_df.to_csv(table_dir / "soft_risk_selected_configs.csv", index=False)
    examples.to_csv(table_dir / "soft_risk_examples.csv", index=False)

    metadata = {
        "variant_id": args.variant,
        "run_name": args.run_name,
        "symbolic_profile": args.symbolic_profile,
        "calibration_split": args.calibration_split,
        "test_split": args.test_split,
        "selection_objective": args.selection_objective,
        "thresholds": args.thresholds,
        "lambda_values": args.lambda_values,
        "rho_values": args.rho_values,
        "risk_weights": weights_as_dict(weights),
        "num_grid_configs": len(configs),
        "notes": "Soft symbolic risk exploratory experiment; parameters selected on validation only.",
    }
    save_json(metadata, meta_path)

    save_split_predictions(
        base_by_split[args.calibration_split],
        selected_configs,
        pred_dir / "soft_risk_valid_predictions.csv",
    )
    save_split_predictions(
        base_by_split[args.test_split],
        selected_configs,
        pred_dir / "soft_risk_test_predictions.csv",
    )

    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_f1_comparison(selected_df, fig_dir, args.test_split)
    plot_pr_scatter(grid_df, selected_df, fig_dir, args.test_split)
    plot_lambda_curve(grid_df, selected_df, args.thresholds, fig_dir, args.test_split)

    print_summary(selected_df, obj)
    print()
    print("Saved tables:", table_dir)
    print("Saved figures:", fig_dir)
    print("Saved predictions:", pred_dir / "soft_risk_*_predictions.csv")


if __name__ == "__main__":
    main()
