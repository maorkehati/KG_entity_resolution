"""Analyze per-constraint impact and leave-one-rule-out symbolic ablations."""

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
from src.decision import DecisionConfig, apply_governed_decision
from src.evaluation import compute_binary_metrics_from_predictions, save_json
from src.symbolic import (
    ALL_ANALYZED_CONSTRAINTS,
    ProductConstraintChecker,
    hard_constraints_for_profile,
    make_symbolic_config,
    parse_symbolic_list,
    remove_constraint_from_symbolic_results,
)

EXAMPLE_COLS = [
    "split",
    "example_group",
    "constraint",
    "pair_id",
    "label",
    "neural_score",
    "neural_pred",
    "symbolic_status",
    "violated_constraints",
    "positive_evidence",
    "uncertain_reasons",
    "governed_pred",
    "left_title",
    "right_title",
    "left_brand",
    "right_brand",
    "left_price",
    "right_price",
    "left_description",
    "right_description",
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Symbolic rule impact analysis.")
    parser.add_argument("--variant", default=DEFAULT_WDC_VARIANT_ID)
    parser.add_argument("--run-name", default="neural_logreg")
    parser.add_argument("--threshold", type=float, default=0.66)
    parser.add_argument(
        "--symbolic-profile",
        default="conservative",
        choices=["conservative", "moderate"],
    )
    parser.add_argument("--splits", nargs="+", default=["valid", "test"])
    parser.add_argument("--max-examples", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_raw_predictions(pred_dir: Path, split: str) -> pd.DataFrame:
    path = pred_dir / f"raw_{split}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw predictions: {path}")
    return pd.read_csv(path)


def _merge_symbolic(raw: pd.DataFrame, symbolic: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    sym = symbolic.drop(columns=["pair_id"], errors="ignore")
    for col in sym.columns:
        out[col] = sym[col].values
    return out


def _apply_governance(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    cfg = DecisionConfig(decision_mode="invalid_blocks", threshold=threshold)
    return apply_governed_decision(df, cfg)


def _violations_series(df: pd.DataFrame) -> pd.Series:
    return df["violated_constraints"].map(parse_symbolic_list)


def _constraint_triggered(violations: pd.Series, constraint: str) -> pd.Series:
    return violations.map(lambda xs: constraint in xs)


def _metrics_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    m = compute_binary_metrics_from_predictions(y_true, y_pred, y_score, threshold)
    return {k: m[k] for k in METRIC_COLS}


def compute_constraint_impact(
    df: pd.DataFrame,
    *,
    variant_id: str,
    run_name: str,
    split: str,
    threshold: float,
    profile: str,
) -> pd.DataFrame:
    violations = _violations_series(df)
    neural_pred = df["neural_pred"].astype(int)
    labels = df["label"].astype(int)
    total = len(df)
    rows: list[dict[str, Any]] = []

    for constraint in ALL_ANALYZED_CONSTRAINTS:
        triggered = _constraint_triggered(violations, constraint)
        triggered_count = int(triggered.sum())
        neural_blocked = neural_pred == 1
        blocked_mask = neural_blocked & triggered
        blocked_fp = int((blocked_mask & (labels == 0)).sum())
        blocked_tp = int((blocked_mask & (labels == 1)).sum())
        blocked_neural_accepts = int(blocked_mask.sum())

        unique_fp = 0
        unique_tp = 0
        for idx in df.index[blocked_mask & (labels == 0)]:
            vset = set(violations.loc[idx])
            if vset == {constraint}:
                unique_fp += 1
        for idx in df.index[blocked_mask & (labels == 1)]:
            vset = set(violations.loc[idx])
            if vset == {constraint}:
                unique_tp += 1

        rows.append(
            {
                "variant_id": variant_id,
                "run_name": run_name,
                "split": split,
                "threshold": threshold,
                "symbolic_profile": profile,
                "constraint": constraint,
                "triggered_count": triggered_count,
                "triggered_rate": triggered_count / total if total else 0.0,
                "triggered_positive_labels": int((triggered & (labels == 1)).sum()),
                "triggered_negative_labels": int((triggered & (labels == 0)).sum()),
                "blocked_neural_accepts": blocked_neural_accepts,
                "blocked_fp": blocked_fp,
                "blocked_tp": blocked_tp,
                "veto_precision": blocked_fp / max(blocked_neural_accepts, 1),
                "veto_harm_rate": blocked_tp / max(blocked_neural_accepts, 1),
                "net_blocked": blocked_fp - blocked_tp,
                "fp_to_tp_block_ratio": blocked_fp / max(blocked_tp, 1),
                "unique_blocked_fp": unique_fp,
                "unique_blocked_tp": unique_tp,
            }
        )
    return pd.DataFrame(rows)


def compute_overlap_matrix(df: pd.DataFrame, split: str) -> pd.DataFrame:
    violations = _violations_series(df)
    neural_blocked = df["neural_pred"].astype(int) == 1
    labels = df["label"].astype(int)
    active = sorted({c for v in violations for c in v})
    rows: list[dict[str, Any]] = []

    for c1 in active:
        t1 = _constraint_triggered(violations, c1) & neural_blocked
        for c2 in active:
            t2 = _constraint_triggered(violations, c2) & neural_blocked
            both = t1 & t2
            rows.append(
                {
                    "split": split,
                    "constraint_i": c1,
                    "constraint_j": c2,
                    "overlap_count": int(both.sum()),
                    "overlap_fp": int((both & (labels == 0)).sum()),
                    "overlap_tp": int((both & (labels == 1)).sum()),
                }
            )
    return pd.DataFrame(rows)


def run_ablations(
    raw: pd.DataFrame,
    symbolic_base: pd.DataFrame,
    *,
    variant_id: str,
    run_name: str,
    split: str,
    threshold: float,
    profile: str,
) -> pd.DataFrame:
    hard = hard_constraints_for_profile(profile)
    cfg = make_symbolic_config(profile)
    require_pos = cfg.require_positive_evidence_for_valid

    merged = _merge_symbolic(raw, symbolic_base)
    merged["neural_pred"] = (merged["neural_score"] >= threshold).astype(int)
    y_true = merged["label"].astype(int).values
    y_score = merged["neural_score"].astype(float).values

    configs: list[tuple[str, str | None, pd.DataFrame]] = [
        ("neural_only", None, merged),
    ]

    full = _apply_governance(merged, threshold)
    configs.append(("full_governance", None, full))

    for constraint in hard:
        ablated_sym = remove_constraint_from_symbolic_results(
            symbolic_base,
            constraint,
            require_positive_evidence=require_pos,
        )
        ablated = _merge_symbolic(raw, ablated_sym)
        ablated["neural_pred"] = (ablated["neural_score"] >= threshold).astype(int)
        ablated = _apply_governance(ablated, threshold)
        configs.append((f"without_{constraint}", constraint, ablated))

    rows: list[dict[str, Any]] = []
    for ablation, removed, frame in configs:
        if ablation == "neural_only":
            preds = frame["neural_pred"].astype(int).values
        else:
            preds = frame["governed_pred"].astype(int).values
        m = _metrics_row(y_true, preds, y_score, threshold)
        rows.append(
            {
                "variant_id": variant_id,
                "run_name": run_name,
                "split": split,
                "threshold": threshold,
                "symbolic_profile": profile,
                "ablation": ablation,
                "removed_constraint": removed if removed else "",
                **m,
            }
        )
    return pd.DataFrame(rows)


def compute_ablation_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in metrics["split"].unique():
        sub = metrics[metrics["split"] == split]
        full = sub[sub["ablation"] == "full_governance"]
        if full.empty:
            continue
        full_row = full.iloc[0]
        for _, row in sub.iterrows():
            if row["ablation"] in ("full_governance",):
                continue
            rows.append(
                {
                    "split": split,
                    "ablation": row["ablation"],
                    "removed_constraint": row["removed_constraint"],
                    "delta_precision_vs_full": row["precision"] - full_row["precision"],
                    "delta_recall_vs_full": row["recall"] - full_row["recall"],
                    "delta_f1_vs_full": row["f1"] - full_row["f1"],
                    "delta_fp_vs_full": row["fp"] - full_row["fp"],
                    "delta_fn_vs_full": row["fn"] - full_row["fn"],
                }
            )
    return pd.DataFrame(rows)


def _json_list_str(items: list[str]) -> str:
    return json.dumps(items)


def collect_examples(
    df: pd.DataFrame,
    split: str,
    max_examples: int,
    hard: list[str],
) -> pd.DataFrame:
    work = df.copy()
    work["violated_constraints"] = _violations_series(work).map(_json_list_str)
    if "positive_evidence" in work.columns:
        work["positive_evidence"] = work["positive_evidence"].map(
            lambda x: _json_list_str(parse_symbolic_list(x))
        )
    if "uncertain_reasons" in work.columns:
        work["uncertain_reasons"] = work["uncertain_reasons"].map(
            lambda x: _json_list_str(parse_symbolic_list(x))
        )

    violations = _violations_series(df)
    rows: list[pd.DataFrame] = []

    for constraint in hard:
        trig = _constraint_triggered(violations, constraint)
        helpful = work[
            (work["neural_pred"] == 1)
            & (work["label"] == 0)
            & trig
        ].sort_values("neural_score", ascending=False).head(max_examples)
        if not helpful.empty:
            h = helpful.copy()
            h["example_group"] = "helpful_blocked_fp"
            h["constraint"] = constraint
            rows.append(h)

        harmful = work[
            (work["neural_pred"] == 1)
            & (work["label"] == 1)
            & trig
        ].sort_values("neural_score", ascending=False).head(max_examples)
        if not harmful.empty:
            hm = harmful.copy()
            hm["example_group"] = "harmful_blocked_tp"
            hm["constraint"] = constraint
            rows.append(hm)

    surviving = work[
        (work["governed_pred"] == 1)
        & (work["label"] == 0)
        & (work["symbolic_status"] != "invalid")
    ].sort_values("neural_score", ascending=False).head(max_examples)
    if not surviving.empty:
        s = surviving.copy()
        s["example_group"] = "surviving_governed_fp"
        s["constraint"] = ""
        rows.append(s)

    if not rows:
        return pd.DataFrame(columns=EXAMPLE_COLS)

    out = pd.concat(rows, ignore_index=True)
    out["split"] = split
    cols = [c for c in EXAMPLE_COLS if c in out.columns]
    return out[cols]


def _save_figure(fig: plt.Figure, png: Path, pdf: Path) -> None:
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def plot_constraint_impact(impact: pd.DataFrame, split: str, fig_dir: Path) -> None:
    sub = impact[impact["split"] == split].copy()
    hard = sub[sub["blocked_neural_accepts"] > 0].sort_values("blocked_fp", ascending=False)
    if hard.empty:
        hard = sub.sort_values("triggered_count", ascending=False).head(8)
    names = hard["constraint"].tolist()
    fp_vals = hard["blocked_fp"].tolist()
    tp_vals = hard["blocked_tp"].tolist()

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.75), 5))
    ax.bar(x - width / 2, fp_vals, width, label="Blocked FP", color="#4a90d9")
    ax.bar(x + width / 2, tp_vals, width, label="Blocked TP", color="#d9534f")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Number of neural accepted pairs blocked")
    ax.set_title(f"Constraint impact: blocked false positives vs blocked true positives ({split})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    stem = fig_dir / f"constraint_impact_{split}"
    _save_figure(fig, stem.with_suffix(".png"), stem.with_suffix(".pdf"))


def plot_net_effect(impact: pd.DataFrame, split: str, fig_dir: Path) -> None:
    sub = impact[impact["split"] == split].copy()
    sub = sub.sort_values("net_blocked", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(sub) * 0.35)))
    ax.barh(sub["constraint"], sub["net_blocked"], color="#5cb85c", edgecolor="black")
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("net_blocked = blocked_fp - blocked_tp")
    ax.set_title(f"Net symbolic veto effect by constraint ({split})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    stem = fig_dir / f"constraint_net_effect_{split}"
    _save_figure(fig, stem.with_suffix(".png"), stem.with_suffix(".pdf"))


def plot_ablation_delta_f1(deltas: pd.DataFrame, split: str, fig_dir: Path) -> None:
    sub = deltas[
        (deltas["split"] == split)
        & (deltas["ablation"].str.startswith("without_"))
    ].copy()
    if sub.empty:
        return
    sub = sub.sort_values("delta_f1_vs_full", ascending=True)
    labels = sub["removed_constraint"].str.replace("_", " ", regex=False)
    fig, ax = plt.subplots(figsize=(8, max(4, len(sub) * 0.4)))
    colors = ["#d9534f" if v < 0 else "#5cb85c" for v in sub["delta_f1_vs_full"]]
    ax.barh(labels, sub["delta_f1_vs_full"], color=colors, edgecolor="black")
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("delta_f1_vs_full")
    ax.set_title(f"Leave-one-out ablation: delta F1 vs full governance ({split})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    stem = fig_dir / f"ablation_delta_f1_{split}"
    _save_figure(fig, stem.with_suffix(".png"), stem.with_suffix(".pdf"))


def plot_ablation_tradeoff(deltas: pd.DataFrame, split: str, fig_dir: Path) -> None:
    sub = deltas[
        (deltas["split"] == split)
        & (deltas["ablation"].str.startswith("without_"))
    ].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in sub.iterrows():
        ax.scatter(row["delta_fp_vs_full"], row["delta_recall_vs_full"], s=80)
        ax.annotate(
            row["removed_constraint"].replace("_", " "),
            (row["delta_fp_vs_full"], row["delta_recall_vs_full"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.axhline(0, color="gray", linewidth=0.6)
    ax.axvline(0, color="gray", linewidth=0.6)
    ax.set_xlabel("delta_fp_vs_full")
    ax.set_ylabel("delta_recall_vs_full")
    ax.set_title(f"Ablation tradeoff when removing a rule ({split})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    stem = fig_dir / f"ablation_fp_vs_recall_tradeoff_{split}"
    _save_figure(fig, stem.with_suffix(".png"), stem.with_suffix(".pdf"))


def plot_blocked_fp_vs_tp_legacy(impact: pd.DataFrame, split: str, fig_dir: Path) -> None:
    """Alias plot matching requested output name blocked_fp_vs_blocked_tp_{split}."""
    plot_constraint_impact(impact, split, fig_dir)
    sub = impact[impact["split"] == split]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(sub["blocked_tp"], sub["blocked_fp"], s=60)
    for _, row in sub.iterrows():
        if row["blocked_neural_accepts"] == 0:
            continue
        ax.annotate(
            row["constraint"],
            (row["blocked_tp"], row["blocked_fp"]),
            fontsize=7,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set_xlabel("blocked_tp")
    ax.set_ylabel("blocked_fp")
    ax.set_title(f"Blocked FP vs blocked TP by constraint ({split})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    stem = fig_dir / f"blocked_fp_vs_blocked_tp_{split}"
    _save_figure(fig, stem.with_suffix(".png"), stem.with_suffix(".pdf"))


def print_summary(
    impact: pd.DataFrame,
    ablation_metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    *,
    threshold: float,
    profile: str,
    table_dir: Path,
    fig_dir: Path,
) -> None:
    print("SYMBOLIC RULE IMPACT ANALYSIS COMPLETE")
    print()
    print("Main config:")
    print(f"  threshold = {threshold}")
    print(f"  symbolic_profile = {profile}")
    print("  decision_mode = invalid_blocks")
    print()

    split = "test"
    if split not in ablation_metrics["split"].values:
        split = ablation_metrics["split"].iloc[-1]

    print(f"{split} split summary:")
    full = ablation_metrics[
        (ablation_metrics["split"] == split) & (ablation_metrics["ablation"] == "full_governance")
    ]
    if not full.empty:
        r = full.iloc[0]
        print("  Full governance:")
        print(f"    precision = {r['precision']:.4f}")
        print(f"    recall    = {r['recall']:.4f}")
        print(f"    F1        = {r['f1']:.4f}")
        print(f"    FP        = {int(r['fp'])}")
    print()

    imp = impact[impact["split"] == split].sort_values("net_blocked", ascending=False)
    print("Most helpful constraints by net_blocked:")
    for i, (_, row) in enumerate(imp.head(3).iterrows(), 1):
        print(
            f"  {i}. {row['constraint']} "
            f"(net={int(row['net_blocked'])}, blocked_fp={int(row['blocked_fp'])}, "
            f"blocked_tp={int(row['blocked_tp'])})"
        )
    print()

    harmful = impact[impact["split"] == split].sort_values("blocked_tp", ascending=False)
    print("Most harmful constraints by blocked_tp:")
    for i, (_, row) in enumerate(harmful.head(3).iterrows(), 1):
        if row["blocked_tp"] == 0:
            continue
        print(
            f"  {i}. {row['constraint']} (blocked_tp={int(row['blocked_tp'])})"
        )
    print()

    print("Ablation findings:")
    dsub = deltas[
        (deltas["split"] == split) & (deltas["ablation"].str.startswith("without_"))
    ].sort_values("delta_f1_vs_full", key=abs, ascending=False)
    for _, row in dsub.head(5).iterrows():
        print(
            f"  Removing {row['removed_constraint']} changes F1 by "
            f"{row['delta_f1_vs_full']:+.4f} and FP by {int(row['delta_fp_vs_full']):+d}"
        )
    print()
    print("Saved tables:")
    print(f"  {table_dir / 'constraint_impact.csv'}")
    print(f"  {table_dir / 'constraint_ablation_metrics.csv'}")
    print(f"  {table_dir / 'constraint_ablation_deltas.csv'}")
    print(f"  {table_dir / 'constraint_overlap_matrix.csv'}")
    print(f"  {table_dir / 'symbolic_analysis_examples.csv'}")
    print("Saved figures:")
    print(f"  {fig_dir}/")


def main() -> None:
    args = parse_args()
    pred_dir = OUTPUTS_DIR / "predictions" / args.variant / args.run_name
    table_dir = OUTPUTS_DIR / "tables" / "symbolic_analysis" / args.variant / args.run_name
    fig_dir = OUTPUTS_DIR / "figures" / "symbolic_analysis" / args.variant / args.run_name

    marker = table_dir / "symbolic_analysis_metadata.json"
    if marker.exists() and not args.overwrite:
        print("Outputs exist; use --overwrite to regenerate.")
        print_summary(
            pd.read_csv(table_dir / "constraint_impact.csv"),
            pd.read_csv(table_dir / "constraint_ablation_metrics.csv"),
            pd.read_csv(table_dir / "constraint_ablation_deltas.csv"),
            threshold=args.threshold,
            profile=args.symbolic_profile,
            table_dir=table_dir,
            fig_dir=fig_dir,
        )
        return

    checker = ProductConstraintChecker(make_symbolic_config(args.symbolic_profile))
    hard = hard_constraints_for_profile(args.symbolic_profile)

    impact_parts: list[pd.DataFrame] = []
    overlap_parts: list[pd.DataFrame] = []
    ablation_parts: list[pd.DataFrame] = []
    example_parts: list[pd.DataFrame] = []
    split_frames: dict[str, pd.DataFrame] = {}

    for split in args.splits:
        raw = _load_raw_predictions(pred_dir, split)
        symbolic = checker.check_dataframe(raw)
        merged = _merge_symbolic(raw, symbolic)
        merged["neural_pred"] = (merged["neural_score"] >= args.threshold).astype(int)
        governed = _apply_governance(merged, args.threshold)
        split_frames[split] = governed

        impact_parts.append(
            compute_constraint_impact(
                governed,
                variant_id=args.variant,
                run_name=args.run_name,
                split=split,
                threshold=args.threshold,
                profile=args.symbolic_profile,
            )
        )
        overlap_parts.append(compute_overlap_matrix(governed, split))
        ablation_parts.append(
            run_ablations(
                raw,
                symbolic,
                variant_id=args.variant,
                run_name=args.run_name,
                split=split,
                threshold=args.threshold,
                profile=args.symbolic_profile,
            )
        )
        example_parts.append(
            collect_examples(governed, split, args.max_examples, hard)
        )

    impact_df = pd.concat(impact_parts, ignore_index=True)
    overlap_df = pd.concat(overlap_parts, ignore_index=True)
    ablation_df = pd.concat(ablation_parts, ignore_index=True)
    deltas_df = compute_ablation_deltas(ablation_df)
    examples_df = (
        pd.concat(example_parts, ignore_index=True)
        if example_parts
        else pd.DataFrame(columns=EXAMPLE_COLS)
    )

    table_dir.mkdir(parents=True, exist_ok=True)
    impact_df.to_csv(table_dir / "constraint_impact.csv", index=False)
    ablation_df.to_csv(table_dir / "constraint_ablation_metrics.csv", index=False)
    deltas_df.to_csv(table_dir / "constraint_ablation_deltas.csv", index=False)
    overlap_df.to_csv(table_dir / "constraint_overlap_matrix.csv", index=False)
    examples_df.to_csv(table_dir / "symbolic_analysis_examples.csv", index=False)

    metadata = {
        "variant_id": args.variant,
        "run_name": args.run_name,
        "threshold": args.threshold,
        "symbolic_profile": args.symbolic_profile,
        "decision_mode": "invalid_blocks",
        "splits": args.splits,
        "hard_constraints": hard,
        "analyzed_constraints": ALL_ANALYZED_CONSTRAINTS,
        "notes": "Per-constraint impact and leave-one-rule-out ablations on raw neural predictions.",
    }
    save_json(metadata, table_dir / "symbolic_analysis_metadata.json")

    fig_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        plot_constraint_impact(impact_df, split, fig_dir)
        plot_net_effect(impact_df, split, fig_dir)
        plot_ablation_delta_f1(deltas_df, split, fig_dir)
        plot_ablation_tradeoff(deltas_df, split, fig_dir)
        plot_blocked_fp_vs_tp_legacy(impact_df, split, fig_dir)

    print_summary(
        impact_df,
        ablation_df,
        deltas_df,
        threshold=args.threshold,
        profile=args.symbolic_profile,
        table_dir=table_dir,
        fig_dir=fig_dir,
    )


if __name__ == "__main__":
    main()
