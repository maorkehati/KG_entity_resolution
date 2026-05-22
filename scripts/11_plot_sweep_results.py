"""Visualize decision-configuration sweep results."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_WDC_VARIANT_ID, OUTPUTS_DIR, PROJECT_ROOT
from src.plotting import (
    METHOD_STYLES,
    add_reference_lines,
    compute_governance_scores,
    config_display_label,
    ensure_dir,
    find_highlight_configs,
    get_style,
    load_sweep_results,
    method_label,
    savefig,
    select_top_configs,
)

ANNOTATE_LABELS = {
    "best_f1": "best F1",
    "best_precision_recall50": "best P@R≥0.5",
    "best_fp_reduction": "best FP↓",
    "neural_thr066": "neural τ=0.66",
    "neural_thr097": "neural τ=0.97",
    "best_governance_score": "best gov.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot decision sweep results.")
    parser.add_argument("--variant", default=DEFAULT_WDC_VARIANT_ID)
    parser.add_argument("--run-name", default="neural_logreg")
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "valid", "test", "train"],
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _splits_to_plot(df: pd.DataFrame, split_arg: str) -> list[str]:
    available = sorted(df["split"].dropna().unique())
    if split_arg == "all":
        return [s for s in ("valid", "test", "train") if s in available]
    return [split_arg] if split_arg in available else []


def _scatter_encode(ax, sub: pd.DataFrame, x_col: str, y_col: str) -> None:
    for method, group in sub.groupby("method"):
        for profile, g2 in group.groupby("symbolic_profile"):
            style = get_style(method, str(profile))
            ax.scatter(
                g2[x_col],
                g2[y_col],
                c=style["color"],
                marker=style["marker"],
                s=50 + 15 * g2["threshold"].fillna(0),
                alpha=0.75,
                edgecolors="black" if str(profile) == "moderate" else "none",
                linewidths=0.5,
                label=f"{style['label']} ({profile})"
                if method != "neural_only"
                else style["label"],
            )


def _add_legend(ax, sub: pd.DataFrame) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="best")


def _annotate_highlights(
    ax,
    sub: pd.DataFrame,
    highlights: dict,
    x_col: str,
    y_col: str,
) -> None:
    for key, label in ANNOTATE_LABELS.items():
        row = highlights.get(key)
        if row is None or pd.isna(row.get(x_col)) or pd.isna(row.get(y_col)):
            continue
        if row["config_id"] not in sub["config_id"].values:
            continue
        ax.scatter(
            row[x_col],
            row[y_col],
            s=180,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            zorder=5,
        )
        ax.annotate(
            label,
            (row[x_col], row[y_col]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="darkred",
        )


def plot_precision_recall(df: pd.DataFrame, split: str, fig_dir: Path) -> None:
    sub = df[df["split"] == split].dropna(subset=["precision", "recall"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter_encode(ax, sub, "recall", "precision")
    highlights = find_highlight_configs(sub)
    _annotate_highlights(ax, sub, highlights, "recall", "precision")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision–Recall trade-off ({split})")
    ax.grid(True, alpha=0.3)
    _add_legend(ax, sub)
    savefig(fig, fig_dir / f"precision_recall_scatter_{split}.png")
    savefig(fig, fig_dir / f"precision_recall_scatter_{split}.pdf")


def plot_fp_vs_recall(df: pd.DataFrame, split: str, fig_dir: Path) -> None:
    sub = df[df["split"] == split].dropna(subset=["fp", "recall"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter_encode(ax, sub, "recall", "fp")
    highlights = find_highlight_configs(sub)
    _annotate_highlights(ax, sub, highlights, "recall", "fp")
    ax.set_xlabel("Recall")
    ax.set_ylabel("False positives")
    ax.set_title(f"False positives vs recall ({split})")
    ax.grid(True, alpha=0.3)
    _add_legend(ax, sub)
    savefig(fig, fig_dir / f"fp_vs_recall_{split}.png")
    savefig(fig, fig_dir / f"fp_vs_recall_{split}.pdf")


def plot_fp_reduction_vs_recall_loss(df: pd.DataFrame, split: str, fig_dir: Path) -> None:
    sub = df[(df["split"] == split) & (df["method"] != "neural_only")].copy()
    sub = sub.dropna(subset=["fp_reduction_rate", "recall_loss_rate"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter_encode(ax, sub, "recall_loss_rate", "fp_reduction_rate")
    ax.axvline(0.25, color="#999999", linestyle="--", linewidth=1, label="recall loss 25%")
    ax.axhline(0.25, color="#bbbbbb", linestyle="--", linewidth=1, label="FP reduction 25%")
    highlights = find_highlight_configs(sub)
    low_loss = sub[sub["recall_loss_rate"] <= 0.25]
    if not low_loss.empty:
        best = low_loss.loc[low_loss["fp_reduction_rate"].idxmax()]
        ax.scatter(
            best["recall_loss_rate"],
            best["fp_reduction_rate"],
            s=200,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            zorder=5,
        )
        ax.annotate(
            "best FP↓ @ R-loss≤25%",
            (best["recall_loss_rate"], best["fp_reduction_rate"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color="darkred",
        )
    ax.set_xlabel("Recall loss rate (vs neural-only @ same τ)")
    ax.set_ylabel("FP reduction rate (vs neural-only @ same τ)")
    ax.set_title(f"Governance trade-off: FP reduction vs recall loss ({split})")
    ax.grid(True, alpha=0.3)
    _add_legend(ax, sub)
    savefig(fig, fig_dir / f"fp_reduction_vs_recall_loss_{split}.png")
    savefig(fig, fig_dir / f"fp_reduction_vs_recall_loss_{split}.pdf")


def _plot_threshold_metric(
    data: pd.DataFrame,
    metric: str,
    title: str,
    path_stem: Path,
) -> None:
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for (method, profile), group in data.groupby(["method", "symbolic_profile"]):
        g = group.sort_values("threshold")
        style = get_style(method, str(profile))
        label = config_display_label(g.iloc[0]) if len(g) == 1 else f"{method_label(method)} ({profile})"
        ax.plot(
            g["threshold"],
            g[metric],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            label=label,
            linewidth=2,
            markersize=6,
        )
    ax.set_xlabel("Neural threshold")
    ax.set_ylabel(metric.capitalize())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    savefig(fig, path_stem.with_suffix(".png"))
    savefig(fig, path_stem.with_suffix(".pdf"))


def plot_threshold_curves(df: pd.DataFrame, split: str, fig_dir: Path) -> None:
    sub = df[df["split"] == split].dropna(subset=["threshold", "f1"])

    neural = sub[sub["method"] == "neural_only"].sort_values("threshold")
    if not neural.empty:
        _plot_threshold_metric(
            neural,
            "f1",
            f"F1 vs threshold — neural-only ({split})",
            fig_dir / f"threshold_curves_neural_only_{split}",
        )
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        for ax, metric in zip(axes, ("precision", "recall", "f1")):
            ax.plot(neural["threshold"], neural[metric], "o-", color="#4d4d4d", linewidth=2)
            ax.set_xlabel("Threshold")
            ax.set_ylabel(metric.capitalize())
            ax.set_title(metric.capitalize())
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Neural-only threshold curves ({split})")
        fig.tight_layout()
        savefig(fig, fig_dir / f"threshold_curves_neural_only_metrics_{split}.png")

    for profile in sub["symbolic_profile"].dropna().unique():
        if profile == "none":
            continue
        inv = sub[
            (sub["method"] == "neuro_symbolic_invalid_blocks")
            & (sub["symbolic_profile"] == profile)
        ].sort_values("threshold")
        if not inv.empty:
            _plot_threshold_metric(
                inv,
                "f1",
                f"F1 vs threshold — invalid-blocks ({profile}, {split})",
                fig_dir / f"threshold_curves_invalid_blocks_{profile}_{split}",
            )

    strict = sub[sub["method"] == "neuro_symbolic_strict_two_threshold"]
    for profile in strict["symbolic_profile"].dropna().unique():
        for tau in sorted(strict["tau_high"].dropna().unique()):
            g = strict[
                (strict["symbolic_profile"] == profile) & (strict["tau_high"] == tau)
            ].sort_values("threshold")
            if g.empty:
                continue
            tau_str = str(tau).replace(".", "p")
            _plot_threshold_metric(
                g,
                "f1",
                f"F1 vs threshold — strict {profile} τ_high={tau:.2f} ({split})",
                fig_dir / f"threshold_curves_strict_{profile}_tauhigh_{tau_str}_{split}",
            )

    # Compact F1 summary: invalid_blocks profiles + strict tau=0.9
    fig, ax = plt.subplots(figsize=(9, 6))
    if not neural.empty:
        ax.plot(
            neural["threshold"],
            neural["f1"],
            "o-",
            color=METHOD_STYLES["neural_only"]["color"],
            label="Neural-only",
            linewidth=2,
        )
    for profile in ("conservative", "moderate"):
        inv = sub[
            (sub["method"] == "neuro_symbolic_invalid_blocks")
            & (sub["symbolic_profile"] == profile)
        ].sort_values("threshold")
        if not inv.empty:
            style = get_style("neuro_symbolic_invalid_blocks", profile)
            ax.plot(
                inv["threshold"],
                inv["f1"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                label=f"Invalid-blocks ({profile})",
                linewidth=2,
            )
    strict90 = strict[np.isclose(strict["tau_high"], 0.9)].sort_values("threshold")
    for profile, g in strict90.groupby("symbolic_profile"):
        style = get_style("neuro_symbolic_strict_two_threshold", str(profile))
        ax.plot(
            g["threshold"],
            g["f1"],
            marker=style["marker"],
            linestyle="--",
            color=style["color"],
            label=f"Strict τ_high=0.90 ({profile})",
            linewidth=2,
        )
    ax.set_xlabel("Neural threshold")
    ax.set_ylabel("F1")
    ax.set_title(f"F1 by method (compact summary, {split})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    savefig(fig, fig_dir / f"threshold_f1_by_method_{split}.png")
    savefig(fig, fig_dir / f"threshold_f1_by_method_{split}.pdf")


def plot_top_configs(df: pd.DataFrame, split: str, fig_dir: Path, top_k: int) -> pd.DataFrame:
    top = select_top_configs(df, split, top_k)
    if top.empty:
        return top
    fig_h = max(6, 0.35 * len(top))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y_labels = [config_display_label(row) for _, row in top.iterrows()]
    ax.barh(range(len(top)), top["governance_score"], color="#5b8db8", alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Governance score (F1 + 0.25·FP↓ − 0.25·R-loss)")
    ax.set_title(f"Top {len(top)} configurations ({split})")
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(
            row["governance_score"] + 0.01,
            i,
            f"F1={row['f1']:.3f} P={row['precision']:.3f} R={row['recall']:.3f} FP={int(row['fp'])}",
            va="center",
            fontsize=7,
        )
    fig.tight_layout()
    savefig(fig, fig_dir / f"top_configs_{split}.png")
    savefig(fig, fig_dir / f"top_configs_{split}.pdf")
    return top


def plot_valid_vs_test(df: pd.DataFrame, fig_dir: Path) -> None:
    if not {"valid", "test"}.issubset(set(df["split"].unique())):
        return
    valid = df[df["split"] == "valid"].set_index("config_id")
    test = df[df["split"] == "test"].set_index("config_id")
    common = valid.index.intersection(test.index)
    if len(common) == 0:
        return
    merged = valid.loc[common].join(
        test.loc[common],
        lsuffix="_valid",
        rsuffix="_test",
        how="inner",
    )

    specs = [
        ("f1_valid", "f1_test", "valid_vs_test_f1", "F1"),
        ("precision_valid", "precision_test", "valid_vs_test_precision", "Precision"),
        ("fp_valid", "fp_test", "valid_vs_test_fp", "False positives"),
    ]
    for x_col, y_col, stem, ylab in specs:
        fig, ax = plt.subplots(figsize=(8, 6))
        for method, group in merged.groupby("method_valid"):
            style = get_style(method, "none")
            ax.scatter(
                group[x_col],
                group[y_col],
                c=style["color"],
                marker=style["marker"],
                alpha=0.7,
                s=40,
                label=method_label(method),
            )
        lims = [
            min(merged[x_col].min(), merged[y_col].min()) * 0.95,
            max(merged[x_col].max(), merged[y_col].max()) * 1.05,
        ]
        if ylab != "False positives":
            ax.plot(lims, lims, "k--", alpha=0.4, label="y=x")
        ax.set_xlabel(f"Valid {ylab}")
        ax.set_ylabel(f"Test {ylab}")
        ax.set_title(f"Valid vs test {ylab.lower()}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        savefig(fig, fig_dir / f"{stem}.png")
        savefig(fig, fig_dir / f"{stem}.pdf")


def _print_test_summary(df: pd.DataFrame) -> None:
    test = df[df["split"] == "test"]
    if test.empty:
        print("\nNo test split rows.")
        return
    h = find_highlight_configs(test)
    scored = test.copy()
    scored["governance_score"] = compute_governance_scores(scored)
    print("\nMost interesting test configurations:")
    items = [
        ("1. Best F1", h.get("best_f1")),
        ("2. Best precision with recall >= 0.50", h.get("best_precision_recall50")),
        ("3. Best FP reduction with recall loss <= 25%", h.get("best_fp_reduction")),
        ("4. Best governance score", h.get("best_governance_score")),
    ]
    for label, row in items:
        if row is None:
            print(f"  {label}: none")
        else:
            print(f"  {label}: {config_display_label(row)} (F1={row['f1']:.4f}, FP={int(row['fp'])})")


def main() -> int:
    args = parse_args()
    sweep_csv = (
        OUTPUTS_DIR
        / "tables"
        / "sweeps"
        / args.variant
        / args.run_name
        / "decision_config_sweep.csv"
    )
    fig_dir = ensure_dir(
        OUTPUTS_DIR / "figures" / "sweeps" / args.variant / args.run_name
    )
    table_dir = OUTPUTS_DIR / "tables" / "sweeps" / args.variant / args.run_name

    if not sweep_csv.exists():
        print(f"Missing sweep CSV: {sweep_csv}", file=sys.stderr)
        print("Run: python scripts/10_sweep_decision_configs.py", file=sys.stderr)
        return 1

    df = load_sweep_results(sweep_csv)
    splits = _splits_to_plot(df, args.split)
    if not splits:
        print("No matching splits to plot.", file=sys.stderr)
        return 1

    selected_frames: list[pd.DataFrame] = []

    for split in splits:
        plot_precision_recall(df, split, fig_dir)
        plot_fp_vs_recall(df, split, fig_dir)
        plot_fp_reduction_vs_recall_loss(df, split, fig_dir)
        plot_threshold_curves(df, split, fig_dir)
        top = plot_top_configs(df, split, fig_dir, args.top_k)
        if not top.empty:
            selected_frames.append(top)

    if args.split in ("all", "valid"):
        plot_valid_vs_test(df, fig_dir)

    if selected_frames:
        selected = pd.concat(selected_frames, ignore_index=True)
        out_csv = table_dir / "plot_selected_configs.csv"
        selected.to_csv(out_csv, index=False)

    print("SWEEP PLOTS COMPLETE\n")
    print(f"Loaded:\n  {sweep_csv.relative_to(PROJECT_ROOT)} ({len(df)} rows)")
    print(f"\nSaved figures to:\n  {fig_dir.relative_to(PROJECT_ROOT)}")
    if selected_frames:
        print(
            f"\nSelected configs table:\n  "
            f"{(table_dir / 'plot_selected_configs.csv').relative_to(PROJECT_ROOT)}"
        )
    _print_test_summary(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
