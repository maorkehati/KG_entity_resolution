"""Select best validation-F1 configs per method family and plot test F1."""

from __future__ import annotations

import argparse
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

BASELINE_NAMES = [
    "always_negative",
    "exact_title_match",
    "bow_cosine",
    "tfidf_cosine_word",
    "tfidf_cosine_char",
    "fieldwise_lexical_heuristic",
]

BASELINE_META: dict[str, tuple[str, str]] = {
    "always_negative": ("Always negative", "naive_baseline"),
    "exact_title_match": ("Exact title", "naive_baseline"),
    "bow_cosine": ("BoW cosine", "lexical_baseline"),
    "tfidf_cosine_word": ("TF-IDF cosine (word)", "lexical_baseline"),
    "tfidf_cosine_char": ("TF-IDF cosine (char)", "lexical_baseline"),
    "fieldwise_lexical_heuristic": ("Field-wise heuristic", "lexical_baseline"),
}

DISPLAY_ORDER = [
    "Always negative",
    "Exact title",
    "BoW cosine",
    "TF-IDF cosine (word)",
    "TF-IDF cosine (char)",
    "Field-wise heuristic",
    "Learned scorer",
    "Neuro-symbolic, moderate",
    "Two-threshold governance",
    "Neuro-symbolic, conservative",
]

GROUP_HATCH: dict[str, str] = {
    "naive_baseline": "",
    "lexical_baseline": "///",
    "learned_ablation": "xx",
    "symbolic_ablation": "..",
    "proposed": "",
    "extension": "++",
}

GROUP_FACE: dict[str, str] = {
    "naive_baseline": "#d9d9d9",
    "lexical_baseline": "#bdbdbd",
    "learned_ablation": "#9e9e9e",
    "symbolic_ablation": "#757575",
    "proposed": "#4a90d9",
    "extension": "#f0ad4e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot main test F1 comparison after validation-based selection."
    )
    parser.add_argument("--variant", default=DEFAULT_WDC_VARIANT_ID)
    parser.add_argument("--run-name", default="neural_logreg")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _thresholds_match(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if isinstance(left, str) or isinstance(right, str):
        return str(left) == str(right)
    try:
        return bool(np.isclose(float(left), float(right)))
    except (TypeError, ValueError):
        return str(left) == str(right)


def _select_best_valid(valid_df: pd.DataFrame) -> pd.Series:
    if valid_df.empty:
        raise ValueError("No validation rows to select from.")
    ranked = valid_df.sort_values(
        ["f1", "precision", "fp", "recall", "threshold"],
        ascending=[False, False, True, False, True],
        kind="mergesort",
    )
    return ranked.iloc[0]


def _row_to_record(
    *,
    group: str,
    display_name: str,
    source: str,
    valid_row: pd.Series,
    test_row: pd.Series,
    config_id: str = "",
) -> dict[str, Any]:
    tau = valid_row.get("tau_high", np.nan)
    if pd.isna(tau):
        tau_out = ""
    else:
        tau_out = tau
    return {
        "group": group,
        "display_name": display_name,
        "source": source,
        "selected_by": "validation_f1",
        "validation_f1": valid_row["f1"],
        "validation_precision": valid_row["precision"],
        "validation_recall": valid_row["recall"],
        "validation_fp": valid_row["fp"],
        "test_f1": test_row["f1"],
        "test_precision": test_row["precision"],
        "test_recall": test_row["recall"],
        "test_fp": test_row["fp"],
        "test_fn": test_row["fn"],
        "threshold": valid_row.get("threshold", ""),
        "tau_high": tau_out,
        "symbolic_profile": valid_row.get("symbolic_profile", ""),
        "decision_mode": valid_row.get("decision_mode", ""),
        "config_id": config_id,
    }


def _select_baseline_family(
    baseline_metrics: pd.DataFrame,
    baseline_name: str,
) -> dict[str, Any] | None:
    valid = baseline_metrics[
        (baseline_metrics["baseline"] == baseline_name)
        & (baseline_metrics["split"] == "valid")
    ]
    if valid.empty:
        return None
    valid_row = _select_best_valid(valid)
    test = baseline_metrics[
        (baseline_metrics["baseline"] == baseline_name)
        & (baseline_metrics["split"] == "test")
    ]
    if test.empty:
        warnings.warn(f"No test rows for baseline '{baseline_name}'.")
        return None
    matches = test[
        test.apply(
            lambda r: _thresholds_match(r["threshold"], valid_row["threshold"])
            and str(r.get("threshold_objective", ""))
            == str(valid_row.get("threshold_objective", "")),
            axis=1,
        )
    ]
    if matches.empty:
        warnings.warn(
            f"Could not match test row for baseline '{baseline_name}' "
            f"(threshold={valid_row['threshold']})."
        )
        return None
    test_row = matches.iloc[0]
    display, group = BASELINE_META[baseline_name]
    return _row_to_record(
        group=group,
        display_name=display,
        source="baseline_metrics",
        valid_row=valid_row,
        test_row=test_row,
    )


def _select_sweep_family(
    sweep: pd.DataFrame,
    *,
    display_name: str,
    group: str,
    mask: pd.Series,
) -> dict[str, Any] | None:
    valid = sweep[(sweep["split"] == "valid") & mask]
    if valid.empty:
        return None
    valid_row = _select_best_valid(valid)
    test = sweep[(sweep["split"] == "test") & (sweep["config_id"] == valid_row["config_id"])]
    if test.empty:
        warnings.warn(
            f"No test row for config_id '{valid_row['config_id']}' ({display_name})."
        )
        return None
    test_row = test.iloc[0]
    return _row_to_record(
        group=group,
        display_name=display_name,
        source="decision_config_sweep",
        valid_row=valid_row,
        test_row=test_row,
        config_id=str(valid_row["config_id"]),
    )


def build_comparison_table(
    baseline_metrics: pd.DataFrame,
    sweep: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for name in BASELINE_NAMES:
        if name not in baseline_metrics["baseline"].unique():
            warnings.warn(f"Baseline '{name}' missing from metrics; skipping.")
            continue
        rec = _select_baseline_family(baseline_metrics, name)
        if rec is not None:
            rows.append(rec)

    sweep_specs = [
        (
            "Learned scorer",
            "learned_ablation",
            sweep["method"] == "neural_only",
        ),
        (
            "Neuro-symbolic, conservative",
            "proposed",
            (sweep["method"] == "neuro_symbolic_invalid_blocks")
            & (sweep["symbolic_profile"] == "conservative")
            & (sweep["decision_mode"] == "invalid_blocks"),
        ),
        (
            "Neuro-symbolic, moderate",
            "symbolic_ablation",
            (sweep["method"] == "neuro_symbolic_invalid_blocks")
            & (sweep["symbolic_profile"] == "moderate")
            & (sweep["decision_mode"] == "invalid_blocks"),
        ),
        (
            "Two-threshold governance",
            "extension",
            (sweep["method"] == "neuro_symbolic_strict_two_threshold")
            & (sweep["symbolic_profile"] == "conservative")
            & (sweep["decision_mode"] == "strict_valid_or_high_confidence"),
        ),
    ]
    for display_name, group, mask in sweep_specs:
        rec = _select_sweep_family(sweep, display_name=display_name, group=group, mask=mask)
        if rec is None:
            warnings.warn(f"No validation configs for '{display_name}'; skipping.")
        else:
            rows.append(rec)

    if not rows:
        raise RuntimeError("No methods selected for comparison.")

    df = pd.DataFrame(rows)
    order_map = {name: i for i, name in enumerate(DISPLAY_ORDER)}
    df["_sort"] = df["display_name"].map(lambda x: order_map.get(x, 999))
    return df.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


def _save_figure(fig: plt.Figure, png_path: Path, pdf_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_test_f1_comparison(df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    names = df["display_name"].tolist()
    f1s = df["test_f1"].astype(float).tolist()
    groups = df["group"].tolist()

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.95), 5.5))
    x = np.arange(len(names))
    colors = [GROUP_FACE.get(g, "#cccccc") for g in groups]
    hatches = [GROUP_HATCH.get(g, "") for g in groups]
    bars = ax.bar(
        x,
        f1s,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        zorder=2,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    prev_group = None
    for i, g in enumerate(groups):
        if prev_group is not None and g != prev_group:
            ax.axvline(i - 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        prev_group = g

    for bar, val in zip(bars, f1s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Test F1")
    ax.set_ylim(0, min(1.0, max(f1s) * 1.15 + 0.05) if f1s else 1.0)
    ax.set_title("Main comparison: test F1 after validation-based selection")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    legend_handles = []
    seen: set[str] = set()
    for g in groups:
        if g in seen:
            continue
        seen.add(g)
        legend_handles.append(
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=GROUP_FACE.get(g, "#cccccc"),
                edgecolor="black",
                hatch=GROUP_HATCH.get(g, ""),
                label=g.replace("_", " "),
            )
        )
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    fig.tight_layout()
    _save_figure(fig, out_png, out_pdf)


def print_summary(df: pd.DataFrame, table_path: Path, fig_png: Path, fig_pdf: Path) -> None:
    print("MAIN METHOD COMPARISON")
    print()
    print("Selection rule:")
    print("  best validation F1 per method family")
    print()
    print("Plotted metric:")
    print("  test F1")
    print()
    print("Selected configurations:")
    print(
        "  display_name | val_f1 | test_f1 | test_precision | test_recall | test_fp | "
        "threshold | tau_high"
    )
    for _, row in df.iterrows():
        tau = row["tau_high"]
        tau_s = "" if tau == "" or (isinstance(tau, float) and pd.isna(tau)) else f"{tau:g}"
        thr = row["threshold"]
        thr_s = str(thr) if isinstance(thr, str) else f"{float(thr):g}"
        print(
            f"  {row['display_name']:<28} "
            f"{row['validation_f1']:.4f}   {row['test_f1']:.4f}   "
            f"{row['test_precision']:.4f}          {row['test_recall']:.4f}       "
            f"{int(row['test_fp']):>4}    {thr_s:<8} {tau_s}"
        )
    print()
    print("Saved table:")
    print(f"  {table_path}")
    print("Saved figures:")
    print(f"  {fig_png}")
    print(f"  {fig_pdf}")


def main() -> None:
    args = parse_args()
    variant_id = args.variant
    run_name = args.run_name

    baseline_path = (
        OUTPUTS_DIR / "tables" / "baselines" / variant_id / "baseline_metrics.csv"
    )
    sweep_path = (
        OUTPUTS_DIR
        / "tables"
        / "sweeps"
        / variant_id
        / run_name
        / "decision_config_sweep.csv"
    )

    if not baseline_path.exists():
        print("Run scripts/12_run_naive_baselines.py first.")
        sys.exit(1)
    if not sweep_path.exists():
        print("Run scripts/10_sweep_decision_configs.py first.")
        sys.exit(1)

    table_dir = OUTPUTS_DIR / "tables" / "main_comparison" / variant_id / run_name
    fig_dir = OUTPUTS_DIR / "figures" / "main_comparison" / variant_id / run_name
    table_path = table_dir / "selected_main_comparison.csv"
    fig_png = fig_dir / "main_test_f1_comparison.png"
    fig_pdf = fig_dir / "main_test_f1_comparison.pdf"

    if (
        table_path.exists()
        and fig_png.exists()
        and fig_pdf.exists()
        and not args.overwrite
    ):
        print("Outputs already exist; use --overwrite to regenerate.")
        df = pd.read_csv(table_path)
        print_summary(df, table_path, fig_png, fig_pdf)
        return

    baseline_metrics = pd.read_csv(baseline_path)
    sweep = pd.read_csv(sweep_path)
    metric_cols = ("f1", "precision", "recall", "fp", "fn", "tau_high")
    for col in metric_cols:
        if col in sweep.columns:
            sweep[col] = pd.to_numeric(sweep[col], errors="coerce")
        if col in baseline_metrics.columns:
            baseline_metrics[col] = pd.to_numeric(baseline_metrics[col], errors="coerce")
    if "threshold" in sweep.columns:
        sweep["threshold"] = pd.to_numeric(sweep["threshold"], errors="coerce")

    df = build_comparison_table(baseline_metrics, sweep)
    table_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_path, index=False)
    plot_test_f1_comparison(df, fig_png, fig_pdf)
    print_summary(df, table_path, fig_png, fig_pdf)


if __name__ == "__main__":
    main()
