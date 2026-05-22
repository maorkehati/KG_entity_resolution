"""Aggregate optional ablation experiment outputs into report-ready tables and figures."""

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
from src.evaluation import save_json

NUMERIC_COLS = [
    "precision",
    "recall",
    "f1",
    "fp",
    "fn",
    "tp",
    "tn",
    "threshold",
    "lambda_risk",
    "rho",
    "iteration",
    "validation_f1",
    "validation_precision",
    "validation_recall",
    "validation_fp",
    "test_f1",
    "test_precision",
    "test_recall",
    "test_fp",
    "test_fn",
    "num_hard_negatives",
]

COLOR_NEURAL = "#8FB3D9"
COLOR_GOVERNANCE = "#6BAE75"
COLOR_SOFT = "#E8A85C"
COLOR_STRUCTURED = "#9B7BB8"
COLOR_STRUCTURED_ALT = "#A8A8C8"

SOFT_RISK_FAMILIES = [
    ("neural_only", "Neural-only", "neural"),
    ("hard_invalid_blocks", "Hard invalid-blocks", "governance"),
    ("soft_score_penalty", "Soft score penalty", "soft"),
    ("soft_risk_gate", "Soft risk gate", "soft"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare report ablation tables and figures from existing outputs."
    )
    parser.add_argument("--variant", default=DEFAULT_WDC_VARIANT_ID)
    parser.add_argument("--run-name", default="neural_logreg")
    parser.add_argument("--hard-negative-experiment", default="symbolic_hn_strong")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_read_csv(path: Path, required: bool = False) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file missing: {path}")
        warnings.warn(f"Optional file missing: {path}")
        return None
    return pd.read_csv(path)


def coerce_numeric_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def save_plot(fig: plt.Figure, path_base: Path) -> None:
    ensure_dir(path_base.parent)
    fig.savefig(path_base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def annotate_bars(
    ax: plt.Axes,
    bars: Any,
    values: list[float],
    fmt: str = "{:.3f}",
    fp_values: list[int] | None = None,
) -> None:
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    if fp_values is not None:
        for bar, fp in zip(bars, fp_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                -0.02,
                f"FP={int(fp)}",
                ha="center",
                va="top",
                fontsize=7,
                color="#555555",
                transform=ax.get_xaxis_transform(),
            )


def save_table_csv_tex(df: pd.DataFrame, path_base: Path) -> None:
    ensure_dir(path_base.parent)
    df.to_csv(path_base.with_suffix(".csv"), index=False)
    try:
        df.to_latex(path_base.with_suffix(".tex"), index=False, escape=False)
    except Exception as exc:
        warnings.warn(f"Could not write LaTeX table {path_base}.tex: {exc}")


def _tiebreak_sort(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["f1", "precision", "fp", "recall"] if c in df.columns]
    asc = [False, False, True, False][: len(cols)]
    return df.sort_values(cols, ascending=asc, kind="mergesort")


def _format_threshold(val: Any) -> str:
    if pd.isna(val):
        return "n/a"
    return f"{float(val):.2f}".rstrip("0").rstrip(".") if float(val) != int(float(val)) else str(int(float(val)))


def _soft_risk_config(row: pd.Series, family: str) -> str:
    thr = _format_threshold(row.get("threshold", row.get("tau", "")))
    if family == "neural_only":
        return f"τ={thr}"
    if family == "hard_invalid_blocks":
        return f"τ={thr}"
    if family == "soft_score_penalty":
        lam = row.get("lambda_risk", "")
        return f"τ={thr}, λ={_format_threshold(lam) if not pd.isna(lam) else lam}"
    if family == "soft_risk_gate":
        rho = row.get("rho", "")
        return f"τ={thr}, ρ={_format_threshold(rho) if not pd.isna(rho) else rho}"
    return ""


def _select_soft_risk_from_grid(grid_df: pd.DataFrame, mode: str) -> pd.Series | None:
    work = grid_df[grid_df["mode"] == mode].copy()
    if work.empty:
        return None
    valid = work[work["split"] == "valid"]
    if valid.empty:
        return None
    best_valid = _tiebreak_sort(valid).iloc[0]
    config_id = best_valid.get("config_id")
    test = work[work["split"] == "test"]
    if config_id is not None and "config_id" in test.columns:
        match = test[test["config_id"] == config_id]
        if not match.empty:
            return match.iloc[0]
    match = test[
        np.isclose(test["threshold"], best_valid["threshold"], rtol=0, atol=1e-9)
    ]
    if mode == "soft_score_penalty" and "lambda_risk" in test.columns:
        match = match[np.isclose(match["lambda_risk"], best_valid["lambda_risk"], rtol=0, atol=1e-9)]
    if mode == "soft_risk_gate" and "rho" in test.columns:
        match = match[np.isclose(match["rho"], best_valid["rho"], rtol=0, atol=1e-9)]
    return match.iloc[0] if not match.empty else None


def prepare_soft_risk_ablation(
    variant_id: str,
    run_name: str,
    table_dir: Path,
    fig_dir: Path,
    overwrite: bool,
) -> dict[str, Any] | None:
    base = OUTPUTS_DIR / "tables" / "soft_risk" / variant_id / run_name
    selected_path = base / "soft_risk_selected_configs.csv"
    grid_path = base / "soft_risk_grid_results.csv"

    table_out = table_dir / "soft_risk_ablation_table"
    fig_out = fig_dir / "soft_risk_ablation_test_f1"
    if not overwrite and table_out.with_suffix(".csv").exists() and fig_out.with_suffix(".png").exists():
        return {"table": table_out.with_suffix(".csv"), "figure": fig_out.with_suffix(".png"), "skipped": True}

    selected_df = safe_read_csv(selected_path)
    grid_df = safe_read_csv(grid_path)
    if selected_df is None and grid_df is None:
        warnings.warn(
            f"Soft risk ablation skipped: no files under {base}"
        )
        return None

    rows: list[dict[str, Any]] = []
    for family, method_label, color_key in SOFT_RISK_FAMILIES:
        row_data: dict[str, Any] | None = None
        if selected_df is not None and not selected_df.empty:
            sub = selected_df[selected_df["method_family"] == family]
            if not sub.empty:
                row_data = sub.iloc[0]
        elif grid_df is not None:
            test_row = _select_soft_risk_from_grid(coerce_numeric_metrics(grid_df), family)
            if test_row is not None:
                row_data = test_row
        if row_data is None:
            warnings.warn(f"Soft risk: no row for method family {family}")
            continue

        prec = float(row_data.get("test_precision", row_data.get("precision", np.nan)))
        rec = float(row_data.get("test_recall", row_data.get("recall", np.nan)))
        f1 = float(row_data.get("test_f1", row_data.get("f1", np.nan)))
        fp = int(row_data.get("test_fp", row_data.get("fp", 0)))
        rows.append(
            {
                "Method": method_label,
                "Configuration": _soft_risk_config(row_data, family),
                "Precision": round(prec, 3),
                "Recall": round(rec, 3),
                "F1": round(f1, 3),
                "FP": fp,
                "_color_key": color_key,
                "_f1_plot": f1,
            }
        )

    if not rows:
        return None

    table_df = pd.DataFrame(rows)[["Method", "Configuration", "Precision", "Recall", "F1", "FP"]]
    save_table_csv_tex(table_df, table_out)

    color_map = {
        "neural": COLOR_NEURAL,
        "governance": COLOR_GOVERNANCE,
        "soft": COLOR_SOFT,
    }
    labels = [r["Method"] for r in rows]
    f1_vals = [r["_f1_plot"] for r in rows]
    fp_vals = [r["FP"] for r in rows]
    colors = [color_map[r["_color_key"]] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, f1_vals, color=colors, edgecolor="#404040", linewidth=0.8, width=0.65)
    annotate_bars(ax, bars, f1_vals, fp_values=fp_vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Test F1")
    ymax = max(f1_vals) if f1_vals else 1.0
    ax.set_ylim(0, min(1.0, ymax * 1.2 + 0.05))
    ax.set_title("Hard symbolic vetoes vs. soft symbolic risk", fontsize=12, fontweight="bold")
    style_axes(ax)
    fig.tight_layout()
    save_plot(fig, fig_out)

    return {"table": table_out.with_suffix(".csv"), "figure": fig_out.with_suffix(".png"), "preview": table_df}


def prepare_hard_negative_ablation(
    variant_id: str,
    experiment: str,
    table_dir: Path,
    fig_dir: Path,
    overwrite: bool,
) -> dict[str, Any] | None:
    hn_dir = OUTPUTS_DIR / "tables" / "hard_negative_training" / variant_id / experiment
    metrics_path = hn_dir / "iteration_metrics.csv"
    mining_path = hn_dir / "mining_summary.csv"

    table_out = table_dir / "hard_negative_ablation_table"
    fig_out = fig_dir / "hard_negative_iterations_test_f1_fp"
    if not overwrite and table_out.with_suffix(".csv").exists() and fig_out.with_suffix(".png").exists():
        return {"table": table_out.with_suffix(".csv"), "figure": fig_out.with_suffix(".png"), "skipped": True}

    metrics_df = safe_read_csv(metrics_path)
    if metrics_df is None or metrics_df.empty:
        warnings.warn(f"Hard-negative ablation skipped: missing {metrics_path}")
        return None

    metrics_df = coerce_numeric_metrics(metrics_df)
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    if test_df.empty:
        warnings.warn("Hard-negative ablation skipped: no test split rows")
        return None

    mining_df = safe_read_csv(mining_path)
    mined_by_iter: dict[int, int] = {0: 0}
    if mining_df is not None and not mining_df.empty:
        mining_df = coerce_numeric_metrics(mining_df)
        for _, r in mining_df.iterrows():
            mined_by_iter[int(r["iteration"])] = int(r.get("num_hard_negatives", 0))

    iterations = sorted(test_df["iteration"].dropna().unique().astype(int))
    table_rows: list[dict[str, Any]] = []
    neural_series: dict[int, dict[str, float]] = {}
    governed_series: dict[int, dict[str, float]] = {}

    for it in iterations:
        neural = test_df[
            (test_df["iteration"] == it) & (test_df["decision_type"] == "neural_only")
        ]
        governed = test_df[
            (test_df["iteration"] == it)
            & (test_df["decision_type"] == "hard_governed_invalid_blocks")
        ]
        if neural.empty or governed.empty:
            warnings.warn(f"Hard-negative: incomplete rows for iteration {it}")
            continue
        n_row = neural.iloc[0]
        g_row = governed.iloc[0]
        mined = mined_by_iter.get(it, 0 if it == 0 else "")
        table_rows.append(
            {
                "Iteration": int(it),
                "Neural Precision": round(float(n_row["precision"]), 3),
                "Neural Recall": round(float(n_row["recall"]), 3),
                "Neural F1": round(float(n_row["f1"]), 3),
                "Neural FP": int(n_row["fp"]),
                "Governed Precision": round(float(g_row["precision"]), 3),
                "Governed Recall": round(float(g_row["recall"]), 3),
                "Governed F1": round(float(g_row["f1"]), 3),
                "Governed FP": int(g_row["fp"]),
                "Mined hard negatives": mined if mined != "" else 0,
            }
        )
        neural_series[it] = {"f1": float(n_row["f1"]), "fp": float(n_row["fp"])}
        governed_series[it] = {"f1": float(g_row["f1"]), "fp": float(g_row["fp"])}

    if not table_rows:
        return None

    table_df = pd.DataFrame(table_rows)
    save_table_csv_tex(table_df, table_out)

    iters = sorted(neural_series.keys())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax_f1 = axes[0]
    ax_f1.plot(
        iters,
        [neural_series[i]["f1"] for i in iters],
        marker="o",
        color=COLOR_NEURAL,
        label="Neural-only",
        linewidth=1.8,
    )
    ax_f1.plot(
        iters,
        [governed_series[i]["f1"] for i in iters],
        marker="s",
        color=COLOR_GOVERNANCE,
        label="Governed",
        linewidth=1.8,
    )
    ax_f1.set_xlabel("Iteration")
    ax_f1.set_ylabel("Test F1")
    ax_f1.set_xticks(iters)
    ax_f1.legend(frameon=False, fontsize=8)
    style_axes(ax_f1)

    ax_fp = axes[1]
    ax_fp.plot(
        iters,
        [neural_series[i]["fp"] for i in iters],
        marker="o",
        color=COLOR_NEURAL,
        label="Neural-only",
        linewidth=1.8,
    )
    ax_fp.plot(
        iters,
        [governed_series[i]["fp"] for i in iters],
        marker="s",
        color=COLOR_GOVERNANCE,
        label="Governed",
        linewidth=1.8,
    )
    ax_fp.set_xlabel("Iteration")
    ax_fp.set_ylabel("Test false positives")
    ax_fp.set_xticks(iters)
    ax_fp.legend(frameon=False, fontsize=8)
    style_axes(ax_fp)

    fig.suptitle(
        "Symbolically guided hard-negative training over iterations",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    save_plot(fig, fig_out)

    return {"table": table_out.with_suffix(".csv"), "figure": fig_out.with_suffix(".png"), "preview": table_df}


def _arch_row_from_test(model_name: str, test_df: pd.DataFrame) -> dict[str, Any] | None:
    sub = test_df[test_df["model_name"] == model_name]
    if sub.empty:
        return None
    r = sub.iloc[0]
    return {
        "Precision": round(float(r["test_precision"]), 3),
        "Recall": round(float(r["test_recall"]), 3),
        "F1": round(float(r["test_f1"]), 3),
        "FP": int(r["test_fp"]),
        "_f1_plot": float(r["test_f1"]),
    }


def _arch_row_from_governed(model_key: str, gov_df: pd.DataFrame) -> dict[str, Any] | None:
    sub = gov_df[gov_df["model_name"] == model_key]
    if sub.empty:
        return None
    r = sub.iloc[0]
    return {
        "Precision": round(float(r["precision"]), 3),
        "Recall": round(float(r["recall"]), 3),
        "F1": round(float(r["f1"]), 3),
        "FP": int(r["fp"]),
        "_f1_plot": float(r["f1"]),
    }


def prepare_architecture_ablation(
    variant_id: str,
    table_dir: Path,
    fig_dir: Path,
    overwrite: bool,
) -> dict[str, Any] | None:
    struct_dir = OUTPUTS_DIR / "tables" / "structured_models" / variant_id
    test_path = struct_dir / "structured_model_test_comparison.csv"
    gov_path = struct_dir / "structured_model_governed_comparison.csv"
    metrics_path = struct_dir / "structured_model_metrics.csv"

    table_out = table_dir / "architecture_ablation_table"
    fig_out = fig_dir / "architecture_ablation_test_f1"
    if not overwrite and table_out.with_suffix(".csv").exists() and fig_out.with_suffix(".png").exists():
        return {"table": table_out.with_suffix(".csv"), "figure": fig_out.with_suffix(".png"), "skipped": True}

    test_df = safe_read_csv(test_path)
    gov_df = safe_read_csv(gov_path)
    metrics_df = safe_read_csv(metrics_path)

    if test_df is None and metrics_df is None:
        warnings.warn(f"Architecture ablation skipped: no files under {struct_dir}")
        return None

    if test_df is not None:
        test_df = coerce_numeric_metrics(test_df)
    if gov_df is not None:
        gov_df = coerce_numeric_metrics(gov_df)

    specs: list[tuple[str, str, str, str]] = [
        ("Flat learned scorer", "existing_scorer", "", "neural"),
        ("Field attention", "field_attention", "", "structured"),
        ("Structured transformer", "structured_transformer", "", "structured_alt"),
        (
            "Field attention + governance",
            "",
            "field_attention_governed",
            "governance",
        ),
        (
            "Structured transformer + governance",
            "",
            "structured_transformer_governed",
            "governance",
        ),
    ]

    rows: list[dict[str, Any]] = []
    for model_label, test_key, gov_key, style in specs:
        metrics: dict[str, Any] | None = None
        if test_key and test_df is not None:
            metrics = _arch_row_from_test(test_key, test_df)
        if metrics is None and gov_key and gov_df is not None:
            metrics = _arch_row_from_governed(gov_key, gov_df)
        if metrics is None and metrics_df is not None and test_key:
            long_df = coerce_numeric_metrics(metrics_df)
            valid = long_df[
                (long_df["model_name"] == test_key) & (long_df["split"] == "valid")
            ]
            test_long = long_df[
                (long_df["model_name"] == test_key) & (long_df["split"] == "test")
            ]
            if not valid.empty and not test_long.empty:
                best = _tiebreak_sort(valid).iloc[0]
                thr = best["threshold"]
                match = test_long[np.isclose(test_long["threshold"], thr, rtol=0, atol=1e-9)]
                if not match.empty:
                    r = match.iloc[0]
                    metrics = {
                        "Precision": round(float(r["precision"]), 3),
                        "Recall": round(float(r["recall"]), 3),
                        "F1": round(float(r["f1"]), 3),
                        "FP": int(r["fp"]),
                        "_f1_plot": float(r["f1"]),
                    }
        if metrics is None:
            warnings.warn(f"Architecture ablation: missing metrics for {model_label}")
            continue
        rows.append({"Model": model_label, **metrics, "_style": style})

    flat_gov = None
    if gov_df is not None:
        flat_gov = _arch_row_from_governed("existing_scorer_governed", gov_df)
    if flat_gov is not None:
        rows.append(
            {
                "Model": "Flat learned scorer + governance",
                **flat_gov,
                "_style": "governance",
                "_extra": True,
            }
        )

    if not rows:
        return None

    report_rows = [r for r in rows if not r.get("_extra")]
    table_df = pd.DataFrame(report_rows)[["Model", "Precision", "Recall", "F1", "FP"]]
    full_df = pd.DataFrame(rows)[["Model", "Precision", "Recall", "F1", "FP"]]
    save_table_csv_tex(full_df, table_out)

    style_colors = {
        "neural": COLOR_NEURAL,
        "structured": COLOR_STRUCTURED,
        "structured_alt": COLOR_STRUCTURED_ALT,
        "governance": COLOR_GOVERNANCE,
    }
    plot_rows = report_rows
    labels = [r["Model"] for r in plot_rows]
    f1_vals = [r["_f1_plot"] for r in plot_rows]
    fp_vals = [r["FP"] for r in plot_rows]
    colors = [style_colors[r["_style"]] for r in plot_rows]
    hatches = ["", "", "", "\\\\", "\\\\"]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        f1_vals,
        color=colors,
        edgecolor="#404040",
        linewidth=0.8,
        width=0.65,
        hatch=[hatches[i] if i < len(hatches) else "" for i in range(len(labels))],
    )
    annotate_bars(ax, bars, f1_vals, fp_values=fp_vals)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            l.replace(" + governance", "\n+ governance").replace(
                "Structured transformer", "Struct. transformer"
            )
            for l in labels
        ],
        rotation=25,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Test F1")
    ymax = max(f1_vals) if f1_vals else 1.0
    ax.set_ylim(0, min(1.0, ymax * 1.22 + 0.05))
    ax.set_title("Field-aware neural architectures", fontsize=12, fontweight="bold")
    style_axes(ax)
    fig.tight_layout()
    save_plot(fig, fig_out)

    return {
        "table": table_out.with_suffix(".csv"),
        "figure": fig_out.with_suffix(".png"),
        "preview": table_df,
    }


def _print_preview(name: str, df: pd.DataFrame | None) -> None:
    if df is None or df.empty:
        return
    print(f"\n{name} preview:")
    text = df.to_string(index=False)
    print(text.encode("ascii", "replace").decode("ascii"))


def main() -> int:
    args = parse_args()
    variant_id = args.variant
    run_name = args.run_name
    experiment = args.hard_negative_experiment

    table_dir = ensure_dir(
        OUTPUTS_DIR / "tables" / "report_ablations" / variant_id / run_name
    )
    fig_dir = ensure_dir(
        OUTPUTS_DIR / "figures" / "report_ablations" / variant_id / run_name
    )

    warn_msgs: list[str] = []
    results: dict[str, Any] = {}

    soft = prepare_soft_risk_ablation(
        variant_id, run_name, table_dir, fig_dir, args.overwrite
    )
    if soft is None:
        warn_msgs.append("Soft risk ablation: skipped (inputs missing).")
    else:
        results["soft_risk"] = {
            "table": str(soft["table"]),
            "figure": str(soft["figure"]),
        }

    hn = prepare_hard_negative_ablation(
        variant_id, experiment, table_dir, fig_dir, args.overwrite
    )
    if hn is None:
        warn_msgs.append("Hard-negative ablation: skipped (inputs missing).")
    else:
        results["hard_negative"] = {
            "table": str(hn["table"]),
            "figure": str(hn["figure"]),
        }

    arch = prepare_architecture_ablation(variant_id, table_dir, fig_dir, args.overwrite)
    if arch is None:
        warn_msgs.append("Architecture ablation: skipped (inputs missing).")
    else:
        results["architecture"] = {
            "table": str(arch["table"]),
            "figure": str(arch["figure"]),
        }

    metadata = {
        "variant_id": variant_id,
        "run_name": run_name,
        "hard_negative_experiment": experiment,
        "selection_rule": "validation_f1",
        "reported_split": "test",
        "subsections": results,
        "warnings": warn_msgs,
    }
    save_json(metadata, table_dir / "report_ablation_metadata.json")

    print("REPORT ABLATION RESULTS PREPARED")
    print("\nSoft risk:")
    if soft:
        print(f"  table: {soft['table']}")
        print(f"  figure: {soft['figure']}")
        _print_preview("Soft risk", soft.get("preview"))
    else:
        print("  (skipped)")

    print("\nHard-negative training:")
    if hn:
        print(f"  table: {hn['table']}")
        print(f"  figure: {hn['figure']}")
        _print_preview("Hard-negative", hn.get("preview"))
    else:
        print("  (skipped)")

    print("\nArchitectures:")
    if arch:
        print(f"  table: {arch['table']}")
        print(f"  figure: {arch['figure']}")
        _print_preview("Architectures", arch.get("preview"))
    else:
        print("  (skipped)")

    if warn_msgs:
        print("\nWarnings:")
        for w in warn_msgs:
            print(f"  {w}")

    print(f"\nOutputs: {table_dir}")
    print(f"         {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
