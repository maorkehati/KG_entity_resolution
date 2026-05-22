"""Generate three final report-ready test-F1 comparison figures."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_WDC_VARIANT_ID, OUTPUTS_DIR
from src.evaluation import compute_binary_metrics_from_predictions, save_json
from src.symbolic import ProductConstraintChecker, make_symbolic_config

GROUP_STYLE: dict[str, dict[str, str]] = {
    "baseline": {"color": "#B8B8B8", "edgecolor": "#404040", "hatch": ""},
    "symbolic-only": {"color": "#D9D9D9", "edgecolor": "#404040", "hatch": "///"},
    "neural-only": {"color": "#8FB3D9", "edgecolor": "#26384D", "hatch": ""},
    "ablation": {"color": "#C9B6E4", "edgecolor": "#4A3A61", "hatch": ".."},
    "neuro-symbolic": {"color": "#6BAE75", "edgecolor": "#1F4D2B", "hatch": ""},
    "variant": {"color": "#F2B66D", "edgecolor": "#6B3F0B", "hatch": "\\\\"},
    "extension": {"color": "#E6A0A0", "edgecolor": "#6B2424", "hatch": "xx"},
}

FALLBACK_STYLE = {"color": "#CCCCCC", "edgecolor": "#404040", "hatch": ""}

LEGEND_ORDER = [
    "baseline",
    "symbolic-only",
    "neural-only",
    "ablation",
    "neuro-symbolic",
    "variant",
    "extension",
]

GROUP_GAP = 0.5
BAR_EDGE_WIDTH = 0.9

METRIC_COLS = ["precision", "recall", "f1", "fp", "fn", "tp", "tn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final comparison figures (test F1).")
    parser.add_argument("--variant", default=DEFAULT_WDC_VARIANT_ID)
    parser.add_argument("--run-name", default="neural_logreg")
    parser.add_argument(
        "--hard-negative-experiment",
        default="symbolic_hn_strong",
    )
    parser.add_argument("--structured-models-enabled", action="store_true")
    parser.add_argument("--soft-risk-enabled", action="store_true")
    parser.add_argument("--symbolic-profile", default="conservative")
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


def coerce_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in (
        "f1",
        "precision",
        "recall",
        "fp",
        "fn",
        "tp",
        "tn",
        "threshold",
        "tau_high",
        "validation_f1",
        "test_f1",
        "validation_precision",
        "validation_recall",
        "validation_fp",
        "test_precision",
        "test_recall",
        "test_fp",
        "test_fn",
        "iteration",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _thresholds_match(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if isinstance(left, str) or isinstance(right, str):
        return str(left) == str(right)
    try:
        return bool(np.isclose(float(left), float(right)))
    except (TypeError, ValueError):
        return str(left) == str(right)


def _tiebreak_sort(df: pd.DataFrame) -> pd.DataFrame:
    extra = [c for c in ("threshold", "tau_high", "lambda_risk", "rho", "iteration") if c in df.columns]
    asc = [False, False, True, False] + [True] * len(extra)
    cols = ["f1", "precision", "fp", "recall"] + extra
    return df.sort_values(
        [c for c in cols if c in df.columns],
        ascending=asc[: len([c for c in cols if c in df.columns])],
        kind="mergesort",
    )


def _base_selected_row(
    *,
    figure: str,
    display_name: str,
    group: str,
    source: str,
    valid_row: pd.Series,
    test_row: pd.Series,
    notes: str = "",
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "figure": figure,
        "display_name": display_name,
        "group": group,
        "source": source,
        "selection_rule": "validation_f1",
        "validation_f1": float(valid_row["f1"]),
        "validation_precision": float(valid_row["precision"]),
        "validation_recall": float(valid_row["recall"]),
        "validation_fp": int(valid_row["fp"]),
        "test_f1": float(test_row["f1"]),
        "test_precision": float(test_row["precision"]),
        "test_recall": float(test_row["recall"]),
        "test_fp": int(test_row["fp"]),
        "test_fn": int(test_row.get("fn", 0)),
        "threshold": valid_row.get("threshold", ""),
        "tau_high": valid_row.get("tau_high", ""),
        "iteration": valid_row.get("iteration", ""),
        "model_name": valid_row.get("model_name", ""),
        "method": valid_row.get("method", ""),
        "decision_mode": valid_row.get("decision_mode", ""),
        "symbolic_profile": valid_row.get("symbolic_profile", ""),
        "config_id": valid_row.get("config_id", ""),
        "notes": notes,
    }
    row.update(extra)
    return row


def select_best_by_validation_f1(
    df: pd.DataFrame,
    method_filter: dict[str, Any] | None,
    id_columns: list[str],
    display_name: str,
    group: str,
    source: str,
    figure: str,
) -> dict[str, Any] | None:
    work = df.copy()
    if method_filter:
        for col, val in method_filter.items():
            if col not in work.columns:
                return None
            work = work[work[col] == val]
    if work.empty:
        return None
    valid = work[work["split"] == "valid"]
    test = work[work["split"] == "test"]
    if valid.empty or test.empty:
        return None
    best_valid = _tiebreak_sort(valid).iloc[0]
    match = test.copy()
    for col in id_columns:
        if col not in best_valid.index or col not in match.columns:
            continue
        val = best_valid[col]
        if col == "threshold":
            match = match[
                match.apply(lambda r: _thresholds_match(r[col], val), axis=1)
            ]
        else:
            match = match[match[col] == val]
    if match.empty and "config_id" in id_columns and "config_id" in best_valid.index:
        match = test[test["config_id"] == best_valid["config_id"]]
    if match.empty:
        warnings.warn(f"No test match for {display_name} ({source})")
        return None
    test_row = match.iloc[0]
    return _base_selected_row(
        figure=figure,
        display_name=display_name,
        group=group,
        source=source,
        valid_row=best_valid,
        test_row=test_row,
    )


def pretty_group_name(group: str) -> str:
    mapping = {
        "baseline": "Baselines",
        "symbolic-only": "Symbolic-only",
        "neural-only": "Neural-only",
        "ablation": "Ablations",
        "neuro-symbolic": "Neuro-symbolic",
        "variant": "Variants",
        "extension": "Extensions",
    }
    return mapping.get(group, group.replace("_", " ").title())


def _group_style(group: str) -> dict[str, str]:
    return GROUP_STYLE.get(group, FALLBACK_STYLE)


def _compute_x_positions(groups: list[str], gap: float = GROUP_GAP) -> np.ndarray:
    positions: list[float] = []
    x = 0.0
    for i, g in enumerate(groups):
        if i > 0 and g != groups[i - 1]:
            x += gap
        positions.append(x)
        x += 1.0
    return np.asarray(positions, dtype=float)


def make_bar_plot(
    df: pd.DataFrame,
    output_path_base: Path,
    title: str,
    ylabel: str = "Test F1",
    value_col: str = "test_f1",
    label_col: str = "display_name",
    group_col: str = "group",
    figsize: tuple[float, float] = (9, 5),
    rotate_xticks: bool = True,
    show_subtitle: bool = True,
    show_legend: bool = False,
) -> None:
    if df.empty:
        warnings.warn(f"Skipping empty plot: {title}")
        return

    labels = df[label_col].tolist()
    values = df[value_col].astype(float).tolist()
    groups = df[group_col].tolist()
    x = _compute_x_positions(groups)
    ymax = max(values) if values else 1.0
    y_top = min(1.0, ymax * 1.18 + 0.03)

    fig, ax = plt.subplots(figsize=figsize)
    bar_width = 0.72

    for xi, val, grp in zip(x, values, groups):
        style = _group_style(grp)
        ax.bar(
            xi,
            val,
            width=bar_width,
            color=style["color"],
            edgecolor=style["edgecolor"],
            hatch=style["hatch"],
            linewidth=BAR_EDGE_WIDTH,
            zorder=3,
        )
        ax.text(
            xi,
            val + 0.012,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            boundary = (x[i - 1] + x[i]) / 2.0
            ax.axvline(boundary, color="#666666", linewidth=0.8, alpha=0.45, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35 if rotate_xticks else 0, ha="right")

    ax.set_ylabel(ylabel)
    ax.set_ylim(0, y_top)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=22 if show_subtitle else 14)

    if show_subtitle:
        ax.text(
            0.5,
            1.01,
            "Configurations selected by validation F1; bars show test F1",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#555555",
        )

    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        seen: list[str] = []
        for g in groups:
            if g not in seen:
                seen.append(g)
        ordered = [g for g in LEGEND_ORDER if g in seen] + [
            g for g in seen if g not in LEGEND_ORDER
        ]
        handles = [
            Patch(
                facecolor=_group_style(g)["color"],
                edgecolor=_group_style(g)["edgecolor"],
                hatch=_group_style(g)["hatch"],
                label=pretty_group_name(g),
            )
            for g in ordered
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=8)

    fig.tight_layout()
    ensure_dir(output_path_base.parent)
    fig.savefig(output_path_base.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(output_path_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _order_rows(rows: list[dict[str, Any]], order: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    order_map = {n: i for i, n in enumerate(order)}
    prefixes = sorted(order, key=len, reverse=True)

    def _rank(name: str) -> int:
        if name in order_map:
            return order_map[name]
        for prefix in prefixes:
            if name.startswith(prefix):
                return order_map[prefix]
        return 999

    df["_ord"] = df["display_name"].map(_rank)
    return df.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


def collect_baseline_rows(
    baseline_df: pd.DataFrame,
    figure: str,
) -> list[dict[str, Any]]:
    specs = [
        ("always_negative", "Always negative", "baseline"),
        ("exact_title_match", "Exact title", "baseline"),
        ("bow_cosine", "BoW cosine", "baseline"),
        ("tfidf_cosine_word", "TF-IDF word", "baseline"),
        ("tfidf_cosine_char", "TF-IDF char", "baseline"),
        ("fieldwise_lexical_heuristic", "Field-wise heuristic", "baseline"),
    ]
    rows: list[dict[str, Any]] = []
    for baseline, display, group in specs:
        sub = baseline_df[baseline_df["baseline"] == baseline]
        if sub.empty:
            warnings.warn(f"Baseline missing: {baseline}")
            continue
        rec = select_best_by_validation_f1(
            sub.rename(columns={"baseline": "method"}),
            None,
            ["method", "threshold", "threshold_objective"],
            display,
            group,
            "baseline_metrics",
            figure,
        )
        if rec:
            rec["method"] = baseline
            rows.append(rec)
    return rows


def collect_sweep_rows(
    sweep_df: pd.DataFrame,
    figure: str,
    method_filter: dict[str, Any],
    display_name: str,
    group: str,
) -> dict[str, Any] | None:
    return select_best_by_validation_f1(
        sweep_df,
        method_filter,
        ["config_id"],
        display_name,
        group,
        "decision_config_sweep",
        figure,
    )


def collect_soft_risk_rows(
    selected_df: pd.DataFrame | None,
    grid_df: pd.DataFrame | None,
    figure: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if selected_df is not None and not selected_df.empty:
        for family, display in [
            ("soft_score_penalty", "soft symbolic risk"),
            ("soft_risk_gate", "symbolic risk gate"),
        ]:
            sub = selected_df[selected_df["method_family"] == family]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append(
                {
                    "figure": figure,
                    "display_name": display,
                    "group": "extension",
                    "source": "soft_risk_selected_configs",
                    "selection_rule": "validation_f1",
                    "validation_f1": float(r["validation_f1"]),
                    "validation_precision": float(r["validation_precision"]),
                    "validation_recall": float(r["validation_recall"]),
                    "validation_fp": int(r["validation_fp"]),
                    "test_f1": float(r["test_f1"]),
                    "test_precision": float(r["test_precision"]),
                    "test_recall": float(r["test_recall"]),
                    "test_fp": int(r["test_fp"]),
                    "test_fn": int(r["test_fn"]),
                    "threshold": r.get("threshold", ""),
                    "tau_high": r.get("rho", r.get("tau_high", "")),
                    "config_id": r.get("config_id", ""),
                    "method": family,
                    "notes": "",
                }
            )
        return rows
    if grid_df is None:
        return rows
    work = grid_df.copy()
    work["split"] = work["split"].astype(str)
    for mode, display in [
        ("soft_score_penalty", "soft symbolic risk"),
        ("soft_risk_gate", "symbolic risk gate"),
    ]:
        sub = work[work["mode"] == mode]
        rec = select_best_by_validation_f1(
            sub,
            None,
            ["config_id"],
            display,
            "extension",
            "soft_risk_grid_results",
            figure,
        )
        if rec:
            rows.append(rec)
    return rows


HN_SELECTION_NOTES = (
    "Selected among hard-negative iterations >= 1 using validation F1."
)


def collect_hard_negative_rows(
    hn_df: pd.DataFrame | None,
    figure: str,
) -> list[dict[str, Any]]:
    if hn_df is None or hn_df.empty:
        return []
    work = hn_df.copy()
    work["iteration"] = pd.to_numeric(work["iteration"], errors="coerce")
    work = work[work["iteration"] >= 1].copy()
    if work.empty:
        warnings.warn(
            "No hard-negative training rows with iteration >= 1 found; skipping HN variant."
        )
        return []

    rows: list[dict[str, Any]] = []
    for dtype, display_base in [
        ("neural_only", "symbolic hard negatives"),
        ("hard_governed_invalid_blocks", "symbolic hard negatives + neuro-symbolic"),
    ]:
        sub = work[work["decision_type"] == dtype]
        if sub.empty:
            warnings.warn(
                f"No hard-negative rows with iteration >= 1 for decision_type={dtype}; "
                "skipping HN variant."
            )
            continue
        rec = select_best_by_validation_f1(
            sub,
            None,
            ["iteration"],
            display_base,
            "extension",
            "hard_negative_training",
            figure,
        )
        if rec:
            it = rec.get("iteration", "")
            if figure == "figure2" and it != "" and not pd.isna(it):
                display = f"{display_base} (iter {int(it)})"
            else:
                display = display_base
            rec["display_name"] = display
            rec["decision_type"] = dtype
            rec["notes"] = HN_SELECTION_NOTES
            rows.append(rec)
    return rows


def collect_structured_model_rows(
    comparison_df: pd.DataFrame | None,
    governed_df: pd.DataFrame | None,
    figure: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if comparison_df is not None:
        for model, display, group in [
            ("field_attention", "field attention", "extension"),
            ("structured_transformer", "structured transformer", "extension"),
        ]:
            sub = comparison_df[comparison_df["model_name"] == model]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append(
                {
                    "figure": figure,
                    "display_name": display,
                    "group": group,
                    "source": "structured_model_test_comparison",
                    "selection_rule": "validation_f1",
                    "validation_f1": float(r["validation_f1"]),
                    "validation_precision": float(r["validation_precision"]),
                    "validation_recall": float(r["validation_recall"]),
                    "validation_fp": int(r["validation_fp"]),
                    "test_f1": float(r["test_f1"]),
                    "test_precision": float(r["test_precision"]),
                    "test_recall": float(r["test_recall"]),
                    "test_fp": int(r["test_fp"]),
                    "test_fn": int(r["test_fn"]),
                    "threshold": r.get("validation_threshold", ""),
                    "model_name": model,
                    "notes": "",
                }
            )
    if governed_df is not None:
        mapping = [
            ("field_attention_governed", "field attention + neuro-symbolic"),
            ("structured_transformer_governed", "structured transformer + neuro-symbolic"),
        ]
        for model_key, display in mapping:
            sub = governed_df[governed_df["model_name"] == model_key]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append(
                {
                    "figure": figure,
                    "display_name": display,
                    "group": "extension",
                    "source": "structured_model_governed_comparison",
                    "selection_rule": "validation_f1",
                    "validation_f1": np.nan,
                    "validation_precision": np.nan,
                    "validation_recall": np.nan,
                    "validation_fp": np.nan,
                    "test_f1": float(r["f1"]),
                    "test_precision": float(r["precision"]),
                    "test_recall": float(r["recall"]),
                    "test_fp": int(r["fp"]),
                    "test_fn": int(r["fn"]),
                    "threshold": r.get("threshold", ""),
                    "model_name": model_key,
                    "notes": "governed at model validation threshold",
                }
            )
    return rows


def compute_symbolic_only_metrics(
    raw_predictions_path: Path,
    symbolic_profile: str,
    split: str,
) -> dict[str, Any]:
    raw = pd.read_csv(raw_predictions_path)
    checker = ProductConstraintChecker(make_symbolic_config(symbolic_profile))
    sym = checker.check_dataframe(raw)
    merged = raw.copy()
    for col in sym.columns:
        if col != "pair_id":
            merged[col] = sym[col].values
    preds = (merged["symbolic_status"] == "valid").astype(int).values
    y_true = merged["label"].astype(int).values
    scores = preds.astype(float)
    m = compute_binary_metrics_from_predictions(y_true, preds, scores, 0.5)
    return {
        "split": split,
        "method": "symbolic_only",
        "display_name": "symbolic-only",
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "fp": m["fp"],
        "fn": m["fn"],
        "tp": m["tp"],
        "tn": m["tn"],
        "threshold": "n/a",
    }


def collect_symbolic_only_row(
    pred_dir: Path,
    table_dir: Path,
    symbolic_profile: str,
    figure: str,
) -> dict[str, Any] | None:
    metrics_rows = []
    for split in ("valid", "test"):
        path = pred_dir / f"raw_{split}_predictions.csv"
        if not path.exists():
            warnings.warn(f"Missing {path} for symbolic-only baseline")
            return None
        metrics_rows.append(
            compute_symbolic_only_metrics(path, symbolic_profile, split)
        )
    sym_df = pd.DataFrame(metrics_rows)
    sym_df.to_csv(table_dir / "symbolic_only_metrics.csv", index=False)
    valid = sym_df[sym_df["split"] == "valid"].iloc[0]
    test = sym_df[sym_df["split"] == "test"].iloc[0]
    return {
        "figure": figure,
        "display_name": "symbolic-only",
        "group": "symbolic-only",
        "source": "symbolic_only_metrics",
        "selection_rule": "validation_f1",
        "validation_f1": float(valid["f1"]),
        "validation_precision": float(valid["precision"]),
        "validation_recall": float(valid["recall"]),
        "validation_fp": int(valid["fp"]),
        "test_f1": float(test["f1"]),
        "test_precision": float(test["precision"]),
        "test_recall": float(test["recall"]),
        "test_fp": int(test["fp"]),
        "test_fn": int(test["fn"]),
        "threshold": "n/a",
        "method": "symbolic_only",
        "notes": "accept iff symbolic_status==valid",
    }


def pick_best_lexical(
    baseline_df: pd.DataFrame,
    figure: str,
) -> dict[str, Any] | None:
    names = [
        "bow_cosine",
        "tfidf_cosine_word",
        "tfidf_cosine_char",
        "fieldwise_lexical_heuristic",
    ]
    candidates: list[dict[str, Any]] = []
    for b in names:
        sub = baseline_df[baseline_df["baseline"] == b]
        if sub.empty:
            continue
        rec = select_best_by_validation_f1(
            sub.rename(columns={"baseline": "method"}),
            None,
            ["method", "threshold", "threshold_objective"],
            "best lexical baseline",
            "baseline",
            "baseline_metrics",
            figure,
        )
        if rec:
            rec["display_name"] = "best lexical baseline"
            rec["notes"] = f"best among lexical baselines ({b})"
            candidates.append(rec)
    if not candidates:
        return None
    return max(candidates, key=lambda r: r["validation_f1"])


def build_figure1(
    baseline_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = collect_baseline_rows(baseline_df, "figure1")
    neural = collect_sweep_rows(
        sweep_df,
        "figure1",
        {"method": "neural_only"},
        "neural-only",
        "neural-only",
    )
    if neural:
        rows.append(neural)
    proposed = collect_sweep_rows(
        sweep_df,
        "figure1",
        {
            "method": "neuro_symbolic_invalid_blocks",
            "decision_mode": "invalid_blocks",
            "symbolic_profile": "conservative",
        },
        "neuro-symbolic (conservative)",
        "neuro-symbolic",
    )
    if proposed:
        rows.append(proposed)
    order = [
        "Always negative",
        "Exact title",
        "BoW cosine",
        "TF-IDF word",
        "TF-IDF char",
        "Field-wise heuristic",
        "neural-only",
        "neuro-symbolic (conservative)",
    ]
    return _order_rows(rows, order).to_dict("records")


def build_figure2(
    sweep_df: pd.DataFrame,
    soft_selected: pd.DataFrame | None,
    soft_grid: pd.DataFrame | None,
    hn_df: pd.DataFrame | None,
    struct_comp: pd.DataFrame | None,
    struct_gov: pd.DataFrame | None,
    include_soft: bool,
    include_struct: bool,
    include_hn: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for filt, display, group in [
        (
            {
                "method": "neuro_symbolic_invalid_blocks",
                "decision_mode": "invalid_blocks",
                "symbolic_profile": "conservative",
            },
            "neuro-symbolic (conservative)",
            "neuro-symbolic",
        ),
        (
            {
                "method": "neuro_symbolic_invalid_blocks",
                "decision_mode": "invalid_blocks",
                "symbolic_profile": "moderate",
            },
            "neuro-symbolic (moderate)",
            "ablation",
        ),
        (
            {
                "method": "neuro_symbolic_strict_two_threshold",
                "decision_mode": "strict_valid_or_high_confidence",
                "symbolic_profile": "conservative",
            },
            "two threshold governance",
            "variant",
        ),
    ]:
        rec = collect_sweep_rows(sweep_df, "figure2", filt, display, group)
        if rec:
            rows.append(rec)
    if include_soft:
        rows.extend(collect_soft_risk_rows(soft_selected, soft_grid, "figure2"))
    if include_hn:
        rows.extend(collect_hard_negative_rows(hn_df, "figure2"))
    if include_struct:
        rows.extend(collect_structured_model_rows(struct_comp, struct_gov, "figure2"))
    order = [
        "neuro-symbolic (conservative)",
        "neuro-symbolic (moderate)",
        "two threshold governance",
        "soft symbolic risk",
        "symbolic risk gate",
        "symbolic hard negatives",
        "symbolic hard negatives + neuro-symbolic",
        "field attention",
        "structured transformer",
        "field attention + neuro-symbolic",
        "structured transformer + neuro-symbolic",
    ]
    return _order_rows(rows, order).to_dict("records")


def build_figure3(
    baseline_df: pd.DataFrame,
    sweep_df: pd.DataFrame,
    pred_dir: Path,
    table_dir: Path,
    symbolic_profile: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    lex = pick_best_lexical(baseline_df, "figure3")
    if lex:
        rows.append(lex)
    sym = collect_symbolic_only_row(pred_dir, table_dir, symbolic_profile, "figure3")
    if sym:
        rows.append(sym)
    neural = collect_sweep_rows(
        sweep_df,
        "figure3",
        {"method": "neural_only"},
        "neural-only",
        "neural-only",
    )
    if neural:
        rows.append(neural)
    base = collect_sweep_rows(
        sweep_df,
        "figure3",
        {
            "method": "neuro_symbolic_invalid_blocks",
            "decision_mode": "invalid_blocks",
            "symbolic_profile": "conservative",
        },
        "neuro-symbolic (conservative)",
        "neuro-symbolic",
    )
    if base:
        rows.append(base)
    order = [
        "best lexical baseline",
        "symbolic-only",
        "neural-only",
        "neuro-symbolic (conservative)",
    ]
    return _order_rows(rows, order).to_dict("records")


def print_summary(fig_name: str, path_base: Path, df: pd.DataFrame) -> None:
    print(f"\n{fig_name}")
    print("Saved:")
    print(f"  {path_base.with_suffix('.png')}")
    print(f"  {path_base.with_suffix('.pdf')}")
    if df.empty:
        return
    print("Selected rows:")
    for _, r in df.iterrows():
        print(
            f"  {r['display_name']:<28} "
            f"{r['validation_f1']:.4f}   {r['test_f1']:.4f}   "
            f"{r['test_precision']:.4f}   {r['test_recall']:.4f}   "
            f"{int(r['test_fp']):>4}   {r['source']}"
        )


def main() -> int:
    args = parse_args()
    variant_id = args.variant
    run_name = args.run_name
    fig_dir = ensure_dir(OUTPUTS_DIR / "figures" / "final_comparison" / variant_id / run_name)
    table_dir = ensure_dir(OUTPUTS_DIR / "tables" / "final_comparison" / variant_id / run_name)
    pred_dir = OUTPUTS_DIR / "predictions" / variant_id / run_name

    baseline_path = OUTPUTS_DIR / "tables" / "baselines" / variant_id / "baseline_metrics.csv"
    sweep_path = (
        OUTPUTS_DIR / "tables" / "sweeps" / variant_id / run_name / "decision_config_sweep.csv"
    )

    baseline_df = safe_read_csv(baseline_path, required=True)
    sweep_df = safe_read_csv(sweep_path, required=True)
    baseline_df = coerce_metric_columns(baseline_df)
    sweep_df = coerce_metric_columns(sweep_df)

    soft_selected = safe_read_csv(
        OUTPUTS_DIR / "tables" / "soft_risk" / variant_id / run_name / "soft_risk_selected_configs.csv"
    )
    soft_grid = safe_read_csv(
        OUTPUTS_DIR / "tables" / "soft_risk" / variant_id / run_name / "soft_risk_grid_results.csv"
    )
    hn_path = (
        OUTPUTS_DIR
        / "tables"
        / "hard_negative_training"
        / variant_id
        / args.hard_negative_experiment
        / "iteration_metrics.csv"
    )
    hn_df = safe_read_csv(hn_path)
    struct_comp = safe_read_csv(
        OUTPUTS_DIR / "tables" / "structured_models" / variant_id / "structured_model_test_comparison.csv"
    )
    struct_gov = safe_read_csv(
        OUTPUTS_DIR
        / "tables"
        / "structured_models"
        / variant_id
        / "structured_model_governed_comparison.csv"
    )

    include_soft = args.soft_risk_enabled or soft_selected is not None or soft_grid is not None
    include_struct = args.structured_models_enabled or struct_comp is not None
    include_hn = hn_df is not None

    if soft_selected is not None:
        soft_selected = coerce_metric_columns(soft_selected)
    if soft_grid is not None:
        soft_grid = coerce_metric_columns(soft_grid)
    if hn_df is not None:
        hn_df = coerce_metric_columns(hn_df)
    if struct_comp is not None:
        struct_comp = coerce_metric_columns(struct_comp)

    fig1_rows = build_figure1(baseline_df, sweep_df)
    fig2_rows = build_figure2(
        sweep_df,
        soft_selected,
        soft_grid,
        hn_df,
        struct_comp,
        struct_gov,
        include_soft,
        include_struct,
        include_hn,
    )
    fig3_rows = build_figure3(
        baseline_df, sweep_df, pred_dir, table_dir, args.symbolic_profile
    )

    fig1_df = pd.DataFrame(fig1_rows)
    fig2_df = pd.DataFrame(fig2_rows)
    fig3_df = pd.DataFrame(fig3_rows)

    fig1_df.to_csv(table_dir / "figure1_selected_rows.csv", index=False)
    fig2_df.to_csv(table_dir / "figure2_selected_rows.csv", index=False)
    fig3_df.to_csv(table_dir / "figure3_selected_rows.csv", index=False)
    all_df = pd.concat([fig1_df, fig2_df, fig3_df], ignore_index=True)
    all_df.to_csv(table_dir / "all_selected_rows_long.csv", index=False)

    save_json(
        {
            "variant_id": variant_id,
            "run_name": run_name,
            "selection_rule": "validation_f1",
            "plotted_metric": "test_f1",
            "optional_included": {
                "soft_risk": include_soft,
                "structured_models": include_struct,
                "hard_negative": include_hn,
            },
        },
        table_dir / "final_comparison_metadata.json",
    )

    make_bar_plot(
        fig1_df,
        fig_dir / "figure1_baselines_vs_proposed_test_f1",
        "Baselines vs neuro-symbolic method",
        figsize=(10, 5.5),
        show_subtitle=True,
    )
    make_bar_plot(
        fig2_df,
        fig_dir / "figure2_proposed_variants_test_f1",
        "Variants of the neuro-symbolic method",
        figsize=(12, 5.5),
        show_subtitle=True,
    )
    make_bar_plot(
        fig3_df,
        fig_dir / "figure3_intro_motivation_test_f1",
        "Why combine neural scoring with symbolic governance?",
        figsize=(8, 5),
        rotate_xticks=True,
        show_subtitle=True,
    )

    print("FINAL COMPARISON FIGURES COMPLETE")
    print_summary(
        "Figure 1: Baselines vs neuro-symbolic method",
        fig_dir / "figure1_baselines_vs_proposed_test_f1",
        fig1_df,
    )
    print_summary(
        "Figure 2: Variants of the neuro-symbolic method",
        fig_dir / "figure2_proposed_variants_test_f1",
        fig2_df,
    )
    print_summary(
        "Figure 3: Introduction motivation figure",
        fig_dir / "figure3_intro_motivation_test_f1",
        fig3_df,
    )
    print(f"\nTables: {table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
