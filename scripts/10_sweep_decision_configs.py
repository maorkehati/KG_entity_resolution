"""Sweep neural-only and neuro-symbolic decision configurations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_WDC_VARIANT_ID, OUTPUTS_DIR, PROJECT_ROOT
from src.decision import DecisionConfig, apply_governed_decision, compute_governance_diagnostics
from src.evaluation import compute_binary_metrics_from_predictions, save_json
from src.symbolic import ProductConstraintChecker, make_symbolic_config

RESULT_COLUMNS = [
    "variant_id",
    "run_name",
    "split",
    "config_id",
    "method",
    "neurosymbolic_enabled",
    "decision_mode",
    "symbolic_profile",
    "threshold",
    "tau_high",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "false_positive_rate",
    "false_negative_rate",
    "tp",
    "fp",
    "tn",
    "fn",
    "support",
    "positives_true",
    "positives_pred",
    "positive_rate_true",
    "positive_rate_pred",
    "auroc",
    "auprc",
    "average_precision",
    "brier_score",
    "log_loss",
    "accepted_count",
    "rejected_count",
    "accept_rate",
    "reject_rate",
    "symbolic_valid_count",
    "symbolic_invalid_count",
    "symbolic_uncertain_count",
    "invalid_block_count",
    "invalid_block_positive_labels",
    "invalid_block_negative_labels",
    "delta_precision",
    "delta_recall",
    "delta_f1",
    "delta_fp",
    "delta_fn",
    "fp_reduction",
    "fp_reduction_rate",
    "recall_loss",
    "recall_loss_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep decision configurations.")
    parser.add_argument("--variant", default=DEFAULT_WDC_VARIANT_ID)
    parser.add_argument("--run-name", default="neural_logreg")
    parser.add_argument(
        "--neural-thresholds",
        nargs="+",
        type=float,
        default=[0.50, 0.60, 0.66, 0.70, 0.80, 0.90, 0.95, 0.97],
    )
    parser.add_argument(
        "--tau-high-values",
        nargs="+",
        type=float,
        default=[0.80, 0.90, 0.95, 0.97],
    )
    parser.add_argument(
        "--symbolic-profiles",
        nargs="+",
        default=["conservative", "moderate"],
    )
    parser.add_argument("--include-train", action="store_true")
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def iter_sweep_configs(args: argparse.Namespace) -> list[dict]:
    """Generate valid decision configuration dicts."""
    configs: list[dict] = []

    for threshold in args.neural_thresholds:
        configs.append(
            {
                "config_id": f"neural_only|thr={threshold:.2f}",
                "method": "neural_only",
                "neurosymbolic_enabled": False,
                "decision_mode": "neural_threshold",
                "symbolic_profile": "none",
                "threshold": threshold,
                "tau_high": np.nan,
            }
        )

    for threshold in args.neural_thresholds:
        for profile in args.symbolic_profiles:
            configs.append(
                {
                    "config_id": f"invalid_blocks|profile={profile}|thr={threshold:.2f}",
                    "method": "neuro_symbolic_invalid_blocks",
                    "neurosymbolic_enabled": True,
                    "decision_mode": "invalid_blocks",
                    "symbolic_profile": profile,
                    "threshold": threshold,
                    "tau_high": np.nan,
                }
            )

    for threshold in args.neural_thresholds:
        for tau_high in args.tau_high_values:
            if tau_high < threshold:
                continue
            for profile in args.symbolic_profiles:
                configs.append(
                    {
                        "config_id": (
                            f"strict_two_threshold|profile={profile}|"
                            f"thr={threshold:.2f}|tau_high={tau_high:.2f}"
                        ),
                        "method": "neuro_symbolic_strict_two_threshold",
                        "neurosymbolic_enabled": True,
                        "decision_mode": "strict_valid_or_high_confidence",
                        "symbolic_profile": profile,
                        "threshold": threshold,
                        "tau_high": tau_high,
                    }
                )

    if args.max_configs is not None:
        configs = configs[: args.max_configs]
    return configs


def _load_raw_split(pred_dir: Path, split: str) -> pd.DataFrame:
    path = pred_dir / f"raw_{split}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def _build_symbolic_cache(
    splits: list[str],
    profiles: list[str],
    pred_dir: Path,
) -> dict[tuple[str, str], pd.DataFrame]:
    cache: dict[tuple[str, str], pd.DataFrame] = {}
    for split in splits:
        raw = _load_raw_split(pred_dir, split)
        for profile in profiles:
            checker = ProductConstraintChecker(make_symbolic_config(profile))
            sym = checker.check_dataframe(raw)
            cache[(split, profile)] = sym
    return cache


def _predict_config(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return dataframe with governed_pred column."""
    if cfg["method"] == "neural_only":
        out = df.copy()
        out["governed_pred"] = (out["neural_score"] >= cfg["threshold"]).astype(int)
        out["final_decision"] = np.where(
            out["governed_pred"] == 1, "accept", "reject"
        )
        return out

    decision_cfg = DecisionConfig(
        decision_mode=cfg["decision_mode"],
        threshold=cfg["threshold"],
        tau_high=float(cfg["tau_high"]) if not pd.isna(cfg["tau_high"]) else 0.9,
        uncertain_action="accept",
    )
    return apply_governed_decision(df, decision_cfg)


def _evaluate_row(
    df: pd.DataFrame,
    cfg: dict,
    variant_id: str,
    run_name: str,
    split: str,
) -> dict:
    y_true = df["label"].astype(int).values
    y_pred = df["governed_pred"].astype(int).values
    y_score = df["neural_score"].values

    metrics = compute_binary_metrics_from_predictions(
        y_true, y_pred, y_score=y_score, threshold=cfg["threshold"]
    )

    row = {
        "variant_id": variant_id,
        "run_name": run_name,
        "split": split,
        "config_id": cfg["config_id"],
        "method": cfg["method"],
        "neurosymbolic_enabled": cfg["neurosymbolic_enabled"],
        "decision_mode": cfg["decision_mode"],
        "symbolic_profile": cfg["symbolic_profile"],
        "threshold": cfg["threshold"],
        "tau_high": cfg["tau_high"],
        **metrics,
    }

    if cfg["neurosymbolic_enabled"]:
        diag = compute_governance_diagnostics(df)
        row["accepted_count"] = diag["accepted"]
        row["rejected_count"] = diag["rejected"]
        row["accept_rate"] = diag["accept_rate"]
        row["reject_rate"] = diag["reject_rate"]
        row["symbolic_valid_count"] = diag["symbolic_valid_count"]
        row["symbolic_invalid_count"] = diag["symbolic_invalid_count"]
        row["symbolic_uncertain_count"] = diag["symbolic_uncertain_count"]
        row["invalid_block_count"] = diag["invalid_block_count"]
        row["invalid_block_positive_labels"] = diag["invalid_block_positive_labels"]
        row["invalid_block_negative_labels"] = diag["invalid_block_negative_labels"]
    else:
        accepted = int((df["governed_pred"] == 1).sum())
        rejected = len(df) - accepted
        row["accepted_count"] = accepted
        row["rejected_count"] = rejected
        row["accept_rate"] = accepted / len(df) if len(df) else 0.0
        row["reject_rate"] = rejected / len(df) if len(df) else 0.0
        row["symbolic_valid_count"] = np.nan
        row["symbolic_invalid_count"] = np.nan
        row["symbolic_uncertain_count"] = np.nan
        row["invalid_block_count"] = np.nan
        row["invalid_block_positive_labels"] = np.nan
        row["invalid_block_negative_labels"] = np.nan

    return row


def _add_deltas(results: pd.DataFrame) -> pd.DataFrame:
    """Add neural-vs-governed delta columns."""
    out = results.copy()
    for col in (
        "delta_precision",
        "delta_recall",
        "delta_f1",
        "delta_fp",
        "delta_fn",
        "fp_reduction",
        "fp_reduction_rate",
        "recall_loss",
        "recall_loss_rate",
    ):
        out[col] = np.nan

    neural = out[out["method"] == "neural_only"].set_index(["split", "threshold"])
    for idx, row in out.iterrows():
        if row["method"] == "neural_only":
            continue
        key = (row["split"], row["threshold"])
        if key not in neural.index:
            continue
        n = neural.loc[key]
        out.at[idx, "delta_precision"] = row["precision"] - n["precision"]
        out.at[idx, "delta_recall"] = row["recall"] - n["recall"]
        out.at[idx, "delta_f1"] = row["f1"] - n["f1"]
        out.at[idx, "delta_fp"] = row["fp"] - n["fp"]
        out.at[idx, "delta_fn"] = row["fn"] - n["fn"]
        fp_red = n["fp"] - row["fp"]
        out.at[idx, "fp_reduction"] = fp_red
        out.at[idx, "fp_reduction_rate"] = fp_red / max(n["fp"], 1)
        rec_loss = n["recall"] - row["recall"]
        out.at[idx, "recall_loss"] = rec_loss
        out.at[idx, "recall_loss_rate"] = rec_loss / max(n["recall"], 1e-12)
    return out


def _print_best_summary(df: pd.DataFrame, split: str) -> None:
    sub = df[df["split"] == split]
    print(f"\n{split.capitalize()}:")

    if sub.empty:
        print("  (no rows)")
        return

    best_f1 = sub.loc[sub["f1"].idxmax()]
    print(
        f"  best F1: {best_f1['config_id']} "
        f"(F1={best_f1['f1']:.4f}, P={best_f1['precision']:.4f}, R={best_f1['recall']:.4f})"
    )

    recall_ok = sub[sub["recall"] >= 0.50]
    if recall_ok.empty:
        print("  best precision with recall >= 0.50: none")
    else:
        best_p = recall_ok.loc[recall_ok["precision"].idxmax()]
        print(
            f"  best precision with recall >= 0.50: {best_p['config_id']} "
            f"(P={best_p['precision']:.4f}, R={best_p['recall']:.4f})"
        )

    governed = sub[sub["method"] != "neural_only"].copy()
    governed = governed[governed["recall_loss_rate"].notna()]
    low_recall_loss = governed[governed["recall_loss_rate"] <= 0.25]
    if low_recall_loss.empty:
        print("  best FP reduction with recall_loss_rate <= 0.25: none")
    else:
        best_fp = low_recall_loss.loc[low_recall_loss["fp_reduction"].idxmax()]
        print(
            f"  best FP reduction with recall_loss_rate <= 0.25: {best_fp['config_id']} "
            f"(FP reduction={int(best_fp['fp_reduction'])}, "
            f"recall_loss_rate={best_fp['recall_loss_rate']:.4f})"
        )


def main() -> int:
    args = parse_args()
    pred_dir = OUTPUTS_DIR / "predictions" / args.variant / args.run_name
    out_dir = OUTPUTS_DIR / "tables" / "sweeps" / args.variant / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "decision_config_sweep.csv"

    if out_csv.exists() and not args.overwrite:
        print(f"Sweep output exists: {out_csv}. Use --overwrite to replace.", file=sys.stderr)
        return 1

    splits = ["valid", "test"]
    if args.include_train:
        splits = ["train", "valid", "test"]

    for split in splits:
        try:
            _load_raw_split(pred_dir, split)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

    configs = iter_sweep_configs(args)
    symbolic_cache = _build_symbolic_cache(splits, args.symbolic_profiles, pred_dir)

    raw_by_split = {s: _load_raw_split(pred_dir, s) for s in splits}
    merged_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for split in splits:
        for profile in args.symbolic_profiles:
            merged_cache[(split, profile)] = raw_by_split[split].merge(
                symbolic_cache[(split, profile)],
                on="pair_id",
                how="left",
            )

    rows: list[dict] = []
    for cfg in configs:
        for split in splits:
            if cfg["method"] == "neural_only":
                work = raw_by_split[split].copy()
            else:
                work = merged_cache[(split, cfg["symbolic_profile"])].copy()
            work = _predict_config(work, cfg)
            rows.append(_evaluate_row(work, cfg, args.variant, args.run_name, split))

    results = _add_deltas(pd.DataFrame(rows))
    results = results[RESULT_COLUMNS]
    results.to_csv(out_csv, index=False)

    metadata = {
        "variant_id": args.variant,
        "run_name": args.run_name,
        "eval_splits": splits,
        "neural_thresholds": args.neural_thresholds,
        "tau_high_values": args.tau_high_values,
        "symbolic_profiles": args.symbolic_profiles,
        "num_configurations": len(configs),
        "num_rows": len(results),
        "output_csv": str(out_csv.relative_to(PROJECT_ROOT)),
        "notes": (
            "Decision sweep over neural-only, invalid-blocking neuro-symbolic "
            "governance, and strict two-threshold neuro-symbolic governance. "
            "Strict mode accepts uncertain cases when neural_score >= tau_high."
        ),
    }
    meta_path = out_dir / "decision_config_sweep_metadata.json"
    save_json(metadata, meta_path)

    print("DECISION CONFIG SWEEP COMPLETE\n")
    print(f"Saved:\n  {out_csv.relative_to(PROJECT_ROOT)}")
    print(f"Configurations: {len(configs)}, rows: {len(results)}")
    print("\nBest by split:")
    for split in splits:
        _print_best_summary(results, split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
