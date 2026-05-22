"""Qualitative error analysis from neural and governed predictions."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import OUTPUTS_DIR
from src.decision import DecisionConfig, apply_governed_decision
from src.evaluation import apply_threshold, save_json, threshold_to_str
from src.symbolic import (
    ACCESSORY_TERMS,
    MAIN_PRODUCT_TERMS,
    ProductConstraintChecker,
    extract_model_like_tokens,
    extract_quantities,
    lists_to_json_columns,
    make_symbolic_config,
    normalize_text,
    parse_symbolic_list,
)

CONSTRAINT_LABELS = {
    "model_token_conflict": "different model identifiers",
    "quantity_conflict": "different capacities / quantities",
    "variant_modifier_conflict": "variant mismatch",
    "accessory_main_product_conflict": "accessory vs main product",
    "brand_conflict": "brand mismatch",
    "bundle_or_kit_conflict": "bundle vs single-item mismatch",
    "same_currency_price_conflict": "price mismatch",
    "color_conflict": "color mismatch",
    "category_keyword_conflict": "category mismatch",
}

EXAMPLE_OUTPUT_COLS = [
    "split",
    "analysis_category",
    "false_negative_type",
    "auto_category",
    "short_explanation",
    "pair_id",
    "label",
    "neural_score",
    "neural_pred",
    "governed_pred",
    "symbolic_status",
    "violated_constraints",
    "positive_evidence",
    "uncertain_reasons",
    "left_title",
    "right_title",
    "left_brand",
    "right_brand",
    "left_price",
    "right_price",
    "left_description",
    "right_description",
    "left_description_short",
    "right_description_short",
    "title_similarity",
    "left_model_tokens",
    "right_model_tokens",
    "left_quantities",
    "right_quantities",
]

SUMMARY_CATEGORIES = [
    "blocked_false_positives",
    "blocked_true_positives",
    "remaining_false_positives",
    "false_negatives_total",
    "false_negatives_score",
    "false_negatives_symbolic",
    "false_negatives_other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qualitative error analysis for governed ER.")
    parser.add_argument("--variant", default="pairwise_50_medium_unseen100")
    parser.add_argument("--run-name", default="neural_logreg")
    parser.add_argument("--threshold", type=float, default=0.66)
    parser.add_argument("--split", default="test")
    parser.add_argument("--decision-mode", default="invalid_blocks")
    parser.add_argument("--symbolic-profile", default="conservative")
    parser.add_argument("--max-examples", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _truncate(text: Any, max_len: int = 300) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _json_list_str(items: list[str]) -> str:
    return json.dumps(items, ensure_ascii=False)


def _title_similarity(left: str, right: str) -> float:
    lt = set(normalize_text(left).split())
    rt = set(normalize_text(right).split())
    if not lt or not rt:
        return 0.0
    return len(lt & rt) / len(lt | rt)


def _brand_norm(brand: Any) -> str:
    return normalize_text(brand) if brand is not None and not pd.isna(brand) else ""


def _has_accessory_term(text: str) -> bool:
    norm = normalize_text(text)
    return any(t in norm for t in ACCESSORY_TERMS)


def _has_main_product_term(text: str) -> bool:
    norm = normalize_text(text)
    return any(t in norm for t in MAIN_PRODUCT_TERMS)


def _violations(row: pd.Series) -> list[str]:
    return parse_symbolic_list(row.get("violated_constraints", []))


def _auto_from_violations(violations: list[str]) -> str | None:
    labels = [CONSTRAINT_LABELS[v] for v in violations if v in CONSTRAINT_LABELS]
    if len(labels) > 1:
        return "multiple symbolic conflicts"
    if len(labels) == 1:
        return labels[0]
    return None


def categorize_example(row: pd.Series, category: str) -> str:
    violations = _violations(row)
    mapped = _auto_from_violations(violations)
    if mapped:
        return mapped

    left_title = str(row.get("left_title", "") or "")
    right_title = str(row.get("right_title", "") or "")
    left_brand = _brand_norm(row.get("left_brand"))
    right_brand = _brand_norm(row.get("right_brand"))
    sim = float(row.get("title_similarity", _title_similarity(left_title, right_title)))

    if category == "blocked_false_positives":
        if violations:
            return "multiple symbolic conflicts"
        return "symbolic invalidation (unspecified constraint)"

    if category == "blocked_true_positives":
        if violations and "model_token_conflict" in violations:
            return "noisy model-token extraction"
        if violations and "quantity_conflict" in violations:
            return "quantity mentioned as compatibility rather than identity"
        if violations and "variant_modifier_conflict" in violations:
            return "multi-variant listing"
        if violations and "brand_conflict" in violations:
            return "brand alias / normalization failure"
        if violations and "accessory_main_product_conflict" in violations:
            return "bundle/accessory ambiguity"
        return "possibly noisy label or strict symbolic rule"

    if category == "remaining_false_positives":
        if left_brand and right_brand and left_brand != right_brand:
            return "possible uncovered brand conflict"
        combined = f"{left_title} {right_title}"
        if _has_accessory_term(combined) and (
            _has_accessory_term(left_title) ^ _has_accessory_term(right_title)
            or _has_main_product_term(left_title) ^ _has_main_product_term(right_title)
        ):
            return "possible accessory/main-product ambiguity"
        if sim >= 0.45 and not violations:
            return "high lexical overlap without symbolic contradiction"
        if not left_brand or not right_brand or not left_title or not right_title:
            return "missing symbolic evidence"
        return "uncovered semantic distinction"

    if category in ("false_negatives", "score_false_negative"):
        if sim < 0.2:
            return "low lexical overlap / paraphrase"
        if not left_brand or not right_brand:
            return "missing brand evidence"
        left_desc = str(row.get("left_description", "") or "")
        right_desc = str(row.get("right_description", "") or "")
        if sim < 0.35 and left_desc and right_desc:
            return "description-only evidence"
        return "score below threshold"

    if category == "symbolic_false_negative":
        if violations and "model_token_conflict" in violations:
            return "noisy model-token extraction"
        if violations and "quantity_conflict" in violations:
            return "quantity mentioned as compatibility rather than identity"
        return "symbolic invalidation blocked true match"

    return "other"


def make_short_explanation(row: pd.Series, category: str) -> str:
    auto = str(row.get("auto_category", categorize_example(row, category)))
    violations = _violations(row)
    vtxt = ", ".join(violations) if violations else "none"

    templates = {
        "blocked_false_positives": (
            f"Neural-only would merge this non-match (score {row['neural_score']:.3f}); "
            f"governance blocked due to: {auto}. Violated: {vtxt}."
        ),
        "blocked_true_positives": (
            f"True match rejected by governance (score {row['neural_score']:.3f}); "
            f"likely: {auto}. Violated: {vtxt}."
        ),
        "remaining_false_positives": (
            f"Governed false positive (score {row['neural_score']:.3f}); "
            f"{auto}. Violated: {vtxt}."
        ),
        "score_false_negative": (
            f"True match missed: score {row['neural_score']:.3f} below threshold; {auto}."
        ),
        "symbolic_false_negative": (
            f"True match blocked despite score {row['neural_score']:.3f}; {auto}. "
            f"Violated: {vtxt}."
        ),
        "both_or_other": (
            f"False negative with score {row['neural_score']:.3f}, status "
            f"{row.get('symbolic_status', '')}; {auto}."
        ),
    }
    key = category if category in templates else "both_or_other"
    return templates.get(key, templates["both_or_other"])


def discover_governed_file(
    pred_dir: Path,
    split: str,
    decision_mode: str,
    threshold: float,
) -> Path | None:
    thr_str = threshold_to_str(threshold)
    exact = pred_dir / f"governed_{split}_{decision_mode}_thr_{thr_str}.csv"
    if exact.exists():
        return exact

    patterns = [
        f"governed_{split}_{decision_mode}_thr_*.csv",
        f"governed_{split}_*{decision_mode}*.csv",
        f"governed_{split}_*.csv",
        f"thresholded_{split}_*.csv",
    ]
    for pattern in patterns:
        matches = sorted(pred_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_raw_predictions(pred_dir: Path, split: str) -> pd.DataFrame:
    path = pred_dir / f"raw_{split}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw predictions: {path}. Run scripts/06_train_neural_scorer.py first."
        )
    return pd.read_csv(path)


def reconstruct_governed(
    raw_df: pd.DataFrame,
    threshold: float,
    decision_mode: str,
    symbolic_profile: str,
) -> pd.DataFrame:
    checker = ProductConstraintChecker(make_symbolic_config(symbolic_profile))
    sym = checker.check_dataframe(raw_df)
    merged = raw_df.merge(sym, on="pair_id", how="left")
    cfg = DecisionConfig(decision_mode=decision_mode, threshold=threshold)
    return apply_governed_decision(merged, cfg)


def load_analysis_frame(
    pred_dir: Path,
    out_dir: Path,
    split: str,
    threshold: float,
    decision_mode: str,
    symbolic_profile: str,
) -> tuple[pd.DataFrame, str]:
    governed_path = discover_governed_file(pred_dir, split, decision_mode, threshold)
    source = "governed_predictions"

    if governed_path is not None:
        df = pd.read_csv(governed_path)
    else:
        warnings.warn(
            f"No governed file under {pred_dir}; reconstructing from raw predictions."
        )
        raw_df = load_raw_predictions(pred_dir, split)
        df = reconstruct_governed(raw_df, threshold, decision_mode, symbolic_profile)
        thr_str = threshold_to_str(threshold)
        recon_path = (
            out_dir
            / f"reconstructed_governed_{split}_{decision_mode}_thr_{thr_str}.csv"
        )
        lists_to_json_columns(df).to_csv(recon_path, index=False)
        source = f"reconstructed:{recon_path.name}"

    df = df.copy()
    df["split"] = split
    if "neural_score" not in df.columns:
        raise ValueError("Predictions must include neural_score.")

    if "neural_pred" not in df.columns:
        df["neural_pred"] = apply_threshold(df["neural_score"].values, threshold)

    if "governed_pred" not in df.columns:
        status = df["symbolic_status"].astype(str)
        score_ok = df["neural_score"] >= threshold
        df["governed_pred"] = np.where(
            status == "invalid",
            0,
            np.where(score_ok, 1, 0),
        ).astype(int)

    for col in (
        "left_description",
        "right_description",
        "left_brand",
        "right_brand",
        "violated_constraints",
        "positive_evidence",
        "uncertain_reasons",
    ):
        if col not in df.columns:
            df[col] = ""

    return df, source


def enrich_row_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["left_description_short"] = out["left_description"].map(lambda x: _truncate(x))
    out["right_description_short"] = out["right_description"].map(lambda x: _truncate(x))
    out["title_similarity"] = [
        _title_similarity(str(a or ""), str(b or ""))
        for a, b in zip(out["left_title"], out["right_title"])
    ]
    out["left_model_tokens"] = [
        _json_list_str(sorted(extract_model_like_tokens(f"{t} {d}")))
        for t, d in zip(out["left_title"], out["left_description"])
    ]
    out["right_model_tokens"] = [
        _json_list_str(sorted(extract_model_like_tokens(f"{t} {d}")))
        for t, d in zip(out["right_title"], out["right_description"])
    ]

    def _qty_str(title: str, desc: str) -> str:
        qs = extract_quantities(f"{title} {desc}")
        return _json_list_str([f"{q.family}:{q.value}{q.unit}" for q in qs])

    out["left_quantities"] = [
        _qty_str(str(t or ""), str(d or "")) for t, d in zip(out["left_title"], out["left_description"])
    ]
    out["right_quantities"] = [
        _qty_str(str(t or ""), str(d or "")) for t, d in zip(out["right_title"], out["right_description"])
    ]
    return out


def classify_false_negative_type(row: pd.Series, threshold: float) -> str:
    if int(row["label"]) != 1 or int(row["governed_pred"]) != 0:
        return ""
    sym_invalid = str(row["symbolic_status"]) == "invalid"
    score_high = float(row["neural_score"]) >= threshold
    if not score_high and not sym_invalid:
        return "score_false_negative"
    if score_high and sym_invalid:
        return "symbolic_false_negative"
    return "both_or_other"


def _prepare_examples(
    df: pd.DataFrame,
    mask: pd.Series,
    category: str,
    threshold: float,
    max_examples: int,
    sort_ascending: bool = False,
) -> pd.DataFrame:
    sub = df[mask].copy()
    if sub.empty:
        return pd.DataFrame(columns=EXAMPLE_OUTPUT_COLS)

    if category.startswith("false_negatives") or category == "false_negatives":
        sub["false_negative_type"] = sub.apply(
            lambda r: classify_false_negative_type(r, threshold), axis=1
        )
        score_fn = sub["false_negative_type"] == "score_false_negative"
        sym_fn = sub["false_negative_type"] == "symbolic_false_negative"
        other_fn = sub["false_negative_type"] == "both_or_other"
        parts = []
        if score_fn.any():
            parts.append(
                sub[score_fn].sort_values("neural_score", ascending=False).head(max_examples)
            )
        if sym_fn.any():
            parts.append(
                sub[sym_fn].sort_values("neural_score", ascending=False).head(max_examples)
            )
        if other_fn.any():
            parts.append(
                sub[other_fn].sort_values("neural_score", ascending=False).head(max_examples)
            )
        if parts:
            sub = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["pair_id"])
        else:
            sub = sub.iloc[0:0]
        sub = sub.head(max_examples)
        analysis_cat = "false_negatives"
    else:
        sub["false_negative_type"] = ""
        sub = sub.sort_values("neural_score", ascending=sort_ascending).head(max_examples)
        analysis_cat = category

    sub["analysis_category"] = analysis_cat
    sub["auto_category"] = sub.apply(
        lambda r: categorize_example(
            r,
            sub.loc[r.name, "false_negative_type"] or analysis_cat,
        ),
        axis=1,
    )
    sub["short_explanation"] = sub.apply(
        lambda r: make_short_explanation(
            r,
            r["false_negative_type"] if r["false_negative_type"] else analysis_cat,
        ),
        axis=1,
    )

    for col in ("violated_constraints", "positive_evidence", "uncertain_reasons"):
        if col in sub.columns:
            sub[col] = sub[col].apply(
                lambda x: _json_list_str(parse_symbolic_list(x))
                if not isinstance(x, str) or not x.strip().startswith("[")
                else x
            )

    cols = [c for c in EXAMPLE_OUTPUT_COLS if c in sub.columns]
    return sub[cols]


def _safe_rate(num: int, denom: int) -> float:
    if denom <= 0:
        return float("nan")
    return round(num / denom, 6)


def build_summary_table(df: pd.DataFrame, split: str, threshold: float) -> pd.DataFrame:
    n = len(df)
    neural_accepts = int((df["neural_pred"] == 1).sum())
    governed_accepts = int((df["governed_pred"] == 1).sum())
    true_pos = int((df["label"] == 1).sum())
    false_pos = int((df["label"] == 0).sum())

    fn_mask = (df["label"] == 1) & (df["governed_pred"] == 0)
    fn_types = df[fn_mask].apply(lambda r: classify_false_negative_type(r, threshold), axis=1)

    specs: list[tuple[str, pd.Series]] = [
        (
            "blocked_false_positives",
            (df["neural_pred"] == 1) & (df["governed_pred"] == 0) & (df["label"] == 0),
        ),
        (
            "blocked_true_positives",
            (df["neural_pred"] == 1) & (df["governed_pred"] == 0) & (df["label"] == 1),
        ),
        (
            "remaining_false_positives",
            (df["governed_pred"] == 1) & (df["label"] == 0),
        ),
        ("false_negatives_total", fn_mask),
        ("false_negatives_score", fn_mask & (fn_types == "score_false_negative")),
        ("false_negatives_symbolic", fn_mask & (fn_types == "symbolic_false_negative")),
        ("false_negatives_other", fn_mask & (fn_types == "both_or_other")),
    ]

    rows: list[dict[str, Any]] = []
    for cat, mask in specs:
        sub = df[mask]
        cnt = len(sub)
        rows.append(
            {
                "split": split,
                "category": cat,
                "count": cnt,
                "rate_over_total": _safe_rate(cnt, n),
                "rate_over_neural_accepts": _safe_rate(
                    int(((df["neural_pred"] == 1) & mask).sum()), neural_accepts
                ),
                "rate_over_governed_accepts": _safe_rate(
                    int(((df["governed_pred"] == 1) & mask).sum()), governed_accepts
                ),
                "rate_over_true_positives": _safe_rate(
                    int(((df["label"] == 1) & mask).sum()), true_pos
                ),
                "rate_over_false_positives": _safe_rate(
                    int(((df["label"] == 0) & mask).sum()), false_pos
                ),
                "mean_neural_score": round(float(sub["neural_score"].mean()), 4)
                if cnt
                else float("nan"),
                "median_neural_score": round(float(sub["neural_score"].median()), 4)
                if cnt
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_category_counts(examples: dict[str, pd.DataFrame], split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cat, df in examples.items():
        if df.empty:
            continue
        for auto_cat, grp in df.groupby("auto_category", dropna=False):
            rows.append(
                {
                    "split": split,
                    "analysis_category": cat,
                    "auto_category": auto_cat,
                    "count": len(grp),
                    "mean_neural_score": round(float(grp["neural_score"].mean()), 4),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["split", "analysis_category", "auto_category", "count", "mean_neural_score"]
        )
    return pd.DataFrame(rows).sort_values(
        ["analysis_category", "count"], ascending=[True, False]
    )


def write_markdown_preview(
    examples: dict[str, pd.DataFrame],
    path: Path,
    split: str,
    threshold: float,
) -> None:
    sections = [
        ("blocked_false_positives", "Blocked false positives"),
        ("blocked_true_positives", "Blocked true positives"),
        ("remaining_false_positives", "Remaining false positives"),
        ("false_negatives", "False negatives"),
    ]
    lines = [
        "# Qualitative Error Analysis Preview",
        "",
        f"Split: `{split}` | Threshold: `{threshold}`",
        "",
    ]
    for key, heading in sections:
        df = examples.get(key, pd.DataFrame())
        lines.append(f"## {heading}")
        lines.append("")
        if df.empty:
            lines.append("_No examples in this category._")
            lines.append("")
            continue
        for i, (_, row) in enumerate(df.head(5).iterrows(), start=1):
            lines.append(f"### Example {i}")
            lines.append(f"- Pair ID: `{row.get('pair_id', '')}`")
            lines.append(f"- Label: `{row.get('label', '')}`")
            lines.append(f"- Neural score: `{row.get('neural_score', '')}`")
            lines.append(f"- Neural pred: `{row.get('neural_pred', '')}`")
            lines.append(f"- Governed pred: `{row.get('governed_pred', '')}`")
            lines.append(f"- Symbolic status: `{row.get('symbolic_status', '')}`")
            lines.append(f"- Violated constraints: `{row.get('violated_constraints', '')}`")
            if row.get("false_negative_type"):
                lines.append(f"- FN type: `{row.get('false_negative_type', '')}`")
            lines.append(f"- Auto-category: {row.get('auto_category', '')}")
            lines.append(f"- Explanation: {row.get('short_explanation', '')}")
            lines.append(f"- Left title: {row.get('left_title', '')}")
            lines.append(f"- Right title: {row.get('right_title', '')}")
            lines.append(f"- Left brand: {row.get('left_brand', '')}")
            lines.append(f"- Right brand: {row.get('right_brand', '')}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    pred_dir = OUTPUTS_DIR / "predictions" / args.variant / args.run_name
    out_dir = ensure_dir(
        OUTPUTS_DIR / "tables" / "qualitative_error_analysis" / args.variant / args.run_name
    )

    marker = out_dir / "qualitative_error_metadata.json"
    if marker.exists() and not args.overwrite:
        print(f"Outputs exist at {out_dir}. Use --overwrite to regenerate.")
        return 0

    df, data_source = load_analysis_frame(
        pred_dir,
        out_dir,
        args.split,
        args.threshold,
        args.decision_mode,
        args.symbolic_profile,
    )
    df = enrich_row_fields(df)
    df["neural_pred"] = (df["neural_score"] >= args.threshold).astype(int)
    df["governed_pred"] = df["governed_pred"].astype(int)
    df["label"] = df["label"].astype(int)

    blocked_fp = _prepare_examples(
        df,
        (df["neural_pred"] == 1) & (df["governed_pred"] == 0) & (df["label"] == 0),
        "blocked_false_positives",
        args.threshold,
        args.max_examples,
    )
    blocked_tp = _prepare_examples(
        df,
        (df["neural_pred"] == 1) & (df["governed_pred"] == 0) & (df["label"] == 1),
        "blocked_true_positives",
        args.threshold,
        args.max_examples,
    )
    remaining_fp = _prepare_examples(
        df,
        (df["governed_pred"] == 1) & (df["label"] == 0),
        "remaining_false_positives",
        args.threshold,
        args.max_examples,
    )
    false_neg = _prepare_examples(
        df,
        (df["label"] == 1) & (df["governed_pred"] == 0),
        "false_negatives",
        args.threshold,
        args.max_examples,
    )

    examples = {
        "blocked_false_positives": blocked_fp,
        "blocked_true_positives": blocked_tp,
        "remaining_false_positives": remaining_fp,
        "false_negatives": false_neg,
    }

    blocked_fp.to_csv(out_dir / "blocked_false_positives_examples.csv", index=False)
    blocked_tp.to_csv(out_dir / "blocked_true_positives_examples.csv", index=False)
    remaining_fp.to_csv(out_dir / "remaining_false_positives_examples.csv", index=False)
    false_neg.to_csv(out_dir / "false_negatives_examples.csv", index=False)

    summary_df = build_summary_table(df, args.split, args.threshold)
    summary_df.to_csv(out_dir / "qualitative_error_summary.csv", index=False)

    cat_df = build_category_counts(
        {
            "blocked_false_positives": blocked_fp,
            "blocked_true_positives": blocked_tp,
            "remaining_false_positives": remaining_fp,
            "false_negatives": false_neg,
        },
        args.split,
    )
    cat_df.to_csv(out_dir / "qualitative_error_categories.csv", index=False)

    write_markdown_preview(
        examples,
        out_dir / "qualitative_error_examples.md",
        args.split,
        args.threshold,
    )

    fn_types = false_neg["false_negative_type"].value_counts() if not false_neg.empty else {}
    top_blocked_fp = (
        blocked_fp["auto_category"].value_counts().head(5)
        if not blocked_fp.empty
        else pd.Series(dtype=int)
    )

    metadata = {
        "variant": args.variant,
        "run_name": args.run_name,
        "split": args.split,
        "threshold": args.threshold,
        "decision_mode": args.decision_mode,
        "symbolic_profile": args.symbolic_profile,
        "data_source": data_source,
        "max_examples": args.max_examples,
        "counts": {
            "blocked_false_positives": len(blocked_fp),
            "blocked_true_positives": len(blocked_tp),
            "remaining_false_positives": len(remaining_fp),
            "false_negatives_total": len(false_neg),
            "false_negatives_score": int(fn_types.get("score_false_negative", 0)),
            "false_negatives_symbolic": int(fn_types.get("symbolic_false_negative", 0)),
            "false_negatives_other": int(fn_types.get("both_or_other", 0)),
        },
        "output_dir": str(out_dir.relative_to(ROOT)).replace("\\", "/"),
    }
    save_json(metadata, out_dir / "qualitative_error_metadata.json")

    print("QUALITATIVE ERROR ANALYSIS COMPLETE")
    print(f"\nSplit: {args.split}")
    print(f"Threshold: {args.threshold}")
    print(f"Data source: {data_source}")
    print("\nCounts (sampled rows):")
    print(f"  blocked false positives: {len(blocked_fp)}")
    print(f"  blocked true positives: {len(blocked_tp)}")
    print(f"  remaining false positives: {len(remaining_fp)}")
    print(f"  false negatives total: {len(false_neg)}")
    print(f"    score false negatives: {metadata['counts']['false_negatives_score']}")
    print(f"    symbolic false negatives: {metadata['counts']['false_negatives_symbolic']}")
    print(f"    other false negatives: {metadata['counts']['false_negatives_other']}")

    if not top_blocked_fp.empty:
        print("\nTop blocked false-positive categories (sampled):")
        for cat, cnt in top_blocked_fp.items():
            print(f"  {cat}: {cnt}")

    print(f"\nSaved:\n  {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
