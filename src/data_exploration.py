"""Exploration utilities for WDC Products pairwise data."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.config import DEFAULT_WDC_VARIANT_ID, OUTPUTS_DIR, PROJECT_ROOT
from src.data_loading import (
    WDCPairwiseDataset,
    WDCPairwiseSplit,
    load_wdc_pairwise_variant,
    processed_variant_dir,
    read_wdc_json_gz,
)


def _configure_stdio() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_stdio()
console = Console(legacy_windows=False)


def safe_print(message: str) -> None:
    """Print via Rich, falling back to ASCII-safe stdout on Windows."""
    try:
        console.print(message)
    except UnicodeEncodeError:
        text = str(message).encode("ascii", errors="replace").decode("ascii")
        print(text)

MISSING_COLUMNS = [
    "left_title",
    "right_title",
    "left_description",
    "right_description",
    "left_brand",
    "right_brand",
    "left_price",
    "right_price",
    "left_price_currency",
    "right_price_currency",
]

TEXT_LENGTH_COLUMNS = [
    "left_title",
    "right_title",
    "left_description",
    "right_description",
    "left_text",
    "right_text",
    "pair_text",
]

EXAMPLE_OUTPUT_COLUMNS = [
    "split",
    "example_type",
    "pair_id",
    "label",
    "title_similarity",
    "left_id",
    "right_id",
    "left_title",
    "right_title",
    "left_brand",
    "right_brand",
    "left_price",
    "right_price",
    "left_description",
    "right_description",
    "pair_text",
]


def exploration_output_dir(variant_id: str) -> Path:
    return OUTPUTS_DIR / "tables" / "data_exploration" / variant_id


def truncate_text(x: Any, max_chars: int = 300) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    text = str(x).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def safe_len(x: Any) -> int:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0
    return len(str(x))


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024**3:
        return f"{num_bytes / 1024**2:.1f} MB"
    return f"{num_bytes / 1024**3:.1f} GB"


def add_title_similarity(df: pd.DataFrame) -> pd.DataFrame:
    """Add title_similarity column using token_set_ratio."""
    out = df.copy()

    def _sim(row: pd.Series) -> float:
        left = row.get("left_title")
        right = row.get("right_title")
        if pd.isna(left) or pd.isna(right):
            return 0.0
        return float(
            fuzz.token_set_ratio(str(left), str(right))
        )

    out["title_similarity"] = out.apply(_sim, axis=1)
    return out


def _brands_equal(left: Any, right: Any) -> bool:
    if pd.isna(left) or pd.isna(right):
        return False
    left_s = str(left).strip().lower()
    right_s = str(right).strip().lower()
    if not left_s or not right_s:
        return False
    if left_s == right_s:
        return True
    return fuzz.ratio(left_s, right_s) >= 85


def _parse_price(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[^\d.\-]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def compute_missingness(df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in MISSING_COLUMNS:
        if col not in df.columns:
            continue
        missing = int(df[col].isna().sum())
        rows.append(
            {
                "split": split,
                "column": col,
                "missing_count": missing,
                "missing_pct": missing / len(df) if len(df) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def compute_text_length_stats(df: pd.DataFrame, split: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in TEXT_LENGTH_COLUMNS:
        if col not in df.columns:
            continue
        lengths = df[col].map(safe_len)
        rows.append(
            {
                "split": split,
                "column": col,
                "mean_len": float(lengths.mean()),
                "median_len": float(lengths.median()),
                "p90_len": float(lengths.quantile(0.9)),
                "max_len": int(lengths.max()),
            }
        )
    return pd.DataFrame(rows)


def compute_split_summary(
    df: pd.DataFrame,
    split: str,
    raw_path: Path,
) -> dict[str, Any]:
    positives = int((df["label"] == 1).sum())
    negatives = int((df["label"] == 0).sum())
    n = len(df)
    pos_rate = positives / n if n else 0.0
    imbalance = negatives / positives if positives else float("inf")

    dup_pairs = int(df.duplicated(subset=["pair_id"]).sum()) if "pair_id" in df.columns else 0
    left_ids = df["left_id"].nunique() if "left_id" in df.columns else 0
    right_ids = df["right_id"].nunique() if "right_id" in df.columns else 0

    def _miss_pct(col: str) -> float:
        if col not in df.columns or n == 0:
            return 0.0
        return float(df[col].isna().mean())

    file_size = raw_path.stat().st_size if raw_path.exists() else 0

    return {
        "split": split,
        "raw_path": str(raw_path),
        "file_size_bytes": file_size,
        "rows": n,
        "columns": len(df.columns),
        "positives": positives,
        "negatives": negatives,
        "positive_rate": pos_rate,
        "class_imbalance_ratio": imbalance,
        "unique_pair_ids": df["pair_id"].nunique() if "pair_id" in df.columns else 0,
        "pair_id_unique": bool(df["pair_id"].is_unique) if "pair_id" in df.columns else False,
        "unique_left_ids": left_ids,
        "unique_right_ids": right_ids,
        "id_overlap_left_right": (
            len(set(df["left_id"].dropna()) & set(df["right_id"].dropna()))
            if "left_id" in df.columns and "right_id" in df.columns
            else 0
        ),
        "duplicate_pairs": dup_pairs,
        "left_title_missing_pct": _miss_pct("left_title"),
        "right_title_missing_pct": _miss_pct("right_title"),
        "left_brand_missing_pct": _miss_pct("left_brand"),
        "right_brand_missing_pct": _miss_pct("right_brand"),
        "left_description_missing_pct": _miss_pct("left_description"),
        "right_description_missing_pct": _miss_pct("right_description"),
    }


def compute_brand_stats(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if "left_brand" not in df.columns or "right_brand" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["brand_equal"] = work.apply(
        lambda r: _brands_equal(r["left_brand"], r["right_brand"]),
        axis=1,
    )

    rows: list[dict[str, Any]] = []
    for label_value in ("all", 0, 1):
        subset = work if label_value == "all" else work[work["label"] == label_value]
        if len(subset) == 0:
            continue
        rows.append(
            {
                "split": split,
                "label": label_value,
                "brand_equal_rate": float(subset["brand_equal"].mean()),
                "left_unique_brands": int(subset["left_brand"].dropna().nunique()),
                "right_unique_brands": int(subset["right_brand"].dropna().nunique()),
            }
        )
    return pd.DataFrame(rows)


def compute_price_stats(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if "left_price" not in df.columns or "right_price" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["left_price_num"] = _parse_price(work["left_price"])
    work["right_price_num"] = _parse_price(work["right_price"])
    parse_ok = work["left_price_num"].notna() & work["right_price_num"].notna()

    same_currency = False
    if "left_price_currency" in work.columns and "right_price_currency" in work.columns:
        lc = work["left_price_currency"].astype(str).str.strip().str.upper()
        rc = work["right_price_currency"].astype(str).str.strip().str.upper()
        same_currency = (lc == rc) & lc.notna() & (lc != "NAN") & (lc != "")

    abs_diff = (work["left_price_num"] - work["right_price_num"]).abs()

    rows: list[dict[str, Any]] = []
    for label_value, group_name in [(None, "all"), (0, "negative"), (1, "positive")]:
        mask = parse_ok if label_value is None else parse_ok & (work["label"] == label_value)
        subset_diff = abs_diff[mask]
        subset_currency = same_currency[mask] if isinstance(same_currency, pd.Series) else pd.Series(dtype=bool)
        rows.append(
            {
                "split": split,
                "label_group": group_name,
                "left_price_missing_pct": float(work["left_price"].isna().mean()),
                "right_price_missing_pct": float(work["right_price"].isna().mean()),
                "numeric_parse_success_rate": float(mask.mean()),
                "median_abs_price_diff": float(subset_diff.median()) if len(subset_diff) else np.nan,
                "p90_abs_price_diff": float(subset_diff.quantile(0.9)) if len(subset_diff) else np.nan,
                "same_currency_rate": float(subset_currency.mean()) if len(subset_currency) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def select_example_pairs(
    df: pd.DataFrame,
    split: str,
    max_each: int = 5,
) -> pd.DataFrame:
    """Select positive, negative, hard-negative, and suspicious-positive examples."""
    work = add_title_similarity(df)
    examples: list[pd.DataFrame] = []

    def _pack(subset: pd.DataFrame, example_type: str) -> pd.DataFrame:
        if subset.empty:
            return pd.DataFrame()
        out = subset.head(max_each).copy()
        out["split"] = split
        out["example_type"] = example_type
        return out

    positives = work[work["label"] == 1].sort_values("title_similarity", ascending=False)
    negatives = work[work["label"] == 0].sort_values("title_similarity", ascending=True)
    examples.append(_pack(positives, "positive"))
    examples.append(_pack(negatives, "negative"))

    hard = work[work["label"] == 0].copy()
    if not hard.empty and "left_brand" in hard.columns:
        hard["brand_equal"] = hard.apply(
            lambda r: _brands_equal(r["left_brand"], r["right_brand"]),
            axis=1,
        )
        hard = hard.sort_values("title_similarity", ascending=False)
        hard = hard[(hard["title_similarity"] >= 60) | hard["brand_equal"]]
        examples.append(_pack(hard, "hard_negative"))

    suspicious = work[work["label"] == 1].copy()
    if not suspicious.empty:
        suspicious["brand_equal"] = suspicious.apply(
            lambda r: _brands_equal(r["left_brand"], r["right_brand"]),
            axis=1,
        )
        if "left_price_num" not in suspicious.columns:
            suspicious["left_price_num"] = _parse_price(suspicious["left_price"])
            suspicious["right_price_num"] = _parse_price(suspicious["right_price"])
        price_conflict = (
            suspicious["left_price_num"].notna()
            & suspicious["right_price_num"].notna()
            & (suspicious["left_price_num"] - suspicious["right_price_num"]).abs()
            > suspicious[["left_price_num", "right_price_num"]].max(axis=1) * 0.5
        )
        suspicious["suspicion_score"] = (
            (~suspicious["brand_equal"]).astype(int) * 2
            + (suspicious["title_similarity"] < 40).astype(int) * 2
            + price_conflict.astype(int)
        )
        suspicious = suspicious.sort_values(
            ["suspicion_score", "title_similarity"],
            ascending=[False, True],
        )
        suspicious = suspicious[suspicious["suspicion_score"] > 0]
        examples.append(_pack(suspicious, "suspicious_positive"))

    if not examples:
        return pd.DataFrame(columns=EXAMPLE_OUTPUT_COLUMNS)

    combined = pd.concat([e for e in examples if not e.empty], ignore_index=True)
    cols = [c for c in EXAMPLE_OUTPUT_COLUMNS if c in combined.columns]
    return combined[cols]


def print_pair_examples(examples: pd.DataFrame, max_chars: int = 300) -> None:
    if examples.empty:
        console.print("[yellow]No examples selected.[/yellow]")
        return

    for _, row in examples.iterrows():
        lines = [
            f"PAIR_ID: {row.get('pair_id', '')}",
            f"LABEL: {row.get('label', '')}",
            f"TITLE SIMILARITY: {row.get('title_similarity', '')}",
            f"TYPE: {row.get('example_type', '')} ({row.get('split', '')})",
            "LEFT:",
            f"  id: {row.get('left_id', '')}",
            f"  title: {truncate_text(row.get('left_title'), max_chars)}",
            f"  brand: {row.get('left_brand', '')}",
            f"  price: {row.get('left_price', '')}",
            f"  description: {truncate_text(row.get('left_description'), max_chars)}",
            "RIGHT:",
            f"  id: {row.get('right_id', '')}",
            f"  title: {truncate_text(row.get('right_title'), max_chars)}",
            f"  brand: {row.get('right_brand', '')}",
            f"  price: {row.get('right_price', '')}",
            f"  description: {truncate_text(row.get('right_description'), max_chars)}",
            "PAIR_TEXT:",
            f"  {truncate_text(row.get('pair_text'), max_chars)}",
        ]
        panel_text = "\n".join(lines)
        try:
            console.print(Panel(panel_text, title=str(row.get("example_type", "example"))))
        except UnicodeEncodeError:
            safe_print(f"--- {row.get('example_type', 'example')} ---\n{panel_text}")


def _detect_structure(df: pd.DataFrame) -> str:
    cols = [c.lower() for c in df.columns]
    if any(c.endswith("_left") or c.endswith("_right") for c in cols):
        return "flat (prefixed left/right columns)"
    if any(c in cols for c in ("left", "right", "record1", "record2", "pair")):
        return "nested (dict-like left/right or pair column)"
    if "label" in cols:
        return "flat (unknown layout, has label)"
    return "unknown"


def _inspect_nested_column(series: pd.Series, col_name: str) -> None:
    sample = series.dropna().head(3)
    for idx, value in sample.items():
        if isinstance(value, dict):
            console.print(f"  [{col_name}] row {idx} keys: {list(value.keys())}")
            preview = {k: truncate_text(v, 80) for k, v in list(value.items())[:6]}
            console.print(f"    example: {preview}")
        elif isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict):
                    console.print(f"  [{col_name}] row {idx} JSON keys: {list(parsed.keys())}")
            except json.JSONDecodeError:
                console.print(f"  [{col_name}] row {idx}: string value (not JSON dict)")


def inspect_raw_split(split_name: str, path: Path, max_rows: int = 5) -> pd.DataFrame:
    """Read and summarize a raw JSONL.GZ split file."""
    console.print(f"\n[bold cyan]Raw split: {split_name}[/bold cyan]")
    console.print(f"  path: {path}")
    if not path.exists():
        console.print(f"  [red]File not found: {path}[/red]")
        return pd.DataFrame()

    size = path.stat().st_size
    console.print(f"  size: {_format_size(size)}")

    df = read_wdc_json_gz(path)
    console.print(f"  shape: {df.shape}")
    console.print(f"  columns ({len(df.columns)}): {list(df.columns)}")
    console.print(f"  dtypes:\n{df.dtypes.to_string()}")
    console.print(f"  structure: {_detect_structure(df)}")

    label_cols = [c for c in df.columns if c.lower() in ("label", "match", "is_match", "y", "target")]
    if label_cols:
        col = label_cols[0]
        console.print(f"  label column: {col}")
        console.print(f"  label distribution:\n{df[col].value_counts().to_string()}")

    nested_candidates = [
        c
        for c in df.columns
        if c.lower() in ("left", "right", "record1", "record2", "pair", "e1", "e2")
        or df[c].dropna().apply(lambda x: isinstance(x, dict)).any()
    ]
    if nested_candidates:
        console.print("  nested column inspection:")
        for col in nested_candidates:
            _inspect_nested_column(df[col], col)

    mapping = Table(title=f"Raw -> normalized mapping hints ({split_name})")
    mapping.add_column("Normalized", style="cyan")
    mapping.add_column("Likely raw column(s)")
    hints = [
        ("pair_id", "pair_id"),
        ("left_id / right_id", "id_left, id_right, left_id, right_id"),
        ("left_title / right_title", "title_left, title_right, left_title, right_title"),
        ("left_brand / right_brand", "brand_left, brand_right"),
        ("left_description / right_description", "description_left, description_right"),
        ("left_price / right_price", "price_left, price_right"),
        ("left_price_currency / right_price_currency", "priceCurrency_left, priceCurrency_right"),
        ("label", ", ".join(label_cols) if label_cols else "label"),
    ]
    for norm, raw in hints:
        mapping.add_row(norm, raw)
    console.print(mapping)

    safe_print(f"  first {max_rows} raw rows:")
    preview = df.head(max_rows).to_string()
    safe_print(preview)

    return df


def print_normalized_split_stats(split: WDCPairwiseSplit) -> None:
    """Print detailed statistics for one normalized split."""
    df = split.df
    split_name = split.name
    path = split.path

    console.rule(f"[bold]NORMALIZED: {split_name}[/bold]")

    console.print("[bold]A. File information[/bold]")
    console.print(f"  raw file: {path}")
    if path.exists():
        console.print(f"  file size: {_format_size(path.stat().st_size)}")
    console.print(f"  rows: {len(df)}")
    console.print(f"  columns: {len(df.columns)}")
    console.print(f"  column names: {list(df.columns)}")
    console.print(f"  dtypes:\n{df.dtypes.to_string()}")

    console.print("\n[bold]B. Label distribution[/bold]")
    pos = int((df["label"] == 1).sum())
    neg = int((df["label"] == 0).sum())
    n = len(df)
    console.print(f"  label 0: {neg}")
    console.print(f"  label 1: {pos}")
    console.print(f"  positive rate: {pos / n:.4f}" if n else "  positive rate: n/a")
    console.print(f"  class imbalance (neg/pos): {neg / pos:.4f}" if pos else "  class imbalance: inf")

    console.print("\n[bold]C. Basic pair information[/bold]")
    if "pair_id" in df.columns:
        console.print(f"  unique pair_id: {df['pair_id'].nunique()}")
        console.print(f"  pair_id unique: {df['pair_id'].is_unique}")
    if "left_id" in df.columns and "right_id" in df.columns:
        left_set = set(df["left_id"].dropna())
        right_set = set(df["right_id"].dropna())
        console.print(f"  unique left_id: {len(left_set)}")
        console.print(f"  unique right_id: {len(right_set)}")
        console.print(f"  left/right id overlap: {len(left_set & right_set)}")
    console.print(f"  duplicate rows (all columns): {int(df.duplicated().sum())}")

    console.print("\n[bold]D. Missingness[/bold]")
    miss = compute_missingness(df, split_name)
    if miss.empty:
        console.print("  (no tracked columns)")
    else:
        table = Table()
        for col in ("column", "missing_count", "missing_pct"):
            table.add_column(col)
        for _, row in miss.iterrows():
            table.add_row(
                str(row["column"]),
                str(int(row["missing_count"])),
                f"{row['missing_pct']:.2%}",
            )
        console.print(table)

    console.print("\n[bold]E. Text length statistics[/bold]")
    tlen = compute_text_length_stats(df, split_name)
    if not tlen.empty:
        console.print(tlen.to_string(index=False))

    console.print("\n[bold]F. Brand statistics[/bold]")
    if "left_brand" in df.columns:
        console.print(f"  unique left brands: {df['left_brand'].dropna().nunique()}")
        console.print(f"  unique right brands: {df['right_brand'].dropna().nunique()}")
        console.print("  top 20 left brands:")
        console.print(df["left_brand"].value_counts().head(20).to_string())
        console.print("  top 20 right brands:")
        console.print(df["right_brand"].value_counts().head(20).to_string())
        bstats = compute_brand_stats(df, split_name)
        if not bstats.empty:
            console.print(bstats.to_string(index=False))

    console.print("\n[bold]G. Price statistics[/bold]")
    pstats = compute_price_stats(df, split_name)
    if pstats.empty:
        console.print("  (price fields unavailable)")
    else:
        console.print(pstats.to_string(index=False))


def print_symbolic_constraint_signals(dataset: WDCPairwiseDataset) -> None:
    """Summarize signals relevant to symbolic governance constraints."""
    console.rule("[bold]POTENTIAL SYMBOLIC CONSTRAINT SIGNALS[/bold]")
    for split in (dataset.train, dataset.valid, dataset.test):
        df = add_title_similarity(split.df)
        console.print(f"\n[bold]{split.name}[/bold]")
        if "left_brand" in df.columns:
            brand_eq = df.apply(
                lambda r: _brands_equal(r["left_brand"], r["right_brand"]),
                axis=1,
            )
            for label, name in [(0, "negatives"), (1, "positives")]:
                sub = df[df["label"] == label]
                if len(sub) == 0:
                    continue
                eq_rate = brand_eq[sub.index].mean()
                console.print(f"  brand match rate among {name}: {eq_rate:.2%}")
        high_sim_neg = ((df["label"] == 0) & (df["title_similarity"] >= 70)).sum()
        low_sim_pos = ((df["label"] == 1) & (df["title_similarity"] < 40)).sum()
        console.print(f"  high title similarity negatives (>=70): {high_sim_neg}")
        console.print(f"  low title similarity positives (<40): {low_sim_pos}")
        if "raw__is_hard_negative" in df.columns:
            console.print(
                f"  raw is_hard_negative=True: {int(df['raw__is_hard_negative'].sum())}"
            )


def check_processed_parquet(
    variant_id: str,
    live_dataset: WDCPairwiseDataset,
) -> None:
    """Compare processed parquet files against freshly loaded normalized data."""
    console.rule("[bold]PROCESSED PARQUET CHECK[/bold]")
    out_dir = processed_variant_dir(variant_id)
    missing = [s for s in ("train", "valid", "test") if not (out_dir / f"{s}.parquet").exists()]

    if missing:
        console.print(
            "[yellow]Processed parquet files not found for: "
            + ", ".join(missing)
            + "[/yellow]\n"
            "Run:\n  python scripts/03_prepare_variant.py"
        )
        return

    from src.data_loading import load_processed_variant

    processed = load_processed_variant(variant_id)
    live_map = {
        "train": live_dataset.train,
        "valid": live_dataset.valid,
        "test": live_dataset.test,
    }
    proc_map = {
        "train": processed.train,
        "valid": processed.valid,
        "test": processed.test,
    }

    for name in ("train", "valid", "test"):
        live_df = live_map[name].df
        proc_df = proc_map[name].df
        path = out_dir / f"{name}.parquet"
        console.print(f"\n[bold]{name}[/bold] parquet: {path}")
        console.print(f"  parquet shape: {proc_df.shape}")
        console.print(f"  live normalized shape: {live_df.shape}")
        console.print(f"  row count match: {len(proc_df) == len(live_df)}")
        live_labels = live_df["label"].value_counts().sort_index().to_dict()
        proc_labels = proc_df["label"].value_counts().sort_index().to_dict()
        console.print(f"  live label dist: {live_labels}")
        console.print(f"  parquet label dist: {proc_labels}")
        console.print(f"  label distribution match: {live_labels == proc_labels}")


def run_exploration(
    variant_id: str = DEFAULT_WDC_VARIANT_ID,
    max_examples: int = 5,
    show_raw: bool = False,
) -> Path:
    """
    Run full data exploration for a WDC variant.

    Returns the output directory where CSV artifacts were saved.
    """
    console.rule("[bold]WDC Products Data Exploration[/bold]")
    console.print(f"variant_id: {variant_id}")

    try:
        dataset = load_wdc_pairwise_variant(variant_id)
    except FileNotFoundError as exc:
        console.print(f"[red]Error: {exc}[/red]")
        console.print(
            "\nPlace the WDC Products pairwise archive (e.g. 50pair.zip) in data/raw/ "
            "and run: python scripts/00_download_data.py"
        )
        raise SystemExit(1) from exc

    out_dir = exploration_output_dir(variant_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    if show_raw:
        console.rule("[bold]RAW DATA INSPECTION[/bold]")
        for split in (dataset.train, dataset.valid, dataset.test):
            inspect_raw_split(split.name, split.path, max_rows=5)

    console.rule("[bold]NORMALIZED DATA INSPECTION[/bold]")
    summary_rows: list[dict[str, Any]] = []
    missing_frames: list[pd.DataFrame] = []
    text_frames: list[pd.DataFrame] = []
    brand_frames: list[pd.DataFrame] = []
    example_frames: list[pd.DataFrame] = []

    for split in (dataset.train, dataset.valid, dataset.test):
        print_normalized_split_stats(split)
        summary_rows.append(compute_split_summary(split.df, split.name, split.path))
        missing_frames.append(compute_missingness(split.df, split.name))
        text_frames.append(compute_text_length_stats(split.df, split.name))
        b = compute_brand_stats(split.df, split.name)
        if not b.empty:
            brand_frames.append(b)
        examples = select_example_pairs(split.df, split.name, max_each=max_examples)
        if not examples.empty:
            example_frames.append(examples)

    pd.DataFrame(summary_rows).to_csv(out_dir / "split_summary.csv", index=False)
    pd.concat(missing_frames, ignore_index=True).to_csv(
        out_dir / "missingness_by_split.csv", index=False
    )
    pd.concat(text_frames, ignore_index=True).to_csv(
        out_dir / "text_length_stats.csv", index=False
    )
    if brand_frames:
        pd.concat(brand_frames, ignore_index=True).to_csv(
            out_dir / "brand_stats.csv", index=False
        )
    if example_frames:
        pd.concat(example_frames, ignore_index=True).to_csv(
            out_dir / "example_pairs.csv", index=False
        )

    check_processed_parquet(variant_id, dataset)
    print_symbolic_constraint_signals(dataset)

    console.rule("[bold]EXAMPLE PAIRS[/bold]")
    if example_frames:
        all_examples = pd.concat(example_frames, ignore_index=True)
        print_pair_examples(all_examples, max_chars=300)
    else:
        console.print("[yellow]No examples collected.[/yellow]")

    console.print(f"\n[green]Artifacts saved to:[/green] {out_dir.relative_to(PROJECT_ROOT)}")
    return out_dir
