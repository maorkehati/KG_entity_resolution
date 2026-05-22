"""Load and normalize WDC Products pairwise benchmark variants."""

from __future__ import annotations

import fnmatch
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    DEFAULT_WDC_VARIANT_ID,
    PROJECT_ROOT,
    WDC_INTERIM_PRODUCTS_DIR,
    WDCVariant,
    get_wdc_variant,
)

SEARCH_ROOTS = (WDC_INTERIM_PRODUCTS_DIR, DATA_RAW_DIR)

LABEL_CANDIDATES = ("label", "match", "is_match", "matching", "y", "target")
PAIR_ID_CANDIDATES = ("pair_id", "pairid", "id")

OFFER_FIELDS = (
    "id",
    "title",
    "description",
    "brand",
    "price",
    "price_currency",
)

NORMALIZED_COLUMNS = (
    "pair_id",
    "left_id",
    "right_id",
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
    "label",
    "left_text",
    "right_text",
    "pair_text",
)


@dataclass
class WDCPairwiseSplit:
    name: str
    path: Path
    df: pd.DataFrame


@dataclass
class WDCPairwiseDataset:
    variant_id: str
    train: WDCPairwiseSplit
    valid: WDCPairwiseSplit
    test: WDCPairwiseSplit


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def extract_offer_field(row_or_obj: Any, field: str) -> Any:
    """Read a product attribute from a dict/Series with several naming conventions."""
    if row_or_obj is None or (isinstance(row_or_obj, float) and pd.isna(row_or_obj)):
        return None

    aliases: dict[str, list[str]] = {
        "id": ["id", "offer_id", "product_id"],
        "title": ["title", "name", "product_name"],
        "description": ["description", "desc", "body"],
        "brand": ["brand", "manufacturer", "maker"],
        "price": ["price", "amount", "cost"],
        "price_currency": [
            "price_currency",
            "pricecurrency",
            "currency",
            "priceCurrency",
        ],
    }
    keys = aliases.get(field, [field])

    if isinstance(row_or_obj, pd.Series):
        row_or_obj = row_or_obj.to_dict()
    if not isinstance(row_or_obj, dict):
        return None

    lower_map = {str(k).lower(): k for k in row_or_obj}
    for key in keys:
        if key in row_or_obj:
            return row_or_obj[key]
        if key.lower() in lower_map:
            return row_or_obj[lower_map[key.lower()]]
    return None


def coerce_label(series: pd.Series) -> pd.Series:
    """Map heterogeneous labels to integers 0/1."""
    def _to_int(value: Any) -> int:
        if pd.isna(value):
            raise ValueError("Missing label value")
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            iv = int(value)
            if iv in (0, 1):
                return iv
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "match", "positive", "pos"}:
            return 1
        if text in {"0", "false", "no", "non-match", "negative", "neg"}:
            return 0
        raise ValueError(f"Unrecognized label value: {value!r}")

    return series.map(_to_int)


def _search_roots(data_root: Path | None) -> list[Path]:
    if data_root is not None:
        return [data_root]
    return [r for r in SEARCH_ROOTS if r.exists()]


def discover_json_gz_nearby(roots: list[Path]) -> list[str]:
    names: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json.gz"):
            names.append(str(path.relative_to(root)))
    return sorted(set(names))


def _glob_matches(root: Path, pattern: str) -> list[Path]:
    matches: list[Path] = []
    for path in root.rglob("*.json.gz"):
        if fnmatch.fnmatch(path.name.lower(), pattern.lower()):
            matches.append(path)
    return sorted(set(matches))


def _rank_candidate(path: Path, split: str, variant: WDCVariant) -> int:
    """Higher score = better match for the configured variant."""
    name = path.name.lower()
    score = 0

    if "pair" in name or "wdcproduct" in name:
        score += 8
    if variant.task == "pairwise" and "multi" not in name:
        score += 4

    split_tokens = {
        "train": ["train"],
        "valid": ["valid", "val"],
        "test": ["test", "gs"],
    }
    if any(tok in name for tok in split_tokens.get(split, [])):
        score += 16

    cc = str(variant.corner_case_ratio)
    if f"{cc}cc" in name or f"cc{cc}" in name or cc in name:
        score += 12

    dev = variant.development_size.lower()
    if dev in name:
        score += 10

    if split == "test":
        unseen = variant.unseen_test_ratio
        if f"rnd{unseen:03d}un" in name or f"{unseen}un" in name:
            score += 20
        if unseen == 100 and "rnd100un" in name:
            score += 5
    else:
        # Train/valid use fully-seen entity pool in WDC naming (rnd000un).
        if "rnd000un" in name or "000un" in name:
            score += 12
        if "gs" in name:
            score -= 50

    if split in ("train", "valid") and "gs" in name and "train" not in name and "valid" not in name:
        score -= 30

    return score


def resolve_variant_files(
    variant_id: str,
    data_root: Path | None = None,
) -> dict[str, Path]:
    """Find train/valid/test JSON-lines gzip files for a registered variant."""
    variant = get_wdc_variant(variant_id)
    roots = _search_roots(data_root)

    if not roots:
        raise FileNotFoundError(
            f"No data directories found for variant {variant_id!r}. "
            f"Expected one of: {WDC_INTERIM_PRODUCTS_DIR}, {DATA_RAW_DIR}. "
            "Download 50pair.zip into data/raw/ and run scripts/00_download_data.py."
        )

    resolved: dict[str, Path] = {}
    for split, patterns in variant.file_patterns.items():
        candidates: list[Path] = []
        for root in roots:
            for pattern in patterns:
                candidates.extend(_glob_matches(root, pattern))

        candidates = sorted(set(candidates))
        if not candidates:
            nearby = discover_json_gz_nearby(roots)
            raise FileNotFoundError(
                f"Could not find {split!r} file for variant {variant_id!r}.\n"
                f"  Searched roots: {[str(r) for r in roots]}\n"
                f"  Patterns tried: {patterns}\n"
                f"  Nearby .json.gz files:\n"
                + "".join(f"    - {n}\n" for n in nearby[:40])
                + (
                    f"    ... ({len(nearby) - 40} more)\n"
                    if len(nearby) > 40
                    else ""
                )
                + "\nPlace the WDC Products pairwise archive (e.g. 50pair.zip) in "
                "data/raw/ and run scripts/00_download_data.py to extract."
            )

        ranked = sorted(
            candidates,
            key=lambda p: (_rank_candidate(p, split, variant), p.name),
            reverse=True,
        )
        chosen = ranked[0]

        print(f"[{variant_id}] split={split!r}")
        print(f"  candidates ({len(candidates)}):")
        for path in candidates:
            marker = " <-- selected" if path == chosen else ""
            print(f"    - {path}{marker}")
        print(f"  selected: {chosen}")

        resolved[split] = chosen

    return resolved


def read_wdc_json_gz(path: Path) -> pd.DataFrame:
    """Read a WDC JSON-lines gzip file."""
    try:
        return pd.read_json(path, lines=True, compression="gzip")
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}: {exc}") from exc


def _nested_pair_columns(df: pd.DataFrame) -> tuple[str, str, str] | None:
    for left_name, right_name, label_name in (
        ("left", "right", "label"),
        ("record1", "record2", "label"),
        ("e1", "e2", "label"),
        ("offer1", "offer2", "label"),
    ):
        if left_name in df.columns and right_name in df.columns:
            label_col = label_name if label_name in df.columns else first_existing_column(
                df, list(LABEL_CANDIDATES)
            )
            if label_col:
                return left_name, right_name, label_col
    if "pair" in df.columns:
        label_col = first_existing_column(df, list(LABEL_CANDIDATES))
        if label_col:
            return "pair", "pair", label_col
    return None


def _flat_column_map(df: pd.DataFrame) -> dict[str, str | None]:
    """Map normalized field names to source column names."""
    mapping: dict[str, str | None] = {}
    for side in ("left", "right"):
        for field in OFFER_FIELDS:
            if field == "price_currency":
                candidates = [
                    f"{side}_price_currency",
                    f"{side}_pricecurrency",
                    f"priceCurrency_{side}",
                    f"{side}_priceCurrency",
                    f"price_currency_{side}",
                ]
            else:
                candidates = [
                    f"{side}_{field}",
                    f"{field}_{side}",
                    f"id_{side}" if field == "id" else f"{side}_{field}",
                    f"ltable_{field}" if side == "left" else f"rtable_{field}",
                ]
            mapping[f"{side}_{field}"] = first_existing_column(df, candidates)
    return mapping


def normalize_wdc_pairwise_df(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Normalize heterogeneous WDC pairwise rows to a consistent schema."""
    work = df.copy()
    out = pd.DataFrame(index=work.index)

    label_col = first_existing_column(work, list(LABEL_CANDIDATES))
    if label_col is None:
        raise ValueError(f"Split {split!r}: no label column found in {list(work.columns)}")

    pair_id_col = first_existing_column(work, list(PAIR_ID_CANDIDATES))
    flat_map = _flat_column_map(work)
    nested = _nested_pair_columns(work)

    if nested and not any(flat_map.values()):
        left_col, right_col, _ = nested
        records: list[dict[str, Any]] = []
        for _, row in work.iterrows():
            rec: dict[str, Any] = {}
            left_obj = row[left_col]
            right_obj = row[right_col]
            if isinstance(left_obj, str):
                try:
                    left_obj = json.loads(left_obj)
                except json.JSONDecodeError:
                    left_obj = None
            if isinstance(right_obj, str):
                try:
                    right_obj = json.loads(right_obj)
                except json.JSONDecodeError:
                    right_obj = None
            for side, obj in (("left", left_obj), ("right", right_obj)):
                for field in OFFER_FIELDS:
                    rec[f"{side}_{field}"] = extract_offer_field(obj, field)
            records.append(rec)
        out = pd.DataFrame(records, index=work.index)
    else:
        for norm_name, src_col in flat_map.items():
            if src_col:
                out[norm_name] = work[src_col]
            else:
                out[norm_name] = None

    out["label"] = coerce_label(work[label_col])
    if pair_id_col:
        out["pair_id"] = work[pair_id_col].astype(str)
    else:
        out["pair_id"] = [f"{split}_{i}" for i in range(len(work))]

    for side in ("left", "right"):
        title_col = f"{side}_title"
        if title_col not in out.columns or out[title_col].isna().all():
            text_cols = [
                c
                for c in work.columns
                if side in c.lower()
                and work[c].dtype == object
                and c not in (label_col, pair_id_col)
            ]
            if text_cols:
                out[title_col] = work[text_cols].astype(str).agg(" ".join, axis=1)

    for col in work.columns:
        if col not in out.columns and not col.startswith("raw__"):
            out[f"raw__{col}"] = work[col]

    out["left_text"] = out.apply(serialize_product_side, axis=1, side="left")
    out["right_text"] = out.apply(serialize_product_side, axis=1, side="right")
    out["pair_text"] = out.apply(serialize_product_pair_row, axis=1)

    return out


def serialize_product_side(row: pd.Series, side: str) -> str:
    """Serialize one product offer for downstream scorers."""
    parts: list[str] = []
    prefix = side.upper()
    field_labels = (
        ("title", "TITLE"),
        ("brand", "BRAND"),
        ("description", "DESCRIPTION"),
        ("price", "PRICE"),
        ("price_currency", "PRICE_CURRENCY"),
    )
    for field, tag in field_labels:
        col = f"{side}_{field}"
        value = row.get(col)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            text = str(value).strip()
            if text:
                parts.append(f"[{tag}] {text}")
    if not parts:
        return f"[{prefix}]"
    return f"[{prefix}] " + " ".join(parts)


def serialize_product_pair_row(row: pd.Series) -> str:
    """Serialize a pairwise example as LEFT/RIGHT product text."""
    left = serialize_product_side(row, "left")
    right = serialize_product_side(row, "right")
    return f"{left}\n{right}"


def validate_pairwise_schema(df: pd.DataFrame, split: str) -> None:
    """Validate normalized pairwise schema and print diagnostics."""
    required = [
        "pair_id",
        "left_id",
        "right_id",
        "left_title",
        "right_title",
        "label",
        "left_text",
        "right_text",
        "pair_text",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Split {split!r} missing required columns: {missing}")

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"Split {split!r}: labels must be 0/1, got {df['label'].unique()}")

    if df["pair_id"].duplicated().any():
        dupes = int(df["pair_id"].duplicated().sum())
        print(
            f"Warning: split {split!r} has {dupes} duplicate pair_id values.",
            file=sys.stderr,
        )

    print(f"[validate] split={split!r} shape={df.shape}")
    print(f"  label distribution:\n{df['label'].value_counts().to_string()}")

    optional = ["left_brand", "right_brand", "left_description", "right_description"]
    for col in optional:
        if col not in df.columns:
            continue
        null_rate = float(df[col].isna().mean())
        if null_rate > 0.5:
            print(
                f"  warning: {col} is {null_rate:.1%} missing",
                file=sys.stderr,
            )


def load_wdc_pairwise_variant(
    variant_id: str = DEFAULT_WDC_VARIANT_ID,
    data_root: Path | None = None,
) -> WDCPairwiseDataset:
    """Load train/valid/test splits for a registered WDC pairwise variant."""
    variant = get_wdc_variant(variant_id)
    if variant.task != "pairwise":
        raise ValueError(f"Variant {variant_id!r} is not a pairwise task.")

    paths = resolve_variant_files(variant_id, data_root=data_root)
    splits: dict[str, WDCPairwiseSplit] = {}

    for split_name in ("train", "valid", "test"):
        path = paths[split_name]
        raw = read_wdc_json_gz(path)
        normalized = normalize_wdc_pairwise_df(raw, split=split_name)
        validate_pairwise_schema(normalized, split=split_name)
        splits[split_name] = WDCPairwiseSplit(
            name=split_name,
            path=path,
            df=normalized,
        )

    return WDCPairwiseDataset(
        variant_id=variant_id,
        train=splits["train"],
        valid=splits["valid"],
        test=splits["test"],
    )


def processed_variant_dir(variant_id: str) -> Path:
    return DATA_PROCESSED_DIR / variant_id


def save_processed_variant(
    dataset: WDCPairwiseDataset,
    source_paths: dict[str, Path],
) -> Path:
    """Write parquet splits and variant_metadata.json."""
    variant = get_wdc_variant(dataset.variant_id)
    out_dir = processed_variant_dir(dataset.variant_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "variant_id": variant.variant_id,
        "task": variant.task,
        "corner_case_ratio": variant.corner_case_ratio,
        "development_size": variant.development_size,
        "unseen_test_ratio": variant.unseen_test_ratio,
        "description": variant.description,
        "source_files": {k: str(v) for k, v in source_paths.items()},
        "splits": {},
        "normalized_columns": list(dataset.train.df.columns),
    }

    for split in (dataset.train, dataset.valid, dataset.test):
        out_path = out_dir / f"{split.name}.parquet"
        split.df.to_parquet(out_path, index=False)
        metadata["splits"][split.name] = {
            "rows": len(split.df),
            "output_path": str(out_path.relative_to(PROJECT_ROOT)),
            "label_distribution": split.df["label"].value_counts().to_dict(),
        }

    meta_path = out_dir / "variant_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir


def load_processed_variant(variant_id: str = DEFAULT_WDC_VARIANT_ID) -> WDCPairwiseDataset:
    """Load previously materialized parquet splits."""
    out_dir = processed_variant_dir(variant_id)
    if not out_dir.exists():
        raise FileNotFoundError(
            f"Processed data not found at {out_dir}. "
            "Run: python scripts/03_prepare_variant.py"
        )

    splits = {}
    for name in ("train", "valid", "test"):
        path = out_dir / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing processed split: {path}")
        splits[name] = WDCPairwiseSplit(name=name, path=path, df=pd.read_parquet(path))

    return WDCPairwiseDataset(
        variant_id=variant_id,
        train=splits["train"],
        valid=splits["valid"],
        test=splits["test"],
    )
