"""Smoke test for processed WDC pairwise parquet splits."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_WDC_VARIANT_ID
from src.data_loading import load_processed_variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test processed WDC variant splits.")
    parser.add_argument("--variant", default=DEFAULT_WDC_VARIANT_ID)
    return parser.parse_args()


def _assert_label_mix(df, split: str) -> None:
    positives = int((df["label"] == 1).sum())
    negatives = int((df["label"] == 0).sum())
    assert positives > 0, f"{split}: expected at least one positive label"
    assert negatives > 0, f"{split}: expected at least one negative label"


def main() -> int:
    variant_id = parse_args().variant
    try:
        dataset = load_processed_variant(variant_id)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print("Run: python scripts/03_prepare_variant.py", file=sys.stderr)
        return 1

    for split in (dataset.train, dataset.valid, dataset.test):
        df = split.df
        print(f"\n=== {split.name} ===")
        print(f"shape: {df.shape}")
        print(f"label distribution:\n{df['label'].value_counts().to_string()}")

        assert "label" in df.columns
        assert set(df["label"].unique()).issubset({0, 1})
        _assert_label_mix(df, split.name)

        print("examples:")
        sample = df.sample(min(3, len(df)), random_state=42)
        for _, row in sample.iterrows():
            text = str(row.get("pair_text", ""))[:500]
            print(f"  pair_id={row['pair_id']}")
            print(f"    left_title={row.get('left_title')!r}")
            print(f"    right_title={row.get('right_title')!r}")
            print(f"    left_brand={row.get('left_brand')!r}")
            print(f"    right_brand={row.get('right_brand')!r}")
            print(f"    label={row['label']}")
            print(f"    pair_text={text!r}...")

    print("\nDataset smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
