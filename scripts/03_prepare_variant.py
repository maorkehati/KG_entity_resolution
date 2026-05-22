"""Materialize the default WDC pairwise variant to processed parquet files."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_WDC_VARIANT_ID
from src.data_loading import load_wdc_pairwise_variant, save_processed_variant
from src.download_wdc import extract_wdc_archives


def main() -> int:
    variant_id = DEFAULT_WDC_VARIANT_ID
    print(f"Preparing WDC variant: {variant_id}")

    try:
        extract_wdc_archives(force=False)
        dataset = load_wdc_pairwise_variant(variant_id)
        source_paths = {
            split.name: split.path
            for split in (dataset.train, dataset.valid, dataset.test)
        }
        out_dir = save_processed_variant(dataset, source_paths)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "\nTo fix:\n"
            "  1. Download the WDC Products pairwise 50% corner-case archive (50pair.zip)\n"
            "     from https://webdatacommons.org/largescaleproductcorpus/wdc-products/\n"
            "  2. Place it in data/raw/\n"
            "  3. Run: python scripts/00_download_data.py\n"
            "  4. Re-run: python scripts/03_prepare_variant.py",
            file=sys.stderr,
        )
        return 1
    except (KeyError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nProcessed data written to: {out_dir}")
    for split in (dataset.train, dataset.valid, dataset.test):
        print(f"  {split.name}: {len(split.df)} rows -> {out_dir / (split.name + '.parquet')}")
    print(f"  metadata: {out_dir / 'variant_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
