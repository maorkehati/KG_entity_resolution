"""Explore raw and normalized WDC Products pairwise data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_WDC_VARIANT_ID
from src.data_exploration import run_exploration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore WDC Products pairwise data (raw, normalized, processed)."
    )
    parser.add_argument(
        "--variant",
        default=DEFAULT_WDC_VARIANT_ID,
        help=f"WDC variant ID (default: {DEFAULT_WDC_VARIANT_ID})",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Maximum examples per category per split (default: 5)",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Print raw JSONL.GZ inspection before normalized stats",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_exploration(
        variant_id=args.variant,
        max_examples=args.max_examples,
        show_raw=args.show_raw,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
