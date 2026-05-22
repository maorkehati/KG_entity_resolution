"""Inspect downloaded files under data/raw and data/interim."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inspect_dataset import inspect_local_data


def main() -> int:
    inspect_local_data()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
