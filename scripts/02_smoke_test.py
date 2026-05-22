"""Quick sanity checks for project scaffolding."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    DATA_INTERIM_DIR,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
)
from src.download_wdc import PAGE_HTML_PATH


REQUIRED_DIRS = (
    DATA_RAW_DIR,
    DATA_INTERIM_DIR,
    DATA_PROCESSED_DIR,
    OUTPUTS_DIR / "figures",
    OUTPUTS_DIR / "tables",
    OUTPUTS_DIR / "predictions",
)


def main() -> int:
    errors: list[str] = []

    try:
        import numpy  # noqa: F401
        import pandas  # noqa: F401
        import requests  # noqa: F401
        import sklearn  # noqa: F401
        from rich.console import Console  # noqa: F401

        import src.config
        import src.download_wdc
        import src.inspect_dataset
    except ImportError as exc:
        errors.append(f"Import failed: {exc}")

    if PROJECT_ROOT != ROOT:
        errors.append(f"PROJECT_ROOT mismatch: {PROJECT_ROOT} vs {ROOT}")

    for directory in REQUIRED_DIRS:
        if not directory.is_dir():
            errors.append(f"Missing directory: {directory}")

    if not PAGE_HTML_PATH.is_file():
        errors.append(
            f"Missing WDC page HTML at {PAGE_HTML_PATH}. "
            "Run: python scripts/00_download_data.py"
        )

    if errors:
        print("Smoke test failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("Smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
