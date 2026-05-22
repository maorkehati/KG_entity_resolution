"""Download WDC Products benchmark page and list candidate data URLs."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_RAW_DIR
from src.download_wdc import PAGE_HTML_PATH, NOTES_PATH, download_wdc_sources


def main() -> int:
    try:
        candidates = download_wdc_sources()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\nCandidate download links:")
    if candidates:
        for i, url in enumerate(candidates, start=1):
            print(f"  {i}. {url}")
    else:
        print("  (none found — open the saved HTML and choose links manually)")

    print(f"\nHTML snapshot: {PAGE_HTML_PATH}")
    print(f"Source notes:  {NOTES_PATH}")
    print(f"Data directory: {DATA_RAW_DIR}")
    print("\nNext: download a chosen archive into data/raw/ or data/interim/, then run:")
    print("  python scripts/01_inspect_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
