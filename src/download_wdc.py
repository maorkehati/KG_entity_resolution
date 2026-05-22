"""Fetch WDC Products benchmark page and discover download candidates."""

from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from zipfile import BadZipFile, ZipFile

import requests

from src.config import (
    DATA_INTERIM_DIR,
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    OUTPUTS_DIR,
    WDC_EXPLICIT_CANDIDATE_URLS,
    WDC_INTERIM_PRODUCTS_DIR,
    WDC_PRODUCTS_GITHUB_URL,
    WDC_PRODUCTS_PAGE_URL,
)

PAGE_HTML_PATH = DATA_RAW_DIR / "wdc_products_page.html"
NOTES_PATH = DATA_RAW_DIR / "dataset_source_notes.txt"

# Extensions and path keywords that often indicate benchmark assets.
_DOWNLOAD_HINTS = (
    ".zip",
    ".gz",
    ".json",
    ".tar",
    ".csv",
    ".tsv",
    "wdc-products",
    "wdcproducts",
    "largescaleproductcorpus",
    "data.dws",
)


class _LinkExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                self.links.append(value.strip())


def ensure_data_directories() -> None:
    """Create data and output directory trees."""
    for path in (
        DATA_RAW_DIR,
        DATA_INTERIM_DIR,
        DATA_PROCESSED_DIR,
        OUTPUTS_DIR / "figures",
        OUTPUTS_DIR / "tables",
        OUTPUTS_DIR / "predictions",
    ):
        path.mkdir(parents=True, exist_ok=True)


def _normalize_link(href: str, base_url: str) -> str | None:
    href = href.strip()
    if not href or href.startswith("#") or href.lower().startswith("javascript:"):
        return None
    absolute = urljoin(base_url, href)
    parsed = urlparse(absolute)
    if parsed.scheme not in ("http", "https"):
        return None
    if parsed.fragment:
        absolute = absolute.split("#", 1)[0]
    return absolute


def _is_download_candidate(url: str) -> bool:
    lower = url.lower()
    path = urlparse(lower).path
    if any(path.endswith(ext) for ext in (".zip", ".gz", ".json", ".tar", ".csv", ".tsv")):
        return True
    if "data.dws.informatik.uni-mannheim.de" in lower and "wdc-products" in lower:
        return True
    return any(hint in lower for hint in _DOWNLOAD_HINTS)


def _dedupe_preserve_order(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for url in urls:
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def extract_candidate_links(html: str, base_url: str) -> list[str]:
    """Parse HTML and return plausible benchmark download URLs."""
    parser = _LinkExtractor()
    try:
        parser.feed(html)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse HTML from {base_url}: {exc}") from exc

    from_page: list[str] = []
    for href in parser.links:
        normalized = _normalize_link(href, base_url)
        if normalized and _is_download_candidate(normalized):
            from_page.append(normalized)

    # Also pick up bare URLs in page text (some pages embed links outside <a>.
    for match in re.findall(r"https?://[^\s\"'<>]+", html):
        cleaned = match.rstrip(".,);]")
        if _is_download_candidate(cleaned):
            from_page.append(cleaned)

    combined = _dedupe_preserve_order(WDC_EXPLICIT_CANDIDATE_URLS + from_page)
    return combined


def fetch_wdc_products_page(timeout: int = 60) -> str:
    """Download the WDC Products benchmark page HTML."""
    try:
        response = requests.get(WDC_PRODUCTS_PAGE_URL, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not fetch WDC Products page at {WDC_PRODUCTS_PAGE_URL}: {exc}"
        ) from exc

    if not response.text.strip():
        raise RuntimeError(
            f"WDC Products page at {WDC_PRODUCTS_PAGE_URL} returned empty content."
        )
    return response.text


def write_source_notes() -> None:
    """Write reproducibility notes about dataset provenance."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    notes = f"""WDC Products benchmark — source notes
=====================================

Benchmark page: {WDC_PRODUCTS_PAGE_URL}
GitHub repository: {WDC_PRODUCTS_GITHUB_URL}
Download captured: {timestamp}

This benchmark was chosen because product records support symbolic governance
constraints (brand, category, identifiers, numeric attributes, title heuristics)
for constraint-aware entity resolution experiments.

The HTML snapshot of the benchmark page is saved as:
  {PAGE_HTML_PATH.name}

Candidate download URLs were extracted from the page and from verified upstream
references. Select the exact benchmark variant (corner-case %, pairwise vs
multi-class) before downloading archives into data/raw/ or data/interim/.
"""
    NOTES_PATH.write_text(notes, encoding="utf-8")


def discover_wdc_files(root: Path | None = None) -> list[Path]:
    """List data files under the WDC products interim/raw trees."""
    roots = [root] if root is not None else [WDC_INTERIM_PRODUCTS_DIR, DATA_RAW_DIR]
    extensions = {".json.gz", ".json", ".parquet", ".csv", ".tsv"}
    found: list[Path] = []
    for base in roots:
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            name = path.name.lower()
            if any(name.endswith(ext) for ext in extensions) or name.endswith(".zip"):
                found.append(path)
    return found


def extract_wdc_archives(force: bool = False) -> list[Path]:
    """
    Extract .zip archives from data/raw into data/interim/wdc_products/.

    Skips extraction when the target directory already has files unless force=True.
    Returns paths of extracted member files (or existing files if skipped).
    """
    ensure_data_directories()
    WDC_INTERIM_PRODUCTS_DIR.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(DATA_RAW_DIR.glob("*.zip"))
    if not zip_files:
        print("No .zip files found in data/raw — skip archive extraction.")
        return discover_wdc_files()

    existing = [p for p in WDC_INTERIM_PRODUCTS_DIR.rglob("*") if p.is_file()]
    if existing and not force:
        print(
            f"Extraction skipped: {WDC_INTERIM_PRODUCTS_DIR} already contains "
            f"{len(existing)} file(s). Use force=True to re-extract."
        )
        return sorted(existing)

    extracted: list[Path] = []
    for zip_path in zip_files:
        print(f"Extracting {zip_path.name} -> {WDC_INTERIM_PRODUCTS_DIR}")
        try:
            with ZipFile(zip_path, "r") as archive:
                for member in archive.namelist():
                    if member.endswith("/"):
                        continue
                    target = WDC_INTERIM_PRODUCTS_DIR / Path(member).name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as src, target.open("wb") as dst:
                        dst.write(src.read())
                    extracted.append(target)
        except BadZipFile as exc:
            raise RuntimeError(f"Invalid zip archive {zip_path}: {exc}") from exc
        except OSError as exc:
            raise RuntimeError(f"Failed to extract {zip_path}: {exc}") from exc

    print(f"Extracted {len(extracted)} file(s) from {len(zip_files)} archive(s).")
    for path in extracted[:20]:
        print(f"  {path.relative_to(WDC_INTERIM_PRODUCTS_DIR.parent.parent)}")
    if len(extracted) > 20:
        print(f"  ... and {len(extracted) - 20} more")
    return extracted


def download_wdc_sources() -> list[str]:
    """
    Ensure directories exist, fetch the benchmark page, save artifacts,
    and return candidate download URLs.
    """
    ensure_data_directories()

    html = fetch_wdc_products_page()
    PAGE_HTML_PATH.write_text(html, encoding="utf-8")
    write_source_notes()

    candidates = extract_candidate_links(html, WDC_PRODUCTS_PAGE_URL)
    if not candidates:
        print(
            "Warning: no download candidate links were discovered. "
            "Inspect the saved HTML and dataset_source_notes.txt.",
            file=sys.stderr,
        )

    extract_wdc_archives(force=False)
    return candidates


def main() -> int:
    try:
        candidates = download_wdc_sources()
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved page HTML: {PAGE_HTML_PATH}")
    print(f"Saved source notes: {NOTES_PATH}")
    print("\nCandidate download links:")
    if candidates:
        for i, url in enumerate(candidates, start=1):
            print(f"  {i}. {url}")
    else:
        print("  (none found — check HTML snapshot manually)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
