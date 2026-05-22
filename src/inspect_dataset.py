"""Inspect files under data/raw and data/interim."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.config import DATA_INTERIM_DIR, DATA_RAW_DIR, PROJECT_ROOT

console = Console()

_PREVIEW_ROWS = 3
_SCAN_DIRS = (DATA_RAW_DIR, DATA_INTERIM_DIR)


def _format_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024**3:
        return f"{num_bytes / 1024**2:.1f} MB"
    return f"{num_bytes / 1024**3:.1f} GB"


def _rel_data_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _iter_scan_files() -> list[Path]:
    files: list[Path] = []
    for root in _SCAN_DIRS:
        if not root.exists():
            continue
        files.extend(p for p in sorted(root.rglob("*")) if p.is_file())
    return files


def _is_tabular(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".json.gz") or name.endswith(".csv.gz") or name.endswith(".tsv.gz"):
        return True
    return path.suffix.lower() in {".csv", ".tsv", ".json", ".parquet", ".jsonl"}


def _read_tabular_preview(path: Path) -> pd.DataFrame | None:
    name = path.name.lower()
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path).head(_PREVIEW_ROWS)
        if name.endswith(".json.gz") or path.suffix.lower() == ".jsonl":
            return pd.read_json(path, lines=True, nrows=_PREVIEW_ROWS)
        if path.suffix.lower() == ".json":
            return pd.read_json(path).head(_PREVIEW_ROWS)
        if path.suffix.lower() == ".tsv" or name.endswith(".tsv.gz"):
            return pd.read_csv(path, sep="\t", nrows=_PREVIEW_ROWS)
        if path.suffix.lower() == ".csv" or name.endswith(".csv.gz"):
            return pd.read_csv(path, nrows=_PREVIEW_ROWS)
    except Exception as exc:
        console.print(f"  [yellow]Could not preview {path.name}: {exc}[/yellow]")
        return None
    return None


def inspect_local_data() -> None:
    """List local data files and preview tabular formats."""
    files = _iter_scan_files()

    table = Table(title="Files in data/raw and data/interim")
    table.add_column("Path", style="cyan")
    table.add_column("Size", justify="right")

    if not files:
        console.print("[yellow]No files found under data/raw or data/interim.[/yellow]")
        console.print("Run: python scripts/00_download_data.py")
        return

    for path in files:
        table.add_row(_rel_data_path(path), _format_size(path.stat().st_size))
    console.print(table)

    for path in files:
        if not _is_tabular(path):
            continue

        console.print(f"\n[bold]Preview: {_rel_data_path(path)}[/bold]")
        preview = _read_tabular_preview(path)
        if preview is None:
            continue
        console.print(f"  shape (preview): {preview.shape}")
        console.print(f"  columns: {list(preview.columns)}")
        preview_text = preview.head(_PREVIEW_ROWS).to_string(index=False)
        try:
            console.print(preview_text)
        except UnicodeEncodeError:
            console.print(preview_text.encode("ascii", errors="replace").decode("ascii"))


def main() -> int:
    inspect_local_data()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
