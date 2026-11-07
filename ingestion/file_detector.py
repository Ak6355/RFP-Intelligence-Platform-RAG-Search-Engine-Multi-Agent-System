"""File discovery and metadata helpers used by the ingestion pipeline."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm"}


def detect_file_type(path: Path) -> str | None:
    """Return 'pdf' or 'html' for supported files, otherwise None."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".html", ".htm"}:
        return "html"
    return None


def infer_doc_type(path: Path, file_type: str) -> str:
    """Infer a document category from its filename without hard-coding bid names."""
    name = path.stem.lower().replace("_", " ").replace("-", " ")

    if file_type == "html":
        return "bid_page"
    if re.search(r"\baddendum\b", name):
        return "addendum"
    if "affidavit" in name:
        return "affidavit"
    if any(word in name for word in ("spec", "specification", "technical")):
        return "specs"
    if any(word in name for word in ("rfp", "porfp", "solicitation", "final", "request for proposal")):
        return "rfp"
    return "other"


def extract_addendum_number(path: Path) -> int | None:
    """Extract the addendum number from names such as 'Addendum 2.pdf'."""
    match = re.search(r"\baddendum\s*[-_# ]*([0-9]+)\b", path.stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def discover_source_files(data_dir: Path) -> tuple[list[Path], list[Path]]:
    """Return supported source files and unsupported files under data_dir."""
    supported: list[Path] = []
    unsupported: list[Path] = []

    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("~$"):
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            supported.append(path)
        else:
            unsupported.append(path)

    return supported, unsupported


def get_bid_id(path: Path, data_dir: Path) -> str:
    """Use the first folder below data_dir as the bid identifier."""
    try:
        relative = path.relative_to(data_dir)
        if len(relative.parts) >= 2:
            return relative.parts[0]
    except ValueError:
        pass
    return path.parent.name
