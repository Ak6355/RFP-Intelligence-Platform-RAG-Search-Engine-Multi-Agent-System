"""HTML bid-page parser using BeautifulSoup."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from .cleaner import normalize_table

logger = logging.getLogger(__name__)


def _first_meta_content(soup: BeautifulSoup, names: tuple[str, ...]) -> str | None:
    for name in names:
        tag = soup.find("meta", attrs={"name": name})
        if tag and tag.get("content"):
            return str(tag.get("content")).strip()
        tag = soup.find("meta", attrs={"property": name})
        if tag and tag.get("content"):
            return str(tag.get("content")).strip()
    return None


def parse_html(path: Path) -> dict[str, Any]:
    """Extract page text, tables, title, publication date (when marked), and URL."""
    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    title = soup.title.get_text(" ", strip=True) if soup.title else None
    if not title:
        first_heading = soup.find(["h1", "h2"])
        title = first_heading.get_text(" ", strip=True) if first_heading else None

    tables: list[list[list[str]]] = []
    for html_table in soup.find_all("table"):
        rows: list[list[str]] = []
        for tr in html_table.find_all("tr"):
            cells = tr.find_all(["th", "td"], recursive=False)
            if not cells:
                cells = tr.find_all(["th", "td"])
            rows.append([cell.get_text(" ", strip=True) for cell in cells])
        normalized = normalize_table(rows)
        if normalized:
            tables.append(normalized)

    # Remove code/style and clearly non-content elements. Keep headers and footers
    # because procurement pages sometimes place bid details there.
    for element in soup.find_all(["script", "style", "noscript", "svg"]):
        element.decompose()

    content_root = soup.find("main") or soup.find("article") or soup.body or soup
    text = content_root.get_text(separator="\n", strip=True)

    canonical = soup.find("link", rel="canonical")
    source_url = None
    if canonical and canonical.get("href"):
        source_url = str(canonical.get("href")).strip()
    if not source_url:
        source_url = _first_meta_content(soup, ("og:url",))

    document_date = _first_meta_content(
        soup,
        ("datePublished", "dateModified", "publishdate", "publication_date", "article:published_time"),
    )
    if not document_date:
        time_tag = soup.find("time", attrs={"datetime": True})
        if time_tag:
            document_date = str(time_tag.get("datetime")).strip()

    return {
        "text": text,
        "tables": tables,
        "title": title,
        "document_date": document_date,
        "source_url": source_url,
    }
