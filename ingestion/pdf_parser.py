"""PDF text and table extraction using PyMuPDF and pdfplumber."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pdfplumber

from .cleaner import normalize_table

logger = logging.getLogger(__name__)


def parse_pdf(path: Path, ocr_text_threshold: int = 50) -> dict[str, Any]:
    """Extract one record per PDF page, preserving page number and tables.

    OCR is not performed in this phase. Pages with very little extractable text are
    flagged with needs_ocr=True so they can be handled in a later OCR/bonus phase.
    """
    pages: list[dict[str, Any]] = []
    pdf_metadata: dict[str, Any] = {}

    # PyMuPDF is used for page-level text extraction and page/image diagnostics.
    with fitz.open(path) as pdf_document:
        pdf_metadata = pdf_document.metadata or {}
        for page_number, page in enumerate(pdf_document, start=1):
            raw_text = page.get_text("text", sort=True) or ""
            compact_text_length = len("".join(raw_text.split()))
            try:
                image_count = len(page.get_images(full=True))
            except Exception:
                image_count = 0

            needs_ocr = compact_text_length < ocr_text_threshold
            if compact_text_length == 0 and image_count == 0:
                status = "empty"
            elif needs_ocr:
                status = "needs_ocr_or_review"
            else:
                status = "ok"

            pages.append(
                {
                    "page_number": page_number,
                    "raw_text": raw_text,
                    "tables": [],
                    "image_count": image_count,
                    "char_count": compact_text_length,
                    "needs_ocr": needs_ocr,
                    "status": status,
                    "removed_repeated_edge_lines": 0,
                }
            )

    # pdfplumber is used separately for table extraction. If table extraction fails,
    # keep the PDF text and log the table-extraction problem instead of losing the file.
    try:
        with pdfplumber.open(path) as pdf_document:
            for index, page in enumerate(pdf_document.pages):
                if index >= len(pages):
                    break
                try:
                    raw_tables = page.extract_tables() or []
                    tables = [normalize_table(table) for table in raw_tables]
                    pages[index]["tables"] = [table for table in tables if table]
                except Exception as exc:  # A problematic table should not stop other pages.
                    logger.warning(
                        "Table extraction failed for %s page %s: %s",
                        path.name,
                        index + 1,
                        exc,
                    )
    except Exception as exc:
        logger.warning("Could not open %s with pdfplumber for tables: %s", path.name, exc)

    return {"pages": pages, "pdf_metadata": pdf_metadata}
