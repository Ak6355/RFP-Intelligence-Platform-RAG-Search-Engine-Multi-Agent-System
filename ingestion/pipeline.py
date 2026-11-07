"""Run the Phase 2 ingestion pipeline over data/<BidFolder>/.

Usage from the project root:
    python -m ingestion.pipeline --data-dir data --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cleaner import clean_text, find_explicit_document_date, remove_repeated_headers_footers, render_tables_as_text
from .file_detector import (
    detect_file_type,
    discover_source_files,
    extract_addendum_number,
    get_bid_id,
    infer_doc_type,
)
from .html_parser import parse_html
from .pdf_parser import parse_pdf


def configure_logging(project_root: Path) -> None:
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "ingestion.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def make_doc_id(relative_path: str, page_number: int | None) -> str:
    raw = f"{relative_path}::{page_number if page_number is not None else 'html'}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def add_tables_to_text(text: str, tables: list[list[list[str]]]) -> str:
    table_text = render_tables_as_text(tables)
    if not table_text:
        return text
    if text.strip():
        return f"{text.strip()}\n\n{table_text}"
    return table_text


def process_pdf(path: Path, data_dir: Path, project_root: Path) -> list[dict[str, Any]]:
    parsed = parse_pdf(path)
    pages = remove_repeated_headers_footers(parsed["pages"])

    file_type = "pdf"
    bid_id = get_bid_id(path, data_dir)
    doc_type = infer_doc_type(path, file_type)
    addendum_number = extract_addendum_number(path)
    relative_path = path.relative_to(project_root).as_posix()
    records: list[dict[str, Any]] = []

    for page in pages:
        page_number = page["page_number"]
        cleaned_page_text = clean_text(page.get("raw_text", ""))
        tables = page.get("tables", [])
        combined_text = clean_text(add_tables_to_text(cleaned_page_text, tables))
        document_date = find_explicit_document_date(cleaned_page_text)

        records.append(
            {
                "doc_id": make_doc_id(relative_path, page_number),
                "bid_id": bid_id,
                "file_name": path.name,
                "relative_path": relative_path,
                "file_type": file_type,
                "doc_type": doc_type,
                "addendum_number": addendum_number,
                "page_number": page_number,
                "document_date": document_date,
                "document_title": parsed.get("pdf_metadata", {}).get("title") or None,
                "text": combined_text,
                "tables": tables,
                "status": page.get("status", "unknown"),
                "needs_ocr": bool(page.get("needs_ocr", False)),
                "image_count": page.get("image_count", 0),
                "char_count": len("".join(combined_text.split())),
                "removed_repeated_edge_lines": page.get("removed_repeated_edge_lines", 0),
                "pdf_metadata": parsed.get("pdf_metadata", {}),
            }
        )
    return records


def process_html(path: Path, data_dir: Path, project_root: Path) -> list[dict[str, Any]]:
    parsed = parse_html(path)
    bid_id = get_bid_id(path, data_dir)
    relative_path = path.relative_to(project_root).as_posix()
    tables = parsed.get("tables", [])
    text = clean_text(add_tables_to_text(parsed.get("text", ""), tables))

    return [
        {
            "doc_id": make_doc_id(relative_path, None),
            "bid_id": bid_id,
            "file_name": path.name,
            "relative_path": relative_path,
            "file_type": "html",
            "doc_type": "bid_page",
            "addendum_number": None,
            # HTML pages are not paginated documents, so page_number is null.
            "page_number": None,
            "document_date": parsed.get("document_date") or find_explicit_document_date(text),
            "document_title": parsed.get("title"),
            "source_url": parsed.get("source_url"),
            "text": text,
            "tables": tables,
            "status": "ok" if text else "empty",
            "needs_ocr": False,
            "image_count": 0,
            "char_count": len("".join(text.split())),
            "removed_repeated_edge_lines": 0,
            "pdf_metadata": {},
        }
    ]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_ingestion(data_dir: Path, output_dir: Path, project_root: Path) -> dict[str, Any]:
    logger = logging.getLogger(__name__)
    data_dir = data_dir.resolve()
    output_dir = output_dir.resolve()
    project_root = project_root.resolve()

    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    supported_files, unsupported_files = discover_source_files(data_dir)
    if not supported_files:
        raise RuntimeError(
            f"No PDF or HTML files found under {data_dir}. Put source files inside data/Bid1 and data/Bid2."
        )

    all_records: list[dict[str, Any]] = []
    failed_files: list[dict[str, str]] = []
    successful_files = 0

    for path in supported_files:
        logger.info("Processing: %s", path)
        try:
            file_type = detect_file_type(path)
            if file_type == "pdf":
                records = process_pdf(path, data_dir, project_root)
            elif file_type == "html":
                records = process_html(path, data_dir, project_root)
            else:
                continue
            all_records.extend(records)
            successful_files += 1
            logger.info("Completed %s -> %s record(s)", path.name, len(records))
        except Exception as exc:
            logger.exception("Failed to process file: %s", path)
            failed_files.append({"file": str(path.relative_to(project_root)), "error": str(exc)})

    normalized_path = output_dir / "normalized_documents.jsonl"
    report_path = output_dir / "ingestion_report.json"
    write_jsonl(normalized_path, all_records)

    records_by_bid: dict[str, int] = {}
    for record in all_records:
        records_by_bid[record["bid_id"]] = records_by_bid.get(record["bid_id"], 0) + 1

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_directory": str(data_dir),
        "total_supported_files": len(supported_files),
        "successfully_processed_files": successful_files,
        "failed_files_count": len(failed_files),
        "unsupported_files_count": len(unsupported_files),
        "total_normalized_records": len(all_records),
        "records_by_bid": records_by_bid,
        "records_needing_ocr_or_review": sum(1 for item in all_records if item.get("needs_ocr")),
        "empty_records": sum(1 for item in all_records if item.get("status") == "empty"),
        "failed_files": failed_files,
        "unsupported_files": [str(path.relative_to(project_root)) for path in unsupported_files],
        "output_jsonl": str(normalized_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Ingestion finished. Records: %s | Failed files: %s", len(all_records), len(failed_files))
    logger.info("Normalized output: %s", normalized_path)
    logger.info("Ingestion report: %s", report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 PDF/HTML ingestion.")
    parser.add_argument("--data-dir", default="data", help="Folder containing one subfolder per bid.")
    parser.add_argument("--output-dir", default="outputs", help="Folder for normalized output and report.")
    args = parser.parse_args()

    project_root = Path.cwd()
    configure_logging(project_root)

    report = run_ingestion(
        data_dir=project_root / args.data_dir,
        output_dir=project_root / args.output_dir,
        project_root=project_root,
    )
    print("\nPHASE 2 INGESTION SUMMARY")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
