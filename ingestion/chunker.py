"""Create retrieval-ready chunks from Phase 2 normalized JSONL records.

This module only chunks text and tables. It does NOT generate embeddings or
write to Qdrant/Elasticsearch; those are separate later phases.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

LOGGER = logging.getLogger("rfp_chunker")
DEFAULT_TOKENIZER = "BAAI/bge-large-en-v1.5"
DEFAULT_CHUNK_SIZE = 450
DEFAULT_OVERLAP = 60


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def make_chunk_id(record: dict[str, Any], content_type: str, index: int, text: str) -> str:
    """Create a repeatable ID so re-running chunking does not create new IDs."""
    identity = "|".join(
        str(value or "")
        for value in (
            record.get("bid_id"),
            record.get("file_name"),
            record.get("page_number"),
            content_type,
            index,
            text,
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def is_heading(paragraph: str) -> bool:
    """A conservative heading heuristic; it never changes the source document."""
    value = re.sub(r"\s+", " ", paragraph).strip()
    if not value or len(value) > 120:
        return False
    if re.match(r"^(section\s+)?\d+(?:\.\d+)*(?:[.)])?\s+\S+", value, re.IGNORECASE):
        return True
    if value.endswith(":") and len(value.split()) <= 14:
        return True
    letters = [character for character in value if character.isalpha()]
    if len(letters) >= 4 and sum(character.isupper() for character in letters) / len(letters) >= 0.85:
        return True
    return False


def split_into_sections(text: str) -> list[tuple[str | None, str]]:
    """Split on blank-line paragraphs and retain likely section headings."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", text) if part.strip()]
    if not paragraphs:
        return []

    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal body
        body_text = "\n\n".join(body).strip()
        if body_text:
            sections.append((current_heading, body_text))
        body = []

    for paragraph in paragraphs:
        if is_heading(paragraph):
            flush()
            current_heading = paragraph
        else:
            body.append(paragraph)
    flush()

    # A document made entirely of short headings should still be chunkable.
    if not sections and paragraphs:
        sections.append((None, "\n\n".join(paragraphs)))
    return sections


def chunk_text_sections(
    text: str,
    tokenizer: Any,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict[str, Any]]:
    """Create token-bounded, heading-aware text chunks with sliding overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    output: list[dict[str, Any]] = []
    for heading, body in split_into_sections(text):
        heading_ids = tokenizer.encode(heading, add_special_tokens=False) if heading else []
        # Reserve room for the heading in every chunk of this section.
        if len(heading_ids) >= chunk_size:
            heading_ids = heading_ids[: max(1, chunk_size // 5)]
        body_budget = chunk_size - len(heading_ids)
        if body_budget <= 0:
            body_budget = chunk_size
            heading_ids = []

        body_ids = tokenizer.encode(body, add_special_tokens=False)
        if not body_ids:
            continue

        step = body_budget - min(overlap, max(0, body_budget - 1))
        start = 0
        while start < len(body_ids):
            end = min(start + body_budget, len(body_ids))
            payload_ids = heading_ids + body_ids[start:end]
            chunk_text = tokenizer.decode(payload_ids, skip_special_tokens=True).strip()
            if chunk_text:
                output.append(
                    {
                        "text": chunk_text,
                        "token_count": len(payload_ids),
                        "section_heading": heading,
                    }
                )
            if end >= len(body_ids):
                break
            start += step
    return output


def table_to_rows(table: Any) -> list[list[str]]:
    """Normalize common pdfplumber table shapes into rows of strings."""
    if isinstance(table, dict):
        candidate = table.get("rows") or table.get("data")
        if candidate is not None:
            table = candidate
        else:
            return [[f"{key}: {value}" for key, value in table.items()]]

    if isinstance(table, list):
        rows: list[list[str]] = []
        for row in table:
            if isinstance(row, (list, tuple)):
                cells = [str(cell).strip() if cell is not None else "" for cell in row]
            elif isinstance(row, dict):
                cells = [f"{key}: {value}" for key, value in row.items()]
            else:
                cells = [str(row).strip()]
            if any(cells):
                rows.append(cells)
        return rows

    return [[str(table).strip()]] if table is not None else []


def table_rows_to_chunks(
    table: Any,
    tokenizer: Any,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    """Keep table rows together and repeat the first row as a header on splits."""
    rows = table_to_rows(table)
    if not rows:
        return []

    header = " | ".join(rows[0])
    data_rows = rows[1:] if len(rows) > 1 else []
    groups: list[list[str]] = []
    current: list[str] = []

    # If a table has only one row, store that row as a single table chunk.
    if not data_rows:
        data_rows = []
        current_text = header
        if len(tokenizer.encode(current_text, add_special_tokens=False)) > chunk_size:
            ids = tokenizer.encode(current_text, add_special_tokens=False)
            for start in range(0, len(ids), chunk_size):
                part = tokenizer.decode(ids[start : start + chunk_size], skip_special_tokens=True).strip()
                if part:
                    groups.append([part])
        else:
            groups.append([current_text])
    else:
        for row in data_rows:
            row_text = " | ".join(row)
            candidate = "\n".join(([header] if header else []) + current + [row_text])
            if current and len(tokenizer.encode(candidate, add_special_tokens=False)) > chunk_size:
                groups.append(current)
                current = [row_text]
            else:
                current.append(row_text)
        if current:
            groups.append(current)

    result: list[dict[str, Any]] = []
    for group in groups:
        text = "\n".join(([header] if header and group != [header] else []) + group).strip()
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= chunk_size:
            result.append({"text": text, "token_count": len(ids), "section_heading": None})
        else:
            # A single unusually large row cannot be split by columns safely;
            # split its token sequence while keeping the header in each segment.
            header_ids = tokenizer.encode(header, add_special_tokens=False) if header and group != [header] else []
            body_text = "\n".join(group)
            body_ids = tokenizer.encode(body_text, add_special_tokens=False)
            budget = max(1, chunk_size - len(header_ids))
            start = 0
            while start < len(body_ids):
                end = min(start + budget, len(body_ids))
                chunk_ids = header_ids + body_ids[start:end]
                result.append(
                    {
                        "text": tokenizer.decode(chunk_ids, skip_special_tokens=True).strip(),
                        "token_count": len(chunk_ids),
                        "section_heading": None,
                    }
                )
                start = end
    return result


def chunk_record(
    record: dict[str, Any],
    tokenizer: Any,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict[str, Any]]:
    """Convert one Phase 2 page/HTML record into chunk records."""
    base_metadata = {
        key: record.get(key)
        for key in (
            "doc_id",
            "bid_id",
            "file_name",
            "file_type",
            "doc_type",
            "addendum_number",
            "page_number",
            "document_date",
        )
    }

    output: list[dict[str, Any]] = []
    local_index = 0

    source_text = str(record.get("text") or "").strip()
    if source_text:
        text_chunks = chunk_text_sections(source_text, tokenizer, chunk_size, overlap)
        for item in text_chunks:
            chunk_text = item["text"]
            output.append(
                {
                    "chunk_id": make_chunk_id(record, "text", local_index, chunk_text),
                    **base_metadata,
                    "content_type": "text",
                    "chunk_index": local_index,
                    "section_heading": item.get("section_heading"),
                    "text": chunk_text,
                    "token_count": item["token_count"],
                    "char_count": len(chunk_text),
                }
            )
            local_index += 1

    tables = record.get("tables") or []
    if isinstance(tables, dict):
        tables = [tables]
    for table_index, table in enumerate(tables):
        table_chunks = table_rows_to_chunks(table, tokenizer, chunk_size)
        for item in table_chunks:
            chunk_text = item["text"]
            output.append(
                {
                    "chunk_id": make_chunk_id(record, "table", local_index, chunk_text),
                    **base_metadata,
                    "content_type": "table",
                    "table_index": table_index,
                    "chunk_index": local_index,
                    "section_heading": None,
                    "text": chunk_text,
                    "token_count": item["token_count"],
                    "char_count": len(chunk_text),
                }
            )
            local_index += 1
    return output


def load_tokenizer(model_name: str) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers is not installed. Run: python -m pip install transformers"
        ) from exc
    LOGGER.info("Loading tokenizer: %s", model_name)
    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping invalid JSON at line %s: %s", line_number, exc)
                continue
            if isinstance(record, dict):
                yield record
            else:
                LOGGER.warning("Skipping non-object JSON at line %s", line_number)


def run_chunking(
    input_path: Path,
    output_dir: Path,
    model_name: str = DEFAULT_TOKENIZER,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(model_name)
    chunks_path = output_dir / "chunks.jsonl"
    report_path = output_dir / "chunking_report.json"

    totals: Counter[str] = Counter()
    by_bid: Counter[str] = Counter()
    by_doc_type: Counter[str] = Counter()
    by_content_type: Counter[str] = Counter()
    records_without_chunks = 0
    max_observed_tokens = 0

    with chunks_path.open("w", encoding="utf-8") as output_file:
        for record in read_jsonl(input_path):
            totals["input_records"] += 1
            record_chunks = chunk_record(record, tokenizer, chunk_size, overlap)
            if not record_chunks:
                records_without_chunks += 1
                totals["records_without_chunks"] += 1
                continue

            for chunk in record_chunks:
                output_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                totals["chunks"] += 1
                by_bid[str(chunk.get("bid_id") or "unknown")] += 1
                by_doc_type[str(chunk.get("doc_type") or "unknown")] += 1
                by_content_type[str(chunk.get("content_type") or "unknown")] += 1
                max_observed_tokens = max(max_observed_tokens, int(chunk.get("token_count") or 0))

    report = {
        "input_file": str(input_path),
        "output_file": str(chunks_path),
        "tokenizer": model_name,
        "chunk_size_tokens": chunk_size,
        "overlap_tokens": overlap,
        "input_records": totals["input_records"],
        "total_chunks": totals["chunks"],
        "records_without_chunks": records_without_chunks,
        "max_observed_chunk_tokens": max_observed_tokens,
        "chunks_by_bid": dict(by_bid),
        "chunks_by_doc_type": dict(by_doc_type),
        "chunks_by_content_type": dict(by_content_type),
        "notes": [
            "PDF page-level source metadata is preserved on each chunk.",
            "Text and extracted tables are emitted as separate content types.",
            "This phase does not generate embeddings or write to a vector/keyword database.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Chunking complete: %s chunks written to %s", totals["chunks"], chunks_path)
    LOGGER.info("Chunking report written to %s", report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk Phase 2 normalized RFP documents.")
    parser.add_argument("--input", default="outputs/normalized_documents.jsonl")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    args = parser.parse_args()

    configure_logging()
    report = run_chunking(
        Path(args.input),
        Path(args.output_dir),
        model_name=args.tokenizer,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
