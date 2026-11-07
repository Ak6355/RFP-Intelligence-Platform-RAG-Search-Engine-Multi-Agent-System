"""Conservative text and table cleaning helpers.

The cleaner deliberately preserves headings, bullets, page boundaries and table rows.
It only joins obvious line-wrap breaks and removes repeated page-edge headers/footers.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


def clean_text(text: str | None) -> str:
    """Normalize extracted text while preserving meaningful structure."""
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\ufeff", "")

    # Join words split by a hyphen at the end of a PDF line: "inter-\\nnational".
    text = re.sub(r"(?<=\w)-[ \t]*\n[ \t]*(?=[a-z])", "", text)

    raw_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    output: list[str] = []

    for index, line in enumerate(raw_lines):
        if not line:
            if output and output[-1] != "":
                output.append("")
            continue

        if not output or output[-1] == "":
            output.append(line)
            continue

        previous = output[-1]
        starts_structural = bool(
            re.match(r"^(?:[•●▪◦*-]\s+|\(?\d+[.)]\s+|\d+(?:\.\d+)*\s+)", line)
        )
        looks_like_heading = (
            len(line) <= 100
            and line.isupper()
            and any(character.isalpha() for character in line)
        )
        previous_ends_sentence = bool(re.search(r"[.!?:;]$", previous))
        next_starts_lowercase = bool(re.match(r"^[a-z]", line))

        # Join only obvious wrapped sentences. Keep headings, bullets and list items apart.
        if not starts_structural and not looks_like_heading and not previous_ends_sentence and next_starts_lowercase:
            output[-1] = previous + " " + line
        else:
            output.append(line)

    cleaned = "\n".join(output)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_table(table: list[list[Any]] | None) -> list[list[str]]:
    """Convert a parser table into a clean rectangular-ish list of strings."""
    if not table:
        return []

    rows: list[list[str]] = []
    for row in table:
        if row is None:
            continue
        cleaned_row = [re.sub(r"\s+", " ", str(cell or "")).strip() for cell in row]
        if any(cleaned_row):
            rows.append(cleaned_row)
    return rows


def render_tables_as_text(tables: list[list[list[str]]]) -> str:
    """Render tables into searchable text without discarding their row structure."""
    blocks: list[str] = []
    for table_number, table in enumerate(tables, start=1):
        if not table:
            continue
        rows = [" | ".join(cell for cell in row) for row in table]
        blocks.append(f"[TABLE {table_number}]\n" + "\n".join(rows))
    return "\n\n".join(blocks)


def remove_repeated_headers_footers(pages: list[dict[str, Any]], threshold: float = 0.5) -> list[dict[str, Any]]:
    """Remove lines repeated at the top/bottom edge of multiple PDF pages.

    Only the first two and last two non-empty lines of each page are considered.
    A line must occur on at least two pages and on the configured share of pages.
    """
    if len(pages) < 2:
        return pages

    edge_counter: Counter[str] = Counter()
    for page in pages:
        lines = [line.strip() for line in (page.get("raw_text") or "").splitlines() if line.strip()]
        edge_lines = lines[:2] + lines[-2:]
        # Count a candidate at most once per page.
        for line in set(edge_lines):
            edge_counter[line] += 1

    minimum = max(2, int(len(pages) * threshold + 0.999))
    repeated = {line for line, count in edge_counter.items() if count >= minimum}

    cleaned_pages: list[dict[str, Any]] = []
    for page in pages:
        updated = dict(page)
        lines = (page.get("raw_text") or "").splitlines()
        nonempty_positions = [i for i, line in enumerate(lines) if line.strip()]
        remove_positions: set[int] = set()
        for i in nonempty_positions[:2] + nonempty_positions[-2:]:
            if lines[i].strip() in repeated:
                remove_positions.add(i)
        updated["raw_text"] = "\n".join(line for i, line in enumerate(lines) if i not in remove_positions)
        updated["removed_repeated_edge_lines"] = len(remove_positions)
        cleaned_pages.append(updated)

    return cleaned_pages


def find_explicit_document_date(text: str) -> str | None:
    """Return a date only when it appears next to an issue/publication label.

    This avoids mistaking a submission due date for the document's publication date.
    The original date text is retained rather than guessing a timezone or date format.
    """
    if not text:
        return None

    date_pattern = (
        r"(?:issue\s+date|date\s+issued|issued\s+on|publication\s+date|"
        r"published\s+on|date\s+published|posted\s+on)\s*[:\-]?\s*"
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"\d{4}-\d{2}-\d{2})"
    )
    match = re.search(date_pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else None
