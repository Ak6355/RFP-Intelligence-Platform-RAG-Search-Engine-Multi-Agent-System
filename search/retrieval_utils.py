"""Dependency-free helpers for query expansion, RRF fusion, and citations."""
from __future__ import annotations

import re
from typing import Any, Iterable

# Conservative domain vocabulary: retain the user's original wording and append
# common alternatives that occur in procurement documents.
QUERY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "deadline": ("submission deadline", "due date", "closing date", "proposal submission date"),
    "due date": ("submission deadline", "closing date", "final date to submit"),
    "submission date": ("submission deadline", "due date", "closing date"),
    "bid bond": ("bid security", "proposal security", "surety bond"),
    "warranty": ("warranty period", "extended warranty", "support coverage"),
    "affidavit": ("sworn statement", "required certification", "affidavit form"),
    "model number": ("model no", "model_no", "product model"),
    "part number": ("part no", "part_no", "sku", "stock keeping unit"),
    "delivery": ("delivery date", "delivery schedule", "shipment"),
    "payment terms": ("payment conditions", "invoice terms", "net payment terms"),
    "pre-bid meeting": ("pre bid meeting", "pre-proposal conference", "bid conference"),
    "installation": ("deployment", "setup", "configuration services"),
}


def expand_query(query: str) -> dict[str, Any]:
    """Return the original query plus conservative domain synonyms.

    This is deterministic and does not call an LLM. The original wording is
    always retained so exact identifiers and phrases are not discarded.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    original = re.sub(r"\s+", " ", query).strip()
    lowered = original.casefold()
    additions: list[str] = []
    matched: list[str] = []
    # Prefer longest matching phrases first. A synonym group is applied once.
    for phrase in sorted(QUERY_SYNONYMS, key=len, reverse=True):
        if phrase in lowered:
            matched.append(phrase)
            for synonym in QUERY_SYNONYMS[phrase]:
                if synonym.casefold() not in lowered and synonym.casefold() not in {x.casefold() for x in additions}:
                    additions.append(synonym)
    expanded = original if not additions else f"{original} {' '.join(additions)}"
    return {"original_query": original, "expanded_query": expanded, "added_terms": additions, "matched_concepts": matched}


def _identity(result: dict[str, Any]) -> str:
    chunk_id = result.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return "|".join(str(result.get(key, "")) for key in ("bid_id", "file_name", "page_number", "text"))


def reciprocal_rank_fusion(
    ranked_lists: Iterable[Iterable[dict[str, Any]]],
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse ranked result lists using RRF: sum(1 / (k + rank))."""
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")
    fused: dict[str, dict[str, Any]] = {}
    for method_index, results in enumerate(ranked_lists):
        method_name = "dense_vector" if method_index == 0 else "bm25_keyword" if method_index == 1 else f"retrieval_{method_index + 1}"
        seen_in_list: set[str] = set()
        for rank, result in enumerate(results, start=1):
            key = _identity(result)
            if key in seen_in_list:
                continue
            seen_in_list.add(key)
            item = fused.setdefault(key, dict(result))
            item.setdefault("retrieval_ranks", {})[method_name] = rank
            item["rrf_score"] = float(item.get("rrf_score", 0.0)) + 1.0 / (rrf_k + rank)
            item.setdefault("retrieval_methods", [])
            if method_name not in item["retrieval_methods"]:
                item["retrieval_methods"].append(method_name)
            # Prefer non-empty values when the same chunk appears in both stores.
            for field, value in result.items():
                if item.get(field) in (None, "") and value not in (None, ""):
                    item[field] = value
    return sorted(fused.values(), key=lambda x: (-float(x.get("rrf_score", 0.0)), _identity(x)))


def make_citation(result: dict[str, Any]) -> dict[str, Any]:
    """Create a traceable citation; page can be null for HTML sources."""
    file_name = result.get("file_name") or "Unknown source"
    page = result.get("page_number")
    bid_id = result.get("bid_id")
    label = f"{file_name}, page {page}" if page is not None else str(file_name)
    if bid_id:
        label = f"{bid_id} — {label}"
    return {
        "bid_id": bid_id,
        "file": file_name,
        "page": page,
        "doc_type": result.get("doc_type"),
        "addendum_number": result.get("addendum_number"),
        "chunk_id": result.get("chunk_id"),
        "label": label,
    }
