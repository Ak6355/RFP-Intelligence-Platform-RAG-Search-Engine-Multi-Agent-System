"""Passage-level retrieval metrics: Recall@k and MRR."""
from __future__ import annotations
from typing import Any, Iterable


def _matches(result: dict[str, Any], expected: dict[str, Any]) -> bool:
    file_hint = str(expected.get("file_contains", "")).casefold()
    actual_file = str(result.get("file_name", "")).casefold()
    if not file_hint or file_hint not in actual_file:
        return False
    expected_page = expected.get("page_number")
    if expected_page == "FILL_PAGE":
        return False
    actual_page = result.get("page_number")
    if expected_page is None:
        return actual_page is None
    try:
        return int(actual_page) == int(expected_page)
    except (TypeError, ValueError):
        return False


def recall_at_k(results: Iterable[dict[str, Any]], expected_passages: list[dict[str, Any]], k: int) -> float:
    expected = [x for x in expected_passages if x.get("page_number") != "FILL_PAGE" and x.get("file_contains")]
    if not expected:
        raise ValueError("No fully annotated expected passages")
    ranked = list(results)
    found = sum(1 for gold in expected if any(_matches(row, gold) for row in ranked[:k]))
    return found / len(expected)


def reciprocal_rank(results: Iterable[dict[str, Any]], expected_passages: list[dict[str, Any]]) -> float:
    expected = [x for x in expected_passages if x.get("page_number") != "FILL_PAGE" and x.get("file_contains")]
    if not expected:
        raise ValueError("No fully annotated expected passages")
    for rank, result in enumerate(results, start=1):
        if any(_matches(result, gold) for gold in expected):
            return 1.0 / rank
    return 0.0


def annotation_complete(item: dict[str, Any]) -> bool:
    passages = item.get("expected_passages") or []
    return bool(passages) and all(
        bool(p.get("file_contains")) and p.get("page_number") != "FILL_PAGE"
        for p in passages
    )


def aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        raise ValueError("No evaluation rows with complete ground truth")
    return {
        "Recall@5": sum(r["recall_at_5"] for r in rows) / len(rows),
        "MRR": sum(r["mrr"] for r in rows) / len(rows),
    }
