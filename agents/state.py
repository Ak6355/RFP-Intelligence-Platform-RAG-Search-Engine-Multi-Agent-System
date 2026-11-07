"""Shared LangGraph state and reducers."""
from __future__ import annotations

from typing import Annotated, Any, TypedDict


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


def merge_lists(left: list[Any] | None, right: list[Any] | None) -> list[Any]:
    return list(left or []) + list(right or [])


class AgentState(TypedDict, total=False):
    mode: str
    bid_id: str
    question: str
    project_root: str
    plan: list[str]
    max_retries: int
    retry_count: int
    retry_fields: list[str]
    retry_feedback: str
    field_results: Annotated[dict[str, dict[str, Any]], merge_dicts]
    evidence_by_group: Annotated[dict[str, list[dict[str, Any]]], merge_dicts]
    addendum_changes: list[dict[str, Any]]
    validation_report: dict[str, Any]
    failed_fields: list[str]
    answer: str
    answer_sources: list[dict[str, Any]]
    confidence: float
    ingestion_result: dict[str, Any]
    final_json_path: str
    trace: Annotated[list[dict[str, Any]], merge_lists]
