"""Addendum reconciliation agent: apply explicit updates from addendum evidence."""
from __future__ import annotations

from typing import Any

from schemas.agent_models import AddendumReviewDraft
from .llm import invoke_structured
from .observability import start_timer, trace_event
from .retrieval_agent import RetrievalAgent, render_evidence


def reconcile_addendums(bid_id: str, fields: dict[str, dict[str, Any]], retrieval: RetrievalAgent, llm: Any = None):
    started = start_timer()
    query = (
        "Find all explicit amendments, corrections, replacements, extensions, and changes in addendum documents. "
        "Check due date, submission time, delivery, product specifications, warranty, bid bond, meetings, forms, and contract conditions."
    )
    evidence = retrieval.retrieve(query, bid_id, doc_type="addendum", top_k=12)
    if not evidence:
        event = trace_event("addendum_reconciliation_agent", "reconcile", started, input_summary={"bid_id": bid_id}, tool_calls=[{"tool": "hybrid_search", "filters": {"bid_id": bid_id, "doc_type": "addendum"}, "result_count": 0}], output_summary={"changes": 0}, status="no_addendums_found")
        return fields, [], evidence, event

    prompt = (
        "You are an addendum reconciliation agent. Compare the current extracted values with the supplied addendum passages. "
        "Only apply a change when the passage explicitly amends, replaces, clarifies, or extends a requirement. "
        "Do not assume every later addendum changes every field. Use addendum number/date and explicit wording to determine the latest applicable instruction. "
        "For every updated field, provide evidence_indexes pointing to the exact supplied addendum passages. "
        "If a passage is merely background and does not change a field, do not update it. Never guess."
    )
    current_field_rows = [
        {"evidence_index": i, "text": f"{key}: {value.get('value')}", "citation": value.get("sources", [])}
        for i, (key, value) in enumerate(fields.items())
    ]
    user = (
        f"Bid: {bid_id}\nCurrent fields:\n{render_evidence(current_field_rows)}"
        f"\nAddendum evidence:\n{render_evidence(evidence)}"
    )
    parsed, usage = invoke_structured(AddendumReviewDraft, [("system", prompt), ("user", user)], llm=llm)
    updated_fields = parsed.updated_fields if hasattr(parsed, "updated_fields") else parsed.get("updated_fields", {})
    changes_draft = parsed.changes if hasattr(parsed, "changes") else parsed.get("changes", [])

    result_fields = dict(fields)
    changes: list[dict[str, Any]] = []
    for field_name, draft in updated_fields.items():
        indexes = draft.evidence_indexes if hasattr(draft, "evidence_indexes") else draft.get("evidence_indexes", [])
        value = draft.value if hasattr(draft, "value") else draft.get("value")
        sources = [evidence[i].get("citation", {}) for i in indexes if isinstance(i, int) and 0 <= i < len(evidence)]
        if field_name in result_fields and value is not None and sources:
            result_fields[field_name] = {
                "value": value,
                "sources": sources,
                "confidence": float(draft.confidence if hasattr(draft, "confidence") else draft.get("confidence", 0.0)),
                "notes": "Updated based on explicit addendum evidence",
            }

    for item in changes_draft:
        indexes = item.evidence_indexes if hasattr(item, "evidence_indexes") else item.get("evidence_indexes", [])
        sources = [evidence[i].get("citation", {}) for i in indexes if isinstance(i, int) and 0 <= i < len(evidence)]
        if not sources:
            continue
        get = (lambda key, default=None: getattr(item, key, default)) if not isinstance(item, dict) else item.get
        changes.append({
            "field": get("field", ""),
            "previous_value": get("previous_value"),
            "new_value": get("new_value"),
            "reason": get("reason", ""),
            "sources": sources,
        })

    event = trace_event(
        "addendum_reconciliation_agent", "reconcile", started,
        input_summary={"bid_id": bid_id, "current_field_count": len(fields)},
        tool_calls=[{"tool": "hybrid_search", "query": query, "filters": {"bid_id": bid_id, "doc_type": "addendum"}, "result_count": len(evidence), "returned_chunks": [item.get("chunk_id") for item in evidence]}],
        output_summary={"changes": changes, "updated_fields": [k for k, v in result_fields.items() if v != fields.get(k)]},
        token_usage=usage,
    )
    return result_fields, changes, evidence, event
