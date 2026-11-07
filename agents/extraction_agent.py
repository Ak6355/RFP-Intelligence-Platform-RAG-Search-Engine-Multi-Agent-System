"""Parallel specialist extraction agents."""
from __future__ import annotations

from typing import Any

from schemas.agent_models import ExtractionDraft, FieldDraft
from .llm import invoke_structured
from .observability import start_timer, trace_event
from .retrieval_agent import RetrievalAgent, render_evidence

FIELD_GROUPS: dict[str, list[str]] = {
    "dates_logistics": [
        "Due Date", "Bid Submission Type", "Term of Bid", "Pre Bid Meeting",
        "Installation", "Delivery Date", "Payment Terms",
    ],
    "commercial_documents": [
        "Bid Bond Requirement", "Any Additional Documentation Required",
        "MFG for Registration", "Contract or Cooperative to use",
    ],
    "product_company": [
        "Bid Number", "Title", "Model_no", "Part_no", "Product",
        "contact_info", "company_name", "Product Specification",
    ],
    "summary": ["Bid Summary"],
}

GROUP_QUERIES: dict[str, str] = {
    "dates_logistics": "Find bid due date and exact submission deadline and time zone, submission method, contract term and renewal, pre-bid meeting, installation, delivery schedule, and payment terms.",
    "commercial_documents": "Find bid bond or bid security requirement and amount, required affidavits and other forms or certificates, manufacturer authorization or registration, and cooperative or state contract vehicle.",
    "product_company": "Find official solicitation number and title, issuing agency, procurement contact name email and phone, products and quantities, manufacturer model numbers, part numbers, and all technical specifications including CPU RAM storage display and warranty.",
    "summary": "Find the purpose, scope, requested products, quantities, key technical requirements, procurement conditions, and important dates for a concise bid summary.",
}


def _source_from_evidence(item: dict[str, Any]) -> dict[str, Any]:
    citation = dict(item.get("citation") or {})
    citation.setdefault("chunk_id", item.get("chunk_id"))
    return citation


def run_extraction_group(
    *,
    group: str,
    bid_id: str,
    retrieval: RetrievalAgent,
    llm: Any = None,
    retry_fields: list[str] | None = None,
    retry_feedback: str = "",
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if group not in FIELD_GROUPS:
        raise ValueError(f"Unknown extraction group: {group}")
    started = start_timer()
    required_fields = FIELD_GROUPS[group]
    requested_fields = [name for name in required_fields if not retry_fields or name in retry_fields]
    query = GROUP_QUERIES[group]
    if retry_fields:
        query += " Focus especially on: " + "; ".join(requested_fields)
        if retry_feedback:
            query += " Validator feedback: " + retry_feedback[:1200]

    evidence = retrieval.retrieve(query, bid_id, top_k=8 if group == "summary" else None)
    if not evidence:
        fields = {
            name: {"value": None, "sources": [], "confidence": 0.0, "notes": "Not found in documents"}
            for name in requested_fields
        }
        event = trace_event("extraction_agent", group, started, input_summary={"bid_id": bid_id, "fields": requested_fields}, tool_calls=[{"tool": "hybrid_search", "query": query, "result_count": 0}], output_summary={"fields": list(fields)}, status="no_evidence")
        return fields, evidence, event

    system_prompt = (
        "You are a specialist RFP information extraction agent. Use ONLY the retrieved evidence below. "
        "Do not use outside knowledge or guess. Extract only the requested field names. "
        "For each value, set evidence_indexes to the zero-based index of one or more evidence items that directly support it. "
        "If evidence does not establish a value, return value=null, evidence_indexes=[], confidence=0, notes='Not found in documents'. "
        "Do not invent dates, time zones, quantities, manufacturer names, contacts, or contract terms. "
        "Keep exact identifiers and preserve time zones as written. If sources conflict, describe the conflict in notes and avoid silently choosing. "
        "For Bid Summary, write 3-6 concise sentences based only on retrieved evidence and cite every material statement with evidence indexes."
    )
    user_prompt = (
        f"Bid ID: {bid_id}\nExtraction group: {group}\nRequested fields: {requested_fields}\n"
        f"Validator feedback: {retry_feedback or 'None'}\n"
        f"Retrieved evidence (the only permitted source):\n{render_evidence(evidence)}"
    )
    parsed, usage = invoke_structured(ExtractionDraft, [("system", system_prompt), ("user", user_prompt)], llm=llm)
    drafts = parsed.fields if hasattr(parsed, "fields") else parsed.get("fields", {})

    mapped: dict[str, dict[str, Any]] = {}
    for field_name in requested_fields:
        draft = drafts.get(field_name)
        if draft is None:
            draft = FieldDraft(value=None, notes="Not found in documents")
        indexes = draft.evidence_indexes if hasattr(draft, "evidence_indexes") else draft.get("evidence_indexes", [])
        value = draft.value if hasattr(draft, "value") else draft.get("value")
        confidence = draft.confidence if hasattr(draft, "confidence") else draft.get("confidence", 0.0)
        notes = draft.notes if hasattr(draft, "notes") else draft.get("notes", "")
        sources = [_source_from_evidence(evidence[i]) for i in indexes if isinstance(i, int) and 0 <= i < len(evidence)]
        # Hard guardrail: a non-null extraction without a retrieved citation is rejected.
        if value is not None and not sources:
            value = None
            confidence = 0.0
            notes = "Not found in documents: no valid supporting evidence citation was returned"
        mapped[field_name] = {
            "value": value,
            "sources": sources,
            "confidence": max(0.0, min(1.0, float(confidence or 0.0))),
            "notes": str(notes or ""),
        }

    event = trace_event(
        "extraction_agent", group, started,
        input_summary={"bid_id": bid_id, "fields": requested_fields, "query": query},
        tool_calls=[{"tool": "hybrid_search", "query": query, "result_count": len(evidence), "filters": {"bid_id": bid_id}, "returned_chunks": [item.get("chunk_id") for item in evidence]}],
        output_summary={"fields": mapped, "cited_fields": sum(bool(v["sources"]) for v in mapped.values())},
        token_usage=usage,
    )
    return mapped, evidence, event
