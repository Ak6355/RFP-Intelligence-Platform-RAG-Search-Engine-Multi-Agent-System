"""Validator/Critic agent with citation checks and bounded re-extraction feedback."""
from __future__ import annotations

from typing import Any

from schemas.agent_models import CriticReport
from .llm import invoke_structured
from .observability import start_timer, trace_event
from .retrieval_agent import render_evidence


def validate_fields(fields: dict[str, dict[str, Any]], evidence_by_group: dict[str, list[dict[str, Any]]], llm: Any = None):
    started = start_timer()
    all_evidence = [item for rows in evidence_by_group.values() for item in rows]
    valid_chunk_ids = {str(item.get("chunk_id")) for item in all_evidence if item.get("chunk_id")}
    issues: list[dict[str, Any]] = []
    not_found = 0
    for field_name, item in fields.items():
        value = item.get("value")
        sources = item.get("sources") or []
        if value is None:
            not_found += 1
            continue
        valid_sources = [s for s in sources if s.get("chunk_id") and str(s.get("chunk_id")) in valid_chunk_ids]
        if not valid_sources:
            issues.append({"field": field_name, "reason": "Non-null value has no citation matching retrieved evidence", "retry": True})

    # Semantic critic: verify that values are actually supported by the passages cited.
    semantic_issues: list[dict[str, Any]] = []
    usage: Any = {}
    if fields and all_evidence:
        try:
            compact = []
            for name, item in fields.items():
                if item.get("value") is not None:
                    compact.append({"field": name, "value": item.get("value"), "sources": item.get("sources", []), "confidence": item.get("confidence", 0)})
            if compact:
                schema_prompt = (
                    "You are a strict evidence validator for bid extraction. Review every non-null field and its citation against the retrieved passages. "
                    "Flag a field if the cited passage does not support the value, if the value conflicts with evidence, or if a material detail was invented. "
                    "Do not flag a field just because it is concise. Return only field names from the supplied record."
                )
                parsed, usage = invoke_structured(
                    CriticReport,
                    [("system", schema_prompt), ("user", f"Extracted fields:\n{render_evidence(compact)}\nRetrieved evidence:\n{render_evidence(all_evidence)}")],
                    llm=llm,
                )
                rows = parsed.issues if hasattr(parsed, "issues") else parsed.get("issues", [])
                for row in rows:
                    get = (lambda key, default=None: getattr(row, key, default)) if not isinstance(row, dict) else row.get
                    if get("retry", True):
                        semantic_issues.append({"field": get("field", ""), "reason": get("reason", "Critic rejected evidence support"), "retry": True})
        except Exception as exc:
            # Keep the workflow alive; deterministic citation checks still apply.
            semantic_issues.append({"field": "__critic_error__", "reason": f"Semantic critic unavailable: {type(exc).__name__}: {exc}", "retry": False})

    seen = {issue["field"] for issue in issues}
    for issue in semantic_issues:
        if issue["field"] not in seen and issue["field"] != "__critic_error__":
            issues.append(issue)
            seen.add(issue["field"])

    failed_fields = sorted({item["field"] for item in issues if item.get("retry") and item["field"] in fields})
    passed = sum(1 for name, item in fields.items() if item.get("value") is not None and name not in failed_fields)
    report = {
        "passed": passed,
        "failed": len(failed_fields),
        "not_found": not_found,
        "issues": issues + [i for i in semantic_issues if i.get("field") == "__critic_error__"],
        "failed_fields": failed_fields,
    }
    event = trace_event(
        "validator_critic_agent", "validate", started,
        input_summary={"fields": len(fields), "evidence_passages": len(all_evidence)},
        output_summary={"passed": passed, "failed": len(failed_fields), "not_found": not_found, "failed_fields": failed_fields},
        token_usage=usage,
        status="needs_retry" if failed_fields else "success",
    )
    return report, event
