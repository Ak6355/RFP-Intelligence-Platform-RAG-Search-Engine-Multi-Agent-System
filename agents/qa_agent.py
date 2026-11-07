"""Citation-grounded free-form bid Q&A agent."""
from __future__ import annotations

from typing import Any

from schemas.agent_models import QADraft
from .llm import invoke_structured
from .observability import start_timer, trace_event
from .retrieval_agent import RetrievalAgent, render_evidence


def answer_question(question: str, bid_id: str, retrieval: RetrievalAgent, llm: Any = None):
    started = start_timer()
    evidence = retrieval.retrieve(question, bid_id)
    if not evidence:
        event = trace_event("qa_report_agent", "answer_question", started, input_summary={"question": question, "bid_id": bid_id}, tool_calls=[{"tool": "hybrid_search", "result_count": 0}], output_summary={"answer": "Not found in documents"}, status="no_evidence")
        return "Not found in documents", [], 0.0, event

    prompt = (
        "You answer questions about procurement bids. Use ONLY the retrieved evidence. "
        "If the answer is not present, say exactly 'Not found in documents'. Do not infer missing terms or dates. "
        "Return evidence_indexes for passages that directly support the answer. If you cannot support the answer, return no indexes."
    )
    parsed, usage = invoke_structured(QADraft, [("system", prompt), ("user", f"Question: {question}\nBid: {bid_id}\nEvidence:\n{render_evidence(evidence)}")], llm=llm)
    answer = parsed.answer if hasattr(parsed, "answer") else parsed.get("answer", "Not found in documents")
    indexes = parsed.evidence_indexes if hasattr(parsed, "evidence_indexes") else parsed.get("evidence_indexes", [])
    confidence = parsed.confidence if hasattr(parsed, "confidence") else parsed.get("confidence", 0.0)
    citations = [evidence[i].get("citation", {}) for i in indexes if isinstance(i, int) and 0 <= i < len(evidence)]
    if not citations or str(answer).strip().casefold() == "not found in documents":
        answer = "Not found in documents"
        citations = []
        confidence = 0.0
    event = trace_event(
        "qa_report_agent", "answer_question", started,
        input_summary={"question": question, "bid_id": bid_id},
        tool_calls=[{"tool": "hybrid_search", "query": question, "filters": {"bid_id": bid_id}, "result_count": len(evidence), "returned_chunks": [item.get("chunk_id") for item in evidence]}],
        output_summary={"answer": answer, "citations": len(citations)}, token_usage=usage,
    )
    return answer, citations, float(confidence or 0.0), event
