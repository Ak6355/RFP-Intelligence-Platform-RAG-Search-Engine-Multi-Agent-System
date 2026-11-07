"""Retrieval agent: the only path by which LLM agents access bid documents."""
from __future__ import annotations

from typing import Any


class RetrievalAgent:
    def __init__(self, search_engine: Any, top_k: int = 6, candidate_k: int = 20) -> None:
        self.search_engine = search_engine
        self.top_k = top_k
        self.candidate_k = max(candidate_k, top_k)

    def retrieve(
        self,
        query: str,
        bid_id: str,
        *,
        doc_type: str | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return short retrieved passages with citation metadata; never load full PDFs."""
        result = self.search_engine.search(
            query=query,
            bid_id=bid_id,
            doc_type=doc_type,
            top_k=top_k or self.top_k,
            candidate_k=self.candidate_k,
        )
        rows = result.get("results", []) if isinstance(result, dict) else result
        evidence: list[dict[str, Any]] = []
        for index, row in enumerate(rows or []):
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            citation = row.get("citation") or {
                "bid_id": row.get("bid_id", bid_id),
                "file": row.get("file_name"),
                "page": row.get("page_number"),
                "doc_type": row.get("doc_type"),
                "addendum_number": row.get("addendum_number"),
                "chunk_id": row.get("chunk_id"),
            }
            evidence.append({
                "evidence_index": len(evidence),
                "chunk_id": row.get("chunk_id"),
                "text": text[:5000],
                "citation": citation,
                "score": row.get("rerank_score", row.get("rrf_score", row.get("score"))),
            })
        return evidence


def render_evidence(evidence: list[dict[str, Any]]) -> str:
    """Compact evidence representation used in prompts."""
    import json
    return json.dumps(evidence, ensure_ascii=False, indent=2, default=str)
