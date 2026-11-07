"""Lazy adapters around the existing Phase 5 search and Phase 6 LangGraph."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class RFPServices:
    """Load heavyweight models and the graph only when an endpoint needs them."""

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self._graph = None
        self._search_engine = None

    def _get_graph(self):
        if self._graph is None:
            from agents.orchestrator import build_agent_graph
            self._graph = build_agent_graph(project_root=self.project_root)
        return self._graph

    def _get_search_engine(self):
        if self._search_engine is None:
            from search.hybrid_search import HybridSearchEngine
            self._search_engine = HybridSearchEngine(
                embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
                reranker_model=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                elasticsearch_url=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
                collection=os.getenv("QDRANT_COLLECTION", "rfp_chunks"),
                es_index=os.getenv("ELASTICSEARCH_INDEX", "rfp_chunks"),
                enable_reranker=True,
            )
        return self._search_engine

    def search(self, **kwargs: Any) -> dict[str, Any]:
        return self._get_search_engine().search(**kwargs)

    def invoke_agent(self, mode: str, bid_id: str, *, question: str | None = None, max_retries: int = 2) -> dict[str, Any]:
        initial_state: dict[str, Any] = {
            "mode": mode,
            "bid_id": bid_id,
            "project_root": str(self.project_root),
            "trace": [],
        }
        if mode == "qa":
            initial_state["question"] = question or ""
        if mode == "extract":
            initial_state.update({"max_retries": max_retries, "field_results": {}, "evidence_by_group": {}})
        return self._get_graph().invoke(initial_state)

    def close(self) -> None:
        if self._search_engine is not None:
            self._search_engine.close()
            self._search_engine = None
