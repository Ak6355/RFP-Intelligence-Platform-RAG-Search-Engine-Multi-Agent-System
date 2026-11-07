"""Cross-encoder reranking for retrieved RFP passages."""
from __future__ import annotations

from typing import Any, Sequence

import torch
from sentence_transformers import CrossEncoder

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER, device: str = "auto", batch_size: int = 16) -> None:
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        if not results:
            return []
        pairs = [(query, str(row.get("text", ""))) for row in results]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        reranked = []
        for row, score in zip(results, scores):
            item = dict(row)
            item["rerank_score"] = float(score)
            reranked.append(item)
        return sorted(reranked, key=lambda item: (-item["rerank_score"], -float(item.get("rrf_score", 0.0))))
