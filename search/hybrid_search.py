"""Hybrid RFP retrieval: query expansion, Qdrant, BM25, RRF, reranking, citations."""
from __future__ import annotations

from typing import Any

from .embeddings import DEFAULT_MODEL, EmbeddingModel
from .keyword_search import ElasticsearchBM25Search
from .reranker import CrossEncoderReranker, DEFAULT_RERANKER
from .retrieval_utils import expand_query, make_citation, reciprocal_rank_fusion
from .vector_store import QdrantVectorSearch


class HybridSearchEngine:
    def __init__(
        self,
        embedding_model: str = DEFAULT_MODEL,
        reranker_model: str = DEFAULT_RERANKER,
        qdrant_url: str = "http://localhost:6333",
        elasticsearch_url: str = "http://localhost:9200",
        collection: str = "rfp_chunks",
        es_index: str = "rfp_chunks",
        enable_reranker: bool = True,
        embedding: Any | None = None,
        dense_search: Any | None = None,
        keyword_search: Any | None = None,
        reranker: Any | None = None,
    ) -> None:
        self.embedding = embedding or EmbeddingModel(model_name=embedding_model)
        self.dense = dense_search or QdrantVectorSearch(url=qdrant_url, collection=collection)
        self.keyword = keyword_search or ElasticsearchBM25Search(url=elasticsearch_url, index=es_index)
        self.reranker = reranker if reranker is not None else (CrossEncoderReranker(reranker_model) if enable_reranker else None)

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 30,
        bid_id: str | None = None,
        doc_type: str | None = None,
        file_name: str | None = None,
        addendum_number: int | None = None,
        content_type: str | None = None,
        rrf_k: int = 60,
    ) -> dict[str, Any]:
        if top_k < 1 or candidate_k < top_k:
            raise ValueError("Require top_k >= 1 and candidate_k >= top_k")
        expansion = expand_query(query)
        expanded_query = expansion["expanded_query"]
        query_vector = self.embedding.encode_query(expanded_query).tolist()
        filters = {
            "bid_id": bid_id, "doc_type": doc_type, "file_name": file_name,
            "addendum_number": addendum_number, "content_type": content_type,
        }
        dense_results = self.dense.search(query_vector, top_k=candidate_k, **filters)
        keyword_results = self.keyword.search(expanded_query, top_k=candidate_k, **filters)
        fused = reciprocal_rank_fusion([dense_results, keyword_results], rrf_k=rrf_k)
        candidates = fused[:candidate_k]
        if self.reranker is not None and candidates:
            final_results = self.reranker.rerank(query, candidates)[:top_k]
            ranking_method = "rrf_then_cross_encoder"
        else:
            final_results = candidates[:top_k]
            ranking_method = "rrf_only"
        output_results = []
        for result in final_results:
            item = dict(result)
            item["citation"] = make_citation(item)
            output_results.append(item)
        return {
            "query": query,
            "expanded_query": expanded_query,
            "query_expansion": expansion,
            "filters": {key: value for key, value in filters.items() if value is not None},
            "retrieval_counts": {"dense": len(dense_results), "bm25": len(keyword_results), "fused_unique": len(fused)},
            "ranking_method": ranking_method,
            "results": output_results,
        }

    def close(self) -> None:
        for client in (self.dense, self.keyword):
            close = getattr(client, "close", None)
            if callable(close):
                close()
