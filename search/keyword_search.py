"""Elasticsearch BM25 retrieval with metadata filters."""
from __future__ import annotations
from typing import Any
from elasticsearch import Elasticsearch


class ElasticsearchBM25Search:
    def __init__(self, url: str = "http://localhost:9200", index: str = "rfp_chunks") -> None:
        self.client = Elasticsearch(url, request_timeout=60)
        self.index = index

    def search(
        self, query: str, top_k: int = 5, bid_id: str | None = None,
        doc_type: str | None = None, file_name: str | None = None,
        addendum_number: int | None = None, content_type: str | None = None,
    ) -> list[dict[str, Any]]:
        filters = []
        for field, value in (("bid_id", bid_id), ("doc_type", doc_type), ("file_name", file_name), ("content_type", content_type)):
            if value:
                filters.append({"term": {field: value}})
        if addendum_number is not None:
            filters.append({"term": {"addendum_number": int(addendum_number)}})
        body = {
            "size": top_k,
            "query": {"bool": {
                "must": [{"multi_match": {
                    "query": query,
                    "fields": ["text^3", "section_heading^2", "file_name"],
                    "type": "best_fields",
                }}],
                "filter": filters,
            }},
        }
        response = self.client.search(index=self.index, body=body)
        rows = []
        for hit in response["hits"]["hits"]:
            source = hit.get("_source", {})
            rows.append({
                "chunk_id": source.get("chunk_id"), "bid_id": source.get("bid_id"),
                "file_name": source.get("file_name"), "doc_type": source.get("doc_type"),
                "file_type": source.get("file_type"), "page_number": source.get("page_number"),
                "addendum_number": source.get("addendum_number"), "content_type": source.get("content_type"),
                "section_heading": source.get("section_heading"), "document_date": source.get("document_date"),
                "text": source.get("text", ""), "score": float(hit.get("_score") or 0.0),
                "retrieval_method": "bm25_keyword",
            })
        return rows

    def close(self) -> None:
        self.client.close()
