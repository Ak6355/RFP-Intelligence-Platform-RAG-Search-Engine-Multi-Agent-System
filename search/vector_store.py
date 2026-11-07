"""Qdrant dense-vector search with bid/document/addendum filters."""
from __future__ import annotations

from typing import Any
from qdrant_client import QdrantClient, models


def build_filter(
    bid_id: str | None = None,
    doc_type: str | None = None,
    file_name: str | None = None,
    addendum_number: int | None = None,
    content_type: str | None = None,
) -> models.Filter | None:
    conditions = []
    for field, value in (("bid_id", bid_id), ("doc_type", doc_type), ("file_name", file_name), ("content_type", content_type)):
        if value:
            conditions.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))
    if addendum_number is not None:
        conditions.append(models.FieldCondition(key="addendum_number", match=models.MatchValue(value=int(addendum_number))))
    return models.Filter(must=conditions) if conditions else None


class QdrantVectorSearch:
    def __init__(self, url: str = "http://localhost:6333", collection: str = "rfp_chunks") -> None:
        self.client = QdrantClient(url=url, timeout=60)
        self.collection = collection

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        bid_id: str | None = None,
        doc_type: str | None = None,
        file_name: str | None = None,
        addendum_number: int | None = None,
        content_type: str | None = None,
    ) -> list[dict[str, Any]]:
        result = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=build_filter(bid_id, doc_type, file_name, addendum_number, content_type),
            limit=top_k,
            with_payload=True,
        )
        rows = []
        for point in result.points:
            payload = point.payload or {}
            rows.append({
                "chunk_id": payload.get("chunk_id"), "bid_id": payload.get("bid_id"),
                "file_name": payload.get("file_name"), "doc_type": payload.get("doc_type"),
                "file_type": payload.get("file_type"), "page_number": payload.get("page_number"),
                "addendum_number": payload.get("addendum_number"), "content_type": payload.get("content_type"),
                "section_heading": payload.get("section_heading"), "document_date": payload.get("document_date"),
                "text": payload.get("text", ""), "score": float(point.score), "retrieval_method": "dense_vector",
            })
        return rows

    def close(self) -> None:
        self.client.close()
