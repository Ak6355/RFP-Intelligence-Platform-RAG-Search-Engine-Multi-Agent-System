"""Run one query against both indexes to verify Phase 4 indexing."""
from __future__ import annotations

import argparse
import json
import os

from .embeddings import DEFAULT_MODEL, EmbeddingModel
from .keyword_search import ElasticsearchBM25Search
from .vector_store import QdrantVectorSearch


def main() -> int:
    parser = argparse.ArgumentParser(description="Test dense and BM25 search")
    parser.add_argument("--query", required=True)
    parser.add_argument("--bid-id", default=None)
    parser.add_argument("--doc-type", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL))
    args = parser.parse_args()

    model = EmbeddingModel(model_name=args.model)
    qdrant = QdrantVectorSearch()
    es = ElasticsearchBM25Search()
    try:
        query_vector = model.encode_query(args.query).tolist()
        dense = qdrant.search(query_vector, args.top_k, args.bid_id, args.doc_type)
        lexical = es.search(args.query, args.top_k, args.bid_id, args.doc_type)
        print("\n=== DENSE VECTOR RESULTS (QDRANT) ===")
        print(json.dumps(dense, indent=2, ensure_ascii=False))
        print("\n=== BM25 KEYWORD RESULTS (ELASTICSEARCH) ===")
        print(json.dumps(lexical, indent=2, ensure_ascii=False))
        return 0
    finally:
        qdrant.close()
        es.close()


if __name__ == "__main__":
    raise SystemExit(main())
