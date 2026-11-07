"""Command-line interface for the hybrid RFP search engine."""
from __future__ import annotations
import argparse
import json
import os
from .embeddings import DEFAULT_MODEL
from .hybrid_search import HybridSearchEngine
from .reranker import DEFAULT_RERANKER


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search RFP documents using hybrid retrieval")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--bid-id")
    parser.add_argument("--doc-type")
    parser.add_argument("--file-name")
    parser.add_argument("--addendum-number", type=int)
    parser.add_argument("--content-type", choices=["text", "table"])
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL))
    parser.add_argument("--reranker-model", default=os.getenv("RERANKER_MODEL", DEFAULT_RERANKER))
    parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder; use RRF ranking only")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--elasticsearch-url", default=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "rfp_chunks"))
    parser.add_argument("--es-index", default=os.getenv("ELASTICSEARCH_INDEX", "rfp_chunks"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    engine = None
    try:
        engine = HybridSearchEngine(
            embedding_model=args.model, reranker_model=args.reranker_model,
            qdrant_url=args.qdrant_url, elasticsearch_url=args.elasticsearch_url,
            collection=args.collection, es_index=args.es_index, enable_reranker=not args.no_rerank,
        )
        result = engine.search(
            query=args.query, top_k=args.top_k, candidate_k=args.candidate_k,
            bid_id=args.bid_id, doc_type=args.doc_type, file_name=args.file_name,
            addendum_number=args.addendum_number, content_type=args.content_type, rrf_k=args.rrf_k,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"Hybrid search failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
