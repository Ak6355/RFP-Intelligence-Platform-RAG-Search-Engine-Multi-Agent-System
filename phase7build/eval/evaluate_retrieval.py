"""Compare dense-only, hybrid RRF, and hybrid+cross-encoder retrieval.

Ground truth must be annotated in questions.json before scores are meaningful.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from eval.metrics import aggregate_metrics, annotation_complete, recall_at_k, reciprocal_rank

load_dotenv()


def run_evaluation(questions_path: Path, output_path: Path, top_k: int = 5) -> dict[str, Any]:
    from search.embeddings import EmbeddingModel
    from search.vector_store import QdrantVectorSearch
    from search.hybrid_search import HybridSearchEngine
    from search.reranker import CrossEncoderReranker

    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    ready = [item for item in questions if annotation_complete(item)]
    pending = [item["id"] for item in questions if not annotation_complete(item)]
    if len(ready) < 15:
        raise RuntimeError(
            f"Only {len(ready)}/{len(questions)} evaluation questions have complete expected file/page labels. "
            f"Annotate all 15 rows in {questions_path}. Pending: {', '.join(pending)}"
        )

    embedding = EmbeddingModel(model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"))
    dense = QdrantVectorSearch(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        collection=os.getenv("QDRANT_COLLECTION", "rfp_chunks"),
    )
    hybrid = HybridSearchEngine(
        embedding=embedding,
        dense_search=dense,
        enable_reranker=False,
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        elasticsearch_url=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
        collection=os.getenv("QDRANT_COLLECTION", "rfp_chunks"),
        es_index=os.getenv("ELASTICSEARCH_INDEX", "rfp_chunks"),
    )
    reranker = CrossEncoderReranker(os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

    per_question: list[dict[str, Any]] = []
    methods: dict[str, list[dict[str, float]]] = {
        "dense_only": [], "hybrid_rrf": [], "hybrid_rrf_reranked": []
    }
    try:
        for item in ready:
            filters = dict(item.get("filters") or {})
            filters["bid_id"] = item["bid_id"]
            query = item["question"]
            # Dense-only uses the original query and Qdrant only.
            vector = embedding.encode_query(query).tolist()
            dense_rows = dense.search(vector, top_k=30, **filters)[:top_k]
            # Hybrid RRF; temporarily keep the engine without a cross encoder.
            hybrid_rows = hybrid.search(query=query, top_k=top_k, candidate_k=30, **filters)["results"]
            # Re-rank the same RRF candidate pool with the cross encoder.
            fused = hybrid.search(query=query, top_k=30, candidate_k=30, **filters)["results"]
            reranked_rows = reranker.rerank(query, fused)[:top_k]

            gold = item["expected_passages"]
            method_results = {
                "dense_only": dense_rows,
                "hybrid_rrf": hybrid_rows,
                "hybrid_rrf_reranked": reranked_rows,
            }
            row: dict[str, Any] = {"id": item["id"], "bid_id": item["bid_id"], "question": query, "methods": {}}
            for method, results in method_results.items():
                metric = {
                    "recall_at_5": recall_at_k(results, gold, top_k),
                    "mrr": reciprocal_rank(results, gold),
                }
                methods[method].append(metric)
                row["methods"][method] = {
                    **metric,
                    "returned_sources": [
                        {"file_name": r.get("file_name"), "page_number": r.get("page_number"), "chunk_id": r.get("chunk_id")}
                        for r in results
                    ],
                }
            per_question.append(row)
    finally:
        hybrid.close()

    summary = {name: aggregate_metrics(rows) for name, rows in methods.items()}
    report = {
        "evaluation_questions": len(ready),
        "top_k": top_k,
        "ground_truth": "file-name substring + page number labels in questions.json",
        "summary": summary,
        "per_question": per_question,
        "note": "Scores are valid only if every expected passage label was manually checked against the source files.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="eval/questions.json")
    parser.add_argument("--output", default="outputs/retrieval_evaluation.json")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    report = run_evaluation(Path(args.questions), Path(args.output), args.top_k)
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
