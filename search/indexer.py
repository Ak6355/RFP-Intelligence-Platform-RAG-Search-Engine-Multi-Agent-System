"""Index chunks into Qdrant (dense vectors) and Elasticsearch (BM25)."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from elasticsearch import Elasticsearch, helpers
from qdrant_client import QdrantClient, models

from .embeddings import DEFAULT_MODEL, EmbeddingModel

LOGGER = logging.getLogger("rfp_indexer")
NAMESPACE = uuid.UUID("a8ed0f80-6be7-5db6-bf55-c530c67d5e22")


def stable_chunk_id(record: dict[str, Any]) -> str:
    """Return a stable ID. Existing Phase 3 chunk_id is preferred."""
    existing = record.get("chunk_id")
    if existing is not None and str(existing).strip():
        identity = str(existing).strip()
    else:
        identity = "|".join(
            str(record.get(key, ""))
            for key in ("bid_id", "file_name", "page_number", "content_type", "chunk_index", "text")
        )
    return str(uuid.uuid5(NAMESPACE, identity))


def _clean_record(record: dict[str, Any]) -> dict[str, Any]:
    text = record.get("text") or record.get("chunk_text") or record.get("content") or ""
    text = str(text).strip()
    if not text:
        raise ValueError("Chunk has no non-empty text field (expected text/chunk_text/content)")

    cleaned = dict(record)
    cleaned["text"] = text
    cleaned["bid_id"] = str(record.get("bid_id") or "unknown")
    cleaned["file_name"] = str(record.get("file_name") or "unknown")
    cleaned["doc_type"] = str(record.get("doc_type") or "unknown")
    cleaned["content_type"] = str(record.get("content_type") or "text")
    cleaned["chunk_id"] = str(record.get("chunk_id") or stable_chunk_id(record))
    cleaned["point_id"] = stable_chunk_id(record)
    return cleaned


def read_jsonl_batches(path: Path, batch_size: int) -> Iterable[list[dict[str, Any]]]:
    """Read a JSONL file in batches so the complete dataset is not held in RAM."""
    batch: list[dict[str, Any]] = []
    line_number = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line_number += 1
            if not raw_line.strip():
                continue
            try:
                parsed = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                LOGGER.warning("Skipping invalid JSON on line %s: %s", line_number, exc)
                continue
            if not isinstance(parsed, dict):
                LOGGER.warning("Skipping non-object JSON on line %s", line_number)
                continue
            try:
                batch.append(_clean_record(parsed))
            except ValueError as exc:
                LOGGER.warning("Skipping line %s: %s", line_number, exc)
                continue
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def ensure_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int) -> bool:
    """Create collection when missing; return True when newly created."""
    if client.collection_exists(collection_name):
        info = client.get_collection(collection_name)
        vectors = info.config.params.vectors
        existing_size = getattr(vectors, "size", None)
        if existing_size is not None and existing_size != vector_size:
            raise RuntimeError(
                f"Qdrant collection '{collection_name}' has vector size {existing_size}, "
                f"but model requires {vector_size}. Use a new collection name or re-create it."
            )
        return False

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )

    # Payload indexes make common bid/document filters more efficient.
    for field in ("bid_id", "doc_type", "file_name", "content_type"):
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    for field in ("page_number", "addendum_number"):
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=models.PayloadSchemaType.INTEGER,
        )
    return True


def ensure_elasticsearch_index(client: Elasticsearch, index_name: str) -> bool:
    if client.indices.exists(index=index_name):
        return False

    mappings = {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "bid_id": {"type": "keyword"},
            "file_name": {"type": "keyword"},
            "doc_type": {"type": "keyword"},
            "file_type": {"type": "keyword"},
            "addendum_number": {"type": "integer"},
            "page_number": {"type": "integer"},
            "content_type": {"type": "keyword"},
            "section_heading": {"type": "text"},
            "text": {"type": "text"},
            "document_date": {"type": "date", "ignore_malformed": True},
            "chunk_index": {"type": "integer"},
            "token_count": {"type": "integer"},
        }
    }
    client.indices.create(index=index_name, mappings=mappings)
    return True


def make_es_document(record: dict[str, Any], model_name: str) -> dict[str, Any]:
    allowed = (
        "chunk_id", "bid_id", "file_name", "doc_type", "file_type", "addendum_number",
        "page_number", "content_type", "section_heading", "text", "document_date",
        "chunk_index", "token_count", "char_count", "doc_id",
    )
    doc = {key: record[key] for key in allowed if key in record and record[key] is not None}
    doc["embedding_model"] = model_name
    return doc


def record_fingerprint(record: dict[str, Any], model_name: str) -> str:
    """Hash text and indexed metadata so unchanged chunks can be skipped."""
    serialized = json.dumps(make_es_document(record, model_name), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def index_chunks(
    input_path: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 16,
    qdrant_url: str = "http://localhost:6333",
    elasticsearch_url: str = "http://localhost:9200",
    collection_name: str = "rfp_chunks",
    es_index: str = "rfp_chunks",
    force_reindex: bool = False,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Chunk file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model = EmbeddingModel(model_name=model_name, batch_size=batch_size)
    qdrant = QdrantClient(url=qdrant_url, timeout=120)
    es = Elasticsearch(elasticsearch_url, request_timeout=120)

    # Fail early with clear messages if either store is not running.
    qdrant.get_collections()
    es.info()
    qdrant_created = ensure_qdrant_collection(qdrant, collection_name, model.dimension)
    es_created = ensure_elasticsearch_index(es, es_index)

    manifest_path = output_dir / "index_manifest.json"
    if force_reindex or qdrant_created or es_created:
        manifest: dict[str, str] = {}
    elif manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                manifest = {}
        except (json.JSONDecodeError, OSError):
            manifest = {}
    else:
        manifest = {}

    total = 0
    skipped_unchanged = 0
    skipped_or_invalid = 0
    bid_counts: dict[str, int] = {}
    file_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()

    for input_batch in read_jsonl_batches(input_path, batch_size):
        batch = []
        fingerprints: dict[str, str] = {}
        for item in input_batch:
            fingerprint = record_fingerprint(item, model_name)
            if not force_reindex and manifest.get(item["chunk_id"]) == fingerprint:
                skipped_unchanged += 1
                continue
            batch.append(item)
            fingerprints[item["chunk_id"]] = fingerprint
        if not batch:
            continue

        texts = [item["text"] for item in batch]
        vectors = model.encode_passages(texts)
        if len(vectors) != len(batch):
            raise RuntimeError("Embedding count does not match the number of chunks")

        points = []
        actions = []
        for record, vector in zip(batch, vectors):
            payload = make_es_document(record, model_name)
            # Store citation metadata and text in Qdrant's payload too.
            qdrant_payload = dict(payload)
            qdrant_payload["text"] = record["text"]
            points.append(
                models.PointStruct(
                    id=record["point_id"],
                    vector=vector.tolist(),
                    payload=qdrant_payload,
                )
            )
            actions.append(
                {
                    "_op_type": "index",
                    "_index": es_index,
                    "_id": record["chunk_id"],
                    "_source": payload,
                }
            )
            bid = record["bid_id"]
            bid_counts[bid] = bid_counts.get(bid, 0) + 1

        qdrant.upsert(collection_name=collection_name, points=points, wait=True)
        success_count, errors = helpers.bulk(es, actions, raise_on_error=False, request_timeout=120)
        if errors:
            LOGGER.error("Elasticsearch reported %s indexing errors", len(errors))
            for error in errors[:5]:
                LOGGER.error("Elasticsearch indexing error: %s", error)
            skipped_or_invalid += len(errors)
        else:
            # Persist fingerprints only after both stores accepted the full batch.
            manifest.update(fingerprints)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        total += len(batch)

    es.indices.refresh(index=es_index)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "input_sha256": file_hash,
        "embedding_model": model_name,
        "embedding_dimension": model.dimension,
        "device": model.device,
        "batch_size": batch_size,
        "qdrant_url": qdrant_url,
        "qdrant_collection": collection_name,
        "elasticsearch_url": elasticsearch_url,
        "elasticsearch_index": es_index,
        "chunks_indexed_this_run": total,
        "chunks_skipped_unchanged": skipped_unchanged,
        "elasticsearch_errors": skipped_or_invalid,
        "manifest_entries": len(manifest),
        "force_reindex": force_reindex,
        "chunks_by_bid": bid_counts,
        "incremental_behavior": "Unchanged chunks are skipped using a local fingerprint manifest; changed chunks are upserted; new chunks are inserted.",
    }
    report_path = output_dir / "indexing_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    qdrant.close()
    es.close()
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index RFP chunks into Qdrant and Elasticsearch")
    parser.add_argument("--input", default="outputs/chunks.jsonl", help="Phase 3 chunks JSONL file")
    parser.add_argument("--output-dir", default="outputs", help="Where indexing_report.json is saved")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("EMBEDDING_BATCH_SIZE", "16")))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--elasticsearch-url", default=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"))
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "rfp_chunks"))
    parser.add_argument("--es-index", default=os.getenv("ELASTICSEARCH_INDEX", "rfp_chunks"))
    parser.add_argument("--force-reindex", action="store_true", help="Re-embed and re-index every chunk")
    return parser


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = build_parser().parse_args()
    try:
        report = index_chunks(
            input_path=Path(args.input),
            output_dir=Path(args.output_dir),
            model_name=args.model,
            batch_size=args.batch_size,
            qdrant_url=args.qdrant_url,
            elasticsearch_url=args.elasticsearch_url,
            collection_name=args.collection,
            es_index=args.es_index,
            force_reindex=args.force_reindex,
        )
    except Exception as exc:
        LOGGER.exception("Indexing failed: %s", exc)
        return 1
    print(json.dumps(report, indent=2))
    print(f"\nIndexing report saved to: {Path(args.output_dir) / 'indexing_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
