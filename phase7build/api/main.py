"""RFP Intelligence REST API.

Run from the project root with: uvicorn api.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query

from .schemas import AskRequest, ExtractRequest, IndexRequest
from .services import RFPServices


def create_app(services: Any | None = None, project_root: str | Path | None = None) -> FastAPI:
    app = FastAPI(
        title="RFP Intelligence Platform API",
        version="1.0.0",
        description="Hybrid RAG search and multi-agent extraction for RFP documents.",
    )
    app.state.services = services or RFPServices(project_root=project_root)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "rfp-intelligence-api"}

    @app.post("/index")
    def index_bid(payload: IndexRequest) -> dict[str, Any]:
        try:
            state = app.state.services.invoke_agent("ingest", payload.bid_id)
            result = state.get("ingestion_result", {})
            if result.get("status") == "failed":
                raise HTTPException(status_code=400, detail=result.get("error", "Indexing failed"))
            return {"status": result.get("status", "completed"), "bid_id": payload.bid_id, "result": result}
        except HTTPException:
            raise
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Indexing failed: {type(exc).__name__}: {exc}") from exc

    @app.get("/search")
    def search_bid(
        q: str = Query(min_length=2, max_length=1000),
        bid_id: str | None = Query(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$"),
        doc_type: str | None = None,
        file_name: str | None = None,
        addendum_number: int | None = Query(default=None, ge=1),
        content_type: str | None = None,
        top_k: int = Query(default=5, ge=1, le=50),
        candidate_k: int = Query(default=30, ge=5, le=200),
    ) -> dict[str, Any]:
        if candidate_k < top_k:
            raise HTTPException(status_code=422, detail="candidate_k must be greater than or equal to top_k")
        try:
            return app.state.services.search(
                query=q, top_k=top_k, candidate_k=candidate_k,
                bid_id=bid_id, doc_type=doc_type, file_name=file_name,
                addendum_number=addendum_number, content_type=content_type,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Search failed: {type(exc).__name__}: {exc}") from exc

    @app.post("/ask")
    def ask_bid(payload: AskRequest) -> dict[str, Any]:
        try:
            state = app.state.services.invoke_agent("qa", payload.bid_id, question=payload.question)
            return {
                "bid_id": payload.bid_id,
                "question": payload.question,
                "answer": state.get("answer", "Not found in documents"),
                "citations": state.get("answer_sources", []),
                "confidence": state.get("confidence", 0.0),
            }
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Question answering failed: {type(exc).__name__}: {exc}") from exc

    @app.post("/extract")
    def extract_bid(payload: ExtractRequest) -> dict[str, Any]:
        try:
            state = app.state.services.invoke_agent("extract", payload.bid_id, max_retries=payload.max_retries)
            output_path = state.get("final_json_path")
            output_json = None
            if output_path:
                resolved = Path(output_path).resolve()
                root = Path(getattr(app.state.services, "project_root", project_root or Path.cwd())).resolve()
                if root not in resolved.parents:
                    raise HTTPException(status_code=500, detail="Generated output path is outside the project directory")
                if resolved.exists():
                    import json
                    output_json = json.loads(resolved.read_text(encoding="utf-8"))
            return {
                "status": "completed" if output_path else "finished_without_output_path",
                "bid_id": payload.bid_id,
                "output_path": str(output_path) if output_path else None,
                "validation": state.get("validation_report", {}),
                "result": output_json,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Extraction failed: {type(exc).__name__}: {exc}") from exc

    return app


app = create_app()
