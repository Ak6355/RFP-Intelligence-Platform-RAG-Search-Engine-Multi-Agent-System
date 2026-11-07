"""LangGraph orchestrator for extraction, addendum review, validation and Q&A."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .addendum_agent import reconcile_addendums
from .extraction_agent import FIELD_GROUPS, run_extraction_group
from .ingestion_agent import run_ingestion_job
from .observability import start_timer, trace_event
from .qa_agent import answer_question
from .retrieval_agent import RetrievalAgent
from .state import AgentState
from .validator_agent import validate_fields

load_dotenv()


class LazyHybridSearch:
    """Do not load embedding/reranker models until the first retrieval call."""
    def __init__(self):
        self._engine = None

    def search(self, **kwargs):
        if self._engine is None:
            from search.hybrid_search import HybridSearchEngine
            self._engine = HybridSearchEngine(enable_reranker=True)
        return self._engine.search(**kwargs)

    def close(self):
        if self._engine is not None:
            self._engine.close()


def build_agent_graph(
    *,
    search_engine: Any = None,
    llm: Any = None,
    project_root: str | Path | None = None,
    top_k_evidence: int | None = None,
    candidate_k: int | None = None,
):
    """Build the compiled LangGraph application. Dependencies can be injected for tests."""
    retrieval = RetrievalAgent(
        search_engine or LazyHybridSearch(),
        top_k=top_k_evidence or int(os.getenv("RFP_TOP_K_EVIDENCE", "6")),
        candidate_k=candidate_k or int(os.getenv("RFP_CANDIDATE_K", "20")),
    )
    root = Path(project_root or Path.cwd()).resolve()

    def orchestrator_node(state: AgentState) -> dict[str, Any]:
        started = start_timer()
        mode = str(state.get("mode", "extract")).lower()
        if mode not in {"extract", "qa", "ingest"}:
            raise ValueError("mode must be one of: extract, qa, ingest")
        bid_id = str(state.get("bid_id", "")).strip()
        if not bid_id:
            raise ValueError("bid_id is required")
        plan = {
            "extract": ["Retrieve evidence for specialist field groups in parallel", "Reconcile addendums", "Validate every field and citation", "Retry rejected fields up to the configured maximum", "Write cited JSON and agent trace"],
            "qa": ["Retrieve relevant passages using hybrid search", "Generate a grounded answer", "Attach citations from retrieved passages"],
            "ingest": ["Parse PDF and HTML source documents", "Create chunks", "Incrementally index chunks in Qdrant and Elasticsearch"],
        }[mode]
        event = trace_event("orchestrator_agent", "plan", started, input_summary={"mode": mode, "bid_id": bid_id, "question": state.get("question")}, output_summary={"plan": plan})
        return {"mode": mode, "plan": plan, "retry_count": 0, "max_retries": int(state.get("max_retries", int(os.getenv("RFP_MAX_RETRIES", "2")))), "trace": [event]}

    def route_after_orchestrator(state: AgentState):
        mode = state["mode"]
        if mode == "qa":
            return "qa_agent"
        if mode == "ingest":
            return "ingestion_agent"
        return [
            Send("extraction_worker", {
                "mode": "extract", "bid_id": state["bid_id"], "group": group,
                "retry_fields": [], "retry_feedback": "",
            })
            for group in FIELD_GROUPS
        ]

    def extraction_worker(state: AgentState) -> dict[str, Any]:
        group = state["group"]
        started = start_timer()
        try:
            fields, evidence, event = run_extraction_group(
                group=group,
                bid_id=state["bid_id"],
                retrieval=retrieval,
                llm=llm,
                retry_fields=state.get("retry_fields") or None,
                retry_feedback=state.get("retry_feedback", ""),
            )
            return {"field_results": fields, "evidence_by_group": {group: evidence}, "trace": [event]}
        except Exception as exc:
            from .extraction_agent import FIELD_GROUPS
            requested = [name for name in FIELD_GROUPS[group] if not state.get("retry_fields") or name in state.get("retry_fields", [])]
            fields = {name: {"value": None, "sources": [], "confidence": 0.0, "notes": f"Extraction failed; review agent trace: {type(exc).__name__}"} for name in requested}
            event = trace_event(
                "extraction_agent", group, started,
                input_summary={"bid_id": state.get("bid_id"), "fields": requested},
                output_summary={"fields": requested, "error": str(exc)},
                status="error", error=f"{type(exc).__name__}: {exc}",
            )
            return {"field_results": fields, "evidence_by_group": {group: []}, "trace": [event]}

    def addendum_node(state: AgentState) -> dict[str, Any]:
        started = start_timer()
        try:
            fields, changes, evidence, event = reconcile_addendums(state["bid_id"], state.get("field_results", {}), retrieval, llm=llm)
            return {"field_results": fields, "addendum_changes": changes, "evidence_by_group": {"addendum": evidence}, "trace": [event]}
        except Exception as exc:
            event = trace_event("addendum_reconciliation_agent", "reconcile", started, input_summary={"bid_id": state.get("bid_id")}, output_summary={}, status="error", error=f"{type(exc).__name__}: {exc}")
            return {"addendum_changes": [], "trace": [event]}

    def validator_node(state: AgentState) -> dict[str, Any]:
        started = start_timer()
        try:
            report, event = validate_fields(state.get("field_results", {}), state.get("evidence_by_group", {}), llm=llm)
        except Exception as exc:
            report = {"passed": 0, "failed": len(state.get("field_results", {})), "not_found": 0, "issues": [{"field": "__validator_error__", "reason": f"{type(exc).__name__}: {exc}", "retry": False}], "failed_fields": []}
            event = trace_event("validator_critic_agent", "validate", started, output_summary=report, status="error", error=f"{type(exc).__name__}: {exc}")
        failed_fields = list(report.get("failed_fields", []))
        retry_count = int(state.get("retry_count", 0))
        can_retry = bool(failed_fields) and retry_count < int(state.get("max_retries", 2))
        feedback = "; ".join(f"{issue.get('field')}: {issue.get('reason')}" for issue in report.get("issues", []) if issue.get("retry"))
        return {
            "validation_report": report,
            "failed_fields": failed_fields if can_retry else [],
            "retry_feedback": feedback,
            "retry_count": retry_count + 1 if can_retry else retry_count,
            "trace": [event],
        }

    def route_after_validation(state: AgentState):
        failed = set(state.get("failed_fields", []))
        if failed:
            sends = []
            for group, fields in FIELD_GROUPS.items():
                retry_subset = sorted(failed.intersection(fields))
                if retry_subset:
                    sends.append(Send("extraction_worker", {
                        "mode": "extract", "bid_id": state["bid_id"], "group": group,
                        "retry_fields": retry_subset,
                        "retry_feedback": state.get("retry_feedback", ""),
                    }))
            if sends:
                return sends
        return "report_agent"

    def report_node(state: AgentState) -> dict[str, Any]:
        started = start_timer()
        bid_id = state["bid_id"]
        all_fields = dict(state.get("field_results", {}))
        # Ensure the output contract contains every required field, even when evidence is absent.
        for group_fields in FIELD_GROUPS.values():
            for field_name in group_fields:
                all_fields.setdefault(field_name, {"value": None, "sources": [], "confidence": 0.0, "notes": "Not found in documents"})
        report = state.get("validation_report", {})
        output = {
            "bid_id": bid_id,
            "fields": all_fields,
            "addendum_changes": state.get("addendum_changes", []),
            "validation": {
                "passed": report.get("passed", 0),
                "failed": report.get("failed", 0),
                "not_found": report.get("not_found", sum(1 for x in all_fields.values() if x.get("value") is None)),
                "issues": report.get("issues", []),
                "retry_count": state.get("retry_count", 0),
                "max_retries": state.get("max_retries", 2),
            },
        }
        output_dir = root / "outputs"
        logs_dir = root / "logs"
        output_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{bid_id}.json"
        trace_path = logs_dir / f"agent_trace_{bid_id}.json"
        json_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        trace = list(state.get("trace", []))
        event = trace_event("qa_report_agent", "write_final_json", started, input_summary={"bid_id": bid_id}, output_summary={"json_path": str(json_path), "fields": len(all_fields)})
        trace.append(event)
        trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return {"field_results": all_fields, "final_json_path": str(json_path), "trace": [event]}

    def qa_node(state: AgentState) -> dict[str, Any]:
        started = start_timer()
        try:
            answer, citations, confidence, event = answer_question(state.get("question", ""), state["bid_id"], retrieval, llm=llm)
            root_logs = root / "logs"
            root_logs.mkdir(parents=True, exist_ok=True)
            qa_path = root_logs / "qa_log.jsonl"
            with qa_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"bid_id": state["bid_id"], "question": state.get("question"), "answer": answer, "citations": citations, "confidence": confidence}, ensure_ascii=False) + "\n")
            return {"answer": answer, "answer_sources": citations, "confidence": confidence, "trace": [event]}
        except Exception as exc:
            event = trace_event("qa_report_agent", "answer_question", started, input_summary={"question": state.get("question"), "bid_id": state.get("bid_id")}, output_summary={"answer": "Unable to answer because an agent step failed"}, status="error", error=f"{type(exc).__name__}: {exc}")
            return {"answer": "Unable to answer because an agent step failed. See agent trace for details.", "answer_sources": [], "confidence": 0.0, "trace": [event]}

    def ingestion_node(state: AgentState) -> dict[str, Any]:
        started = start_timer()
        try:
            result, event = run_ingestion_job(state.get("project_root", root), state["bid_id"])
            return {"ingestion_result": result, "trace": [event]}
        except Exception as exc:
            event = trace_event("ingestion_agent", "ingest_and_index", started, input_summary={"bid_id": state.get("bid_id")}, output_summary={}, status="error", error=f"{type(exc).__name__}: {exc}")
            return {"ingestion_result": {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, "trace": [event]}

    builder = StateGraph(AgentState)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("extraction_worker", extraction_worker)
    builder.add_node("addendum_reconciliation", addendum_node)
    builder.add_node("validator", validator_node)
    builder.add_node("report_agent", report_node)
    builder.add_node("qa_agent", qa_node)
    builder.add_node("ingestion_agent", ingestion_node)

    builder.add_edge(START, "orchestrator")
    builder.add_conditional_edges("orchestrator", route_after_orchestrator)
    builder.add_edge("extraction_worker", "addendum_reconciliation")
    builder.add_edge("addendum_reconciliation", "validator")
    builder.add_conditional_edges("validator", route_after_validation)
    builder.add_edge("report_agent", END)
    builder.add_edge("qa_agent", END)
    builder.add_edge("ingestion_agent", END)
    return builder.compile()
