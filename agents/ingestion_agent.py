"""Ingestion agent that invokes existing Phase 2, 3 and 4 pipelines."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .observability import start_timer, trace_event


def run_ingestion_job(project_root: str | Path, bid_id: str, timeout_seconds: int = 7200) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reprocess the data folder, re-chunk it, then incrementally index all current bids.

    The Phase 4 indexer's manifest skips unchanged chunks, so an added bid does not
    require recomputing embeddings for unchanged bids.
    """
    started = start_timer()
    root = Path(project_root).resolve()
    bid_folder = root / "data" / bid_id
    if not bid_folder.is_dir():
        raise FileNotFoundError(f"Bid folder not found: {bid_folder}")

    commands = [
        [sys.executable, "-m", "ingestion.pipeline", "--data-dir", "data", "--output-dir", "outputs"],
        [sys.executable, "-m", "ingestion.chunker", "--input", "outputs/normalized_documents.jsonl", "--output-dir", "outputs"],
        [sys.executable, "-m", "search.indexer", "--input", "outputs/chunks.jsonl", "--output-dir", "outputs"],
    ]
    steps = []
    for command in commands:
        completed = subprocess.run(
            command, cwd=root, capture_output=True, text=True, timeout=timeout_seconds,
            check=False, env=os.environ.copy(),
        )
        step = {
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-5000:],
            "stderr_tail": completed.stderr[-5000:],
        }
        steps.append(step)
        if completed.returncode != 0:
            result = {"status": "failed", "bid_id": bid_id, "failed_command": command, "steps": steps}
            event = trace_event("ingestion_agent", "ingest_and_index", started, input_summary={"bid_id": bid_id, "project_root": str(root)}, tool_calls=[{"tool": "subprocess", "command": s["command"], "returncode": s["returncode"]} for s in steps], output_summary={"status": "failed"}, status="error", error=completed.stderr[-2000:])
            return result, event

    result = {"status": "success", "bid_id": bid_id, "steps": steps, "reports": [
        str(root / "outputs" / "ingestion_report.json"),
        str(root / "outputs" / "chunking_report.json"),
        str(root / "outputs" / "indexing_report.json"),
    ]}
    event = trace_event("ingestion_agent", "ingest_and_index", started, input_summary={"bid_id": bid_id, "project_root": str(root)}, tool_calls=[{"tool": "subprocess", "command": s["command"], "returncode": s["returncode"]} for s in steps], output_summary={"status": "success", "reports": result["reports"]})
    return result, event
