"""Small structured trace helpers: inputs, retrieval tool calls, outputs, tokens, latency."""
from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from typing import Any


def start_timer() -> float:
    return perf_counter()


def trace_event(
    agent: str,
    action: str,
    started: float,
    *,
    input_summary: Any = None,
    tool_calls: list[dict[str, Any]] | None = None,
    output_summary: Any = None,
    token_usage: Any = None,
    status: str = "success",
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "agent": agent,
        "action": action,
        "status": status,
        "input": input_summary,
        "tool_calls": tool_calls or [],
        "output": output_summary,
        "token_usage": token_usage or {},
        "latency_ms": round((perf_counter() - started) * 1000, 2),
        "error": error,
    }
