"""Lazy Groq client and structured-output invocation helpers."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Type

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


@lru_cache(maxsize=1)
def get_chat_model():
    """Create the Groq chat model only when an LLM agent actually runs."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise RuntimeError("langchain-groq is not installed. Install requirements-phase6.txt.") from exc

    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    return ChatGroq(model=model_name, temperature=0, api_key=api_key, timeout=90, max_retries=2)


def invoke_structured(schema: Type[BaseModel], messages: list[tuple[str, str]], llm: Any = None):
    """Invoke a schema-constrained model; return (Pydantic result, token usage)."""
    model = llm or get_chat_model()
    structured = model.with_structured_output(
        schema,
        method="function_calling",
        include_raw=True,
    )
    response = structured.invoke(messages)
    if not isinstance(response, dict):
        raise TypeError("Structured model response was not a dictionary")
    if response.get("parsing_error"):
        raise ValueError(f"Structured output parsing failed: {response['parsing_error']}")
    parsed = response.get("parsed")
    if parsed is None:
        raise ValueError("LLM returned no parsed structured output")
    raw = response.get("raw")
    usage = getattr(raw, "usage_metadata", None) or {}
    if not usage and raw is not None:
        response_metadata = getattr(raw, "response_metadata", {}) or {}
        usage = response_metadata.get("token_usage", {}) or response_metadata.get("usage", {}) or {}
    return parsed, usage
