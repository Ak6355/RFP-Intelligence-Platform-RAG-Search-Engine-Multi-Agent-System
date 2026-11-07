"""Validated request and response schemas for the RFP API."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing_extensions import Annotated

BidId = Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")]


class StrictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class IndexRequest(StrictRequest):
    bid_id: BidId


class AskRequest(StrictRequest):
    bid_id: BidId
    question: str = Field(min_length=3, max_length=2000)


class ExtractRequest(StrictRequest):
    bid_id: BidId
    max_retries: int = Field(default=2, ge=0, le=5)


class SearchResponse(BaseModel):
    query: str
    results: list[dict[str, Any]]
