"""Structured message schemas exchanged by the RFP agents."""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, ConfigDict


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FieldDraft(StrictModel):
    """An extracted field plus references to evidence items returned by retrieval."""
    value: str | None = Field(default=None, description="Extracted value, or null when not found")
    evidence_indexes: list[int] = Field(default_factory=list, description="Zero-based indexes of supporting evidence items")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str = Field(default="", description="Explain interpretation or why the field is not found")


class ExtractionDraft(StrictModel):
    fields: dict[str, FieldDraft] = Field(default_factory=dict)


class AddendumChangeDraft(StrictModel):
    field: str
    previous_value: str | None = None
    new_value: str | None = None
    evidence_indexes: list[int] = Field(default_factory=list)
    reason: str = ""


class AddendumReviewDraft(StrictModel):
    updated_fields: dict[str, FieldDraft] = Field(default_factory=dict)
    changes: list[AddendumChangeDraft] = Field(default_factory=list)


class ValidationIssue(StrictModel):
    field: str
    reason: str
    retry: bool = True


class CriticReport(StrictModel):
    issues: list[ValidationIssue] = Field(default_factory=list)
    overall_note: str = ""


class QADraft(StrictModel):
    answer: str = Field(description="Answer using only the supplied evidence")
    evidence_indexes: list[int] = Field(default_factory=list, description="Indexes of passages supporting the answer")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
