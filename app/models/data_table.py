"""Pydantic models for NotebookLM-style grounded data tables."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    source_id: str
    evidence_id: str
    document_name: str | None = None
    kind: str
    title: str | None = None
    page: int | None = None
    section: str | None = None
    image_path: str | None = None
    char_start: int | None = None
    char_end: int | None = None


class EvidenceBlock(BaseModel):
    evidence_id: str
    source_id: str
    document_name: str | None = None
    kind: str
    title: str | None = None
    text: str
    table_markdown: str | None = None
    image_path: str | None = None
    source_ref: SourceRef
    keywords: list[str] = Field(default_factory=list)


class DataTableColumn(BaseModel):
    name: str
    role: Literal["entity", "attribute", "metric", "date", "category", "notes"]
    description: str
    value_type: Literal["string", "number", "boolean", "date", "enum", "list", "unknown"] = "string"
    required: bool = False


class DataTableSchema(BaseModel):
    title: str
    table_type: Literal["synthesized_data_table", "source_table"] = "synthesized_data_table"
    intent: str
    columns: list[DataTableColumn]


class RowEntity(BaseModel):
    entity_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    source_refs: list[SourceRef] = Field(default_factory=list)
    confidence: float = 0.0


class CellCitation(BaseModel):
    source_ref: SourceRef
    quote: str
    support_type: Literal["direct", "inferred", "conflicting"] = "direct"


class GroundedCell(BaseModel):
    value: str | int | float | bool | None
    status: Literal["supported", "not_reported", "conflicting", "inferred", "unsupported"]
    citations: list[CellCitation] = Field(default_factory=list)
    confidence: float = 0.0
    verification_notes: list[str] = Field(default_factory=list)


class GroundedRow(BaseModel):
    entity: RowEntity
    cells: dict[str, GroundedCell]


class GroundedDataTable(BaseModel):
    schema_: DataTableSchema = Field(alias="schema")
    rows: list[GroundedRow] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    debug_trace: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}
