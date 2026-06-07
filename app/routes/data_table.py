"""POST /data-table — NotebookLM-style grounded data table generation."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.models.schemas import DocumentInput
from app.services.data_table.exporters import to_citation_table, to_debug_json, to_simple_table
from app.services.data_table.pipeline import generate_data_table
from app.services.pipeline import build_evidence, resolve_documents

router = APIRouter()


class DataTableRequest(BaseModel):
    documents: list[DocumentInput] = Field(description="One or more markdown documents")
    hint: str = Field(default="", description="What kind of table the user wants")
    analyze_images: bool = Field(default=True)
    max_rows: int = Field(default=20, ge=1, le=100)
    max_columns: int = Field(default=6, ge=3, le=12)


@router.post("/data-table", summary="Documents + hint → NotebookLM-style grounded data table")
async def data_table_endpoint(body: DataTableRequest, settings: Settings = Depends(get_settings)):
    parsed_docs = resolve_documents(body.documents)
    evidence_items = await build_evidence(parsed_docs, settings, body.analyze_images)

    data_table = await generate_data_table(
        evidence_items=evidence_items,
        hint=body.hint,
        settings=settings,
        max_rows=body.max_rows,
        max_columns=body.max_columns,
    )

    simple = to_simple_table(data_table)
    citations = to_citation_table(data_table)
    debug = to_debug_json(data_table)

    return {
        "type": "data_table",
        "table": simple,
        "citations": citations,
        "grounded_table": debug,
        "metrics": data_table.metrics,
        "debug_trace": data_table.debug_trace,
        "warnings": data_table.warnings,
    }
