"""Main pipeline: documents → GroundedDataTable (semantic plan-and-compose)."""

from __future__ import annotations

import logging

from app.config import Settings
from app.models.data_table import (
    DataTableColumn,
    DataTableSchema,
    GroundedCell,
    GroundedDataTable,
    GroundedRow,
    RowEntity,
)
from app.services.data_table.evidence_store import build_evidence_store
from app.services.data_table.exporters import compute_table_metrics
from app.services.data_table.source_table_summary import summarize_source_tables
from app.services.data_table.table_composer import compose_data_table
from app.services.data_table.table_planner import plan_data_table
from app.services.data_table.table_verifier import verify_draft_table
from app.services.data_table.text_table_extractor import extract_tables_from_text_blocks, inject_extracted_tables

logger = logging.getLogger(__name__)

_REPAIR_ROUNDS = 1
_SEVERE_ERROR_KEYWORDS = ("excluded evidence", "quote not found", "numeric value", "no evidence_id")


async def generate_data_table(
    evidence_items: list,
    hint: str,
    settings: Settings,
    max_rows: int = 20,
    max_columns: int = 6,
) -> GroundedDataTable:
    debug_trace: list = []
    warnings: list[str] = []

    # Stage 1: evidence store
    evidence_store = build_evidence_store(evidence_items)
    if not evidence_store:
        warnings.append("No evidence found. Cannot generate data table.")
        return GroundedDataTable(
            **{"schema": _empty_schema(hint), "warnings": warnings, "debug_trace": debug_trace}
        )

    debug_trace.append({"stage": "evidence_store", "block_count": len(evidence_store)})

    # Stage 1b: extract structured tables from plain-text blocks
    extracted_tables = await extract_tables_from_text_blocks(
        evidence_store, settings, hint=hint, debug_trace=debug_trace
    )
    if extracted_tables:
        evidence_store = inject_extracted_tables(evidence_store, extracted_tables)
        debug_trace.append({
            "stage": "text_table_injection",
            "injected_count": len(extracted_tables),
        })

    # Stage 2: source table summaries (planner input, not final output)
    source_table_summaries = summarize_source_tables(evidence_store)
    debug_trace.append({
        "stage": "source_table_summary",
        "tables": [s.model_dump() for s in source_table_summaries],
    })

    # Stage 3: LLM table planner
    plan = await plan_data_table(hint, evidence_store, source_table_summaries, settings)
    debug_trace.append({
        "stage": "table_planning",
        "plan": plan.model_dump(),
    })
    if plan.warnings:
        warnings.extend(plan.warnings)

    # cap columns to max_columns
    if len(plan.columns) > max_columns:
        plan.columns = plan.columns[:max_columns]

    # Stage 4: LLM table composer
    draft = await compose_data_table(hint, plan, evidence_store, source_table_summaries, settings)
    debug_trace.append({
        "stage": "table_composition",
        "draft_row_count": len(draft.rows),
        "draft_headers": draft.headers,
    })

    # cap rows to max_rows
    if len(draft.rows) > max_rows:
        draft.rows = draft.rows[:max_rows]

    # Stage 5: deterministic verifier
    rows, schema, errors = verify_draft_table(draft, plan, evidence_store)
    debug_trace.append({
        "stage": "verification",
        "warnings": errors,
    })

    # Stage 6: repair loop for severe errors
    severe_errors = [e for e in errors if any(kw in e for kw in _SEVERE_ERROR_KEYWORDS)]
    if severe_errors and _REPAIR_ROUNDS > 0:
        logger.info("Triggering repair loop: %d severe errors", len(severe_errors))
        repaired_draft = await compose_data_table(
            hint, plan, evidence_store, source_table_summaries, settings,
            repair_errors=severe_errors,
        )
        repaired_rows, schema, repaired_errors = verify_draft_table(repaired_draft, plan, evidence_store)
        debug_trace.append({
            "stage": "repair",
            "errors_before": len(severe_errors),
            "errors_after": len([e for e in repaired_errors if any(kw in e for kw in _SEVERE_ERROR_KEYWORDS)]),
        })
        if len(repaired_errors) <= len(errors):
            rows = repaired_rows
            errors = repaired_errors
        else:
            warnings.append("Repair loop did not improve the table. Using original output with warnings.")

    # forward non-severe errors as warnings
    for e in errors:
        if e not in warnings:
            warnings.append(e)

    if not rows:
        warnings.append("No rows were produced. Evidence may not support the requested table.")

    data_table = GroundedDataTable(
        **{
            "schema": schema,
            "rows": rows,
            "warnings": warnings,
            "debug_trace": debug_trace,
        }
    )
    data_table.metrics = compute_table_metrics(data_table)
    return data_table


def _empty_schema(hint: str) -> DataTableSchema:
    return DataTableSchema(
        title="Empty Table",
        intent=hint or "No data available.",
        columns=[
            DataTableColumn(name="Entity", role="entity", description="No data.", required=True)
        ],
    )
