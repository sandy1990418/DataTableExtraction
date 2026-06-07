"""Main pipeline: documents → GroundedDataTable (semantic plan-and-compose)."""

from __future__ import annotations

import logging

from app.config import Settings
from app.models.data_table import (
    DataTableColumn,
    DataTableSchema,
    GroundedDataTable,
)
from app.services.data_table.evidence_store import build_evidence_store
from app.services.data_table.exporters import compute_table_metrics
from app.services.data_table.source_table_summary import summarize_source_tables
from app.services.data_table.table_composer import (
    RESULT_SUMMARY_HEADERS,
    build_draft_from_source_table,
    compose_data_table,
    compose_long_form_from_source_tables,
    compose_result_summary,
)
from app.services.data_table.table_planner import _FALLBACK_PLAN, plan_data_table
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
    plan = await plan_data_table(hint, evidence_store, source_table_summaries, settings, debug_trace=debug_trace)
    debug_trace.append({
        "stage": "table_planning",
        "table_purpose_type": plan.table_purpose_type,
        "table_format": plan.table_format,
        "row_grain": plan.row_grain,
        "used_source_table_ids": [
            s.table_id for s in source_table_summaries
            if s.evidence_id not in {e.evidence_id for e in plan.excluded_source_tables}
            and any(
                d.evidence_id == s.evidence_id and d.decision in ("use", "maybe")
                for d in plan.evidence_decisions
            )
        ],
        "excluded_source_tables": [e.model_dump() for e in plan.excluded_source_tables],
        "candidate_rows": plan.candidate_rows,
        "excluded_candidate_rows": [e.model_dump() for e in plan.excluded_candidate_rows],
        "plan": plan.model_dump(),
    })
    if plan.warnings:
        warnings.extend(plan.warnings)

    # cap columns to max_columns
    if len(plan.columns) > max_columns:
        plan.columns = plan.columns[:max_columns]

    # Stage 4: compose
    # Priority routing by table_purpose_type:
    #   result_summary      → LLM result-summary composer (one row per method/system)
    #   raw_metric_extraction + long format → deterministic long-form source-table expansion
    #   fallback plan (planner LLM failed)  → direct source-table reconstruction
    #   anything else       → general LLM composer
    is_fallback_plan = plan.reason == _FALLBACK_PLAN.reason
    draft = None

    if plan.table_purpose_type == "result_summary" and not is_fallback_plan:
        draft = await compose_result_summary(hint, plan, evidence_store, source_table_summaries, settings)
        debug_trace.append({
            "stage": "table_composition",
            "method": "llm_result_summary",
            "table_purpose_type": plan.table_purpose_type,
            "draft_row_count": len(draft.rows),
            "draft_headers": draft.headers,
        })

    elif plan.table_purpose_type == "raw_metric_extraction" and plan.table_format == "long":
        draft = compose_long_form_from_source_tables(plan, evidence_store, source_table_summaries)
        if draft:
            debug_trace.append({
                "stage": "table_composition",
                "method": "deterministic_long_form_source_table",
                "table_purpose_type": plan.table_purpose_type,
                "generated_rows": len(draft.rows),
                "draft_headers": draft.headers,
            })

    if draft is None and is_fallback_plan:
        draft = build_draft_from_source_table(evidence_store, hint)
        if draft:
            debug_trace.append({
                "stage": "table_composition",
                "method": "source_table_fallback",
                "draft_row_count": len(draft.rows),
                "draft_headers": draft.headers,
            })

    if draft is None:
        draft = await compose_data_table(hint, plan, evidence_store, source_table_summaries, settings)
        debug_trace.append({
            "stage": "table_composition",
            "method": "llm_composer",
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
        if plan.table_purpose_type == "result_summary" and not is_fallback_plan:
            repaired_draft = await compose_result_summary(
                hint, plan, evidence_store, source_table_summaries, settings,
                repair_errors=severe_errors,
            )
        else:
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
