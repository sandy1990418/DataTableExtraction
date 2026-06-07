"""Main pipeline: documents → GroundedDataTable."""

from __future__ import annotations

import logging

from app.config import Settings
from app.models.data_table import (
    DataTableColumn,
    GroundedCell,
    GroundedDataTable,
    GroundedRow,
)
from app.services.data_table.cell_filler import fill_cell
from app.services.data_table.cell_retriever import retrieve_cell_evidence
from app.services.data_table.cell_verifier import verify_cell
from app.services.data_table.entity_discovery import discover_entities
from app.services.data_table.evidence_store import build_evidence_store
from app.services.data_table.exporters import compute_table_metrics
from app.services.data_table.intent import detect_table_intent
from app.services.data_table.schema_inducer import induce_schema

logger = logging.getLogger(__name__)

DATA_TABLE_MAX_ROWS = 20
DATA_TABLE_MAX_COLUMNS = 6
DATA_TABLE_CELL_EVIDENCE_K = 5
DATA_TABLE_REPAIR_ROUNDS = 1
DATA_TABLE_DROP_SPARSE_ROWS = True
DATA_TABLE_DROP_UNSUPPORTED_COLUMNS = False


async def generate_data_table(
    evidence_items: list,
    hint: str,
    settings: Settings,
    max_rows: int = DATA_TABLE_MAX_ROWS,
    max_columns: int = DATA_TABLE_MAX_COLUMNS,
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

    # Stage 2: intent
    intent = await detect_table_intent(hint, evidence_store, settings)
    debug_trace.append({"stage": "intent", "result": intent})

    # Stage 3: schema
    schema = await induce_schema(hint, intent, evidence_store, settings, max_columns=max_columns)
    debug_trace.append({"stage": "schema", "columns": [c.name for c in schema.columns]})

    # Stage 4: entity discovery
    entities = await discover_entities(hint, schema, evidence_store, settings, max_rows=max_rows)
    debug_trace.append({"stage": "entity_discovery", "entity_count": len(entities)})

    if not entities:
        warnings.append("No entities discovered from evidence. Table is empty.")
        return GroundedDataTable(
            **{"schema": schema, "warnings": warnings, "debug_trace": debug_trace}
        )

    # Stages 5-7: per-entity, per-cell retrieval → fill → verify → optional repair
    rows: list[GroundedRow] = []
    for entity in entities:
        cells: dict[str, GroundedCell] = {}

        for column in schema.columns:
            if column.role == "entity":
                cells[column.name] = GroundedCell(
                    value=entity.name,
                    status="supported",
                    citations=[],
                    confidence=entity.confidence,
                )
                continue

            # Stage 5: retrieval
            evidence = retrieve_cell_evidence(
                entity=entity,
                column=column,
                evidence_store=evidence_store,
                hint=hint,
                k=DATA_TABLE_CELL_EVIDENCE_K,
                debug_trace=debug_trace,
            )

            # Stage 6: fill
            cell = await fill_cell(
                entity=entity,
                column=column,
                evidence_blocks=evidence,
                hint=hint,
                settings=settings,
            )

            # Stage 7: verify
            cell = verify_cell(
                cell=cell,
                entity=entity,
                column=column,
                evidence_blocks=evidence,
            )

            # Stage 8: repair loop
            if cell.status == "unsupported" and DATA_TABLE_REPAIR_ROUNDS > 0:
                expanded_evidence = retrieve_cell_evidence(
                    entity=entity,
                    column=column,
                    evidence_store=evidence_store,
                    hint=hint + " " + column.description,
                    k=DATA_TABLE_CELL_EVIDENCE_K + 3,
                    debug_trace=debug_trace,
                )
                repaired = await fill_cell(
                    entity=entity,
                    column=column,
                    evidence_blocks=expanded_evidence,
                    hint=hint,
                    settings=settings,
                )
                repaired = verify_cell(
                    cell=repaired,
                    entity=entity,
                    column=column,
                    evidence_blocks=expanded_evidence,
                )
                debug_trace.append({
                    "stage": "repair",
                    "entity": entity.name,
                    "column": column.name,
                    "before": cell.status,
                    "after": repaired.status,
                })
                cell = repaired

            cells[column.name] = cell

        row = GroundedRow(entity=entity, cells=cells)

        # drop sparse rows
        if DATA_TABLE_DROP_SPARSE_ROWS:
            non_entity_cells = [
                c for name, c in cells.items()
                if any(col.name == name and col.role != "entity" for col in schema.columns)
            ]
            supported_count = sum(1 for c in non_entity_cells if c.status == "supported")
            if non_entity_cells and supported_count == 0:
                warnings.append(f"Dropped sparse row: {entity.name} (no supported cells)")
                continue

        rows.append(row)

    # table-level warnings
    if rows:
        col_names = [c.name for c in schema.columns if c.role != "entity"]
        for col_name in col_names:
            col_cells = [row.cells.get(col_name) for row in rows if row.cells.get(col_name)]
            not_reported_ratio = sum(1 for c in col_cells if c.status == "not_reported") / max(len(col_cells), 1)
            if not_reported_ratio > 0.8:
                warnings.append(
                    f"Column '{col_name}' has {not_reported_ratio:.0%} not_reported cells. "
                    "Schema may not be well supported by sources."
                )

        all_non_entity = [
            c for row in rows for name, c in row.cells.items()
            if any(col.name == name and col.role != "entity" for col in schema.columns)
        ]
        if all_non_entity:
            not_reported_ratio = sum(1 for c in all_non_entity if c.status == "not_reported") / len(all_non_entity)
            if not_reported_ratio > 0.4:
                warnings.append(
                    "Many cells are not reported. Schema may not be well supported by sources."
                )

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


def _empty_schema(hint: str):
    from app.models.data_table import DataTableColumn, DataTableSchema
    return DataTableSchema(
        title="Empty Table",
        intent=hint or "No data available.",
        columns=[
            DataTableColumn(name="Entity", role="entity", description="No data.", required=True)
        ],
    )
