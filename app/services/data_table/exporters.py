"""Stage 9 + 10: exporters and metrics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.models.data_table import GroundedDataTable


def compute_table_metrics(data_table: GroundedDataTable) -> dict:
    row_count = len(data_table.rows)
    col_count = len(data_table.schema_.columns)

    status_counts: dict[str, int] = {
        "supported": 0,
        "not_reported": 0,
        "unsupported": 0,
        "conflicting": 0,
        "inferred": 0,
    }
    citation_count = 0

    for row in data_table.rows:
        for cell in row.cells.values():
            status_counts[cell.status] = status_counts.get(cell.status, 0) + 1
            citation_count += len(cell.citations)

    cell_count = row_count * col_count
    supported = status_counts["supported"]
    supported_ratio = round(supported / cell_count, 4) if cell_count else 0.0

    return {
        "row_count": row_count,
        "column_count": col_count,
        "cell_count": cell_count,
        "supported_cell_count": supported,
        "not_reported_cell_count": status_counts["not_reported"],
        "unsupported_cell_count": status_counts["unsupported"],
        "conflicting_cell_count": status_counts["conflicting"],
        "citation_count": citation_count,
        "supported_cell_ratio": supported_ratio,
    }


def to_simple_table(data_table: GroundedDataTable) -> dict:
    headers = [col.name for col in data_table.schema_.columns]
    rows = []
    for row in data_table.rows:
        cells = []
        for col in data_table.schema_.columns:
            cell = row.cells.get(col.name)
            cells.append(str(cell.value) if cell and cell.value is not None else "")
        rows.append(cells)
    return {"headers": headers, "rows": rows}


def to_citation_table(data_table: GroundedDataTable) -> dict:
    headers = [
        "row_entity",
        "column",
        "value",
        "status",
        "source_id",
        "evidence_id",
        "quote",
        "support_type",
    ]
    rows = []
    for row in data_table.rows:
        for col in data_table.schema_.columns:
            cell = row.cells.get(col.name)
            if not cell:
                continue
            value_str = str(cell.value) if cell.value is not None else ""
            if cell.citations:
                for cit in cell.citations:
                    rows.append([
                        row.entity.name,
                        col.name,
                        value_str,
                        cell.status,
                        cit.source_ref.source_id,
                        cit.source_ref.evidence_id,
                        cit.quote,
                        cit.support_type,
                    ])
            else:
                rows.append([row.entity.name, col.name, value_str, cell.status, "", "", "", ""])
    return {"headers": headers, "rows": rows}


def to_debug_json(data_table: GroundedDataTable) -> dict:
    return {
        "schema": data_table.schema_.model_dump(),
        "metrics": data_table.metrics,
        "warnings": data_table.warnings,
        "debug_trace": data_table.debug_trace,
        "rows": [
            {
                "entity": row.entity.model_dump(),
                "cells": {
                    col_name: cell.model_dump()
                    for col_name, cell in row.cells.items()
                },
            }
            for row in data_table.rows
        ],
    }


def write_table_csv(data_table: GroundedDataTable, path: str | Path) -> None:
    simple = to_simple_table(data_table)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(simple["headers"])
        writer.writerows(simple["rows"])


def write_citations_csv(data_table: GroundedDataTable, path: str | Path) -> None:
    cit = to_citation_table(data_table)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(cit["headers"])
        writer.writerows(cit["rows"])


def write_debug_json(data_table: GroundedDataTable, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_debug_json(data_table), f, ensure_ascii=False, indent=2)
