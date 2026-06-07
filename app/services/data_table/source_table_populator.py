"""Deterministic cell filling from source markdown tables (benchmark/result mode)."""

from __future__ import annotations

import re

from app.models.data_table import (
    CellCitation,
    DataTableColumn,
    EvidenceBlock,
    GroundedCell,
    RowEntity,
    SourceRef,
)
from app.services.data_table.source_table_rows import parse_markdown_table


def _normalize(text: str) -> str:
    return re.sub(r"[\s\-_]+", "", text.lower())


def _row_as_quote(headers: list[str], row: list[str]) -> str:
    """Serialize a table row as a readable quote for citations."""
    parts = [f"{h}={v}" for h, v in zip(headers, row) if v.strip()]
    return " | ".join(parts)


def fill_cell_from_source_table(
    entity: RowEntity,
    column: DataTableColumn,
    evidence_blocks: list[EvidenceBlock],
) -> GroundedCell | None:
    """Try to fill a cell deterministically from a source markdown table.

    Returns None if no matching table row + column is found.
    """
    if column.role == "entity":
        return None

    entity_names = {_normalize(entity.name), *{_normalize(a) for a in entity.aliases}}
    col_name_norm = _normalize(column.name)

    for block in evidence_blocks:
        if not block.table_markdown:
            continue

        headers, rows = parse_markdown_table(block.table_markdown)
        if not headers or not rows:
            continue

        # find entity column
        entity_col_idx = 0
        for i, h in enumerate(headers):
            if _normalize(h) in {"method", "model", "system", "approach", "baseline", "name", "algorithm"}:
                entity_col_idx = i
                break

        # find column matching target
        target_col_idx = None
        for i, h in enumerate(headers):
            if _normalize(h) == col_name_norm:
                target_col_idx = i
                break
        if target_col_idx is None:
            # try partial match
            for i, h in enumerate(headers):
                if col_name_norm in _normalize(h) or _normalize(h) in col_name_norm:
                    target_col_idx = i
                    break
        if target_col_idx is None:
            continue

        # find matching entity row
        for row in rows:
            if entity_col_idx >= len(row):
                continue
            row_entity_norm = _normalize(row[entity_col_idx])
            if row_entity_norm not in entity_names:
                # check aliases too
                if not any(alias == row_entity_norm for alias in entity_names):
                    continue

            if target_col_idx >= len(row):
                continue
            value_str = row[target_col_idx].strip()
            if not value_str:
                return GroundedCell(value=None, status="not_reported", citations=[], confidence=0.0)

            # try to parse as number
            value: str | float = value_str
            try:
                value = float(value_str.replace(",", ""))
            except ValueError:
                pass

            quote = _row_as_quote(headers, row)
            citation = CellCitation(
                source_ref=block.source_ref,
                quote=quote,
                support_type="direct",
            )
            return GroundedCell(
                value=value,
                status="supported",
                citations=[citation],
                confidence=0.95,
            )

    return None
