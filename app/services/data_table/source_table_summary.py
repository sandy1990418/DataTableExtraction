"""Summarize source tables from evidence for use as planner input."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from app.models.data_table import EvidenceBlock
from app.services.data_table.source_table_rows import parse_markdown_table

_NUMERIC_RE = re.compile(r"^\s*-?\d")

_GRAIN_HINTS: list[tuple[list[str], str]] = [
    (["method", "model", "system", "approach", "algorithm", "baseline"], "base_model_plus_method"),
    (["memory", "mem", "memgpt", "memorybank", "a-mem", "amem"], "memory_system"),
    (["dataset", "data", "corpus", "benchmark"], "dataset"),
    (["task", "subtask", "category"], "task"),
    (["paper", "study", "work", "publication"], "paper"),
    (["ablation", "variant", "configuration", "setting"], "ablation_variant"),
]

RowGrain = Literal[
    "base_model_plus_method",
    "memory_system",
    "method",
    "dataset",
    "task",
    "ablation_variant",
    "paper",
    "unknown",
]


class SourceTableSummary(BaseModel):
    table_id: str
    evidence_id: str
    title: str | None = None
    headers: list[str]
    sample_rows: list[list[str]]
    all_row_labels: list[str] = []
    row_count: int
    numeric_column_count: int
    guessed_row_grain: RowGrain
    notes: str = ""


def _normalize(text: str) -> str:
    return re.sub(r"[\s\-_]+", "", text.lower())


def _guess_row_grain(headers: list[str], sample_rows: list[list[str]]) -> RowGrain:
    first_col_values = [row[0] for row in sample_rows if row] if sample_rows else []
    all_text = " ".join(headers + first_col_values).lower()

    for terms, grain in _GRAIN_HINTS:
        if any(t in all_text for t in terms):
            return grain  # type: ignore[return-value]
    return "unknown"


def _count_numeric_columns(headers: list[str], rows: list[list[str]]) -> int:
    if not rows:
        return 0
    count = 0
    for col_idx in range(len(headers)):
        sample = [row[col_idx] for row in rows[:5] if col_idx < len(row) and row[col_idx].strip()]
        if sample and all(_NUMERIC_RE.match(v) for v in sample):
            count += 1
    return count


def summarize_source_tables(evidence_store: list[EvidenceBlock]) -> list[SourceTableSummary]:
    summaries: list[SourceTableSummary] = []
    table_idx = 0

    for block in evidence_store:
        if not block.table_markdown:
            continue
        headers, rows = parse_markdown_table(block.table_markdown)
        if not headers or not rows:
            continue

        sample_rows = rows[:5]
        grain = _guess_row_grain(headers, sample_rows)
        numeric_cols = _count_numeric_columns(headers, rows)
        all_row_labels = [row[0].strip() for row in rows if row and row[0].strip()]

        table_idx += 1
        summaries.append(
            SourceTableSummary(
                table_id=f"tbl_{table_idx}",
                evidence_id=block.evidence_id,
                title=block.title,
                headers=headers,
                sample_rows=sample_rows,
                all_row_labels=all_row_labels,
                row_count=len(rows),
                numeric_column_count=numeric_cols,
                guessed_row_grain=grain,
                notes=f"from {block.document_name or block.source_id}",
            )
        )

    return summaries
