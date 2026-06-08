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

# Header names that indicate a backbone/base-model column (not the true method grain).
_BASE_MODEL_HEADERS = {"model", "base model", "llm", "backbone", "base_model", "lm"}
# Header names that indicate the actual method/system column.
_METHOD_HEADERS = {"method", "system", "approach", "algorithm", "strategy", "memory"}

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


def _row_label_column_index(headers: list[str]) -> int:
    """
    Return the column index that contains the actual row-grain labels.

    For tables like  Model | Method | F1 | BLEU  the first column is a base
    model (backbone LLM), while the second column holds the actual memory
    method/system being compared.  Return 1 in that case; return 0 otherwise.
    """
    if len(headers) < 2:
        return 0
    h0 = headers[0].strip().lower()
    h1 = headers[1].strip().lower()
    if h0 in _BASE_MODEL_HEADERS and h1 in _METHOD_HEADERS:
        return 1
    return 0


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
        label_col = _row_label_column_index(headers)
        all_row_labels = [
            row[label_col].strip()
            for row in rows
            if len(row) > label_col and row[label_col].strip()
        ]
        # Deduplicate while preserving order (a base-model×method table repeats base models).
        seen: set[str] = set()
        unique_row_labels: list[str] = []
        for lbl in all_row_labels:
            if lbl.lower() not in seen:
                seen.add(lbl.lower())
                unique_row_labels.append(lbl)
        all_row_labels = unique_row_labels
        # Upgrade grain when the method column was chosen.
        if label_col == 1 and grain == "base_model_plus_method":
            grain = "method"

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
