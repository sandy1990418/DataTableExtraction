"""Deterministic verifier: checks draft table citations and row-grain coherence."""

from __future__ import annotations

import re

from app.models.data_table import (
    CellCitation,
    DataTableColumn,
    DataTableSchema,
    EvidenceBlock,
    GroundedCell,
    GroundedRow,
    RowEntity,
    SourceRef,
)
from app.services.data_table.table_composer import DraftCell, DraftDataTable
from app.services.data_table.table_planner import DataTablePlan

_NUMERIC_RE = re.compile(r"-?\d+\.?\d*")

_VAGUE_HEADER_PATTERNS = re.compile(
    r"^(primary\s+)?metric\s*[12345]$|^score\s*[12345]$|^value\s*[12345]$|^column\s*[12345]$",
    re.IGNORECASE,
)


def _ev_text(block: EvidenceBlock) -> str:
    parts = [block.text or ""]
    if block.table_markdown:
        parts.append(block.table_markdown)
    return " ".join(parts)


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _quote_in_evidence(quote: str, text: str) -> bool:
    if not quote:
        return False
    q = _normalize_ws(quote)
    t = _normalize_ws(text)
    if q in t:
        return True
    # partial overlap: 60% of words match
    q_words = q.split()
    t_words = set(t.split())
    overlap = sum(1 for w in q_words if w in t_words)
    return len(q_words) > 0 and overlap / len(q_words) >= 0.6


def _numeric_in_quote(value: float | int, quote: str) -> bool:
    nums = _NUMERIC_RE.findall(quote)
    try:
        return any(abs(float(n) - float(value)) < 1e-4 for n in nums)
    except ValueError:
        return False


def _build_grounded_cell(
    draft_cell: DraftCell,
    evidence_index: dict[str, EvidenceBlock],
    excluded_ids: set[str],
    errors: list[str],
    entity_name: str,
    col_name: str,
) -> GroundedCell:
    status = draft_cell.status
    ev_id = draft_cell.evidence_id
    quote = draft_cell.quote or ""
    value = draft_cell.value

    citations: list[CellCitation] = []
    verification_notes: list[str] = []

    if status == "supported":
        if not ev_id:
            errors.append(f"Cell {entity_name}/{col_name}: status=supported but no evidence_id")
            status = "unsupported"
        elif ev_id in excluded_ids:
            errors.append(f"Cell {entity_name}/{col_name}: cites excluded evidence {ev_id}")
            status = "unsupported"
        elif ev_id not in evidence_index:
            errors.append(f"Cell {entity_name}/{col_name}: unknown evidence_id {ev_id}")
            status = "unsupported"
        else:
            block = evidence_index[ev_id]
            ev_text = _ev_text(block)

            if not quote:
                errors.append(f"Cell {entity_name}/{col_name}: supported but no quote")
                status = "unsupported"
            elif not _quote_in_evidence(quote, ev_text):
                errors.append(f"Cell {entity_name}/{col_name}: quote not found in evidence {ev_id}")
                status = "unsupported"
            else:
                if isinstance(value, (int, float)) and not _numeric_in_quote(value, quote):
                    errors.append(
                        f"Cell {entity_name}/{col_name}: numeric value {value} not found in quote"
                    )
                    status = "unsupported"
                else:
                    citations.append(
                        CellCitation(
                            source_ref=block.source_ref,
                            quote=quote,
                            support_type="direct",
                        )
                    )

    return GroundedCell(
        value=value,
        status=status,
        citations=citations,
        confidence=0.9 if status == "supported" else 0.0,
        verification_notes=verification_notes,
    )


def verify_draft_table(
    draft: DraftDataTable,
    plan: DataTablePlan,
    evidence_store: list[EvidenceBlock],
) -> tuple[list[GroundedRow], DataTableSchema, list[str]]:
    """Verify a DraftDataTable and return (grounded_rows, schema, errors).

    Errors are severe issues that should trigger a repair. Warnings are
    appended directly to a returned list of string warnings.
    """
    evidence_index = {b.evidence_id: b for b in evidence_store}
    excluded_ids = {d.evidence_id for d in plan.evidence_decisions if d.decision == "exclude"}

    errors: list[str] = []
    warnings: list[str] = []

    # Build schema from plan
    columns: list[DataTableColumn] = []
    for i, pc in enumerate(plan.columns):
        role_map = {"string": "attribute", "number": "metric", "boolean": "attribute",
                    "date": "date", "list": "attribute", "unknown": "attribute"}
        role = "entity" if i == 0 else role_map.get(pc.value_type, "attribute")
        columns.append(DataTableColumn(
            name=pc.name,
            role=role,
            description=pc.description,
            value_type=pc.value_type if pc.value_type != "unknown" else "string",
            required=(i == 0),
        ))

    schema = DataTableSchema(
        title=plan.table_title,
        intent=plan.table_purpose,
        columns=columns,
    )

    rows: list[GroundedRow] = []
    col_names = draft.headers

    for draft_row in draft.rows:
        entity_name = draft_row.row_label
        entity = RowEntity(
            entity_id=re.sub(r"\W+", "_", entity_name.lower())[:20],
            name=entity_name,
            confidence=0.8,
        )

        cells: dict[str, GroundedCell] = {}
        for col_name in col_names:
            draft_cell = draft_row.cells.get(col_name, DraftCell(value=None, status="not_reported"))

            # first column is entity column — use row_label
            if col_name == col_names[0]:
                cells[col_name] = GroundedCell(
                    value=entity_name,
                    status="supported",
                    citations=[],
                    confidence=0.9,
                )
                continue

            cells[col_name] = _build_grounded_cell(
                draft_cell=draft_cell,
                evidence_index=evidence_index,
                excluded_ids=excluded_ids,
                errors=errors,
                entity_name=entity_name,
                col_name=col_name,
            )

        rows.append(GroundedRow(entity=entity, cells=cells))

    # table-level warnings
    if rows:
        all_data_cells = [
            cell
            for row in rows
            for name, cell in row.cells.items()
            if name != col_names[0]
        ]
        if all_data_cells:
            empty_ratio = sum(
                1 for c in all_data_cells if c.status in ("not_reported", "unsupported")
            ) / len(all_data_cells)
            if empty_ratio > 0.3:
                warnings.append(
                    f"{empty_ratio:.0%} of cells are empty or unsupported. "
                    "Evidence may not cover the requested table."
                )

    # vague header names
    vague_headers = [h for h in col_names if _VAGUE_HEADER_PATTERNS.match(h.strip())]
    if vague_headers:
        warnings.append(
            f"Vague column names detected: {vague_headers}. "
            "Use specific metric names (e.g. 'Single-Hop F1') or switch to long format."
        )

    # row count much smaller than candidate count
    candidate_count = len(plan.candidate_rows)
    if candidate_count > 0 and len(rows) < candidate_count * 0.5:
        warnings.append(
            f"Only {len(rows)} rows produced from {candidate_count} candidates. "
            "Some candidate rows may have been silently dropped."
        )

    return rows, schema, errors + warnings
