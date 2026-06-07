"""LLM-based table composer: plan + evidence → DraftDataTable with citations."""

from __future__ import annotations

import json
import logging
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.config import Settings
from app.models.data_table import EvidenceBlock
from app.prompts.data_table import TABLE_COMPOSER_SYSTEM
from app.services.data_table.source_table_summary import SourceTableSummary
from app.services.data_table.table_planner import DataTablePlan

logger = logging.getLogger(__name__)


class DraftCell(BaseModel):
    value: str | int | float | bool | None = None
    status: Literal["supported", "not_reported", "conflicting", "inferred"] = "not_reported"
    evidence_id: str | None = None
    quote: str | None = None


class DraftRow(BaseModel):
    row_label: str
    cells: dict[str, DraftCell]


class DraftDataTable(BaseModel):
    headers: list[str]
    rows: list[DraftRow]
    notes: list[str] = []


def _format_plan(plan: DataTablePlan) -> str:
    col_lines = "\n".join(
        f"  - {c.name} ({c.value_type}, policy={c.evidence_policy}): {c.description}"
        for c in plan.columns
    )
    used_ev = [d.evidence_id for d in plan.evidence_decisions if d.decision in ("use", "maybe")]
    return (
        f"Table title: {plan.table_title}\n"
        f"Purpose: {plan.table_purpose}\n"
        f"Row grain: {plan.row_grain}\n"
        f"Generation policy: {plan.generation_policy}\n"
        f"Columns:\n{col_lines}\n"
        f"Evidence to use: {used_ev or 'all available'}"
    )


def _allowed_evidence_ids(plan: DataTablePlan, evidence_store: list[EvidenceBlock]) -> set[str]:
    decided = {d.evidence_id for d in plan.evidence_decisions if d.decision == "exclude"}
    return {b.evidence_id for b in evidence_store if b.evidence_id not in decided}


def _format_evidence(evidence_store: list[EvidenceBlock], allowed_ids: set[str]) -> str:
    lines = []
    for b in evidence_store:
        if b.evidence_id not in allowed_ids:
            continue
        text = (b.text or "")[:500]
        table_section = f"\nTable:\n{b.table_markdown[:600]}" if b.table_markdown else ""
        lines.append(f"[evidence_id={b.evidence_id}]\n{text}{table_section}")
    return "\n\n---\n\n".join(lines) or "(no evidence)"


def _format_summaries(summaries: list[SourceTableSummary], allowed_ids: set[str]) -> str:
    lines = []
    for s in summaries:
        if s.evidence_id not in allowed_ids:
            continue
        sample_str = "\n".join("  | " + " | ".join(r) for r in s.sample_rows)
        lines.append(
            f"[{s.evidence_id}] {s.title or 'untitled'} (grain={s.guessed_row_grain})\n"
            f"  Headers: {s.headers}\n{sample_str}"
        )
    return "\n".join(lines) or "(none)"


def _parse_draft(data: dict, plan: DataTablePlan) -> DraftDataTable:
    headers = data.get("headers", [c.name for c in plan.columns])
    raw_rows = data.get("rows", [])
    rows: list[DraftRow] = []

    for raw_row in raw_rows:
        label = str(raw_row.get("row_label", "")).strip()
        if not label:
            continue
        raw_cells = raw_row.get("cells", {})
        cells: dict[str, DraftCell] = {}
        for header in headers:
            raw_cell = raw_cells.get(header, {})
            if isinstance(raw_cell, dict):
                status = raw_cell.get("status", "not_reported")
                if status not in ("supported", "not_reported", "conflicting", "inferred"):
                    status = "not_reported"
                cells[header] = DraftCell(
                    value=raw_cell.get("value"),
                    status=status,  # type: ignore[arg-type]
                    evidence_id=raw_cell.get("evidence_id") or None,
                    quote=raw_cell.get("quote") or None,
                )
            else:
                cells[header] = DraftCell(value=None, status="not_reported")
        rows.append(DraftRow(row_label=label, cells=cells))

    return DraftDataTable(
        headers=headers,
        rows=rows,
        notes=data.get("notes", []),
    )


async def compose_data_table(
    hint: str,
    plan: DataTablePlan,
    evidence_store: list[EvidenceBlock],
    source_table_summaries: list[SourceTableSummary],
    settings: Settings,
    repair_errors: list[str] | None = None,
) -> DraftDataTable:
    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
    )

    allowed_ids = _allowed_evidence_ids(plan, evidence_store)

    repair_section = ""
    if repair_errors:
        repair_section = (
            "\n\nPREVIOUS ATTEMPT HAD ERRORS — repair these issues:\n"
            + "\n".join(f"- {e}" for e in repair_errors)
            + "\n"
        )

    user_msg = (
        f"User hint: {hint}\n\n"
        f"Table plan:\n{_format_plan(plan)}\n\n"
        f"Source table summaries (allowed):\n{_format_summaries(source_table_summaries, allowed_ids)}\n\n"
        f"Evidence:\n{_format_evidence(evidence_store, allowed_ids)}"
        f"{repair_section}\n\n"
        "Return JSON only."
    )

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": TABLE_COMPOSER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_completion_tokens=4096,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        return _parse_draft(data, plan)
    except Exception as exc:
        logger.warning("compose_data_table LLM call failed: %s", exc)
        return DraftDataTable(headers=[c.name for c in plan.columns], rows=[], notes=[str(exc)])
