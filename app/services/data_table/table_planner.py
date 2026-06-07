"""LLM-based table planner: understand hint + evidence → produce DataTablePlan."""

from __future__ import annotations

import json
import logging
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.config import Settings
from app.models.data_table import EvidenceBlock
from app.prompts.data_table import TABLE_PLANNER_SYSTEM
from app.services.data_table.source_table_summary import SourceTableSummary

logger = logging.getLogger(__name__)


class PlannedColumn(BaseModel):
    name: str
    description: str
    value_type: Literal["string", "number", "boolean", "date", "list", "unknown"] = "unknown"
    evidence_policy: Literal["source_table", "text", "mixed"] = "mixed"


class EvidenceDecision(BaseModel):
    evidence_id: str
    decision: Literal["use", "maybe", "exclude"]
    reason: str


class DataTablePlan(BaseModel):
    table_title: str
    table_purpose: str
    row_grain: str
    columns: list[PlannedColumn]
    evidence_decisions: list[EvidenceDecision]
    generation_policy: Literal[
        "single_source_table_reconstruction",
        "coherent_synthesis",
        "system_summary_with_metrics",
    ] = "coherent_synthesis"
    warnings: list[str] = []
    reason: str = ""


_FALLBACK_PLAN = DataTablePlan(
    table_title="Data Table",
    table_purpose="Compare entities from provided sources.",
    row_grain="unknown",
    columns=[
        PlannedColumn(name="Entity", description="The subject being compared.", value_type="string", evidence_policy="mixed"),
        PlannedColumn(name="Description", description="Brief description.", value_type="string", evidence_policy="text"),
    ],
    evidence_decisions=[],
    generation_policy="coherent_synthesis",
    reason="Fallback plan — LLM planning failed.",
)


def _format_summaries(summaries: list[SourceTableSummary]) -> str:
    if not summaries:
        return "(no structured tables found in evidence)"
    lines = []
    for s in summaries:
        sample_str = "; ".join(" | ".join(r) for r in s.sample_rows[:3])
        lines.append(
            f"[{s.table_id} / {s.evidence_id}] title={s.title or 'untitled'} "
            f"grain={s.guessed_row_grain} headers={s.headers} "
            f"rows={s.row_count} numeric_cols={s.numeric_column_count}\n"
            f"  sample: {sample_str}"
        )
    return "\n".join(lines)


def _format_text_evidence(evidence_store: list[EvidenceBlock], max_items: int = 8) -> str:
    lines = []
    for b in evidence_store[:max_items]:
        if b.table_markdown:
            continue  # tables already in summaries
        preview = (b.text or "")[:300].replace("\n", " ")
        lines.append(f"[{b.evidence_id}] {b.title or ''}: {preview}")
    return "\n".join(lines) or "(no text evidence)"


def _parse_plan(data: dict) -> DataTablePlan:
    columns = []
    for col in data.get("columns", []):
        try:
            columns.append(PlannedColumn.model_validate(col))
        except Exception:
            pass

    evidence_decisions = []
    for dec in data.get("evidence_decisions", []):
        try:
            evidence_decisions.append(EvidenceDecision.model_validate(dec))
        except Exception:
            pass

    policy = data.get("generation_policy", "coherent_synthesis")
    if policy not in ("single_source_table_reconstruction", "coherent_synthesis", "system_summary_with_metrics"):
        policy = "coherent_synthesis"

    return DataTablePlan(
        table_title=str(data.get("table_title", "Data Table")).strip() or "Data Table",
        table_purpose=str(data.get("table_purpose", "")).strip(),
        row_grain=str(data.get("row_grain", "unknown")).strip() or "unknown",
        columns=columns if len(columns) >= 1 else _FALLBACK_PLAN.columns,
        evidence_decisions=evidence_decisions,
        generation_policy=policy,  # type: ignore[arg-type]
        warnings=data.get("warnings", []),
        reason=str(data.get("reason", "")),
    )


async def plan_data_table(
    hint: str,
    evidence_store: list[EvidenceBlock],
    source_table_summaries: list[SourceTableSummary],
    settings: Settings,
) -> DataTablePlan:
    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
    )

    user_msg = (
        f"User hint: {hint}\n\n"
        f"Source table summaries:\n{_format_summaries(source_table_summaries)}\n\n"
        f"Text evidence snippets:\n{_format_text_evidence(evidence_store)}\n\n"
        "Return JSON only."
    )

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": TABLE_PLANNER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_completion_tokens=1024,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        return _parse_plan(data)
    except Exception as exc:
        logger.warning("plan_data_table LLM call failed: %s — using fallback plan", exc)
        return _FALLBACK_PLAN
