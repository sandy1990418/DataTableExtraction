"""Stage 6: fill a single cell with LLM + grounded citations."""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from app.config import Settings
from app.models.data_table import (
    CellCitation,
    DataTableColumn,
    EvidenceBlock,
    GroundedCell,
    RowEntity,
    SourceRef,
)
from app.prompts.data_table import CELL_FILL_SYSTEM

logger = logging.getLogger(__name__)

_NOT_REPORTED = GroundedCell(value=None, status="not_reported", citations=[], confidence=0.0)


def _evidence_context(blocks: list[EvidenceBlock]) -> str:
    lines = []
    for b in blocks:
        text = (b.text or "")[:600]
        table_hint = f"\nTable:\n{b.table_markdown[:400]}" if b.table_markdown else ""
        lines.append(f"[evidence_id={b.evidence_id}] source={b.source_id}\n{text}{table_hint}")
    return "\n\n---\n\n".join(lines) or "(no evidence)"


def _build_citation(raw: dict, evidence_index: dict[str, EvidenceBlock]) -> CellCitation | None:
    ev_id = str(raw.get("evidence_id", "")).strip()
    quote = str(raw.get("quote", "")).strip()
    support_type = raw.get("support_type", "direct")
    if support_type not in ("direct", "inferred", "conflicting"):
        support_type = "direct"

    block = evidence_index.get(ev_id)
    if not block:
        return None

    return CellCitation(
        source_ref=block.source_ref,
        quote=quote,
        support_type=support_type,
    )


async def fill_cell(
    entity: RowEntity,
    column: DataTableColumn,
    evidence_blocks: list[EvidenceBlock],
    hint: str,
    settings: Settings,
) -> GroundedCell:
    """Call LLM to fill one cell value with citations."""
    if not evidence_blocks:
        return _NOT_REPORTED

    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
    )

    evidence_index = {b.evidence_id: b for b in evidence_blocks}

    user_msg = (
        f"Entity: {entity.name}\n"
        f"Aliases: {', '.join(entity.aliases) or 'none'}\n\n"
        f"Column: {column.name}\n"
        f"Description: {column.description}\n"
        f"Value type: {column.value_type}\n\n"
        f"User hint: {hint}\n\n"
        f"Evidence:\n{_evidence_context(evidence_blocks)}\n\n"
        "Return JSON only. Fill exactly this one cell."
    )

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": CELL_FILL_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_completion_tokens=512,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
    except Exception as exc:
        logger.warning("fill_cell LLM failed for %s/%s: %s", entity.name, column.name, exc)
        return _NOT_REPORTED

    status = data.get("status", "not_reported")
    if status not in ("supported", "not_reported", "conflicting", "inferred", "unsupported"):
        status = "not_reported"

    value = data.get("value")
    confidence = float(data.get("confidence", 0.0))

    citations: list[CellCitation] = []
    for raw_cite in data.get("citations", []):
        cit = _build_citation(raw_cite, evidence_index)
        if cit:
            citations.append(cit)

    # enforce: supported must have citations
    if status == "supported" and not citations:
        status = "unsupported"

    return GroundedCell(
        value=value,
        status=status,
        citations=citations,
        confidence=confidence,
    )
