"""Multi-stage concept-table workflow: outline → per-row compose → critique → assemble.

Stage A (1 LLM call)   read all evidence → per-concept outline (key points, evidence map,
                       narrative row order)
Stage B (N LLM calls)  one focused call per concept with only its relevant evidence → rich row
Stage C (1 LLM call)   richness critic over the drafted table → targeted rewrite of thin rows
Stage D (no LLM call)  deterministic assembly in Stage A's narrative order

This replaces the single giant compose call for concept_summary tables: per-row
focused context avoids lost-in-the-middle dilution over long transcripts, and the
critic pass applies the evaluator-optimizer pattern (Self-Refine) so generic or
thin cells get rewritten with directed feedback. Falls back to the single-call
composer when Stage A fails.
"""

from __future__ import annotations

import asyncio
import json
import logging

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.config import Settings
from app.services.data_table.llm_client import make_client
from app.models.data_table import EvidenceBlock
from app.prompts.data_table import (
    CONCEPT_CRITIC_SYSTEM,
    CONCEPT_OUTLINE_SYSTEM,
    CONCEPT_ROW_SYSTEM,
)
from app.services.data_table.table_composer import (
    DraftDataTable,
    DraftRow,
    _allowed_evidence_ids,
    _format_all_evidence_for_concepts,
    _rows_from_fixed_columns,
    compose_concept_summary,
)
from app.services.data_table.table_planner import DataTablePlan

logger = logging.getLogger(__name__)

_MAX_CONCEPTS = 20
_MAX_CONCURRENT_ROW_CALLS = 5
_MAX_CRITIC_REWRITES = 8
_ROW_EVIDENCE_CHAR_LIMIT = 9000


class ConceptOutline(BaseModel):
    name: str
    evidence_ids: list[str] = []
    key_points: list[str] = []
    related_concepts: list[str] = []


class TableOutline(BaseModel):
    concepts: list[ConceptOutline] = []
    row_order: list[str] = []


def parse_table_outline(data: dict) -> TableOutline:
    concepts = []
    for raw in data.get("concepts", []):
        if not isinstance(raw, dict) or not str(raw.get("name", "")).strip():
            continue
        try:
            concepts.append(ConceptOutline.model_validate(raw))
        except Exception:
            concepts.append(ConceptOutline(name=str(raw.get("name", "")).strip()))
    row_order = [str(n) for n in data.get("row_order", []) if n]
    return TableOutline(concepts=concepts[:_MAX_CONCEPTS], row_order=row_order)


def select_evidence_for_concept(
    outline: ConceptOutline,
    evidence_store: list[EvidenceBlock],
    allowed_ids: set[str],
    char_limit: int = _ROW_EVIDENCE_CHAR_LIMIT,
) -> str:
    """Stage B evidence: blocks the outline mapped to this concept, plus blocks whose
    text lexically mentions the concept name. Keeps the per-row context small and
    relevant so the row call gets full attention on its own concept."""
    outline_ids = set(outline.evidence_ids)
    name_lower = outline.name.lower()

    selected: list[EvidenceBlock] = []
    seen: set[str] = set()
    # outline-mapped blocks first (highest relevance), then lexical matches
    for b in evidence_store:
        if b.evidence_id in outline_ids and b.evidence_id in allowed_ids:
            selected.append(b)
            seen.add(b.evidence_id)
    for b in evidence_store:
        if b.evidence_id in seen or b.evidence_id not in allowed_ids:
            continue
        if b.text and name_lower in b.text.lower():
            selected.append(b)
            seen.add(b.evidence_id)

    lines: list[str] = []
    used = 0
    for b in selected:
        text = (b.text or "").replace("\n", " ")
        table_section = f"\nTable:\n{b.table_markdown[:800]}" if b.table_markdown else ""
        chunk = f"[evidence_id={b.evidence_id}] {b.title or ''}\n{text}{table_section}"
        if used + len(chunk) > char_limit and lines:
            break
        lines.append(chunk[: char_limit - used])
        used += len(chunk)
    return "\n\n---\n\n".join(lines) or "(no evidence)"


def order_rows(rows: list[DraftRow], row_order: list[str]) -> list[DraftRow]:
    """Stage D: deterministic narrative ordering using Stage A's row_order.
    Rows not in row_order keep their relative position at the end."""
    pos = {name.lower(): i for i, name in enumerate(row_order)}
    base = len(pos)
    keyed = [
        (pos.get(r.row_label.lower(), base + i), i, r) for i, r in enumerate(rows)
    ]
    return [r for _, _, r in sorted(keyed, key=lambda t: (t[0], t[1]))]


async def _chat_json(
    client: AsyncOpenAI,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 16000,
) -> dict:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_completion_tokens=max_tokens,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content or "{}")


async def _stage_a_outline(
    client: AsyncOpenAI,
    model: str,
    hint: str,
    plan: DataTablePlan,
    evidence_store: list[EvidenceBlock],
    allowed_ids: set[str],
) -> TableOutline | None:
    evidence_str = _format_all_evidence_for_concepts(evidence_store, allowed_ids)
    candidates = ", ".join(plan.candidate_rows) if plan.candidate_rows else "(none — enumerate from evidence)"
    user_msg = (
        f"User hint: {hint}\n\n"
        f"Table title: {plan.table_title}\n"
        f"Candidate concepts (must all appear): {candidates}\n\n"
        f"=== Evidence ===\n{evidence_str}\n\n"
        f"Produce the outline for at most {_MAX_CONCEPTS} concepts. Return compact JSON only."
    )
    try:
        data = await _chat_json(client, model, CONCEPT_OUTLINE_SYSTEM, user_msg)
        outline = parse_table_outline(data)
        return outline if outline.concepts else None
    except Exception as exc:
        logger.warning("concept workflow stage A failed: %s", exc)
        return None


async def _stage_b_row(
    client: AsyncOpenAI,
    model: str,
    hint: str,
    concept: ConceptOutline,
    columns: list[str],
    evidence_str: str,
    feedback: list[str] | None = None,
) -> DraftRow | None:
    key_points = "\n".join(f"- {p}" for p in concept.key_points) or "(none)"
    related = ", ".join(concept.related_concepts) or "(none)"
    feedback_section = ""
    if feedback:
        feedback_section = (
            "\n\nEDITOR FEEDBACK on the previous draft of this row — fix every issue:\n"
            + "\n".join(f"- {i}" for i in feedback)
        )
    user_msg = (
        f"User hint: {hint}\n\n"
        f"Concept: {concept.name}\n"
        f"Columns to fill (use exactly, in order): {columns}\n"
        f"Key points from the source (work in every applicable one):\n{key_points}\n"
        f"Related concepts: {related}\n\n"
        f"=== Evidence for this concept ===\n{evidence_str}"
        f"{feedback_section}\n\n"
        "Return compact JSON only."
    )
    try:
        data = await _chat_json(client, model, CONCEPT_ROW_SYSTEM, user_msg)
        rows = _rows_from_fixed_columns([data], columns)
        return rows[0] if rows else None
    except Exception as exc:
        logger.warning("concept workflow stage B failed for %r: %s", concept.name, exc)
        return None


def _render_rows_for_critic(rows: list[DraftRow]) -> str:
    rendered = [
        {
            "row_label": r.row_label,
            "cells": {col: cell.value for col, cell in r.cells.items()},
        }
        for r in rows
    ]
    return json.dumps(rendered, ensure_ascii=False)


async def _stage_c_critic(
    client: AsyncOpenAI,
    model: str,
    rows: list[DraftRow],
    outlines: dict[str, ConceptOutline],
) -> dict[str, list[str]]:
    """Return {row_label: [issues]} for rows that need a rewrite."""
    outline_str = json.dumps(
        {name: o.key_points for name, o in outlines.items()}, ensure_ascii=False
    )
    user_msg = (
        f"=== Drafted table rows ===\n{_render_rows_for_critic(rows)}\n\n"
        f"=== Per-concept key points from the sources ===\n{outline_str}\n\n"
        "Judge every row. Return compact JSON only."
    )
    try:
        data = await _chat_json(client, model, CONCEPT_CRITIC_SYSTEM, user_msg)
    except Exception as exc:
        logger.warning("concept workflow stage C failed: %s", exc)
        return {}

    flagged: dict[str, list[str]] = {}
    for v in data.get("verdicts", []):
        if not isinstance(v, dict) or v.get("ok", True):
            continue
        label = str(v.get("row_label", "")).strip()
        issues = [str(i) for i in v.get("issues", []) if i]
        if label and issues:
            flagged[label] = issues
    return flagged


async def run_concept_workflow(
    hint: str,
    plan: DataTablePlan,
    evidence_store: list[EvidenceBlock],
    settings: Settings,
    debug_trace: list | None = None,
) -> DraftDataTable:
    """Run the four-stage concept workflow; fall back to the single-call composer
    if the outline stage fails."""
    client = make_client(settings)
    model = settings.OPENAI_MODEL
    allowed_ids = _allowed_evidence_ids(plan, evidence_store)
    columns = [c.name for c in plan.columns] if plan.columns else []

    # Stage A — outline
    outline = await _stage_a_outline(client, model, hint, plan, evidence_store, allowed_ids)
    if outline is None:
        if debug_trace is not None:
            debug_trace.append({
                "stage": "concept_workflow",
                "outcome": "outline_failed_fallback_single_call",
            })
        return await compose_concept_summary(hint, plan, evidence_store, settings)

    outlines_by_name = {c.name: c for c in outline.concepts}
    if debug_trace is not None:
        debug_trace.append({
            "stage": "concept_workflow_outline",
            "concepts": [c.name for c in outline.concepts],
            "row_order": outline.row_order,
        })

    # Stage B — one focused call per concept, bounded concurrency
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_ROW_CALLS)
    concept_evidence = {
        c.name: select_evidence_for_concept(c, evidence_store, allowed_ids)
        for c in outline.concepts
    }

    async def _bounded_row(concept: ConceptOutline, feedback: list[str] | None = None):
        async with semaphore:
            return await _stage_b_row(
                client, model, hint, concept, columns,
                concept_evidence[concept.name], feedback=feedback,
            )

    row_results = await asyncio.gather(*(_bounded_row(c) for c in outline.concepts))
    rows: list[DraftRow] = [r for r in row_results if r is not None]

    if debug_trace is not None:
        debug_trace.append({
            "stage": "concept_workflow_rows",
            "requested": len(outline.concepts),
            "produced": len(rows),
        })

    # Stage C — critic + targeted rewrites
    if rows:
        flagged = await _stage_c_critic(client, model, rows, outlines_by_name)
        rewrite_labels = list(flagged.keys())[:_MAX_CRITIC_REWRITES]
        if rewrite_labels:
            label_to_concept = {
                r.row_label: outlines_by_name.get(r.row_label) for r in rows
            }
            rewritten = await asyncio.gather(*(
                _bounded_row(label_to_concept[label], feedback=flagged[label])
                for label in rewrite_labels
                if label_to_concept.get(label) is not None
            ))
            by_label = {r.row_label: r for r in rewritten if r is not None}
            rows = [by_label.get(r.row_label, r) for r in rows]
        if debug_trace is not None:
            debug_trace.append({
                "stage": "concept_workflow_critic",
                "flagged_rows": rewrite_labels,
                "issues": flagged,
            })

    # Stage D — deterministic narrative ordering
    rows = order_rows(rows, outline.row_order)

    return DraftDataTable(headers=columns, rows=rows, notes=[])
