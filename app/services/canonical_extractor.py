from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings
from app.prompts import PLAN_SYSTEM, POPULATE_SYSTEM
from app.services.evidence_layer import EvidenceItem, summarize_evidence

logger = logging.getLogger(__name__)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _evidence_to_prompt(items: list[EvidenceItem], content_limit: int = 4000) -> str:
    parts = [summarize_evidence(items), "", "Full evidence details:", ""]
    for i, item in enumerate(items, 1):
        parts.append(f"### Evidence {i}: [{item.kind}] {item.title}")
        parts.append(f"Source: {item.source_ref}")
        if item.headers:
            parts.append(f"Headers: {', '.join(item.headers)}")
            for row in item.rows[:40]:
                parts.append(f"  Row: {', '.join(str(c) for c in row)}")
            if len(item.rows) > 40:
                parts.append(f"  ... ({len(item.rows) - 40} more rows)")
        else:
            text = item.content[:content_limit]
            if len(item.content) > content_limit:
                text += " …[truncated]"
            parts.append(f"Content: {text}")
        parts.append("")
    return "\n".join(parts)


def _evidence_overview(items: list[EvidenceItem]) -> str:
    """Lighter view used for planning: titles, sources, and table catalogs only."""
    parts = [summarize_evidence(items)]
    return "\n".join(parts)


async def _discover_table_plan(
    client: AsyncOpenAI,
    items: list[EvidenceItem],
    settings: Settings,
    hint: str,
) -> list[dict[str, Any]]:
    prompt = _evidence_overview(items)
    if hint:
        prompt = f"Report goal: {hint}\n\n{prompt}"

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": PLAN_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=settings.MAX_TOKENS,
            temperature=settings.TEMPERATURE,
        )
        content = response.choices[0].message.content or "{}"
        result = json.loads(_strip_fences(content))
        return result.get("tables", [])
    except json.JSONDecodeError as exc:
        logger.error("table-plan JSON parse error: %s", exc)
        return []
    except Exception as exc:
        logger.error("table-plan discovery failed: %s", exc, exc_info=True)
        return []


def _spec_to_prompt(spec: dict[str, Any], evidence_block: str) -> str:
    cols = spec.get("columns", [])
    col_lines = []
    for c in cols:
        col_lines.append(
            f'- "{c.get("name")}": {c.get("description", "")} (e.g. {c.get("example", "")})'
        )
    return (
        f"Table: {spec.get('title')}\n"
        f"Description: {spec.get('description', '')}\n"
        f"Each row represents: {spec.get('row_entity', 'one entity')}\n\n"
        f"Columns (use these exact headers, in this order):\n"
        + "\n".join(col_lines)
        + "\n\n=== EVIDENCE ===\n"
        + evidence_block
    )


async def _populate_table(
    client: AsyncOpenAI,
    spec: dict[str, Any],
    evidence_block: str,
    settings: Settings,
) -> dict[str, Any] | None:
    headers = [c.get("name", "") for c in spec.get("columns", [])]
    if not headers:
        return None

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": POPULATE_SYSTEM},
                {"role": "user", "content": _spec_to_prompt(spec, evidence_block)},
            ],
            max_completion_tokens=settings.MAX_TOKENS,
            temperature=settings.TEMPERATURE,
        )
        content = response.choices[0].message.content or "{}"
        result = json.loads(_strip_fences(content))
    except json.JSONDecodeError as exc:
        logger.error("populate JSON parse error for %r: %s", spec.get("name"), exc)
        return None
    except Exception as exc:
        logger.error("populate failed for %r: %s", spec.get("name"), exc, exc_info=True)
        return None

    rows = result.get("rows", [])
    # Trust the spec's headers over whatever the model echoes back.
    out_headers = result.get("headers") or headers
    if len(out_headers) != len(headers):
        out_headers = headers
    if not rows:
        return None

    return {
        "name": spec.get("name", ""),
        "title": spec.get("title", spec.get("name", "Table")),
        "description": spec.get("description", ""),
        "headers": out_headers,
        "rows": rows,
    }


async def extract_canonical_tables(
    items: list[EvidenceItem],
    settings: Settings,
    hint: str = "",
) -> list[dict[str, Any]]:
    """Two-stage extraction: discover the table plan (facets), then populate each
    table from the full evidence. This produces human-meaningful comparison tables
    instead of generic one-shot output."""
    if not items:
        return []

    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
    )

    # Stage 1: plan
    plan = await _discover_table_plan(client, items, settings, hint)
    if not plan:
        return []

    # Stage 2: populate each planned table in parallel against the full evidence
    evidence_block = _evidence_to_prompt(items)
    populated = await asyncio.gather(
        *[_populate_table(client, spec, evidence_block, settings) for spec in plan]
    )
    return [t for t in populated if t]
