from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings
from app.services.evidence_layer import EvidenceItem, summarize_evidence

logger = logging.getLogger(__name__)

# ── Stage 1: table-plan (facet) discovery ────────────────────────────────────
# Instead of one-shot "extract tables", we first ask the model — anchored to the
# ACTUAL collection — what comparison tables a domain expert would actually build,
# and what the rows and columns should be. This avoids generic template columns
# like "Summary" / "Main contribution" and produces human-meaningful tables.

PLAN_SYSTEM = """You are a senior research analyst planning comparison tables for a report.

You are given evidence distilled from multiple documents. Your job is NOT to fill in tables yet — it is to decide which tables a domain expert would actually want, and design their structure.

Think like a human analyst:
- What are the natural ROW entities here? (e.g. the papers/methods being compared, the benchmarks, the open issues)
- What COLUMNS would let a reader compare those entities at a glance? Pick dimensions that actually vary across rows and that matter for decisions — not generic filler.
- Avoid vague columns like "Summary", "Main Contribution", "Notes". Prefer specific, comparable dimensions (e.g. "Storage Backend", "Retrieval Method", "Forgetting/Update Rule", "Benchmark", "Reported Metric").

Return JSON:
{
  "tables": [
    {
      "name": "snake_case_id",
      "title": "Human readable title",
      "description": "What a reader learns from this table",
      "row_entity": "what each row represents (e.g. 'one paper/method')",
      "columns": [
        {"name": "Column header", "description": "what goes in this cell", "example": "a concrete example value"}
      ]
    }
  ]
}

Rules:
- Propose 2-5 tables, ordered by usefulness.
- Every column needs a concrete example so the populator knows the expected granularity.
- Only propose tables that the evidence can actually support."""

# ── Stage 2: example-anchored population ─────────────────────────────────────

POPULATE_SYSTEM = """You are a meticulous research data extractor.

You are given (a) ONE table specification with named columns, descriptions and example values, and (b) the full evidence layer. Populate the table.

Rules:
- Use EXACTLY the column headers given in the spec, in order. Do not invent or rename columns.
- One row per distinct row-entity. Match the granularity shown by the column examples.
- Fill every cell from the evidence. If a value is genuinely not present, use "-" (never guess or hallucinate).
- Keep cells concise (a phrase or short value), not paragraphs.
- Preserve numbers/metrics exactly as written in the evidence.

Return JSON: {"headers": [...], "rows": [[...], ...]}"""


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
