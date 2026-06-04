from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings
from app.models import EvidenceItem
from app.prompts import PLAN_SYSTEM, POPULATE_SYSTEM
from app.services.evidence_layer import summarize_evidence

logger = logging.getLogger(__name__)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _evidence_content_limit(item: EvidenceItem, default_limit: int) -> int:
    title = item.title.lower()
    if title.startswith(("table ", "figure ", "fig.")):
        return 5000
    if any(keyword in title for keyword in ("result", "experiment", "evaluation", "benchmark")):
        return 3000
    return default_limit


def _evidence_to_prompt(items: list[EvidenceItem], content_limit: int = 1400) -> str:
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
            limit = _evidence_content_limit(item, content_limit)
            text = item.content[:limit]
            if len(item.content) > limit:
                text += " …[truncated]"
            parts.append(f"Content: {text}")
        parts.append("")
    return "\n".join(parts)


def _planning_evidence_to_prompt(
    items: list[EvidenceItem],
    max_items: int = 28,
    max_chars: int = 60000,
) -> str:
    """Table-aware evidence view used for schema planning.

    Stage 1 is where the table grain is decided. A title-only overview makes the
    model invent generic schemas, so planning gets the most table/metric/method
    heavy snippets with their actual row-like text preserved.
    """
    parts = [
        summarize_evidence(items),
        "",
        "Detailed evidence for table planning:",
        "",
    ]

    scored = sorted(
        ((_planning_score_item(item), index, item) for index, item in enumerate(items)),
        key=lambda entry: (-entry[0], entry[1]),
    )
    selected = [item for score, _, item in scored if score > 0][:max_items]
    if len(selected) < min(8, len(items)):
        selected.extend(item for _, _, item in scored if item not in selected)
        selected = selected[:max_items]

    char_count = sum(len(part) for part in parts)
    for item_index, item in enumerate(selected, 1):
        block = _format_planning_item(item_index, item)
        if char_count + len(block) > max_chars:
            break
        parts.append(block)
        char_count += len(block)

    return "\n".join(parts)


def _planning_score_item(item: EvidenceItem) -> float:
    title = item.title.lower()
    content = item.content.lower()
    score = 0.0

    if item.kind in {"markdown_table", "image_table"}:
        score += 14
    if title.startswith("table "):
        score += 14
    if title.startswith(("figure ", "fig.")):
        score += 4
    if any(
        keyword in title or keyword in content[:1000]
        for keyword in (
            "benchmark",
            "dataset",
            "metric",
            "metrics",
            "accuracy",
            "f1",
            "bleu",
            "rouge",
            "meteor",
            "sbert",
            "result",
            "results",
            "evaluation",
            "experiment",
            "ablation",
            "efficiency",
            "retrieval time",
        )
    ):
        score += 5
    if any(
        keyword in title or keyword in content[:1000]
        for keyword in (
            "architecture",
            "methodology",
            "method",
            "memory",
            "storage",
            "retrieval",
            "update",
            "evolution",
        )
    ):
        score += 3
    if not item.content.strip() and not item.rows:
        return 0
    return score


def _format_planning_item(index: int, item: EvidenceItem) -> str:
    lines = [f"### Evidence {index}: [{item.kind}] {item.title}", f"Source: {item.source_ref}"]
    if item.headers:
        lines.append(f"Headers: {', '.join(item.headers)}")
        for row in item.rows[:40]:
            lines.append(f"  Row: {', '.join(str(c) for c in row)}")
        if len(item.rows) > 40:
            lines.append(f"  ... ({len(item.rows) - 40} more rows)")
    else:
        limit = _planning_content_limit(item)
        text = item.content[:limit]
        if len(item.content) > limit:
            text += " ...[truncated]"
        lines.append(f"Content: {text}")
    lines.append("")
    return "\n".join(lines)


def _planning_content_limit(item: EvidenceItem) -> int:
    title = item.title.lower()
    if title.startswith("table "):
        return 5000
    if title.startswith(("figure ", "fig.")):
        return 2200
    if any(keyword in title for keyword in ("result", "experiment", "evaluation", "benchmark")):
        return 2600
    if any(keyword in title for keyword in ("method", "architecture", "memory", "retrieval")):
        return 1800
    return 1200


async def _discover_table_plan(
    client: AsyncOpenAI,
    items: list[EvidenceItem],
    settings: Settings,
    hint: str,
) -> list[dict[str, Any]]:
    prompt = _planning_evidence_to_prompt(items)
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


def _evidence_for_spec(spec: dict[str, Any], evidence_block: str | dict[str, str]) -> str:
    if isinstance(evidence_block, dict):
        return evidence_block.get(str(spec.get("name")), "") or evidence_block.get("", "")
    return evidence_block


async def _populate_table(
    client: AsyncOpenAI,
    spec: dict[str, Any],
    evidence_block: str | dict[str, str],
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
                {"role": "user", "content": _spec_to_prompt(spec, _evidence_for_spec(spec, evidence_block))},
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


def _client(settings: Settings) -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL or None)


# ── Public two-stage API (call separately for lazy population) ────────────────


async def plan_tables(
    items: list[EvidenceItem],
    settings: Settings,
    hint: str = "",
) -> tuple[list[dict[str, Any]], str]:
    """Stage 1 (cheap): discover table specs (facets) from the evidence.

    Returns (specs, evidence_block). `evidence_block` is the full rendered evidence
    that populate_tables needs later — cache it so population can run lazily without
    re-rendering. No rows are produced here.
    """
    if not items:
        return [], ""
    specs = await _discover_table_plan(_client(settings), items, settings, hint)
    evidence_block = _evidence_to_prompt(items)
    return specs, evidence_block


async def populate_tables(
    specs: list[dict[str, Any]],
    evidence_block: str | dict[str, str],
    settings: Settings,
) -> list[dict[str, Any]]:
    """Stage 2 (expensive): fill the given specs with rows from the evidence, in
    parallel. Call this ONLY for the tables you actually need."""
    if not specs:
        return []
    client = _client(settings)
    populated = await asyncio.gather(
        *[_populate_table(client, spec, evidence_block, settings) for spec in specs]
    )
    return [t for t in populated if t]


async def extract_canonical_tables(
    items: list[EvidenceItem],
    settings: Settings,
    hint: str = "",
) -> list[dict[str, Any]]:
    """Eager two-stage extraction: plan + populate ALL tables. Prefer the lazy path
    (plan_tables then populate_tables for only referenced tables) when you don't need
    every table rendered."""
    specs, evidence_block = await plan_tables(items, settings, hint)
    return await populate_tables(specs, evidence_block, settings)
