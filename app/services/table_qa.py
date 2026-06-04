from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings
from app.prompts import TABLE_QA_SYSTEM

logger = logging.getLogger(__name__)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _review_prompt(table: dict, evidence_block: str) -> str:
    payload = {
        "table": {
            "table_id": table.get("table_id") or table.get("name") or table.get("title"),
            "title": table.get("title"),
            "headers": table.get("headers", []),
            "rows": table.get("rows", []),
            "summary": table.get("summary", ""),
            "source_ref": table.get("source_ref", ""),
        },
        "evidence": evidence_block,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _evidence_for_table(table: dict, evidence_block: str | dict[str, str]) -> str:
    if not isinstance(evidence_block, dict):
        return evidence_block
    table_id = str(table.get("table_id") or table.get("name") or table.get("title") or "")
    return evidence_block.get(table_id, "") or evidence_block.get("", "")


async def _review_table(
    client: AsyncOpenAI,
    table: dict,
    evidence_block: str | dict[str, str],
    settings: Settings,
) -> dict[str, Any]:
    table_id = table.get("table_id") or table.get("name") or table.get("title") or "table"
    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": TABLE_QA_SYSTEM},
                {"role": "user", "content": _review_prompt(table, _evidence_for_table(table, evidence_block))},
            ],
            max_completion_tokens=min(settings.MAX_TOKENS, 2048),
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        result = json.loads(_strip_fences(content))
    except json.JSONDecodeError as exc:
        logger.warning("table QA JSON parse error for %r: %s", table_id, exc)
        return {
            "table_id": table_id,
            "status": "needs_revision",
            "warnings": ["QA reviewer returned invalid JSON."],
            "unsupported_cells": [],
        }
    except Exception as exc:
        logger.warning("table QA failed for %r: %s", table_id, exc, exc_info=True)
        return {
            "table_id": table_id,
            "status": "needs_revision",
            "warnings": [f"QA reviewer failed: {exc}"],
            "unsupported_cells": [],
        }

    warnings = result.get("warnings") or []
    unsupported = result.get("unsupported_cells") or []
    status = result.get("status") or ("needs_revision" if warnings or unsupported else "pass")
    return {
        "table_id": result.get("table_id") or table_id,
        "status": status if status in {"pass", "needs_revision"} else "needs_revision",
        "warnings": [str(w) for w in warnings],
        "unsupported_cells": unsupported if isinstance(unsupported, list) else [],
    }


async def review_tables(
    tables: list[dict],
    evidence_block: str | dict[str, str],
    settings: Settings,
) -> list[dict[str, Any]]:
    """Run an LLM reviewer agent over each populated table."""
    if not tables:
        return []
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL or None)
    return await asyncio.gather(
        *[_review_table(client, table, evidence_block, settings) for table in tables]
    )


def review_warnings(reviews: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for review in reviews:
        table_id = review.get("table_id") or "table"
        for warning in review.get("warnings", []):
            warnings.append(f"QA {table_id}: {warning}")
        unsupported = review.get("unsupported_cells") or []
        if unsupported:
            warnings.append(f"QA {table_id}: {len(unsupported)} unsupported cells flagged.")
    return warnings
