from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings
from app.prompts import TABLE_REVISION_SYSTEM

logger = logging.getLogger(__name__)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _evidence_for_table(table: dict, evidence_block: str | dict[str, str]) -> str:
    if not isinstance(evidence_block, dict):
        return evidence_block
    table_id = str(table.get("table_id") or table.get("name") or table.get("title") or "")
    return evidence_block.get(table_id, "") or evidence_block.get("", "")


def _revision_prompt(table: dict, review: dict, evidence_block: str) -> str:
    return json.dumps(
        {
            "table": table,
            "review": review,
            "evidence": evidence_block,
        },
        ensure_ascii=False,
        indent=2,
    )


async def _revise_table(
    client: AsyncOpenAI,
    table: dict,
    review: dict,
    evidence_block: str | dict[str, str],
    settings: Settings,
) -> dict[str, Any]:
    table_id = table.get("table_id") or table.get("name") or table.get("title") or "table"
    headers = table.get("headers") or []
    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": TABLE_REVISION_SYSTEM},
                {
                    "role": "user",
                    "content": _revision_prompt(table, review, _evidence_for_table(table, evidence_block)),
                },
            ],
            max_completion_tokens=settings.MAX_TOKENS,
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        result = json.loads(_strip_fences(content))
    except json.JSONDecodeError as exc:
        logger.warning("table revision JSON parse error for %r: %s", table_id, exc)
        return table
    except Exception as exc:
        logger.warning("table revision failed for %r: %s", table_id, exc, exc_info=True)
        return table

    revised_headers = result.get("headers") or headers
    if revised_headers != headers:
        revised_headers = headers
    rows = result.get("rows") or table.get("rows") or []
    return {
        **table,
        "headers": revised_headers,
        "rows": rows,
    }


async def revise_tables(
    tables: list[dict],
    reviews: list[dict],
    evidence_block: str | dict[str, str],
    settings: Settings,
) -> list[dict[str, Any]]:
    if not tables:
        return []

    review_by_id = {str(review.get("table_id")): review for review in reviews}
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL or None)
    return await asyncio.gather(
        *[
            _revise_table(
                client,
                table,
                review_by_id.get(str(table.get("table_id") or table.get("name") or table.get("title")), {}),
                evidence_block,
                settings,
            )
            for table in tables
        ]
    )
