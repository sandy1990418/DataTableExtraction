from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings
from app.prompts import PPT_PLANNER_SYSTEM

logger = logging.getLogger(__name__)


def _catalog_entry(t: dict) -> dict:
    """Lightweight table descriptor for outline selection. Accepts either a plan
    spec (with `columns`) or a populated table (with `headers`/`rows`)."""
    entry = {"name": t.get("name"), "title": t.get("title"), "description": t.get("description")}
    columns = t.get("columns")
    if columns:
        entry["columns"] = [c.get("name") if isinstance(c, dict) else c for c in columns]
    elif t.get("headers"):
        entry["columns"] = t["headers"]
    if t.get("rows"):
        entry["row_count"] = len(t["rows"])
    return entry


async def generate_ppt_plan(
    table_catalog: list[dict],
    evidence_summary: str,
    settings: Settings,
    presentation_hint: str = "",
    n_slides: int | None = None,
) -> dict[str, Any]:
    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
    )

    # Only the lightweight catalog is sent at outline time — no rows. The model
    # picks which tables deserve a slide via table_ref; rows are filled later.
    table_summary = json.dumps(
        [_catalog_entry(t) for t in table_catalog],
        ensure_ascii=False,
        indent=2,
    )

    user_content = f"Evidence summary:\n{evidence_summary}\n\nCanonical tables available:\n{table_summary}"
    if n_slides:
        user_content = f"Target slide count: {n_slides}\n\n{user_content}"
    if presentation_hint:
        user_content = f"Presentation goal: {presentation_hint}\n\n{user_content}"

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": PPT_PLANNER_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=2048,
            temperature=0.3,
        )
        content = response.choices[0].message.content or "{}"
        text = content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("ppt_planner JSON parse error: %s", exc)
        return {"presentation_title": "Presentation", "slides": []}
    except Exception as exc:
        logger.error("ppt_planner failed: %s", exc, exc_info=True)
        return {"presentation_title": "Presentation", "slides": []}
