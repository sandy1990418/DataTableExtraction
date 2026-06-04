from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings

logger = logging.getLogger(__name__)

PPT_PLANNER_SYSTEM = """You are a presentation designer. Given a set of canonical tables and document summaries, create a structured PPT slide plan.

For each slide specify:
- slide_number (int)
- title (str)
- slide_type: one of "title", "overview", "table", "chart_summary", "key_findings", "comparison", "timeline", "conclusion"
- content: brief description of what goes on this slide
- table_ref: name of the canonical table to display (if slide_type is "table"), or null
- speaker_notes: optional talking points

Return JSON: {"presentation_title": "...", "slides": [...]}

Guidelines:
- Start with a title slide
- Include an overview/agenda slide
- SELECT only the tables that genuinely deserve their own slide. Not every available table must be used — skip redundant or low-value tables. Reference a chosen table via table_ref (its name).
- A slide that shows a table MUST set table_ref to an existing table name. A slide without a table MUST set table_ref to null.
- Add key findings / insight slides (table_ref null) where prose tells the story better than a table.
- End with a conclusion slide
- Respect the requested slide count when provided."""


async def generate_ppt_plan(
    canonical_tables: list[dict],
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
    # picks which tables deserve a slide; full rows are pulled later at render.
    table_summary = json.dumps(
        [
            {
                "name": t.get("name"),
                "title": t.get("title"),
                "description": t.get("description"),
                "row_count": len(t.get("rows", [])),
            }
            for t in canonical_tables
        ],
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
