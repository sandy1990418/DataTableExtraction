from __future__ import annotations

import asyncio
import json
import logging

from openai import AsyncOpenAI

from app.config import Settings

logger = logging.getLogger(__name__)

IMAGE_SYSTEM = """You are a visual data extraction assistant. Analyze the provided image and return a JSON object with these fields:
- "type": one of "table", "chart", "diagram", "screenshot", "other"
- "title": short descriptive title
- "caption": 1-2 sentence description of what this image shows
- "insight": key insight or finding (for charts/diagrams), or empty string
- "table": if type is "table" or "screenshot" containing a table, include {"headers": [...], "rows": [[...], ...]}, otherwise null

Return only valid JSON, no markdown fences."""


async def analyze_image(data_b64: str, alt: str, settings: Settings) -> dict:
    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
    )
    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": IMAGE_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Image alt text: {alt or 'not provided'}. Analyze this image."},
                        {"type": "image_url", "image_url": {"url": data_b64}},
                    ],
                },
            ],
            max_completion_tokens=1024,
            temperature=0.1,
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
    except Exception as exc:
        logger.warning("Image analysis failed for %r: %s", alt, exc)
        return {"type": "other", "title": alt, "caption": alt, "insight": "", "table": None}


async def analyze_images_batch(images: list[dict], settings: Settings) -> list[dict]:
    tasks = [analyze_image(img["data_b64"], img["alt"], settings) for img in images]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out = []
    for r in results:
        if isinstance(r, Exception):
            out.append({"type": "other", "caption": "", "insight": "", "table": None})
        else:
            out.append(r)
    return out
