from __future__ import annotations

import asyncio
import json
import logging

from openai import AsyncOpenAI

from app.config import Settings
from app.prompts.image import IMAGE_SYSTEM

logger = logging.getLogger(__name__)


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
                        {
                            "type": "text",
                            "text": f"Image alt text: {alt or 'not provided'}. Analyze this image.",
                        },
                        {"type": "image_url", "image_url": {"url": data_b64}},
                    ],
                },
            ],
            # Table screenshots can have many rows; 1024 tokens truncates the JSON
            # mid-table and the result is silently dropped as unparseable.
            max_completion_tokens=min(settings.MAX_TOKENS, 4096),
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
