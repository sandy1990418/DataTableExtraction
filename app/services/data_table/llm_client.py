"""Shared AsyncOpenAI client factory for the data-table pipeline.

Centralizes timeout/retry config: the SDK default is a 600s timeout with 2
retries, which can stall a single pipeline stage for ~30 minutes when the
provider hangs. All data-table stages should create clients through here.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from app.config import Settings


def make_client(settings: Settings) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
        timeout=getattr(settings, "LLM_TIMEOUT_SECONDS", 120.0),
        max_retries=getattr(settings, "LLM_MAX_RETRIES", 1),
    )
