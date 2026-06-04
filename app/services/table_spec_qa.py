from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from app.config import Settings
from app.prompts import TABLE_SPEC_QA_SYSTEM, TABLE_SPEC_REVISION_SYSTEM

logger = logging.getLogger(__name__)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _spec_review_prompt(specs: list[dict], hint: str, evidence_block: str) -> str:
    return json.dumps(
        {
            "goal": hint,
            "table_specs": specs,
            "focused_evidence": evidence_block,
        },
        ensure_ascii=False,
        indent=2,
    )


def _spec_revision_prompt(
    specs: list[dict],
    reviews: list[dict],
    hint: str,
    evidence_block: str,
) -> str:
    return json.dumps(
        {
            "goal": hint,
            "table_specs": specs,
            "reviews": reviews,
            "focused_evidence": evidence_block,
        },
        ensure_ascii=False,
        indent=2,
    )


def _normalize_reviews(specs: list[dict], result: dict[str, Any]) -> list[dict[str, Any]]:
    by_name = {str(review.get("name")): review for review in result.get("reviews", [])}
    reviews: list[dict[str, Any]] = []
    for spec in specs:
        name = str(spec.get("name") or spec.get("title") or "table")
        review = by_name.get(name, {})
        warnings = review.get("warnings") or []
        unsupported = review.get("unsupported_columns") or []
        status = review.get("status") or ("needs_revision" if warnings or unsupported else "pass")
        reviews.append(
            {
                "name": name,
                "status": status if status in {"pass", "needs_revision"} else "needs_revision",
                "warnings": [str(warning) for warning in warnings],
                "unsupported_columns": unsupported if isinstance(unsupported, list) else [],
            }
        )
    return reviews


async def review_table_specs(
    specs: list[dict],
    hint: str,
    evidence_block: str,
    settings: Settings,
) -> list[dict[str, Any]]:
    if not specs:
        return []

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL or None)
    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": TABLE_SPEC_QA_SYSTEM},
                {"role": "user", "content": _spec_review_prompt(specs, hint, evidence_block)},
            ],
            max_completion_tokens=min(settings.MAX_TOKENS, 4096),
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        result = json.loads(_strip_fences(content))
        return _normalize_reviews(specs, result)
    except json.JSONDecodeError as exc:
        logger.warning("table spec QA JSON parse error: %s", exc)
        return [
            {
                "name": str(spec.get("name") or spec.get("title") or "table"),
                "status": "needs_revision",
                "warnings": ["Spec reviewer returned invalid JSON."],
                "unsupported_columns": [],
            }
            for spec in specs
        ]
    except Exception as exc:
        logger.warning("table spec QA failed: %s", exc, exc_info=True)
        return [
            {
                "name": str(spec.get("name") or spec.get("title") or "table"),
                "status": "pass",
                "warnings": [f"Spec reviewer failed: {exc}"],
                "unsupported_columns": [],
            }
            for spec in specs
        ]


async def revise_table_specs(
    specs: list[dict],
    reviews: list[dict],
    hint: str,
    evidence_block: str,
    settings: Settings,
) -> list[dict[str, Any]]:
    if not specs:
        return []

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_BASE_URL or None)
    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": TABLE_SPEC_REVISION_SYSTEM},
                {"role": "user", "content": _spec_revision_prompt(specs, reviews, hint, evidence_block)},
            ],
            max_completion_tokens=settings.MAX_TOKENS,
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        result = json.loads(_strip_fences(content))
    except json.JSONDecodeError as exc:
        logger.warning("table spec revision JSON parse error: %s", exc)
        return specs
    except Exception as exc:
        logger.warning("table spec revision failed: %s", exc, exc_info=True)
        return specs

    revised = result.get("tables") or []
    return revised if isinstance(revised, list) and revised else specs


def spec_review_warnings(reviews: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for review in reviews:
        name = review.get("name") or "table_spec"
        for warning in review.get("warnings", []):
            warnings.append(f"Spec QA {name}: {warning}")
        unsupported = review.get("unsupported_columns") or []
        if unsupported:
            warnings.append(f"Spec QA {name}: {len(unsupported)} unsupported columns flagged.")
    return warnings
