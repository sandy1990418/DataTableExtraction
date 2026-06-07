"""Stage 2: detect what kind of data table the user wants."""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from app.config import Settings
from app.models.data_table import EvidenceBlock
from app.prompts.data_table import TABLE_INTENT_SYSTEM

logger = logging.getLogger(__name__)

_VALID_TABLE_KINDS = {
    "comparison",
    "timeline",
    "action_items",
    "entity_attribute",
    "experiment_results",
    "literature_review",
    "generic",
}

_FALLBACK = {
    "intent": "Extract structured information from the provided sources.",
    "table_kind": "generic",
    "row_grain": "unknown",
    "expected_columns": [],
    "notes": "Fallback: LLM intent detection failed.",
}


def _evidence_summary(evidence_store: list[EvidenceBlock], max_items: int = 15) -> str:
    lines = []
    for block in evidence_store[:max_items]:
        preview = (block.text or "")[:200].replace("\n", " ")
        lines.append(f"- [{block.evidence_id}] ({block.kind}) {block.title or ''}: {preview}")
    return "\n".join(lines) or "(no evidence)"


async def detect_table_intent(hint: str, evidence_store: list[EvidenceBlock], settings: Settings) -> dict:
    """Use LLM to detect the table kind and expected shape."""
    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
    )

    user_msg = (
        f"User hint: {hint}\n\n"
        f"Evidence summary:\n{_evidence_summary(evidence_store)}\n\n"
        "Return JSON only."
    )

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": TABLE_INTENT_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_completion_tokens=512,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        result = json.loads(raw)
    except Exception as exc:
        logger.warning("detect_table_intent LLM call failed: %s — using fallback", exc)
        return dict(_FALLBACK)

    # validate / coerce
    if result.get("table_kind") not in _VALID_TABLE_KINDS:
        result["table_kind"] = "generic"
    result.setdefault("intent", hint or _FALLBACK["intent"])
    result.setdefault("row_grain", "unknown")
    result.setdefault("expected_columns", [])
    result.setdefault("notes", "")
    return result
