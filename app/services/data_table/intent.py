"""Stage 2: detect table intent and select pipeline strategy."""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from app.config import Settings
from app.models.data_table import EvidenceBlock
from app.prompts.data_table import TABLE_INTENT_SYSTEM
from app.services.data_table.strategy import select_strategy, strategy_to_row_discovery_mode

logger = logging.getLogger(__name__)

_VALID_TABLE_KINDS = {
    "comparison",
    "timeline",
    "action_items",
    "entity_attribute",
    "experiment_results",
    "benchmark_results",
    "literature_review",
    "generic",
}

_FALLBACK = {
    "intent": "Extract structured information from the provided sources.",
    "table_kind": "generic",
    "row_grain": "unknown",
    "expected_columns": [],
    "row_discovery_mode": "primary_subjects",
    "strategy": "synthesized_entity_comparison",
    "notes": "Fallback: LLM intent detection failed.",
}


def _evidence_summary(evidence_store: list[EvidenceBlock], max_items: int = 15) -> str:
    lines = []
    for block in evidence_store[:max_items]:
        preview = (block.text or "")[:200].replace("\n", " ")
        table_hint = f" [table headers: {block.table_markdown.splitlines()[0][:80]}]" if block.table_markdown else ""
        lines.append(f"- [{block.evidence_id}] ({block.kind}) {block.title or ''}: {preview}{table_hint}")
    return "\n".join(lines) or "(no evidence)"


async def detect_table_intent(
    hint: str,
    evidence_store: list[EvidenceBlock],
    settings: Settings,
    debug_trace: list | None = None,
) -> dict:
    """Detect table intent and select strategy based on hint + evidence shape."""
    # evidence-aware strategy selection (deterministic, no LLM)
    strategy, score_report = select_strategy(hint, evidence_store, debug_trace=debug_trace)
    row_discovery_mode = strategy_to_row_discovery_mode(strategy)

    # LLM call for table_kind / intent / row_grain description (not for routing)
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
        result = {}

    # validate LLM output
    if result.get("table_kind") not in _VALID_TABLE_KINDS:
        result["table_kind"] = "generic"

    result.setdefault("intent", hint or _FALLBACK["intent"])
    result.setdefault("row_grain", "unknown")
    result.setdefault("expected_columns", [])
    result.setdefault("notes", "")

    # strategy and row_discovery_mode come from the evidence-aware scorer, not LLM
    result["strategy"] = strategy
    result["row_discovery_mode"] = row_discovery_mode
    result["strategy_scores"] = score_report

    return result
