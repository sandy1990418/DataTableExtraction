"""Stage 3: induce a DataTableSchema from hint + intent + evidence."""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from app.config import Settings
from app.models.data_table import DataTableColumn, DataTableSchema, EvidenceBlock
from app.prompts.data_table import SCHEMA_INDUCTION_SYSTEM

logger = logging.getLogger(__name__)

_DEFAULT_COLUMNS = [
    DataTableColumn(name="Entity", role="entity", description="The subject being compared.", required=True),
    DataTableColumn(name="Description", role="attribute", description="Brief description from sources."),
    DataTableColumn(name="Key Feature", role="attribute", description="Notable characteristic."),
]


def _evidence_sample(evidence_store: list[EvidenceBlock], max_items: int = 10) -> str:
    lines = []
    for b in evidence_store[:max_items]:
        preview = (b.text or "")[:300].replace("\n", " ")
        lines.append(f"[{b.evidence_id}] {b.title or ''}: {preview}")
    return "\n".join(lines) or "(no evidence)"


def _parse_columns(raw_columns: list[dict]) -> list[DataTableColumn]:
    seen_names: set[str] = set()
    columns: list[DataTableColumn] = []
    for col in raw_columns:
        name = str(col.get("name", "")).strip()
        if not name or name in seen_names:
            continue
        seen_names.add(name)
        try:
            columns.append(DataTableColumn.model_validate(col))
        except Exception:
            pass
    return columns


async def induce_schema(
    hint: str,
    table_intent: dict,
    evidence_store: list[EvidenceBlock],
    settings: Settings,
    max_columns: int = 6,
) -> DataTableSchema:
    """Call LLM to design table columns, return validated DataTableSchema."""
    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
    )

    user_msg = (
        f"User hint: {hint}\n\n"
        f"Detected intent: {json.dumps(table_intent, ensure_ascii=False)}\n\n"
        f"Evidence sample:\n{_evidence_sample(evidence_store)}\n\n"
        f"Max columns: {max_columns}\n\n"
        "Return JSON only."
    )

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SCHEMA_INDUCTION_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_completion_tokens=1024,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
    except Exception as exc:
        logger.warning("induce_schema LLM call failed: %s — using defaults", exc)
        return DataTableSchema(
            title="Data Table",
            intent=hint or "Extract structured data from sources.",
            columns=_DEFAULT_COLUMNS,
        )

    columns = _parse_columns(data.get("columns", []))

    # ensure min 3 columns
    if len(columns) < 3:
        columns = _DEFAULT_COLUMNS

    # cap at max_columns
    columns = columns[:max_columns]

    # ensure at least one entity column
    if not any(c.role == "entity" for c in columns):
        columns[0] = DataTableColumn(
            name=columns[0].name,
            role="entity",
            description=columns[0].description,
            value_type=columns[0].value_type,
            required=True,
        )

    return DataTableSchema(
        title=str(data.get("title", "Data Table")).strip() or "Data Table",
        intent=str(data.get("intent", hint)).strip() or hint,
        columns=columns,
    )
