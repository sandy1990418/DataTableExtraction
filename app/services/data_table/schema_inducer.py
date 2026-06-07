"""Stage 3: induce a DataTableSchema from hint + intent + evidence."""

from __future__ import annotations

import json
import logging
import re

from openai import AsyncOpenAI

from app.config import Settings
from app.models.data_table import DataTableColumn, DataTableSchema, EvidenceBlock
from app.prompts.data_table import SCHEMA_INDUCTION_SYSTEM
from app.services.data_table.source_table_rows import extract_source_table_candidates

logger = logging.getLogger(__name__)

_RESULT_TABLE_KINDS = {"benchmark_results", "experiment_results"}
_NUMERIC_RE = re.compile(r"^\d")

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


def _infer_value_type(header: str, sample_values: list[str]) -> str:
    """Guess value_type from header name and sample cell values."""
    h = header.lower().replace("-", "_").replace(" ", "_")
    if any(num in h for num in ["score", "rouge", "bleu", "f1", "accuracy", "meteor", "sbert", "rate", "ratio"]):
        return "number"
    if sample_values and all(_NUMERIC_RE.match(v.strip()) for v in sample_values if v.strip()):
        return "number"
    return "string"


def induce_schema_from_source_tables(
    hint: str,
    intent: dict,
    evidence_store: list[EvidenceBlock],
    max_columns: int,
) -> DataTableSchema | None:
    """Build schema directly from best matching source table headers.

    Used for benchmark/result mode to avoid LLM hallucinating column names.
    """
    candidates = extract_source_table_candidates(evidence_store, hint)
    if not candidates:
        return None

    best = candidates[0]
    if best.score < 5.0 or not best.headers:
        return None

    columns: list[DataTableColumn] = []
    for i, header in enumerate(best.headers[:max_columns]):
        name = header.strip() or f"col_{i}"
        if not name:
            continue
        # sample values for type inference
        sample = [row[i] for row in best.rows[:3] if i < len(row)]
        role = "entity" if i == 0 else "metric"
        # check if it looks like a category/label column
        if i > 0 and not any(_NUMERIC_RE.match(v.strip()) for v in sample if v.strip()):
            role = "attribute"
        vtype = _infer_value_type(name, sample) if i > 0 else "string"
        columns.append(
            DataTableColumn(
                name=name,
                role=role,
                description=f"{name} from benchmark results.",
                value_type=vtype,
                required=(i == 0),
            )
        )

    if len(columns) < 2:
        return None

    return DataTableSchema(
        title=best.title or "Benchmark Results",
        intent=intent.get("intent", hint),
        columns=columns,
    )


async def induce_schema(
    hint: str,
    table_intent: dict,
    evidence_store: list[EvidenceBlock],
    settings: Settings,
    max_columns: int = 6,
) -> DataTableSchema:
    """Call LLM to design table columns, return validated DataTableSchema."""
    # for benchmark/result mode, use source table headers directly
    _use_source_table = (
        table_intent.get("strategy") in ("source_table_reconstruction", "hybrid_table_synthesis")
        or table_intent.get("table_kind") in _RESULT_TABLE_KINDS
        or table_intent.get("row_discovery_mode") == "source_table_rows"
    )
    if _use_source_table:
        table_schema = induce_schema_from_source_tables(hint, table_intent, evidence_store, max_columns)
        if table_schema:
            return table_schema

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
