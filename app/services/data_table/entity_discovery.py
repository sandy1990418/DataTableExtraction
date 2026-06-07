"""Stage 4: discover row entities from evidence (does NOT generate table cells)."""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from app.config import Settings
from app.models.data_table import DataTableSchema, EvidenceBlock, RowEntity, SourceRef
from app.prompts.data_table import ENTITY_DISCOVERY_SYSTEM

logger = logging.getLogger(__name__)


def _evidence_for_discovery(evidence_store: list[EvidenceBlock], max_items: int = 20) -> str:
    lines = []
    for b in evidence_store[:max_items]:
        text = (b.text or "")[:400].replace("\n", " ")
        table_hint = f" [table: {b.table_markdown[:100]}]" if b.table_markdown else ""
        doc = f"document={b.document_name}" if b.document_name else ""
        lines.append(f"[{b.evidence_id}] ({b.kind}) {doc} title={b.title or ''}: {text}{table_hint}")
    return "\n".join(lines) or "(no evidence)"


def _build_row_entity(raw: dict, evidence_index: dict[str, EvidenceBlock]) -> RowEntity | None:
    name = str(raw.get("name", "")).strip()
    if not name:
        return None

    entity_id = str(raw.get("entity_id", f"ent_{name[:8]}")).strip()
    aliases = [str(a) for a in raw.get("aliases", []) if str(a).strip()]
    description = raw.get("description") or None
    confidence = float(raw.get("confidence", 0.5))

    source_refs: list[SourceRef] = []
    for ev_id in raw.get("source_refs", []):
        block = evidence_index.get(str(ev_id))
        if block:
            source_refs.append(block.source_ref)

    if not source_refs:
        confidence = min(confidence, 0.3)

    return RowEntity(
        entity_id=entity_id,
        name=name,
        aliases=aliases,
        description=description,
        source_refs=source_refs,
        confidence=confidence,
    )


def _dedup_entities(entities: list[RowEntity]) -> list[RowEntity]:
    """Merge entities with the same canonical name (case-insensitive)."""
    seen: dict[str, RowEntity] = {}
    for ent in entities:
        key = ent.name.lower()
        if key not in seen:
            seen[key] = ent
        else:
            existing = seen[key]
            # merge aliases
            merged_aliases = list({*existing.aliases, *ent.aliases, ent.name})
            merged_aliases = [a for a in merged_aliases if a.lower() != key]
            existing.aliases = merged_aliases
            # merge source refs by evidence_id
            existing_ev_ids = {r.evidence_id for r in existing.source_refs}
            for ref in ent.source_refs:
                if ref.evidence_id not in existing_ev_ids:
                    existing.source_refs.append(ref)
            existing.confidence = max(existing.confidence, ent.confidence)
    return list(seen.values())


async def discover_entities(
    hint: str,
    schema: DataTableSchema,
    evidence_store: list[EvidenceBlock],
    settings: Settings,
    max_rows: int = 20,
) -> list[RowEntity]:
    """Extract candidate row entities from evidence."""
    if not evidence_store:
        logger.warning("discover_entities: empty evidence store")
        return []

    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
    )

    evidence_index = {b.evidence_id: b for b in evidence_store}

    schema_summary = (
        f"Table title: {schema.title}\n"
        f"Intent: {schema.intent}\n"
        f"Row grain: what each row should represent (infer from intent)\n"
        f"Columns: {', '.join(c.name for c in schema.columns)}"
    )

    doc_names = sorted({b.document_name for b in evidence_store if b.document_name})
    doc_list = ", ".join(doc_names) if doc_names else "unknown"

    user_msg = (
        f"User hint: {hint}\n\n"
        f"Source documents (these are the PRIMARY subjects): {doc_list}\n\n"
        f"Table schema:\n{schema_summary}\n\n"
        f"Evidence:\n{_evidence_for_discovery(evidence_store)}\n\n"
        f"Max rows: {max_rows}\n\n"
        "IMPORTANT: Only extract entities that are the PRIMARY subject of one of the source documents listed above. "
        "Do NOT extract entities that are only mentioned as baselines, citations, or prior work.\n\n"
        "Return JSON only. Return an object with 'entities' (list) and optionally 'warning' (string)."
    )

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": ENTITY_DISCOVERY_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_completion_tokens=2048,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
    except Exception as exc:
        logger.warning("discover_entities LLM call failed: %s", exc)
        return []

    raw_entities = data.get("entities", [])
    entities: list[RowEntity] = []
    for raw in raw_entities[:max_rows]:
        ent = _build_row_entity(raw, evidence_index)
        if ent:
            entities.append(ent)

    entities = _dedup_entities(entities)

    if not entities:
        logger.warning("discover_entities: no entities found")

    return entities
