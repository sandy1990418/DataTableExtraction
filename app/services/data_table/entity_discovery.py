"""Stage 4: discover row entities from evidence (does NOT generate table cells)."""

from __future__ import annotations

import json
import logging

from openai import AsyncOpenAI

from app.config import Settings
from app.models.data_table import DataTableSchema, EvidenceBlock, RowEntity, SourceRef
from app.prompts.data_table import ENTITY_DISCOVERY_SYSTEM
from app.services.data_table.source_table_rows import (
    _normalize,
    extract_source_table_candidates,
    find_entity_column_index,
)

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


def _discover_entities_from_source_tables(
    evidence_store: list[EvidenceBlock],
    hint: str,
    max_rows: int,
    debug_trace: list | None = None,
) -> list[RowEntity]:
    """Extract row entities from top-scoring source table rows."""
    candidates = extract_source_table_candidates(evidence_store, hint)
    evidence_index = {b.evidence_id: b for b in evidence_store}

    if not candidates:
        return []

    if debug_trace is not None:
        debug_trace.append({
            "stage": "source_table_candidates",
            "candidates": [
                {"evidence_id": c.evidence_id, "headers": c.headers, "score": round(c.score, 2)}
                for c in candidates[:5]
            ],
        })

    # use top candidate as schema reference; merge compatible tables for rows
    best = candidates[0]
    best_headers_norm = [_normalize(h) for h in best.headers]
    entity_col = find_entity_column_index(best.headers)

    # collect compatible tables: same entity col position, overlapping headers
    compatible = [best]
    for cand in candidates[1:]:
        if cand.score < 3.0:
            break
        if find_entity_column_index(cand.headers) != entity_col:
            continue
        cand_norm = [_normalize(h) for h in cand.headers]
        overlap = len(set(best_headers_norm) & set(cand_norm))
        if overlap >= max(2, len(best_headers_norm) // 2):
            compatible.append(cand)

    entities: list[RowEntity] = []
    seen: set[str] = set()
    ent_idx = 0

    for cand in compatible:
        block = evidence_index.get(cand.evidence_id)
        for row in cand.rows:
            if entity_col >= len(row):
                continue
            name = row[entity_col].strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            ent_idx += 1
            if ent_idx > max_rows:
                break

            source_refs = [block.source_ref] if block else []
            entities.append(
                RowEntity(
                    entity_id=f"ent_{ent_idx}",
                    name=name,
                    aliases=[],
                    description=None,
                    source_refs=source_refs,
                    confidence=0.9 if block else 0.5,
                )
            )

    if debug_trace is not None:
        debug_trace.append({
            "stage": "entity_discovery",
            "mode": "source_table_rows",
            "entities": [e.name for e in entities],
            "tables_used": [c.evidence_id for c in compatible],
        })

    return entities


async def discover_entities(
    hint: str,
    schema: DataTableSchema,
    evidence_store: list[EvidenceBlock],
    settings: Settings,
    max_rows: int = 20,
    row_discovery_mode: str = "primary_subjects",
    table_intent: dict | None = None,
    debug_trace: list | None = None,
) -> list[RowEntity]:
    """Extract candidate row entities from evidence."""
    if not evidence_store:
        logger.warning("discover_entities: empty evidence store")
        return []

    strategy = (table_intent or {}).get("strategy", row_discovery_mode)

    # source_table_reconstruction: rows exclusively from table rows
    if strategy == "source_table_reconstruction" or row_discovery_mode == "source_table_rows":
        entities = _discover_entities_from_source_tables(evidence_store, hint, max_rows, debug_trace)
        if entities:
            return entities
        logger.warning("source_table_rows mode: no table candidates — falling back to LLM discovery")

    # hybrid: merge source-table rows with LLM-discovered primary subjects
    elif strategy == "hybrid_table_synthesis":
        table_entities = _discover_entities_from_source_tables(evidence_store, hint, max_rows, debug_trace)
        # LLM discovery continues below; will be merged after

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
    llm_entities: list[RowEntity] = []
    for raw in raw_entities[:max_rows]:
        ent = _build_row_entity(raw, evidence_index)
        if ent:
            llm_entities.append(ent)

    # hybrid: merge table-derived rows with LLM primary subjects
    if strategy == "hybrid_table_synthesis":
        pre_table = locals().get("table_entities", [])
        combined = [*pre_table, *llm_entities]
        entities = _dedup_entities(combined)
    else:
        entities = _dedup_entities(llm_entities)

    if not entities:
        logger.warning("discover_entities: no entities found")

    return entities
