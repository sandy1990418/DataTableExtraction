"""Stage 5: per-cell evidence retrieval using local scoring."""

from __future__ import annotations

import re
from collections import Counter

from app.models.data_table import DataTableColumn, EvidenceBlock, RowEntity
from app.services.text_match import keywords as extract_keywords
from app.services.text_match import overlap_score

_NUMBER_RE = re.compile(r"\d+\.?\d*")


def _query_keywords(
    entity: RowEntity,
    column: DataTableColumn,
    hint: str,
) -> Counter:
    parts = [entity.name, *entity.aliases, column.name, column.description, hint]
    combined = " ".join(parts)
    return extract_keywords(combined)


def _score_block(
    block: EvidenceBlock,
    query: Counter,
    entity: RowEntity,
    column: DataTableColumn,
) -> float:
    target = extract_keywords(f"{block.title or ''} {block.text} {block.table_markdown or ''}")
    score = float(overlap_score(query, target))

    # entity name/alias boost
    text_lower = (block.text + (block.table_markdown or "")).lower()
    entity_names = [entity.name.lower(), *[a.lower() for a in entity.aliases]]
    if any(name in text_lower for name in entity_names):
        score += 5.0

    # source table boost for metric columns
    if column.role == "metric" and block.kind == "markdown_table":
        score += 3.0

    # image/table boost for visual evidence
    if block.kind in ("image_table", "markdown_table") and block.table_markdown:
        score += 2.0

    # numeric column boost if block contains numbers
    if column.value_type == "number" and _NUMBER_RE.search(block.text or ""):
        score += 2.0

    return score


def retrieve_cell_evidence(
    entity: RowEntity,
    column: DataTableColumn,
    evidence_store: list[EvidenceBlock],
    hint: str,
    k: int = 5,
    debug_trace: list | None = None,
) -> list[EvidenceBlock]:
    """Score all evidence blocks for a specific entity × column pair, return top-k."""
    query = _query_keywords(entity, column, hint)

    scored = []
    for block in evidence_store:
        score = _score_block(block, query, entity, column)
        if score > 0:
            scored.append((block, score))

    scored.sort(key=lambda x: -x[1])
    selected = [b for b, _ in scored[:k]]

    if debug_trace is not None:
        debug_trace.append(
            {
                "stage": "cell_retrieval",
                "entity": entity.name,
                "column": column.name,
                "selected_evidence": [
                    {
                        "evidence_id": b.evidence_id,
                        "score": round(s, 2),
                        "reason": "overlap + boost",
                        "preview": (b.text or "")[:80],
                    }
                    for b, s in scored[:k]
                ],
            }
        )

    return selected
