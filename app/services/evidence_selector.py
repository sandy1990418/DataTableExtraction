from __future__ import annotations

import re
from collections import Counter

from app.models import EvidenceItem
from app.services.evidence_layer import summarize_evidence

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "table",
    "column",
    "method",
    "model",
    "result",
    "results",
    "comparison",
    "performance",
}


def select_table_evidence_blocks(
    specs: list[dict],
    items: list[EvidenceItem],
    max_items: int = 14,
    max_chars: int = 18000,
) -> dict[str, str]:
    """Build a focused evidence prompt for each planned table spec.

    This deterministic selector acts as the evidence-selection member of the
    pipeline: table/figure/result sections are preferred, then keyword overlap
    between the table spec and each evidence item decides the rest.
    """
    return {
        str(spec.get("name")): _build_spec_evidence_block(spec, items, max_items, max_chars)
        for spec in specs
        if spec.get("name")
    }


def _build_spec_evidence_block(
    spec: dict,
    items: list[EvidenceItem],
    max_items: int,
    max_chars: int,
) -> str:
    query_terms = _keywords(_spec_text(spec))
    scored = sorted(
        ((_score_item(item, query_terms), index, item) for index, item in enumerate(items)),
        key=lambda entry: (-entry[0], entry[1]),
    )
    selected = [item for score, _, item in scored if score > 0][:max_items]
    if len(selected) < min(5, len(items)):
        selected.extend(item for _, _, item in scored if item not in selected)
        selected = selected[:max_items]

    parts = [
        "Evidence summary:",
        summarize_evidence(selected),
        "",
        "Focused evidence details:",
        "",
    ]
    for index, item in enumerate(selected, 1):
        parts.append(_format_item(index, item))
        if sum(len(part) for part in parts) > max_chars:
            break
    return "\n".join(parts)[:max_chars]


def _spec_text(spec: dict) -> str:
    parts = [
        str(spec.get("name", "")),
        str(spec.get("title", "")),
        str(spec.get("description", "")),
        str(spec.get("row_entity", "")),
    ]
    anchors = spec.get("evidence_anchors", [])
    if isinstance(anchors, list):
        parts.extend(str(anchor) for anchor in anchors)
    elif anchors:
        parts.append(str(anchors))
    for col in spec.get("columns", []):
        if isinstance(col, dict):
            parts.extend([str(col.get("name", "")), str(col.get("description", "")), str(col.get("example", ""))])
        else:
            parts.append(str(col))
    return " ".join(parts)


def _keywords(text: str) -> Counter[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower())
    return Counter(word for word in words if word not in STOPWORDS)


def _score_item(item: EvidenceItem, query_terms: Counter[str]) -> float:
    item_text = f"{item.title} {item.content} {' '.join(item.headers)}"
    item_terms = _keywords(item_text)
    overlap = sum(min(count, item_terms.get(term, 0)) for term, count in query_terms.items())

    score = float(overlap)
    title = item.title.lower()
    content = item.content.lower()
    if item.kind in {"markdown_table", "image_table"}:
        score += 6
    if title.startswith(("table ", "figure ", "fig.")):
        score += 5
    if any(term in title or term in content[:800] for term in ("benchmark", "metric", "f1", "bleu", "accuracy")):
        score += 3
    if any(term in title or term in content[:800] for term in ("architecture", "storage", "retrieval", "memory")):
        score += 2
    return score


def _format_item(index: int, item: EvidenceItem) -> str:
    lines = [f"### Evidence {index}: [{item.kind}] {item.title}", f"Source: {item.source_ref}"]
    if item.headers:
        lines.append(f"Headers: {', '.join(item.headers)}")
        for row in item.rows[:40]:
            lines.append(f"  Row: {', '.join(str(c) for c in row)}")
        if len(item.rows) > 40:
            lines.append(f"  ... ({len(item.rows) - 40} more rows)")
    else:
        limit = 5000 if item.title.lower().startswith(("table ", "figure ", "fig.")) else 1600
        text = item.content[:limit]
        if len(item.content) > limit:
            text += " ...[truncated]"
        lines.append(f"Content: {text}")
    lines.append("")
    return "\n".join(lines)
