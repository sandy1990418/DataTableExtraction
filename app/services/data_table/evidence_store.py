"""Stage 1: normalize existing evidence items into EvidenceBlock."""

from __future__ import annotations

import hashlib
import logging

from app.models.data_table import EvidenceBlock, SourceRef
from app.services.text_match import keywords as extract_keywords

logger = logging.getLogger(__name__)

_MAX_KEYWORDS = 20


def _make_evidence_id(source_id: str, idx: int) -> str:
    digest = hashlib.md5(f"{source_id}:{idx}".encode()).hexdigest()[:6]
    return f"ev_{digest}"


def _make_source_id(document_name: str | None, idx: int) -> str:
    base = document_name or f"doc_{idx}"
    return base.replace(" ", "_").replace("/", "_")


def _build_keywords(text: str, title: str | None) -> list[str]:
    combined = f"{title or ''} {text}"
    kw = extract_keywords(combined)
    return [term for term, _ in kw.most_common(_MAX_KEYWORDS)]


def _table_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    if not headers:
        return ""
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    header_row = "| " + " | ".join(headers) + " |"
    data_rows = ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join([header_row, sep, *data_rows])


def build_evidence_store(evidence_items: list) -> list[EvidenceBlock]:
    """Convert pipeline evidence items into EvidenceBlock list.

    Accepts EvidenceItem dataclasses from the existing evidence layer.
    """
    blocks: list[EvidenceBlock] = []

    for idx, item in enumerate(evidence_items):
        kind = getattr(item, "kind", "text_fact")
        source_ref_str = getattr(item, "source_ref", "") or ""
        title = getattr(item, "title", None) or None
        content = getattr(item, "content", "") or ""
        headers = getattr(item, "headers", []) or []
        rows = getattr(item, "rows", []) or []

        # derive document name from source_ref (format: "docname:lines:X-Y")
        document_name = source_ref_str.split(":")[0] if source_ref_str else None
        source_id = _make_source_id(document_name, idx)
        evidence_id = _make_evidence_id(source_id, idx)

        table_markdown = _table_to_markdown(headers, rows) if headers else None

        # combine text + table for keyword extraction
        text_for_keywords = content
        if table_markdown:
            text_for_keywords = f"{content}\n{table_markdown}"

        source_ref = SourceRef(
            source_id=source_id,
            evidence_id=evidence_id,
            document_name=document_name,
            kind=kind,
            title=title,
            section=source_ref_str or None,
        )

        block = EvidenceBlock(
            evidence_id=evidence_id,
            source_id=source_id,
            document_name=document_name,
            kind=kind,
            title=title,
            text=content,
            table_markdown=table_markdown,
            source_ref=source_ref,
            keywords=_build_keywords(text_for_keywords, title),
        )
        blocks.append(block)

    if not blocks:
        logger.warning("evidence_store is empty — pipeline will return warnings")

    return blocks
