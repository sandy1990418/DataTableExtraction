"""Stage 1: normalize existing evidence items into EvidenceBlock."""

from __future__ import annotations

import hashlib
import logging

from app.models.data_table import EvidenceBlock, SourceRef
from app.services.text_match import keywords as extract_keywords

logger = logging.getLogger(__name__)

_MAX_KEYWORDS = 20

# Long text blocks (e.g. a transcript with no headings → one giant section) are
# split into chunks so downstream per-block char caps don't silently discard
# everything past the first ~2000 chars of the document.
_CHUNK_TARGET_CHARS = 1200
_CHUNK_MAX_CHARS = 1800

_SENTENCE_BOUNDARY = ("。", "！", "？", ". ", "! ", "? ", "\n")


def _split_long_text(text: str) -> list[str]:
    """Split text into ~_CHUNK_TARGET_CHARS chunks at paragraph/sentence boundaries."""
    if len(text) <= _CHUNK_MAX_CHARS:
        return [text]

    chunks: list[str] = []
    current = ""
    # prefer paragraph boundaries; fall back to sentence boundaries inside long paragraphs
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) + 2 <= _CHUNK_TARGET_CHARS:
            current = f"{current}\n\n{para}" if current else para
            continue
        if current:
            chunks.append(current)
            current = ""
        while len(para) > _CHUNK_MAX_CHARS:
            cut = -1
            window = para[:_CHUNK_TARGET_CHARS]
            for sep in _SENTENCE_BOUNDARY:
                pos = window.rfind(sep)
                if pos > cut:
                    cut = pos + len(sep)
            if cut <= 0:
                cut = _CHUNK_TARGET_CHARS
            chunks.append(para[:cut].strip())
            para = para[cut:].strip()
        current = para
    if current:
        chunks.append(current)
    return chunks or [text]


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

        table_markdown = _table_to_markdown(headers, rows) if headers else None

        # split heading-less long documents (e.g. transcripts) into chunks;
        # table blocks are never split
        text_parts = [content] if table_markdown else _split_long_text(content)

        for part_idx, part_text in enumerate(text_parts):
            evidence_id = _make_evidence_id(source_id, idx) if len(text_parts) == 1 \
                else _make_evidence_id(f"{source_id}:{part_idx}", idx)
            part_title = title if len(text_parts) == 1 \
                else f"{title or document_name or 'text'} (part {part_idx + 1})"

            # combine text + table for keyword extraction
            text_for_keywords = part_text
            if table_markdown:
                text_for_keywords = f"{part_text}\n{table_markdown}"

            source_ref = SourceRef(
                source_id=source_id,
                evidence_id=evidence_id,
                document_name=document_name,
                kind=kind,
                title=part_title,
                section=source_ref_str or None,
            )

            blocks.append(EvidenceBlock(
                evidence_id=evidence_id,
                source_id=source_id,
                document_name=document_name,
                kind=kind,
                title=part_title,
                text=part_text,
                table_markdown=table_markdown,
                source_ref=source_ref,
                keywords=_build_keywords(text_for_keywords, part_title),
            ))

    if not blocks:
        logger.warning("evidence_store is empty — pipeline will return warnings")

    return blocks
