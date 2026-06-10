"""Extract structured tables from plain-text evidence (e.g. PDF-converted academic papers).

Academic papers converted from PDF often have tables as plain text where the
caption, headers, and data rows are split into separate evidence blocks.
This module re-groups blocks by document and sends full document sections to
the LLM for table reconstruction.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict

from openai import AsyncOpenAI

from app.config import Settings
from app.models.data_table import EvidenceBlock, SourceRef
from app.services.data_table.source_table_rows import _normalize, _score_table

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM = """\
You are a table extraction assistant for academic papers.
Given text from a research paper (which may have been converted from PDF),
extract every benchmark or experiment result table present.

Return a JSON object with:
- tables: list of table objects, each with:
  - title: table caption/title (string)
  - headers: flat list of column header strings
  - rows: list of rows; each row is a list of cell value strings

Rules:
- Only extract tables that compare methods/models/systems using numeric metrics.
- The first column should be the method/model/system name.
- Each row must have the same number of cells as headers.
- If a table has grouped/nested headers (e.g. "Multi Hop" spanning F1 and BLEU),
  flatten into "Multi Hop F1", "Multi Hop BLEU", etc.
- Do NOT invent values; only extract what is present.
- If no benchmark/result table is found, return {"tables": []}.
"""

# how many chars to send per document to the LLM
_MAX_CHARS_PER_DOC = 12000
# minimum score to keep an extracted table
_MIN_SCORE = 4.0


def _markdown_from_headers_rows(headers: list[str], rows: list[list[str]]) -> str:
    if not headers:
        return ""
    width = len(headers)
    sep = "| " + " | ".join(["---"] * width) + " |"
    header_row = "| " + " | ".join(str(h) for h in headers) + " |"
    data_rows = [
        "| " + " | ".join(str(c) for c in (row[:width] + [""] * max(0, width - len(row)))) + " |"
        for row in rows
    ]
    return "\n".join([header_row, sep, *data_rows])


def _group_by_document(evidence_store: list[EvidenceBlock]) -> dict[str, list[EvidenceBlock]]:
    groups: dict[str, list[EvidenceBlock]] = defaultdict(list)
    for block in evidence_store:
        if block.kind == "text_fact" and block.text and len(block.text) > 50:
            key = block.document_name or block.source_id
            groups[key].append(block)
    return dict(groups)


_TABLE_WORDS = ("table", "result", "experiment", "benchmark", "comparison", "performance")


def _is_table_anchor(b: EvidenceBlock) -> bool:
    title = (b.title or "").lower()
    return any(w in title for w in _TABLE_WORDS)


def _count_number_lines(text: str) -> int:
    return sum(1 for ln in text.splitlines() if any(c.isdigit() for c in ln))


def _select_rich_sections(blocks: list[EvidenceBlock], max_chars: int) -> str:
    """Select sections likely to contain benchmark tables.

    Strategy:
    1. Find "anchor" blocks whose title contains table/result/benchmark keywords.
    2. For each anchor, include it plus the following 4 blocks (caption → data continuity).
    3. Fill remaining budget with the most number-dense blocks.
    Blocks are de-duplicated and rendered in original document order.
    """
    n = len(blocks)
    {b.evidence_id: i for i, b in enumerate(blocks)}

    # Step 1: collect anchor windows
    selected_indices: set[int] = set()
    for i, b in enumerate(blocks):
        if _is_table_anchor(b):
            for j in range(i, min(i + 5, n)):
                selected_indices.add(j)

    # Step 2: fill with number-dense blocks
    if len(selected_indices) < 30:
        density_order = sorted(
            range(n),
            key=lambda i: _count_number_lines(blocks[i].text),
            reverse=True,
        )
        for i in density_order:
            if len(selected_indices) >= 40:
                break
            selected_indices.add(i)

    # Step 3: render in original order, respecting char budget
    ordered = sorted(selected_indices)
    parts: list[str] = []
    total = 0
    for i in ordered:
        b = blocks[i]
        chunk = f"[Section: {b.title or 'untitled'}]\n{b.text}"
        if total + len(chunk) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                parts.append(chunk[:remaining])
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n\n---\n\n".join(parts)


async def extract_tables_from_text_blocks(
    evidence_store: list[EvidenceBlock],
    settings: Settings,
    hint: str = "",
    debug_trace: list | None = None,
) -> list[EvidenceBlock]:
    """Group evidence blocks by document; send each document's full text to LLM.

    Returns synthetic EvidenceBlocks with table_markdown for high-confidence tables.
    """
    doc_groups = _group_by_document(evidence_store)
    if not doc_groups:
        return []

    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
    )

    extracted: list[EvidenceBlock] = []

    for doc_name, blocks in doc_groups.items():
        combined_text = _select_rich_sections(blocks, _MAX_CHARS_PER_DOC)
        if not combined_text.strip():
            continue

        # representative source ref from first block
        first_block = blocks[0]

        user_msg = (
            f"Hint: {hint}\n\n"
            f"Document: {doc_name}\n\n"
            f"{combined_text}\n\n"
            "Extract any benchmark/result tables from the above text. Return JSON only."
        )

        try:
            response = await client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": _EXTRACTION_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                max_completion_tokens=128000,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
        except Exception as exc:
            logger.warning("text_table_extractor LLM failed for %s: %s", doc_name, exc)
            continue

        for i, tbl in enumerate(data.get("tables", [])):
            headers = [str(h).strip() for h in tbl.get("headers", []) if str(h).strip()]
            rows = [[str(c).strip() for c in row] for row in tbl.get("rows", []) if row]
            if not headers or not rows:
                continue

            # normalize row width
            width = len(headers)
            rows = [r[:width] + [""] * max(0, width - len(r)) for r in rows]

            score = _score_table(headers, rows, tbl.get("title") or "")
            if score < _MIN_SCORE:
                continue

            table_md = _markdown_from_headers_rows(headers, rows)
            synth_ev_id = f"txt_{_normalize(doc_name)[:8]}_{i}"
            synth_src_ref = SourceRef(
                source_id=first_block.source_id,
                evidence_id=synth_ev_id,
                document_name=doc_name,
                kind="extracted_table",
                title=tbl.get("title") or f"Table from {doc_name}",
            )
            synth_block = EvidenceBlock(
                evidence_id=synth_ev_id,
                source_id=first_block.source_id,
                document_name=doc_name,
                kind="markdown_table",
                title=tbl.get("title") or f"Table from {doc_name}",
                text=f"{len(rows)} rows x {len(headers)} columns (extracted from {doc_name})",
                table_markdown=table_md,
                source_ref=synth_src_ref,
                keywords=headers[:10],
            )
            extracted.append(synth_block)

            if debug_trace is not None:
                debug_trace.append({
                    "stage": "text_table_extraction",
                    "document": doc_name,
                    "extracted_evidence_id": synth_ev_id,
                    "headers": headers,
                    "row_count": len(rows),
                    "score": round(score, 2),
                })

    return extracted


def inject_extracted_tables(
    evidence_store: list[EvidenceBlock],
    extracted: list[EvidenceBlock],
) -> list[EvidenceBlock]:
    """Prepend extracted tables so they score highest in candidate selection."""
    return [*extracted, *evidence_store]
