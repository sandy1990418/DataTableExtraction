"""Extract and score source markdown tables from evidence for benchmark/result mode."""

from __future__ import annotations

import re

from pydantic import BaseModel

from app.models.data_table import EvidenceBlock

_RESULT_HEADER_TERMS = {
    "method", "model", "system", "approach", "baseline", "name",
    "multi_hop", "multihop", "multi-hop",
    "temporal", "open_domain", "open-domain", "opendomain",
    "single_hop", "singelhop", "single-hop",
    "overall", "overall_score", "average", "avg",
    "rouge", "bleu", "meteor", "sbert", "f1", "accuracy",
    "precision", "recall", "score", "result", "performance",
}

_RESULT_TITLE_TERMS = {
    "result", "results", "benchmark", "evaluation", "experiment",
    "performance", "comparison", "score", "leaderboard",
}

_NUMBER_RE = re.compile(r"\d")


class SourceTableCandidate(BaseModel):
    evidence_id: str
    source_id: str
    document_name: str | None = None
    title: str | None = None
    headers: list[str]
    rows: list[list[str]]
    score: float = 0.0


def parse_markdown_table(table_markdown: str) -> tuple[list[str], list[list[str]]]:
    """Parse a simple pipe-delimited markdown table into headers + rows."""
    lines = [ln.strip() for ln in table_markdown.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        return [], []

    def split_row(line: str) -> list[str]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        return cells

    headers = split_row(lines[0])
    # skip separator line (---) if present
    data_lines = lines[1:]
    if data_lines and re.match(r"^[\|\s\-:]+$", data_lines[0]):
        data_lines = data_lines[1:]

    rows = [split_row(ln) for ln in data_lines if "|" in ln]
    # normalize row width
    width = len(headers)
    rows = [r[:width] + [""] * max(0, width - len(r)) for r in rows]
    return headers, rows


def _normalize(text: str) -> str:
    return re.sub(r"[\s\-_]+", "", text.lower())


def _score_table(headers: list[str], rows: list[list[str]], title: str | None) -> float:
    score = 0.0
    norm_headers = [_normalize(h) for h in headers]
    header_set = set(norm_headers)

    # entity/method column present
    if any(h in {"method", "model", "system", "approach", "baseline", "name"} for h in header_set):
        score += 5.0

    # result-domain headers
    result_header_count = sum(
        1 for h in header_set
        if any(_normalize(t) in h or h in _normalize(t) for t in _RESULT_HEADER_TERMS)
    )
    score += result_header_count * 2.0

    # numeric cells
    numeric_cells = sum(
        1 for row in rows for cell in row if _NUMBER_RE.search(cell)
    )
    score += min(numeric_cells * 0.5, 10.0)

    # result-related title
    if title:
        title_lower = title.lower()
        if any(t in title_lower for t in _RESULT_TITLE_TERMS):
            score += 5.0

    # penalty: too few rows
    if len(rows) < 2:
        score -= 3.0

    return score


def extract_source_table_candidates(
    evidence_store: list[EvidenceBlock],
    hint: str,
) -> list[SourceTableCandidate]:
    """Find and rank markdown tables in evidence that look like result/benchmark tables."""
    candidates: list[SourceTableCandidate] = []

    for block in evidence_store:
        if not block.table_markdown:
            continue
        headers, rows = parse_markdown_table(block.table_markdown)
        if not headers or not rows:
            continue
        score = _score_table(headers, rows, block.title)
        candidates.append(
            SourceTableCandidate(
                evidence_id=block.evidence_id,
                source_id=block.source_id,
                document_name=block.document_name,
                title=block.title,
                headers=headers,
                rows=rows,
                score=score,
            )
        )

    candidates.sort(key=lambda c: -c.score)
    return candidates


def find_entity_column_index(headers: list[str]) -> int:
    """Return index of the entity/method column (first non-numeric looking column)."""
    entity_names = {"method", "model", "system", "approach", "baseline", "name", "algorithm"}
    for i, h in enumerate(headers):
        if _normalize(h) in entity_names:
            return i
    # fallback: first column
    return 0
