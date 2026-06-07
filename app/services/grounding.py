"""Deterministic numeric grounding for populated tables.

The LLM reviewer (``table_qa``) is a second opinion, not a guarantee. This module
adds a cheap, deterministic check with teeth: every number a cell asserts must
literally appear in that table's evidence block, otherwise the cell is flagged
and the row-revision loop is forced to re-derive it. This catches the most common
failure mode — a fabricated or transformed metric — that an LLM self-review often
rubber-stamps.
"""

from __future__ import annotations

import re

# Matches integers, decimals, thousands-separated and percentage numbers.
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


def _evidence_for_table(table: dict, evidence_block: str | dict[str, str]) -> str:
    if not isinstance(evidence_block, dict):
        return evidence_block
    table_id = str(table.get("table_id") or table.get("name") or table.get("title") or "")
    return evidence_block.get(table_id, "") or evidence_block.get("", "")


def _significant_numbers(text: str) -> list[str]:
    """Numeric tokens worth grounding (normalized: no commas, no trailing %).

    Single-digit integers are skipped — they are almost always present somewhere
    and would only add false positives.
    """
    numbers: list[str] = []
    for match in _NUM.findall(text):
        core = match.rstrip("%").replace(",", "")
        digits = core.replace(".", "")
        if not digits:
            continue
        if len(digits) >= 2 or "." in core:
            numbers.append(core)
    return numbers


def check_numeric_grounding(table: dict, evidence: str) -> list[dict]:
    """Return one entry per cell number that is absent from ``evidence``."""
    evidence_norm = evidence.replace(",", "").lower()
    if not evidence_norm.strip():
        return []

    headers = table.get("headers") or []
    unsupported: list[dict] = []
    for row_idx, row in enumerate(table.get("rows") or []):
        for col_idx, cell in enumerate(row):
            for number in _significant_numbers(str(cell)):
                if number.lower() not in evidence_norm:
                    unsupported.append(
                        {
                            "row": row_idx,
                            "column": headers[col_idx] if col_idx < len(headers) else col_idx,
                            "value": str(cell),
                            "number": number,
                            "reason": "number not found in evidence",
                        }
                    )
    return unsupported


def ground_table_reviews(
    tables: list[dict],
    evidence_block: str | dict[str, str],
    reviews: list[dict],
) -> list[dict]:
    """Merge deterministic numeric-grounding failures into the LLM review list.

    Flagged cells are appended to the matching review's ``unsupported_cells`` and
    the review is forced to ``needs_revision`` so the agent loop re-derives them.
    Returns a new review list (input reviews are not mutated).
    """
    merged = [dict(review) for review in reviews]
    review_by_id = {str(review.get("table_id")): review for review in merged}

    for table in tables:
        table_id = str(table.get("table_id") or table.get("name") or table.get("title") or "")
        unsupported = check_numeric_grounding(table, _evidence_for_table(table, evidence_block))
        if not unsupported:
            continue

        review = review_by_id.get(table_id)
        if review is None:
            review = {
                "table_id": table_id,
                "status": "pass",
                "warnings": [],
                "unsupported_cells": [],
            }
            merged.append(review)
            review_by_id[table_id] = review

        review["status"] = "needs_revision"
        review["unsupported_cells"] = [*(review.get("unsupported_cells") or []), *unsupported]
        review["warnings"] = [
            *(review.get("warnings") or []),
            f"{len(unsupported)} cell value(s) contain numbers absent from the evidence.",
        ]

    return merged
