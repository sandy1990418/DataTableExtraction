"""Stage 7: deterministic cell verification."""

from __future__ import annotations

import re

from app.models.data_table import DataTableColumn, EvidenceBlock, GroundedCell, RowEntity

_NUMBER_RE = re.compile(r"\d+\.?\d*")

_GENERIC_VALUES = {"yes", "no", "high", "low", "moderate", "true", "false", "n/a", "unknown"}


def _fuzzy_contains(haystack: str, needle: str, min_overlap: float = 0.7) -> bool:
    """True if enough words from needle appear in haystack."""
    if not needle:
        return False
    haystack_lower = haystack.lower()
    needle_lower = needle.lower()
    if needle_lower in haystack_lower:
        return True
    words = [w for w in re.findall(r"[A-Za-z一-鿿]{2,}", needle_lower)]
    if not words:
        return False
    matched = sum(1 for w in words if w in haystack_lower)
    return (matched / len(words)) >= min_overlap


def _quote_in_evidence(quote: str, blocks: list[EvidenceBlock]) -> bool:
    # source-table quotes have format "col=val | col=val" — check if values appear in table
    if " | " in quote and "=" in quote:
        pairs = [p.strip() for p in quote.split("|")]
        values = [p.split("=", 1)[1].strip() for p in pairs if "=" in p]
        for block in blocks:
            text = (block.text or "") + (block.table_markdown or "")
            if all(v in text for v in values if v):
                return True
        return bool(values)  # trust source-table quotes

    for block in blocks:
        text = (block.text or "") + (block.table_markdown or "")
        if _fuzzy_contains(text, quote):
            return True
    return False


def _entity_in_evidence(entity: RowEntity, blocks: list[EvidenceBlock]) -> bool:
    names = [entity.name.lower(), *[a.lower() for a in entity.aliases]]
    for block in blocks:
        text = ((block.text or "") + (block.table_markdown or "")).lower()
        if any(name in text for name in names):
            return True
    return False


def _numeric_in_quote(value, quote: str) -> bool:
    if value is None:
        return True
    str_value = str(value)
    numbers_in_value = set(_NUMBER_RE.findall(str_value))
    numbers_in_quote = set(_NUMBER_RE.findall(quote))
    if not numbers_in_value:
        return True
    return bool(numbers_in_value & numbers_in_quote)


def verify_cell(
    cell: GroundedCell,
    entity: RowEntity,
    column: DataTableColumn,
    evidence_blocks: list[EvidenceBlock],
) -> GroundedCell:
    """Run deterministic checks; downgrade status if checks fail."""
    if cell.status in ("not_reported", "unsupported"):
        # not_reported with citations → suspicious
        if cell.status == "not_reported" and cell.citations:
            cell.verification_notes.append("not_reported cell had citations — citations cleared")
            cell.citations = []
        return cell

    notes: list[str] = []

    # 1. supported must have citations
    if cell.status == "supported" and not cell.citations:
        notes.append("supported cell has no citations → downgraded to unsupported")
        cell.status = "unsupported"
        cell.verification_notes.extend(notes)
        return cell

    failed = False
    for citation in cell.citations:
        quote = citation.quote

        # 2. quote must appear in evidence
        if not _quote_in_evidence(quote, evidence_blocks):
            notes.append(f"quote not found in evidence: '{quote[:60]}...'")
            failed = True

        # 3. numeric mismatch
        if column.value_type == "number" and not _numeric_in_quote(cell.value, quote):
            notes.append(f"numeric value '{cell.value}' not found in quote: '{quote[:60]}'")
            failed = True

        # 4. generic boolean must be justified
        if column.value_type == "boolean":
            str_val = str(cell.value or "").lower()
            if str_val in ("yes", "no", "true", "false"):
                if not quote.strip():
                    notes.append("boolean cell has empty quote")
                    failed = True

    # 5. generic value must have a meaningful quote
    str_val = str(cell.value or "").lower()
    if str_val in _GENERIC_VALUES and cell.citations:
        if all(not c.quote.strip() for c in cell.citations):
            notes.append(f"generic value '{cell.value}' has no supporting quote")
            failed = True

    if failed and cell.status == "supported":
        cell.status = "unsupported"
        notes.insert(0, "verification failed → downgraded to unsupported")

    cell.verification_notes.extend(notes)
    return cell
