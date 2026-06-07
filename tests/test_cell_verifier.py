"""Tests for deterministic cell verification."""

from __future__ import annotations

from app.models.data_table import (
    CellCitation,
    DataTableColumn,
    EvidenceBlock,
    GroundedCell,
    RowEntity,
    SourceRef,
)
from app.services.data_table.cell_verifier import verify_cell


def _col(name="Score", role="metric", value_type="number") -> DataTableColumn:
    return DataTableColumn(name=name, role=role, description="A score.", value_type=value_type)


def _entity(name="ModelA") -> RowEntity:
    return RowEntity(entity_id="ent_1", name=name, confidence=0.8)


def _source_ref(ev_id="ev_1") -> SourceRef:
    return SourceRef(source_id="src_1", evidence_id=ev_id, kind="text_fact")


def _block(ev_id="ev_1", text="ModelA achieves a BLEU score of 45.6.") -> EvidenceBlock:
    return EvidenceBlock(
        evidence_id=ev_id,
        source_id="src_1",
        kind="text_fact",
        text=text,
        source_ref=_source_ref(ev_id),
    )


def _citation(quote: str, ev_id="ev_1") -> CellCitation:
    return CellCitation(source_ref=_source_ref(ev_id), quote=quote)


def test_supported_cell_exact_quote_passes():
    cell = GroundedCell(
        value=45.6,
        status="supported",
        citations=[_citation("ModelA achieves a BLEU score of 45.6.")],
        confidence=0.9,
    )
    result = verify_cell(cell, _entity(), _col(), [_block()])
    assert result.status == "supported"
    assert result.verification_notes == []


def test_supported_cell_no_citation_becomes_unsupported():
    cell = GroundedCell(value="Yes", status="supported", citations=[], confidence=0.5)
    result = verify_cell(cell, _entity(), _col(value_type="string"), [_block()])
    assert result.status == "unsupported"
    assert any("no citations" in n for n in result.verification_notes)


def test_numeric_mismatch_becomes_unsupported():
    cell = GroundedCell(
        value=99.9,
        status="supported",
        citations=[_citation("BLEU score of 45.6.")],
        confidence=0.9,
    )
    result = verify_cell(cell, _entity(), _col(value_type="number"), [_block()])
    assert result.status == "unsupported"
    assert any("numeric" in n for n in result.verification_notes)


def test_not_reported_cell_no_citation_passes():
    cell = GroundedCell(value=None, status="not_reported", citations=[], confidence=0.0)
    result = verify_cell(cell, _entity(), _col(), [_block()])
    assert result.status == "not_reported"
    assert result.citations == []


def test_not_reported_with_citations_cleared():
    cell = GroundedCell(
        value=None,
        status="not_reported",
        citations=[_citation("some quote")],
        confidence=0.0,
    )
    result = verify_cell(cell, _entity(), _col(), [_block()])
    assert result.status == "not_reported"
    assert result.citations == []
