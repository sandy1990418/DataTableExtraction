"""Tests for entity discovery helpers (dedup / alias merge)."""

from __future__ import annotations

from app.models.data_table import RowEntity, SourceRef
from app.services.data_table.entity_discovery import _build_row_entity, _dedup_entities


def _src_ref(ev_id: str) -> SourceRef:
    return SourceRef(source_id="src_1", evidence_id=ev_id, kind="text_fact")


def test_dedup_merges_same_name():
    a = RowEntity(entity_id="ent_1", name="GPT-4o", source_refs=[_src_ref("ev_1")], confidence=0.8)
    b = RowEntity(entity_id="ent_2", name="gpt-4o", source_refs=[_src_ref("ev_2")], confidence=0.6)
    merged = _dedup_entities([a, b])
    assert len(merged) == 1
    assert merged[0].name == "GPT-4o"
    assert len(merged[0].source_refs) == 2
    assert merged[0].confidence == 0.8


def test_dedup_merges_aliases():
    a = RowEntity(
        entity_id="ent_1",
        name="GPT-4o",
        aliases=["OpenAI GPT-4o"],
        source_refs=[_src_ref("ev_1")],
        confidence=0.9,
    )
    b = RowEntity(
        entity_id="ent_2",
        name="gpt-4o",
        aliases=["GPT 4o"],
        source_refs=[_src_ref("ev_2")],
        confidence=0.7,
    )
    merged = _dedup_entities([a, b])
    assert len(merged) == 1
    aliases = {al.lower() for al in merged[0].aliases}
    assert "gpt 4o" in aliases or "openai gpt-4o" in aliases


def test_dedup_keeps_distinct_entities():
    a = RowEntity(entity_id="ent_1", name="GPT-4o", source_refs=[_src_ref("ev_1")], confidence=0.8)
    b = RowEntity(entity_id="ent_2", name="Claude 3", source_refs=[_src_ref("ev_2")], confidence=0.7)
    merged = _dedup_entities([a, b])
    assert len(merged) == 2


def test_build_row_entity_missing_source_gets_low_confidence():
    from app.models.data_table import EvidenceBlock

    block = EvidenceBlock(
        evidence_id="ev_1",
        source_id="src_1",
        kind="text_fact",
        text="some text",
        source_ref=SourceRef(source_id="src_1", evidence_id="ev_1", kind="text_fact"),
    )
    evidence_index = {"ev_1": block}

    raw = {
        "entity_id": "ent_1",
        "name": "ModelX",
        "aliases": [],
        "description": "A model",
        "source_refs": ["ev_999"],  # unknown evidence_id
        "confidence": 0.9,
    }
    ent = _build_row_entity(raw, evidence_index)
    assert ent is not None
    assert ent.confidence <= 0.3  # low confidence because no valid source refs


def test_build_row_entity_empty_name_returns_none():
    raw = {"entity_id": "ent_1", "name": "", "aliases": [], "source_refs": [], "confidence": 0.5}
    ent = _build_row_entity(raw, {})
    assert ent is None
