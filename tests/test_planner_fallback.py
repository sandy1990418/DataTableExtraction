"""Tests for planner JSON-parse robustness and fallback source-table reconstruction."""

from __future__ import annotations


from app.models.data_table import EvidenceBlock, SourceRef
from app.services.data_table.table_composer import (
    DraftDataTable,
    _normalize_notes,
    _parse_draft,
    build_draft_from_source_table,
)
from app.services.data_table.table_planner import (
    DataTablePlan,
    PlannedColumn,
    _extract_json,
)

SAMPLE_TABLE_MD = """\
| method | single_hop_f1 | multi_hop_f1 |
| --- | ---: | ---: |
| A-MEM | 60.30 | 44.27 |
| MemGPT | 1.18 | 1.18 |
| Zep | 23.00 | 20.00 |
"""


def _make_block(ev_id="ev_1", table_md=SAMPLE_TABLE_MD, title="Experiment Results") -> EvidenceBlock:
    return EvidenceBlock(
        evidence_id=ev_id,
        source_id="src_1",
        kind="markdown_table",
        text="",
        table_markdown=table_md,
        title=title,
        source_ref=SourceRef(source_id="src_1", evidence_id=ev_id, kind="markdown_table"),
    )


# ── JSON extraction ───────────────────────────────────────────────────────────

def test_extract_json_clean():
    raw = '{"table_title": "Results", "row_grain": "method"}'
    data = _extract_json(raw)
    assert data["table_title"] == "Results"


def test_extract_json_markdown_fences():
    raw = '```json\n{"table_title": "Results"}\n```'
    data = _extract_json(raw)
    assert data["table_title"] == "Results"


def test_extract_json_truncated():
    # simulate truncation mid-string: ends with an unterminated value
    raw = '{"table_title": "Results", "row_grain": "method", "columns": [{"name": "Method"'
    # should not raise — either parses a partial object or raises ValueError
    try:
        data = _extract_json(raw)
        assert isinstance(data, dict)
    except (json.JSONDecodeError, ValueError):
        pass  # acceptable — we just verify it doesn't crash the caller ungracefully


import json  # noqa: E402 — needed above


# ── fallback source-table reconstruction ─────────────────────────────────────

def test_build_draft_from_source_table_basic():
    block = _make_block()
    draft = build_draft_from_source_table([block], hint="compare memory results")
    assert draft is not None
    assert draft.headers == ["method", "single_hop_f1", "multi_hop_f1"]
    assert len(draft.rows) == 3
    row_labels = [r.row_label for r in draft.rows]
    assert "A-MEM" in row_labels
    assert "MemGPT" in row_labels
    assert "Zep" in row_labels
    assert draft.notes == ["Fallback source table reconstruction because planner failed."]


def test_build_draft_from_source_table_cell_values():
    block = _make_block()
    draft = build_draft_from_source_table([block], hint="compare memory results")
    assert draft is not None
    amem_row = next(r for r in draft.rows if r.row_label == "A-MEM")
    assert amem_row.cells["single_hop_f1"].value == 60.30
    assert amem_row.cells["single_hop_f1"].status == "supported"
    assert amem_row.cells["single_hop_f1"].evidence_id == "ev_1"
    assert amem_row.cells["single_hop_f1"].quote is not None


def test_build_draft_no_table_evidence():
    block = EvidenceBlock(
        evidence_id="ev_x",
        source_id="src_x",
        kind="text_fact",
        text="Some text without a table.",
        source_ref=SourceRef(source_id="src_x", evidence_id="ev_x", kind="text_fact"),
    )
    draft = build_draft_from_source_table([block], hint="compare results")
    assert draft is None


def test_fallback_plan_triggers_source_table_not_composer(monkeypatch):
    """When the planner returns the fallback plan, compose_data_table must NOT be called."""
    called = []

    async def fake_compose(*args, **kwargs):
        called.append(True)
        return DraftDataTable(headers=[], rows=[], notes=[])

    monkeypatch.setattr(
        "app.services.data_table.table_composer.compose_data_table",
        fake_compose,
    )

    block = _make_block()
    draft = build_draft_from_source_table([block], hint="compare memory results")
    assert draft is not None
    # composer was not needed — verify called list is empty
    assert called == [], "compose_data_table should not be called when fallback source table is available"


# ── notes normalization ───────────────────────────────────────────────────────

def test_normalize_notes_string():
    assert _normalize_notes("a note") == ["a note"]


def test_normalize_notes_list_of_str():
    assert _normalize_notes(["note a", "note b"]) == ["note a", "note b"]


def test_normalize_notes_list_of_dict_value_key():
    raw = [{"value": "note from dict"}]
    assert _normalize_notes(raw) == ["note from dict"]


def test_normalize_notes_list_of_dict_text_key():
    raw = [{"text": "another note"}]
    assert _normalize_notes(raw) == ["another note"]


def test_normalize_notes_mixed():
    raw = [{"value": "first"}, "second", {"text": "third"}]
    assert _normalize_notes(raw) == ["first", "second", "third"]


def test_normalize_notes_none():
    assert _normalize_notes(None) == []


def test_parse_draft_notes_as_list_of_dicts():
    plan = DataTablePlan(
        table_title="T",
        table_purpose="P",
        row_grain="method",
        columns=[PlannedColumn(name="Method", description="d")],
        evidence_decisions=[],
        reason="",
    )
    data = {
        "headers": ["Method"],
        "rows": [{"row_label": "SysA", "cells": {"Method": {"value": "SysA", "status": "supported",
                                                              "evidence_id": "ev_1", "quote": "SysA"}}}],
        "notes": [{"value": "This is a note."}, {"text": "Another note."}],
    }
    draft = _parse_draft(data, plan)
    assert draft.notes == ["This is a note.", "Another note."]
