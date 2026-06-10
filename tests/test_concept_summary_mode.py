"""Tests for the concept_summary (NotebookLM-style knowledge synthesis) path."""

from __future__ import annotations

from app.models.data_table import EvidenceBlock, SourceRef
from app.services.data_table.table_planner import (
    CONCEPT_SUMMARY_HEADERS,
    _format_text_evidence,
    _parse_plan,
)


def _text_block(ev_id: str, text: str, title: str = "") -> EvidenceBlock:
    return EvidenceBlock(
        evidence_id=ev_id,
        source_id="src_1",
        kind="text_fact",
        text=text,
        title=title,
        source_ref=SourceRef(source_id="src_1", evidence_id=ev_id, kind="text_fact"),
    )


def test_parse_plan_accepts_concept_summary():
    plan = _parse_plan({
        "table_title": "LLM 核心概念彙整",
        "table_purpose_type": "concept_summary",
        "row_grain": "concept",
        "columns": [
            {"name": "概念名稱", "description": "概念", "value_type": "string"},
            {"name": "定義與核心功能", "description": "定義", "value_type": "string"},
            {"name": "關鍵特性或運作原理", "description": "特性", "value_type": "string"},
        ],
        "candidate_rows": ["大型語言模型", "Token", "RAG"],
    })
    assert plan.table_purpose_type == "concept_summary"
    assert [c.name for c in plan.columns] == ["概念名稱", "定義與核心功能", "關鍵特性或運作原理"]
    assert plan.candidate_rows == ["大型語言模型", "Token", "RAG"]


def test_parse_plan_concept_summary_default_columns_when_too_few():
    plan = _parse_plan({
        "table_title": "Concepts",
        "table_purpose_type": "concept_summary",
        "columns": [{"name": "Concept", "description": "x", "value_type": "string"}],
    })
    assert [c.name for c in plan.columns] == CONCEPT_SUMMARY_HEADERS


def test_parse_plan_unknown_purpose_still_defaults_to_result_summary():
    plan = _parse_plan({"table_title": "T", "table_purpose_type": "nonsense"})
    assert plan.table_purpose_type == "result_summary"


def test_format_text_evidence_includes_text_blocks():
    blocks = [
        _text_block("ev_1", "Token 是大模型處理文本的最基本單位。", title="Token"),
        _text_block("ev_2", "RAG 從海量數據中抽取與問題匹配的片段。", title="RAG"),
    ]
    out = _format_text_evidence(blocks)
    assert "[ev_1]" in out
    assert "Token 是大模型處理文本的最基本單位" in out
    assert "[ev_2]" in out


def test_format_text_evidence_empty():
    assert _format_text_evidence([]) == "(no text evidence)"
