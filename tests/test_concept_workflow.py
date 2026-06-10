"""Tests for the deterministic parts of the multi-stage concept workflow."""

from __future__ import annotations

from app.models.data_table import EvidenceBlock, SourceRef
from app.services.data_table.concept_workflow import (
    ConceptOutline,
    order_rows,
    parse_table_outline,
    select_evidence_for_concept,
)
from app.services.data_table.table_composer import DraftCell, DraftRow


def _text_block(ev_id: str, text: str, title: str = "") -> EvidenceBlock:
    return EvidenceBlock(
        evidence_id=ev_id,
        source_id="src_1",
        kind="text_fact",
        text=text,
        title=title,
        source_ref=SourceRef(source_id="src_1", evidence_id=ev_id, kind="text_fact"),
    )


def _row(label: str) -> DraftRow:
    return DraftRow(row_label=label, cells={"定義": DraftCell(value=f"{label} 的定義")})


def test_parse_table_outline_basic():
    outline = parse_table_outline({
        "concepts": [
            {"name": "Token", "evidence_ids": ["ev_1"], "key_points": ["最基本單位", "1 token ≈ 0.75 英文單詞"]},
            {"name": "RAG", "evidence_ids": ["ev_2"], "related_concepts": ["Context Window"]},
        ],
        "row_order": ["Token", "RAG"],
    })
    assert [c.name for c in outline.concepts] == ["Token", "RAG"]
    assert outline.concepts[0].key_points == ["最基本單位", "1 token ≈ 0.75 英文單詞"]
    assert outline.row_order == ["Token", "RAG"]


def test_parse_table_outline_skips_invalid_entries():
    outline = parse_table_outline({
        "concepts": [{"name": ""}, "not a dict", {"name": "LLM"}],
        "row_order": [],
    })
    assert [c.name for c in outline.concepts] == ["LLM"]


def test_select_evidence_prefers_outline_ids_then_lexical():
    blocks = [
        _text_block("ev_1", "Token 是大模型處理文本的最基本單位。"),
        _text_block("ev_2", "RAG 從海量數據中抽取片段。"),
        _text_block("ev_3", "Tokenizer 利用 BPE 算法將文字切分為 Token ID。"),
    ]
    outline = ConceptOutline(name="Token", evidence_ids=["ev_1"])
    out = select_evidence_for_concept(outline, blocks, {"ev_1", "ev_2", "ev_3"})
    # ev_1 from outline mapping; ev_3 via lexical match on "Token"; ev_2 excluded
    assert "[evidence_id=ev_1]" in out
    assert "[evidence_id=ev_3]" in out
    assert "[evidence_id=ev_2]" not in out
    assert out.index("ev_1") < out.index("ev_3")


def test_select_evidence_respects_allowed_ids():
    blocks = [_text_block("ev_1", "Token 基本單位")]
    outline = ConceptOutline(name="Token", evidence_ids=["ev_1"])
    assert select_evidence_for_concept(outline, blocks, set()) == "(no evidence)"


def test_order_rows_narrative_order():
    rows = [_row("RAG"), _row("LLM"), _row("Token")]
    ordered = order_rows(rows, ["LLM", "Token", "RAG"])
    assert [r.row_label for r in ordered] == ["LLM", "Token", "RAG"]


def test_order_rows_unknown_labels_keep_relative_position_at_end():
    rows = [_row("Bonus"), _row("LLM"), _row("Extra")]
    ordered = order_rows(rows, ["LLM"])
    assert [r.row_label for r in ordered] == ["LLM", "Bonus", "Extra"]
