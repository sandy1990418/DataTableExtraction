"""Tests for long-text chunking in the data-table evidence store."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.data_table.evidence_store import (
    _CHUNK_MAX_CHARS,
    _split_long_text,
    build_evidence_store,
)


@dataclass
class _Item:
    kind: str = "text_fact"
    source_ref: str = "transcript.md:lines:1-999"
    title: str | None = None
    content: str = ""
    headers: list = field(default_factory=list)
    rows: list = field(default_factory=list)


def test_short_text_not_split():
    assert _split_long_text("短文字。") == ["短文字。"]


def test_long_text_split_at_sentence_boundaries():
    text = "這是一句話。" * 600  # ~3600 chars, no paragraph breaks
    chunks = _split_long_text(text)
    assert len(chunks) > 1
    assert all(len(c) <= _CHUNK_MAX_CHARS for c in chunks)
    assert "".join(chunks) == text


def test_long_text_split_preserves_paragraphs():
    paras = [f"第{i}段內容。" * 40 for i in range(10)]  # each ~280 chars
    text = "\n\n".join(paras)
    chunks = _split_long_text(text)
    assert len(chunks) > 1
    joined = "\n\n".join(chunks)
    for p in paras:
        assert p in joined


def test_build_evidence_store_splits_giant_transcript_block():
    transcript = "大模型的生成方式像文字接龍。" * 500  # ~7000 chars, one block
    blocks = build_evidence_store([_Item(content=transcript)])
    assert len(blocks) > 1
    # unique evidence ids, part-numbered titles, full content preserved
    ids = [b.evidence_id for b in blocks]
    assert len(set(ids)) == len(ids)
    assert all("(part" in (b.title or "") for b in blocks)
    assert "".join(b.text for b in blocks) == transcript


def test_build_evidence_store_keeps_short_block_intact():
    blocks = build_evidence_store([_Item(content="短內容。", title="標題")])
    assert len(blocks) == 1
    assert blocks[0].title == "標題"
    assert blocks[0].text == "短內容。"


def test_build_evidence_store_never_splits_tables():
    item = _Item(
        kind="markdown_table",
        content="x" * 5000,
        headers=["A", "B"],
        rows=[["1", "2"]],
    )
    blocks = build_evidence_store([item])
    assert len(blocks) == 1
    assert blocks[0].table_markdown is not None
