"""Tests for evidence-aware strategy scoring."""

from __future__ import annotations

from app.models.data_table import EvidenceBlock, SourceRef
from app.services.data_table.strategy import select_strategy, strategy_to_row_discovery_mode

RESULT_TABLE_MD = """\
| method | multi_hop | temporal | open_domain | single_hop | overall_score |
| --- | ---: | ---: | ---: | ---: | ---: |
| A-Mem | 44.27 | 31.20 | 50.10 | 60.30 | 46.47 |
| MemGPT | 1.18 | 2.00 | 3.00 | 1.18 | 1.84 |
"""

TEXT_BLOCK_MD = (
    "A-MEM is an agentic memory system for LLM agents. "
    "It uses semantic embedding for retrieval. "
    "Key limitation: scaling cost at large memory sizes."
)


def _table_block(ev_id="ev_t1", table_md=RESULT_TABLE_MD, title="Experiment Results") -> EvidenceBlock:
    return EvidenceBlock(
        evidence_id=ev_id, source_id="src_1", kind="markdown_table",
        text="", table_markdown=table_md, title=title,
        source_ref=SourceRef(source_id="src_1", evidence_id=ev_id, kind="markdown_table"),
    )


def _text_block(ev_id="ev_x1", text=TEXT_BLOCK_MD, title="AMem Overview") -> EvidenceBlock:
    return EvidenceBlock(
        evidence_id=ev_id, source_id="src_1", kind="text_fact",
        text=text, title=title,
        source_ref=SourceRef(source_id="src_1", evidence_id=ev_id, kind="text_fact"),
    )


# ── source_table_reconstruction ───────────────────────────────────────────────

def test_benchmark_hint_plus_table_selects_reconstruction():
    store = [_table_block()]
    strategy, scores = select_strategy("Compare these memory experiment result", store)
    assert strategy == "source_table_reconstruction"


def test_result_hint_no_table_does_not_select_reconstruction():
    # No source table → reconstruction is disqualified
    store = [_text_block()]
    strategy, scores = select_strategy("Compare these memory experiment result", store)
    assert strategy != "source_table_reconstruction"
    assert scores["source_table_reconstruction"]["score"] < 0


def test_short_score_hint_selects_reconstruction():
    store = [_table_block()]
    strategy, _ = select_strategy("benchmark scores", store)
    assert strategy == "source_table_reconstruction"


# ── synthesized_entity_comparison ────────────────────────────────────────────

def test_architecture_hint_no_table_selects_synthesized():
    store = [_text_block(), _text_block("ev_x2", title="MemGPT design")]
    strategy, _ = select_strategy(
        "Compare these memory systems by architecture design and retrieval mechanism", store
    )
    assert strategy == "synthesized_entity_comparison"


def test_arch_hint_with_table_penalizes_reconstruction():
    store = [_table_block(), _text_block()]
    strategy, scores = select_strategy(
        "Compare these memory systems by architecture design and retrieval mechanism", store
    )
    # architecture terms push reconstruction score down
    assert scores["source_table_reconstruction"]["score"] < scores["synthesized_entity_comparison"]["score"]


# ── hybrid_table_synthesis ────────────────────────────────────────────────────

def test_mixed_hint_with_table_selects_hybrid():
    store = [_table_block(), _text_block(), _text_block("ev_x2"), _text_block("ev_x3")]
    strategy, _ = select_strategy(
        "Compare these memory systems by architecture design, retrieval mechanism, benchmark performance, and key limitations.",
        store,
    )
    assert strategy == "hybrid_table_synthesis"


# ── row_discovery_mode mapping ────────────────────────────────────────────────

def test_strategy_maps_to_row_discovery_mode():
    assert strategy_to_row_discovery_mode("source_table_reconstruction") == "source_table_rows"
    assert strategy_to_row_discovery_mode("synthesized_entity_comparison") == "primary_subjects"
    assert strategy_to_row_discovery_mode("hybrid_table_synthesis") == "hybrid"


# ── debug trace ───────────────────────────────────────────────────────────────

def test_debug_trace_records_strategy_selection():
    store = [_table_block()]
    trace = []
    select_strategy("Compare experiment results", store, debug_trace=trace)
    stages = [t["stage"] for t in trace]
    assert "strategy_selection" in stages
    entry = next(t for t in trace if t["stage"] == "strategy_selection")
    assert "winner" in entry
    assert "scores" in entry
    assert "evidence_signals" in entry
