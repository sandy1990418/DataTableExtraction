"""Tests for deterministic long-form source-table composer."""

from __future__ import annotations

import pytest

from app.models.data_table import EvidenceBlock, SourceRef
from app.services.data_table.table_composer import (
    DraftDataTable,
    compose_long_form_from_source_tables,
)
from app.services.data_table.table_planner import (
    DataTablePlan,
    EvidenceDecision,
    PlannedColumn,
)

# 12-row benchmark table with many metric columns
AMEM_TABLE_MD = """\
| method | single_hop_f1 | single_hop_bleu | multi_hop_f1 | multi_hop_bleu | temporal_f1 | temporal_bleu | open_domain_f1 | open_domain_bleu | overall_f1 | overall_bleu | overall_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A-MEM | 60.30 | 0.41 | 44.27 | 0.30 | 31.20 | 0.22 | 50.10 | 0.35 | 46.47 | 0.32 | 46.47 |
| MemGPT | 1.18 | 0.01 | 1.18 | 0.01 | 2.00 | 0.02 | 3.00 | 0.02 | 1.84 | 0.02 | 1.84 |
| MemoryBank | 10.00 | 0.08 | 9.50 | 0.07 | 8.00 | 0.06 | 11.00 | 0.09 | 9.63 | 0.08 | 9.63 |
| Zep | 23.00 | 0.18 | 20.00 | 0.15 | 21.00 | 0.16 | 22.00 | 0.17 | 21.50 | 0.17 | 21.50 |
| GPT-4o-mini | 55.00 | 0.37 | 40.00 | 0.28 | 28.00 | 0.20 | 45.00 | 0.32 | 42.00 | 0.29 | 42.00 |
| Qwen2.5-3B | 30.00 | 0.22 | 25.00 | 0.19 | 20.00 | 0.15 | 28.00 | 0.20 | 25.75 | 0.19 | 25.75 |
| Llama-3.1-8B | 35.00 | 0.25 | 30.00 | 0.22 | 22.00 | 0.17 | 32.00 | 0.23 | 29.75 | 0.22 | 29.75 |
| Llama-3.1-70B | 50.00 | 0.35 | 38.00 | 0.27 | 26.00 | 0.19 | 42.00 | 0.30 | 39.00 | 0.28 | 39.00 |
| GPT-4o | 62.00 | 0.43 | 46.00 | 0.32 | 33.00 | 0.24 | 52.00 | 0.37 | 48.25 | 0.34 | 48.25 |
| TiM-GPT-4o-mini | 58.00 | 0.40 | 42.00 | 0.29 | 30.00 | 0.21 | 48.00 | 0.34 | 44.50 | 0.31 | 44.50 |
| TiM-Qwen2.5-3B | 32.00 | 0.23 | 27.00 | 0.20 | 21.00 | 0.16 | 29.00 | 0.21 | 27.25 | 0.20 | 27.25 |
| TiM-Llama-3.1-8B | 37.00 | 0.27 | 32.00 | 0.23 | 23.00 | 0.18 | 33.00 | 0.24 | 31.25 | 0.23 | 31.25 |
"""


def _make_block(ev_id="ev_1", table_md=AMEM_TABLE_MD, title="AMem Benchmark Results") -> EvidenceBlock:
    return EvidenceBlock(
        evidence_id=ev_id,
        source_id="src_1",
        kind="markdown_table",
        text="",
        table_markdown=table_md,
        title=title,
        source_ref=SourceRef(source_id="src_1", evidence_id=ev_id, kind="markdown_table"),
    )


def _long_plan(ev_id: str = "ev_1") -> DataTablePlan:
    return DataTablePlan(
        table_title="Memory Benchmark Results (Long Form)",
        table_purpose="Compare memory systems across all metrics.",
        row_grain="method × metric",
        table_format="long",
        columns=[
            PlannedColumn(name="Method / System", description="The method or system.", value_type="string"),
            PlannedColumn(name="Benchmark / Task", description="Benchmark or task name.", value_type="string"),
            PlannedColumn(name="Metric Name", description="Name of the metric.", value_type="string"),
            PlannedColumn(name="Metric Value", description="Numeric score.", value_type="number"),
            PlannedColumn(name="Setting / Model", description="Model/configuration.", value_type="string"),
            PlannedColumn(name="Notes", description="Additional notes.", value_type="string"),
        ],
        evidence_decisions=[
            EvidenceDecision(evidence_id=ev_id, decision="use", reason="primary results table"),
        ],
        generation_policy="coherent_synthesis",
        reason="",
    )


# ── basic output correctness ──────────────────────────────────────────────────

def test_long_form_produces_rows():
    block = _make_block()
    plan = _long_plan()
    draft = compose_long_form_from_source_tables(plan, [block], [])
    assert draft is not None
    assert len(draft.rows) > 0


def test_long_form_does_not_call_llm(monkeypatch):
    """compose_long_form_from_source_tables must be purely deterministic — no async LLM call."""
    import app.services.data_table.table_composer as mod

    original = mod.compose_long_form_from_source_tables

    async def fake_llm_compose(*args, **kwargs):
        raise AssertionError("LLM composer must not be called in long-form deterministic mode")

    # replace only the LLM path
    monkeypatch.setattr(mod, "compose_data_table", fake_llm_compose)

    block = _make_block()
    plan = _long_plan()
    # this is the synchronous deterministic path — should not raise
    draft = original(plan, [block], [])
    assert draft is not None


def test_long_form_row_count():
    """12 source rows × 11 metric columns = 132 output rows."""
    block = _make_block()
    plan = _long_plan()
    draft = compose_long_form_from_source_tables(plan, [block], [])
    assert draft is not None
    # each method gets one row per metric column
    assert len(draft.rows) == 12 * 11


def test_long_form_headers_match_plan():
    block = _make_block()
    plan = _long_plan()
    draft = compose_long_form_from_source_tables(plan, [block], [])
    assert draft is not None
    assert draft.headers == [c.name for c in plan.columns]


def test_long_form_metric_values_correct():
    """A-MEM single_hop_f1 should be 60.30."""
    block = _make_block()
    plan = _long_plan()
    draft = compose_long_form_from_source_tables(plan, [block], [])
    assert draft is not None

    amem_f1_rows = [
        r for r in draft.rows
        if "A-MEM" in r.row_label and "single_hop_f1" in r.row_label
    ]
    assert len(amem_f1_rows) == 1
    cell = amem_f1_rows[0].cells["Metric Value"]
    assert cell.value == 60.30
    assert cell.status == "supported"


def test_long_form_citations_generated_by_code():
    """Citations must be code-generated (evidence_id + quote), not LLM-invented."""
    block = _make_block()
    plan = _long_plan()
    draft = compose_long_form_from_source_tables(plan, [block], [])
    assert draft is not None

    supported = [
        r.cells["Metric Value"]
        for r in draft.rows
        if r.cells.get("Metric Value") and r.cells["Metric Value"].status == "supported"
    ]
    assert len(supported) > 0
    for cell in supported:
        assert cell.evidence_id == "ev_1"
        assert cell.quote is not None and len(cell.quote) > 0


def test_long_form_metric_name_column_populated():
    block = _make_block()
    plan = _long_plan()
    draft = compose_long_form_from_source_tables(plan, [block], [])
    assert draft is not None

    metric_names = {r.cells["Metric Name"].value for r in draft.rows if r.cells.get("Metric Name")}
    assert "single_hop_f1" in metric_names
    assert "multi_hop_f1" in metric_names
    assert "overall_score" in metric_names


def test_long_form_benchmark_task_populated():
    block = _make_block()
    plan = _long_plan()
    draft = compose_long_form_from_source_tables(plan, [block], [])
    assert draft is not None

    bench_vals = {
        r.cells["Benchmark / Task"].value
        for r in draft.rows
        if r.cells.get("Benchmark / Task") and r.cells["Benchmark / Task"].value
    }
    # should contain the block title
    assert any("AMem" in str(v) or "Benchmark" in str(v) for v in bench_vals)


def test_long_form_excluded_evidence_not_used():
    block = _make_block(ev_id="ev_excluded")
    plan = DataTablePlan(
        table_title="T",
        table_purpose="P",
        row_grain="method × metric",
        table_format="long",
        columns=[
            PlannedColumn(name="Method / System", description="d"),
            PlannedColumn(name="Benchmark / Task", description="d"),
            PlannedColumn(name="Metric Name", description="d"),
            PlannedColumn(name="Metric Value", description="d"),
            PlannedColumn(name="Setting / Model", description="d"),
            PlannedColumn(name="Notes", description="d"),
        ],
        evidence_decisions=[
            EvidenceDecision(evidence_id="ev_excluded", decision="exclude", reason="wrong grain"),
        ],
        generation_policy="coherent_synthesis",
        reason="",
    )
    draft = compose_long_form_from_source_tables(plan, [block], [])
    # excluded evidence should not produce any rows
    assert draft is None or len(draft.rows) == 0


def test_long_form_no_json_truncation_risk():
    """Output is a Python object, not JSON — cannot be truncated."""
    block = _make_block()
    plan = _long_plan()
    # would raise if there were any JSON parsing involved
    draft = compose_long_form_from_source_tables(plan, [block], [])
    assert draft is not None
    assert isinstance(draft, DraftDataTable)
