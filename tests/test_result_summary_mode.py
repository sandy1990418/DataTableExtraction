"""Tests for result_summary composition mode."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.data_table import EvidenceBlock, SourceRef
from app.services.data_table.table_composer import (
    RESULT_SUMMARY_HEADERS,
    _format_source_tables_for_summary,
    compose_result_summary,
)
from app.services.data_table.table_planner import (
    DataTablePlan,
    EvidenceDecision,
    PlannedColumn,
    _parse_plan,
)

MEMORY_TABLE_MD = """\
| method | single_hop_f1 | single_hop_bleu | multi_hop_f1 | multi_hop_bleu | temporal_f1 | temporal_bleu | overall_score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A-MEM | 60.30 | 0.41 | 44.27 | 0.30 | 31.20 | 0.22 | 46.47 |
| MemGPT | 1.18 | 0.01 | 1.18 | 0.01 | 2.00 | 0.02 | 1.84 |
| MemoryBank | 10.00 | 0.08 | 9.50 | 0.07 | 8.00 | 0.06 | 9.63 |
| MemoryOS | 23.00 | 0.18 | 20.00 | 0.15 | 21.00 | 0.16 | 21.50 |
"""


def _make_block(ev_id="ev_1", table_md=MEMORY_TABLE_MD, title="Memory Benchmark") -> EvidenceBlock:
    return EvidenceBlock(
        evidence_id=ev_id,
        source_id="src_1",
        kind="markdown_table",
        text="A-MEM achieves best overall score of 46.47 across all tasks.",
        table_markdown=table_md,
        title=title,
        source_ref=SourceRef(source_id="src_1", evidence_id=ev_id, kind="markdown_table"),
    )


def _summary_plan(ev_id: str = "ev_1") -> DataTablePlan:
    return DataTablePlan(
        table_title="Memory System Comparison",
        table_purpose="Summarize key experimental results for each memory system.",
        table_purpose_type="result_summary",
        row_grain="memory system",
        table_format="wide",
        columns=[PlannedColumn(name=h, description=h, value_type="string") for h in RESULT_SUMMARY_HEADERS],
        evidence_decisions=[EvidenceDecision(evidence_id=ev_id, decision="use", reason="primary table")],
        candidate_rows=["A-MEM", "MemGPT"],
        generation_policy="coherent_synthesis",
        reason="",
    )


# ── planner output parsing ────────────────────────────────────────────────────

def test_planner_parses_result_summary_purpose_type():
    data = {
        "table_title": "Memory Results",
        "table_purpose": "Summarize memory system benchmark results.",
        "table_purpose_type": "result_summary",
        "row_grain": "memory system",
        "table_format": "wide",
        "columns": [{"name": "Method / System", "description": "d", "value_type": "string", "evidence_policy": "mixed"}],
        "evidence_decisions": [],
        "excluded_source_tables": [],
        "candidate_rows": ["A-MEM", "MemGPT"],
        "excluded_candidate_rows": [],
        "generation_policy": "coherent_synthesis",
        "warnings": [],
        "reason": "Broad hint → result_summary.",
    }
    plan = _parse_plan(data)
    assert plan.table_purpose_type == "result_summary"


def test_planner_defaults_to_result_summary_on_unknown_purpose():
    data = {
        "table_title": "T",
        "table_purpose": "P",
        "table_purpose_type": "some_unknown_value",
        "row_grain": "method",
        "table_format": "wide",
        "columns": [{"name": "Method", "description": "d"}],
        "evidence_decisions": [],
        "excluded_source_tables": [],
        "candidate_rows": [],
        "excluded_candidate_rows": [],
        "generation_policy": "coherent_synthesis",
        "warnings": [],
        "reason": "",
    }
    plan = _parse_plan(data)
    assert plan.table_purpose_type == "result_summary"


def test_planner_parses_raw_metric_extraction():
    data = {
        "table_title": "All Metrics",
        "table_purpose": "Extract every metric value.",
        "table_purpose_type": "raw_metric_extraction",
        "row_grain": "method × metric",
        "table_format": "long",
        "columns": [{"name": "Method / System", "description": "d"}],
        "evidence_decisions": [],
        "excluded_source_tables": [],
        "candidate_rows": [],
        "excluded_candidate_rows": [],
        "generation_policy": "coherent_synthesis",
        "warnings": [],
        "reason": "",
    }
    plan = _parse_plan(data)
    assert plan.table_purpose_type == "raw_metric_extraction"
    assert plan.table_format == "long"


# ── result summary schema ─────────────────────────────────────────────────────

def test_result_summary_headers_constant():
    assert RESULT_SUMMARY_HEADERS == [
        "Method / System",
        "Main Benchmark / Task",
        "Representative Result",
        "Compared Against",
        "Key Takeaway",
        "Limitations / Notes",
        "Sources",
    ]


def test_result_summary_has_no_metric_name_metric_value_columns():
    assert "Metric Name" not in RESULT_SUMMARY_HEADERS
    assert "Metric Value" not in RESULT_SUMMARY_HEADERS


# ── LLM result summary composer ──────────────────────────────────────────────

def _make_row_dict(row_label: str, f1: str, bleu: str, ev_id: str = "ev_1") -> dict:
    """Single row dict used inside the single composer rows response."""
    return {
        "row_label": row_label,
        "cells": {
            "Method / System": {"value": row_label, "status": "supported", "evidence_id": ev_id, "quote": row_label},
            "F1 Score (%)": {"value": f1, "status": "supported", "evidence_id": ev_id, "quote": f1},
            "BLEU-1 (%)": {"value": bleu, "status": "supported", "evidence_id": ev_id, "quote": bleu},
            "Compared Against": {"value": "MemGPT, MemoryBank, MemoryOS", "status": "inferred",
                                 "evidence_id": ev_id, "quote": "MemGPT"},
            "Key Takeaway": {"value": f"{row_label} key finding.", "status": "inferred",
                             "evidence_id": ev_id, "quote": row_label},
        },
    }


def _make_rows_response(rows_data: list[dict]) -> dict:
    """Single composer response: all rows nested under one {"rows": [...]} object."""
    return {"rows": rows_data}


_DISCOVERED_COLS = ["Method / System", "F1 Score (%)", "BLEU-1 (%)", "Compared Against", "Key Takeaway"]


@pytest.fixture
def llm_result_summary_response():
    col_response = {"columns": _DISCOVERED_COLS}
    rows_response = _make_rows_response(
        [
            _make_row_dict("A-MEM", "44.27", "20.09"),
            _make_row_dict("MemGPT", "1.18", "0.01"),
        ],
    )
    return col_response, rows_response


def _make_mock_client(column_response: dict, rows_response: dict) -> AsyncMock:
    """Create a mock OpenAI client: call 1 returns column discovery, call 2 returns all rows."""
    mock_client = AsyncMock()
    col_resp = MagicMock()
    col_resp.choices[0].message.content = json.dumps(column_response)
    rows_resp = MagicMock()
    rows_resp.choices[0].message.content = json.dumps(rows_response)
    mock_client.chat.completions.create = AsyncMock(side_effect=[col_resp, rows_resp])
    return mock_client


@pytest.mark.asyncio
async def test_compose_result_summary_calls_llm_not_long_form(llm_result_summary_response):
    """compose_result_summary makes exactly 2 LLM calls: column discovery + one single rows call."""
    block = _make_block()
    plan = _summary_plan()

    from app.config import Settings
    settings = MagicMock(spec=Settings)
    settings.OPENAI_API_KEY = "test"
    settings.OPENAI_BASE_URL = None
    settings.OPENAI_MODEL = "gpt-4o-mini"

    col_response, rows_response = llm_result_summary_response
    with patch("app.services.data_table.table_composer.make_client") as mock_openai:
        mock_client = _make_mock_client(col_response, rows_response)
        mock_openai.return_value = mock_client

        draft = await compose_result_summary("Compare memory results", plan, [block], [], settings)

    assert mock_client.chat.completions.create.call_count == 2  # 1 discovery + 1 single rows call
    assert len(draft.rows) == 2
    assert "Method / System" in draft.headers


@pytest.mark.asyncio
async def test_compose_result_summary_row_grain_is_method_level(llm_result_summary_response):
    """Each row must represent one method/system, not one metric."""
    block = _make_block()
    plan = _summary_plan()

    from app.config import Settings
    settings = MagicMock(spec=Settings)
    settings.OPENAI_API_KEY = "test"
    settings.OPENAI_BASE_URL = None
    settings.OPENAI_MODEL = "gpt-4o-mini"

    col_response, rows_response = llm_result_summary_response
    with patch("app.services.data_table.table_composer.make_client") as mock_openai:
        mock_client = _make_mock_client(col_response, rows_response)
        mock_openai.return_value = mock_client

        draft = await compose_result_summary("Compare memory results", plan, [block], [], settings)

    row_labels = [r.row_label for r in draft.rows]
    assert "A-MEM" in row_labels
    assert "MemGPT" in row_labels
    assert not any("—" in label and any(m in label for m in ["f1", "bleu", "score"]) for label in row_labels)


@pytest.mark.asyncio
async def test_compose_result_summary_f1_cell_has_number(llm_result_summary_response):
    """Discovered metric column must contain a numeric value."""
    block = _make_block()
    plan = _summary_plan()

    from app.config import Settings
    settings = MagicMock(spec=Settings)
    settings.OPENAI_API_KEY = "test"
    settings.OPENAI_BASE_URL = None
    settings.OPENAI_MODEL = "gpt-4o-mini"

    col_response, rows_response = llm_result_summary_response
    with patch("app.services.data_table.table_composer.make_client") as mock_openai:
        mock_client = _make_mock_client(col_response, rows_response)
        mock_openai.return_value = mock_client

        draft = await compose_result_summary("Compare memory results", plan, [block], [], settings)

    amem_row = next(r for r in draft.rows if r.row_label == "A-MEM")
    f1_cell = amem_row.cells.get("F1 Score (%)")
    assert f1_cell is not None
    assert any(c.isdigit() for c in str(f1_cell.value or ""))


@pytest.mark.asyncio
async def test_compose_result_summary_headers_discovered_from_data(llm_result_summary_response):
    """Headers are discovered from the LLM responses, not taken from a fixed schema."""
    block = _make_block()
    plan = _summary_plan()

    from app.config import Settings
    settings = MagicMock(spec=Settings)
    settings.OPENAI_API_KEY = "test"
    settings.OPENAI_BASE_URL = None
    settings.OPENAI_MODEL = "gpt-4o-mini"

    col_response, rows_response = llm_result_summary_response
    with patch("app.services.data_table.table_composer.make_client") as mock_openai:
        mock_client = _make_mock_client(col_response, rows_response)
        mock_openai.return_value = mock_client

        draft = await compose_result_summary("Compare memory results", plan, [block], [], settings)

    # Headers should be discovered from Phase 1 column discovery
    assert "Method / System" in draft.headers
    assert "F1 Score (%)" in draft.headers
    assert len(draft.headers) >= 3


# ── format_source_tables_for_summary ────────────────────────────────────────

def test_format_source_tables_includes_full_markdown():
    block = _make_block()
    result = _format_source_tables_for_summary([block], {"ev_1"})
    assert "method" in result
    assert "A-MEM" in result
    assert "46.47" in result


def test_format_source_tables_excludes_non_allowed():
    block = _make_block()
    result = _format_source_tables_for_summary([block], set())  # empty allowed set
    assert result == "(no source tables)"


# ── schema vs long-form distinction ──────────────────────────────────────────

def test_result_summary_plan_does_not_route_to_long_form():
    """A result_summary plan should never route to compose_long_form_from_source_tables."""
    plan = _summary_plan()
    assert plan.table_purpose_type == "result_summary"
    assert plan.table_format != "long" or plan.table_purpose_type != "raw_metric_extraction"


def test_raw_metric_extraction_plan_routes_to_long_form():
    from app.services.data_table.table_planner import _parse_plan
    data = {
        "table_title": "All Metrics",
        "table_purpose": "Extract all raw metrics.",
        "table_purpose_type": "raw_metric_extraction",
        "row_grain": "method × metric",
        "table_format": "long",
        "columns": [{"name": "Method / System", "description": "d"}],
        "evidence_decisions": [],
        "excluded_source_tables": [],
        "candidate_rows": [],
        "excluded_candidate_rows": [],
        "generation_policy": "coherent_synthesis",
        "warnings": [],
        "reason": "",
    }
    plan = _parse_plan(data)
    # pipeline routes: table_purpose_type == raw_metric_extraction AND table_format == long
    assert plan.table_purpose_type == "raw_metric_extraction"
    assert plan.table_format == "long"
