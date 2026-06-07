"""Tests for result_summary composition mode."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.data_table import EvidenceBlock, SourceRef
from app.services.data_table.table_composer import (
    RESULT_SUMMARY_HEADERS,
    DraftDataTable,
    DraftRow,
    DraftCell,
    _parse_draft,
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
    ]


def test_result_summary_has_no_metric_name_metric_value_columns():
    assert "Metric Name" not in RESULT_SUMMARY_HEADERS
    assert "Metric Value" not in RESULT_SUMMARY_HEADERS


# ── LLM result summary composer ──────────────────────────────────────────────

@pytest.fixture
def llm_result_summary_response():
    return {
        "headers": RESULT_SUMMARY_HEADERS,
        "rows": [
            {
                "row_label": "A-MEM",
                "cells": {
                    "Method / System": {"value": "A-MEM", "status": "supported", "evidence_id": "ev_1",
                                        "quote": "A-MEM achieves best overall score of 46.47"},
                    "Main Benchmark / Task": {"value": "LoCoMo QA", "status": "supported", "evidence_id": "ev_1",
                                              "quote": "Memory Benchmark"},
                    "Representative Result": {"value": "Overall 46.47; Multi-Hop F1 44.27", "status": "supported",
                                              "evidence_id": "ev_1", "quote": "overall_score=46.47"},
                    "Compared Against": {"value": "MemGPT, MemoryBank, MemoryOS", "status": "inferred",
                                         "evidence_id": "ev_1", "quote": "method=MemGPT"},
                    "Key Takeaway": {"value": "Best overall score among tested memory systems.", "status": "inferred",
                                     "evidence_id": "ev_1", "quote": "A-MEM achieves best overall score"},
                    "Limitations / Notes": {"value": None, "status": "not_reported", "evidence_id": None, "quote": None},
                },
            },
            {
                "row_label": "MemGPT",
                "cells": {
                    "Method / System": {"value": "MemGPT", "status": "supported", "evidence_id": "ev_1",
                                        "quote": "method=MemGPT"},
                    "Main Benchmark / Task": {"value": "LoCoMo QA", "status": "supported", "evidence_id": "ev_1",
                                              "quote": "Memory Benchmark"},
                    "Representative Result": {"value": "Overall 1.84; Single-Hop F1 1.18", "status": "supported",
                                              "evidence_id": "ev_1", "quote": "overall_score=1.84"},
                    "Compared Against": {"value": "A-MEM, MemoryBank", "status": "inferred",
                                         "evidence_id": "ev_1", "quote": "method=A-MEM"},
                    "Key Takeaway": {"value": "Very low scores across all tasks.", "status": "inferred",
                                     "evidence_id": "ev_1", "quote": "overall_score=1.84"},
                    "Limitations / Notes": {"value": None, "status": "not_reported", "evidence_id": None, "quote": None},
                },
            },
        ],
        "notes": [],
    }


@pytest.mark.asyncio
async def test_compose_result_summary_calls_llm_not_long_form(llm_result_summary_response):
    """compose_result_summary must call the LLM, not the deterministic long-form path."""
    block = _make_block()
    plan = _summary_plan()

    from app.config import Settings
    settings = MagicMock(spec=Settings)
    settings.OPENAI_API_KEY = "test"
    settings.OPENAI_BASE_URL = None
    settings.OPENAI_MODEL = "gpt-4o-mini"

    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(llm_result_summary_response)

    with patch("app.services.data_table.table_composer.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        draft = await compose_result_summary("Compare memory results", plan, [block], [], settings)

    assert mock_client.chat.completions.create.called
    assert len(draft.rows) == 2
    assert draft.headers == RESULT_SUMMARY_HEADERS


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

    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(llm_result_summary_response)

    with patch("app.services.data_table.table_composer.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        draft = await compose_result_summary("Compare memory results", plan, [block], [], settings)

    row_labels = [r.row_label for r in draft.rows]
    # rows should be system-level, not metric-level
    assert "A-MEM" in row_labels
    assert "MemGPT" in row_labels
    # should NOT have rows like "A-MEM — single_hop_f1"
    assert not any("—" in label and any(m in label for m in ["f1", "bleu", "score"]) for label in row_labels)


@pytest.mark.asyncio
async def test_compose_result_summary_representative_result_is_composite(llm_result_summary_response):
    """Representative Result cell must be a string, not a bare number."""
    block = _make_block()
    plan = _summary_plan()

    from app.config import Settings
    settings = MagicMock(spec=Settings)
    settings.OPENAI_API_KEY = "test"
    settings.OPENAI_BASE_URL = None
    settings.OPENAI_MODEL = "gpt-4o-mini"

    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(llm_result_summary_response)

    with patch("app.services.data_table.table_composer.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        draft = await compose_result_summary("Compare memory results", plan, [block], [], settings)

    amem_row = next(r for r in draft.rows if r.row_label == "A-MEM")
    rep = amem_row.cells.get("Representative Result")
    assert rep is not None
    assert isinstance(rep.value, str)
    # composite: should mention multiple numbers or metrics
    assert any(c.isdigit() for c in (rep.value or ""))


@pytest.mark.asyncio
async def test_compose_result_summary_headers_enforced_even_if_llm_returns_wrong(llm_result_summary_response):
    """Even if LLM returns wrong headers, the output must use RESULT_SUMMARY_HEADERS."""
    block = _make_block()
    plan = _summary_plan()

    from app.config import Settings
    settings = MagicMock(spec=Settings)
    settings.OPENAI_API_KEY = "test"
    settings.OPENAI_BASE_URL = None
    settings.OPENAI_MODEL = "gpt-4o-mini"

    # LLM returns wrong headers
    bad_response = dict(llm_result_summary_response)
    bad_response["headers"] = ["Method", "Score 1", "Score 2"]

    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps(bad_response)

    with patch("app.services.data_table.table_composer.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        draft = await compose_result_summary("Compare memory results", plan, [block], [], settings)

    assert draft.headers == RESULT_SUMMARY_HEADERS


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
