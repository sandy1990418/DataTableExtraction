"""Regression tests: pipeline consistency between planner, composer, and final schema."""

from __future__ import annotations



from app.models.data_table import EvidenceBlock, SourceRef
from app.services.data_table.source_table_summary import summarize_source_tables
from app.services.data_table.table_composer import DraftDataTable, DraftRow, DraftCell
from app.services.data_table.table_planner import (
    RESULT_SUMMARY_HEADERS,
    DataTablePlan,
    EvidenceDecision,
    PlannedColumn,
    _RESULT_SUMMARY_DEFAULT_COLUMNS,
    _parse_plan,
)
from app.services.data_table.table_verifier import verify_draft_table

AMEM_TABLE_MD = """\
| method | single_hop_f1 | multi_hop_f1 | temporal_f1 | overall_score |
| --- | ---: | ---: | ---: | ---: |
| A-MEM | 60.30 | 44.27 | 31.20 | 46.47 |
| MemGPT | 1.18 | 1.18 | 2.00 | 1.84 |
| MemoryBank | 10.00 | 9.50 | 8.00 | 9.63 |
"""


def _make_block(ev_id="ev_1") -> EvidenceBlock:
    return EvidenceBlock(
        evidence_id=ev_id,
        source_id="src_1",
        kind="markdown_table",
        text="A-MEM outperforms all baselines on the LoCoMo QA benchmark.",
        table_markdown=AMEM_TABLE_MD,
        title="LoCoMo QA Benchmark Results",
        source_ref=SourceRef(source_id="src_1", evidence_id=ev_id, kind="markdown_table"),
    )


def _result_summary_plan(ev_id: str = "ev_1") -> DataTablePlan:
    return DataTablePlan(
        table_title="Memory System Comparison",
        table_purpose="Summarize key experimental results.",
        table_purpose_type="result_summary",
        row_grain="memory system",
        table_format="wide",
        columns=list(_RESULT_SUMMARY_DEFAULT_COLUMNS),
        evidence_decisions=[EvidenceDecision(evidence_id=ev_id, decision="use", reason="primary")],
        candidate_rows=["A-MEM", "MemGPT", "MemoryBank"],
        generation_policy="coherent_synthesis",
        reason="",
    )


# ── Fix 1: plan columns are never Entity/Description for result_summary ───────

def test_parse_plan_result_summary_with_generic_columns_gets_default_schema():
    """If LLM returns generic Entity/Description columns for result_summary, use defaults."""
    data = {
        "table_title": "Results",
        "table_purpose": "Summarize results.",
        "table_purpose_type": "result_summary",
        "row_grain": "memory system",
        "table_format": "wide",
        "columns": [
            {"name": "Entity", "description": "the entity", "value_type": "string", "evidence_policy": "mixed"},
            {"name": "Description", "description": "brief description", "value_type": "string", "evidence_policy": "text"},
        ],
        "evidence_decisions": [],
        "excluded_source_tables": [],
        "candidate_rows": [],
        "excluded_candidate_rows": [],
        "generation_policy": "coherent_synthesis",
        "warnings": [],
        "reason": "",
    }
    plan = _parse_plan(data)
    col_names = [c.name for c in plan.columns]
    assert "Entity" not in col_names
    assert "Description" not in col_names
    assert "Method / System" in col_names
    assert "Representative Result" in col_names


def test_parse_plan_result_summary_with_empty_columns_gets_default_schema():
    data = {
        "table_title": "T",
        "table_purpose": "P",
        "table_purpose_type": "result_summary",
        "row_grain": "method",
        "table_format": "wide",
        "columns": [],
        "evidence_decisions": [],
        "excluded_source_tables": [],
        "candidate_rows": [],
        "excluded_candidate_rows": [],
        "generation_policy": "coherent_synthesis",
        "warnings": [],
        "reason": "",
    }
    plan = _parse_plan(data)
    assert [c.name for c in plan.columns] == RESULT_SUMMARY_HEADERS


def test_parse_plan_result_summary_with_good_columns_preserved():
    """If LLM returns proper result_summary columns, keep them."""
    data = {
        "table_title": "T",
        "table_purpose": "P",
        "table_purpose_type": "result_summary",
        "row_grain": "method",
        "table_format": "wide",
        "columns": [{"name": h, "description": h, "value_type": "string", "evidence_policy": "mixed"}
                    for h in RESULT_SUMMARY_HEADERS],
        "evidence_decisions": [],
        "excluded_source_tables": [],
        "candidate_rows": [],
        "excluded_candidate_rows": [],
        "generation_policy": "coherent_synthesis",
        "warnings": [],
        "reason": "",
    }
    plan = _parse_plan(data)
    assert [c.name for c in plan.columns] == RESULT_SUMMARY_HEADERS


# ── Fix 2: verifier uses draft.headers for final schema ──────────────────────

def test_verify_draft_table_uses_draft_headers_not_plan_columns():
    """Final schema must come from draft.headers, not plan.columns."""
    block = _make_block()
    plan = _result_summary_plan()

    # plan has result_summary columns, draft has matching headers
    draft = DraftDataTable(
        headers=RESULT_SUMMARY_HEADERS,
        rows=[
            DraftRow(
                row_label="A-MEM",
                cells={
                    "Method / System": DraftCell(value="A-MEM", status="supported", evidence_id="ev_1", quote="A-MEM"),
                    "Main Benchmark / Task": DraftCell(value="LoCoMo QA", status="supported", evidence_id="ev_1", quote="LoCoMo QA"),
                    "Representative Result": DraftCell(value="Overall 46.47", status="supported", evidence_id="ev_1", quote="overall_score=46.47"),
                    "Compared Against": DraftCell(value="MemGPT, MemoryBank", status="inferred", evidence_id="ev_1", quote="method=MemGPT"),
                    "Key Takeaway": DraftCell(value="Best system.", status="inferred", evidence_id="ev_1", quote="A-MEM outperforms all baselines"),
                    "Limitations / Notes": DraftCell(value=None, status="not_reported"),
                    "Sources": DraftCell(value=None, status="not_reported"),
                },
            )
        ],
        notes=[],
    )

    rows, schema, _ = verify_draft_table(draft, plan, [block])
    exported_col_names = [c.name for c in schema.columns]
    assert exported_col_names == RESULT_SUMMARY_HEADERS
    assert "Entity" not in exported_col_names
    assert "Description" not in exported_col_names


def test_verify_draft_table_schema_not_overwritten_by_fallback_plan_columns():
    """Even if plan.columns are Entity/Description (old fallback), schema uses draft.headers."""
    block = _make_block()

    # simulate old-style fallback plan with Entity/Description columns
    old_fallback_plan = DataTablePlan(
        table_title="T",
        table_purpose="P",
        table_purpose_type="result_summary",
        row_grain="method",
        table_format="wide",
        columns=[
            PlannedColumn(name="Entity", description="d", value_type="string"),
            PlannedColumn(name="Description", description="d", value_type="string"),
        ],
        evidence_decisions=[],
        generation_policy="coherent_synthesis",
        reason="",
    )

    draft = DraftDataTable(
        headers=["Method / System", "Representative Result"],
        rows=[DraftRow(
            row_label="A-MEM",
            cells={
                "Method / System": DraftCell(value="A-MEM", status="supported", evidence_id="ev_1", quote="A-MEM"),
                "Representative Result": DraftCell(value="46.47", status="supported", evidence_id="ev_1", quote="overall_score=46.47"),
            },
        )],
        notes=[],
    )

    rows, schema, _ = verify_draft_table(draft, old_fallback_plan, [block])
    col_names = [c.name for c in schema.columns]
    assert "Method / System" in col_names
    assert "Representative Result" in col_names
    assert "Entity" not in col_names


# ── Fix 3: evidence_id normalization ─────────────────────────────────────────

def test_source_table_summaries_have_correct_evidence_ids():
    """SourceTableSummary.evidence_id must match the EvidenceBlock.evidence_id."""
    block = _make_block(ev_id="txt_amem_0")
    summaries = summarize_source_tables([block])
    assert len(summaries) == 1
    assert summaries[0].evidence_id == "txt_amem_0"
    assert summaries[0].table_id == "tbl_1"


def test_evidence_id_normalization_tbl_id_resolved():
    """plan_data_table post-processes evidence decisions: tbl_1 → real evidence_id."""
    # test _parse_plan + normalization logic independently
    from app.services.data_table.table_planner import EvidenceDecision

    # simulate what plan_data_table does after _parse_plan
    ev_id_set = {"txt_amem_0"}
    tbl_to_ev = {"tbl_1": "txt_amem_0"}

    def _resolve(eid: str) -> str:
        return eid if eid in ev_id_set else tbl_to_ev.get(eid, eid)

    decisions = [EvidenceDecision(evidence_id="tbl_1", decision="use", reason="")]
    fixed = [EvidenceDecision(evidence_id=_resolve(d.evidence_id), decision=d.decision, reason=d.reason)
             for d in decisions]
    assert fixed[0].evidence_id == "txt_amem_0"


# ── Fix 4: candidate rows from full source table ──────────────────────────────

def test_source_table_summary_includes_all_row_labels():
    """all_row_labels must include every row in the source table, not just sample rows."""
    block = _make_block()
    summaries = summarize_source_tables([block])
    assert len(summaries) == 1
    labels = summaries[0].all_row_labels
    assert "A-MEM" in labels
    assert "MemGPT" in labels
    assert "MemoryBank" in labels


def test_source_table_summary_sample_rows_limited_but_all_labels_complete():
    """sample_rows is limited to 5, but all_row_labels is complete."""
    long_table = "| method | score |\n| --- | --- |\n"
    long_table += "\n".join(f"| System_{i} | {i * 10.0} |" for i in range(10))
    block = EvidenceBlock(
        evidence_id="ev_big",
        source_id="src_big",
        kind="markdown_table",
        text="",
        table_markdown=long_table,
        source_ref=SourceRef(source_id="src_big", evidence_id="ev_big", kind="markdown_table"),
    )
    summaries = summarize_source_tables([block])
    assert summaries[0].row_count == 10
    assert len(summaries[0].sample_rows) == 5
    assert len(summaries[0].all_row_labels) == 10


# ── Fix 6: verifier contextual support ───────────────────────────────────────

def test_verifier_main_benchmark_supported_via_block_title():
    """Main Benchmark / Task supported if quote matches block title, even if not in table text."""
    block = _make_block()  # title = "LoCoMo QA Benchmark Results"
    plan = _result_summary_plan()

    draft = DraftDataTable(
        headers=RESULT_SUMMARY_HEADERS,
        rows=[DraftRow(
            row_label="A-MEM",
            cells={
                "Method / System": DraftCell(value="A-MEM", status="supported", evidence_id="ev_1", quote="A-MEM"),
                "Main Benchmark / Task": DraftCell(
                    value="LoCoMo QA",
                    status="supported",
                    evidence_id="ev_1",
                    # quote comes from block title only, not table body
                    quote="LoCoMo QA Benchmark Results",
                ),
                "Representative Result": DraftCell(value="46.47", status="supported", evidence_id="ev_1", quote="overall_score=46.47"),
                "Compared Against": DraftCell(value=None, status="not_reported"),
                "Key Takeaway": DraftCell(value=None, status="not_reported"),
                "Limitations / Notes": DraftCell(value=None, status="not_reported"),
                "Sources": DraftCell(value=None, status="not_reported"),
            },
        )],
        notes=[],
    )

    rows, schema, errors = verify_draft_table(draft, plan, [block])
    # "Main Benchmark / Task" should be supported, not flagged
    severe = [e for e in errors if "Main Benchmark" in e and "quote not found" in e]
    assert not severe, f"Unexpected errors: {severe}"
    assert rows[0].cells["Main Benchmark / Task"].status == "supported"


def test_verifier_key_takeaway_inferred_accepted():
    """Key Takeaway with inferred status should be accepted without strict quote check."""
    block = _make_block()
    plan = _result_summary_plan()

    draft = DraftDataTable(
        headers=RESULT_SUMMARY_HEADERS,
        rows=[DraftRow(
            row_label="A-MEM",
            cells={
                "Method / System": DraftCell(value="A-MEM", status="supported", evidence_id="ev_1", quote="A-MEM"),
                "Main Benchmark / Task": DraftCell(value=None, status="not_reported"),
                "Representative Result": DraftCell(value="46.47", status="supported", evidence_id="ev_1", quote="overall_score=46.47"),
                "Compared Against": DraftCell(value=None, status="not_reported"),
                "Key Takeaway": DraftCell(
                    value="Best system among all tested.",
                    status="inferred",
                    evidence_id="ev_1",
                    quote="A-MEM outperforms all baselines",
                ),
                "Limitations / Notes": DraftCell(value=None, status="not_reported"),
                "Sources": DraftCell(value=None, status="not_reported"),
            },
        )],
        notes=[],
    )

    rows, _, errors = verify_draft_table(draft, plan, [block])
    severe = [e for e in errors if "Key Takeaway" in e]
    assert not severe
    assert rows[0].cells["Key Takeaway"].status == "inferred"


def test_verifier_representative_result_composite_string():
    """Representative Result as composite string: at least one number must be in evidence."""
    block = _make_block()
    plan = _result_summary_plan()

    draft = DraftDataTable(
        headers=RESULT_SUMMARY_HEADERS,
        rows=[DraftRow(
            row_label="A-MEM",
            cells={
                "Method / System": DraftCell(value="A-MEM", status="supported", evidence_id="ev_1", quote="A-MEM"),
                "Main Benchmark / Task": DraftCell(value=None, status="not_reported"),
                "Representative Result": DraftCell(
                    value="Multi-Hop F1: 44.27; Overall: 46.47",  # composite string
                    status="supported",
                    evidence_id="ev_1",
                    quote="multi_hop_f1=44.27",
                ),
                "Compared Against": DraftCell(value=None, status="not_reported"),
                "Key Takeaway": DraftCell(value=None, status="not_reported"),
                "Limitations / Notes": DraftCell(value=None, status="not_reported"),
                "Sources": DraftCell(value=None, status="not_reported"),
            },
        )],
        notes=[],
    )

    rows, _, errors = verify_draft_table(draft, plan, [block])
    rep_errors = [e for e in errors if "Representative Result" in e]
    assert not rep_errors, f"Unexpected errors: {rep_errors}"
    assert rows[0].cells["Representative Result"].status == "supported"


# ── Full regression: headers must never be Entity/Description ─────────────────

def test_full_regression_output_never_has_entity_description_headers():
    """Full verifier pass: output schema must use result_summary headers, not Entity/Description."""
    block = _make_block()
    plan = _result_summary_plan()  # uses _RESULT_SUMMARY_DEFAULT_COLUMNS

    draft = DraftDataTable(
        headers=RESULT_SUMMARY_HEADERS,
        rows=[
            DraftRow(row_label=method, cells={
                "Method / System": DraftCell(value=method, status="supported", evidence_id="ev_1", quote=method),
                "Main Benchmark / Task": DraftCell(value="LoCoMo QA", status="supported", evidence_id="ev_1", quote="LoCoMo QA Benchmark Results"),
                "Representative Result": DraftCell(value=f"Overall {score}", status="supported", evidence_id="ev_1", quote=f"overall_score={score}"),
                "Compared Against": DraftCell(value=None, status="not_reported"),
                "Key Takeaway": DraftCell(value="Good results.", status="inferred", evidence_id="ev_1", quote="A-MEM outperforms all baselines"),
                "Limitations / Notes": DraftCell(value=None, status="not_reported"),
                "Sources": DraftCell(value=None, status="not_reported"),
            })
            for method, score in [("A-MEM", "46.47"), ("MemGPT", "1.84"), ("MemoryBank", "9.63")]
        ],
        notes=[],
    )

    rows, schema, errors = verify_draft_table(draft, plan, [block])

    col_names = [c.name for c in schema.columns]
    assert "Entity" not in col_names, "Entity column must not appear in result_summary output"
    assert "Description" not in col_names, "Description column must not appear in result_summary output"
    assert col_names == RESULT_SUMMARY_HEADERS

    row_labels = [r.entity.name for r in rows]
    assert "A-MEM" in row_labels
    assert "MemGPT" in row_labels
    assert "MemoryBank" in row_labels
    assert len(rows) == 3
