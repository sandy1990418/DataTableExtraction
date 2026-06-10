"""Tests for benchmark/result table mode."""

from __future__ import annotations


from app.models.data_table import (
    DataTableColumn,
    EvidenceBlock,
    RowEntity,
    SourceRef,
)
from app.services.data_table.strategy import select_strategy
from app.services.data_table.source_table_rows import (
    extract_source_table_candidates,
    parse_markdown_table,
)
from app.services.data_table.entity_discovery import _discover_entities_from_source_tables
from app.services.data_table.source_table_populator import fill_cell_from_source_table

SAMPLE_TABLE_MD = """\
| method | multi_hop | temporal | open_domain | single_hop | overall_score |
| --- | ---: | ---: | ---: | ---: | ---: |
| A-Mem | 44.27 | 31.20 | 50.10 | 60.30 | 46.47 |
| MemGPT | 1.18 | 2.00 | 3.00 | 1.18 | 1.84 |
| Zep | 20.00 | 21.00 | 22.00 | 23.00 | 21.50 |
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


# ── intent heuristic ──────────────────────────────────────────────────────────

def test_short_hint_detects_result_mode():
    block = _make_block()
    strategy, _ = select_strategy("Compare these memory experiment result", [block])
    assert strategy == "source_table_reconstruction"


def test_architecture_hint_not_result_mode():
    block = _make_block()
    strategy, _ = select_strategy(
        "Compare these memory systems by architecture design and retrieval mechanism", [block]
    )
    assert strategy != "source_table_reconstruction"


def test_benchmark_keyword_detected():
    block = _make_block()
    strategy, _ = select_strategy("Show me benchmark scores", [block])
    assert strategy == "source_table_reconstruction"


# ── markdown table parsing ────────────────────────────────────────────────────

def test_parse_markdown_table():
    headers, rows = parse_markdown_table(SAMPLE_TABLE_MD)
    assert headers == ["method", "multi_hop", "temporal", "open_domain", "single_hop", "overall_score"]
    assert len(rows) == 3
    assert rows[0][0] == "A-Mem"
    assert rows[0][1] == "44.27"
    assert rows[2][4] == "23.00"


def test_parse_empty_table():
    headers, rows = parse_markdown_table("not a table")
    assert headers == []
    assert rows == []


# ── source table candidates ───────────────────────────────────────────────────

def test_schema_from_result_table():
    block = _make_block()
    candidates = extract_source_table_candidates([block], hint="experiment result")
    assert len(candidates) == 1
    assert candidates[0].headers == ["method", "multi_hop", "temporal", "open_domain", "single_hop", "overall_score"]
    assert candidates[0].score > 5.0


def test_entities_from_source_table_rows():
    block = _make_block()
    entities = _discover_entities_from_source_tables([block], hint="experiment result", max_rows=10)
    names = [e.name for e in entities]
    assert "A-Mem" in names
    assert "MemGPT" in names
    assert "Zep" in names
    assert len(entities) == 3


# ── deterministic cell filling ────────────────────────────────────────────────

def _entity(name: str, ev_id: str = "ev_1") -> RowEntity:
    return RowEntity(
        entity_id="ent_1",
        name=name,
        source_refs=[SourceRef(source_id="src_1", evidence_id=ev_id, kind="markdown_table")],
        confidence=0.9,
    )


def _col(name: str, role="metric", vtype="number") -> DataTableColumn:
    return DataTableColumn(name=name, role=role, description=f"{name} score.", value_type=vtype)


def test_cells_filled_from_source_table():
    block = _make_block()
    entity = _entity("A-Mem")

    cell = fill_cell_from_source_table(entity, _col("multi_hop"), [block])
    assert cell is not None
    assert cell.value == 44.27
    assert cell.status == "supported"
    assert cell.citations

    cell2 = fill_cell_from_source_table(entity, _col("temporal"), [block])
    assert cell2 is not None
    assert cell2.value == 31.20

    cell3 = fill_cell_from_source_table(_entity("MemGPT"), _col("open_domain"), [block])
    assert cell3 is not None
    assert cell3.value == 3.0

    cell4 = fill_cell_from_source_table(_entity("Zep"), _col("overall_score"), [block])
    assert cell4 is not None
    assert cell4.value == 21.5


def test_no_sparse_result_table():
    block = _make_block()
    entities = _discover_entities_from_source_tables([block], hint="experiment result", max_rows=10)

    total_cells = 0
    supported_cells = 0
    for entity in entities:
        for col_name in ["multi_hop", "temporal", "open_domain", "single_hop", "overall_score"]:
            col = _col(col_name)
            cell = fill_cell_from_source_table(entity, col, [block])
            total_cells += 1
            if cell and cell.status == "supported":
                supported_cells += 1

    ratio = supported_cells / total_cells
    assert ratio > 0.8, f"Supported ratio too low: {ratio:.1%}"
