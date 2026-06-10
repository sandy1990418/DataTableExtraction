"""Tests for data table exporters and metrics."""

from __future__ import annotations

from app.models.data_table import (
    CellCitation,
    DataTableColumn,
    DataTableSchema,
    GroundedCell,
    GroundedDataTable,
    GroundedRow,
    RowEntity,
    SourceRef,
)
from app.services.data_table.exporters import (
    compute_table_metrics,
    to_citation_table,
    to_debug_json,
    to_pptx_table,
    to_simple_table,
)


def _make_table() -> GroundedDataTable:
    schema = DataTableSchema(
        title="Speech Models",
        intent="Compare speech models",
        columns=[
            DataTableColumn(name="Model", role="entity", description="Model name", required=True),
            DataTableColumn(name="Streaming", role="attribute", description="Supports streaming", value_type="boolean"),
            DataTableColumn(name="BLEU", role="metric", description="BLEU score", value_type="number"),
        ],
    )
    src_ref = SourceRef(source_id="src_1", evidence_id="ev_1", kind="text_fact")
    citation = CellCitation(source_ref=src_ref, quote="Supports streaming speech interaction.", support_type="direct")

    row1 = GroundedRow(
        entity=RowEntity(entity_id="ent_1", name="ModelA", source_refs=[src_ref], confidence=0.9),
        cells={
            "Model": GroundedCell(value="ModelA", status="supported", confidence=0.9),
            "Streaming": GroundedCell(value="Yes", status="supported", citations=[citation], confidence=0.9),
            "BLEU": GroundedCell(value=45.6, status="supported", citations=[citation], confidence=0.85),
        },
    )
    row2 = GroundedRow(
        entity=RowEntity(entity_id="ent_2", name="ModelB", source_refs=[], confidence=0.5),
        cells={
            "Model": GroundedCell(value="ModelB", status="supported", confidence=0.5),
            "Streaming": GroundedCell(value=None, status="not_reported"),
            "BLEU": GroundedCell(value=None, status="not_reported"),
        },
    )
    table = GroundedDataTable(**{"schema": schema, "rows": [row1, row2]})
    table.metrics = compute_table_metrics(table)
    return table


def test_main_table_export():
    table = _make_table()
    simple = to_simple_table(table)
    assert simple["headers"] == ["Model", "Streaming", "BLEU"]
    assert len(simple["rows"]) == 2
    assert simple["rows"][0][0] == "ModelA"
    assert simple["rows"][1][1] == ""  # not_reported → empty


def test_pptx_table_export():
    pptx_table = to_pptx_table(_make_table())

    assert pptx_table == {
        "table_id": "data_table",
        "title": "Speech Models",
        "kind": "extracted_summary",
        "headers": ["Model", "Streaming", "BLEU"],
        "rows": [["ModelA", "Yes", "45.6"], ["ModelB", "", ""]],
        "layout": "table_only",
    }


def test_citation_table_export():
    table = _make_table()
    cit = to_citation_table(table)
    assert "row_entity" in cit["headers"]
    assert "quote" in cit["headers"]
    # ModelA has 2 cited cells (Streaming + BLEU)
    entity_rows = [r for r in cit["rows"] if r[0] == "ModelA" and r[6]]
    assert len(entity_rows) >= 2


def test_debug_json_export():
    table = _make_table()
    debug = to_debug_json(table)
    assert "schema" in debug
    assert "metrics" in debug
    assert "rows" in debug
    assert debug["schema"]["title"] == "Speech Models"


def test_metrics_are_correct():
    table = _make_table()
    m = table.metrics
    assert m["row_count"] == 2
    assert m["column_count"] == 3
    assert m["cell_count"] == 6
    # ModelA: 3 supported, ModelB: 1 supported (entity col) + 2 not_reported
    assert m["supported_cell_count"] == 4
    assert m["not_reported_cell_count"] == 2
    assert m["citation_count"] == 2
    assert m["supported_cell_ratio"] == round(4 / 6, 4)
