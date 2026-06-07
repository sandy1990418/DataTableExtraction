"""Tests for DataTableExtraction v0.2 Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.data_table import (
    DataTableColumn,
    DataTableSchema,
    GroundedCell,
    GroundedDataTable,
    GroundedRow,
    RowEntity,
    SourceRef,
)


def make_source_ref(**kw) -> SourceRef:
    defaults = dict(source_id="src_1", evidence_id="ev_1", kind="text_fact")
    return SourceRef(**{**defaults, **kw})


def make_schema(**kw) -> DataTableSchema:
    defaults = dict(
        title="Test Table",
        intent="Compare things",
        columns=[
            DataTableColumn(name="Entity", role="entity", description="The entity", required=True),
            DataTableColumn(name="Score", role="metric", description="A score", value_type="number"),
        ],
    )
    return DataTableSchema(**{**defaults, **kw})


def make_cell(**kw) -> GroundedCell:
    defaults = dict(value="Yes", status="supported", confidence=0.9)
    return GroundedCell(**{**defaults, **kw})


def test_schema_validates():
    schema = make_schema()
    assert schema.title == "Test Table"
    assert len(schema.columns) == 2
    assert schema.columns[0].role == "entity"


def test_schema_column_requires_description():
    with pytest.raises(ValidationError):
        DataTableColumn(name="X", role="metric")  # missing description


def test_schema_column_no_empty_description():
    col = DataTableColumn(name="X", role="metric", description="Something")
    assert col.description == "Something"


def test_grounded_cell_validates():
    cell = make_cell()
    assert cell.status == "supported"
    assert cell.value == "Yes"


def test_grounded_cell_bad_status():
    with pytest.raises(ValidationError):
        GroundedCell(value="x", status="INVALID")


def test_grounded_data_table_serializes():
    schema = make_schema()
    entity = RowEntity(
        entity_id="ent_1",
        name="ModelA",
        source_refs=[make_source_ref()],
        confidence=0.8,
    )
    cell = make_cell()
    row = GroundedRow(entity=entity, cells={"Entity": cell, "Score": make_cell(value=42, status="supported")})
    table = GroundedDataTable(**{"schema": schema, "rows": [row]})
    d = table.model_dump(by_alias=True)
    assert d["schema"]["title"] == "Test Table"
    assert len(d["rows"]) == 1
    assert d["rows"][0]["entity"]["name"] == "ModelA"


def test_grounded_data_table_empty():
    schema = make_schema()
    table = GroundedDataTable(**{"schema": schema})
    assert table.rows == []
    assert table.warnings == []
    assert table.metrics == {}
