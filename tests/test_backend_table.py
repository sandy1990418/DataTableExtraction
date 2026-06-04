from __future__ import annotations

import io
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
from pptx import Presentation
import pytest

from app.main import create_app
from app.config import Settings
from app.services.llm_service import (
    TableSpec,
    _merge_source_and_generated_tables,
    _source_tables_from_message,
    _table_catalog_summary,
    _tables_from_tool_args,
    chat,
)
from app.services.table_extraction import extract_source_tables
from app.tools import table_pptx


def test_build_pptx_rejects_empty_headers() -> None:
    with pytest.raises(ValueError, match="headers must not be empty"):
        table_pptx.build_pptx("Title", [], [["value"]])


def test_table_spec_normalizes_rows_and_layout() -> None:
    table = TableSpec(
        title="  Quarterly Review  ",
        headers=["Name", "Status"],
        rows=[["Alice"]],
        text="   ",
        layout="text_left",
    )

    assert table.title == "Quarterly Review"
    assert table.rows == [["Alice", ""]]
    assert table.layout == "table_only"
    assert table.table_ratio == 0.5


def test_download_uses_stored_filename(tmp_path) -> None:
    original_store_dir = table_pptx.STORE_DIR
    table_pptx.STORE_DIR = tmp_path
    try:
        token = table_pptx.store_pptx(b"pptx-bytes", "中文 / 報表?.pptx", 60)
        client = TestClient(create_app())

        response = client.get(f"/download/{token}")

        assert response.status_code == 200
        assert response.content == b"pptx-bytes"
        assert response.headers["content-disposition"] == (
            'attachment; filename="table.pptx"; '
            "filename*=UTF-8''%E4%B8%AD%E6%96%87%20%E5%A0%B1%E8%A1%A8%3F.pptx"
        )
    finally:
        table_pptx.STORE_DIR = original_store_dir


def test_safe_filename_preserves_unicode() -> None:
    assert table_pptx.safe_pptx_filename("中文 / 報表?.pptx") == "中文 報表?.pptx"
    assert table_pptx.ascii_fallback_filename("中文 / 報表?.pptx") == "table.pptx"


def test_multi_table_args_preserve_source_table_before_derived_table() -> None:
    tables = _tables_from_tool_args(
        {
            "tables": [
                {
                    "title": "Original Metrics",
                    "kind": "source_table",
                    "headers": ["Metric", "Value"],
                    "rows": [["Revenue", "$10M"]],
                },
                {
                    "title": "Key Takeaways",
                    "kind": "extracted_summary",
                    "headers": ["Point", "Evidence"],
                    "rows": [["Revenue grew", "Revenue = $10M"]],
                    "text": "Derived from the surrounding prose.",
                    "layout": "table_bottom",
                    "table_ratio": 0.33,
                },
            ]
        }
    )

    assert [table.kind for table in tables] == ["source_table", "extracted_summary"]
    assert tables[0].rows == [["Revenue", "$10M"]]
    assert tables[1].layout == "table_bottom"
    assert tables[1].table_ratio == 0.33


def test_build_tables_pptx_creates_slide_per_table() -> None:
    pptx_bytes = table_pptx.build_tables_pptx(
        [
            {
                "title": "Original Table",
                "kind": "source_table",
                "headers": ["A", "B"],
                "rows": [["1", "2"]],
            },
            {
                "title": "Summary Table",
                "kind": "extracted_summary",
                "headers": ["Theme", "Detail"],
                "rows": [["Risk", "Needs follow-up"]],
            },
        ]
    )

    presentation = Presentation(io.BytesIO(pptx_bytes))
    assert len(presentation.slides) == 2


def test_table_bottom_ratio_places_table_in_lower_third() -> None:
    pptx_bytes = table_pptx.build_pptx(
        "Lower Third",
        ["A", "B"],
        [["1", "2"], ["3", "4"], ["5", "6"]],
        text="Context above the table.",
        layout="table_bottom",
        table_ratio=0.33,
    )

    presentation = Presentation(io.BytesIO(pptx_bytes))
    slide = presentation.slides[0]
    table_shape = next(shape for shape in slide.shapes if getattr(shape, "has_table", False))

    assert table_shape.top > table_pptx.CONTENT_TOP + table_pptx.CONTENT_H * 0.5
    assert table_shape.height <= table_pptx.CONTENT_H * 0.33


def test_extract_source_tables_builds_catalog_from_markdown() -> None:
    tables = extract_source_tables(
        """Quarterly Revenue

| Product | Revenue | Growth |
| --- | ---: | ---: |
| A | 120 | 20% |
| B | 80 | -5% |
"""
    )

    assert len(tables) == 1
    assert tables[0]["table_id"] == "tbl_001"
    assert tables[0]["kind"] == "source_table"
    assert tables[0]["title"] == "Quarterly Revenue"
    assert tables[0]["headers"] == ["Product", "Revenue", "Growth"]
    assert tables[0]["rows"] == [["A", "120", "20%"], ["B", "80", "-5%"]]
    assert tables[0]["summary"] == "2 rows x 3 columns (Product, Revenue, Growth)"
    assert tables[0]["source_ref"] == "lines:3-6"


def test_extract_source_tables_builds_catalog_from_html() -> None:
    tables = extract_source_tables(
        """
<table>
  <caption>Pipeline Status</caption>
  <tr><th>Stage</th><th>Status</th></tr>
  <tr><td>Outline</td><td>Done</td></tr>
</table>
"""
    )

    assert len(tables) == 1
    assert tables[0]["table_id"] == "tbl_001"
    assert tables[0]["title"] == "Pipeline Status"
    assert tables[0]["headers"] == ["Stage", "Status"]
    assert tables[0]["rows"] == [["Outline", "Done"]]
    assert tables[0]["source_ref"] == "html:table"


def test_source_table_catalog_is_preserved_before_generated_tables() -> None:
    source_tables = _source_tables_from_message(
        """Original Data

| Name | Score |
| --- | --- |
| Alice | 95 |
"""
    )
    generated_tables = _tables_from_tool_args(
        {
            "tables": [
                {
                    "title": "Duplicate Source",
                    "kind": "source_table",
                    "headers": ["Name", "Score"],
                    "rows": [["Alice", "95"]],
                },
                {
                    "title": "Derived Summary",
                    "kind": "extracted_summary",
                    "headers": ["Finding", "Evidence"],
                    "rows": [["Alice leads", "Score 95"]],
                },
            ]
        }
    )

    merged = _merge_source_and_generated_tables(source_tables, generated_tables)

    assert [table.kind for table in merged] == ["source_table", "extracted_summary"]
    assert merged[0].table_id == "tbl_001"
    assert merged[0].title == "Original Data"
    assert "tbl_001: Original Data" in _table_catalog_summary(source_tables)


async def test_chat_returns_error_for_invalid_tool_table_args(monkeypatch) -> None:
    invalid_args = {
        "tables": [
            {
                "title": "Broken Table",
                "kind": "extracted_summary",
                "headers": ["Name", ""],
                "rows": [["Alice", "95"]],
            }
        ]
    }

    class FakeCompletions:
        async def create(self, **_kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(arguments=json.dumps(invalid_args))
                                )
                            ],
                        )
                    )
                ]
            )

    class FakeAsyncOpenAI:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("app.services.llm_service.AsyncOpenAI", FakeAsyncOpenAI)

    response = await chat("make a table", Settings(OPENAI_API_KEY="test"))

    assert response["type"] == "error"
    assert response["message"].startswith("Tool args validation error:")
