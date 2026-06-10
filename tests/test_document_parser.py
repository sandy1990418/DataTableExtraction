from __future__ import annotations

from app.services.document_parser import parse_markdown
from app.services.table_extraction import extract_source_tables


def test_parse_markdown_collects_pipe_and_html_tables() -> None:
    document = parse_markdown(
        """# Results

| Model | Score |
| --- | --- |
| A | 0.91 |

<table>
<caption>Latency</caption>
<tr><th>Model</th><th>Milliseconds</th></tr>
<tr><td>A</td><td>12</td></tr>
</table>
""",
        doc_name="paper.md",
    )

    tables = {table.title: table for table in document.tables}
    assert tables["Results"].rows == [["A", "0.91"]]
    assert tables["Latency"].rows == [["A", "12"]]


def test_extract_source_tables_parses_markdown() -> None:
    tables = extract_source_tables(
        """Quarterly Revenue

| Product | Revenue |
| --- | ---: |
| A | 120 |
"""
    )

    assert tables[0]["title"] == "Quarterly Revenue"
    assert tables[0]["headers"] == ["Product", "Revenue"]
    assert tables[0]["rows"] == [["A", "120"]]
