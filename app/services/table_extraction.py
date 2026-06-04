from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser


def extract_source_tables(text: str) -> list[dict]:
    """Extract tables that already exist in the source text."""
    tables = _extract_html_tables(text)
    tables.extend(_extract_markdown_tables(text))
    tables.extend(_extract_tsv_tables(text))
    return _dedupe_tables(tables)


def _make_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    source_ref: str = "",
) -> dict | None:
    headers, rows = _normalize_table(headers, rows)
    if not headers or not rows:
        return None

    return {
        "table_id": "",
        "title": title.strip() or "Source Table",
        "kind": "source_table",
        "headers": headers,
        "rows": rows,
        "summary": _summarize_table(headers, rows),
        "source_ref": source_ref,
        "text": "",
        "layout": "table_only",
        "table_ratio": 0.5,
    }


def _normalize_table(
    headers: list[str],
    rows: list[list[str]],
) -> tuple[list[str], list[list[str]]]:
    headers = [_clean_cell(header) for header in headers]
    rows = [[_clean_cell(cell) for cell in row] for row in rows]
    rows = [row for row in rows if any(row)]

    width = max([len(headers), *(len(row) for row in rows)] or [0])
    if width < 2:
        return [], []

    headers = (headers + [f"Column {i + 1}" for i in range(len(headers), width)])[:width]
    headers = [header or f"Column {i + 1}" for i, header in enumerate(headers)]

    normalized_rows = []
    for row in rows:
        normalized_rows.append((row + [""] * width)[:width])
    return headers, normalized_rows


def _clean_cell(value: str) -> str:
    return re.sub(r"\s+", " ", unescape(str(value))).strip()


def _summarize_table(headers: list[str], rows: list[list[str]]) -> str:
    preview_headers = ", ".join(headers[:4])
    if len(headers) > 4:
        preview_headers += ", ..."
    return f"{len(rows)} rows x {len(headers)} columns ({preview_headers})"


def _dedupe_tables(tables: list[dict | None]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple] = set()
    source_index = 1

    for table in tables:
        if table is None:
            continue
        key = (
            tuple(table["headers"]),
            tuple(tuple(row) for row in table["rows"]),
        )
        if key in seen:
            continue
        seen.add(key)
        if table["title"] == "Source Table":
            table["title"] = f"Source Table {source_index}"
        table["table_id"] = f"tbl_{source_index:03d}"
        source_index += 1
        deduped.append(table)

    return deduped


def _extract_markdown_tables(text: str) -> list[dict]:
    lines = text.splitlines()
    tables: list[dict] = []
    i = 0

    while i < len(lines) - 1:
        if not _looks_like_pipe_row(lines[i]) or not _is_markdown_separator(lines[i + 1]):
            i += 1
            continue

        headers = _split_pipe_row(lines[i])
        rows: list[list[str]] = []
        j = i + 2
        while j < len(lines) and _looks_like_pipe_row(lines[j]):
            if not _is_markdown_separator(lines[j]):
                rows.append(_split_pipe_row(lines[j]))
            j += 1

        title = _nearest_heading(lines, i) or "Source Table"
        tables.append(_make_table(title, headers, rows, f"lines:{i + 1}-{j}"))
        i = j

    return tables


def _looks_like_pipe_row(line: str) -> bool:
    return "|" in line and len(_split_pipe_row(line)) >= 2


def _split_pipe_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [_clean_cell(cell.replace(r"\|", "|")) for cell in re.split(r"(?<!\\)\|", stripped)]


def _is_markdown_separator(line: str) -> bool:
    cells = _split_pipe_row(line)
    return len(cells) >= 2 and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def _nearest_heading(lines: list[str], index: int) -> str | None:
    for i in range(index - 1, max(-1, index - 5), -1):
        candidate = lines[i].strip()
        if not candidate:
            continue
        if _looks_like_pipe_row(candidate):
            continue
        candidate = re.sub(r"^#{1,6}\s*", "", candidate).strip(" :")
        return candidate or None
    return None


def _extract_tsv_tables(text: str) -> list[dict]:
    lines = text.splitlines()
    tables: list[dict] = []
    i = 0

    while i < len(lines):
        if "\t" not in lines[i]:
            i += 1
            continue

        block: list[list[str]] = []
        start = i
        while i < len(lines) and "\t" in lines[i]:
            block.append([_clean_cell(cell) for cell in lines[i].split("\t")])
            i += 1

        widths = {len(row) for row in block}
        if len(block) >= 2 and len(widths) == 1 and next(iter(widths)) >= 2:
            title = _nearest_heading(lines, start) or "Source Table"
            tables.append(_make_table(title, block[0], block[1:], f"lines:{start + 1}-{i}"))
        else:
            i += 1

    return tables


class _HTMLTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[dict | None] = []
        self._in_table = False
        self._in_caption = False
        self._in_cell = False
        self._cell_is_header = False
        self._caption = ""
        self._cell_parts: list[str] = []
        self._current_row: list[str] = []
        self._current_row_has_header = False
        self._rows: list[tuple[list[str], bool]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "table":
            self._in_table = True
            self._caption = ""
            self._rows = []
        elif self._in_table and tag == "caption":
            self._in_caption = True
        elif self._in_table and tag == "tr":
            self._current_row = []
            self._current_row_has_header = False
        elif self._in_table and tag in {"th", "td"}:
            self._in_cell = True
            self._cell_is_header = tag == "th"
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_caption:
            self._caption += data
        elif self._in_cell:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._in_table and tag == "caption":
            self._in_caption = False
        elif self._in_table and tag in {"th", "td"} and self._in_cell:
            self._current_row.append(_clean_cell("".join(self._cell_parts)))
            self._current_row_has_header = self._current_row_has_header or self._cell_is_header
            self._in_cell = False
            self._cell_is_header = False
            self._cell_parts = []
        elif self._in_table and tag == "tr":
            if any(self._current_row):
                self._rows.append((self._current_row, self._current_row_has_header))
            self._current_row = []
            self._current_row_has_header = False
        elif tag == "table" and self._in_table:
            self.tables.append(self._to_table())
            self._in_table = False

    def _to_table(self) -> dict | None:
        if not self._rows:
            return None

        first_row, first_row_has_header = self._rows[0]
        if first_row_has_header:
            headers = first_row
            rows = [row for row, _ in self._rows[1:]]
        else:
            width = max(len(row) for row, _ in self._rows)
            headers = [f"Column {i + 1}" for i in range(width)]
            rows = [row for row, _ in self._rows]

        return _make_table(_clean_cell(self._caption) or "Source Table", headers, rows, "html:table")


def _extract_html_tables(text: str) -> list[dict]:
    if "<table" not in text.lower():
        return []

    parser = _HTMLTableParser()
    parser.feed(text)
    return parser.tables
