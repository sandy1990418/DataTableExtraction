"""
create_table_pptx — build a PPTX presentation with a table (and optional text).

Layouts
-------
- "table_only"   : title + full-slide table (default)
- "text_above"   : title + text block + table below
- "text_left"    : title + text on left half + table on right half
- "table_bottom" : title + text above + table in the lower portion
- "table_top"    : title + table in the upper portion + text below
- "table_left"   : title + table on the left + text on the right
- "table_right"  : title + text on the left + table on the right

Overflow handling
-----------------
1. Shrink font down to MIN_FONT_PT to fit on one slide.
2. If still too many rows, split into multiple slides (each with its own header row).
"""

from __future__ import annotations

import io
import re
import time
import uuid
from pathlib import Path
from urllib.parse import quote
from typing import Literal

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

# ── Constants ────────────────────────────────────────────────────────────────

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

MARGIN = Inches(0.4)
TITLE_H = Inches(0.7)
TITLE_TOP = Inches(0.2)
CONTENT_TOP = TITLE_TOP + TITLE_H + Inches(0.15)
CONTENT_H = SLIDE_H - CONTENT_TOP - MARGIN   # space below title

ROW_H = Inches(0.38)
HEADER_FONT = 12
DATA_FONT = 11
MIN_FONT_PT = 7          # smallest readable font before we paginate
MAX_ROWS_HARD = 60       # safety cap per slide even at min font

HEADER_BG = RGBColor(0x2F, 0x54, 0x96)
HEADER_FG = RGBColor(0xFF, 0xFF, 0xFF)
ALT_BG    = RGBColor(0xDD, 0xE8, 0xF5)
WHITE     = RGBColor(0xFF, 0xFF, 0xFF)

Layout = Literal[
    "table_only",
    "text_above",
    "text_left",
    "table_bottom",
    "table_top",
    "table_left",
    "table_right",
]
TableKind = Literal[
    "source_table",
    "extracted_summary",
    "comparison",
    "timeline",
    "matrix",
    "qa",
    "other",
]

# ── In-memory store ──────────────────────────────────────────────────────────

STORE_DIR = Path("/tmp/backend-table-pptx")
_store: dict[str, tuple[Path, str, float]] = {}


def safe_pptx_filename(filename: str) -> str:
    clean = re.sub(r"[\\/\r\n\t]+", " ", filename).strip()
    clean = re.sub(r"\s+", " ", clean)
    clean = clean or "table.pptx"
    if not clean.lower().endswith(".pptx"):
        clean = f"{clean}.pptx"
    return clean


def ascii_fallback_filename(filename: str) -> str:
    clean = safe_pptx_filename(filename)
    ascii_name = re.sub(r"[^A-Za-z0-9._ -]+", "", clean).strip(" .")
    if ascii_name.lower() == "pptx":
        return "table.pptx"
    return ascii_name or "table.pptx"


def content_disposition(filename: str) -> str:
    safe_name = safe_pptx_filename(filename)
    fallback = ascii_fallback_filename(safe_name)
    encoded = quote(safe_name)
    return f'attachment; filename="{fallback}"; filename*=UTF-8\'\'{encoded}'


def store_pptx(data: bytes, filename_or_ttl: str | int = "table.pptx", ttl: int | None = None) -> str:
    if ttl is None:
        filename = "table.pptx"
        ttl_seconds = int(filename_or_ttl)
    else:
        filename = safe_pptx_filename(str(filename_or_ttl))
        ttl_seconds = ttl

    token = uuid.uuid4().hex
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    path = STORE_DIR / f"{token}.pptx"
    path.write_bytes(data)
    _store[token] = (path, filename, time.monotonic() + ttl_seconds)
    return token


def get_pptx(token: str) -> bytes | None:
    entry = get_pptx_entry(token)
    if entry is None:
        return None
    data, _filename = entry
    return data


def get_pptx_entry(token: str) -> tuple[bytes, str] | None:
    entry = _store.get(token)
    if entry is None:
        return None
    path, filename, expiry = entry
    if time.monotonic() > expiry:
        del _store[token]
        path.unlink(missing_ok=True)
        return None
    if not path.exists():
        del _store[token]
        return None
    return path.read_bytes(), filename


def evict_expired() -> None:
    now = time.monotonic()
    for k, (path, _, _) in [(k, entry) for k, entry in _store.items() if now > entry[2]]:
        path.unlink(missing_ok=True)
        del _store[k]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _add_title(slide, text: str) -> None:
    txb = slide.shapes.add_textbox(MARGIN, TITLE_TOP, SLIDE_W - MARGIN * 2, TITLE_H)
    tf = txb.text_frame
    tf.word_wrap = True
    tf.text = text
    runs = tf.paragraphs[0].runs
    if runs:
        runs[0].font.size = Pt(24)
        runs[0].font.bold = True


def _add_text_box(slide, text: str, left, top, width, height) -> None:
    txb = slide.shapes.add_textbox(left, top, width, height)
    tf = txb.text_frame
    tf.word_wrap = True
    tf.text = text
    for para in tf.paragraphs:
        for run in para.runs:
            run.font.size = Pt(12)


def _add_body_slide(prs: Presentation, blank_layout, title: str, content: str) -> None:
    slide = prs.slides.add_slide(blank_layout)
    _add_title(slide, title)
    body = content.strip()
    if body:
        _add_text_box(slide, body, MARGIN, CONTENT_TOP, SLIDE_W - MARGIN * 2, CONTENT_H)


def _rows_per_slide(available_h, font_pt: float) -> int:
    rh = Inches(font_pt / 72 * 1.8)   # approx row height from font size
    return max(1, int(available_h / rh) - 1)  # -1 for header


def _col_widths(headers: list[str], rows: list[list[str]], total_width) -> list[int]:
    """Distribute total_width proportionally by max char length per column."""
    n_cols = len(headers)
    max_lens = [len(str(h)) for h in headers]
    for row in rows:
        for c in range(n_cols):
            max_lens[c] = max(max_lens[c], len(str(row[c])) if c < len(row) else 0)
    total_chars = sum(max_lens) or 1
    min_w = int(total_width / n_cols * 0.4)  # floor: 40% of even share
    widths = [max(min_w, int(total_width * length / total_chars)) for length in max_lens]
    # Scale down proportionally if total exceeds available width
    total = sum(widths)
    if total > total_width:
        scale = total_width / total
        widths = [max(min_w, int(w * scale)) for w in widths]
    # Last column absorbs rounding remainder, clamped to min_w
    widths[-1] = max(min_w, total_width - sum(widths[:-1]))
    return widths


def _add_table(slide, headers, rows, left, top, width, height, font_pt: float) -> None:
    n_cols = len(headers)
    n_rows = len(rows) + 1

    tbl = slide.shapes.add_table(n_rows, n_cols, left, top, width, int(height)).table

    # Apply proportional column widths
    col_widths = _col_widths(headers, rows, int(width))
    for c, cw in enumerate(col_widths):
        tbl.columns[c].width = cw

    for c, text in enumerate(headers):
        cell = tbl.cell(0, c)
        cell.text = str(text)
        cell.fill.solid()
        cell.fill.fore_color.rgb = HEADER_BG
        runs = cell.text_frame.paragraphs[0].runs
        if runs:
            runs[0].font.bold = True
            runs[0].font.size = Pt(font_pt)
            runs[0].font.color.rgb = HEADER_FG

    for r, row in enumerate(rows):
        bg = ALT_BG if r % 2 == 0 else WHITE
        for c in range(n_cols):
            cell = tbl.cell(r + 1, c)
            cell.text = str(row[c]) if c < len(row) else ""
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            runs = cell.text_frame.paragraphs[0].runs
            if runs:
                runs[0].font.size = Pt(font_pt)


def _validate_table_inputs(headers: list[str], rows: list[list[str]]) -> None:
    if not headers:
        raise ValueError("headers must not be empty")
    if any(not str(header).strip() for header in headers):
        raise ValueError("headers must not contain blank values")
    if rows is None:
        raise ValueError("rows must not be None")


def _normalize_table_ratio(table_ratio: float) -> float:
    try:
        ratio = float(table_ratio)
    except (TypeError, ValueError):
        ratio = 0.5
    return min(0.8, max(0.2, ratio))


def _compute_regions(layout: Layout, table_ratio: float, has_text: bool):
    content_left = MARGIN
    content_top = CONTENT_TOP
    content_w = SLIDE_W - MARGIN * 2
    content_h = CONTENT_H
    gap = Inches(0.15)

    if not has_text or layout == "table_only":
        return None, (content_left, content_top, content_w, content_h), "table_only"

    ratio = _normalize_table_ratio(table_ratio)
    resolved_layout = layout
    if layout == "text_above":
        resolved_layout = "table_bottom"
        ratio = 0.75
    elif layout == "text_left":
        resolved_layout = "table_right"
        ratio = 0.5

    if resolved_layout == "table_bottom":
        tbl_h = int(content_h * ratio)
        text_h = content_h - tbl_h - gap
        return (
            (content_left, content_top, content_w, text_h),
            (content_left, content_top + text_h + gap, content_w, tbl_h),
            resolved_layout,
        )

    if resolved_layout == "table_top":
        tbl_h = int(content_h * ratio)
        text_h = content_h - tbl_h - gap
        return (
            (content_left, content_top + tbl_h + gap, content_w, text_h),
            (content_left, content_top, content_w, tbl_h),
            resolved_layout,
        )

    if resolved_layout == "table_left":
        tbl_w = int(content_w * ratio)
        text_w = content_w - tbl_w - gap
        return (
            (content_left + tbl_w + gap, content_top, text_w, content_h),
            (content_left, content_top, tbl_w, content_h),
            resolved_layout,
        )

    if resolved_layout == "table_right":
        tbl_w = int(content_w * ratio)
        text_w = content_w - tbl_w - gap
        return (
            (content_left, content_top, text_w, content_h),
            (content_left + text_w + gap, content_top, tbl_w, content_h),
            resolved_layout,
        )

    return None, (content_left, content_top, content_w, content_h), "table_only"


def _compute_font_and_chunks(
    n_rows: int,
    available_h,
    font_pt: float = DATA_FONT,
) -> tuple[float, list[int]]:
    """
    Return (font_pt, chunk_sizes) where chunk_sizes is the number of data
    rows per slide.  Shrinks font first; paginates if still needed.
    """
    # Try shrinking font
    while font_pt >= MIN_FONT_PT:
        rph = _rows_per_slide(available_h, font_pt)
        if rph >= n_rows:
            return font_pt, [n_rows]
        if font_pt == MIN_FONT_PT:
            break
        font_pt -= 1

    # Font at minimum — paginate
    rph = min(_rows_per_slide(available_h, MIN_FONT_PT), MAX_ROWS_HARD)
    chunks = []
    remaining = n_rows
    while remaining > 0:
        take = min(rph, remaining)
        chunks.append(take)
        remaining -= take
    return MIN_FONT_PT, chunks


# ── Main build function ──────────────────────────────────────────────────────

def build_pptx(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    text: str = "",
    layout: Layout = "table_only",
    table_ratio: float = 0.5,
) -> bytes:
    _validate_table_inputs(headers, rows)
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    blank_layout = prs.slide_layouts[6]
    _add_table_slides(prs, blank_layout, title, headers, rows, text, layout, table_ratio)

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def _add_table_slides(
    prs: Presentation,
    blank_layout,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    text: str = "",
    layout: Layout = "table_only",
    table_ratio: float = 0.5,
) -> None:
    _validate_table_inputs(headers, rows)

    text_region, table_region, layout = _compute_regions(layout, table_ratio, bool(text))
    tbl_left, tbl_top, tbl_w, tbl_h = table_region
    available_h = tbl_h

    font_pt, chunks = _compute_font_and_chunks(len(rows), available_h)

    offset = 0
    for i, chunk_size in enumerate(chunks):
        slide = prs.slides.add_slide(blank_layout)

        page_title = title if len(chunks) == 1 else f"{title} ({i + 1}/{len(chunks)})"
        _add_title(slide, page_title)

        # Text area (only on first slide for multi-page)
        if text_region and text and i == 0:
            _add_text_box(slide, text, *text_region)

        chunk_rows = rows[offset: offset + chunk_size]
        offset += chunk_size

        # Recalc row height for this chunk
        n_rows_slide = len(chunk_rows) + 1
        actual_h = min(tbl_h, ROW_H * n_rows_slide)

        _add_table(slide, headers, chunk_rows, tbl_left, tbl_top, tbl_w, actual_h, font_pt)


def build_tables_pptx(tables: list[dict]) -> bytes:
    if not tables:
        raise ValueError("tables must not be empty")

    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank_layout = prs.slide_layouts[6]

    for table in tables:
        _add_table_slides(
            prs,
            blank_layout,
            str(table["title"]),
            [str(header) for header in table["headers"]],
            [[str(cell) for cell in row] for row in table["rows"]],
            str(table.get("text", "")),
            table.get("layout", "table_only"),
            table.get("table_ratio", 0.5),
        )

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def build_plan_pptx(ppt_plan: dict | None, tables: list[dict]) -> bytes:
    """Render the full slide plan, preserving non-table slides."""
    if not ppt_plan or not ppt_plan.get("slides"):
        return build_tables_pptx(tables)

    table_by_id = {str(table.get("table_id") or table.get("name")): table for table in tables}
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank_layout = prs.slide_layouts[6]

    for index, slide_plan in enumerate(ppt_plan.get("slides", []), 1):
        title = str(slide_plan.get("title") or f"Slide {index}")
        content = str(slide_plan.get("content") or "")
        table_ref = slide_plan.get("table_ref")
        table = table_by_id.get(str(table_ref)) if table_ref else None

        if table and table.get("headers") and table.get("rows"):
            text = content or str(table.get("text") or table.get("summary") or "")
            _add_table_slides(
                prs,
                blank_layout,
                title,
                [str(header) for header in table["headers"]],
                [[str(cell) for cell in row] for row in table["rows"]],
                text,
                "table_bottom" if text else "table_only",
                0.68,
            )
        else:
            _add_body_slide(prs, blank_layout, title, content)

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


# ── OpenAI function-calling schema ───────────────────────────────────────────

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_table_pptx",
        "description": (
            "Generate a PowerPoint (.pptx) file containing one or more tables. "
            "Use source_table tables to preserve tables already present in the input, and use "
            "derived table kinds to summarize, compare, or reorganize key points."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "presentation_title": {
                    "type": "string",
                    "description": "Specific filename/presentation title for the generated deck.",
                },
                "tables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Specific slide title shown above this table.",
                            },
                            "kind": {
                                "type": "string",
                                "enum": [
                                    "source_table",
                                    "extracted_summary",
                                    "comparison",
                                    "timeline",
                                    "matrix",
                                    "qa",
                                    "other",
                                ],
                                "description": (
                                    "Use source_table for a table copied from the original content; "
                                    "use the other values for generated/derived tables."
                                ),
                            },
                            "headers": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Column header labels.",
                            },
                            "rows": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "string"}},
                                "description": "Table rows; each inner array must match headers length.",
                            },
                            "text": {
                                "type": "string",
                                "description": (
                                    "Optional explanatory text to accompany the table "
                                    "(e.g. summary, key findings, source note). Leave empty if not needed."
                                ),
                            },
                            "layout": {
                                "type": "string",
                                "enum": [
                                    "table_only",
                                    "text_above",
                                    "text_left",
                                    "table_bottom",
                                    "table_top",
                                    "table_left",
                                    "table_right",
                                ],
                                "description": (
                                    "Slide layout. Use table_bottom with table_ratio=0.33 for a lower-third "
                                    "table with text above; use table_right with table_ratio=0.33 for a "
                                    "right-third table with text on the left. Legacy text_above/text_left are "
                                    "also accepted."
                                ),
                            },
                            "table_ratio": {
                                "type": "number",
                                "minimum": 0.2,
                                "maximum": 0.8,
                                "description": (
                                    "Fraction of the content area occupied by the table for table_top, "
                                    "table_bottom, table_left, or table_right. Use 0.33 for one-third."
                                ),
                            },
                        },
                        "required": ["title", "kind", "headers", "rows"],
                        "additionalProperties": False,
                    },
                    "description": "Ordered tables to place into the deck, one table section per slide group.",
                },
                "title": {
                    "type": "string",
                    "description": "Legacy single-table slide title. Prefer tables[].title.",
                },
                "headers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Legacy single-table column headers. Prefer tables[].headers.",
                },
                "rows": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "string"}},
                    "description": "Legacy single-table rows. Prefer tables[].rows.",
                },
                "text": {
                    "type": "string",
                    "description": (
                        "Optional explanatory text to accompany the table "
                        "(e.g. summary, key findings, context). Leave empty if not needed."
                    ),
                },
                "layout": {
                    "type": "string",
                    "enum": [
                        "table_only",
                        "text_above",
                        "text_left",
                        "table_bottom",
                        "table_top",
                        "table_left",
                        "table_right",
                    ],
                    "description": (
                        "Legacy single-table layout. Prefer tables[].layout."
                    ),
                },
                "table_ratio": {
                    "type": "number",
                    "minimum": 0.2,
                    "maximum": 0.8,
                    "description": (
                        "Legacy single-table table ratio. Prefer tables[].table_ratio."
                    ),
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}
