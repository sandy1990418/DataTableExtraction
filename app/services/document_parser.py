from __future__ import annotations

import base64
import re
from pathlib import Path

from app.models import ImageRef, MarkdownTable, ParsedDocument, TextSection

MAX_TEXT_SECTION_CHARS = 3500


def parse_markdown(content: str, doc_name: str = "doc", base_dir: str | None = None) -> ParsedDocument:
    doc = ParsedDocument(name=doc_name)
    lines = content.splitlines()
    i = 0
    current_heading = ""
    current_text_lines: list[str] = []
    current_text_start = 1

    def append_text_line(line_no: int, text: str) -> None:
        nonlocal current_text_start
        if not current_text_lines:
            current_text_start = line_no
        current_text_lines.append(text)

    def flush_section(end_line: int | None = None) -> None:
        nonlocal current_text_lines
        text = "\n".join(current_text_lines).strip()
        if text:
            chunks = _chunk_lines(current_text_lines, MAX_TEXT_SECTION_CHARS)
            for chunk_idx, chunk in enumerate(chunks, 1):
                suffix = f" (part {chunk_idx})" if len(chunks) > 1 else ""
                doc.sections.append(
                    TextSection(
                        heading=(current_heading or doc_name) + suffix,
                        content="\n".join(chunk).strip(),
                        source_ref=f"{doc_name}:lines:{current_text_start}-{end_line or i}",
                    )
                )
        current_text_lines = []

    while i < len(lines):
        line = lines[i]

        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush_section(i)
            current_heading = m.group(2).strip()
            i += 1
            continue

        paper_heading = _paper_heading_at(lines, i)
        if paper_heading:
            flush_section(i)
            current_heading, i = paper_heading
            continue

        caption_kind = _caption_kind(line)
        if caption_kind:
            flush_section(i)
            block, j = _collect_caption_block(lines, i, caption_kind)
            doc.sections.append(
                TextSection(
                    heading=block[0].strip(),
                    content="\n".join(block).strip(),
                    source_ref=f"{doc_name}:lines:{i + 1}-{j}",
                )
            )
            i = j
            continue

        if "|" in line and i + 1 < len(lines) and re.match(r"^[\|\s]*:?-{3,}:?[\|\s]*", lines[i + 1]):
            flush_section(i)
            headers = _split_pipe(line)
            rows: list[list[str]] = []
            j = i + 2
            while j < len(lines) and "|" in lines[j]:
                rows.append(_split_pipe(lines[j]))
                j += 1
            doc.tables.append(MarkdownTable(
                title=current_heading or "Table",
                headers=headers,
                rows=rows,
                source_ref=f"{doc_name}:lines:{i+1}-{j}",
            ))
            i = j
            continue

        img_matches = re.findall(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        for alt, path in img_matches:
            ref = ImageRef(alt=alt.strip(), path=path.strip(), source_ref=f"{doc_name}:line:{i+1}")
            if base_dir:
                abs_path = Path(base_dir) / path
                if abs_path.exists():
                    data = abs_path.read_bytes()
                    suffix = abs_path.suffix.lower().lstrip(".")
                    mime = {
                        "jpg": "image/jpeg",
                        "jpeg": "image/jpeg",
                        "png": "image/png",
                        "gif": "image/gif",
                        "webp": "image/webp",
                    }.get(suffix, "image/png")
                    ref.data_b64 = f"data:{mime};base64,{base64.b64encode(data).decode()}"
            doc.images.append(ref)

        append_text_line(i + 1, line)
        i += 1

    flush_section(len(lines))
    return doc


def _split_pipe(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def _chunk_lines(lines: list[str], max_chars: int) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for line in lines:
        line_len = len(line) + 1
        if current and current_len + line_len > max_chars:
            chunks.append(current)
            current = []
            current_len = 0
        current.append(line)
        current_len += line_len
    if current:
        chunks.append(current)
    return chunks


def _paper_heading_at(lines: list[str], index: int) -> tuple[str, int] | None:
    line = lines[index].strip()
    if not line:
        return None

    if re.fullmatch(r"\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z0-9 ,:;()/-]{2,80}", line):
        if not _caption_kind(line):
            return line.rstrip("."), index + 1

    if re.fullmatch(r"\d+(?:\.\d+)*", line) and index + 1 < len(lines):
        title = lines[index + 1].strip()
        if _is_title_like(title):
            return f"{line} {title}", index + 2

    return None


def _is_title_like(line: str) -> bool:
    if not line or len(line) > 90:
        return False
    if _caption_kind(line) or line.startswith("!["):
        return False
    if re.search(r"\d+\.\d+", line):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z-]*", line)
    return 1 <= len(words) <= 10


def _caption_kind(line: str) -> str:
    match = re.match(r"^\s*(Table|Figure|Fig\.)\s+\d+[.:]\s+", line, flags=re.IGNORECASE)
    if not match:
        return ""
    kind = match.group(1).lower()
    return "figure" if kind.startswith("fig") else kind


def _collect_caption_block(lines: list[str], start: int, kind: str) -> tuple[list[str], int]:
    max_lines = 90 if kind == "table" else 20
    block: list[str] = []
    i = start
    while i < len(lines) and len(block) < max_lines:
        if i > start:
            line = lines[i].strip()
            if _caption_kind(line) or _paper_heading_at(lines, i) or line.startswith("!["):
                break
        block.append(lines[i])
        i += 1
    return block, i
