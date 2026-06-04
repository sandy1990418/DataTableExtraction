from __future__ import annotations

import base64
import re
from pathlib import Path

from app.models import ImageRef, MarkdownTable, ParsedDocument, TextSection


def parse_markdown(content: str, doc_name: str = "doc", base_dir: str | None = None) -> ParsedDocument:
    doc = ParsedDocument(name=doc_name)
    lines = content.splitlines()
    i = 0
    current_heading = ""
    current_text_lines: list[str] = []

    def flush_section() -> None:
        nonlocal current_text_lines
        text = "\n".join(current_text_lines).strip()
        if text:
            doc.sections.append(TextSection(heading=current_heading, content=text, source_ref=doc_name))
        current_text_lines = []

    while i < len(lines):
        line = lines[i]

        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            flush_section()
            current_heading = m.group(2).strip()
            i += 1
            continue

        if "|" in line and i + 1 < len(lines) and re.match(r"^[\|\s]*:?-{3,}:?[\|\s]*", lines[i + 1]):
            flush_section()
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

        current_text_lines.append(line)
        i += 1

    flush_section()
    return doc


def _split_pipe(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]
