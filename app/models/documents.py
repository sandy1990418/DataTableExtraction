"""Domain dataclasses for parsed documents (produced by document_parser)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextSection:
    heading: str
    content: str
    source_ref: str = ""


@dataclass
class MarkdownTable:
    title: str
    headers: list[str]
    rows: list[list[str]]
    source_ref: str = ""


@dataclass
class ImageRef:
    alt: str
    path: str
    source_ref: str = ""
    data_b64: str = ""


@dataclass
class ParsedDocument:
    name: str
    sections: list[TextSection] = field(default_factory=list)
    tables: list[MarkdownTable] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)
