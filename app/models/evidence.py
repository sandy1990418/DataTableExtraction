"""Domain dataclass for the unified evidence layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

EvidenceKind = Literal[
    "text_fact",
    "markdown_table",
    "image_table",
    "image_caption",
    "chart_insight",
    "diagram_summary",
]


@dataclass
class EvidenceItem:
    kind: EvidenceKind
    source_ref: str
    title: str
    content: str
    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
