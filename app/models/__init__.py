"""Data models: domain dataclasses and HTTP API schemas.

- documents / evidence: internal dataclasses passed between pipeline stages
- schemas: Pydantic request bodies for the HTTP API
"""

from app.models.documents import ImageRef, MarkdownTable, ParsedDocument, TextSection
from app.models.evidence import EvidenceItem, EvidenceKind
from app.models.schemas import (
    AnalyzeRequest,
    ChatRequest,
    DocumentInput,
    EvidenceRequest,
    OutlineRequest,
    RenderRequest,
)

__all__ = [
    # domain dataclasses
    "TextSection",
    "MarkdownTable",
    "ImageRef",
    "ParsedDocument",
    "EvidenceItem",
    "EvidenceKind",
    # API schemas
    "DocumentInput",
    "EvidenceRequest",
    "OutlineRequest",
    "RenderRequest",
    "AnalyzeRequest",
    "ChatRequest",
]
