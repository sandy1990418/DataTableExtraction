"""Pydantic request schemas for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    name: str = Field(default="", description="Document name; defaults to filename if file_path is given")
    file_path: str | None = Field(default=None, description="Absolute or relative path to a .md file; server reads it directly")
    content: str = Field(default="", description="Markdown content (used when file_path is not given)")
    base_dir: str | None = Field(default=None, description="Base directory for resolving image paths; defaults to file_path's parent")


class EvidenceRequest(BaseModel):
    documents: list[DocumentInput] = Field(description="One or more markdown documents")
    hint: str = Field(default="", description="Optional goal that steers which tables get extracted")
    analyze_images: bool = Field(default=True, description="Run vision analysis on embedded images")


class OutlineRequest(BaseModel):
    session_id: str = Field(description="session_id from /evidence")
    hint: str = Field(default="", description="Presentation goal; defaults to the /evidence hint")
    n_slides: int | None = Field(default=None, description="Target slide count")


class RenderRequest(BaseModel):
    session_id: str = Field(description="session_id from /evidence")
    ppt_plan: dict = Field(description="Plan from /outline")
    presentation_title: str | None = Field(default=None)


class AnalyzeRequest(BaseModel):
    documents: list[DocumentInput] = Field(description="One or more markdown documents to analyze")
    hint: str = Field(default="", description="Optional instruction or goal for the analysis")
    n_slides: int | None = Field(default=None, description="Target slide count")
    analyze_images: bool = Field(default=True)


class ChatRequest(BaseModel):
    message: str
