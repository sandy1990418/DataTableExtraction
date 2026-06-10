"""Pydantic request schemas for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    name: str = Field(default="", description="Document name; defaults to filename if file_path is given")
    file_path: str | None = Field(default=None, description="Absolute or relative path to a .md file; server reads it directly")
    content: str = Field(default="", description="Markdown content (used when file_path is not given)")
    base_dir: str | None = Field(default=None, description="Base directory for resolving image paths; defaults to file_path's parent")
