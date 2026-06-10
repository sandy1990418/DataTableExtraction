"""Shared document, evidence, and PPTX helpers for the data-table route."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import HTTPException

from app.config import Settings
from app.models.schemas import DocumentInput
from app.services.document_parser import parse_markdown
from app.services.evidence_layer import build_evidence_layer
from app.services.image_analysis import analyze_images_batch
from app.tools.table_pptx import build_tables_pptx, evict_expired, store_pptx

logger = logging.getLogger(__name__)


def resolve_documents(documents: list[DocumentInput]):
    """Turn API document inputs (file_path or inline content) into ParsedDocuments."""
    parsed_docs = []
    for doc in documents:
        if doc.file_path:
            p = Path(doc.file_path)
            if not p.exists():
                raise HTTPException(status_code=400, detail=f"File not found: {doc.file_path}")
            name = doc.name or p.name
            content = p.read_text()
            base_dir = doc.base_dir or str(p.parent)
        else:
            content = doc.content
            name = doc.name or "doc"
            base_dir = doc.base_dir
        parsed_docs.append(parse_markdown(content, doc_name=name, base_dir=base_dir))
    return parsed_docs


async def build_evidence(parsed_docs, settings: Settings, analyze_images: bool):
    """Run vision on embedded images (optional) and unify into the evidence layer."""
    images_with_data = [
        {"data_b64": img.data_b64, "alt": img.alt}
        for doc in parsed_docs
        for img in doc.images
        if img.data_b64
    ]
    image_analyses = (
        await analyze_images_batch(images_with_data, settings)
        if (analyze_images and images_with_data)
        else []
    )

    image_analysis_map: list[dict] = []
    analysis_idx = 0
    for doc in parsed_docs:
        for img in doc.images:
            if img.data_b64 and analyze_images:
                image_analysis_map.append(
                    image_analyses[analysis_idx] if analysis_idx < len(image_analyses) else {}
                )
                analysis_idx += 1
            else:
                image_analysis_map.append({"type": "other", "caption": img.alt, "insight": "", "table": None})

    return build_evidence_layer(parsed_docs, image_analysis_map)


def render_pptx(pptx_tables: list[dict], title: str, settings: Settings) -> dict:
    """Render tables to a PPTX and return a download descriptor."""
    if not pptx_tables:
        return {"type": "text", "message": "No table was produced."}
    try:
        pptx_bytes = build_tables_pptx(pptx_tables)
    except Exception as exc:
        logger.error("build_tables_pptx failed: %s", exc, exc_info=True)
        return {"type": "error", "message": f"PPTX generation failed: {exc}"}
    evict_expired()
    filename = f"{title}.pptx"
    token = store_pptx(pptx_bytes, filename, settings.DOWNLOAD_TTL_SECONDS)
    return {"type": "download", "url": f"/download/{token}", "filename": filename}
