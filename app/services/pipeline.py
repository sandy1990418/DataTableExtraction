"""Pipeline orchestration shared by the analyze routes.

Pure business logic — parsing, evidence building, table selection, and PPTX
rendering — kept out of the route handlers so the routes stay thin.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import HTTPException

from app.config import Settings
from app.models import DocumentInput
from app.services.canonical_extractor import populate_tables
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
            content = p.read_text()
            name = doc.name or p.name
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


def spec_catalog(specs: list[dict]) -> list[dict]:
    """Lightweight catalog from plan specs (no rows yet) for the /evidence response."""
    return [
        {
            "name": s.get("name"),
            "title": s.get("title"),
            "description": s.get("description", ""),
            "columns": [c.get("name") for c in s.get("columns", [])],
            "column_count": len(s.get("columns", [])),
        }
        for s in specs
    ]


def _referenced_names(ppt_plan: dict) -> list[str]:
    """Distinct table_refs in slide order."""
    names = [s.get("table_ref") for s in ppt_plan.get("slides", []) if s.get("table_ref")]
    return list(dict.fromkeys(names))


async def populate_referenced_tables(
    ppt_plan: dict,
    specs: list[dict],
    evidence_block: str,
    settings: Settings,
    cache: dict[str, dict] | None = None,
) -> list[dict]:
    """Lazily populate ONLY the tables the plan references.

    `cache` (e.g. the session's populated dict) is consulted first and updated with
    any newly populated tables, so re-renders don't re-pay for the same table.
    Returns the populated table dicts for referenced tables, in slide order.
    """
    cache = cache if cache is not None else {}
    referenced = _referenced_names(ppt_plan)
    specs_by_name = {s.get("name"): s for s in specs}

    to_populate = [specs_by_name[n] for n in referenced if n in specs_by_name and n not in cache]
    for table in await populate_tables(to_populate, evidence_block, settings):
        cache[table["name"]] = table

    return [cache[n] for n in referenced if n in cache]


def select_referenced_tables(ppt_plan: dict, tables: list[dict]) -> list[dict]:
    """Build pptx table dicts for ONLY the tables a slide references, in slide order."""
    by_name = {t.get("name"): t for t in tables}
    selected: list[dict] = []
    seen: set[str] = set()
    for slide in ppt_plan.get("slides", []):
        ref = slide.get("table_ref")
        if not ref or ref in seen or ref not in by_name:
            continue
        t = by_name[ref]
        if not t.get("headers") or not t.get("rows"):
            continue
        seen.add(ref)
        selected.append(
            {
                "title": t.get("title") or t.get("name", "Table"),
                "kind": "extracted_summary",
                "headers": t.get("headers", []),
                "rows": t.get("rows", []),
                "text": t.get("description", ""),
                "layout": "table_only" if not t.get("description") else "text_above",
                "table_ratio": 0.6,
                "summary": t.get("description", ""),
                "source_ref": "",
                "table_id": t.get("name", ""),
            }
        )
    return selected


def render_pptx(pptx_tables: list[dict], title: str, settings: Settings) -> dict:
    """Render selected tables to a PPTX and return a download descriptor.

    Synchronous (python-pptx) — call via asyncio.to_thread from async routes.
    """
    if not pptx_tables:
        return {"type": "text", "message": "No tables were referenced by the plan — nothing to render."}
    try:
        pptx_bytes = build_tables_pptx(pptx_tables)
    except Exception as exc:
        logger.error("build_tables_pptx failed: %s", exc, exc_info=True)
        return {"type": "error", "message": f"PPTX generation failed: {exc}"}
    evict_expired()
    filename = f"{title}.pptx"
    token = store_pptx(pptx_bytes, filename, settings.DOWNLOAD_TTL_SECONDS)
    return {"type": "download", "url": f"/download/{token}", "filename": filename}
