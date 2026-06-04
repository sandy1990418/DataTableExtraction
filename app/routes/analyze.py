from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.services.canonical_extractor import extract_canonical_tables
from app.services.document_parser import parse_markdown
from app.services.evidence_layer import build_evidence_layer, summarize_evidence
from app.services.image_analysis import analyze_images_batch
from app.services.ppt_planner import generate_ppt_plan
from app.tools.table_pptx import build_tables_pptx, evict_expired, store_pptx

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Shared models ────────────────────────────────────────────────────────────


class DocumentInput(BaseModel):
    name: str = Field(default="", description="Document name; defaults to filename if file_path is given")
    file_path: str | None = Field(default=None, description="Absolute or relative path to a .md file; server reads it directly")
    content: str = Field(default="", description="Markdown content (used when file_path is not given)")
    base_dir: str | None = Field(default=None, description="Base directory for resolving image paths; defaults to file_path's parent")


class TableModel(BaseModel):
    name: str
    title: str = ""
    description: str = ""
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


# ── Stage 1: evidence + table pool ───────────────────────────────────────────


class EvidenceRequest(BaseModel):
    documents: list[DocumentInput] = Field(description="One or more markdown documents")
    hint: str = Field(default="", description="Optional goal that steers which tables get extracted")
    analyze_images: bool = Field(default=True, description="Run vision analysis on embedded images")


def _resolve_documents(documents: list[DocumentInput]):
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


async def _build_evidence(parsed_docs, settings: Settings, analyze_images: bool):
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


@router.post("/evidence", summary="Documents → evidence layer + canonical table pool (no rendering)")
async def evidence_endpoint(body: EvidenceRequest, settings: Settings = Depends(get_settings)):
    """Stage 1 — run ONCE per document set and cache the result. Returns the full
    table pool plus a lightweight catalog you feed to the outline step."""
    parsed_docs = _resolve_documents(body.documents)
    evidence_items = await _build_evidence(parsed_docs, settings, body.analyze_images)
    evidence_summary = summarize_evidence(evidence_items)
    canonical_tables = await extract_canonical_tables(evidence_items, settings, hint=body.hint)

    return {
        "evidence_summary": evidence_summary,
        # Full tables (with rows) — cache these for the render step.
        "tables": canonical_tables,
        # Lightweight catalog — pass THIS to the outline step (no rows).
        "table_catalog": [
            {
                "name": t.get("name"),
                "title": t.get("title"),
                "description": t.get("description", ""),
                "row_count": len(t.get("rows", [])),
                "column_count": len(t.get("headers", [])),
            }
            for t in canonical_tables
        ],
    }


# ── Stage 2: outline (which tables become slides) ────────────────────────────


class OutlineRequest(BaseModel):
    tables: list[TableModel] = Field(description="Table pool from /evidence (rows optional here)")
    evidence_summary: str = Field(default="", description="evidence_summary from /evidence")
    hint: str = Field(default="", description="Presentation goal")
    n_slides: int | None = Field(default=None, description="Target slide count")


@router.post("/outline", summary="Table pool → slide outline; LLM picks which tables deserve a slide")
async def outline_endpoint(body: OutlineRequest, settings: Settings = Depends(get_settings)):
    """Stage 2 — maps to Presenton's outline generation. Each slide may carry a
    table_ref (a table name) or null. Tables not referenced won't be rendered."""
    table_dicts = [t.model_dump() for t in body.tables]
    ppt_plan = await generate_ppt_plan(
        table_dicts,
        body.evidence_summary,
        settings,
        presentation_hint=body.hint,
        n_slides=body.n_slides,
    )
    referenced = [s.get("table_ref") for s in ppt_plan.get("slides", []) if s.get("table_ref")]
    return {
        "ppt_plan": ppt_plan,
        "referenced_tables": referenced,
    }


# ── Stage 3: render (only referenced tables) ─────────────────────────────────


class RenderRequest(BaseModel):
    ppt_plan: dict = Field(description="Plan from /outline")
    tables: list[TableModel] = Field(description="Full table pool from /evidence (must include rows)")
    presentation_title: str | None = Field(default=None)


def _select_referenced_tables(ppt_plan: dict, tables: list[TableModel]) -> list[dict]:
    by_name = {t.name: t for t in tables}
    selected: list[dict] = []
    seen: set[str] = set()
    for slide in ppt_plan.get("slides", []):
        ref = slide.get("table_ref")
        if not ref or ref in seen or ref not in by_name:
            continue
        t = by_name[ref]
        if not t.headers or not t.rows:
            continue
        seen.add(ref)
        selected.append(
            {
                "title": t.title or t.name,
                "kind": "extracted_summary",
                "headers": t.headers,
                "rows": t.rows,
                "text": t.description,
                "layout": "table_only" if not t.description else "text_above",
                "table_ratio": 0.6,
                "summary": t.description,
                "source_ref": "",
                "table_id": t.name,
            }
        )
    return selected


@router.post("/render", summary="Plan + table pool → PPTX; renders ONLY tables referenced by the plan")
async def render_endpoint(body: RenderRequest, settings: Settings = Depends(get_settings)):
    """Stage 3 — maps to Presenton's per-slide content + export. Only tables whose
    name appears in a slide's table_ref are rendered; the rest stay in the pool."""
    pptx_tables = _select_referenced_tables(body.ppt_plan, body.tables)
    if not pptx_tables:
        return {"type": "text", "message": "No tables were referenced by the plan — nothing to render."}

    try:
        pptx_bytes = await asyncio.to_thread(build_tables_pptx, pptx_tables)
    except Exception as exc:
        logger.error("build_tables_pptx failed: %s", exc, exc_info=True)
        return {"type": "error", "message": f"PPTX generation failed: {exc}"}

    evict_expired()
    title = body.presentation_title or body.ppt_plan.get("presentation_title") or "Presentation"
    filename = f"{title}.pptx"
    token = store_pptx(pptx_bytes, filename, settings.DOWNLOAD_TTL_SECONDS)
    return {"type": "download", "url": f"/download/{token}", "filename": filename}


# ── Convenience: all-in-one (chains the three stages) ─────────────────────────


class AnalyzeRequest(BaseModel):
    documents: list[DocumentInput] = Field(description="One or more markdown documents to analyze")
    hint: str = Field(default="", description="Optional instruction or goal for the analysis")
    n_slides: int | None = Field(default=None, description="Target slide count")
    analyze_images: bool = Field(default=True)


@router.post("/analyze", summary="All-in-one: documents → evidence → tables → outline → PPTX (renders only referenced tables)")
async def analyze_endpoint(body: AnalyzeRequest, settings: Settings = Depends(get_settings)):
    parsed_docs = _resolve_documents(body.documents)
    evidence_items = await _build_evidence(parsed_docs, settings, body.analyze_images)
    evidence_summary = summarize_evidence(evidence_items)
    canonical_tables = await extract_canonical_tables(evidence_items, settings, hint=body.hint)

    ppt_plan = await generate_ppt_plan(
        canonical_tables, evidence_summary, settings,
        presentation_hint=body.hint, n_slides=body.n_slides,
    )

    table_models = [TableModel(**t) for t in canonical_tables]
    pptx_tables = _select_referenced_tables(ppt_plan, table_models)

    if not pptx_tables:
        return {
            "type": "text",
            "message": "No tables were referenced by the plan.",
            "evidence_summary": evidence_summary,
            "ppt_plan": ppt_plan,
            "table_catalog": [
                {"name": t.get("name"), "title": t.get("title"), "rows": len(t.get("rows", []))}
                for t in canonical_tables
            ],
        }

    try:
        pptx_bytes = await asyncio.to_thread(build_tables_pptx, pptx_tables)
    except Exception as exc:
        logger.error("build_tables_pptx failed: %s", exc, exc_info=True)
        return {"type": "error", "message": f"PPTX generation failed: {exc}"}

    evict_expired()
    presentation_title = ppt_plan.get("presentation_title") or canonical_tables[0].get("title", "Analysis")
    filename = f"{presentation_title}.pptx"
    token = store_pptx(pptx_bytes, filename, settings.DOWNLOAD_TTL_SECONDS)

    return {
        "type": "download",
        "url": f"/download/{token}",
        "filename": filename,
        "ppt_plan": ppt_plan,
        "rendered_tables": [t["table_id"] for t in pptx_tables],
        "table_catalog": [
            {"name": t.get("name"), "title": t.get("title"), "rows": len(t.get("rows", []))}
            for t in canonical_tables
        ],
        "evidence_summary": evidence_summary,
    }
