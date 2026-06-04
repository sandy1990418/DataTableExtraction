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
from app.services.session_store import get_session, store_session
from app.tools.table_pptx import build_tables_pptx, evict_expired, store_pptx

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Shared models ────────────────────────────────────────────────────────────


class DocumentInput(BaseModel):
    name: str = Field(default="", description="Document name; defaults to filename if file_path is given")
    file_path: str | None = Field(default=None, description="Absolute or relative path to a .md file; server reads it directly")
    content: str = Field(default="", description="Markdown content (used when file_path is not given)")
    base_dir: str | None = Field(default=None, description="Base directory for resolving image paths; defaults to file_path's parent")


# ── Shared pipeline steps (used by both the staged endpoints and /analyze) ────


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


def _table_catalog(tables: list[dict]) -> list[dict]:
    """Lightweight view (no rows) for outline-time table selection."""
    return [
        {
            "name": t.get("name"),
            "title": t.get("title"),
            "description": t.get("description", ""),
            "row_count": len(t.get("rows", [])),
            "column_count": len(t.get("headers", [])),
        }
        for t in tables
    ]


def _select_referenced_tables(ppt_plan: dict, tables: list[dict]) -> list[dict]:
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


def _render_pptx(pptx_tables: list[dict], title: str, settings: Settings) -> dict:
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


# ── Stage 1: /evidence — parse + extract tables, cache, return session_id ─────


class EvidenceRequest(BaseModel):
    documents: list[DocumentInput] = Field(description="One or more markdown documents")
    hint: str = Field(default="", description="Optional goal that steers which tables get extracted")
    analyze_images: bool = Field(default=True, description="Run vision analysis on embedded images")


@router.post("/evidence", summary="Documents → evidence + canonical table pool; cached under a session_id")
async def evidence_endpoint(body: EvidenceRequest, settings: Settings = Depends(get_settings)):
    """Stage 1 — run ONCE per document set. The full table pool is cached server-side;
    you get back a session_id to hand to /outline and /render."""
    parsed_docs = _resolve_documents(body.documents)
    evidence_items = await _build_evidence(parsed_docs, settings, body.analyze_images)
    evidence_summary = summarize_evidence(evidence_items)
    canonical_tables = await extract_canonical_tables(evidence_items, settings, hint=body.hint)

    session_id = store_session(
        {
            "evidence_summary": evidence_summary,
            "tables": canonical_tables,
            "hint": body.hint,
        },
        settings.SESSION_TTL_SECONDS,
    )

    return {
        "session_id": session_id,
        "evidence_summary": evidence_summary,
        "table_catalog": _table_catalog(canonical_tables),
    }


# ── Stage 2: /outline — session_id → slide plan ──────────────────────────────


class OutlineRequest(BaseModel):
    session_id: str = Field(description="session_id from /evidence")
    hint: str = Field(default="", description="Presentation goal; defaults to the /evidence hint")
    n_slides: int | None = Field(default=None, description="Target slide count")


@router.post("/outline", summary="session_id → slide outline; LLM picks which tables deserve a slide")
async def outline_endpoint(body: OutlineRequest, settings: Settings = Depends(get_settings)):
    """Stage 2 — maps to Presenton's outline generation. Reads the FULL cached tables
    (so the planner sees real row counts). Each slide carries table_ref (a table name)
    or null; unreferenced tables won't be rendered."""
    session = get_session(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session_id; call /evidence first.")

    ppt_plan = await generate_ppt_plan(
        session["tables"],
        session["evidence_summary"],
        settings,
        presentation_hint=body.hint or session.get("hint", ""),
        n_slides=body.n_slides,
    )
    referenced = [s.get("table_ref") for s in ppt_plan.get("slides", []) if s.get("table_ref")]
    return {"ppt_plan": ppt_plan, "referenced_tables": referenced}


# ── Stage 3: /render — session_id + plan → PPTX (only referenced tables) ──────


class RenderRequest(BaseModel):
    session_id: str = Field(description="session_id from /evidence")
    ppt_plan: dict = Field(description="Plan from /outline")
    presentation_title: str | None = Field(default=None)


@router.post("/render", summary="session_id + plan → PPTX; renders ONLY tables referenced by the plan")
async def render_endpoint(body: RenderRequest, settings: Settings = Depends(get_settings)):
    """Stage 3 — maps to Presenton's per-slide content + export. Pulls full rows from
    the cached session for tables whose name appears in a slide's table_ref."""
    session = get_session(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session_id; call /evidence first.")

    pptx_tables = _select_referenced_tables(body.ppt_plan, session["tables"])
    title = body.presentation_title or body.ppt_plan.get("presentation_title") or "Presentation"
    return await asyncio.to_thread(_render_pptx, pptx_tables, title, settings)


# ── Convenience: /analyze — all-in-one (recommended for simple use) ──────────


class AnalyzeRequest(BaseModel):
    documents: list[DocumentInput] = Field(description="One or more markdown documents to analyze")
    hint: str = Field(default="", description="Optional instruction or goal for the analysis")
    n_slides: int | None = Field(default=None, description="Target slide count")
    analyze_images: bool = Field(default=True)


@router.post("/analyze", summary="All-in-one: documents → evidence → tables → outline → PPTX")
async def analyze_endpoint(body: AnalyzeRequest, settings: Settings = Depends(get_settings)):
    parsed_docs = _resolve_documents(body.documents)
    evidence_items = await _build_evidence(parsed_docs, settings, body.analyze_images)
    evidence_summary = summarize_evidence(evidence_items)
    canonical_tables = await extract_canonical_tables(evidence_items, settings, hint=body.hint)

    ppt_plan = await generate_ppt_plan(
        canonical_tables, evidence_summary, settings,
        presentation_hint=body.hint, n_slides=body.n_slides,
    )

    pptx_tables = _select_referenced_tables(ppt_plan, canonical_tables)
    title = ppt_plan.get("presentation_title") or (canonical_tables[0].get("title") if canonical_tables else "Analysis")
    result = await asyncio.to_thread(_render_pptx, pptx_tables, title, settings)

    result.update(
        {
            "ppt_plan": ppt_plan,
            "rendered_tables": [t["table_id"] for t in pptx_tables],
            "table_catalog": _table_catalog(canonical_tables),
            "evidence_summary": evidence_summary,
        }
    )
    return result
