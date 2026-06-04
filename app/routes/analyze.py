from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from app.config import Settings, get_settings
from app.models import AnalyzeRequest, EvidenceRequest, OutlineRequest, RenderRequest
from app.services.canonical_extractor import plan_tables
from app.services.evidence_layer import summarize_evidence
from app.services.pipeline import (
    build_evidence,
    render_pptx,
    resolve_documents,
    select_referenced_tables,
    spec_catalog,
    validate_ppt_plan,
    validate_render_inputs,
    validate_table_quality,
)
from app.services.ppt_planner import generate_ppt_plan
from app.services.report_agent_team import run_table_report_agent_team
from app.services.session_store import get_session, store_session
from app.services.table_agent_team import run_table_agent_team
from app.services.table_qa import review_warnings
from app.services.table_spec_qa import spec_review_warnings

router = APIRouter()


@router.post("/evidence", summary="Documents → evidence + table PLAN (cheap); cached under a session_id")
async def evidence_endpoint(body: EvidenceRequest, settings: Settings = Depends(get_settings)):
    """Stage 1 — run ONCE per document set. Does only the cheap planning (which tables
    could exist + their columns); rows are NOT filled yet. Returns a session_id."""
    parsed_docs = resolve_documents(body.documents)
    evidence_items = await build_evidence(parsed_docs, settings, body.analyze_images)
    evidence_summary = summarize_evidence(evidence_items)
    specs, evidence_block = await plan_tables(evidence_items, settings, hint=body.hint)

    session_id = store_session(
        {
            "evidence_summary": evidence_summary,
            "evidence_block": evidence_block,
            "evidence_items": evidence_items,
            "specs": specs,
            "populated": {},  # lazy cache: name -> populated table
            "hint": body.hint,
        },
        settings.SESSION_TTL_SECONDS,
    )

    return {
        "session_id": session_id,
        "evidence_summary": evidence_summary,
        "table_catalog": spec_catalog(specs),
    }


@router.post("/outline", summary="session_id → slide outline; LLM picks which tables deserve a slide")
async def outline_endpoint(body: OutlineRequest, settings: Settings = Depends(get_settings)):
    """Stage 2 — maps to Presenton's outline generation. Selects tables from the plan
    (by columns/description) via table_ref. No rows are needed here."""
    session = get_session(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session_id; call /evidence first.")

    ppt_plan = await generate_ppt_plan(
        session["specs"],
        session["evidence_summary"],
        settings,
        presentation_hint=body.hint or session.get("hint", ""),
        n_slides=body.n_slides,
    )
    referenced = [s.get("table_ref") for s in ppt_plan.get("slides", []) if s.get("table_ref")]
    return {
        "ppt_plan": ppt_plan,
        "referenced_tables": referenced,
        "warnings": validate_ppt_plan(ppt_plan, session["specs"]),
    }


@router.post("/render", summary="session_id + plan → PPTX; populates & renders ONLY referenced tables")
async def render_endpoint(body: RenderRequest, settings: Settings = Depends(get_settings)):
    """Stage 3 — the expensive row population happens HERE, lazily, only for tables the
    plan references (cached on the session so re-renders are free)."""
    session = get_session(body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown or expired session_id; call /evidence first.")

    team_result = await run_table_agent_team(
        body.ppt_plan,
        session["specs"],
        session["evidence_items"],
        session["evidence_block"],
        settings,
        cache=session["populated"],
    )
    pptx_tables = select_referenced_tables(body.ppt_plan, team_result.tables)
    title = body.presentation_title or body.ppt_plan.get("presentation_title") or "Presentation"
    result = await asyncio.to_thread(render_pptx, pptx_tables, title, settings, body.ppt_plan)
    result["rendered_tables"] = [t["table_id"] for t in pptx_tables]
    result["qa_reviews"] = team_result.qa_reviews
    result["agent_trace"] = team_result.agent_trace
    result["warnings"] = [
        *validate_ppt_plan(body.ppt_plan, session["specs"]),
        *validate_render_inputs(body.ppt_plan, pptx_tables),
        *validate_table_quality(pptx_tables),
        *review_warnings(team_result.qa_reviews),
    ]
    return result


@router.post("/analyze", summary="All-in-one agent team: documents + hint → tables → PPTX")
async def analyze_endpoint(body: AnalyzeRequest, settings: Settings = Depends(get_settings)):
    parsed_docs = resolve_documents(body.documents)
    evidence_items = await build_evidence(parsed_docs, settings, body.analyze_images)
    evidence_summary = summarize_evidence(evidence_items)

    team_result = await run_table_report_agent_team(
        evidence_items,
        settings,
        hint=body.hint,
        n_slides=body.n_slides,
    )
    ppt_plan = team_result.ppt_plan
    specs = team_result.specs
    pptx_tables = select_referenced_tables(ppt_plan, team_result.tables)
    title = ppt_plan.get("presentation_title") or (specs[0].get("title") if specs else "Analysis")
    result = await asyncio.to_thread(render_pptx, pptx_tables, title, settings, ppt_plan)

    result.update(
        {
            "ppt_plan": ppt_plan,
            "rendered_tables": [t["table_id"] for t in pptx_tables],
            "table_catalog": spec_catalog(specs),
            "evidence_summary": evidence_summary,
            "focused_evidence_count": team_result.focused_evidence_count,
            "spec_reviews": team_result.spec_reviews,
            "qa_reviews": team_result.qa_reviews,
            "agent_trace": team_result.agent_trace,
            "warnings": [
                *validate_ppt_plan(ppt_plan, specs),
                *validate_render_inputs(ppt_plan, pptx_tables),
                *validate_table_quality(pptx_tables),
                *spec_review_warnings(team_result.spec_reviews),
                *review_warnings(team_result.qa_reviews),
            ],
        }
    )
    return result
