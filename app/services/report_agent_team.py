from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.config import Settings
from app.models import EvidenceItem
from app.services.canonical_extractor import plan_tables, populate_tables
from app.services.evidence_selector import select_table_evidence_blocks
from app.services.table_qa import review_tables
from app.services.table_revision import revise_tables
from app.services.table_spec_qa import review_table_specs, revise_table_specs

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "this",
    "that",
    "what",
    "make",
    "table",
    "tables",
    "compare",
    "comparison",
    "summary",
    "content",
    "document",
    "documents",
    "paper",
    "papers",
}


@dataclass
class ReportAgentTraceEvent:
    agent: str
    action: str
    detail: dict = field(default_factory=dict)


@dataclass
class TableReportAgentResult:
    specs: list[dict]
    tables: list[dict]
    ppt_plan: dict
    focused_evidence_count: int
    table_evidence_blocks: dict[str, str]
    spec_reviews: list[dict]
    qa_reviews: list[dict]
    agent_trace: list[dict]


class TableReportAgentState(TypedDict, total=False):
    hint: str
    evidence_items: list[EvidenceItem]
    settings: Settings
    n_slides: int | None
    max_spec_revision_rounds: int
    spec_revision_round: int
    max_row_revision_rounds: int
    row_revision_round: int
    focused_evidence_items: list[EvidenceItem]
    specs: list[dict]
    planning_evidence_block: str
    spec_reviews: list[dict]
    table_evidence_blocks: dict[str, str]
    tables: list[dict]
    qa_reviews: list[dict]
    ppt_plan: dict
    agent_trace: list[dict]


async def run_table_report_agent_team(
    evidence_items: list[EvidenceItem],
    settings: Settings,
    hint: str = "",
    n_slides: int | None = None,
    max_spec_revision_rounds: int = 1,
    max_row_revision_rounds: int = 1,
) -> TableReportAgentResult:
    """Run the all-in-one table report agent team.

    The user-facing flow is path(s) + hint in, PPT-ready tables out. Internally the
    graph keeps explicit checkpoints so wrong table specs can be reviewed before
    rows are populated.
    """
    graph = _build_table_report_agent_graph()
    state = await graph.ainvoke(
        {
            "hint": hint,
            "evidence_items": evidence_items,
            "settings": settings,
            "n_slides": n_slides,
            "max_spec_revision_rounds": max_spec_revision_rounds,
            "spec_revision_round": 0,
            "max_row_revision_rounds": max_row_revision_rounds,
            "row_revision_round": 0,
            "agent_trace": [],
        }
    )
    return TableReportAgentResult(
        specs=state.get("specs", []),
        tables=state.get("tables", []),
        ppt_plan=state.get("ppt_plan", _build_agent_ppt_plan(hint, [], [], n_slides)),
        focused_evidence_count=len(state.get("focused_evidence_items", [])),
        table_evidence_blocks=state.get("table_evidence_blocks", {}),
        spec_reviews=state.get("spec_reviews", []),
        qa_reviews=state.get("qa_reviews", []),
        agent_trace=state.get("agent_trace", []),
    )


def _build_table_report_agent_graph():
    workflow = StateGraph(TableReportAgentState)
    workflow.add_node("supervisor", _supervisor_node)
    workflow.add_node("select_report_evidence", _select_report_evidence_node)
    workflow.add_node("plan_table_specs", _plan_table_specs_node)
    workflow.add_node("review_table_specs", _review_table_specs_node)
    workflow.add_node("revise_table_specs", _revise_table_specs_node)
    workflow.add_node("select_table_evidence", _select_table_evidence_node)
    workflow.add_node("populate_tables", _populate_tables_node)
    workflow.add_node("review_tables", _review_tables_node)
    workflow.add_node("revise_tables", _revise_tables_node)
    workflow.add_node("build_ppt_plan", _build_ppt_plan_node)

    workflow.set_entry_point("supervisor")
    workflow.add_edge("supervisor", "select_report_evidence")
    workflow.add_edge("select_report_evidence", "plan_table_specs")
    workflow.add_edge("plan_table_specs", "review_table_specs")
    workflow.add_conditional_edges(
        "review_table_specs",
        _route_after_spec_review,
        {"revise": "revise_table_specs", "populate": "select_table_evidence"},
    )
    workflow.add_edge("revise_table_specs", "review_table_specs")
    workflow.add_edge("select_table_evidence", "populate_tables")
    workflow.add_edge("populate_tables", "review_tables")
    workflow.add_conditional_edges(
        "review_tables",
        _route_after_table_review,
        {"revise": "revise_tables", "done": "build_ppt_plan"},
    )
    workflow.add_edge("revise_tables", "review_tables")
    workflow.add_edge("build_ppt_plan", END)
    return workflow.compile()


async def _supervisor_node(state: TableReportAgentState) -> dict[str, Any]:
    return {
        "agent_trace": [
            *state.get("agent_trace", []),
            _trace(
                "ReportSupervisorAgent",
                "resolved_workflow",
                {
                    "workflow": (
                        "select_report_evidence -> plan_table_specs -> review/revise specs "
                        "-> select_table_evidence -> populate/review/revise rows -> build_ppt_plan"
                    ),
                    "hint": state.get("hint", ""),
                },
            ),
        ],
    }


async def _select_report_evidence_node(state: TableReportAgentState) -> dict[str, Any]:
    items = select_report_evidence_items(state.get("evidence_items", []), state.get("hint", ""))
    return {
        "focused_evidence_items": items,
        "agent_trace": [
            *state.get("agent_trace", []),
            _trace(
                "ReportEvidenceAgent",
                "selected_hint_relevant_evidence",
                {
                    "selected": len(items),
                    "available": len(state.get("evidence_items", [])),
                    "top_titles": [item.title for item in items[:8]],
                },
            ),
        ],
    }


async def _plan_table_specs_node(state: TableReportAgentState) -> dict[str, Any]:
    all_items = state.get("evidence_items", [])
    focused = state.get("focused_evidence_items", []) or all_items
    specs, evidence_block = await plan_tables(
        focused,
        state["settings"],
        hint=state.get("hint", ""),
    )
    if not specs and focused != all_items:
        specs, evidence_block = await plan_tables(
            all_items,
            state["settings"],
            hint=state.get("hint", ""),
        )
    return {
        "specs": specs,
        "planning_evidence_block": evidence_block,
        "agent_trace": [
            *state.get("agent_trace", []),
            _trace(
                "TableIntentAgent",
                "planned_table_specs",
                {"specs": [spec.get("name") for spec in specs]},
            ),
        ],
    }


async def _review_table_specs_node(state: TableReportAgentState) -> dict[str, Any]:
    specs = state.get("specs", [])
    reviews = await review_table_specs(
        specs,
        state.get("hint", ""),
        state.get("planning_evidence_block", ""),
        state["settings"],
    )
    action = "reviewed_revised_specs" if state.get("spec_revision_round", 0) else "reviewed_specs"
    detail = {"needs_revision": _needs_spec_revision_names(reviews)}
    if state.get("spec_revision_round", 0):
        detail["round"] = state["spec_revision_round"]
    return {
        "spec_reviews": reviews,
        "agent_trace": [
            *state.get("agent_trace", []),
            _trace("TableSpecReviewAgent", action, detail),
        ],
    }


async def _revise_table_specs_node(state: TableReportAgentState) -> dict[str, Any]:
    revised = await revise_table_specs(
        state.get("specs", []),
        state.get("spec_reviews", []),
        state.get("hint", ""),
        state.get("planning_evidence_block", ""),
        state["settings"],
    )
    revision_round = state.get("spec_revision_round", 0) + 1
    return {
        "specs": revised,
        "spec_revision_round": revision_round,
        "agent_trace": [
            *state.get("agent_trace", []),
            _trace(
                "TableSpecRevisionAgent",
                "revised_specs",
                {"round": revision_round, "specs": [spec.get("name") for spec in revised]},
            ),
        ],
    }


async def _select_table_evidence_node(state: TableReportAgentState) -> dict[str, Any]:
    items = state.get("focused_evidence_items", []) or state.get("evidence_items", [])
    table_evidence_blocks = select_table_evidence_blocks(state.get("specs", []), items)
    table_evidence_blocks[""] = state.get("planning_evidence_block", "")
    return {
        "table_evidence_blocks": table_evidence_blocks,
        "agent_trace": [
            *state.get("agent_trace", []),
            _trace(
                "TableEvidenceAgent",
                "selected_table_specific_evidence",
                {"table_count": len(table_evidence_blocks) - 1},
            ),
        ],
    }


async def _populate_tables_node(state: TableReportAgentState) -> dict[str, Any]:
    tables = await populate_tables(
        state.get("specs", []),
        state.get("table_evidence_blocks", {}),
        state["settings"],
    )
    return {
        "tables": tables,
        "agent_trace": [
            *state.get("agent_trace", []),
            _trace(
                "TablePopulatorAgent",
                "populated_tables",
                {"tables": [table.get("name") for table in tables]},
            ),
        ],
    }


async def _review_tables_node(state: TableReportAgentState) -> dict[str, Any]:
    reviewable = _reviewable_tables(state.get("tables", []))
    reviews = await review_tables(
        reviewable,
        state.get("table_evidence_blocks", {}),
        state["settings"],
    )
    action = "reviewed_revised_rows" if state.get("row_revision_round", 0) else "reviewed_rows"
    detail = {"needs_revision": _needs_row_revision_names(reviews)}
    if state.get("row_revision_round", 0):
        detail["round"] = state["row_revision_round"]
    return {
        "qa_reviews": reviews,
        "agent_trace": [
            *state.get("agent_trace", []),
            _trace("TableGroundingAgent", action, detail),
        ],
    }


async def _revise_tables_node(state: TableReportAgentState) -> dict[str, Any]:
    revision_names = set(_needs_row_revision_names(state.get("qa_reviews", [])))
    reviewable = _reviewable_tables(state.get("tables", []))
    revision_tables = [table for table in reviewable if _table_id(table) in revision_names]
    revised = await revise_tables(
        revision_tables,
        state.get("qa_reviews", []),
        state.get("table_evidence_blocks", {}),
        state["settings"],
    )
    revised_by_id = {_table_id(table): _canonical_from_reviewable(table) for table in revised}
    tables = [revised_by_id.get(_table_id(table), table) for table in state.get("tables", [])]
    revision_round = state.get("row_revision_round", 0) + 1
    return {
        "tables": tables,
        "row_revision_round": revision_round,
        "agent_trace": [
            *state.get("agent_trace", []),
            _trace(
                "TableRevisionAgent",
                "revised_rows",
                {"round": revision_round, "tables": list(revised_by_id)},
            ),
        ],
    }


async def _build_ppt_plan_node(state: TableReportAgentState) -> dict[str, Any]:
    ppt_plan = _build_agent_ppt_plan(
        state.get("hint", ""),
        state.get("specs", []),
        state.get("tables", []),
        state.get("n_slides"),
    )
    return {
        "ppt_plan": ppt_plan,
        "agent_trace": [
            *state.get("agent_trace", []),
            _trace(
                "ReportComposerAgent",
                "built_ppt_plan",
                {
                    "slides": len(ppt_plan.get("slides", [])),
                    "table_refs": [
                        slide.get("table_ref")
                        for slide in ppt_plan.get("slides", [])
                        if slide.get("table_ref")
                    ],
                },
            ),
        ],
    }


def select_report_evidence_items(
    items: list[EvidenceItem],
    hint: str,
    max_items: int = 48,
) -> list[EvidenceItem]:
    if not items:
        return []
    query_terms = _keywords(hint)
    scored = sorted(
        ((_score_report_item(item, query_terms), index, item) for index, item in enumerate(items)),
        key=lambda entry: (-entry[0], entry[1]),
    )
    selected = [item for score, _, item in scored if score > 0][:max_items]
    if len(selected) < min(12, len(items)):
        selected.extend(item for _, _, item in scored if item not in selected)
        selected = selected[:max_items]
    return selected


def _score_report_item(item: EvidenceItem, query_terms: Counter[str]) -> float:
    item_text = f"{item.title} {item.content} {' '.join(item.headers)}"
    item_terms = _keywords(item_text)
    overlap = sum(min(count, item_terms.get(term, 0)) for term, count in query_terms.items())

    title = item.title.lower()
    content = item.content.lower()
    score = float(overlap)
    if item.kind in {"markdown_table", "image_table"}:
        score += 12
    if title.startswith("table "):
        score += 12
    if any(term in title or term in content[:1000] for term in query_terms):
        score += 3
    if any(term in title or term in content[:1200] for term in ("benchmark", "metric", "result", "evaluation")):
        score += 5
    if any(term in title or term in content[:1200] for term in ("architecture", "memory", "storage", "retrieval")):
        score += 4
    if not item.content.strip() and not item.rows:
        return 0
    return score


def _keywords(text: str) -> Counter[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text.lower())
    return Counter(word for word in words if word not in STOPWORDS)


def _route_after_spec_review(state: TableReportAgentState) -> str:
    if _needs_spec_revision_names(state.get("spec_reviews", [])) and (
        state.get("spec_revision_round", 0) < state.get("max_spec_revision_rounds", 1)
    ):
        return "revise"
    return "populate"


def _route_after_table_review(state: TableReportAgentState) -> str:
    if _needs_row_revision_names(state.get("qa_reviews", [])) and (
        state.get("row_revision_round", 0) < state.get("max_row_revision_rounds", 1)
    ):
        return "revise"
    return "done"


def _build_agent_ppt_plan(
    hint: str,
    specs: list[dict],
    tables: list[dict],
    n_slides: int | None,
) -> dict:
    title = _presentation_title(hint)
    table_names = [str(table.get("name") or table.get("table_id") or table.get("title")) for table in tables]
    specs_by_name = {str(spec.get("name")): spec for spec in specs if spec.get("name")}

    slides: list[dict] = [
        {
            "slide_number": 1,
            "slide_type": "title",
            "title": title,
            "content": f"Table report generated from the requested goal: {hint}".strip(),
            "table_ref": None,
            "speaker_notes": "",
        }
    ]

    include_overview = n_slides is None or n_slides > len(table_names) + 1
    if include_overview:
        slides.append(
            {
                "slide_number": len(slides) + 1,
                "slide_type": "overview",
                "title": "Table Overview",
                "content": _overview_content(specs_by_name, table_names),
                "table_ref": None,
                "speaker_notes": "",
            }
        )

    for name in table_names:
        spec = specs_by_name.get(name, {})
        slides.append(
            {
                "slide_number": len(slides) + 1,
                "slide_type": "table",
                "title": str(spec.get("title") or name),
                "content": str(spec.get("description") or ""),
                "table_ref": name,
                "speaker_notes": "",
            }
        )

    if n_slides is None or len(slides) < n_slides:
        slides.append(
            {
                "slide_number": len(slides) + 1,
                "slide_type": "conclusion",
                "title": "Synthesis",
                "content": "Use the tables as the primary evidence for comparing the requested dimensions.",
                "table_ref": None,
                "speaker_notes": "",
            }
        )

    for index, slide in enumerate(slides, 1):
        slide["slide_number"] = index
    return {"presentation_title": title, "slides": slides}


def _presentation_title(hint: str) -> str:
    cleaned = re.sub(r"\s+", " ", hint).strip(" .")
    if not cleaned:
        return "Table Report"
    cleaned = re.sub(r"^(make|create|generate|compare|summarize)\s+", "", cleaned, flags=re.IGNORECASE)
    words = cleaned.split()
    title = " ".join(word[:1].upper() + word[1:] for word in words[:10])
    return title[:90] or "Table Report"


def _overview_content(specs_by_name: dict[str, dict], table_names: list[str]) -> str:
    if not table_names:
        return "No populated tables were produced."
    lines = []
    for name in table_names:
        spec = specs_by_name.get(name, {})
        lines.append(f"- {spec.get('title') or name}: {spec.get('description') or 'generated table'}")
    return "\n".join(lines)


def _reviewable_tables(tables: list[dict]) -> list[dict]:
    return [{**table, "table_id": _table_id(table)} for table in tables]


def _canonical_from_reviewable(table: dict) -> dict:
    out = dict(table)
    out["name"] = str(out.get("name") or out.get("table_id") or out.get("title") or "")
    out.pop("table_id", None)
    return out


def _table_id(table: dict) -> str:
    return str(table.get("table_id") or table.get("name") or table.get("title") or "")


def _needs_spec_revision_names(reviews: list[dict]) -> list[str]:
    names = []
    for review in reviews:
        if review.get("status") == "needs_revision" or review.get("unsupported_columns"):
            names.append(str(review.get("name")))
    return names


def _needs_row_revision_names(reviews: list[dict]) -> list[str]:
    names = []
    for review in reviews:
        if review.get("status") == "needs_revision" or review.get("unsupported_cells"):
            names.append(str(review.get("table_id")))
    return names


def _trace(agent: str, action: str, detail: dict | None = None) -> dict:
    return asdict(ReportAgentTraceEvent(agent, action, detail or {}))
