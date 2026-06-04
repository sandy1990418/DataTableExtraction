from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.config import Settings
from app.models import EvidenceItem
from app.services.canonical_extractor import populate_tables
from app.services.evidence_selector import select_table_evidence_blocks
from app.services.table_qa import review_tables
from app.services.table_revision import revise_tables


@dataclass
class AgentTraceEvent:
    agent: str
    action: str
    detail: dict = field(default_factory=dict)


@dataclass
class TableAgentTeamResult:
    tables: list[dict]
    table_evidence_blocks: dict[str, str]
    qa_reviews: list[dict]
    agent_trace: list[dict]


class TableAgentState(TypedDict, total=False):
    ppt_plan: dict
    specs: list[dict]
    evidence_items: list[EvidenceItem]
    fallback_evidence_block: str
    settings: Settings
    cache: dict[str, dict]
    max_revision_rounds: int
    revision_round: int
    referenced: list[str]
    table_evidence_blocks: dict[str, str]
    tables: list[dict]
    qa_reviews: list[dict]
    agent_trace: list[dict]


async def run_table_agent_team(
    ppt_plan: dict,
    specs: list[dict],
    evidence_items: list[EvidenceItem],
    fallback_evidence_block: str,
    settings: Settings,
    cache: dict[str, dict] | None = None,
    max_revision_rounds: int = 1,
) -> TableAgentTeamResult:
    """Run the LangGraph table-focused agent team for the referenced tables."""
    graph = _build_table_agent_graph()
    state = await graph.ainvoke(
        {
            "ppt_plan": ppt_plan,
            "specs": specs,
            "evidence_items": evidence_items,
            "fallback_evidence_block": fallback_evidence_block,
            "settings": settings,
            "cache": cache if cache is not None else {},
            "max_revision_rounds": max_revision_rounds,
            "revision_round": 0,
            "agent_trace": [],
        }
    )
    return TableAgentTeamResult(
        tables=state.get("tables", []),
        table_evidence_blocks=state.get("table_evidence_blocks", {}),
        qa_reviews=state.get("qa_reviews", []),
        agent_trace=state.get("agent_trace", []),
    )


def _build_table_agent_graph():
    workflow = StateGraph(TableAgentState)
    workflow.add_node("supervisor", _supervisor_node)
    workflow.add_node("select_evidence", _select_evidence_node)
    workflow.add_node("populate_tables", _populate_tables_node)
    workflow.add_node("review_tables", _review_tables_node)
    workflow.add_node("revise_tables", _revise_tables_node)

    workflow.set_entry_point("supervisor")
    workflow.add_edge("supervisor", "select_evidence")
    workflow.add_edge("select_evidence", "populate_tables")
    workflow.add_edge("populate_tables", "review_tables")
    workflow.add_conditional_edges(
        "review_tables",
        _route_after_review,
        {"revise": "revise_tables", "done": END},
    )
    workflow.add_edge("revise_tables", "review_tables")
    return workflow.compile()


async def _supervisor_node(state: TableAgentState) -> dict[str, Any]:
    referenced = _referenced_names(state["ppt_plan"])
    return {
        "referenced": referenced,
        "agent_trace": [
            *state.get("agent_trace", []),
            _trace(
                "SupervisorAgent",
                "resolved_fixed_workflow",
                {
                    "workflow": "select_evidence -> populate_tables -> review_tables -> revise_tables?",
                    "referenced_tables": referenced,
                },
            ),
        ],
    }


async def _select_evidence_node(state: TableAgentState) -> dict[str, Any]:
    table_evidence_blocks = select_table_evidence_blocks(state["specs"], state["evidence_items"])
    table_evidence_blocks[""] = state["fallback_evidence_block"]
    referenced = state.get("referenced", [])
    return {
        "table_evidence_blocks": table_evidence_blocks,
        "agent_trace": [
            *state.get("agent_trace", []),
            _trace(
                "EvidenceSelectorAgent",
                "selected_focused_evidence",
                {
                    "table_count": len(table_evidence_blocks) - 1,
                    "referenced_tables": [name for name in referenced if name in table_evidence_blocks],
                },
            ),
        ],
    }


async def _populate_tables_node(state: TableAgentState) -> dict[str, Any]:
    cache = state.get("cache", {})
    referenced = state.get("referenced", [])
    specs_by_name = {str(spec.get("name")): spec for spec in state.get("specs", []) if spec.get("name")}
    missing_specs = [specs_by_name[name] for name in referenced if name in specs_by_name and name not in cache]

    if missing_specs:
        populated = await populate_tables(missing_specs, state["table_evidence_blocks"], state["settings"])
        for table in populated:
            cache[str(table.get("name"))] = table

    return {
        "cache": cache,
        "tables": _selected_tables(referenced, cache),
        "agent_trace": [
            *state.get("agent_trace", []),
            _trace(
                "TablePopulatorAgent",
                "populated_tables",
                {"requested": [spec.get("name") for spec in missing_specs], "cached": len(cache)},
            ),
        ],
    }


async def _review_tables_node(state: TableAgentState) -> dict[str, Any]:
    tables = state.get("tables", [])
    reviews = await review_tables(
        _reviewable_tables(tables),
        state["table_evidence_blocks"],
        state["settings"],
    )
    action = "reviewed_revisions" if state.get("revision_round", 0) else "reviewed_tables"
    detail = {"needs_revision": _needs_revision_names(reviews)}
    if state.get("revision_round", 0):
        detail["round"] = state["revision_round"]
    return {
        "qa_reviews": reviews,
        "agent_trace": [
            *state.get("agent_trace", []),
            _trace("TableReviewerAgent", action, detail),
        ],
    }


async def _revise_tables_node(state: TableAgentState) -> dict[str, Any]:
    revision_names = set(_needs_revision_names(state.get("qa_reviews", [])))
    reviewable = _reviewable_tables(state.get("tables", []))
    revision_tables = [table for table in reviewable if _table_id(table) in revision_names]
    revised = await revise_tables(
        revision_tables,
        state.get("qa_reviews", []),
        state["table_evidence_blocks"],
        state["settings"],
    )
    cache = state.get("cache", {})
    for table in revised:
        name = _table_id(table)
        if name:
            cache[name] = _canonical_from_reviewable(table)

    revision_round = state.get("revision_round", 0) + 1
    referenced = state.get("referenced", [])
    return {
        "cache": cache,
        "tables": _selected_tables(referenced, cache),
        "revision_round": revision_round,
        "agent_trace": [
            *state.get("agent_trace", []),
            _trace(
                "TableRevisionAgent",
                "revised_tables",
                {"round": revision_round, "tables": [_table_id(table) for table in revised]},
            ),
        ],
    }


def _route_after_review(state: TableAgentState) -> str:
    if _needs_revision_names(state.get("qa_reviews", [])) and (
        state.get("revision_round", 0) < state.get("max_revision_rounds", 1)
    ):
        return "revise"
    return "done"


def _trace(agent: str, action: str, detail: dict | None = None) -> dict:
    return asdict(AgentTraceEvent(agent, action, detail or {}))


def _referenced_names(ppt_plan: dict) -> list[str]:
    names = [str(slide.get("table_ref")) for slide in ppt_plan.get("slides", []) if slide.get("table_ref")]
    return list(dict.fromkeys(names))


def _selected_tables(referenced: list[str], cache: dict[str, dict]) -> list[dict]:
    return [cache[name] for name in referenced if name in cache]


def _reviewable_tables(tables: list[dict]) -> list[dict]:
    out = []
    for table in tables:
        table_id = str(table.get("name") or table.get("table_id") or table.get("title") or "")
        out.append({**table, "table_id": table_id})
    return out


def _canonical_from_reviewable(table: dict) -> dict:
    out = dict(table)
    out["name"] = str(out.get("name") or out.get("table_id") or out.get("title") or "")
    out.pop("table_id", None)
    return out


def _table_id(table: dict) -> str:
    return str(table.get("table_id") or table.get("name") or table.get("title") or "")


def _needs_revision_names(reviews: list[dict]) -> list[str]:
    names = []
    for review in reviews:
        if review.get("status") == "needs_revision" or review.get("unsupported_cells"):
            names.append(str(review.get("table_id")))
    return names
