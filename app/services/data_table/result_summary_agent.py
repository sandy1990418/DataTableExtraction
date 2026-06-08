"""Bounded Result Summary Agent: paper-level understanding → ResultSummaryPlan.

Uses a real LangGraph ReAct agent (create_react_agent) with 7 fixed tools.
The LLM steers the Thought→Action→Observation loop and terminates by calling
`create_summary_plan`, which produces the ResultSummaryPlan.

The plan is consumed by table_planner and compose_result_summary to anchor row
selection on actual methods/systems rather than raw source-table labels.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from app.config import Settings
from app.models.data_table import EvidenceBlock
from app.services.data_table.source_table_summary import SourceTableSummary

logger = logging.getLogger(__name__)

# Standard columns for result_summary tables.
RESULT_SUMMARY_COLUMNS = [
    "Method / System",
    "Main Benchmark / Task",
    "Representative Result",
    "Compared Against",
    "Key Takeaway",
    "Notes",
    "Sources",
]

# Labels that are almost always datasets/benchmarks, not method rows.
_KNOWN_DATASET_LABELS = {
    "locomo", "longmemeval", "memgpt-bench", "memoryhalluc",
    "longbench", "convqa", "multi-hop", "single-hop", "temporal",
}


class ResultSummaryPlan(BaseModel):
    """Paper-level result understanding produced by the ResultSummaryAgent."""

    row_grain: str = "method / system"
    columns: list[str] = RESULT_SUMMARY_COLUMNS

    # Methods/systems confirmed to exist in evidence → must become rows.
    must_include: list[str] = []

    # Labels that are datasets, benchmarks, or task names → exclude as row subjects.
    exclude_as_rows: list[str] = []

    # Names identified as baselines only (not the proposed system).
    baseline_labels: list[str] = []

    # table_id → classification: "main_result" | "ablation" | "efficiency" | "external"
    table_classifications: dict[str, str] = {}

    # method → one-line description of headline metric evidence.
    representative_metrics: dict[str, str] = {}

    # Groups of name variants that should be merged into one row.
    row_groupings: list[list[str]] = []

    notes: list[str] = []


# ---------------------------------------------------------------------------
# Serialised context shared across tools (passed as JSON strings in tool args)
# ---------------------------------------------------------------------------

def _serialize_evidence(evidence_store: list[EvidenceBlock]) -> str:
    """Compact JSON snapshot of evidence for tool arguments."""
    items = []
    for b in evidence_store:
        items.append({
            "evidence_id": b.evidence_id,
            "document_name": b.document_name or b.source_id,
            "title": b.title,
            "kind": b.kind,
            "text_preview": (b.text or "")[:600].replace("\n", " "),
            "has_table": bool(b.table_markdown),
            "table_header_preview": b.table_markdown.splitlines()[0][:120] if b.table_markdown else None,
        })
    return json.dumps(items, ensure_ascii=False)


def _serialize_summaries(summaries: list[SourceTableSummary]) -> str:
    """Compact JSON snapshot of source table summaries for tool arguments."""
    items = []
    for s in summaries:
        items.append({
            "table_id": s.table_id,
            "evidence_id": s.evidence_id,
            "title": s.title,
            "grain": s.guessed_row_grain,
            "headers": s.headers,
            "row_count": s.row_count,
            "all_row_labels": s.all_row_labels,
            "numeric_cols": s.numeric_column_count,
            "notes": s.notes,
        })
    return json.dumps(items, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool factory — returns closure-based tools bound to specific data snapshots.
# This avoids the anti-pattern of passing large JSON strings as tool arguments,
# which causes LLM truncation and json.loads "Extra data" crashes.
# ---------------------------------------------------------------------------

def _make_tools(evidence_snapshot: list[dict], summaries_snapshot: list[dict]):
    """Return 7 LangChain tools bound to the provided data via closures."""

    @tool
    def inspect_source_overview() -> str:
        """
        Action 1: Inspect the source documents.

        Returns a summary of all submitted documents: their names and aggregated
        text content. Use this to understand what papers/systems were submitted.
        No arguments needed.
        """
        doc_map: dict[str, dict] = {}
        for item in evidence_snapshot:
            name = item.get("document_name") or "(unknown)"
            if name not in doc_map:
                doc_map[name] = {"document_name": name, "text": "", "block_count": 0, "has_table": False}
            if len(doc_map[name]["text"]) < 1500:
                doc_map[name]["text"] += " " + item.get("text_preview", "")
            doc_map[name]["block_count"] += 1
            if item.get("has_table"):
                doc_map[name]["has_table"] = True
        return json.dumps({"documents": list(doc_map.values()), "total_documents": len(doc_map)}, ensure_ascii=False, indent=2)

    @tool
    def inspect_candidate_tables() -> str:
        """
        Action 2: Inspect all extracted source tables.

        Returns each table's ID, title, headers, row count, and all row labels.
        No arguments needed.
        """
        return json.dumps({"tables": summaries_snapshot, "table_count": len(summaries_snapshot)}, ensure_ascii=False, indent=2)

    @tool
    def inspect_table_context() -> str:
        """
        Action 3: Inspect the surrounding text context for each table.

        Returns text near each table giving context about what is being compared.
        No arguments needed.
        """
        ev_by_id = {e["evidence_id"]: e for e in evidence_snapshot}
        contexts = []
        for s in summaries_snapshot:
            ev = ev_by_id.get(s["evidence_id"], {})
            contexts.append({
                "table_id": s["table_id"],
                "table_title": s.get("title"),
                "document_name": ev.get("document_name"),
                "surrounding_text": ev.get("text_preview", "")[:400],
            })
        return json.dumps({"table_contexts": contexts}, ensure_ascii=False, indent=2)

    @tool
    def identify_methods_and_baselines() -> str:
        """
        Action 4: Identify which row labels are methods/systems vs. baselines vs. datasets.

        Classifies all row labels into candidate_methods, candidate_baselines, candidate_datasets.
        Also shows the primary document→system mapping.
        No arguments needed.
        """
        doc_names = list({e["document_name"] for e in evidence_snapshot if e.get("document_name")})
        primary_from_docs = []
        for name in doc_names:
            stem = re.sub(r"\.(md|pdf|txt)$", "", name, flags=re.IGNORECASE).strip()
            primary_from_docs.append(stem)

        all_labels: set[str] = set()
        dataset_grain_labels: set[str] = set()
        for s in summaries_snapshot:
            all_labels.update(s.get("all_row_labels", []))
            if s.get("grain") == "dataset":
                dataset_grain_labels.update(s.get("all_row_labels", []))

        candidate_methods, candidate_baselines, candidate_datasets = [], [], []
        for label in all_labels:
            ll = label.lower().strip()
            if any(d in ll for d in _KNOWN_DATASET_LABELS) or ll in dataset_grain_labels:
                candidate_datasets.append(label)
            elif any(tok in ll for tok in ("baseline", "vanilla", "no memory", "w/o", "without")):
                candidate_baselines.append(label)
            else:
                candidate_methods.append(label)

        return json.dumps({
            "primary_systems_from_documents": primary_from_docs,
            "candidate_methods_from_tables": sorted(set(candidate_methods) - dataset_grain_labels),
            "candidate_baselines": sorted(set(candidate_baselines)),
            "candidate_datasets": sorted(set(candidate_datasets) | dataset_grain_labels),
            "note": (
                "primary_systems_from_documents are the MOST RELIABLE signal — "
                "each submitted document IS a primary system to compare."
            ),
        }, ensure_ascii=False, indent=2)

    @tool
    def classify_experiment_tables() -> str:
        """
        Action 5: Classify each source table by its experiment type.

        Returns main_result / ablation / efficiency / external for each table.
        No arguments needed.
        """
        ev_by_id = {e["evidence_id"]: e for e in evidence_snapshot}
        classifications: dict[str, str] = {}
        for s in summaries_snapshot:
            ev = ev_by_id.get(s["evidence_id"], {})
            context = ((s.get("title") or "") + " " + ev.get("text_preview", "")).lower()
            if any(kw in context for kw in ("ablation", "ablate", "variant", "w/o", "without", "effect of")):
                classifications[s["table_id"]] = "ablation"
            elif any(kw in context for kw in ("efficiency", "latency", "speed", "memory usage", "cost", "throughput")):
                classifications[s["table_id"]] = "efficiency"
            elif any(kw in context for kw in ("appendix", "supplementary", "additional result")):
                classifications[s["table_id"]] = "external"
            else:
                classifications[s["table_id"]] = "main_result"
        return json.dumps({"table_classifications": classifications}, ensure_ascii=False, indent=2)

    @tool
    def select_representative_results() -> str:
        """
        Action 6: Identify the most headline-worthy metric for each method.

        Returns metric column hints per method for the Representative Result column.
        No arguments needed.
        """
        ev_by_id = {e["evidence_id"]: e for e in evidence_snapshot}
        method_hints: dict[str, str] = {}
        for s in summaries_snapshot:
            if s.get("numeric_cols", 0) < 1:
                continue
            ev = ev_by_id.get(s["evidence_id"], {})
            context = ((s.get("title") or "") + ev.get("text_preview", "")).lower()
            if any(kw in context for kw in ("ablation", "variant")):
                continue
            metric_cols = [h for h in s.get("headers", [])[1:] if h.strip()][:4]
            metric_summary = ", ".join(metric_cols) if metric_cols else "unknown"
            for label in s.get("all_row_labels", []):
                if label not in method_hints:
                    method_hints[label] = f"Table '{s.get('title') or s['table_id']}': metrics = {metric_summary}"
        return json.dumps({"method_metric_hints": method_hints}, ensure_ascii=False, indent=2)

    @tool
    def create_summary_plan(plan_json: str) -> str:
        """
        Action 7 (FINAL): Create the ResultSummaryPlan and terminate the agent.

        Call this LAST after inspecting evidence, identifying methods, classifying
        tables, and selecting metrics. This tool signals that the agent is done.

        Args:
            plan_json: A JSON string with the ResultSummaryPlan fields:
                {
                  "row_grain": "method / system",
                  "columns": [...],
                  "must_include": ["A-MEM", "MemGPT", ...],
                  "exclude_as_rows": ["LOCOMO", ...],
                  "baseline_labels": ["READAGENT", ...],
                  "table_classifications": {"tbl_1": "main_result", ...},
                  "representative_metrics": {"A-MEM": "Avg F1/BLEU 27.02/20.09", ...},
                  "row_groupings": [["A-MEM", "AMEM"]],
                  "notes": [...]
                }
        """
        try:
            data = json.loads(plan_json)
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"Invalid JSON: {exc}", "status": "failed"})
        plan = _parse_result_summary_plan(data)
        return json.dumps({"status": "done", "plan": plan.model_dump()}, ensure_ascii=False, indent=2)

    return [
        inspect_source_overview,
        inspect_candidate_tables,
        inspect_table_context,
        identify_methods_and_baselines,
        classify_experiment_tables,
        select_representative_results,
        create_summary_plan,
    ]


# ---------------------------------------------------------------------------
# Parser / helpers
# ---------------------------------------------------------------------------

def _parse_result_summary_plan(data: dict) -> ResultSummaryPlan:
    columns = data.get("columns", RESULT_SUMMARY_COLUMNS)
    if not isinstance(columns, list) or not columns:
        columns = RESULT_SUMMARY_COLUMNS

    row_groupings = data.get("row_groupings", [])
    if not isinstance(row_groupings, list):
        row_groupings = []
    row_groupings = [g for g in row_groupings if isinstance(g, list) and len(g) >= 2]

    table_classifications = data.get("table_classifications", {})
    if not isinstance(table_classifications, dict):
        table_classifications = {}

    representative_metrics = data.get("representative_metrics", {})
    if not isinstance(representative_metrics, dict):
        representative_metrics = {}

    return ResultSummaryPlan(
        row_grain=str(data.get("row_grain", "method / system")),
        columns=columns,
        must_include=_str_list(data.get("must_include", [])),
        exclude_as_rows=_str_list(data.get("exclude_as_rows", [])),
        baseline_labels=_str_list(data.get("baseline_labels", [])),
        table_classifications=table_classifications,
        representative_metrics=representative_metrics,
        row_groupings=row_groupings,
        notes=_str_list(data.get("notes", [])),
    )


def _str_list(val) -> list[str]:
    if not isinstance(val, list):
        return []
    return [str(v) for v in val if v]


def _robust_json_loads(text: str) -> dict:
    text = text.strip()
    start = text.find('{')
    if start == -1:
        raise ValueError("No JSON object found")
    # Try full string first
    try:
        return json.loads(text[start:])
    except json.JSONDecodeError:
        pass
    # Extract first balanced JSON object
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    pass
    raise ValueError("Could not extract valid JSON")


def _extract_plan_from_messages(messages) -> ResultSummaryPlan | None:
    """Parse the ResultSummaryPlan from the agent's final tool call result."""
    for msg in reversed(messages):
        # Look for tool call result from create_summary_plan
        if hasattr(msg, "content") and isinstance(msg.content, str):
            try:
                data = _robust_json_loads(msg.content)
                if data.get("status") == "done" and "plan" in data:
                    return _parse_result_summary_plan(data["plan"])
            except (json.JSONDecodeError, AttributeError, ValueError):
                pass
        # Also check tool_calls on AIMessage
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
            for tc in (msg.tool_calls or []):
                if tc.get("name") == "create_summary_plan":
                    try:
                        raw = tc.get("args", {}).get("plan_json", "{}")
                        data = json.loads(raw)
                        return _parse_result_summary_plan(data)
                    except Exception:
                        pass
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Architecture-comparison columns — used when hint is about design/mechanism, not benchmarks.
ARCH_COMPARISON_COLUMNS = [
    "System",
    "Memory Architecture",
    "Memory Update / Retrieval",
    "Key Innovation",
    "Main Evaluation",
    "Best Reported Result",
    "Key Takeaway",
]

# Keywords that signal an architecture / design comparison rather than a benchmark summary.
_ARCH_KEYWORDS = {
    "架構", "architecture", "design", "mechanism", "how", "compare system",
    "比較架構", "compare architecture", "memory design", "system design",
    "memory mechanism", "how does", "approaches", "技術比較",
}


def _is_architecture_hint(hint: str) -> bool:
    h = hint.lower()
    return any(kw in h for kw in _ARCH_KEYWORDS)


_SYSTEM_PROMPT = """\
You are a paper-reading assistant. Your job is to understand a set of submitted academic \
papers and produce a ResultSummaryPlan — NOT the final table.

You have exactly 7 tools. Use them in roughly this order:
1. inspect_source_overview — understand what documents were submitted
2. inspect_candidate_tables — see what structured tables exist
3. inspect_table_context — read the context around each table
4. identify_methods_and_baselines — classify row labels: primary systems vs baselines vs datasets
5. classify_experiment_tables — label each table as main_result/ablation/efficiency/external
6. select_representative_results — find headline metrics per method
7. create_summary_plan — OUTPUT THE FINAL PLAN (this terminates the agent)

Critical rules:
- ALWAYS call create_summary_plan as your final action.
- must_include = the PRIMARY SYSTEMS being compared (one per submitted document).
  Each submitted document IS a primary system. AMem.md → A-MEM, MemGPT.md → MemGPT, etc.
- exclude_as_rows = datasets, benchmarks, evaluation tasks (e.g. LOCOMO, LongMemEval).
  LOCOMO is an evaluation benchmark, NOT a memory system.
- baseline_labels = systems that appear ONLY as baselines in other papers' tables.
- row_grain must be "method / system".
- Do NOT generate the table itself. Only produce the plan JSON.

=== CHOOSING COLUMNS ===

If the user hint is about ARCHITECTURE, DESIGN, or MECHANISM comparison
(e.g. "比較這些架構", "compare architectures", "how do these systems work",
"compare system design", "memory mechanism"):
  → Set columns to:
    ["System", "Memory Architecture", "Memory Update / Retrieval",
     "Key Innovation", "Main Evaluation", "Best Reported Result", "Key Takeaway"]
  → These columns should be filled from the paper text, not just tables.

If the user hint is about EXPERIMENT RESULTS or BENCHMARKS
(e.g. "compare results", "which system performs best", "summarize benchmark"):
  → Use default columns:
    ["Method / System", "Main Benchmark / Task", "Representative Result",
     "Compared Against", "Key Takeaway", "Limitations / Notes", "Sources"]
"""


async def run_result_summary_agent(
    hint: str,
    evidence_store: list[EvidenceBlock],
    source_table_summaries: list[SourceTableSummary],
    settings: Settings,
    debug_trace: list | None = None,
) -> ResultSummaryPlan | None:
    """
    Run the bounded 7-tool LangGraph ReAct agent to produce a ResultSummaryPlan.

    Always runs when evidence is available — the agent handles all hint languages
    and query types. Returns None only if evidence is empty.
    """
    if not source_table_summaries and not any(b.text for b in evidence_store):
        return None

    if debug_trace is not None:
        debug_trace.append({
            "stage": "result_summary_agent_start",
            "hint": hint,
            "source_tables": len(source_table_summaries),
        })

    # Build closure-based tools bound to this call's data.
    # Tools access evidence/summaries directly — no JSON passed as args.
    evidence_snapshot = json.loads(_serialize_evidence(evidence_store))
    summaries_snapshot = json.loads(_serialize_summaries(source_table_summaries))
    tools = _make_tools(evidence_snapshot, summaries_snapshot)

    # Describe the documents and tables in the user message (plain text, no raw JSON).
    doc_names = list({e.get("document_name") or e.get("evidence_id") for e in evidence_snapshot})
    table_ids = [s.get("table_id") for s in summaries_snapshot]
    arch_hint = (
        "\nHint type: ARCHITECTURE/DESIGN comparison — suggest architecture-focused columns."
        if _is_architecture_hint(hint) else
        "\nHint type: BENCHMARK/RESULTS comparison — use default result_summary columns."
    )
    user_content = (
        f"User hint: {hint}\n"
        f"{arch_hint}\n\n"
        f"Submitted documents: {', '.join(doc_names)}\n"
        f"Available source tables: {', '.join(table_ids)}\n\n"
        "Use the tools to inspect the evidence and produce a ResultSummaryPlan. "
        "Call create_summary_plan as your final action."
    )

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
        temperature=0.0,
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=_SYSTEM_PROMPT,
    )

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_content}]},
            config={"recursion_limit": 20},
        )
        messages = result.get("messages", [])
        plan = _extract_plan_from_messages(messages)

        if plan is None:
            logger.warning("ResultSummaryAgent: create_summary_plan was not called — using empty plan")
            plan = ResultSummaryPlan()

        if debug_trace is not None:
            debug_trace.append({
                "stage": "result_summary_agent_plan",
                "must_include": plan.must_include,
                "exclude_as_rows": plan.exclude_as_rows,
                "baseline_labels": plan.baseline_labels,
                "table_classifications": plan.table_classifications,
                "agent_steps": len(messages),
            })

        return plan

    except Exception as exc:
        logger.warning("ResultSummaryAgent failed: %s — using empty plan", exc)
        if debug_trace is not None:
            debug_trace.append({
                "stage": "result_summary_agent_error",
                "error": str(exc),
            })
        return ResultSummaryPlan()
