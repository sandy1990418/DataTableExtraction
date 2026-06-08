"""LLM-based table planner: understand hint + evidence → produce DataTablePlan."""

from __future__ import annotations

import json
import logging
import re
from typing import Literal

from typing import TYPE_CHECKING

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.config import Settings
from app.models.data_table import EvidenceBlock
from app.prompts.data_table import TABLE_PLANNER_SYSTEM
from app.services.data_table.source_table_summary import SourceTableSummary

if TYPE_CHECKING:
    from app.services.data_table.result_summary_agent import ResultSummaryPlan

logger = logging.getLogger(__name__)

RESULT_SUMMARY_HEADERS = [
    "Method / System",
    "Main Benchmark / Task",
    "Representative Result",
    "Compared Against",
    "Key Takeaway",
    "Limitations / Notes",
    "Sources",
]


class PlannedColumn(BaseModel):
    name: str
    description: str
    value_type: Literal["string", "number", "boolean", "date", "list", "unknown"] = "unknown"
    evidence_policy: Literal["source_table", "text", "mixed"] = "mixed"


class EvidenceDecision(BaseModel):
    evidence_id: str
    decision: Literal["use", "maybe", "exclude"]
    reason: str


class ExcludedSourceTable(BaseModel):
    table_id: str
    evidence_id: str
    reason: str


class ExcludedCandidateRow(BaseModel):
    row_label: str
    reason: str


class DataTablePlan(BaseModel):
    table_title: str
    table_purpose: str
    table_purpose_type: Literal[
        "result_summary",
        "raw_metric_extraction",
        "source_table_reconstruction",
        "system_comparison",
    ] = "result_summary"
    row_grain: str
    table_format: Literal["wide", "long"] = "wide"
    columns: list[PlannedColumn]
    evidence_decisions: list[EvidenceDecision]
    excluded_source_tables: list[ExcludedSourceTable] = []
    candidate_rows: list[str] = []
    excluded_candidate_rows: list[ExcludedCandidateRow] = []
    generation_policy: Literal[
        "single_source_table_reconstruction",
        "coherent_synthesis",
        "system_summary_with_metrics",
    ] = "coherent_synthesis"
    warnings: list[str] = []
    reason: str = ""


# Default columns for result_summary when LLM omits or returns generic columns.
_RESULT_SUMMARY_DEFAULT_COLUMNS = [
    PlannedColumn(name=h, description=h, value_type="string", evidence_policy="mixed")
    for h in RESULT_SUMMARY_HEADERS
]

_FALLBACK_PLAN = DataTablePlan(
    table_title="Data Table",
    table_purpose="Compare entities from provided sources.",
    table_purpose_type="result_summary",
    row_grain="unknown",
    table_format="wide",
    columns=_RESULT_SUMMARY_DEFAULT_COLUMNS,
    evidence_decisions=[],
    generation_policy="coherent_synthesis",
    reason="Fallback plan — LLM planning failed.",
)


def _extract_json(raw: str) -> dict:
    """Strip markdown fences and extract the first JSON object from raw LLM output."""
    # strip ```json ... ``` or ``` ... ``` fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw.strip())

    # find first '{' and try progressively longer substrings to handle truncation
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in output")

    # try full string first (common happy path)
    candidate = raw[start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # try to find balanced braces up to the last complete top-level field
    depth = 0
    last_good_pos = start
    for i, ch in enumerate(candidate):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(candidate[: i + 1])
                except json.JSONDecodeError:
                    pass
        if depth == 1 and ch == ",":
            last_good_pos = i

    # last resort: truncate at last comma at depth-1 and close the object
    truncated = candidate[: last_good_pos] + "\n}"
    return json.loads(truncated)


def _format_summaries(summaries: list[SourceTableSummary]) -> str:
    if not summaries:
        return "(no structured tables found in evidence)"
    lines = []
    for s in summaries:
        sample_str = "; ".join(" | ".join(r) for r in s.sample_rows[:3])
        # include ALL row labels so the planner can enumerate complete candidate_rows
        labels_str = ", ".join(s.all_row_labels) if s.all_row_labels else "(unknown)"
        lines.append(
            f"[{s.table_id} / {s.evidence_id}] title={s.title or 'untitled'} "
            f"grain={s.guessed_row_grain} headers={s.headers} "
            f"rows={s.row_count} numeric_cols={s.numeric_column_count}\n"
            f"  all_row_labels: {labels_str}\n"
            f"  sample: {sample_str}"
        )
    return "\n".join(lines)


def _parse_plan(data: dict) -> DataTablePlan:
    columns = []
    for col in data.get("columns", []):
        try:
            columns.append(PlannedColumn.model_validate(col))
        except Exception:
            pass

    evidence_decisions = []
    for dec in data.get("evidence_decisions", []):
        try:
            evidence_decisions.append(EvidenceDecision.model_validate(dec))
        except Exception:
            pass

    excluded_source_tables = []
    for exc in data.get("excluded_source_tables", []):
        try:
            excluded_source_tables.append(ExcludedSourceTable.model_validate(exc))
        except Exception:
            pass

    excluded_candidate_rows = []
    for exc in data.get("excluded_candidate_rows", []):
        try:
            excluded_candidate_rows.append(ExcludedCandidateRow.model_validate(exc))
        except Exception:
            pass

    policy = data.get("generation_policy", "coherent_synthesis")
    if policy not in ("single_source_table_reconstruction", "coherent_synthesis", "system_summary_with_metrics"):
        policy = "coherent_synthesis"

    table_format = data.get("table_format", "wide")
    if table_format not in ("wide", "long"):
        table_format = "wide"

    purpose_type = data.get("table_purpose_type", "result_summary")
    if purpose_type not in ("result_summary", "raw_metric_extraction", "source_table_reconstruction", "system_comparison"):
        purpose_type = "result_summary"

    # For result_summary: enforce standard columns if LLM returned generic/empty columns.
    _generic_names = {"entity", "description", "column", "attribute"}
    if purpose_type == "result_summary" and (
        not columns
        or all(c.name.lower() in _generic_names for c in columns)
        or len(columns) < 3
    ):
        columns = list(_RESULT_SUMMARY_DEFAULT_COLUMNS)

    return DataTablePlan(
        table_title=str(data.get("table_title", "Data Table")).strip() or "Data Table",
        table_purpose=str(data.get("table_purpose", "")).strip(),
        table_purpose_type=purpose_type,  # type: ignore[arg-type]
        row_grain=str(data.get("row_grain", "unknown")).strip() or "unknown",
        table_format=table_format,  # type: ignore[arg-type]
        columns=columns,
        evidence_decisions=evidence_decisions,
        excluded_source_tables=excluded_source_tables,
        candidate_rows=data.get("candidate_rows", []),
        excluded_candidate_rows=excluded_candidate_rows,
        generation_policy=policy,  # type: ignore[arg-type]
        warnings=data.get("warnings", []),
        reason=str(data.get("reason", "")),
    )


async def plan_data_table(
    hint: str,
    evidence_store: list[EvidenceBlock],
    source_table_summaries: list[SourceTableSummary],
    settings: Settings,
    result_summary_plan: ResultSummaryPlan | None = None,
    debug_trace: list | None = None,
) -> DataTablePlan:
    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
    )

    # Build result_summary_plan guidance block if available.
    plan_guidance = ""
    if result_summary_plan is not None:
        from app.services.data_table.result_summary_agent import RESULT_SUMMARY_COLUMNS  # noqa: PLC0415
        rsp = result_summary_plan
        column_line = ""
        if rsp.columns != RESULT_SUMMARY_COLUMNS:
            column_line = f"- suggested_columns: {rsp.columns}\n"
        plan_guidance = (
            "\n\nResultSummaryAgent guidance (use to anchor your decisions):\n"
            f"- must_include rows (confirmed methods): {rsp.must_include}\n"
            f"- exclude_as_rows (datasets/benchmarks, NOT row subjects): {rsp.exclude_as_rows}\n"
            f"- baseline_labels: {rsp.baseline_labels}\n"
            f"- table_classifications: {rsp.table_classifications}\n"
            f"- row_groupings: {rsp.row_groupings}\n"
            f"{column_line}"
            "Use must_include to populate candidate_rows. Add exclude_as_rows to excluded_candidate_rows.\n"
        )

    # keep the user message short: summaries only, no long evidence text
    user_msg = (
        f"User hint: {hint}\n\n"
        f"Source table summaries:\n{_format_summaries(source_table_summaries)}"
        f"{plan_guidance}\n\n"
        "Return compact JSON only. No markdown fences."
    )

    raw = ""
    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": TABLE_PLANNER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_completion_tokens=2048,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = _extract_json(raw)
        plan = _parse_plan(data)
    except Exception as exc:
        logger.warning("plan_data_table LLM call failed: %s — using fallback plan", exc)
        if debug_trace is not None:
            debug_trace.append({
                "stage": "table_planning_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "raw_output_preview": raw[:300] if raw else "(no output)",
            })
        return _FALLBACK_PLAN

    # Apply ResultSummaryPlan overrides: enforce must_include, columns, exclude_as_rows.
    if result_summary_plan is not None:
        from app.services.data_table.result_summary_agent import RESULT_SUMMARY_COLUMNS as _RSC  # noqa: PLC0415
        rsp = result_summary_plan
        if rsp.must_include:
            # Agent read the papers — its must_include is authoritative
            plan.candidate_rows = list(rsp.must_include)

        # If the agent suggested custom columns (e.g. architecture-focused), use them.
        if rsp.columns and rsp.columns != _RSC:
            plan.columns = [
                PlannedColumn(name=col, description=col, value_type="string", evidence_policy="mixed")
                for col in rsp.columns
            ]

        # Remove dataset/benchmark labels from candidate rows
        excl_lower = {e.lower() for e in rsp.exclude_as_rows}
        plan.candidate_rows = [r for r in plan.candidate_rows if r.lower() not in excl_lower]

        # Add exclude_as_rows to excluded_candidate_rows
        existing_excluded_lower = {e.row_label.lower() for e in plan.excluded_candidate_rows}
        for label in rsp.exclude_as_rows:
            if label.lower() not in existing_excluded_lower:
                plan.excluded_candidate_rows.append(
                    ExcludedCandidateRow(row_label=label, reason="dataset/benchmark — not a method/system row")
                )

    # Fix 3: normalize evidence_ids — LLM may write "tbl_1" instead of the actual evidence_id.
    ev_id_set = {b.evidence_id for b in evidence_store}
    tbl_to_ev = {s.table_id: s.evidence_id for s in source_table_summaries}

    def _resolve(eid: str) -> str:
        return eid if eid in ev_id_set else tbl_to_ev.get(eid, eid)

    plan.evidence_decisions = [
        EvidenceDecision(evidence_id=_resolve(d.evidence_id), decision=d.decision, reason=d.reason)
        for d in plan.evidence_decisions
    ]
    plan.excluded_source_tables = [
        ExcludedSourceTable(table_id=e.table_id, evidence_id=_resolve(e.evidence_id), reason=e.reason)
        for e in plan.excluded_source_tables
    ]

    return plan
