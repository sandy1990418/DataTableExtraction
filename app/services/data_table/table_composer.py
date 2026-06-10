"""LLM-based table composer: plan + evidence → DraftDataTable with citations."""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.config import Settings
from app.models.data_table import EvidenceBlock
from app.prompts.data_table import (
    CONCEPT_SUMMARY_SYSTEM,
    DISCOVER_COLUMNS_SYSTEM,
    TABLE_COMPOSER_SYSTEM,
    TABLE_SINGLE_CALL_SYSTEM,
)
from app.services.data_table.source_table_rows import (
    extract_source_table_candidates,
    parse_markdown_table,
)
from app.services.data_table.source_table_summary import SourceTableSummary
from app.services.data_table.table_planner import RESULT_SUMMARY_HEADERS, DataTablePlan

if TYPE_CHECKING:
    from app.services.data_table.result_summary_agent import ResultSummaryPlan

logger = logging.getLogger(__name__)


class DraftCell(BaseModel):
    value: str | int | float | bool | None = None
    status: Literal["supported", "not_reported", "conflicting", "inferred"] = "not_reported"
    evidence_id: str | None = None
    quote: str | None = None


class DraftRow(BaseModel):
    row_label: str
    cells: dict[str, DraftCell]


class DraftDataTable(BaseModel):
    headers: list[str]
    rows: list[DraftRow]
    notes: list[str] = []


def _format_plan(plan: DataTablePlan) -> str:
    col_lines = "\n".join(
        f"  - {c.name} ({c.value_type}, policy={c.evidence_policy}): {c.description}"
        for c in plan.columns
    )
    used_ev = [d.evidence_id for d in plan.evidence_decisions if d.decision in ("use", "maybe")]
    return (
        f"Table title: {plan.table_title}\n"
        f"Purpose: {plan.table_purpose}\n"
        f"Row grain: {plan.row_grain}\n"
        f"Generation policy: {plan.generation_policy}\n"
        f"Columns:\n{col_lines}\n"
        f"Evidence to use: {used_ev or 'all available'}"
    )


def _allowed_evidence_ids(plan: DataTablePlan, evidence_store: list[EvidenceBlock]) -> set[str]:
    decided = {d.evidence_id for d in plan.evidence_decisions if d.decision == "exclude"}
    return {b.evidence_id for b in evidence_store if b.evidence_id not in decided}


def _format_evidence(evidence_store: list[EvidenceBlock], allowed_ids: set[str]) -> str:
    lines = []
    for b in evidence_store:
        if b.evidence_id not in allowed_ids:
            continue
        text = (b.text or "")[:500]
        table_section = f"\nTable:\n{b.table_markdown[:600]}" if b.table_markdown else ""
        lines.append(f"[evidence_id={b.evidence_id}]\n{text}{table_section}")
    return "\n\n---\n\n".join(lines) or "(no evidence)"


def _format_summaries(summaries: list[SourceTableSummary], allowed_ids: set[str]) -> str:
    lines = []
    for s in summaries:
        if s.evidence_id not in allowed_ids:
            continue
        sample_str = "\n".join("  | " + " | ".join(r) for r in s.sample_rows)
        lines.append(
            f"[{s.evidence_id}] {s.title or 'untitled'} (grain={s.guessed_row_grain})\n"
            f"  Headers: {s.headers}\n{sample_str}"
        )
    return "\n".join(lines) or "(none)"


def _normalize_notes(raw_notes) -> list[str]:
    """Normalize notes to list[str], handling str, list[str], and list[dict]."""
    if raw_notes is None:
        return []
    if isinstance(raw_notes, str):
        return [raw_notes] if raw_notes else []
    if not isinstance(raw_notes, list):
        return [str(raw_notes)]
    result = []
    for item in raw_notes:
        if isinstance(item, str):
            result.append(item)
        elif isinstance(item, dict):
            text = item.get("value") or item.get("text") or item.get("note") or ""
            result.append(str(text) if text else json.dumps(item))
        else:
            result.append(str(item))
    return result


def _parse_draft(data: dict, plan: DataTablePlan) -> DraftDataTable:
    headers = data.get("headers", [c.name for c in plan.columns])
    raw_rows = data.get("rows", [])
    rows: list[DraftRow] = []

    for raw_row in raw_rows:
        label = str(raw_row.get("row_label", "")).strip()
        if not label:
            continue
        raw_cells = raw_row.get("cells", {})
        cells: dict[str, DraftCell] = {}
        for header in headers:
            raw_cell = raw_cells.get(header, {})
            if isinstance(raw_cell, dict):
                status = raw_cell.get("status", "not_reported")
                if status not in ("supported", "not_reported", "conflicting", "inferred"):
                    status = "not_reported"
                cells[header] = DraftCell(
                    value=raw_cell.get("value"),
                    status=status,  # type: ignore[arg-type]
                    evidence_id=raw_cell.get("evidence_id") or None,
                    quote=raw_cell.get("quote") or None,
                )
            else:
                cells[header] = DraftCell(value=None, status="not_reported")
        rows.append(DraftRow(row_label=label, cells=cells))

    return DraftDataTable(
        headers=headers,
        rows=rows,
        notes=_normalize_notes(data.get("notes")),
    )


def build_draft_from_source_table(
    evidence_store: list[EvidenceBlock],
    hint: str,
) -> DraftDataTable | None:
    """Fallback: build a DraftDataTable directly from the best matching source table.

    Used when the planner fails (returns the generic fallback plan) so we still
    produce useful output without an LLM composer call.
    """
    candidates = extract_source_table_candidates(evidence_store, hint)
    if not candidates:
        return None
    best = candidates[0]
    if best.score < 5.0 or not best.headers or not best.rows:
        return None

    headers = best.headers
    rows: list[DraftRow] = []
    for src_row in best.rows:
        label = src_row[0].strip() if src_row else ""
        if not label:
            continue
        cells: dict[str, DraftCell] = {}
        for i, header in enumerate(headers):
            if i == 0:
                cells[header] = DraftCell(value=label, status="supported",
                                          evidence_id=best.evidence_id,
                                          quote=label)
                continue
            value_str = src_row[i].strip() if i < len(src_row) else ""
            if not value_str:
                cells[header] = DraftCell(value=None, status="not_reported")
                continue
            value: str | float = value_str
            try:
                value = float(value_str.replace(",", ""))
            except ValueError:
                pass
            row_quote = " | ".join(f"{h}={v}" for h, v in zip(headers, src_row) if v.strip())
            cells[header] = DraftCell(
                value=value,
                status="supported",
                evidence_id=best.evidence_id,
                quote=row_quote,
            )
        rows.append(DraftRow(row_label=label, cells=cells))

    return DraftDataTable(
        headers=headers,
        rows=rows,
        notes=["Fallback source table reconstruction because planner failed."],
    )


_ENTITY_HEADERS = re.compile(
    r"^(method|model|model[\s_/]?method|system|approach|algorithm|baseline|name)$",
    re.IGNORECASE,
)
_METRIC_HEADERS = re.compile(
    r"f1|bleu|accuracy|recall|precision|score|em|overall|average|ranking|rouge|meteor|sbert",
    re.IGNORECASE,
)
_SETTING_HEADERS = re.compile(
    r"^(setting|model|config|configuration|backbone|base[\s_]?model|llm)$",
    re.IGNORECASE,
)
_CATEGORY_HEADERS = re.compile(
    r"^(category|type|group|domain|task|subtask|benchmark|dataset)$",
    re.IGNORECASE,
)
_NUMERIC_CELL_RE = re.compile(r"^\s*-?\d")


def _row_quote(headers: list[str], row: list[str]) -> str:
    return " | ".join(f"{h}={v}" for h, v in zip(headers, row) if v.strip())


def _find_col_idx(headers: list[str], pattern: re.Pattern) -> int | None:
    for i, h in enumerate(headers):
        if pattern.match(h.strip()):
            return i
    return None


def _find_metric_col_indices(headers: list[str], rows: list[list[str]]) -> list[int]:
    """Return indices of columns that are metric/numeric columns (not entity/setting/category)."""
    indices = []
    for i, h in enumerate(headers):
        h_stripped = h.strip()
        if _ENTITY_HEADERS.match(h_stripped):
            continue
        if _SETTING_HEADERS.match(h_stripped):
            continue
        # include if header looks like a metric OR if sample values are numeric
        if _METRIC_HEADERS.search(h_stripped):
            indices.append(i)
            continue
        sample = [row[i] for row in rows[:5] if i < len(row) and row[i].strip()]
        if sample and sum(1 for v in sample if _NUMERIC_CELL_RE.match(v)) >= len(sample) * 0.5:
            indices.append(i)
    return indices


def _long_form_headers(plan: DataTablePlan) -> list[str]:
    """Return the long-form column headers, using plan columns when they match, else defaults."""
    defaults = ["Method / System", "Benchmark / Task", "Metric Name", "Metric Value",
                 "Setting / Model", "Notes"]
    if len(plan.columns) >= 4:
        return [c.name for c in plan.columns]
    return defaults


def compose_long_form_from_source_tables(
    plan: DataTablePlan,
    evidence_store: list[EvidenceBlock],
    source_table_summaries: list[SourceTableSummary],  # noqa: ARG001
) -> DraftDataTable | None:
    """Deterministic long-form composer: one output row per (source-row × metric-column).

    Used when plan.table_format == "long" and source tables are available.
    Never calls LLM — no JSON truncation possible.
    """
    # collect allowed evidence ids from plan
    excluded_ids = {d.evidence_id for d in plan.evidence_decisions if d.decision == "exclude"}
    # prefer explicitly used tables; fall back to any non-excluded table with markdown
    used_ev_ids = {d.evidence_id for d in plan.evidence_decisions if d.decision in ("use", "maybe")}
    if not used_ev_ids:
        used_ev_ids = {b.evidence_id for b in evidence_store if b.table_markdown and b.evidence_id not in excluded_ids}

    out_headers = _long_form_headers(plan)
    # resolve output column positions by name (case-insensitive)
    col_map: dict[str, int] = {h.lower(): i for i, h in enumerate(out_headers)}

    def _col_pos(candidates: list[str]) -> int | None:
        for c in candidates:
            pos = col_map.get(c.lower())
            if pos is not None:
                return pos
        return None

    pos_method = _col_pos(["method / system", "method", "system", "approach", out_headers[0]])
    pos_bench = _col_pos(["benchmark / task", "benchmark", "task", "dataset"])
    pos_metric_name = _col_pos(["metric name", "metric"])
    pos_metric_value = _col_pos(["metric value", "value", "score"])
    pos_setting = _col_pos(["setting / model", "setting", "model", "backbone"])
    pos_notes = _col_pos(["notes", "note"])

    # mandatory columns — if the plan has no matching slots we cannot proceed
    if pos_method is None or pos_metric_name is None or pos_metric_value is None:
        return None

    output_rows: list[DraftRow] = []
    used_table_ids: list[str] = []

    for block in evidence_store:
        if block.evidence_id not in used_ev_ids:
            continue
        if not block.table_markdown:
            continue

        headers, rows = parse_markdown_table(block.table_markdown)
        if not headers or not rows:
            continue

        entity_col = _find_col_idx(headers, _ENTITY_HEADERS)
        if entity_col is None:
            entity_col = 0

        setting_col = _find_col_idx(headers, _SETTING_HEADERS)
        # avoid using the entity column as the setting column
        if setting_col == entity_col:
            setting_col = None

        category_col = _find_col_idx(headers, _CATEGORY_HEADERS)

        metric_cols = _find_metric_col_indices(headers, rows)
        if not metric_cols:
            continue

        bench_name = block.title or block.document_name or ""
        used_table_ids.append(block.evidence_id)

        for src_row in rows:
            entity_val = src_row[entity_col].strip() if entity_col < len(src_row) else ""
            if not entity_val:
                continue

            setting_val = ""
            if setting_col is not None and setting_col < len(src_row):
                setting_val = src_row[setting_col].strip()

            category_val = ""
            if category_col is not None and category_col < len(src_row):
                category_val = src_row[category_col].strip()

            quote = _row_quote(headers, src_row)

            for m_idx in metric_cols:
                if m_idx >= len(src_row):
                    continue
                metric_val_str = src_row[m_idx].strip()
                if not metric_val_str:
                    continue
                metric_name = headers[m_idx].strip()

                # parse numeric value
                metric_val: str | float = metric_val_str
                try:
                    metric_val = float(metric_val_str.replace(",", ""))
                except ValueError:
                    pass

                # build the output cell dict aligned to out_headers
                cell_values: list[str | float | None] = [None] * len(out_headers)
                cell_values[pos_method] = entity_val
                if pos_bench is not None:
                    cell_values[pos_bench] = bench_name
                cell_values[pos_metric_name] = metric_name
                cell_values[pos_metric_value] = metric_val
                if pos_setting is not None:
                    cell_values[pos_setting] = setting_val or None
                if pos_notes is not None:
                    cell_values[pos_notes] = category_val or None

                cells: dict[str, DraftCell] = {}
                for i, h in enumerate(out_headers):
                    v = cell_values[i]
                    if v is None or v == "":
                        cells[h] = DraftCell(value=None, status="not_reported")
                    else:
                        cells[h] = DraftCell(
                            value=v,
                            status="supported",
                            evidence_id=block.evidence_id,
                            quote=quote,
                        )

                row_label = f"{entity_val} — {metric_name}"
                output_rows.append(DraftRow(row_label=row_label, cells=cells))

    if not output_rows:
        return None

    return DraftDataTable(
        headers=out_headers,
        rows=output_rows,
        notes=[f"Long-form deterministic expansion from source tables: {used_table_ids}"],
    )


# LLM composer hard limits
_LLM_MAX_ROWS = 12
_LLM_MAX_CELLS = 60


def _format_source_tables_for_summary(
    evidence_store: list[EvidenceBlock],
    allowed_ids: set[str],
    max_tables: int = 12,
) -> str:
    """Format full source table markdown for the result summary LLM call.

    Tables are fed near-complete (the model needs every row to extract metrics);
    with a 128k output budget and only a handful of papers there is ample room.
    """
    lines = []
    count = 0
    for b in evidence_store:
        if b.evidence_id not in allowed_ids:
            continue
        if not b.table_markdown:
            continue
        if count >= max_tables:
            break
        lines.append(
            f"[evidence_id={b.evidence_id}] {b.title or b.document_name or 'untitled'}\n"
            f"{b.table_markdown[:6000]}"
        )
        count += 1
    return "\n\n---\n\n".join(lines) or "(no source tables)"


# Sections worth feeding in full so qualitative cells can be richly synthesized.
_RICH_SECTION_RE = re.compile(
    r"architect|memory|retriev|method|approach|framework|mechanism|"
    r"updat|evolv|storage|design|model|innovation|contribut|"
    r"result|experiment|evaluat|benchmark|performance|abstract|introduction",
    re.IGNORECASE,
)


def _is_rich_block(b: EvidenceBlock) -> bool:
    title = b.title or ""
    text_head = (b.text or "")[:160]
    return bool(_RICH_SECTION_RE.search(title) or _RICH_SECTION_RE.search(text_head))


def _format_text_for_summary(
    evidence_store: list[EvidenceBlock],
    allowed_ids: set[str],
    max_items: int = 40,
    char_limit: int = 1600,
) -> str:
    """Feed rich method/architecture/results text so cells can be detailed.

    Blocks whose title/head matches architecture/method/result sections are
    prioritized and fed at a generous char limit; the model only writes dense
    NotebookLM-style cells when it actually has the source material to do so.
    """
    candidates = [
        b for b in evidence_store
        if b.evidence_id in allowed_ids and not b.table_markdown and b.text
    ]
    # Rich (architecture/method/result) sections first, then the rest.
    rich = [b for b in candidates if _is_rich_block(b)]
    other = [b for b in candidates if not _is_rich_block(b)]
    ordered = rich + other

    lines = []
    for b in ordered[:max_items]:
        preview = (b.text or "")[:char_limit].replace("\n", " ")
        lines.append(f"[{b.evidence_id}] {b.title or ''}: {preview}")
    return "\n".join(lines) or "(no text evidence)"


async def _discover_unified_columns(
    hint: str,
    evidence_store: list[EvidenceBlock],
    source_table_summaries: list[SourceTableSummary],
    client: AsyncOpenAI,
    model: str,
) -> list[str]:
    """Phase 1: one LLM call across all papers to decide a unified column schema.

    Looks at table headers + sample rows + text previews from all papers,
    then returns a normalized column list that can be filled across most papers.
    """
    # Build compact cross-paper summary
    lines: list[str] = []
    ev_text: dict[str, str] = {}
    for b in evidence_store:
        if b.text and b.evidence_id not in ev_text:
            ev_text[b.evidence_id] = (b.text or "")[:200].replace("\n", " ")

    for s in source_table_summaries:
        sample = "; ".join(" | ".join(r) for r in s.sample_rows[:3])
        text_preview = ev_text.get(s.evidence_id, "")
        lines.append(
            f"[{s.evidence_id}] {s.title or s.table_id} — headers: {s.headers} | sample: {sample}"
            + (f" | text: {text_preview}" if text_preview else "")
        )

    # Also include text-only blocks (no table) for context
    for b in evidence_store:
        if not b.table_markdown and b.text:
            lines.append(f"[{b.evidence_id}] {b.document_name or ''} {b.title or ''}: {(b.text or '')[:200].replace(chr(10), ' ')}")

    user_msg = (
        f"User hint: {hint}\n\n"
        f"Papers available:\n" + "\n".join(lines) +
        "\n\nDesign a unified column schema for these papers. Return JSON: {\"columns\": [...]}"
    )

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DISCOVER_COLUMNS_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_completion_tokens=256,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        cols = data.get("columns", [])
        if isinstance(cols, list) and cols:
            return [str(c) for c in cols if c]
    except Exception as exc:
        logger.warning("_discover_unified_columns failed: %s", exc)

    return RESULT_SUMMARY_HEADERS


def _rows_from_fixed_columns(raw_rows: list, fixed_columns: list[str]) -> list[DraftRow]:
    """Align raw LLM rows to a fixed column list (single-call composer output)."""
    rows: list[DraftRow] = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue
        label = str(raw_row.get("row_label", "")).strip()
        if not label:
            continue
        raw_cells = raw_row.get("cells", {}) or {}
        cells: dict[str, DraftCell] = {}
        for col in fixed_columns:
            raw_cell = raw_cells.get(col, {})
            if isinstance(raw_cell, dict):
                status = raw_cell.get("status", "not_reported")
                if status not in ("supported", "not_reported", "conflicting", "inferred"):
                    status = "not_reported"
                cells[col] = DraftCell(
                    value=raw_cell.get("value"),
                    status=status,  # type: ignore[arg-type]
                    evidence_id=raw_cell.get("evidence_id") or None,
                    quote=raw_cell.get("quote") or None,
                )
            else:
                cells[col] = DraftCell(value=None, status="not_reported")
        rows.append(DraftRow(row_label=label, cells=cells))
    return rows


async def compose_result_summary(
    hint: str,
    plan: DataTablePlan,
    evidence_store: list[EvidenceBlock],
    source_table_summaries: list[SourceTableSummary],
    settings: Settings,
    result_summary_plan: ResultSummaryPlan | None = None,
    repair_errors: list[str] | None = None,
) -> DraftDataTable:
    """NotebookLM-style single long-context composer for result_summary mode.

    Phase 1: discover (or take from plan) a unified column schema.
    Phase 2: ONE LLM call over ALL source tables + text evidence that emits the
    full table at once. No per-source decomposition, no dedup/merge step.
    """
    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
    )

    allowed_ids = _allowed_evidence_ids(plan, evidence_store)

    # Phase 1 — column schema (anchors the single call)
    _ARCH_COL_MARKERS = {"Memory Architecture", "Memory Update / Retrieval", "Key Innovation"}
    plan_cols = [c.name for c in plan.columns] if plan.columns else []
    if _ARCH_COL_MARKERS & set(plan_cols):
        # Architecture mode: use plan columns directly (no discovery call)
        fixed_columns = plan_cols
    else:
        # Benchmark/result mode: discover unified schema from all papers
        fixed_columns = await _discover_unified_columns(
            hint, evidence_store, source_table_summaries, client, settings.OPENAI_MODEL
        )

    # Build the single-call user message
    source_tables_str = _format_source_tables_for_summary(evidence_store, allowed_ids, max_tables=8)
    text_str = _format_text_for_summary(evidence_store, allowed_ids, max_items=8)

    rsp_guidance = ""
    if result_summary_plan is not None:
        rsp = result_summary_plan
        must = ", ".join(rsp.must_include) if rsp.must_include else "(none)"
        excl = ", ".join(rsp.exclude_as_rows) if rsp.exclude_as_rows else "(none)"
        row_groupings = getattr(rsp, "row_groupings", None)
        groupings = (
            "; ".join("/".join(g) for g in row_groupings) if row_groupings else "(none)"
        )
        rsp_guidance = (
            "\n\nPlanning guidance:\n"
            f"- must_include (MUST be rows): {must}\n"
            f"- exclude_as_rows (NEVER rows): {excl}\n"
            f"- row_groupings (merge variants): {groupings}\n"
            f"- row_grain: {rsp.row_grain}\n"
        )

    repair_section = ""
    if repair_errors:
        repair_section = (
            "\n\nPREVIOUS ATTEMPT HAD ERRORS — repair these issues:\n"
            + "\n".join(f"- {e}" for e in repair_errors)
            + "\n"
        )

    user_msg = (
        f"User hint: {hint}\n\n"
        f"Columns to fill (use exactly, in order): {fixed_columns}\n\n"
        f"=== Source result tables ===\n{source_tables_str}\n\n"
        f"=== Key text evidence ===\n{text_str}"
        f"{rsp_guidance}"
        f"{repair_section}\n\n"
        f"Produce at most {_LLM_MAX_ROWS} rows. Return compact JSON only."
    )

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": TABLE_SINGLE_CALL_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_completion_tokens=128000,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
    except Exception as exc:
        logger.warning("compose_result_summary single call failed: %s", exc)
        return DraftDataTable(headers=fixed_columns, rows=[], notes=[str(exc)])

    rows = _rows_from_fixed_columns(data.get("rows", []), fixed_columns)
    if len(rows) > _LLM_MAX_ROWS:
        rows = rows[:_LLM_MAX_ROWS]

    return DraftDataTable(headers=fixed_columns, rows=rows, notes=[])


# Concept tables can be larger than benchmark summaries (one row per concept).
_CONCEPT_MAX_ROWS = 20


def _format_all_evidence_for_concepts(
    evidence_store: list[EvidenceBlock],
    allowed_ids: set[str],
    max_items: int = 120,
    char_limit: int = 2000,
) -> str:
    """Feed full text evidence (and any table markdown) for concept synthesis.

    Generous limits: rich NotebookLM-style cells require the model to actually
    see the explanations, analogies, and numbers in the source text.
    """
    lines = []
    for b in evidence_store:
        if b.evidence_id not in allowed_ids:
            continue
        text = (b.text or "")[:char_limit].replace("\n", " ")
        table_section = f"\nTable:\n{b.table_markdown[:1000]}" if b.table_markdown else ""
        if not text and not table_section:
            continue
        lines.append(f"[evidence_id={b.evidence_id}] {b.title or ''}\n{text}{table_section}")
        if len(lines) >= max_items:
            break
    return "\n\n---\n\n".join(lines) or "(no evidence)"


async def compose_concept_summary(
    hint: str,
    plan: DataTablePlan,
    evidence_store: list[EvidenceBlock],
    settings: Settings,
    repair_errors: list[str] | None = None,
) -> DraftDataTable:
    """NotebookLM-style concept-synthesis composer for explanatory sources.

    ONE long-context LLM call over all text evidence; one rich row per concept
    from plan.candidate_rows, columns taken directly from the plan.
    """
    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
    )

    allowed_ids = _allowed_evidence_ids(plan, evidence_store)
    fixed_columns = [c.name for c in plan.columns] if plan.columns else []

    evidence_str = _format_all_evidence_for_concepts(evidence_store, allowed_ids)
    concepts = ", ".join(plan.candidate_rows) if plan.candidate_rows else "(enumerate them from the evidence)"

    repair_section = ""
    if repair_errors:
        repair_section = (
            "\n\nPREVIOUS ATTEMPT HAD ERRORS — repair these issues:\n"
            + "\n".join(f"- {e}" for e in repair_errors)
            + "\n"
        )

    user_msg = (
        f"User hint: {hint}\n\n"
        f"Table title: {plan.table_title}\n"
        f"Purpose: {plan.table_purpose}\n"
        f"Columns to fill (use exactly, in order): {fixed_columns}\n"
        f"Concepts that must become rows: {concepts}\n\n"
        f"=== Evidence ===\n{evidence_str}"
        f"{repair_section}\n\n"
        f"Produce at most {_CONCEPT_MAX_ROWS} rows. Return compact JSON only."
    )

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": CONCEPT_SUMMARY_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_completion_tokens=128000,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
    except Exception as exc:
        logger.warning("compose_concept_summary call failed: %s", exc)
        return DraftDataTable(headers=fixed_columns, rows=[], notes=[str(exc)])

    rows = _rows_from_fixed_columns(data.get("rows", []), fixed_columns)
    if len(rows) > _CONCEPT_MAX_ROWS:
        rows = rows[:_CONCEPT_MAX_ROWS]

    return DraftDataTable(headers=fixed_columns, rows=rows, notes=[])


async def compose_data_table(
    hint: str,
    plan: DataTablePlan,
    evidence_store: list[EvidenceBlock],
    source_table_summaries: list[SourceTableSummary],
    settings: Settings,
    result_summary_plan: ResultSummaryPlan | None = None,
    repair_errors: list[str] | None = None,
) -> DraftDataTable:
    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
    )

    allowed_ids = _allowed_evidence_ids(plan, evidence_store)

    # apply hard limits to keep LLM output small
    max_rows = _LLM_MAX_ROWS
    max_cells = _LLM_MAX_CELLS

    repair_section = ""
    if repair_errors:
        repair_section = (
            "\n\nPREVIOUS ATTEMPT HAD ERRORS — repair these issues:\n"
            + "\n".join(f"- {e}" for e in repair_errors)
            + "\n"
        )

    rsp_guidance = ""
    if result_summary_plan is not None:
        rsp = result_summary_plan
        must = ", ".join(rsp.must_include) if rsp.must_include else "(none)"
        excl = ", ".join(rsp.exclude_as_rows) if rsp.exclude_as_rows else "(none)"
        rsp_guidance = (
            f"\n\nResultSummaryAgent guidance:\n"
            f"- PRIMARY SYSTEMS to compare (these MUST be rows): {must}\n"
            f"- DO NOT create rows for these (they are datasets/baselines): {excl}\n"
            f"- row_grain: {rsp.row_grain}\n"
        )

    user_msg = (
        f"User hint: {hint}\n\n"
        f"Table plan:\n{_format_plan(plan)}\n\n"
        f"Source table summaries (allowed):\n{_format_summaries(source_table_summaries, allowed_ids)}\n\n"
        f"Evidence:\n{_format_evidence(evidence_store, allowed_ids)}"
        f"{repair_section}"
        f"{rsp_guidance}\n\n"
        f"HARD LIMITS: max {max_rows} rows, max {max_cells} total cells.\n"
        "For each cell use short quote (≤60 chars). Return compact JSON only. No markdown fences."
    )

    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": TABLE_COMPOSER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_completion_tokens=128000,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        data = json.loads(raw)
        draft = _parse_draft(data, plan)
        # enforce hard row cap
        if len(draft.rows) > max_rows:
            draft.rows = draft.rows[:max_rows]
        return draft
    except Exception as exc:
        logger.warning("compose_data_table LLM call failed: %s", exc)
        return DraftDataTable(headers=[c.name for c in plan.columns], rows=[], notes=[str(exc)])
