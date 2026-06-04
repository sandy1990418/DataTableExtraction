from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.config import Settings
from app.prompts import SYSTEM_PROMPT
from app.services.table_extraction import extract_source_tables
from app.tools.table_pptx import TOOL_SCHEMA, build_tables_pptx, evict_expired, store_pptx

logger = logging.getLogger(__name__)


Layout = Literal[
    "table_only",
    "text_above",
    "text_left",
    "table_bottom",
    "table_top",
    "table_left",
    "table_right",
]
TableKind = Literal[
    "source_table",
    "extracted_summary",
    "comparison",
    "timeline",
    "matrix",
    "qa",
    "other",
]


class TableSpec(BaseModel):
    table_id: str | None = None
    title: str = Field(default="Table")
    kind: TableKind = "extracted_summary"
    headers: list[str]
    rows: list[list[str]]
    summary: str = ""
    source_ref: str = ""
    text: str = ""
    layout: Layout = "table_only"
    table_ratio: float = 0.5

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        return value.strip() or "Table"

    @field_validator("headers")
    @classmethod
    def normalize_headers(cls, value: list[str]) -> list[str]:
        headers = [str(header).strip() for header in value]
        if not headers:
            raise ValueError("headers must not be empty")
        if any(not header for header in headers):
            raise ValueError("headers must not contain blank values")
        return headers

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("summary", "source_ref")
    @classmethod
    def normalize_optional_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("table_ratio")
    @classmethod
    def normalize_table_ratio(cls, value: float) -> float:
        return min(0.8, max(0.2, float(value)))

    @model_validator(mode="after")
    def normalize_rows_and_layout(self) -> "TableSpec":
        width = len(self.headers)
        normalized_rows: list[list[str]] = []
        for row in self.rows:
            cells = [str(cell).strip() for cell in row[:width]]
            cells.extend([""] * (width - len(cells)))
            normalized_rows.append(cells)

        self.rows = normalized_rows
        if not self.text:
            self.layout = "table_only"
        return self

    def as_pptx_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "title": self.title,
            "kind": self.kind,
            "headers": self.headers,
            "rows": self.rows,
            "summary": self.summary,
            "source_ref": self.source_ref,
            "text": self.text,
            "layout": self.layout,
            "table_ratio": self.table_ratio,
        }


def _tables_from_tool_args(args: dict) -> list[TableSpec]:
    if "tables" in args:
        return [TableSpec.model_validate(table) for table in args["tables"]]

    if "headers" not in args or "rows" not in args:
        return []

    return [
        TableSpec(
            title=args.get("title", "Table"),
            headers=args["headers"],
            rows=args["rows"],
            text=args.get("text", ""),
            layout=args.get("layout", "table_only"),
            kind=args.get("kind", "extracted_summary"),
            table_ratio=args.get("table_ratio", 0.5),
        )
    ]


def _source_tables_from_message(message: str) -> list[TableSpec]:
    return [TableSpec.model_validate(table) for table in extract_source_tables(message)]


def _table_catalog_summary(source_tables: list[TableSpec]) -> str:
    if not source_tables:
        return "No source tables were detected."

    lines = ["Detected source table catalog:"]
    for table in source_tables:
        lines.append(
            "- "
            f"{table.table_id}: {table.title} "
            f"({table.summary or f'{len(table.rows)} rows x {len(table.headers)} columns'}; "
            f"source={table.source_ref or 'input'})"
        )
    return "\n".join(lines)


def _merge_source_and_generated_tables(
    source_tables: list[TableSpec],
    generated_tables: list[TableSpec],
) -> list[TableSpec]:
    if not source_tables:
        return generated_tables
    derived_tables = [table for table in generated_tables if table.kind != "source_table"]
    return [*source_tables, *derived_tables]


async def chat(message: str, settings: Settings) -> dict:
    client = AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL or None,
    )
    source_tables = _source_tables_from_message(message)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if source_tables:
        messages.append(
            {
                "role": "system",
                "content": (
                    f"The backend already extracted {len(source_tables)} original source table(s) "
                    "from the user's content and will preserve them in the PPTX. Do not recreate "
                    "`source_table` entries. Only add derived tables when the prose contains useful "
                    "summary, comparison, timeline, matrix, or Q&A structure.\n\n"
                    f"{_table_catalog_summary(source_tables)}"
                ),
            }
        )
    messages.append({"role": "user", "content": message})

    response = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=messages,
        tools=[TOOL_SCHEMA],
        tool_choice="auto",
        max_completion_tokens=settings.MAX_TOKENS,
        temperature=settings.TEMPERATURE,
    )

    choice = response.choices[0]

    # Text-only response
    if not choice.message.tool_calls:
        if source_tables:
            args = {}
            tables = source_tables
        else:
            return {"type": "text", "message": choice.message.content or ""}
    else:
        # Tool call
        tc = choice.message.tool_calls[0]
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError as exc:
            return {"type": "error", "message": f"Tool args parse error: {exc}"}

        try:
            generated_tables = _tables_from_tool_args(args)
        except ValidationError as exc:
            return {"type": "error", "message": f"Tool args validation error: {exc}"}

        tables = _merge_source_and_generated_tables(source_tables, generated_tables)

    if not tables:
        return {"type": "text", "message": choice.message.content or ""}

    try:
        pptx_bytes = await asyncio.to_thread(
            build_tables_pptx,
            [table.as_pptx_dict() for table in tables],
        )
    except Exception as exc:
        logger.error("build_pptx failed: %s", exc, exc_info=True)
        return {"type": "error", "message": f"PPTX generation failed: {exc}"}

    evict_expired()
    presentation_title = args.get("presentation_title") or tables[0].title
    filename = f"{presentation_title}.pptx"
    token = store_pptx(pptx_bytes, filename, settings.DOWNLOAD_TTL_SECONDS)

    return {
        "type": "download",
        "url": f"/download/{token}",
        "filename": filename,
        "table_catalog": [
            {
                "table_id": table.table_id,
                "title": table.title,
                "summary": table.summary,
                "source_ref": table.source_ref,
                "row_count": len(table.rows),
                "column_count": len(table.headers),
            }
            for table in tables
            if table.kind == "source_table"
        ],
        "tables": [table.as_pptx_dict() for table in tables],
        "table": {
            "title": tables[0].title,
            "headers": tables[0].headers,
            "rows": tables[0].rows,
        },
    }
