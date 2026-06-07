#!/usr/bin/env python
"""Quick smoke test for data table pipeline using AMem/MemGPT/MemoryBank/MemoryOS papers."""
import asyncio
import json
import sys
from pathlib import Path

# allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import Settings
from app.services.data_table.pipeline import generate_data_table
from app.services.document_parser import parse_markdown
from app.services.evidence_layer import build_evidence_layer

DATA_DIR = Path(__file__).parent.parent / "data" / "parsed"
HINT = "Compare these memory experiment result."

async def main():
    settings = Settings()

    parsed_docs = []
    for md_file in sorted(DATA_DIR.glob("*.md")):
        print(f"Parsing {md_file.name}...")
        doc = parse_markdown(md_file.read_text(), doc_name=md_file.name, base_dir=str(DATA_DIR))
        parsed_docs.append(doc)

    evidence_items = build_evidence_layer(parsed_docs, [{"type": "other", "caption": "", "insight": "", "table": None}] * sum(len(d.images) for d in parsed_docs))

    print(f"\nTotal evidence items: {len(evidence_items)}")
    print(f"Hint: {HINT}\n")

    result = await generate_data_table(evidence_items, HINT, settings, max_rows=15, max_columns=7)

    print(f"Strategy: {next((t['strategy'] for t in result.debug_trace if t.get('stage') == 'intent'), 'unknown')}")
    print(f"Columns: {[c.name for c in result.schema_.columns]}")
    print(f"Rows: {len(result.rows)}")

    text_inject = next((t for t in result.debug_trace if t.get('stage') == 'text_table_injection'), None)
    if text_inject:
        print(f"Text tables injected: {text_inject['injected_count']}")
    else:
        print("WARNING: text_table_injection stage NOT found")

    text_extractions = [t for t in result.debug_trace if t.get('stage') == 'text_table_extraction']
    for t in text_extractions:
        print(f"  Extracted from {t['document']}: {t['row_count']} rows, score={t['score']}, headers={t['headers'][:4]}")

    print()
    print("=== TABLE ===")
    col_names = [c.name for c in result.schema_.columns]
    header = " | ".join(f"{n:20}" for n in col_names)
    print(header)
    print("-" * len(header))
    for row in result.rows:
        cells = []
        for col in result.schema_.columns:
            cell = row.cells.get(col.name)
            val = str(cell.value) if cell and cell.value is not None else "(none)"
            status = cell.status if cell else "?"
            cells.append(f"{val[:18]:18}({status[0]})")
        print(" | ".join(cells))

    all_cells = [c for row in result.rows for c in row.cells.values()]
    supported = sum(1 for c in all_cells if c.status == "supported")
    print(f"\nSupported: {supported}/{len(all_cells)} = {supported/max(len(all_cells),1):.0%}")

    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

    # write debug trace
    out = Path("dt_debug.json")
    out.write_text(json.dumps(result.debug_trace, ensure_ascii=False, indent=2))
    print(f"\nDebug trace written to {out}")

if __name__ == "__main__":
    asyncio.run(main())
