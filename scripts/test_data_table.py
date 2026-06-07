#!/usr/bin/env python3
"""Smoke test for the NotebookLM-style data table pipeline.

Usage:
    python scripts/test_data_table.py [path/to/sample.md]

Outputs:
    data_table.json
    data_table.csv
    data_table_citations.csv
    data_table_debug.json
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.services.data_table.evidence_store import build_evidence_store
from app.services.data_table.exporters import (
    compute_table_metrics,
    to_citation_table,
    to_debug_json,
    to_simple_table,
    write_citations_csv,
    write_debug_json,
    write_table_csv,
)
from app.services.data_table.pipeline import generate_data_table
from app.services.document_parser import parse_markdown


SAMPLE_MD = """\
# Speech Models Comparison

## Qwen2.5-Omni

Qwen2.5-Omni is an omni-modal speech-language model developed by Alibaba.
It supports streaming speech interaction in real time.
BLEU score on CoVoST2: 45.6.

## Whisper v3

OpenAI Whisper v3 is an open-source speech recognition model.
It does not support real-time streaming inference.
BLEU score on CoVoST2: 41.2.

## SeamlessM4T

SeamlessM4T by Meta supports multilingual speech translation.
Streaming support: partial (some endpoints only).
"""


async def main():
    settings = Settings()

    if not settings.OPENAI_API_KEY:
        print("No OPENAI_API_KEY configured — skipping LLM integration test.")
        print("Using minimal offline evidence store test instead.")
        _offline_test()
        return

    md_path = sys.argv[1] if len(sys.argv) > 1 else None
    if md_path:
        content = Path(md_path).read_text()
        doc_name = Path(md_path).name
    else:
        content = SAMPLE_MD
        doc_name = "sample.md"

    print(f"Parsing document: {doc_name}")
    parsed = parse_markdown(content, doc_name=doc_name)

    from app.models.evidence import EvidenceItem
    from app.services.evidence_layer import build_evidence_layer

    evidence_items = build_evidence_layer([parsed], image_analyses=[])
    print(f"Evidence blocks: {len(evidence_items)}")

    hint = "Compare speech models by architecture, streaming support, BLEU scores, and limitations."
    print(f"Hint: {hint}\n")

    data_table = await generate_data_table(
        evidence_items=evidence_items,
        hint=hint,
        settings=settings,
        max_rows=10,
        max_columns=5,
    )

    simple = to_simple_table(data_table)
    citations = to_citation_table(data_table)
    metrics = compute_table_metrics(data_table)

    print(f"Table: {data_table.schema_.title}")
    print(f"Columns: {', '.join(h for h in simple['headers'])}")
    print(f"Rows: {metrics['row_count']}")
    print(f"Supported cell ratio: {metrics['supported_cell_ratio']:.1%}")
    print(f"Citations: {metrics['citation_count']}")
    if data_table.warnings:
        print(f"Warnings ({len(data_table.warnings)}):")
        for w in data_table.warnings:
            print(f"  - {w}")

    out_dir = Path(".")
    write_table_csv(data_table, out_dir / "data_table.csv")
    write_citations_csv(data_table, out_dir / "data_table_citations.csv")
    write_debug_json(data_table, out_dir / "data_table_debug.json")
    (out_dir / "data_table.json").write_text(
        json.dumps(to_debug_json(data_table), ensure_ascii=False, indent=2)
    )
    print("\nOutput files written:")
    print("  data_table.csv")
    print("  data_table_citations.csv")
    print("  data_table_debug.json")
    print("  data_table.json")


def _offline_test():
    from app.models.evidence import EvidenceItem
    from app.services.data_table.evidence_store import build_evidence_store

    items = [
        EvidenceItem(
            kind="text_fact",
            source_ref="sample.md:lines:1-5",
            title="Qwen2.5-Omni",
            content="Qwen2.5-Omni supports streaming and achieves BLEU 45.6.",
        ),
        EvidenceItem(
            kind="text_fact",
            source_ref="sample.md:lines:6-10",
            title="Whisper v3",
            content="Whisper v3 does not support streaming. BLEU 41.2.",
        ),
    ]
    store = build_evidence_store(items)
    print(f"Evidence blocks created: {len(store)}")
    for b in store:
        print(f"  [{b.evidence_id}] {b.title}: keywords={b.keywords[:5]}")
    print("Offline test passed.")


if __name__ == "__main__":
    asyncio.run(main())
