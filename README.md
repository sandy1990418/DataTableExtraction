# DataTableExtraction

Turn one or more Markdown documents (research papers, reports, issue trackers) into
**hint-focused comparison tables** and a **PPTX deck**. The service builds a unified
*evidence layer* from text, Markdown tables, and images (via vision), then runs an
agent team that selects relevant evidence, designs table specs, reviews table quality
before row population, fills grounded rows, and renders the deck.

```
documents (markdown + images)
        │
        ▼
┌─────────────────────────────────────────────┐
│ 1. parse        document_parser.py           │  text sections · markdown tables · image refs
│ 2. vision       image_analysis.py            │  image → {type, caption, insight, table}
│ 3. evidence     evidence_layer.py            │  unify into typed EvidenceItems
└─────────────────────────────────────────────┘
        │   text facts · markdown tables · image tables ·
        │   image captions · chart insights · diagram summaries
        ▼
┌─────────────────────────────────────────────┐
│ 4. agent team   report_agent_team.py         │  ReportEvidenceAgent:
│                                              │    select hint-relevant evidence
│                                              │  TableIntentAgent:
│                                              │    plan table specs
│                                              │  TableSpecReviewAgent:
│                                              │    catch bad/vague specs before rows
│                                              │  TablePopulator + Grounding:
│                                              │    fill and verify table rows
└─────────────────────────────────────────────┘
        │   grounded tables  (metrics, architecture, efficiency, ...)
        ▼
┌─────────────────────────────────────────────┐
│ 5. compose      report_agent_team.py         │  PPT plan from produced tables
│ 6. render       tools/table_pptx.py          │  table slides + text slides → PPTX
└─────────────────────────────────────────────┘
        │
        ▼
   downloadable .pptx
```

Why the agent team? One-shot "extract tables" tends to produce generic template columns
(*Summary*, *Main Contribution*, *Accuracy*) even when the evidence contains precise
metrics such as `F1`, `BLEU-1`, categories, model names, and benchmark tables. The
agent team keeps checkpoints: select evidence by hint, plan table specs, review and
revise bad specs, then populate and verify rows. The user-facing `/analyze` endpoint is
still a single call.

---

## Project structure

```
app/
├── main.py                       FastAPI app factory; wires routers
├── config.py                     Settings (loads .env relative to project root)
├── models/                       ← ALL data definitions live here
│   ├── documents.py              dataclasses: TextSection, MarkdownTable, ImageRef, ParsedDocument
│   ├── evidence.py               dataclass: EvidenceItem (+ EvidenceKind)
│   └── schemas.py                Pydantic API request bodies (DocumentInput, *Request, ChatRequest)
├── prompts/                      ← ALL system prompts live here, one module per stage
│   ├── chat.py                   SYSTEM_PROMPT          (/chat assistant)
│   ├── image.py                  IMAGE_SYSTEM           (vision analysis)
│   ├── canonical.py              PLAN_SYSTEM, POPULATE_SYSTEM  (two-stage extraction)
│   ├── planner.py                PPT_PLANNER_SYSTEM     (staged slide outline)
│   └── qa.py                     row/spec review + revision prompts
├── routes/                       ← thin: route definitions only, no business logic
│   ├── analyze.py                /evidence · /outline · /render · /analyze
│   ├── chat.py                   /chat
│   └── download.py               /download/{token}
├── services/
│   ├── pipeline.py               orchestration shared by the analyze routes
│   ├── document_parser.py        markdown → text sections, tables, image refs (+base64)
│   ├── image_analysis.py         vision API per image (parallel)
│   ├── evidence_layer.py         unify everything into typed EvidenceItems
│   ├── canonical_extractor.py    two-stage table extraction
│   ├── report_agent_team.py      all-in-one agent team for /analyze
│   ├── table_spec_qa.py          table spec review/revision before row population
│   ├── table_agent_team.py       staged render-time table population/review
│   ├── table_qa.py               row evidence-grounding review
│   ├── table_revision.py         row revision after QA findings
│   ├── evidence_selector.py      focused evidence selection for table specs
│   ├── ppt_planner.py            staged slide-outline generation
│   ├── session_store.py          TTL cache: /evidence result keyed by session_id
│   ├── llm_service.py            /chat orchestration
│   └── table_extraction.py       in-source table detection (markdown/html/tsv) for /chat
└── tools/
    └── table_pptx.py             PPTX builder + download-token store
```

Layering: **routes** (HTTP only) → **services/pipeline** (orchestration) → **services/\*** (single-stage logic) → **models** (data) + **prompts** (LLM instructions). Dependencies point downward; `models` and `prompts` import nothing from the app.

---

## Setup

```bash
# from DataTableExtraction/
pip install -e ".[dev]"          # or: uv sync
cp .env.example .env             # then fill in credentials

uvicorn app.main:app --reload --port 8000
```

### Environment (`.env`)

| Variable               | Default        | Purpose                                              |
|------------------------|----------------|------------------------------------------------------|
| `OPENAI_API_KEY`       | —              | **Required.** API key                                |
| `OPENAI_BASE_URL`      | (OpenAI)       | Point at any OpenAI-compatible endpoint              |
| `OPENAI_MODEL`         | `gpt-4o-mini`  | Model for extraction/planning/vision (falls back to `AI_MODEL`) |
| `MAX_TOKENS`           | `4096`         | Max completion tokens for extraction                 |
| `TEMPERATURE`          | `0.3`          | Sampling temperature                                 |
| `DOWNLOAD_TTL_SECONDS` | `600`          | How long a generated PPTX stays downloadable         |
| `SESSION_TTL_SECONDS`  | `1800`         | How long an `/evidence` session is cached            |

> The model must accept `max_completion_tokens` (o-series / GPT-5-class) — the code uses
> that parameter, not the legacy `max_tokens`.

### Preprocessing PDFs → Markdown

The API consumes **Markdown**. To turn PDFs into the expected input (extract page text +
dump embedded images so they can be analyzed), preprocess them first — e.g. with PyMuPDF:

```python
import fitz  # pip install pymupdf
doc = fitz.open("paper.pdf")
# write page text to paper.md and embedded images to paper_images/,
# inserting ![alt](paper_images/...) refs into the markdown.
```

Then pass the `.md` via `file_path` (with `base_dir` pointing at the images folder).

---

## API

The primary path is **all-in-one agent mode** (`/analyze`): pass document paths plus a
hint, and the agent team decides which evidence and tables matter. The staged endpoints
(`/evidence` → `/outline` → `/render`) remain available for compatibility and for
systems that need to drive outlines separately.

### Document input (shared)

```jsonc
{
  "name": "MemGPT.md",                 // optional; defaults to filename
  "file_path": "data/parsed/MemGPT.md",// server reads it directly (relative to server CWD)
  "content": "# ...",                  // OR pass markdown inline instead of file_path
  "base_dir": "data/parsed"            // resolves ![](...) image paths; defaults to file's dir
}
```

---

### `POST /analyze` — all-in-one agent team

Runs the report agent team and returns a download URL. The caller only supplies
documents and a hint; the agent team selects evidence, plans/reviews table specs,
populates grounded rows, composes a PPT plan, and renders the deck.

```jsonc
// request
{
  "documents": [ {"file_path": "data/parsed/MemGPT.md"} ],
  "hint": "compare LLM long-term memory architectures",
  "n_slides": 7,            // optional target slide count
  "analyze_images": true    // run vision on embedded images
}

// response
{
  "type": "download",
  "url": "/download/<token>",
  "filename": "Comparing ... .pptx",
  "ppt_plan": { "presentation_title": "...", "slides": [] },
  "rendered_tables": ["architecture_comparison", "benchmark_results"],
  "table_catalog": [ {"name": "...", "title": "...", "columns": ["..."], "column_count": 6} ],
  "evidence_summary": "Evidence layer: 46 items ...",
  "focused_evidence_count": 28,
  "spec_reviews": [
    {"name": "benchmark_results", "status": "pass", "warnings": [], "unsupported_columns": []}
  ],
  "qa_reviews": [],
  "agent_trace": [
    {"agent": "ReportEvidenceAgent", "action": "selected_hint_relevant_evidence", "detail": {}},
    {"agent": "TableSpecReviewAgent", "action": "reviewed_specs", "detail": {}},
    {"agent": "TableGroundingAgent", "action": "reviewed_rows", "detail": {}}
  ],
  "warnings": []
}
```

The important quality gate is `spec_reviews`: table specs are checked before row
population. This is where vague or unsupported columns should be caught before the
system spends tokens filling a bad table.

#### Script smoke test

The smoke-test script now defaults to `/analyze` agent mode:

```bash
uvicorn app.main:app --port 8000
python scripts/test_staged.py
```

It writes the PPTX to `~/Desktop/test_agent_report.pptx` by default and prints focused
evidence count, planned tables, spec reviews, row QA reviews, and agent trace.

To run the compatibility staged flow instead:

```bash
python scripts/test_staged.py --mode staged
```

---

### Staged flow (compatibility / Presenton-style)

Run `/evidence` once, then drive outline and render separately. This path is useful when
another system owns the slide outline. State is cached server-side under a `session_id`,
so you only pass a short string between calls.

#### `POST /evidence` — plan the tables (run once, cheap)

Parses the docs, builds the evidence layer, and **plans** which tables could exist (their
columns/facets) — but does **not** fill in rows yet. That deferred work happens at
`/render`.

```jsonc
// request
{ "documents": [], "hint": "...", "analyze_images": true }

// response
{
  "session_id": "e8f2e768...",          // hand this to /outline and /render
  "evidence_summary": "...",
  "table_catalog": [                    // planned tables: columns only, no rows
    {"name": "architecture_comparison", "title": "...", "description": "...",
     "columns": ["System", "Memory paradigm", "Retrieval method", "..."],
     "column_count": 6}
  ]
}
```

#### `POST /outline` — pick which tables become slides

Maps to Presenton's outline generation. The planner sees each table's title, description,
and **column list** and selects which deserve a slide via `table_ref`.

```jsonc
// request
{ "session_id": "e8f2e768...", "hint": "...", "n_slides": 6 }

// response
{
  "ppt_plan": {
    "presentation_title": "...",
    "slides": [
      {"slide_number": 3, "slide_type": "table", "title": "...",
       "table_ref": "architecture_comparison", "content": "...", "speaker_notes": "..."},
      {"slide_number": 4, "slide_type": "key_findings", "title": "...",
       "table_ref": null}
    ]
  },
  "referenced_tables": ["architecture_comparison"]
}
```

> **Not every table goes on a slide.** A pool of 6 planned tables might yield 3
> `table_ref`s; the rest are never populated. Slides with `table_ref: null` are text-only.

#### `POST /render` — populate referenced tables + build the PPTX

This is where row population happens — **lazily**, only for the tables the plan
references, then cached on the session (a second `/render` is instant). Unreferenced
tables are never populated.

Rendering now runs a LangGraph table-agent workflow:

```
SupervisorAgent
  -> EvidenceSelectorAgent
  -> TablePopulatorAgent
  -> TableReviewerAgent
      -> TableRevisionAgent -> TableReviewerAgent  (only when review fails)
  -> deterministic PPTX renderer
```

The reviewer checks evidence grounding, sparse tables, and generic values such as
`Yes` / `High` / `Moderate`. The PPTX renderer preserves non-table slides from the
plan instead of dropping title, overview, or conclusion slides.

```jsonc
// request
{ "session_id": "e8f2e768...", "ppt_plan": {}, "presentation_title": "..." }

// response
{
  "type": "download",
  "url": "/download/<token>",
  "filename": "....pptx",
  "rendered_tables": ["architecture_comparison"],
  "qa_reviews": [
    {"table_id": "architecture_comparison", "status": "pass",
     "warnings": [], "unsupported_cells": []}
  ],
  "agent_trace": [
    {"agent": "SupervisorAgent", "action": "resolved_fixed_workflow", "detail": {}},
    {"agent": "EvidenceSelectorAgent", "action": "selected_focused_evidence", "detail": {}},
    {"agent": "TablePopulatorAgent", "action": "populated_tables", "detail": {}},
    {"agent": "TableReviewerAgent", "action": "reviewed_tables", "detail": {}}
  ],
  "warnings": []
}
```

If you render slides yourself instead of calling `/render`, the equivalent is: decide
your `table_ref`s, then call the extractor's `populate_tables` for just those specs.

#### Curl smoke test

```bash
BASE=http://localhost:8000

curl -sS -X POST "$BASE/analyze" \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": [
      {"file_path": "data/parsed/MemoryBank.md"},
      {"file_path": "data/parsed/MemoryOS.md"},
      {"file_path": "data/parsed/AMem.md"},
      {"file_path": "data/parsed/MemGPT.md"}
    ],
    "hint": "compare LLM long-term memory architectures: mechanism, storage, retrieval, benchmarks, limitations",
    "n_slides": 7,
    "analyze_images": false
  }'
```

Use the returned `url` to download:

```bash
curl -L -o /tmp/test_agent_report.pptx "$BASE/download/<TOKEN>"
```

For the compatibility staged flow, use `/evidence`, `/outline`, and `/render`:

```bash
curl -sS -X POST "$BASE/evidence" \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": [
      {"file_path": "data/parsed/MemoryBank.md"},
      {"file_path": "data/parsed/MemoryOS.md"},
      {"file_path": "data/parsed/AMem.md"},
      {"file_path": "data/parsed/MemGPT.md"}
    ],
    "hint": "compare LLM long-term memory architectures: mechanism, storage, retrieval, benchmarks, limitations",
    "analyze_images": false
  }'
```

Use the returned `session_id`:

```bash
curl -sS -X POST "$BASE/outline" \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "<SESSION_ID>", "n_slides": 7}'
```

Use the returned `ppt_plan`:

```bash
curl -sS -X POST "$BASE/render" \
  -H 'Content-Type: application/json' \
  -d '{"session_id": "<SESSION_ID>", "ppt_plan": <PPT_PLAN_JSON>, "presentation_title": "memory_architecture_comparison"}'
```

Download the generated deck:

```bash
curl -L -o /tmp/test_staged.pptx "$BASE/download/<TOKEN>"
```

---

### Other endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/chat`  | POST   | Single-shot: send text, get a table PPTX if the content warrants one. Body: `{"message": "..."}` |
| `/download/{token}` | GET | Download a generated PPTX (valid for `DOWNLOAD_TTL_SECONDS`) |
| `/health` | GET   | Liveness check |

---

## Evidence kinds

`evidence_layer.py` normalizes everything into typed `EvidenceItem`s, which feed table
extraction:

| Kind              | Source                                   |
|-------------------|------------------------------------------|
| `text_fact`       | a prose section (by heading)             |
| `markdown_table`  | a table already in the markdown          |
| `image_table`     | a table extracted from an image (vision) |
| `image_caption`   | vision description of an image           |
| `chart_insight`   | key finding read off a chart             |
| `diagram_summary` | summary of a diagram/architecture figure |

---

## Notes & limitations

- **State is in-process** (`session_store`, pptx token store). For multi-worker
  deployments, back them with Redis or similar.
- **PDF image extraction** pulls *embedded raster assets*; vector figures (many paper
  diagrams) won't be captured and decorative icons may slip through. The vision prompt
  classifies obvious icons/logos as `other`; tighten further with a size filter in the
  preprocessing step if needed.
- The model must support `max_completion_tokens` and (for `analyze_images`) vision.
