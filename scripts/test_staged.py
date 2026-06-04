"""End-to-end test of the staged pipeline: /evidence -> /outline -> /render.

Run the server first:
    cd DataTableExtraction && ../.venv/bin/uvicorn app.main:app --port 8001

Then:
    ../.venv/bin/python scripts/test_staged.py
"""

import json
import urllib.request

BASE = "http://localhost:8001"


def post(path, payload, timeout=600):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# Stage 1 — build the table pool once (set analyze_images=True to run vision)
docs = [
    {"file_path": "data/parsed/MemoryBank.md"},
    {"file_path": "data/parsed/MemoryOS.md"},
    {"file_path": "data/parsed/AMem.md"},
    {"file_path": "data/parsed/MemGPT.md"},
]
ev = post("/evidence", {
    "documents": docs,
    "hint": "compare LLM long-term memory architectures: mechanism, storage, retrieval, benchmarks, limitations",
    "analyze_images": False,
})
print("STAGE 1  /evidence")
print("  session_id:", ev["session_id"])
print("  table pool:")
for t in ev["table_catalog"]:
    print(f"    [{t['name']}] {t['title']} — {t['row_count']}x{t['column_count']}")

# Stage 2 — outline: LLM picks which tables deserve a slide
out = post("/outline", {"session_id": ev["session_id"], "n_slides": 7})
plan = out["ppt_plan"]
print("\nSTAGE 2  /outline ->", plan["presentation_title"])
for s in plan["slides"]:
    ref = f"  -> {s['table_ref']}" if s.get("table_ref") else ""
    print(f"    {s['slide_number']}. [{s['slide_type']}] {s['title']}{ref}")
print("  pool:", len(ev["table_catalog"]), "tables | referenced:", sorted(set(out["referenced_tables"])))

# Stage 3 — render only the referenced tables
rnd = post("/render", {"session_id": ev["session_id"], "ppt_plan": plan})
print("\nSTAGE 3  /render ->", rnd.get("type"), "|", rnd.get("filename"))
if rnd.get("url"):
    token = rnd["url"].split("/")[-1]
    out_path = "/tmp/test_staged.pptx"
    urllib.request.urlretrieve(BASE + "/download/" + token, out_path)
    print("  downloaded:", out_path)
