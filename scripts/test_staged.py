"""End-to-end smoke test for table-to-PPTX generation.

Run the server first from this project root:
    uvicorn app.main:app --port 8000

Then run the current agent-team all-in-one path:
    python scripts/test_staged.py

Useful options:
    python scripts/test_staged.py --mode staged
    python scripts/test_staged.py --base-url http://localhost:8001
    python scripts/test_staged.py --output ~/Desktop/test_staged.pptx
    python scripts/test_staged.py --analyze-images
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_HINT = (
    "compare LLM long-term memory architectures: mechanism, storage, retrieval, "
    "benchmarks, limitations"
)
DEFAULT_DOCS = (
    PROJECT_ROOT / "data/parsed/MemoryBank.md",
    PROJECT_ROOT / "data/parsed/MemoryOS.md",
    PROJECT_ROOT / "data/parsed/AMem.md",
    PROJECT_ROOT / "data/parsed/MemGPT.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the table report smoke test.")
    parser.add_argument(
        "--mode",
        choices=("agent", "staged"),
        default="agent",
        help="agent uses /analyze; staged uses /evidence -> /outline -> /render.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL.")
    parser.add_argument("--hint", default=DEFAULT_HINT, help="Analysis/presentation hint.")
    parser.add_argument("--slides", type=int, default=7, help="Target slide count.")
    parser.add_argument(
        "--output",
        default="",
        help="Where to write the downloaded PPTX.",
    )
    parser.add_argument(
        "--analyze-images",
        action="store_true",
        help="Run vision analysis on embedded images during /evidence.",
    )
    parser.add_argument(
        "documents",
        nargs="*",
        help="Markdown files to analyze. Defaults to data/parsed/*.md sample papers.",
    )
    return parser.parse_args()


def request_json(base_url: str, method: str, path: str, payload: dict | None = None, timeout: int = 600) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed: HTTP {exc.code}\n{body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach {base_url.rstrip('/')}. Start the server first, for example:\n"
            "  uvicorn app.main:app --port 8000"
        ) from exc


def post(base_url: str, path: str, payload: dict, timeout: int = 600) -> Any:
    return request_json(base_url, "POST", path, payload, timeout)


def get(base_url: str, path: str, timeout: int = 600) -> Any:
    return request_json(base_url, "GET", path, timeout=timeout)


def document_payload(paths: list[str]) -> list[dict[str, str]]:
    selected = [Path(path).expanduser() for path in paths] if paths else list(DEFAULT_DOCS)
    docs = []
    for path in selected:
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        if not resolved.exists():
            raise RuntimeError(f"Document not found: {resolved}")
        docs.append(
            {
                "name": resolved.name,
                "file_path": str(resolved),
                "base_dir": str(resolved.parent),
            }
        )
    return docs


def download_file(base_url: str, path: str, output: str) -> Path:
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = base_url.rstrip("/") + path
    try:
        urllib.request.urlretrieve(url, destination)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Download failed: HTTP {exc.code}\n{body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Download failed: {exc}") from exc
    return destination


def run_staged_mode(args: argparse.Namespace, base_url: str, docs: list[dict[str, str]]) -> dict:
    ev = post(
        base_url,
        "/evidence",
        {
            "documents": docs,
            "hint": args.hint,
            "analyze_images": args.analyze_images,
        },
    )
    print("\nSTAGE 1  /evidence  (plan only; no rows yet)")
    print("  session_id:", ev["session_id"])
    print("  planned tables:")
    for table in ev.get("table_catalog", []):
        print(f"    [{table['name']}] {table['title']} | cols: {table['columns']}")

    out = post(base_url, "/outline", {"session_id": ev["session_id"], "hint": args.hint, "n_slides": args.slides})
    plan = out["ppt_plan"]
    print("\nSTAGE 2  /outline ->", plan.get("presentation_title", "Presentation"))
    for slide in plan.get("slides", []):
        ref = f" -> {slide['table_ref']}" if slide.get("table_ref") else ""
        print(f"    {slide.get('slide_number')}. [{slide.get('slide_type')}] {slide.get('title')}{ref}")
    print(
        "  planned:",
        len(ev.get("table_catalog", [])),
        "| referenced:",
        sorted(set(out.get("referenced_tables", []))),
    )
    if out.get("warnings"):
        print("  warnings:", out["warnings"])

    rnd = post(base_url, "/render", {"session_id": ev["session_id"], "ppt_plan": plan})
    print("\nSTAGE 3  /render ->", rnd.get("type"), "|", rnd.get("filename"))
    print_render_summary(rnd)
    return rnd


def run_agent_mode(args: argparse.Namespace, base_url: str, docs: list[dict[str, str]]) -> dict:
    result = post(
        base_url,
        "/analyze",
        {
            "documents": docs,
            "hint": args.hint,
            "n_slides": args.slides,
            "analyze_images": args.analyze_images,
        },
    )
    print("\nAGENT TEAM  /analyze ->", result.get("type"), "|", result.get("filename"))
    print("  focused evidence count:", result.get("focused_evidence_count"))
    print("  planned tables:")
    for table in result.get("table_catalog", []):
        print(f"    [{table['name']}] {table['title']} | cols: {table['columns']}")

    plan = result.get("ppt_plan", {})
    print("\n  composed slides:")
    for slide in plan.get("slides", []):
        ref = f" -> {slide['table_ref']}" if slide.get("table_ref") else ""
        print(f"    {slide.get('slide_number')}. [{slide.get('slide_type')}] {slide.get('title')}{ref}")

    print_render_summary(result)
    print_agent_trace(result.get("agent_trace", []))
    return result


def print_render_summary(result: dict) -> None:
    print("  rendered tables:", result.get("rendered_tables", []))
    if result.get("spec_reviews"):
        print("  spec reviews:")
        for review in result["spec_reviews"]:
            print(f"    [{review.get('status')}] {review.get('name')}: {review.get('warnings', [])}")
    if result.get("qa_reviews"):
        print("  row QA reviews:")
        for review in result["qa_reviews"]:
            print(f"    [{review.get('status')}] {review.get('table_id')}: {review.get('warnings', [])}")
    if result.get("warnings"):
        print("  warnings:", result["warnings"])


def print_agent_trace(agent_trace: list[dict]) -> None:
    if not agent_trace:
        return
    print("  agent trace:")
    for event in agent_trace:
        detail = event.get("detail") or {}
        print(f"    {event.get('agent')}: {event.get('action')} {detail}")


def default_output_path(mode: str) -> str:
    filename = "test_agent_report.pptx" if mode == "agent" else "test_staged.pptx"
    return str(Path.home() / "Desktop" / filename)


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    docs = document_payload(args.documents)

    print("Health check:", get(base_url, "/health"))
    if args.mode == "agent":
        result = run_agent_mode(args, base_url, docs)
    else:
        result = run_staged_mode(args, base_url, docs)

    if result.get("type") != "download" or not result.get("url"):
        print("  no PPTX download was produced:", result)
        return 1

    saved_to = download_file(base_url, result["url"], args.output or default_output_path(args.mode))
    print("  downloaded:", saved_to)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
