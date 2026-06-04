from __future__ import annotations

from collections import Counter

from app.models import EvidenceItem


def build_evidence_layer(
    parsed_docs: list,
    image_analyses: list[dict],
) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    img_idx = 0

    for doc in parsed_docs:
        for section in doc.sections:
            if section.content.strip():
                items.append(EvidenceItem(
                    kind="text_fact",
                    source_ref=section.source_ref,
                    title=section.heading or doc.name,
                    content=section.content,
                ))

        for table in doc.tables:
            items.append(EvidenceItem(
                kind="markdown_table",
                source_ref=table.source_ref,
                title=table.title,
                content=f"{len(table.rows)} rows x {len(table.headers)} columns",
                headers=table.headers,
                rows=table.rows,
            ))

        for image in doc.images:
            if img_idx < len(image_analyses):
                analysis = image_analyses[img_idx]
            else:
                analysis = {"type": "other", "caption": image.alt, "insight": "", "table": None}
            img_idx += 1

            img_type = analysis.get("type", "other")
            if img_type == "chart":
                items.append(EvidenceItem(
                    kind="chart_insight",
                    source_ref=image.source_ref,
                    title=analysis.get("title", image.alt),
                    content=analysis.get("insight") or analysis.get("caption", ""),
                ))
            elif img_type == "diagram":
                items.append(EvidenceItem(
                    kind="diagram_summary",
                    source_ref=image.source_ref,
                    title=analysis.get("title", image.alt),
                    content=analysis.get("caption", ""),
                ))
            else:
                caption = analysis.get("caption", image.alt)
                if caption:
                    items.append(EvidenceItem(
                        kind="image_caption",
                        source_ref=image.source_ref,
                        title=analysis.get("title", image.alt),
                        content=caption,
                    ))
                tbl = analysis.get("table")
                if tbl and tbl.get("headers") and tbl.get("rows"):
                    items.append(EvidenceItem(
                        kind="image_table",
                        source_ref=image.source_ref,
                        title=analysis.get("title", image.alt),
                        content=caption,
                        headers=tbl["headers"],
                        rows=tbl["rows"],
                    ))

    return items


def summarize_evidence(items: list[EvidenceItem]) -> str:
    lines = [f"Evidence layer: {len(items)} items"]
    counts = Counter(item.kind for item in items)
    for kind, count in sorted(counts.items()):
        lines.append(f"  {kind}: {count}")
    lines.append("")
    for item in items:
        if item.kind in ("markdown_table", "image_table"):
            lines.append(f"[{item.kind}] {item.title} — {item.content} (source: {item.source_ref})")
        else:
            preview = item.content[:120].replace("\n", " ")
            lines.append(f"[{item.kind}] {item.title}: {preview}")
    return "\n".join(lines)
