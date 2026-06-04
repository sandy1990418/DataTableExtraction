"""Centralized system prompts for every LLM call in the pipeline.

One module per stage so prompts are easy to find, diff, and tune in isolation:
- chat       → /chat single-shot table-from-text assistant
- image      → vision analysis of embedded images
- canonical  → two-stage table extraction (plan facets → populate rows)
- planner    → slide-outline generation
- qa         → table evidence-grounding review
"""

from app.prompts.canonical import PLAN_SYSTEM, POPULATE_SYSTEM
from app.prompts.chat import SYSTEM_PROMPT
from app.prompts.image import IMAGE_SYSTEM
from app.prompts.planner import PPT_PLANNER_SYSTEM
from app.prompts.qa import (
    TABLE_QA_SYSTEM,
    TABLE_REVISION_SYSTEM,
    TABLE_SPEC_QA_SYSTEM,
    TABLE_SPEC_REVISION_SYSTEM,
)

__all__ = [
    "SYSTEM_PROMPT",
    "IMAGE_SYSTEM",
    "PLAN_SYSTEM",
    "POPULATE_SYSTEM",
    "PPT_PLANNER_SYSTEM",
    "TABLE_QA_SYSTEM",
    "TABLE_REVISION_SYSTEM",
    "TABLE_SPEC_QA_SYSTEM",
    "TABLE_SPEC_REVISION_SYSTEM",
]
