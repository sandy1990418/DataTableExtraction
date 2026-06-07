"""Evidence-aware candidate strategy scoring.

Chooses one of three strategies based on hint signals AND evidence shape.
LLM is never the sole router; scoring is deterministic.
LLM may optionally generate a human-readable reason string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from app.models.data_table import EvidenceBlock
from app.services.data_table.source_table_rows import SourceTableCandidate, extract_source_table_candidates

Strategy = Literal[
    "source_table_reconstruction",
    "synthesized_entity_comparison",
    "hybrid_table_synthesis",
]

# ── hint signal term sets ──────────────────────────────────────────────────────

_RESULT_TERMS = {
    "experiment", "experiments", "result", "results", "benchmark", "benchmarks",
    "performance", "score", "scores", "metric", "metrics", "evaluation",
    "accuracy", "f1", "bleu", "rouge", "meteor", "sbert", "leaderboard",
    "multi_hop", "multihop", "temporal", "open_domain", "opendomain",
    "single_hop", "singlehop", "overall", "overall_score",
}

_ARCHITECTURE_TERMS = {
    "architecture", "design", "retrieval", "mechanism", "limitation", "limitations",
    "feature", "features", "approach", "methods", "system", "systems",
    "compare", "comparison", "how", "overview", "description", "components",
    "structure", "workflow", "pipeline",
}


def _hint_tokens(hint: str) -> frozenset[str]:
    raw = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]*", hint.lower()))
    normalized = {t.replace("-", "_") for t in raw}
    return frozenset(raw | normalized)


@dataclass
class EvidenceSignals:
    has_source_table: bool = False
    best_table_score: float = 0.0
    best_table_candidate: SourceTableCandidate | None = None
    entity_col_present: bool = False
    metric_col_count: int = 0
    text_block_count: int = 0


@dataclass
class StrategyScore:
    strategy: Strategy
    score: float
    signals: dict = field(default_factory=dict)
    reason: str = ""


_ENTITY_COL_NAMES = {"method", "model", "system", "approach", "baseline", "name", "algorithm"}
_METRIC_HEADER_TERMS = {
    "score", "rouge", "bleu", "f1", "accuracy", "meteor", "sbert", "rate", "ratio",
    "result", "performance", "hop", "temporal", "domain", "overall", "avg", "average",
}


def _is_metric_header(header: str) -> bool:
    norm = re.sub(r"[\s\-_]+", "", header.lower())
    return any(t in norm for t in _METRIC_HEADER_TERMS)


def _analyze_evidence(evidence_store: list[EvidenceBlock], hint: str) -> EvidenceSignals:
    candidates = extract_source_table_candidates(evidence_store, hint)
    text_blocks = sum(1 for b in evidence_store if b.text and len(b.text) > 100)

    if not candidates:
        return EvidenceSignals(text_block_count=text_blocks)

    best = candidates[0]
    headers = best.headers

    entity_col = any(
        re.sub(r"[\s\-_]+", "", h.lower()) in _ENTITY_COL_NAMES
        for h in headers
    )
    metric_cols = sum(1 for h in headers[1:] if _is_metric_header(h))

    return EvidenceSignals(
        has_source_table=True,
        best_table_score=best.score,
        best_table_candidate=best,
        entity_col_present=entity_col,
        metric_col_count=metric_cols,
        text_block_count=text_blocks,
    )


def _score_source_table_reconstruction(
    hint_tokens: frozenset[str],
    ev: EvidenceSignals,
) -> tuple[float, dict]:
    signals: dict = {}
    score = 0.0

    # evidence requirements — must have a good source table
    if not ev.has_source_table:
        return -100.0, {"disqualified": "no source table in evidence"}

    if ev.best_table_score < 5.0:
        score -= 3.0
        signals["low_table_score"] = ev.best_table_score
    else:
        score += 5.0
        signals["good_table_score"] = ev.best_table_score

    if ev.entity_col_present:
        score += 2.0
        signals["entity_col"] = True

    if ev.metric_col_count >= 2:
        score += 2.0
        signals["metric_col_count"] = ev.metric_col_count
    elif ev.metric_col_count == 1:
        score += 0.5

    # hint signals
    result_hits = hint_tokens & _RESULT_TERMS
    arch_hits = hint_tokens & _ARCHITECTURE_TERMS

    if result_hits:
        score += 3.0
        signals["result_hint_terms"] = sorted(result_hits)[:5]

    if arch_hits:
        score -= 4.0
        signals["architecture_hint_penalty"] = sorted(arch_hits)[:5]

    return score, signals


def _score_synthesized_entity_comparison(
    hint_tokens: frozenset[str],
    ev: EvidenceSignals,
) -> tuple[float, dict]:
    signals: dict = {}
    score = 0.0

    arch_hits = hint_tokens & _ARCHITECTURE_TERMS
    result_hits = hint_tokens & _RESULT_TERMS

    if arch_hits:
        score += 4.0
        signals["architecture_hint_terms"] = sorted(arch_hits)[:5]

    if not result_hits:
        score += 2.0
        signals["no_result_terms"] = True
    elif result_hits and ev.has_source_table:
        score -= 3.0
        signals["result_terms_with_table_penalty"] = True

    if not ev.has_source_table:
        score += 3.0
        signals["no_source_table_bonus"] = True

    if ev.text_block_count >= 3:
        score += 1.0
        signals["rich_text"] = ev.text_block_count

    return score, signals


def _score_hybrid_table_synthesis(
    hint_tokens: frozenset[str],
    ev: EvidenceSignals,
) -> tuple[float, dict]:
    signals: dict = {}
    score = 0.0

    arch_hits = hint_tokens & _ARCHITECTURE_TERMS
    result_hits = hint_tokens & _RESULT_TERMS

    if arch_hits and result_hits:
        score += 4.0
        signals["both_arch_and_result"] = True

    if ev.has_source_table and arch_hits:
        score += 4.0
        signals["table_plus_arch_hint"] = True

    if ev.text_block_count >= 3 and ev.has_source_table:
        score += 2.0
        signals["rich_evidence_mix"] = True

    # long/detailed hint suggests synthesis
    if len(hint_tokens) >= 8:
        score += 2.0
        signals["detailed_hint"] = len(hint_tokens)

    return score, signals


def select_strategy(
    hint: str,
    evidence_store: list[EvidenceBlock],
    debug_trace: list | None = None,
) -> tuple[Strategy, dict]:
    """Score all three strategies and return the winner.

    Returns (strategy_name, full_score_dict_for_debug).
    """
    hint_tokens = _hint_tokens(hint)
    ev = _analyze_evidence(evidence_store, hint)

    s1, sig1 = _score_source_table_reconstruction(hint_tokens, ev)
    s2, sig2 = _score_synthesized_entity_comparison(hint_tokens, ev)
    s3, sig3 = _score_hybrid_table_synthesis(hint_tokens, ev)

    scores = {
        "source_table_reconstruction": (s1, sig1),
        "synthesized_entity_comparison": (s2, sig2),
        "hybrid_table_synthesis": (s3, sig3),
    }

    winner = max(scores, key=lambda k: scores[k][0])
    winner_score, winner_signals = scores[winner]

    score_report = {
        strategy: {"score": round(sc, 2), "signals": sig}
        for strategy, (sc, sig) in scores.items()
    }

    if debug_trace is not None:
        debug_trace.append({
            "stage": "strategy_selection",
            "winner": winner,
            "scores": score_report,
            "evidence_signals": {
                "has_source_table": ev.has_source_table,
                "best_table_score": round(ev.best_table_score, 2),
                "entity_col_present": ev.entity_col_present,
                "metric_col_count": ev.metric_col_count,
                "text_block_count": ev.text_block_count,
            },
            "hint_tokens": sorted(hint_tokens)[:20],
        })

    return winner, score_report  # type: ignore[return-value]


def strategy_to_row_discovery_mode(strategy: Strategy) -> str:
    if strategy == "source_table_reconstruction":
        return "source_table_rows"
    if strategy == "hybrid_table_synthesis":
        return "hybrid"
    return "primary_subjects"
