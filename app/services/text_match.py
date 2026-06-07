"""Language-aware keyword tokenizer shared by the evidence selectors.

The original selectors tokenized with ``[A-Za-z]`` only, so a Chinese (or any
CJK) hint produced an empty query and every item scored 0 on overlap. This
module keeps the Latin behaviour and adds CJK uni/bi-grams so non-English hints
actually drive retrieval.
"""

from __future__ import annotations

import re
from collections import Counter

# Latin/alphanumeric tokens of length >= 3 (e.g. "bleu", "f1-score", "memory").
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9-]{2,}")
# Contiguous runs of CJK ideographs and Japanese kana.
#   ぀-ヿ  Hiragana + Katakana
#   㐀-䶿  CJK Unified Ideographs Extension A
#   一-鿿  CJK Unified Ideographs
#   豈-﫿  CJK Compatibility Ideographs
_CJK_RUN = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿]+")


def _cjk_terms(text: str) -> list[str]:
    """Unigrams + adjacent bigrams within each CJK run.

    CJK has no word delimiters, so a single token boundary doesn't exist. Bigrams
    capture the most common compound terms (e.g. 錯誤/誤率/錯誤率) while unigrams
    keep recall for single-character matches.
    """
    terms: list[str] = []
    for run in _CJK_RUN.findall(text):
        terms.extend(run)
        terms.extend(run[i : i + 2] for i in range(len(run) - 1))
    return terms


def keywords(text: str, stopwords: frozenset[str] | set[str] = frozenset()) -> Counter[str]:
    """Tokenize ``text`` into a multiset of weighted terms (Latin words + CJK n-grams)."""
    lowered = text.lower()
    terms = [word for word in _LATIN.findall(lowered) if word not in stopwords]
    terms.extend(term for term in _cjk_terms(text) if term not in stopwords)
    return Counter(terms)


def matched_terms(query: Counter[str], target: Counter[str]) -> list[str]:
    """Terms present in both query and target, ordered by query weight (for tracing)."""
    matched = [
        (term, min(count, target.get(term, 0)))
        for term, count in query.items()
        if target.get(term, 0)
    ]
    matched.sort(key=lambda entry: -entry[1])
    return [term for term, _ in matched]


def overlap_score(query: Counter[str], target: Counter[str]) -> int:
    """Capped term-frequency overlap between a query and a target multiset."""
    return sum(min(count, target.get(term, 0)) for term, count in query.items())
