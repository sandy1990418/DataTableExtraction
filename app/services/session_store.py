"""In-memory store for analysis sessions.

`/evidence` runs the expensive parse → vision → table-extraction pipeline once and
caches the full result here, returning a short `session_id`. `/outline` and
`/render` then take just that id instead of shuttling the whole table pool back
and forth — and they always see the FULL tables (with rows), so the planner gets
real row counts.

TTL-based, single-process (matches the pptx token store). For multi-worker
deployments swap this for Redis or similar.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

# token -> (payload, expiry_monotonic)
_store: dict[str, tuple[dict[str, Any], float]] = {}


def store_session(payload: dict[str, Any], ttl_seconds: int) -> str:
    evict_expired()
    token = uuid.uuid4().hex
    _store[token] = (payload, time.monotonic() + ttl_seconds)
    return token


def get_session(session_id: str) -> dict[str, Any] | None:
    entry = _store.get(session_id)
    if entry is None:
        return None
    payload, expiry = entry
    if time.monotonic() > expiry:
        del _store[session_id]
        return None
    return payload


def evict_expired() -> None:
    now = time.monotonic()
    for token in [t for t, (_, expiry) in _store.items() if now > expiry]:
        del _store[token]
