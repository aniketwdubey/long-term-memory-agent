"""Map public conversation names to user-scoped checkpoint identifiers."""

from __future__ import annotations

import hashlib
import json

from langgraph.store.base import BaseStore

THREAD_INDEX_NAMESPACE = "threads"


class LegacyThreadError(ValueError):
    """An old, globally keyed transcript needs operator migration."""


def checkpoint_thread_id(store: BaseStore, user_id: str, thread_id: str) -> str:
    """Resolve a conversation without ever falling back to a shared transcript.

    JSON encodes the pair unambiguously even when IDs contain separators. Keep
    the public thread name in memory records and API responses, not this key.
    """
    pair = json.dumps([user_id, thread_id], ensure_ascii=True, separators=(",", ":"))
    key = "engram:v2:" + hashlib.sha256(pair.encode()).hexdigest()
    entry = store.get((THREAD_INDEX_NAMESPACE, user_id), thread_id)
    if entry is not None and entry.value.get("checkpoint_id") != key:
        raise LegacyThreadError(
            "This user has a legacy conversation requiring operator migration; "
            "see docs/operations.md#upgrading-existing-data."
        )
    return key
