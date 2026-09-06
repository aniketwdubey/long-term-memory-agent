"""The read path: given a turn, surface the few memories that bear on it.

Retrieval is per-user and semantic. It returns a handful of memories, not the
store — injecting everything would put memory back inside the context window it
exists to escape, and would bury the relevant fact among dozens of irrelevant
ones.
"""

from __future__ import annotations

from datetime import datetime

import structlog
from langgraph.store.base import BaseStore

from engram.schemas import MEMORY_NAMESPACE, Memory, Recall

log = structlog.get_logger(__name__)

# Inactive memories — expired, or superseded by a newer fact — are filtered
# after the store returns its matches, so we ask for more than we need. Without
# this, a user whose top matches are all stale gets an empty recall.
_OVERFETCH = 3


class MemoryReader:
    """Semantic recall over one user's long-term memories."""

    def __init__(self, store: BaseStore, top_k: int = 5) -> None:
        self._store = store
        self._top_k = top_k

    def recall(
        self,
        user_id: str,
        query: str,
        *,
        limit: int | None = None,
        now: datetime | None = None,
    ) -> list[Recall]:
        """Return the memories most relevant to ``query``, best first."""
        k = limit or self._top_k
        if not query.strip():
            return []

        items = self._store.search(
            (MEMORY_NAMESPACE, user_id),
            query=query,
            limit=k * _OVERFETCH,
        )

        recalls: list[Recall] = []
        for item in items:
            memory = Memory.from_value(dict(item.value))
            if not memory.is_active(now):
                continue
            recalls.append(Recall(memory=memory, score=item.score or 0.0))
            if len(recalls) == k:
                break

        log.debug(
            "memory.recall",
            user_id=user_id,
            query=query,
            candidates=len(items),
            returned=len(recalls),
            memory_ids=[r.memory.id for r in recalls],
        )
        return recalls

    def all_memories(self, user_id: str, *, limit: int = 1000) -> list[Memory]:
        """Every stored memory for a user, active or not.

        For inspection, eval scoring and the ``/memories`` surface — not for the
        agent loop, which must stay top-k.
        """
        items = self._store.search((MEMORY_NAMESPACE, user_id), limit=limit)
        return [Memory.from_value(dict(item.value)) for item in items]
