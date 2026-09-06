"""Forgetting on request — the delete path.

Decay handles facts that stop being true. This handles a different requirement:
a person asking for their data to be gone. That is a **hard delete**, not a
supersession — nothing is retired, nothing is kept as history, and afterwards
the store cannot answer questions about them at all.

Three things have to go, and missing any one of them makes "forget me" a lie:

* **memories** — including the retired ones. A superseded record still says
  where someone used to live.
* **quarantine** — blocked content is still content about them.
* **thread transcripts** — the checkpointer holds the raw conversation, which is
  usually the most sensitive of the three.

The transcripts are the awkward part: LangGraph's checkpointer is keyed by
thread, with no notion of a user, so there is no query that finds "this user's
threads". Rather than leave a hole, the agent maintains a small index — a store
entry per ``(user, thread)`` — so a complete delete is possible. Building that
index is the price of being able to honour the request.
"""

from __future__ import annotations

from typing import Any

import structlog
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from pydantic import BaseModel, ConfigDict, Field

from engram.memory.gate import QUARANTINE_NAMESPACE
from engram.schemas import MEMORY_NAMESPACE, Memory, utcnow

log = structlog.get_logger(__name__)

THREAD_INDEX_NAMESPACE = "threads"

_PAGE = 200


class ForgetReport(BaseModel):
    """What a delete actually removed. Auditable, because it has to be."""

    model_config = ConfigDict(frozen=True)

    user_id: str
    memories_deleted: int = 0
    quarantine_deleted: int = 0
    threads_deleted: list[str] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return self.memories_deleted + self.quarantine_deleted + len(self.threads_deleted)


def _all_keys(store: BaseStore, namespace: tuple[str, ...]) -> list[tuple[str, dict[str, Any]]]:
    """Every key in a namespace, paged.

    A single ``search`` returns one page; deleting only that page would leave a
    user believing they had been forgotten while records remained.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    offset = 0
    while True:
        page = store.search(namespace, limit=_PAGE, offset=offset)
        if not page:
            return out
        out.extend((item.key, dict(item.value)) for item in page)
        offset += len(page)


def record_thread(store: BaseStore, user_id: str, thread_id: str) -> None:
    """Note that this user has spoken on this thread.

    Idempotent, and the only reason a complete delete is possible at all.
    """
    store.put(
        (THREAD_INDEX_NAMESPACE, user_id),
        thread_id,
        {"thread_id": thread_id, "last_seen": utcnow().isoformat()},
        index=False,
    )


def known_threads(store: BaseStore, user_id: str) -> list[str]:
    """Every thread this user is known to have used."""
    return sorted(key for key, _ in _all_keys(store, (THREAD_INDEX_NAMESPACE, user_id)))


def forget_memory(store: BaseStore, user_id: str, memory_id: str) -> bool:
    """Delete one memory outright. Returns whether it existed."""
    namespace = (MEMORY_NAMESPACE, user_id)
    if store.get(namespace, memory_id) is None:
        return False
    store.delete(namespace, memory_id)
    log.info("memory.forget.one", user_id=user_id, memory_id=memory_id)
    return True


def forget_thread(
    store: BaseStore,
    checkpointer: BaseCheckpointSaver,  # type: ignore[type-arg]
    user_id: str,
    thread_id: str,
) -> ForgetReport:
    """Delete one conversation: its transcript and anything learned from it."""
    deleted = 0
    for key, value in _all_keys(store, (MEMORY_NAMESPACE, user_id)):
        if Memory.from_value(value).thread_id == thread_id:
            store.delete((MEMORY_NAMESPACE, user_id), key)
            deleted += 1

    checkpointer.delete_thread(thread_id)
    store.delete((THREAD_INDEX_NAMESPACE, user_id), thread_id)

    log.info("memory.forget.thread", user_id=user_id, thread_id=thread_id, memories=deleted)
    return ForgetReport(user_id=user_id, memories_deleted=deleted, threads_deleted=[thread_id])


def forget_user(
    store: BaseStore,
    checkpointer: BaseCheckpointSaver,  # type: ignore[type-arg]
    user_id: str,
) -> ForgetReport:
    """Erase a user completely: memories, quarantine, and every transcript.

    Deletes retired and expired records too — a superseded memory still says
    where someone used to live, and "we only kept the outdated copy" is not a
    defence anyone would accept.
    """
    memories = _all_keys(store, (MEMORY_NAMESPACE, user_id))
    for key, _ in memories:
        store.delete((MEMORY_NAMESPACE, user_id), key)

    quarantined = _all_keys(store, (QUARANTINE_NAMESPACE, user_id))
    for key, _ in quarantined:
        store.delete((QUARANTINE_NAMESPACE, user_id), key)

    threads = known_threads(store, user_id)
    for thread_id in threads:
        checkpointer.delete_thread(thread_id)
        store.delete((THREAD_INDEX_NAMESPACE, user_id), thread_id)

    report = ForgetReport(
        user_id=user_id,
        memories_deleted=len(memories),
        quarantine_deleted=len(quarantined),
        threads_deleted=threads,
    )
    log.info(
        "memory.forget.user",
        user_id=user_id,
        memories=report.memories_deleted,
        quarantine=report.quarantine_deleted,
        threads=len(report.threads_deleted),
    )
    return report
