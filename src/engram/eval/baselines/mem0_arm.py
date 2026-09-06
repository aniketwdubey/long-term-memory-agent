"""mem0 as a benchmark arm — the managed memory layer, on identical footing.

mem0 is one of the products that exists because everyone building on a stateless
API has to solve this. Running it as an arm answers the only question that
matters about the custom write path: was building it worth doing?

**Everything except the memory logic is held constant.** Same Nova model, same
Titan embeddings, same Postgres instance, same prompt builder, same scoring.
mem0 is wired through its ``langchain`` provider so it receives the *same
objects* our agent uses, not merely equivalent configuration.

That routing is also a necessity. mem0 2.0.20's own ``aws_bedrock`` adapter is
broken for Amazon models: ``_format_messages_amazon`` returns
``{"role": ..., "content": "<text>"}`` where the Bedrock Converse API requires
content blocks (``[{"text": ...}]``) — its Anthropic formatter gets this right,
its Amazon one does not, so every Nova call fails parameter validation. Going
through ``langchain`` sidesteps mem0's adapter entirely.

Two things this arm is *not*:

* It is not mem0 tuned. It is mem0's default extraction behaviour on our models.
  mem0 supports custom prompts, graph memory and a hosted platform, none of
  which are used here — a fair reading of the numbers is "default mem0", not
  "the best mem0 can do".
* It is not offline. mem0 needs a real LLM for every write, so this arm cannot
  join the CI gate and is opt-in via ``--arms mem0``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

# Set before importing mem0: it ships PostHog telemetry that is on by default,
# and a benchmark run should not quietly report itself to a third party.
os.environ.setdefault("MEM0_TELEMETRY_ENABLED", "False")
os.environ.setdefault("MEM0_TELEMETRY", "False")

import structlog  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from engram.graph import TurnTrace  # noqa: E402
from engram.memory.write import MemoryDecision, MemoryOp, WriteReport  # noqa: E402
from engram.models import build_chat_model  # noqa: E402
from engram.prompts import build_system_prompt  # noqa: E402
from engram.schemas import Memory, Provenance, Recall  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from engram.config import Settings

log = structlog.get_logger(__name__)

COLLECTION = "engram_mem0_baseline"

try:  # pragma: no cover - depends on the optional extra
    from mem0 import Memory as Mem0Memory

    MEM0_AVAILABLE = True
except ImportError:  # pragma: no cover
    MEM0_AVAILABLE = False


class _Mem0Reader:
    """Adapts mem0's records to the shape the eval's store metrics expect."""

    def __init__(self, memory: Any) -> None:
        self._memory = memory

    def all_memories(self, user_id: str) -> list[Memory]:
        results = self._memory.get_all(filters={"user_id": user_id}, top_k=500)
        return [_to_memory(r, user_id) for r in results.get("results", [])]


def _to_memory(record: dict[str, Any], user_id: str) -> Memory:
    """Map one mem0 record onto our record type.

    ``attribute``/``value`` stay empty: mem0 stores memories as free text and
    has no slot concept, which is precisely the design difference under test.
    Everything the metrics read — the text, and whether it is live — is present.
    """
    fields: dict[str, Any] = {
        "user_id": user_id,
        "text": str(record.get("memory", "")).strip() or "(empty)",
        "source": Provenance.USER,
    }
    # Omit the id rather than passing None: the field has a default factory, and
    # an explicit None fails validation instead of falling back to it.
    if record.get("id"):
        fields["id"] = str(record["id"])
    return Memory(**fields)


class Mem0Agent:
    """An agent whose long-term memory is mem0 instead of ours.

    Duck-types the surface :func:`engram.eval.runner.run_case` uses, so the same
    cases and the same scoring run against it unchanged.
    """

    def __init__(self, settings: Settings, memory: Any) -> None:
        self.settings = settings
        self.memory_enabled = True
        self._memory = memory
        self._model = build_chat_model(settings)
        self._top_k = settings.recall_top_k
        self.reader = _Mem0Reader(memory)

    # -- the surface run_case drives ---------------------------------------

    def chat(self, user_id: str, thread_id: str, message: str) -> TurnTrace:
        recalls = self._recall(user_id, message)
        prompt = [
            SystemMessage(content=build_system_prompt(recalls)),
            HumanMessage(content=message),
        ]
        reply = self._model.invoke(prompt)
        report = self._write(user_id, message, thread_id=thread_id)

        text = reply.content if isinstance(reply.content, str) else ""
        return TurnTrace(reply=text, recalled=recalls, written=report.written)

    def observe(
        self,
        user_id: str,
        text: str,
        *,
        thread_id: str | None = None,
        source: Provenance = Provenance.TOOL,
    ) -> WriteReport:
        """Content the agent read rather than content the user said.

        mem0 has no provenance concept, so this is an ordinary write. That is
        not a shortcut in the adapter — it is the finding the injection cases
        are there to measure.
        """
        return self._write(user_id, text, thread_id=thread_id)

    def history(self, thread_id: str) -> list[str]:
        """No thread memory here.

        Only the probe is scored and it is always asked in a fresh thread, so a
        transcript would give neither arm an advantage. Leaving it out keeps the
        comparison about long-term memory alone.
        """
        return []

    # -- internals ----------------------------------------------------------

    def _recall(self, user_id: str, query: str) -> list[Recall]:
        if not query.strip():
            return []
        found = self._memory.search(query, filters={"user_id": user_id}, top_k=self._top_k)
        return [
            Recall(memory=_to_memory(r, user_id), score=float(r.get("score") or 0.0))
            for r in found.get("results", [])
        ]

    def _write(self, user_id: str, text: str, *, thread_id: str | None) -> WriteReport:
        text = text.strip()
        if not text:
            return WriteReport()
        try:
            result = self._memory.add([{"role": "user", "content": text}], user_id=user_id)
        except Exception as exc:
            log.warning("mem0.add.failed", user_id=user_id, error=str(exc)[:200])
            return WriteReport()

        written: list[Memory] = []
        decisions: list[MemoryDecision] = []
        for record in result.get("results", []):
            event = str(record.get("event", "")).upper()
            memory = _to_memory(record, user_id)
            # mem0 reports its own decision per record; map it onto ours so the
            # traces read the same way in both arms.
            op = {
                "ADD": MemoryOp.WRITE,
                "UPDATE": MemoryOp.SUPERSEDE,
                "DELETE": MemoryOp.SUPERSEDE,
                "NONE": MemoryOp.DEDUPE,
            }.get(event, MemoryOp.WRITE)
            if op is MemoryOp.WRITE:
                written.append(memory)
            decisions.append(
                MemoryDecision(op=op, candidate=text, memory_id=memory.id, reason=f"mem0 {event}")
            )
        return WriteReport(written=written, decisions=decisions)


def build_mem0_agent(settings: Settings) -> Mem0Agent:
    """Wire mem0 onto the same model, embeddings and database we use."""
    if not MEM0_AVAILABLE:
        raise RuntimeError('The mem0 arm needs the optional extra: pip install -e ".[baseline]"')
    if settings.chat_provider != "bedrock":
        raise RuntimeError(
            "The mem0 arm needs a real LLM — it calls one on every write. "
            "Run with --provider bedrock --embedder bedrock."
        )

    from langchain_aws import BedrockEmbeddings

    memory = Mem0Memory.from_config(
        {
            # Handed the same objects our agent uses, not merely the same config.
            "llm": {"provider": "langchain", "config": {"model": build_chat_model(settings)}},
            "embedder": {
                "provider": "langchain",
                "config": {
                    "model": BedrockEmbeddings(
                        model_id=settings.bedrock_embed_model_id,
                        region_name=settings.aws_region,
                    )
                },
            },
            "vector_store": {
                "provider": "pgvector",
                "config": {
                    "connection_string": settings.postgres_dsn,
                    "collection_name": COLLECTION,
                    "embedding_model_dims": 1024,
                },
            },
        }
    )
    # A benchmark starts from an empty store, or the second run measures the
    # first one's leftovers. Scoped to our own collection.
    try:
        memory.reset()
    except Exception as exc:  # pragma: no cover - first run has nothing to reset
        log.debug("mem0.reset.skipped", error=str(exc)[:120])
    return Mem0Agent(settings, memory)
