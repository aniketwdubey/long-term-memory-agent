"""The agent graph: recall → respond → remember.

Three nodes, matching the three things a stateful agent does per turn. The write
node runs *after* the response on purpose — a fact learned from this turn should
not be recalled into the answer to that same turn, and the ordering keeps the
read path's inputs stable and reproducible.

Two arms are built from the same code. ``memory=False`` compiles a graph with
only the response node: a stateless agent that still has thread memory but no
long-term memory at all. That is the eval's control, and building it from the
same graph rather than a separate script is what makes the comparison fair.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from pydantic import BaseModel, ConfigDict

from engram.embeddings import build_embedder
from engram.memory.extract import build_extractor
from engram.memory.forget import ForgetReport, forget_thread, forget_user, record_thread
from engram.memory.gate import InjectionGate
from engram.memory.manager import MemoryManager
from engram.memory.read import MemoryReader
from engram.memory.write import MemoryDecision, MemoryWriter, NaiveMemoryWriter, WriteReport
from engram.models import build_chat_model
from engram.prompts import build_system_prompt
from engram.schemas import Memory, Provenance, Recall
from engram.tracing import set_attributes, span

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from engram.config import Settings

log = structlog.get_logger(__name__)


class AgentState(MessagesState):
    """Graph state: the thread's messages plus this turn's memory activity."""

    user_id: str
    # Carried in state rather than read from the run config, so the write path
    # can stamp every memory with the conversation it was learned in — which is
    # what makes "forget this conversation" and per-session tracing possible.
    thread_id: str
    # Per-turn, not accumulated — these are the memories that shaped *this*
    # answer, and they are what the turn trace reports.
    recalled: list[dict[str, Any]]
    written: list[dict[str, Any]]
    decisions: list[dict[str, Any]]


class TurnTrace(BaseModel):
    """What memory did during one turn — the observability unit of this system."""

    model_config = ConfigDict(frozen=True)

    reply: str
    recalled: list[Recall] = []
    written: list[Memory] = []
    # What the write path decided — including the candidates it deduplicated or
    # superseded, which produce no new memory but are the interesting outcomes.
    decisions: list[MemoryDecision] = []


def _last_human_text(state: AgentState) -> str:
    """The user's most recent message, which is both query and write candidate."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message.content
    return ""


def build_graph(
    model: BaseChatModel,
    *,
    checkpointer: BaseCheckpointSaver,  # type: ignore[type-arg]
    store: BaseStore | None = None,
    reader: MemoryReader | None = None,
    writer: MemoryWriter | None = None,
    memory: bool = True,
) -> Any:
    """Compile the agent graph.

    With ``memory=False`` the recall and remember nodes are left out entirely,
    producing the stateless control arm.
    """
    if memory and reader is None:
        raise ValueError("memory=True requires a reader")

    def recall_node(state: AgentState) -> dict[str, Any]:
        assert reader is not None
        with span("memory.recall", **{"user.id": state["user_id"]}) as current:
            recalls = reader.recall(state["user_id"], _last_human_text(state))
            set_attributes(
                current,
                {
                    "memory.recalled.count": len(recalls),
                    "memory.recalled.ids": ",".join(r.memory.id for r in recalls),
                    "memory.recalled.slots": ",".join(
                        r.memory.attribute for r in recalls if r.memory.attribute
                    ),
                },
            )
        return {"recalled": [r.model_dump(mode="json") for r in recalls]}

    def respond_node(state: AgentState) -> dict[str, Any]:
        recalls = [Recall.model_validate(r) for r in state.get("recalled", [])]
        with span("memory.respond", **{"memory.injected.count": len(recalls)}):
            prompt = [SystemMessage(content=build_system_prompt(recalls)), *state["messages"]]
            reply = model.invoke(prompt)
        return {"messages": [reply]}

    def remember_node(state: AgentState) -> dict[str, Any]:
        if writer is None:
            return {"written": [], "decisions": []}
        with span("memory.remember", **{"user.id": state["user_id"]}) as current:
            report = writer.apply(
                state["user_id"],
                _last_human_text(state),
                thread_id=state.get("thread_id"),
                source=Provenance.USER,
            )
            # The decisions are the interesting part: a turn that deduped or
            # superseded wrote nothing, and a span reporting only "0 written"
            # would make that look like a turn where nothing happened.
            set_attributes(
                current,
                {
                    "memory.written.count": len(report.written),
                    "memory.decisions": ",".join(d.op.value for d in report.decisions),
                    "memory.superseded.count": sum(len(d.superseded_ids) for d in report.decisions),
                },
            )
        return {
            "written": [m.to_value() for m in report.written],
            "decisions": [d.model_dump(mode="json") for d in report.decisions],
        }

    graph: StateGraph[AgentState, Any, Any, Any] = StateGraph(AgentState)
    graph.add_node("respond", respond_node)

    if memory:
        graph.add_node("recall", recall_node)
        graph.add_node("remember", remember_node)
        graph.add_edge(START, "recall")
        graph.add_edge("recall", "respond")
        graph.add_edge("respond", "remember")
        graph.add_edge("remember", END)
    else:
        graph.add_edge(START, "respond")
        graph.add_edge("respond", END)

    return graph.compile(checkpointer=checkpointer, store=store)


def build_writer(
    settings: Settings,
    store: BaseStore,
    *,
    model: BaseChatModel,
    embeddings: Embeddings | None = None,
) -> MemoryWriter:
    """Construct the configured write path.

    ``naive`` is kept reachable so the benchmark can run it as a control arm —
    the manager's improvement should be a measured delta, not a claim.
    """
    if settings.memory_writer == "naive":
        return NaiveMemoryWriter(store)
    return MemoryManager(
        store,
        extractor=build_extractor(settings, model),
        embeddings=embeddings or build_embedder(settings).embeddings,
        dedupe_similarity=settings.dedupe_similarity,
    )


class Agent:
    """A conversational agent with thread memory and long-term memory.

    Wraps the compiled graph in the call shape the CLI and the eval harness
    actually want: one user message in, a reply plus a memory trace out.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        checkpointer: BaseCheckpointSaver,  # type: ignore[type-arg]
        store: BaseStore,
        memory: bool = True,
        model: BaseChatModel | None = None,
        writer: MemoryWriter | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        self.settings = settings
        self.memory_enabled = memory
        self.store = store
        self.checkpointer = checkpointer
        self.reader = MemoryReader(store, top_k=settings.recall_top_k)
        chat_model = model or build_chat_model(settings)
        self.writer = writer or build_writer(
            settings, store, model=chat_model, embeddings=embeddings
        )
        self._graph = build_graph(
            chat_model,
            checkpointer=checkpointer,
            store=store,
            reader=self.reader if memory else None,
            writer=self.writer if memory else None,
            memory=memory,
        )

    def chat(self, user_id: str, thread_id: str, message: str) -> TurnTrace:
        """Send one user message on one thread and return the reply plus trace.

        ``thread_id`` selects the conversation (checkpointer); ``user_id``
        selects the long-term memory (store). They are independent on purpose:
        that separation is what makes a fact learned in one session available in
        the next, which is the whole point of the system.
        """
        if self.memory_enabled:
            # The checkpointer is keyed by thread and knows nothing about users,
            # so without this index there is no way to find — or delete — a
            # given person's transcripts. See engram.memory.forget.
            record_thread(self.store, user_id, thread_id)

        result = self._graph.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "user_id": user_id,
                "thread_id": thread_id,
                "recalled": [],
                "written": [],
                "decisions": [],
            },
            config={"configurable": {"thread_id": thread_id}},
        )

        reply = ""
        last = result["messages"][-1]
        if isinstance(last, AIMessage) and isinstance(last.content, str):
            reply = last.content

        trace = TurnTrace(
            reply=reply,
            recalled=[Recall.model_validate(r) for r in result.get("recalled", [])],
            written=[Memory.from_value(m) for m in result.get("written", [])],
            decisions=[MemoryDecision.model_validate(d) for d in result.get("decisions", [])],
        )
        log.info(
            "agent.turn",
            user_id=user_id,
            thread_id=thread_id,
            memory_enabled=self.memory_enabled,
            recalled=[r.memory.id for r in trace.recalled],
            written=[m.id for m in trace.written],
            ops=[d.op.value for d in trace.decisions],
        )
        return trace

    def observe(
        self,
        user_id: str,
        text: str,
        *,
        thread_id: str | None = None,
        source: Provenance = Provenance.TOOL,
    ) -> WriteReport:
        """Feed the agent content it *read* rather than content the user said.

        A tool result, a retrieved document, a file. This is the path an
        injection arrives on, and it is separate from :meth:`chat` precisely so
        that provenance is explicit at the call site rather than inferred: text
        the agent merely read must never be able to masquerade as the user
        speaking. The gate does the rest.
        """
        return self.writer.apply(user_id, text, thread_id=thread_id, source=source)

    def quarantined(self, user_id: str) -> list[dict[str, object]]:
        """Content the injection gate refused to store, for inspection."""
        writer = self.writer
        gate = writer.gate if isinstance(writer, MemoryManager) else InjectionGate(self.store)
        return gate.quarantined(user_id)

    def forget(self, user_id: str) -> ForgetReport:
        """Erase a user: every memory, everything quarantined, every transcript."""
        return forget_user(self.store, self.checkpointer, user_id)

    def forget_conversation(self, user_id: str, thread_id: str) -> ForgetReport:
        """Erase one conversation and anything learned from it."""
        return forget_thread(self.store, self.checkpointer, user_id, thread_id)

    def history(self, thread_id: str) -> list[str]:
        """The persisted transcript of a thread, oldest first.

        Reads straight from the checkpointer, so it survives process restarts —
        this is what demonstrates thread memory is real rather than in-process.
        """
        state = self._graph.get_state({"configurable": {"thread_id": thread_id}})
        if not state or not state.values:
            return []
        return [
            f"{m.type}: {m.content}"
            for m in state.values.get("messages", [])
            if isinstance(m.content, str)
        ]
