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
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.store.base import BaseStore
from pydantic import BaseModel, ConfigDict

from engram.memory.read import MemoryReader
from engram.memory.write import MemoryWriter, NaiveMemoryWriter
from engram.models import build_chat_model
from engram.prompts import build_system_prompt
from engram.schemas import Memory, Provenance, Recall

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from engram.config import Settings

log = structlog.get_logger(__name__)


class AgentState(MessagesState):
    """Graph state: the thread's messages plus this turn's memory activity."""

    user_id: str
    # Per-turn, not accumulated — these are the memories that shaped *this*
    # answer, and they are what the turn trace reports.
    recalled: list[dict[str, Any]]
    written: list[dict[str, Any]]


class TurnTrace(BaseModel):
    """What memory did during one turn — the observability unit of this system."""

    model_config = ConfigDict(frozen=True)

    reply: str
    recalled: list[Recall] = []
    written: list[Memory] = []


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
        recalls = reader.recall(state["user_id"], _last_human_text(state))
        return {"recalled": [r.model_dump(mode="json") for r in recalls]}

    def respond_node(state: AgentState) -> dict[str, Any]:
        recalls = [Recall.model_validate(r) for r in state.get("recalled", [])]
        prompt = [SystemMessage(content=build_system_prompt(recalls)), *state["messages"]]
        reply = model.invoke(prompt)
        return {"messages": [reply]}

    def remember_node(state: AgentState) -> dict[str, Any]:
        if writer is None:
            return {"written": []}
        written = writer.write(
            state["user_id"],
            _last_human_text(state),
            source=Provenance.USER,
        )
        return {"written": [m.to_value() for m in written]}

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
    ) -> None:
        self.settings = settings
        self.memory_enabled = memory
        self.store = store
        self.reader = MemoryReader(store, top_k=settings.recall_top_k)
        self.writer = writer or NaiveMemoryWriter(store)
        self._graph = build_graph(
            model or build_chat_model(settings),
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
        result = self._graph.invoke(
            {
                "messages": [HumanMessage(content=message)],
                "user_id": user_id,
                "recalled": [],
                "written": [],
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
        )
        log.info(
            "agent.turn",
            user_id=user_id,
            thread_id=thread_id,
            memory_enabled=self.memory_enabled,
            recalled=[r.memory.id for r in trace.recalled],
            written=[m.id for m in trace.written],
        )
        return trace

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
