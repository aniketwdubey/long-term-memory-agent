"""HTTP surface for the memory subsystem.

Three groups of endpoint, and the second two matter as much as the first:

* ``/v1/chat`` — talk to the agent.
* ``/v1/memories`` — **see** what it has stored about you, and delete it. A
  memory system whose contents cannot be inspected or removed is one users have
  to take on trust, and "forget me" is not a feature that can live only in a
  Python function.
* ``/v1/observe`` and ``/v1/quarantine`` — feed the agent content it merely
  *read*, and see what the gate refused. This is the injection defence made
  demonstrable over HTTP rather than described in a README.

Every response carries the turn's memory trace: what was recalled, what was
written, and what the write path decided. That is the observability story, and
serving it is cheaper than asking someone to read logs.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from engram.graph import Agent
from engram.memory.forget import ForgetReport
from engram.schemas import Memory, Provenance

router = APIRouter(prefix="/v1")


def get_agent(request: Request) -> Agent:
    """The process-wide agent, built once at startup."""
    agent: Agent | None = getattr(request.app.state, "agent", None)
    if agent is None:  # pragma: no cover - only if the lifespan did not run
        raise HTTPException(status_code=503, detail="agent is not ready")
    return agent


AgentDep = Annotated[Agent, Depends(get_agent)]


# --- request/response models -------------------------------------------------


class ChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str = Field(min_length=1, description="Whose long-term memory to use.")
    message: str = Field(min_length=1)
    thread_id: str = Field(
        default="default",
        description="Which conversation. Independent of user_id — that separation "
        "is what carries a fact from one session to the next.",
    )


class DecisionOut(BaseModel):
    op: str
    candidate: str
    reason: str = ""


class MemoryOut(BaseModel):
    id: str
    text: str
    kind: str
    source: str
    attribute: str = ""
    value: str = ""
    importance: float
    active: bool
    created_at: str
    expires_at: str | None = None
    superseded_by: str | None = None

    @classmethod
    def of(cls, memory: Memory) -> MemoryOut:
        return cls(
            id=memory.id,
            text=memory.text,
            kind=memory.kind.value,
            source=memory.source.value,
            attribute=memory.attribute,
            value=memory.value,
            importance=memory.importance,
            active=memory.is_active(),
            created_at=memory.created_at.isoformat(),
            expires_at=memory.expires_at.isoformat() if memory.expires_at else None,
            superseded_by=memory.superseded_by,
        )


class ChatResponse(BaseModel):
    reply: str
    recalled: list[MemoryOut] = Field(default_factory=list)
    written: list[MemoryOut] = Field(default_factory=list)
    decisions: list[DecisionOut] = Field(default_factory=list)


class ObserveRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    thread_id: str | None = None
    source: Provenance = Field(
        default=Provenance.TOOL,
        description="Where this content came from. Untrusted sources cannot write "
        "user memory, whatever the text says.",
    )


class ObserveResponse(BaseModel):
    written: list[MemoryOut] = Field(default_factory=list)
    decisions: list[DecisionOut] = Field(default_factory=list)


class MemoriesResponse(BaseModel):
    user_id: str
    total: int
    active: int
    memories: list[MemoryOut] = Field(default_factory=list)


# --- endpoints ---------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, agent: AgentDep) -> ChatResponse:
    """One turn: recall, respond, remember."""
    trace = agent.chat(body.user_id, body.thread_id, body.message)
    return ChatResponse(
        reply=trace.reply,
        recalled=[MemoryOut.of(r.memory) for r in trace.recalled],
        written=[MemoryOut.of(m) for m in trace.written],
        decisions=[
            DecisionOut(op=d.op.value, candidate=d.candidate, reason=d.reason)
            for d in trace.decisions
        ],
    )


@router.post("/observe", response_model=ObserveResponse)
def observe(body: ObserveRequest, agent: AgentDep) -> ObserveResponse:
    """Feed the agent content it read rather than content the user said.

    Separate from ``/chat`` so provenance is explicit at the call site: text the
    agent merely read must never be able to arrive looking like the user
    speaking.
    """
    report = agent.observe(body.user_id, body.text, thread_id=body.thread_id, source=body.source)
    return ObserveResponse(
        written=[MemoryOut.of(m) for m in report.written],
        decisions=[
            DecisionOut(op=d.op.value, candidate=d.candidate, reason=d.reason)
            for d in report.decisions
        ],
    )


@router.get("/memories/{user_id}", response_model=MemoriesResponse)
def list_memories(user_id: str, agent: AgentDep, include_retired: bool = True) -> MemoriesResponse:
    """Everything stored about a user, retired records included by default.

    Retired records are shown because hiding them would make the store look
    tidier than it is: a superseded fact still exists, and anyone auditing what
    is held about them should see it.
    """
    memories = agent.reader.all_memories(user_id)
    shown = memories if include_retired else [m for m in memories if m.is_active()]
    return MemoriesResponse(
        user_id=user_id,
        total=len(memories),
        active=len([m for m in memories if m.is_active()]),
        memories=[MemoryOut.of(m) for m in shown],
    )


@router.get("/memories/{user_id}/history/{thread_id}")
def thread_history(user_id: str, thread_id: str, agent: AgentDep) -> dict[str, Any]:
    """The transcript of one conversation, from the checkpointer."""
    return {"user_id": user_id, "thread_id": thread_id, "history": agent.history(thread_id)}


@router.get("/quarantine/{user_id}")
def quarantine(user_id: str, agent: AgentDep) -> dict[str, Any]:
    """Content the injection gate refused, kept so an attempt stays visible."""
    held = agent.quarantined(user_id)
    return {"user_id": user_id, "count": len(held), "items": held}


@router.delete("/memories/{user_id}", response_model=ForgetReport)
def forget_user(user_id: str, agent: AgentDep) -> ForgetReport:
    """Erase a user: memories, quarantine and every transcript.

    A hard delete, not a supersession — including retired records, since a
    superseded memory still says where someone used to live.
    """
    return agent.forget(user_id)


@router.delete("/memories/{user_id}/threads/{thread_id}", response_model=ForgetReport)
def forget_thread(user_id: str, thread_id: str, agent: AgentDep) -> ForgetReport:
    """Erase one conversation and anything learned from it."""
    return agent.forget_conversation(user_id, thread_id)


@router.delete("/memories/{user_id}/{memory_id}")
def forget_memory(user_id: str, memory_id: str, agent: AgentDep) -> dict[str, Any]:
    """Delete a single memory."""
    from engram.memory.forget import forget_memory as delete_one

    if not delete_one(agent.store, user_id, memory_id):
        raise HTTPException(status_code=404, detail="no such memory")
    return {"user_id": user_id, "memory_id": memory_id, "deleted": True}
