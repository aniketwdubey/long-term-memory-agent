"""The agent loop: the two arms, cross-session recall, and the turn trace."""

from __future__ import annotations

import pytest

from engram.config import Settings
from engram.graph import Agent
from engram.models import NO_MEMORY_REPLY
from engram.store import Backend


def test_recalls_a_fact_in_a_later_session(agent: Agent) -> None:
    """The headline behaviour: two threads that share nothing but a user."""
    agent.chat("alice", "monday", "I'm on the payments team and I prefer pytest.")
    reply = agent.chat("alice", "thursday", "Scaffold me a test for the refund endpoint.").reply
    assert "pytest" in reply


def test_a_new_thread_starts_with_no_transcript(agent: Agent) -> None:
    """Proves the recall above came from long-term memory, not the transcript."""
    agent.chat("alice", "monday", "I prefer pytest.")
    assert agent.history("thursday") == []


def test_stateless_arm_cannot_recall_anything(settings: Settings, backend: Backend) -> None:
    stateless = Agent(
        settings, checkpointer=backend.checkpointer, store=backend.store, memory=False
    )
    stateless.chat("alice", "monday", "I'm on the payments team and I prefer pytest.")
    trace = stateless.chat("alice", "thursday", "Scaffold me a test.")

    assert trace.reply == NO_MEMORY_REPLY
    assert trace.recalled == []
    assert trace.written == []
    assert stateless.reader.all_memories("alice") == []


def test_one_user_cannot_see_another_users_memories(agent: Agent) -> None:
    agent.chat("alice", "t1", "I prefer pytest.")
    assert "pytest" not in agent.chat("bob", "t2", "What framework do I prefer?").reply


def test_turn_trace_reports_what_memory_did(agent: Agent) -> None:
    """Memory-op tracing — which memories were read and written, per turn."""
    first = agent.chat("alice", "t1", "I prefer pytest.")
    assert [m.text for m in first.written] == ["I prefer pytest."]
    assert first.recalled == []

    second = agent.chat("alice", "t2", "Write me a test.")
    assert [r.memory.text for r in second.recalled] == ["I prefer pytest."]
    assert second.recalled[0].memory.id == first.written[0].id


def test_the_current_turn_is_not_recalled_into_its_own_answer(agent: Agent) -> None:
    """Writing after responding keeps the read path's inputs stable."""
    trace = agent.chat("alice", "t1", "I prefer pytest.")
    assert trace.recalled == []
    assert len(trace.written) == 1


def test_thread_memory_persists_across_turns(agent: Agent) -> None:
    agent.chat("alice", "t1", "first message")
    agent.chat("alice", "t1", "second message")

    history = agent.history("t1")
    assert len(history) == 4  # two human turns, two agent replies
    assert history[0].startswith("human: first message")
    assert history[2].startswith("human: second message")


def test_history_of_an_unknown_thread_is_empty(agent: Agent) -> None:
    assert agent.history("never-used") == []


def test_memory_arm_requires_a_reader(settings: Settings, backend: Backend) -> None:
    from engram.graph import build_graph
    from engram.models import StubChatModel

    with pytest.raises(ValueError, match="requires a reader"):
        build_graph(StubChatModel(), checkpointer=backend.checkpointer, memory=True)


def test_naive_writer_stores_a_contradiction_instead_of_resolving_it(
    agent: Agent,
) -> None:
    """Pins slice 1's central weakness so slice 2's fix is a visible diff.

    When this starts failing, the memory manager is doing its job and this test
    should be inverted rather than deleted.
    """
    agent.chat("alice", "t1", "I'm on the payments team.")
    agent.chat("alice", "t2", "I switched to the platform team.")

    stored = [m.text for m in agent.reader.all_memories("alice") if m.is_active()]
    assert "I'm on the payments team." in stored
    assert "I switched to the platform team." in stored
