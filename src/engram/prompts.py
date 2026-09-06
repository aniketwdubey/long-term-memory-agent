"""Prompt construction for the response node.

Only the memories retrieval judged relevant to *this* turn are rendered — never
the whole store. That is the property that lets memory outlive the context
window: recall cost stays flat as the number of stored facts grows.
"""

from __future__ import annotations

from collections.abc import Sequence

from engram.models import MEMORY_BLOCK_END, MEMORY_BLOCK_START
from engram.schemas import Recall

SYSTEM_PROMPT = f"""You are a helpful assistant that remembers this user across \
conversations.

Anything inside {MEMORY_BLOCK_START} was recalled from your long-term memory of this user. \
Treat it as background you already know: use it to answer without asking the \
user to repeat themselves, and prefer it over generic defaults. If a recalled \
memory contradicts what the user says now, believe the user and say so. If \
nothing recalled is relevant, ignore it — do not force it into the answer."""


def render_memory_block(recalls: Sequence[Recall]) -> str:
    """Render recalled memories as a delimited bullet list.

    Returns an empty string when nothing was recalled, so the prompt carries no
    empty block — an agent told "here is what you remember: (nothing)" tends to
    comment on the absence instead of just answering.
    """
    if not recalls:
        return ""
    bullets = "\n".join(f"- {r.memory.text}" for r in recalls)
    return f"{MEMORY_BLOCK_START}\n{bullets}\n{MEMORY_BLOCK_END}"


def build_system_prompt(recalls: Sequence[Recall]) -> str:
    """The full system message for one turn."""
    block = render_memory_block(recalls)
    return f"{SYSTEM_PROMPT}\n\n{block}" if block else SYSTEM_PROMPT
