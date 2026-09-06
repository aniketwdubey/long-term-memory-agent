"""Chat model provider: a deterministic offline stub, or Claude on Bedrock.

Two providers behind one factory, the same shape project 07 uses. ``stub`` is
the default so a fresh checkout runs tests, the demo and the whole eval harness
with no credentials and no network.

**What the stub is for, and what it is not.** It is not a language model and does
not pretend to reason. It answers by reporting exactly the memories that were put
in its context, which makes it a *transparent instrument*: an eval run against it
measures the memory subsystem — did retrieval surface the right fact, did the
write path store the wrong ones — with model quality held at a constant. That is
the property a CI regression gate needs. Point ``ENGRAM_CHAT_PROVIDER=bedrock`` at
real Claude to measure the pair together.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from engram.config import Settings

# The stub finds the recalled-memory block the prompt builder emits by looking
# for these markers, rather than trying to parse prose. Kept in sync with
# engram.prompts.
MEMORY_BLOCK_START = "<recalled_memories>"
MEMORY_BLOCK_END = "</recalled_memories>"

NO_MEMORY_REPLY = (
    "I don't have anything stored about you, so I'm answering from the current conversation only."
)


def extract_memory_block(messages: list[BaseMessage]) -> list[str]:
    """Pull the recalled memory lines out of a rendered prompt.

    Returns the bullet texts between the markers, in order, across every message
    in the prompt. Empty when nothing was recalled.
    """
    lines: list[str] = []
    for message in messages:
        content = message.content
        if not isinstance(content, str):
            continue
        for block in re.findall(
            f"{re.escape(MEMORY_BLOCK_START)}(.*?){re.escape(MEMORY_BLOCK_END)}",
            content,
            flags=re.DOTALL,
        ):
            for raw in block.splitlines():
                line = raw.strip()
                if line.startswith("- "):
                    lines.append(line[2:].strip())
    return lines


def _default_responder(messages: list[BaseMessage]) -> str:
    """Report the memories in context, verbatim and in order.

    Deliberately mechanical. Because the reply contains a recalled fact if and
    only if retrieval supplied it, an eval assertion like "the answer mentions
    pytest" becomes a direct measurement of the memory system.

    It also surfaces the naive write path's central flaw rather than hiding it:
    when contradicting facts have both been stored, both appear here, so a case
    that forbids the stale answer fails — which is exactly the gap the memory
    manager has to close.
    """
    recalled = extract_memory_block(messages)
    if not recalled:
        return NO_MEMORY_REPLY
    facts = " ".join(f"You told me: {fact}" for fact in recalled)
    return f"Based on what I remember about you — {facts}"


class StubChatModel(BaseChatModel):
    """A deterministic, offline stand-in for the chat model.

    ``responder`` is overridable so a test can simulate a specific model
    behaviour (a refusal, a hallucination, a malformed structured output)
    without reaching for mock patching.
    """

    responder: Callable[[list[BaseMessage]], str] | None = None

    @property
    def _llm_type(self) -> str:
        return "engram-stub"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        respond = self.responder or _default_responder
        text = respond(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Construct the configured chat model."""
    if settings.chat_provider == "stub":
        return StubChatModel()

    if settings.chat_provider == "bedrock":
        from langchain_aws import ChatBedrockConverse

        # Sampling parameters are deliberately omitted: the current Claude
        # models reject temperature/top_p/top_k with a 400.
        return ChatBedrockConverse(
            model=settings.bedrock_model_id,
            region_name=settings.aws_region,
            max_tokens=settings.max_tokens,
        )

    raise ValueError(f"Unknown chat provider: {settings.chat_provider!r}")
