"""The demo: memory that survives a session boundary, and memory that doesn't.

Runs four acts against whatever backend is configured. Acts 1–3 are what slice 1
set out to prove. Act 4 is a failure, on purpose — it shows the naive write path
losing to a contradiction, which is the thing the memory manager has to fix, and
it is more useful to watch than to be told about.

    python scripts/demo.py                     # in-process, offline
    ENGRAM_STORE_BACKEND=postgres python scripts/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from engram.config import Settings  # noqa: E402
from engram.graph import Agent  # noqa: E402
from engram.logging import configure_logging  # noqa: E402
from engram.store import open_backend  # noqa: E402

USER = "demo-user"

BOLD, DIM, GREEN, RED, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[0m"


def act(n: int, title: str) -> None:
    print(f"\n{BOLD}── Act {n}: {title}{RESET}")


def say(agent: Agent, thread: str, message: str) -> str:
    trace = agent.chat(USER, thread, message)
    print(f"  {DIM}[{thread}]{RESET} user  › {message}")
    print(f"  {DIM}[{thread}]{RESET} agent › {trace.reply}")
    if trace.recalled:
        print(
            f"  {DIM}       recalled {len(trace.recalled)}: "
            f"{'; '.join(r.memory.text for r in trace.recalled)}{RESET}"
        )
    return trace.reply


def verdict(ok: bool, message: str) -> None:
    mark = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    print(f"  {mark} {message}")


def main() -> int:
    settings = Settings()
    configure_logging(settings)

    print(
        f"{BOLD}engram demo{RESET}  ·  store={settings.store_backend} "
        f"· model={settings.chat_provider} · embedder={settings.embedder}"
    )

    with open_backend(settings) as backend:
        agent = Agent(settings, checkpointer=backend.checkpointer, store=backend.store)

        act(1, "Monday — the user mentions something in passing")
        say(agent, "monday", "I'm on the payments team and I prefer pytest for everything.")

        act(2, "Thursday — a brand new session, no shared transcript")
        reply = say(agent, "thursday", "Scaffold me a test for the refund endpoint.")
        verdict("pytest" in reply.lower(), "recalled a fact across a session boundary")

        act(3, "Thread memory is durable, and separate from long-term memory")
        print(f"  {DIM}transcript of 'monday', read back from the checkpointer:{RESET}")
        for line in agent.history("monday"):
            print(f"    {DIM}{line[:96]}{RESET}")
        verdict(len(agent.history("monday")) >= 2, "the earlier thread persisted")

        act(4, "Friday — the user contradicts themselves (slice 1 fails this)")
        say(agent, "friday", "I've moved off payments — I'm on the platform team now.")
        reply = say(agent, "monday-after", "Who should review my infrastructure change?")
        stale = "payments" in reply.lower()
        verdict(
            not stale,
            "used the current fact and dropped the stale one"
            if not stale
            else "surfaced BOTH team facts — the naive writer stored the "
            "contradiction instead of resolving it (this is what the memory "
            "manager fixes next)",
        )

        total = len(agent.reader.all_memories(USER))
        print(
            f"\n{DIM}{total} memories stored for {USER}. Every turn was kept "
            f"verbatim — precision is the other thing slice 2 fixes.{RESET}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
