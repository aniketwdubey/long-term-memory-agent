"""Interactive REPL against the agent — the by-hand version of the demo.

    python scripts/chat.py --user alice --thread monday
    python scripts/chat.py --user alice --thread thursday   # same user, new session

Commands: /memories to list what is stored, /history for this thread, /quit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from engram.config import Settings  # noqa: E402
from engram.graph import Agent  # noqa: E402
from engram.logging import configure_logging  # noqa: E402
from engram.store import open_backend  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with the memory agent.")
    parser.add_argument("--user", default="alice", help="Long-term memory owner.")
    parser.add_argument("--thread", default="main", help="Conversation id.")
    parser.add_argument("--no-memory", action="store_true", help="Stateless control arm.")
    args = parser.parse_args()

    settings = Settings()
    configure_logging(settings)

    with open_backend(settings) as backend:
        agent = Agent(
            settings,
            checkpointer=backend.checkpointer,
            store=backend.store,
            memory=not args.no_memory,
        )
        print(
            f"user={args.user} thread={args.thread} store={settings.store_backend} "
            f"memory={'off' if args.no_memory else 'on'}  ·  /memories /history /quit"
        )
        while True:
            try:
                line = input("› ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                continue
            if line in {"/quit", "/exit"}:
                return 0
            if line == "/memories":
                for m in agent.reader.all_memories(args.user):
                    flag = " " if m.is_active() else "×"
                    print(f"  {flag} [{m.kind.value:<10} {m.source.value:<8}] {m.text}")
                continue
            if line == "/history":
                for entry in agent.history(args.thread):
                    print(f"  {entry}")
                continue

            trace = agent.chat(args.user, args.thread, line)
            print(trace.reply)
            if trace.recalled:
                print(f"  ({len(trace.recalled)} recalled)")


if __name__ == "__main__":
    raise SystemExit(main())
