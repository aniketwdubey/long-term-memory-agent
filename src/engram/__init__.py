"""engram — long-term memory for conversational agents.

An *engram* is the physical trace a single memory leaves behind; this package is
about producing good ones. The public LLM APIs are stateless, so everything that
lets an agent know a user across sessions has to be built here: what to store,
how to retrieve only what matters, how to reconcile facts that contradict each
other, when to forget, and what is never allowed to become a memory at all.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
