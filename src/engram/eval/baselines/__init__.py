"""Managed memory services, benchmarked as arms alongside our own write path.

The point of a baseline is not to lose gracefully to it. It is to find out
whether building the write path by hand was worth doing at all, on the same
cases, with the same model, the same embeddings and the same database — so that
the only thing that differs is the memory logic.
"""

from engram.eval.baselines.mem0_arm import MEM0_AVAILABLE, Mem0Agent, build_mem0_agent

__all__ = ["MEM0_AVAILABLE", "Mem0Agent", "build_mem0_agent"]
