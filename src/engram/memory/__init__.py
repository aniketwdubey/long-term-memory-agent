"""The memory subsystem: the read path, the write path, and the contract between.

The read path is largely a thin, honest wrapper over the store's vector search.
The write path is where memory systems live or die, and where this project's
custom code belongs — see :mod:`engram.memory.manager`.
"""

from engram.memory.extract import (
    CandidateFact,
    ExtractionResult,
    FactExtractor,
    LLMFactExtractor,
    RuleFactExtractor,
    build_extractor,
)
from engram.memory.manager import MemoryManager
from engram.memory.read import MemoryReader
from engram.memory.write import (
    MemoryDecision,
    MemoryOp,
    MemoryWriter,
    NaiveMemoryWriter,
    WriteReport,
)

__all__ = [
    "CandidateFact",
    "ExtractionResult",
    "FactExtractor",
    "LLMFactExtractor",
    "MemoryDecision",
    "MemoryManager",
    "MemoryOp",
    "MemoryReader",
    "MemoryWriter",
    "NaiveMemoryWriter",
    "RuleFactExtractor",
    "WriteReport",
    "build_extractor",
]
