"""The memory subsystem: the read path, the write path, and the contract between.

The read path is largely a thin, honest wrapper over the store's vector search.
The write path is where memory systems live or die, and where this project's
custom code belongs — see :mod:`engram.memory.write`.
"""

from engram.memory.read import MemoryReader
from engram.memory.write import MemoryWriter, NaiveMemoryWriter

__all__ = ["MemoryReader", "MemoryWriter", "NaiveMemoryWriter"]
