"""Application configuration.

Every setting comes from the environment (12-factor) via ``pydantic-settings``;
nothing here reads a secret from source. A single cached ``Settings`` instance is
imported across the app, while tests construct their own ``Settings(...)`` to
override individual fields.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ChatProvider = Literal["stub", "bedrock"]
EmbedderName = Literal["hashing", "fastembed", "bedrock"]
StoreBackend = Literal["memory", "postgres"]
WriterName = Literal["naive", "manager"]
LogFormat = Literal["console", "json"]


class Settings(BaseSettings):
    """Typed, validated application settings loaded from env / ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="ENGRAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # --- Chat model --------------------------------------------------------
    # `stub` is a deterministic offline model (see engram.models). It is the
    # default so that tests, CI and `make eval` never touch the network.
    chat_provider: ChatProvider = "stub"
    aws_region: str = "us-east-1"
    # Haiku 4.5 is the cheapest current Claude on Bedrock and is the right tier
    # for the high-volume half of the memory workload (fact extraction, dedupe
    # triage). Current Claude models are not invokable by their bare id — an
    # inference-profile id (`us.` prefix) is required.
    bedrock_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    # Conflict resolution ("does this new fact update, supersede or coexist with
    # that one?") is the reasoning-heavy half and gets its own, stronger model.
    # Defaults to the same id so a fresh checkout costs nothing extra; point it
    # at a larger model when you want the better judgement.
    bedrock_reasoning_model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    bedrock_embed_model_id: str = "amazon.titan-embed-text-v2:0"
    max_tokens: int = Field(default=2048, ge=256, le=8192)
    llm_timeout_seconds: float = Field(default=30.0, gt=0)

    # --- Embeddings --------------------------------------------------------
    # `hashing` is deterministic and dependency-free, which is what CI wants.
    # `fastembed` gives real sentence semantics offline (install the `embed`
    # extra); `bedrock` uses Titan v2. See engram.embeddings.
    embedder: EmbedderName = "hashing"
    hashing_embed_dims: int = Field(default=256, ge=32, le=4096)
    fastembed_model: str = "BAAI/bge-small-en-v1.5"

    # --- Stores ------------------------------------------------------------
    # `memory` keeps checkpoints and memories in-process (tests, quick demos).
    # `postgres` is the real deliverable: PostgresSaver + pgvector-backed
    # PostgresStore, both from LangGraph itself.
    store_backend: StoreBackend = "memory"
    postgres_dsn: str = "postgresql://engram:engram@localhost:5432/engram"

    # --- Memory write path -------------------------------------------------
    # `manager` extracts, dedupes, resolves conflicts and decays. `naive` stores
    # every turn verbatim and exists so the benchmark can run it as a control —
    # the manager's improvement should be a measured delta, not a claim.
    memory_writer: WriterName = "manager"
    # Cosine similarity above which two unslotted facts are the same fact.
    # Slotted facts do not use this: they dedupe on the slot's value.
    dedupe_similarity: float = Field(default=0.9, ge=0.0, le=1.0)

    # --- Retrieval ---------------------------------------------------------
    # Only a handful of memories are injected per turn — the whole point is to
    # stay cheap as the store grows past what a context window could hold.
    recall_top_k: int = Field(default=5, ge=1, le=50)

    # --- Logging -----------------------------------------------------------
    log_level: str = "INFO"
    log_format: LogFormat = "console"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings()
