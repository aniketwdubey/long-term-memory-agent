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
OtelExporter = Literal["none", "console", "otlp"]


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
    # Amazon Nova Lite: cheap enough for the high-volume half of the workload
    # (every turn is a candidate for extraction), and — being an Amazon model
    # rather than a marketplace one — covered by AWS credits. Nova Micro is
    # cheaper still but was measurably worse at choosing slot keys on the live
    # check: it filed "pytest" under `editor`. Run scripts/check_bedrock.py to
    # compare on your own account.
    bedrock_model_id: str = "amazon.nova-lite-v1:0"
    # A stronger model for judgement-heavy work. Nothing routes here today —
    # conflict resolution is deterministic policy code, not a model call — but
    # the knob exists for when something does.
    bedrock_reasoning_model_id: str = "amazon.nova-pro-v1:0"
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

    # --- Injection gate ----------------------------------------------------
    # The provenance rule — untrusted content may never write user memory — is
    # structural and not configurable. This toggles only the second, heuristic
    # layer that scans *trusted* turns for text aimed at the assistant.
    scan_trusted_content: bool = True

    # --- Retrieval ---------------------------------------------------------
    # Only a handful of memories are injected per turn — the whole point is to
    # stay cheap as the store grows past what a context window could hold.
    recall_top_k: int = Field(default=5, ge=1, le=50)

    # --- Tracing -----------------------------------------------------------
    # Off by default, and genuinely off: with no provider configured the OTel
    # API hands back a no-op tracer. For a system holding personal facts,
    # "nothing leaves the process unless asked" is the only sane default.
    # `console` prints spans; `otlp` ships them to a collector — Jaeger, a
    # self-hosted Langfuse, or CloudWatch via ADOT.
    otel_exporter: OtelExporter = "none"
    otel_endpoint: str = ""
    otel_service_name: str = "engram"

    # --- Logging -----------------------------------------------------------
    log_level: str = "INFO"
    log_format: LogFormat = "console"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (cached)."""
    return Settings()
