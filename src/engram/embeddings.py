"""Pluggable embeddings, behind LangChain's ``Embeddings`` interface.

Implementing that interface (rather than inventing our own) is what lets the same
object be handed straight to a LangGraph store's ``index={"embed": ...}`` config,
so the vector search in Postgres and the one in the in-memory store are driven by
identical vectors.

Three backends, picked by ``ENGRAM_EMBEDDER``:

``hashing``    deterministic, dependency-free, identical across processes and
               machines. The default, and what CI uses — an eval whose numbers
               move because a model file changed is not a regression gate.
``fastembed``  real sentence semantics offline via ONNX (no torch). This is the
               one that can tell "I moved to Berlin" from "I live in Mumbai",
               which is what the dedupe and conflict paths actually need.
``bedrock``    Amazon Titan v2, for parity with a deployed run.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, NamedTuple

from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from engram.config import Settings

# Output dimensions for embedding models we do not want to pay a probe call to
# discover. Anything absent is measured by embedding a short probe string.
_KNOWN_DIMS: dict[str, int] = {
    "amazon.titan-embed-text-v2:0": 1024,
    "amazon.titan-embed-text-v1": 1536,
}

_PROBE = "engram dimension probe"


class EmbedderSpec(NamedTuple):
    """An embeddings backend together with its output dimensionality.

    The store's vector index has to be created with the right ``dims``, and not
    every backend advertises that as an attribute, so the factory resolves it
    once and passes the pair around.
    """

    embeddings: Embeddings
    dims: int


# Function words carry no signal about *what* a memory is about, and because
# they appear in nearly every sentence they otherwise dominate the vector: an
# unfiltered bag-of-words ranks "I had a great coffee at the place near the
# office" above "I prefer pytest and FastAPI" for the query "scaffold me a
# test", purely on shared articles and pronouns. Removing them is the single
# largest quality win available to a lexical embedder.
_STOPWORD_TEXT = """
    a an the and or but if then than that this these those there here
    i me my mine myself you your yours we us our ours they them their
    he him his she her hers it its
    is am are was were be been being do does did doing done
    have has had having will would shall should can could may might must
    of in on at to from for with by about into onto over under again
    as so just also very too much many more most some any all
    what which who whom whose when where why how
    not no nor only own same now
    """
_STOPWORDS: frozenset[str] = frozenset(_STOPWORD_TEXT.split())

# Whole words are the primary signal; character n-grams are a secondary one that
# lets morphological variants meet ("test" ~ "pytest" ~ "testing"), which a pure
# word model treats as unrelated strings. The weight keeps exact word matches
# dominant over incidental substring overlap.
_NGRAM = 3
_NGRAM_WEIGHT = 0.4


def _words(text: str) -> list[str]:
    """Lowercase alphanumeric word tokens, function words removed."""
    out: list[str] = []
    current: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            current.append(ch)
        elif current:
            out.append("".join(current))
            current = []
    if current:
        out.append("".join(current))
    return [w for w in out if w not in _STOPWORDS]


def _features(text: str) -> list[tuple[str, float]]:
    """Weighted features for one string: whole words plus character n-grams."""
    feats: list[tuple[str, float]] = []
    for word in _words(text):
        feats.append((word, 1.0))
        padded = f"#{word}#"
        for i in range(len(padded) - _NGRAM + 1):
            feats.append((padded[i : i + _NGRAM], _NGRAM_WEIGHT))
    return feats


class HashingEmbeddings(Embeddings):
    """Deterministic hashed lexical vectors over content words and n-grams.

    Uses BLAKE2b rather than :func:`hash` so that vectors are stable across
    processes — Python salts string hashing per interpreter, which would make
    stored vectors incomparable with query vectors from a later run.

    These are lexical vectors, not semantic ones: they match on shared words and
    subwords, so they will never connect "Berlin" to "Germany's capital". That
    is a deliberate floor, not an oversight. It keeps CI hermetic and makes the
    eval a real regression gate, and it means semantic wins measured later are
    attributable to the memory manager rather than to a better embedding model.
    Switch to ``fastembed`` when you want true paraphrase matching.
    """

    def __init__(self, dims: int = 256) -> None:
        if dims < 1:
            raise ValueError("dims must be positive")
        self.dims = dims

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for token, weight in _features(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            slot = int.from_bytes(digest[:4], "big") % self.dims
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[slot] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            # An empty or punctuation-only string. Return a fixed unit vector so
            # cosine distance stays defined instead of producing NaN.
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class FastEmbedEmbeddings(Embeddings):
    """Local ONNX sentence embeddings via ``fastembed``.

    The model downloads once (~130MB for bge-small) and is cached on disk; every
    run after that is offline. Imported lazily so the package stays installable
    without the ``embed`` extra.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised by hand
            raise ImportError(
                "The 'fastembed' embedder needs the optional extra: pip install -e \".[embed]\""
            ) from exc
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embedder(settings: Settings) -> EmbedderSpec:
    """Construct the configured embeddings backend and resolve its dimensions."""
    if settings.embedder == "hashing":
        return EmbedderSpec(
            HashingEmbeddings(dims=settings.hashing_embed_dims),
            settings.hashing_embed_dims,
        )

    if settings.embedder == "fastembed":
        embeddings: Embeddings = FastEmbedEmbeddings(settings.fastembed_model)
        return EmbedderSpec(embeddings, len(embeddings.embed_query(_PROBE)))

    if settings.embedder == "bedrock":
        from langchain_aws import BedrockEmbeddings

        embeddings = BedrockEmbeddings(
            model_id=settings.bedrock_embed_model_id,
            region_name=settings.aws_region,
        )
        dims = _KNOWN_DIMS.get(settings.bedrock_embed_model_id)
        return EmbedderSpec(embeddings, dims or len(embeddings.embed_query(_PROBE)))

    raise ValueError(f"Unknown embedder: {settings.embedder!r}")
