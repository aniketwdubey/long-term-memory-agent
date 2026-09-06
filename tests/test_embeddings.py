"""The deterministic embedder — the one CI's numbers depend on."""

from __future__ import annotations

import subprocess
import sys

import pytest

from engram.config import Settings
from engram.embeddings import HashingEmbeddings, build_embedder


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.fixture
def embedder() -> HashingEmbeddings:
    return HashingEmbeddings(dims=256)


def test_is_deterministic_across_processes(embedder: HashingEmbeddings) -> None:
    """The property the whole regression gate rests on.

    Python salts string hashing per interpreter, so an embedder built on the
    builtin hash() would produce vectors that differ between the run that stored
    a memory and the run that queries it. This asserts against a *separate*
    process, which is the only way to catch that.
    """
    code = (
        "from engram.embeddings import HashingEmbeddings;"
        "print(HashingEmbeddings(256).embed_query('I prefer pytest')[:5])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == str(embedder.embed_query("I prefer pytest")[:5])


def test_vectors_are_unit_length(embedder: HashingEmbeddings) -> None:
    vec = embedder.embed_query("I'm on the payments team")
    assert dot(vec, vec) == pytest.approx(1.0)


def test_handles_text_with_no_content_words(embedder: HashingEmbeddings) -> None:
    """Must stay a defined unit vector rather than dividing by a zero norm."""
    for text in ("", "...", "the and of"):
        vec = embedder.embed_query(text)
        assert len(vec) == 256
        assert dot(vec, vec) == pytest.approx(1.0)


def test_ranks_a_relevant_memory_above_unrelated_chatter(
    embedder: HashingEmbeddings,
) -> None:
    """Regression test for stopword domination.

    A bag-of-words that keeps function words scores the coffee anecdote higher
    than the relevant fact, purely on shared articles and pronouns — which
    silently turns the whole benchmark into noise.
    """
    query = embedder.embed_query("Scaffold me a test for the refund endpoint")
    relevant = embedder.embed_query("I prefer pytest for everything I write")
    chatter = embedder.embed_query("I had a great coffee this morning near the office")
    assert dot(query, relevant) > dot(query, chatter)


def test_subword_matching_connects_morphological_variants(
    embedder: HashingEmbeddings,
) -> None:
    """ "test" and "pytest" share no whole word; n-grams are what bridges them."""
    query = embedder.embed_query("write a test")
    assert dot(query, embedder.embed_query("pytest")) > 0.0


def test_embed_documents_matches_embed_query(embedder: HashingEmbeddings) -> None:
    assert embedder.embed_documents(["a b", "c d"]) == [
        embedder.embed_query("a b"),
        embedder.embed_query("c d"),
    ]


def test_rejects_nonsense_dimensions() -> None:
    with pytest.raises(ValueError):
        HashingEmbeddings(dims=0)


def test_factory_reports_matching_dims() -> None:
    spec = build_embedder(Settings(_env_file=None, embedder="hashing", hashing_embed_dims=64))  # type: ignore[call-arg]
    assert spec.dims == 64
    assert len(spec.embeddings.embed_query("hello")) == 64


def test_factory_rejects_unknown_backend() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="Unknown embedder"):
        build_embedder(settings.model_copy(update={"embedder": "nope"}))
