"""
tests/test_query_rewriter.py
------------------------------
Query rewriting with a fake chat model: symbol normalization before the
model call, cleaning of the rewrites, modes, failure handling, and the
sensitivity of the model used.
"""

import pytest
from langchain_core.documents import Document as LCDocument

from services import llm_factory
from services.query_rewriter import QueryRewrites, corpus_languages, rewrite_query
from services.vector_store import add_document_chunks
from tests.test_leak import RecordingChat


class RewriteChat(RecordingChat):
    """Records prompts and returns a fixed QueryRewrites (or raises)."""

    result: object = None

    def with_structured_output(self, schema, **kwargs):
        from langchain_core.runnables import RunnableLambda

        def run(value):
            # Ollama's structured() passes a message list, the others a PromptValue.
            self._record(value.to_messages() if hasattr(value, "to_messages") else list(value))
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

        return RunnableLambda(run)


@pytest.fixture
def models(monkeypatch):
    external = RewriteChat(provider="openai", result=QueryRewrites(expanded="x", translations=[]))
    local = RewriteChat(provider="ollama", result=QueryRewrites(expanded="x", translations=[]))
    monkeypatch.setitem(llm_factory.CHAT_BUILDERS, "openai", lambda *a: external)
    monkeypatch.setitem(llm_factory.CHAT_BUILDERS, "ollama", lambda *a: local)
    monkeypatch.setattr(llm_factory.settings, "deployment_mode", "hybride")
    llm_factory.reset_cache()
    yield external, local
    llm_factory.reset_cache()


def test_symbol_is_normalized_before_the_model_call(models):
    external, _ = models
    external.result = QueryRewrites(expanded="Vote on resolution A/RES/77/276", translations=[])
    rewrites = rewrite_query("vote sur la résolution 77-276", ["en", "fr"])

    assert rewrites[0] == "vote sur la résolution 77-276 (A/RES/77/276)"
    assert "A/RES/77/276" in external.log[0]  # the model saw the official symbol


def test_symbol_rewrite_survives_model_failure(models):
    external, _ = models
    external.result = TimeoutError("slow")
    assert rewrite_query("vote sur A/80/L65", ["fr"]) == ["vote sur A/80/L65 (A/80/L.65)"]


def test_model_failure_without_symbol_gives_nothing(models):
    external, _ = models
    external.result = ValueError("broken")
    assert rewrite_query("position de la France sur les CDN", ["fr"]) == []


def test_rewrites_are_cleaned_and_capped(models):
    external, _ = models
    external.result = QueryRewrites(
        expanded="Position of France on nationally determined contributions",
        translations=["", "position de la France sur les CDN", "Posición de Francia", "Position Frankreichs",
                      "Posizione della Francia"],
    )
    rewrites = rewrite_query("position de la France sur les CDN", ["fr", "en", "es", "de", "it"])
    # empty and identical-to-question dropped, then at most 2 translations
    assert rewrites == [
        "Position of France on nationally determined contributions",
        "Posición de Francia",
        "Position Frankreichs",
    ]


def test_at_most_three_rewrites(models):
    external, _ = models
    external.result = QueryRewrites(expanded="Expanded with A/RES/77/276", translations=["t1", "t2"])
    rewrites = rewrite_query("vote on resolution 77-276", ["en", "fr", "es"])
    assert rewrites == ["vote on resolution 77-276 (A/RES/77/276)", "Expanded with A/RES/77/276", "t1"]


def test_no_lang_mode_drops_translations(models):
    external, _ = models
    external.result = QueryRewrites(expanded="Expanded question", translations=["Question traduite"])
    assert rewrite_query("the question", ["en", "fr"], modes={"expand"}) == ["Expanded question"]
    assert "always return an empty list" in external.log[0]


def test_failed_restricted_context_is_rewritten_locally(models):
    external, local = models
    secret = LCDocument(page_content="Valmont secret instruction", metadata={"classification": "DR_SIMULE"})
    rewrite_query("what is the instruction", ["fr"], failed_context=[secret])
    assert external.log == [] and "Valmont secret instruction" in local.log[0]


def test_corpus_languages(fake_embeddings):
    meta = {"document_name": "a.pdf", "page": 1, "page_end": 1}
    add_document_chunks("owner", [
        LCDocument(page_content="texte", metadata={**meta, "document_id": "a", "chunk_index": 0, "language": "fr"}),
        LCDocument(page_content="text", metadata={**meta, "document_id": "b", "chunk_index": 0, "language": "en"}),
        LCDocument(page_content="none", metadata={**meta, "document_id": "c", "chunk_index": 0}),
    ])
    assert corpus_languages("owner") == ["en", "fr"]
