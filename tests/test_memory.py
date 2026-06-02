"""Acceptance tests for MongoMemoryStore (semantic / episodic vector memory).

Non-Atlas tests run on `mongomock` (no infra). `$vectorSearch` itself is not executed
by mongomock, so we assert on:
- write behaviour (BYO vector stored verbatim; auto_embed rejects client vectors),
- episodic recency recall + scope isolation,
- the index-definition shape for both embedding paths,
- the embedding-path guardrails on recall (no silent fallback).

A live `$vectorSearch` round-trip auto-skips unless ATLAS_URI is set.
"""

from __future__ import annotations

import os

import pytest

from agency_swarm_mongodb import (
    APP_NAME,
    DEFAULT_AUTO_EMBED_MODEL,
    DRIVER_NAME,
    MongoMemoryStore,
)


@pytest.fixture()
def mem_store(monkeypatch):
    """A bring-your-own-vector MongoMemoryStore backed by mongomock."""
    from conftest import make_mongomock_factory

    import agency_swarm_mongodb.memory as memory_mod

    monkeypatch.setattr(memory_mod, "MongoClient", make_mongomock_factory())
    return MongoMemoryStore("mongodb://localhost:27017", database_name="test_mem")


@pytest.fixture()
def auto_store(monkeypatch):
    """An Atlas-Automated-Embedding MongoMemoryStore backed by mongomock."""
    from conftest import make_mongomock_factory

    import agency_swarm_mongodb.memory as memory_mod

    monkeypatch.setattr(memory_mod, "MongoClient", make_mongomock_factory())
    return MongoMemoryStore(
        "mongodb://localhost:27017", database_name="test_mem_auto", auto_embed=True
    )


# Criterion 1 — BYO vector is stored verbatim.
def test_byo_vector_stored(mem_store):
    vec = [0.1, 0.2, 0.3]
    _id = mem_store.add_memory("user-1", "user", "Likes window seats.",
                               kind="semantic", embedding=vec)
    doc = mem_store.memories.find_one({"_id": __import__("bson").ObjectId(_id)})
    assert doc["embedding"] == vec
    assert doc["kind"] == "semantic"
    assert doc["scope"] == "user-1"


# Criterion 2 — auto_embed rejects a client-supplied vector on write.
def test_auto_embed_rejects_client_vector(auto_store):
    with pytest.raises(ValueError, match="auto_embed=True manages embeddings"):
        auto_store.add_memory("user-1", "user", "hi", embedding=[0.1, 0.2])


# Criterion 3 — episodic recency recall returns chronological order.
def test_episodic_recency(mem_store):
    for i in range(5):
        mem_store.add_memory("user-2", "user", f"turn {i}", kind="episodic")
    recent = mem_store.get_recent("user-2", n=3, kind="episodic")
    assert [d["content"] for d in recent] == ["turn 2", "turn 3", "turn 4"]


# Criterion 4 — scope isolation.
def test_scope_isolation(mem_store):
    mem_store.add_memory("scope-A", "user", "a", kind="episodic")
    mem_store.add_memory("scope-B", "user", "b", kind="episodic")
    assert len(mem_store.get_recent("scope-A")) == 1
    assert len(mem_store.get_recent("scope-B")) == 1


# Criterion 5 — BYO index definition uses a `vector` field + scope/kind filters.
def test_byo_index_definition(mem_store):
    defn = mem_store.vector_index_definition(num_dimensions=1024)
    types = [f["type"] for f in defn["fields"]]
    assert "vector" in types
    vec_field = next(f for f in defn["fields"] if f["type"] == "vector")
    assert vec_field["path"] == "embedding"
    assert vec_field["numDimensions"] == 1024
    assert {"type": "filter", "path": "scope"} in defn["fields"]
    assert {"type": "filter", "path": "kind"} in defn["fields"]


# Criterion 5b — BYO index requires num_dimensions.
def test_byo_index_requires_dims(mem_store):
    with pytest.raises(ValueError, match="num_dimensions is required"):
        mem_store.vector_index_definition()


# Criterion 6 — auto_embed index definition uses an `autoEmbed` field + model.
def test_auto_embed_index_definition(auto_store):
    defn = auto_store.vector_index_definition()
    auto = next(f for f in defn["fields"] if f["type"] == "autoEmbed")
    assert auto["path"] == "content"
    assert auto["modality"] == "text"
    assert auto["model"] == DEFAULT_AUTO_EMBED_MODEL


# Criterion 7 — recall without a vector (and no auto_embed) raises (no silent fallback).
def test_recall_requires_vector(mem_store):
    with pytest.raises(ValueError, match="No `query_vector` provided"):
        mem_store.recall_semantic("user-1")


# Criterion 7b — auto_embed recall requires query text, not a vector.
def test_auto_recall_requires_text(auto_store):
    with pytest.raises(ValueError, match="expects `query` text"):
        auto_store.recall_semantic("user-1")
    with pytest.raises(ValueError, match="uses query text, not"):
        auto_store.recall_semantic("user-1", query_vector=[0.1, 0.2])


# Criterion 8 — as_save_hook captures the last turn into episodic memory.
def test_save_hook_captures_turn(mem_store):
    hook = mem_store.as_save_hook("user-hook")
    hook([
        {"role": "user", "content": "first", "agent": "A"},
        {"role": "assistant", "content": "second", "agent": "A"},
    ])
    recent = mem_store.get_recent("user-hook")
    assert len(recent) == 1
    assert recent[0]["content"] == "second"
    assert recent[0]["meta"]["source"] == "thread"


# Criterion 8b — save hook with embedder embeds the captured content.
def test_save_hook_with_embedder(mem_store):
    hook = mem_store.as_save_hook("user-emb", embedder=lambda t: [float(len(t))])
    hook([{"role": "assistant", "content": "hello", "agent": "A"}])
    recent = mem_store.get_recent("user-emb")
    assert recent[0]["embedding"] == [5.0]


# Criterion 9 — clear_scope removes all memories for a scope.
def test_clear_scope(mem_store):
    mem_store.add_memory("user-del", "user", "a", kind="episodic")
    mem_store.add_memory("user-del", "user", "b", kind="semantic", embedding=[0.1])
    assert mem_store.clear_scope("user-del") == 2
    assert mem_store.get_recent("user-del") == []


# Criterion 10 — tracking constants are well-formed (shared with the thread store).
def test_tracking_constants():
    assert APP_NAME == "devrel-integ-agencyswarm-python"
    assert DRIVER_NAME == "agency-swarm-mongodb"


# Criterion 10b — tracking is not overridable by the caller.
def test_tracking_not_overridable(monkeypatch):
    from conftest import make_mongomock_factory

    import agency_swarm_mongodb.memory as memory_mod

    captured = {}

    def factory(*args, **kwargs):
        captured.update(kwargs)
        return make_mongomock_factory()(*args, **kwargs)

    monkeypatch.setattr(memory_mod, "MongoClient", factory)
    MongoMemoryStore(
        "mongodb://localhost:27017",
        database_name="x",
        appname="evil-app",
        driver="not-a-driver",
    )
    assert captured.get("appname") == APP_NAME
    assert captured["driver"].name == DRIVER_NAME


# Criterion 11 (Atlas, optional) — live $vectorSearch round-trip with BYO vectors.
@pytest.mark.skipif(not os.environ.get("ATLAS_URI"), reason="ATLAS_URI not set")
def test_live_vector_recall(atlas_uri):
    store = MongoMemoryStore(atlas_uri, database_name="agency_swarm_mem_test")
    try:
        # NOTE: requires a pre-built vector index named idx_agent_memory.
        store.add_memory("it-scope", "user", "Window seats preferred.",
                         kind="semantic", embedding=[0.0] * 1024)
        hits = store.recall_semantic("it-scope", query_vector=[0.0] * 1024, k=1)
        assert isinstance(hits, list)
    finally:
        store.clear_scope("it-scope")
        store.close()
