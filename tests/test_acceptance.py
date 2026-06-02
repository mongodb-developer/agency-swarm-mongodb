"""Acceptance tests for the Agency Swarm × MongoDB Session/Thread Store.

Each test maps to a numbered criterion in PLAN.md (Phase 2). Non-Atlas tests run on
`mongomock` and need no infrastructure.
"""

from __future__ import annotations

import os

import pytest

from agency_swarm_mongodb import APP_NAME, DRIVER_NAME, MongoThreadStore



def _sample_messages() -> list[dict]:
    return [
        {
            "role": "user",
            "content": "Hello, remember my color is blue.",
            "agent": "AssistantAgent",
            "callerAgent": None,
            "timestamp": 1780379726123,
        },
        {
            "role": "assistant",
            "content": "Got it — blue.",
            "agent": "AssistantAgent",
            "callerAgent": None,
            "timestamp": 1780379726456,
        },
    ]


# Criterion 1 — Round-trip preserves order + all metadata fields.
def test_round_trip_preserves_messages(mock_store):
    msgs = _sample_messages()
    mock_store.save_threads("chat-1", msgs)
    loaded = mock_store.load_threads("chat-1")

    assert loaded == msgs
    assert [m["role"] for m in loaded] == ["user", "assistant"]
    assert loaded[0]["agent"] == "AssistantAgent"
    assert loaded[0]["callerAgent"] is None
    assert loaded[0]["timestamp"] == 1780379726123


# Criterion 2 — Idempotent upsert: one doc, latest write wins.
def test_idempotent_upsert(mock_store):
    mock_store.save_threads("chat-1", _sample_messages())
    mock_store.save_threads("chat-1", _sample_messages())  # save again

    assert mock_store.threads.count_documents({"_id": "chat-1"}) == 1

    extended = _sample_messages() + [
        {"role": "user", "content": "and lucky number 77", "agent": "AssistantAgent",
         "callerAgent": None, "timestamp": 1780379727000}
    ]
    mock_store.save_threads("chat-1", extended)
    loaded = mock_store.load_threads("chat-1")
    assert len(loaded) == 3
    assert mock_store.threads.count_documents({"_id": "chat-1"}) == 1
    doc = mock_store.threads.find_one({"_id": "chat-1"})
    assert doc["message_count"] == 3


# Criterion 3 — Scope isolation between chat_ids.
def test_scope_isolation(mock_store):
    mock_store.save_threads("chat-A", _sample_messages())
    mock_store.save_threads("chat-B", [{"role": "user", "content": "different", "agent": "X",
                                        "callerAgent": None, "timestamp": 1}])

    a = mock_store.load_threads("chat-A")
    b = mock_store.load_threads("chat-B")
    assert len(a) == 2
    assert len(b) == 1
    assert a != b


# Criterion 4 — Unknown chat_id returns [] (not None/error).
def test_empty_load_returns_empty_list(mock_store):
    assert mock_store.load_threads("never-seen") == []


# Criterion 5 — as_callbacks returns signatures matching Agency Swarm.
def test_as_callbacks_signatures(mock_store):
    import inspect

    load_cb, save_cb = mock_store.as_callbacks("chat-cb")

    # load callback is zero-arg and returns [] initially
    assert len(inspect.signature(load_cb).parameters) == 0
    assert load_cb() == []

    # save callback takes exactly one positional arg (messages)
    assert len(inspect.signature(save_cb).parameters) == 1

    msgs = _sample_messages()
    save_cb(msgs)
    assert load_cb() == msgs


# Criterion 6 — TTL index created when ttl_seconds is set.
def test_ttl_index_created(monkeypatch):
    from conftest import make_mongomock_factory

    import agency_swarm_mongodb.store as store_mod

    monkeypatch.setattr(store_mod, "MongoClient", make_mongomock_factory())
    store = MongoThreadStore(
        "mongodb://localhost:27017", database_name="ttl_db", ttl_seconds=3600
    )
    indexes = store.threads.index_information()

    ttl = [v for v in indexes.values() if v.get("expireAfterSeconds") == 3600]
    assert ttl, f"expected a TTL index with expireAfterSeconds=3600, got {indexes}"


# Criterion 7a — appName + driver-info constants are well-formed (100 Integs convention).
def test_appname_and_driver_constants():
    assert APP_NAME == "devrel-integ-agencyswarm-python"
    assert DRIVER_NAME == "agency-swarm-mongodb"


# Criterion 7d — tracking is not overridable by the caller.
def test_tracking_not_overridable(monkeypatch):
    from conftest import make_mongomock_factory

    import agency_swarm_mongodb.store as store_mod

    captured = {}

    def factory(*args, **kwargs):
        captured.update(kwargs)
        return make_mongomock_factory()(*args, **kwargs)

    monkeypatch.setattr(store_mod, "MongoClient", factory)
    # Caller tries to sneak in their own appname/driver — must be ignored.
    MongoThreadStore(
        "mongodb://localhost:27017",
        database_name="x",
        appname="evil-app",
        appName="evil-app",
        driver="not-a-driver",
    )
    assert captured.get("appname") == APP_NAME
    assert captured.get("driver") is not None
    assert captured["driver"].name == DRIVER_NAME


# Criterion 7b — clear() removes the conversation document.

def test_clear_removes_document(mock_store):
    mock_store.save_threads("chat-del", _sample_messages())
    assert mock_store.load_threads("chat-del")
    deleted = mock_store.clear("chat-del")
    assert deleted == 1
    assert mock_store.load_threads("chat-del") == []


# Criterion 7c (Atlas, optional) — real client carries appName + driver_info handshake.
@pytest.mark.skipif(not os.environ.get("ATLAS_URI"), reason="ATLAS_URI not set")
def test_real_client_handshake_metadata(atlas_uri):
    store = MongoThreadStore(atlas_uri, database_name="agency_swarm_test")
    try:
        opts = store.client.options
        assert opts.pool_options.metadata["application"]["name"] == APP_NAME
        # driver_info name is appended to the driver metadata
        driver_name = opts.pool_options.metadata["driver"]["name"]
        assert DRIVER_NAME in driver_name
    finally:
        store.clear("chat-handshake-noop")
        store.close()
