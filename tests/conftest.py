"""Shared pytest fixtures for the Agency Swarm × MongoDB acceptance suite.

Non-search tests run on `mongomock` (no infra needed). Because the store now **always
owns its own client** (Option B — connection-string only), the offline fixture patches
``MongoClient`` in the store module with a mongomock-backed shim that tolerates the
``appname``/``driver`` kwargs the store always sets. The optional Atlas-backed handshake
check is skipped automatically when ATLAS_URI is unset.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / "demo" / ".env")
except ImportError:
    pass


def make_mongomock_factory():
    """Return a ``MongoClient`` replacement backed by mongomock.

    mongomock's client ignores (and rejects) the real driver's ``appname``/``driver``
    kwargs, so we strip them before delegating.
    """
    import mongomock

    def factory(*args, **kwargs):
        kwargs.pop("appname", None)
        kwargs.pop("appName", None)
        kwargs.pop("driver", None)
        return mongomock.MongoClient(*args, **kwargs)

    return factory


@pytest.fixture()
def mock_store(monkeypatch):
    """A MongoThreadStore backed by mongomock (works offline).

    The store builds its own client; we patch ``MongoClient`` in the store module so
    that client is a mongomock instance.
    """
    import agency_swarm_mongodb.store as store_mod
    from agency_swarm_mongodb import MongoThreadStore

    monkeypatch.setattr(store_mod, "MongoClient", make_mongomock_factory())
    store = MongoThreadStore("mongodb://localhost:27017", database_name="test_db")
    yield store


@pytest.fixture()
def atlas_uri() -> str | None:
    return os.environ.get("ATLAS_URI")
