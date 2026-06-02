"""MongoDB-backed thread/session store for Agency Swarm.

Agency Swarm persists conversations through two hooks on the ``Agency`` class:

- ``load_threads_callback: () -> list[dict]`` (zero-arg; the ``chat_id`` is captured
  via a closure)
- ``save_threads_callback: (messages: list[dict]) -> None`` (receives the full flat
  list of all messages for the conversation)

This module stores one document per ``chat_id`` and exposes :meth:`MongoThreadStore.as_callbacks`
which returns closures matching those signatures exactly.

Conventions applied here (100 Integs — baked in, non-overridable):
- ``appName = devrel-integ-agencyswarm-python`` so server telemetry attributes traffic.
- ``driver_info`` handshake metadata identifies the ``agency-swarm-mongodb`` library
  (distinct from appName; see the ``add-client-metadata`` convention).
- The store **always constructs and owns its own** ``MongoClient`` from a connection
  string, so these are guaranteed present on every connection with no caller opt-out.
"""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING, Any, Callable

from pymongo import ASCENDING, MongoClient
from pymongo.driver_info import DriverInfo

if TYPE_CHECKING:
    from pymongo.collection import Collection
    from pymongo.database import Database

APP_NAME = "devrel-integ-agencyswarm-python"
"""MongoDB connection appName for server-side attribution (100 Integs convention)."""

DRIVER_NAME = "agency-swarm-mongodb"
"""driver_info name attached to the MongoDB handshake (distinct from appName)."""

# Resolve the package version lazily so driver_info reports the installed version.
try:  # pragma: no cover - trivial
    from . import __version__ as _PKG_VERSION
except Exception:  # pragma: no cover
    _PKG_VERSION = "0.0.0"


# Type aliases for the Agency Swarm callback signatures.
Message = dict[str, Any]
LoadCallback = Callable[[], list[Message]]
SaveCallback = Callable[[list[Message]], None]


class MongoThreadStore:
    """Persistence layer for Agency Swarm conversations backed by MongoDB / Atlas.

    Each conversation is stored as a single document keyed by ``chat_id`` (used as
    ``_id``), holding the full flat ``messages`` list exactly as Agency Swarm emits it.
    Writes are idempotent upserts, so the adapter is safe to call repeatedly.

    The store **always constructs and owns its own** :class:`MongoClient` from the
    supplied ``connection_string``. This guarantees the 100 Integs telemetry conventions
    are baked in and cannot be bypassed: ``appName`` and ``driver_info`` are always set on
    the connection, with no caller opt-out. Any ``appname``/``appName``/``driver`` value a
    caller tries to pass via ``client_kwargs`` is ignored in favor of the convention values.

    Args:
        connection_string: MongoDB / Atlas connection URI. **Required.**
        database_name: Database to use. Default ``"agency_swarm"``.
        collection_name: Collection for thread documents. Default ``"threads"``.
        ttl_seconds: If set, a TTL index is created on ``updated_at`` to auto-expire
            idle conversations after the given number of seconds.
        **client_kwargs: Extra keyword arguments forwarded to :class:`MongoClient`
            (e.g. ``tls=True``, ``maxPoolSize=50``). ``appname``/``appName``/``driver``
            are reserved and will be overridden with the convention values.
    """

    def __init__(
        self,
        connection_string: str,
        *,
        database_name: str = "agency_swarm",
        collection_name: str = "threads",
        ttl_seconds: int | None = None,
        **client_kwargs: Any,
    ) -> None:
        if not connection_string:
            raise ValueError("connection_string is required")

        self.ttl_seconds = ttl_seconds

        driver_info = DriverInfo(name=DRIVER_NAME, version=_PKG_VERSION)

        # The integration owns the client. appName + driver_info are mandatory and
        # non-overridable: strip any caller-supplied values so tracking is always present.
        client_kwargs.pop("appname", None)
        client_kwargs.pop("appName", None)
        client_kwargs.pop("driver", None)

        self._owns_client = True
        self.client = MongoClient(
            connection_string,
            appname=APP_NAME,
            driver=driver_info,
            **client_kwargs,
        )

        self.db: Database = self.client[database_name]
        self.threads: Collection = self.db[collection_name]

        self._ensure_indexes()

    # -- setup -----------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        """Create the optional TTL index on ``updated_at``."""
        if self.ttl_seconds is not None:
            self.threads.create_index(
                [("updated_at", ASCENDING)],
                name="ttl_updated_at",
                expireAfterSeconds=self.ttl_seconds,
            )

    # -- core surface ----------------------------------------------------------

    def load_threads(self, chat_id: str) -> list[Message]:
        """Return the stored message list for ``chat_id`` (``[]`` if none)."""
        doc = self.threads.find_one({"_id": chat_id})
        if not doc:
            return []
        return doc.get("messages", [])

    def save_threads(self, chat_id: str, messages: list[Message]) -> None:
        """Idempotently upsert the full ``messages`` list for ``chat_id``."""
        self.threads.update_one(
            {"_id": chat_id},
            {
                "$set": {
                    "messages": messages,
                    "message_count": len(messages),
                    "updated_at": _dt.datetime.now(_dt.timezone.utc),
                }
            },
            upsert=True,
        )

    def as_callbacks(self, chat_id: str) -> tuple[LoadCallback, SaveCallback]:
        """Return ``(load_threads_callback, save_threads_callback)`` for ``chat_id``.

        The returned callables match the Agency Swarm signatures exactly:

        - ``load_threads_callback() -> list[dict]`` (zero-arg; ``chat_id`` captured)
        - ``save_threads_callback(messages: list[dict]) -> None``

        Use directly::

            load_cb, save_cb = store.as_callbacks("user-123")
            agency = Agency(agent, load_threads_callback=load_cb, save_threads_callback=save_cb)
        """

        def load_cb() -> list[Message]:
            return self.load_threads(chat_id)

        def save_cb(messages: list[Message]) -> None:
            self.save_threads(chat_id, messages)

        return load_cb, save_cb

    # -- maintenance -----------------------------------------------------------

    def clear(self, chat_id: str) -> int:
        """Delete the conversation document for ``chat_id``. Returns deleted count."""
        return self.threads.delete_one({"_id": chat_id}).deleted_count

    def close(self) -> None:
        """Close the underlying client (the store always owns it)."""
        self.client.close()


__all__ = ["MongoThreadStore", "APP_NAME", "DRIVER_NAME"]
