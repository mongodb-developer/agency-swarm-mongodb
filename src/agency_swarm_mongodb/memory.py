"""Semantic / episodic vector memory for Agency Swarm, backed by MongoDB / Atlas.

This module adds long-term agent memory (MS) and vector recall (VS) alongside the
thread/session store in :mod:`agency_swarm_mongodb.store`. It stays **embedding
source-agnostic** — the package itself never calls an embedding provider. Two
first-class paths are supported:

1. **Bring-your-own vector (default).** You produce embeddings with whatever provider
   you already use (OpenAI, Voyage SDK, Cohere, your agency's embedding client, ...)
   and pass the resulting ``list[float]`` to :meth:`MongoMemoryStore.add_memory`
   (``embedding=...``) and :meth:`recall_semantic` (``query_vector=...``). The store
   only stores and queries the vectors via ``$vectorSearch``.

2. **Atlas Automated Embedding.** Construct the store with ``auto_embed=True``. The
   Atlas Vector Search index is created with the ``autoEmbed`` type so Atlas generates
   embeddings server-side at index-time (from ``content``) and at query-time (from the
   ``query`` text). No client-side embedding code is involved.

There is intentionally **no hidden Voyage fallback** in the package: if you neither
pass a vector nor enable ``auto_embed``, recall raises ``ValueError`` instead of
silently calling a provider.

Memory kinds:
- ``"episodic"`` — time-ordered conversation turns / events (recency recall).
- ``"semantic"`` — distilled facts / knowledge (vector recall).

Conventions applied here (100 Integs — baked in, non-overridable):
- ``appName = devrel-integ-agencyswarm-python`` so server telemetry attributes traffic.
- ``driver_info`` handshake metadata identifies the ``agency-swarm-mongodb`` library.
- The store **always constructs and owns its own** ``MongoClient`` from a connection
  string, so these are guaranteed present on every connection with no caller opt-out.
"""

from __future__ import annotations

import datetime as _dt
from typing import TYPE_CHECKING, Any, Callable, Literal

from pymongo import ASCENDING, MongoClient
from pymongo.driver_info import DriverInfo

from .store import APP_NAME, DRIVER_NAME, _PKG_VERSION

if TYPE_CHECKING:
    from pymongo.collection import Collection
    from pymongo.database import Database

# Default Atlas Automated Embedding model (the recommended general-text model).
DEFAULT_AUTO_EMBED_MODEL = "voyage-4"

MemoryKind = Literal["episodic", "semantic"]


class MongoMemoryStore:
    """Semantic / episodic vector memory backed by MongoDB / Atlas.

    Each memory is one document in the ``memories`` collection::

        {
          "scope":   "user-123",          # tenant / conversation / agent scope
          "kind":    "semantic",          # "episodic" | "semantic"
          "role":    "user",              # who produced it (free-form)
          "content": "Prefers window seats.",
          "embedding": [ ... ],            # present only on the BYO-vector path
          "ts":      ISODate(...),
          "meta":    { ... }
        }

    The store **always constructs and owns its own** :class:`MongoClient` from the
    supplied ``connection_string``; ``appName`` and ``driver_info`` are baked in and
    cannot be overridden.

    Args:
        connection_string: MongoDB / Atlas connection URI. **Required.**
        database_name: Database to use. Default ``"agency_swarm"``.
        collection_name: Collection for memory documents. Default ``"memories"``.
        vector_search_index: Name of the Atlas Vector Search index. Default
            ``"idx_agent_memory"``.
        auto_embed: When True, enables the **Atlas Automated Embedding** path —
            :meth:`ensure_vector_index` builds an ``autoEmbed`` index and recall sends
            query *text* instead of a vector. When False (default), you must supply
            vectors yourself (bring-your-own).
        auto_embed_model: Voyage AI model name used by Atlas Automated Embedding
            (only relevant when ``auto_embed=True``). Default ``"voyage-4"``.
        ttl_seconds: If set, a TTL index on ``ts`` auto-expires old memories.
        **client_kwargs: Extra keyword arguments forwarded to :class:`MongoClient`.
            ``appname``/``appName``/``driver`` are reserved and overridden.
    """

    def __init__(
        self,
        connection_string: str,
        *,
        database_name: str = "agency_swarm",
        collection_name: str = "memories",
        vector_search_index: str = "idx_agent_memory",
        auto_embed: bool = False,
        auto_embed_model: str = DEFAULT_AUTO_EMBED_MODEL,
        ttl_seconds: int | None = None,
        **client_kwargs: Any,
    ) -> None:
        if not connection_string:
            raise ValueError("connection_string is required")

        self.vector_search_index = vector_search_index
        self.auto_embed = auto_embed
        self.auto_embed_model = auto_embed_model
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
        self.memories: Collection = self.db[collection_name]

        self._ensure_indexes()

    # -- setup -----------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        """Create the compound scope/kind/ts index and the optional TTL index."""
        self.memories.create_index(
            [("scope", ASCENDING), ("kind", ASCENDING), ("ts", ASCENDING)],
            name="scope_kind_ts",
        )
        if self.ttl_seconds is not None:
            self.memories.create_index(
                [("ts", ASCENDING)],
                name="ttl_ts",
                expireAfterSeconds=self.ttl_seconds,
            )

    # -- Atlas Vector Search index ---------------------------------------------

    def vector_index_definition(
        self,
        *,
        num_dimensions: int | None = None,
        similarity: str = "cosine",
        filter_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return the Atlas Vector Search index ``definition`` for this store.

        Shape depends on the embedding path:

        - ``auto_embed=True`` → an ``autoEmbed`` field on ``content`` (Atlas embeds
          server-side using ``auto_embed_model``).
        - ``auto_embed=False`` → a classic ``vector`` field on ``embedding`` (requires
          ``num_dimensions``, since you supply the vectors).

        ``scope`` is always added as a ``filter`` path, plus any extra ``filter_paths``.
        """
        fields: list[dict[str, Any]]
        if self.auto_embed:
            fields = [
                {
                    "type": "autoEmbed",
                    "modality": "text",
                    "path": "content",
                    "model": self.auto_embed_model,
                }
            ]
        else:
            if not num_dimensions:
                raise ValueError(
                    "num_dimensions is required for the bring-your-own-vector index "
                    "(set it to your embedding model's dimensionality, e.g. 1024)."
                )
            fields = [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": num_dimensions,
                    "similarity": similarity,
                }
            ]
        fields.append({"type": "filter", "path": "scope"})
        fields.append({"type": "filter", "path": "kind"})
        for path in filter_paths or []:
            fields.append({"type": "filter", "path": path})
        return {"fields": fields}

    def ensure_vector_index(
        self,
        *,
        num_dimensions: int | None = None,
        similarity: str = "cosine",
        filter_paths: list[str] | None = None,
        wait: bool = True,
        timeout: int = 180,
    ) -> bool:
        """Create the Atlas Vector Search index if missing. Returns True once queryable.

        Requires an Atlas cluster (``createSearchIndexes`` is unsupported on local
        MongoDB / mongomock). No-op-safe if the index already exists.
        """
        from pymongo.operations import SearchIndexModel

        existing = {idx["name"] for idx in self.memories.list_search_indexes()}
        if self.vector_search_index not in existing:
            model = SearchIndexModel(
                definition=self.vector_index_definition(
                    num_dimensions=num_dimensions,
                    similarity=similarity,
                    filter_paths=filter_paths,
                ),
                name=self.vector_search_index,
                type="vectorSearch",
            )
            self.memories.create_search_index(model)

        if not wait:
            return False

        import time

        deadline = time.time() + timeout
        while time.time() < deadline:
            for idx in self.memories.list_search_indexes():
                if idx["name"] == self.vector_search_index and idx.get("queryable"):
                    return True
            time.sleep(3)
        return False

    # -- writes ----------------------------------------------------------------

    def add_memory(
        self,
        scope: str,
        role: str,
        content: str,
        *,
        kind: MemoryKind = "episodic",
        embedding: list[float] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> str:
        """Insert one memory. Returns the inserted ``_id`` as a string.

        Embedding behaviour:

        - **Bring-your-own:** pass ``embedding=<list[float]>`` to store a vector you
          produced with your own provider. Stored verbatim under ``embedding``.
        - **Atlas Automated Embedding** (``auto_embed=True``): do **not** pass
          ``embedding`` — Atlas generates it from ``content`` at index-time. Passing a
          vector here is a misconfiguration and raises ``ValueError``.
        - Episodic-only turns can omit a vector entirely (recency recall via
          :meth:`get_recent`).
        """
        if self.auto_embed and embedding is not None:
            raise ValueError(
                "auto_embed=True manages embeddings server-side; do not pass `embedding`. "
                "Atlas generates it from `content`."
            )

        doc: dict[str, Any] = {
            "scope": scope,
            "kind": kind,
            "role": role,
            "content": content,
            "ts": _dt.datetime.now(_dt.timezone.utc),
            "meta": meta or {},
        }
        if embedding is not None:
            doc["embedding"] = embedding
        result = self.memories.insert_one(doc)
        return str(result.inserted_id)

    # -- recall ----------------------------------------------------------------

    def get_recent(
        self,
        scope: str,
        n: int = 20,
        *,
        kind: MemoryKind | None = None,
    ) -> list[dict[str, Any]]:
        """Return the most recent ``n`` memories for ``scope`` (chronological order).

        Optionally restrict to a single ``kind``. Sorted by ``(ts, _id)`` so inserts
        within the same millisecond keep a stable, monotonic order.
        """
        query: dict[str, Any] = {"scope": scope}
        if kind is not None:
            query["kind"] = kind
        cursor = self.memories.find(query).sort([("ts", -1), ("_id", -1)]).limit(n)
        docs = list(cursor)
        docs.reverse()  # chronological (oldest → newest)
        return docs

    def recall_semantic(
        self,
        scope: str,
        *,
        query: str | None = None,
        query_vector: list[float] | None = None,
        k: int = 5,
        kind: MemoryKind | None = "semantic",
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic recall over ``scope`` via Atlas ``$vectorSearch``.

        Two mutually exclusive query modes:

        - **Bring-your-own vector (default):** pass ``query_vector=<list[float]>``. The
          stage uses ``queryVector`` against the stored ``embedding`` field.
        - **Atlas Automated Embedding** (``auto_embed=True``): pass ``query=<text>``.
          The stage uses the ``query`` (text) option and Atlas embeds it server-side
          with the same model declared on the index.

        The ``filter`` always pins ``scope`` (and ``kind`` when given), merging any
        extra ``filters`` for prefiltering. Raises ``ValueError`` if the supplied
        arguments don't match the configured embedding path — there is no silent
        provider fallback.
        """
        vector_stage: dict[str, Any] = {
            "index": self.vector_search_index,
            "path": "embedding",
            "numCandidates": max(k * 20, 100),
            "limit": k,
        }

        if self.auto_embed:
            if query_vector is not None:
                raise ValueError(
                    "auto_embed=True uses query text, not `query_vector`. "
                    "Pass `query=...` instead."
                )
            if query is None:
                raise ValueError(
                    "auto_embed=True expects `query` text (Atlas embeds it server-side); "
                    "got none."
                )
            vector_stage["query"] = query

        else:
            if query_vector is None:
                raise ValueError(
                    "No `query_vector` provided. Either pass a query vector produced by "
                    "your embedding provider, or construct the store with auto_embed=True "
                    "to let Atlas embed query text server-side."
                )
            vector_stage["queryVector"] = query_vector

        vs_filter: dict[str, Any] = {"scope": scope}
        if kind is not None:
            vs_filter["kind"] = kind
        if filters:
            vs_filter.update(filters)
        vector_stage["filter"] = vs_filter

        pipeline = [
            {"$vectorSearch": vector_stage},
            {
                "$project": {
                    "embedding": 0,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        return list(self.memories.aggregate(pipeline))

    # -- Agency Swarm integration ----------------------------------------------

    def as_save_hook(
        self,
        scope: str,
        *,
        kind: MemoryKind = "episodic",
        embedder: Callable[[str], list[float]] | None = None,
    ) -> Callable[[list[dict[str, Any]]], None]:
        """Return a ``save_threads_callback``-compatible hook that captures turns.

        Wrap or chain this with the :class:`~agency_swarm_mongodb.MongoThreadStore`
        save callback to mirror each turn into episodic memory. When ``embedder`` is
        provided (bring-your-own), the content is embedded before storage; otherwise
        text is stored as-is (use ``auto_embed=True`` for server-side embeddings).

        The callback is idempotent per call only in the trivial sense — it appends the
        newest turn each time it is invoked, matching Agency Swarm's full-history
        ``save_threads_callback(messages)`` contract by inspecting the last message.
        """

        def hook(messages: list[dict[str, Any]]) -> None:
            if not messages:
                return
            last = messages[-1]
            content = last.get("content")
            if not content:
                return
            embedding = embedder(content) if embedder is not None else None
            self.add_memory(
                scope,
                last.get("role", "assistant"),
                content,
                kind=kind,
                embedding=embedding,
                meta={"agent": last.get("agent"), "source": "thread"},
            )

        return hook

    # -- maintenance -----------------------------------------------------------

    def clear_scope(self, scope: str) -> int:
        """Delete all memories for ``scope``. Returns deleted count."""
        return self.memories.delete_many({"scope": scope}).deleted_count

    def close(self) -> None:
        """Close the underlying client (the store always owns it)."""
        self.client.close()


__all__ = ["MongoMemoryStore", "DEFAULT_AUTO_EMBED_MODEL", "APP_NAME", "DRIVER_NAME"]
