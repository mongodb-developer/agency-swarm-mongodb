"""MongoDB-backed persistence + vector memory for Agency Swarm.

Two complementary surfaces, both backed by MongoDB / Atlas:

- :class:`MongoThreadStore` — drop-in ``load_threads_callback`` /
  ``save_threads_callback`` for the ``Agency`` class (session/thread persistence).
- :class:`MongoMemoryStore` — semantic / episodic long-term memory with Atlas Vector
  Search recall. Embedding source-agnostic: bring your own query vector (default) or
  enable Atlas Automated Embedding (server-side embeddings, no client code).

Example (threads)::

    from agency_swarm import Agency, Agent
    from agency_swarm_mongodb import MongoThreadStore

    store = MongoThreadStore("mongodb://localhost:27017")
    load_cb, save_cb = store.as_callbacks("user-123")

    agency = Agency(
        Agent(name="Assistant", instructions="..."),
        load_threads_callback=load_cb,
        save_threads_callback=save_cb,
    )

Example (memory — bring your own vector)::

    from agency_swarm_mongodb import MongoMemoryStore

    mem = MongoMemoryStore("mongodb+srv://...")
    mem.add_memory("user-123", "user", "Prefers window seats.",
                   kind="semantic", embedding=my_provider.embed(text))
    hits = mem.recall_semantic("user-123", query_vector=my_provider.embed("seating?"))

Example (memory — Atlas Automated Embedding)::

    mem = MongoMemoryStore("mongodb+srv://...", auto_embed=True)
    mem.ensure_vector_index()                       # builds an autoEmbed index
    mem.add_memory("user-123", "user", "Prefers window seats.", kind="semantic")
    hits = mem.recall_semantic("user-123", query="seating preferences")
"""

from __future__ import annotations

__version__ = "0.1.1"

from .memory import DEFAULT_AUTO_EMBED_MODEL, MongoMemoryStore
from .store import APP_NAME, DRIVER_NAME, MongoThreadStore

__all__ = [
    "MongoThreadStore",
    "MongoMemoryStore",
    "DEFAULT_AUTO_EMBED_MODEL",
    "APP_NAME",
    "DRIVER_NAME",
    "__version__",
]
