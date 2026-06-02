"""MongoDB-backed thread/session persistence for Agency Swarm.

Drop-in ``load_threads_callback`` / ``save_threads_callback`` for the
``Agency`` class, backed by MongoDB / Atlas.

Example::

    from agency_swarm import Agency, Agent
    from agency_swarm_mongodb import MongoThreadStore

    store = MongoThreadStore("mongodb://localhost:27017")
    load_cb, save_cb = store.as_callbacks("user-123")

    agency = Agency(
        Agent(name="Assistant", instructions="..."),
        load_threads_callback=load_cb,
        save_threads_callback=save_cb,
    )
"""

from __future__ import annotations

__version__ = "0.1.0"

from .store import APP_NAME, DRIVER_NAME, MongoThreadStore

__all__ = ["MongoThreadStore", "APP_NAME", "DRIVER_NAME", "__version__"]
