"""Atlas Vector Search helper for the Agency Swarm demo.

Loads the canonical 17-person team directory (`data/embeddings.json`, precomputed
Voyage 3.5 vectors) into a collection, builds a prefilterable Atlas Vector Search index,
and exposes `recall_semantic` for the agent's staffing tool. Query strings are embedded
at runtime with Voyage 3.5 (`voyage-3.5`, 1024-dim) per the 100 Integs conventions.

Kept in the demo (not the core package) so the published `agency-swarm-mongodb` library
stays a focused, dependency-light thread store.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from pymongo import MongoClient
from pymongo.driver_info import DriverInfo
from pymongo.operations import SearchIndexModel

from agency_swarm_mongodb import APP_NAME, DRIVER_NAME

VOYAGE_MODEL = "voyage-3.5"
VOYAGE_DIM = 1024
DIRECTORY_SCOPE = "corpus:team"


def embed_text(text: str, *, input_type: str = "document") -> list[float]:
    """Embed a single string with Voyage AI 3.5 → 1024-dim vector."""
    import voyageai

    client = voyageai.Client()  # reads VOYAGE_API_KEY
    return client.embed([text], model=VOYAGE_MODEL, input_type=input_type).embeddings[0]


class AtlasDirectory:
    """Employee directory backed by Atlas Vector Search (demo support, not core API)."""

    def __init__(self, uri: str, *, database_name: str, index_name: str = "idx_team_directory"):
        self.client = MongoClient(
            uri,
            appname=APP_NAME,
            driver=DriverInfo(name=DRIVER_NAME, version="0.1.0"),
        )
        self.col = self.client[database_name]["team_directory"]
        self.index_name = index_name

    def seed(self, dataset: Path) -> None:
        docs = json.loads(dataset.read_text())
        rows: list[dict[str, Any]] = []
        for d in docs:
            skills = ", ".join(
                s["name"] if isinstance(s, dict) else str(s) for s in d.get("skills", [])
            )
            rows.append(
                {
                    "scope": DIRECTORY_SCOPE,
                    "content": f"{d['name']} — {d['title']} ({d['department']}). Skills: {skills}.",
                    "embedding": d["embedding"],  # Voyage 3.5, precomputed
                    "meta": {
                        "name": d["name"],
                        "title": d["title"],
                        "department": d["department"],
                        "availability": d.get("availability"),
                    },
                }
            )
        self.col.delete_many({"scope": DIRECTORY_SCOPE})
        self.col.insert_many(rows)

    def ensure_index(self, *, timeout: int = 180) -> bool:
        existing = {idx["name"] for idx in self.col.list_search_indexes()}
        if self.index_name not in existing:
            model = SearchIndexModel(
                definition={
                    "fields": [
                        {"type": "vector", "path": "embedding",
                         "numDimensions": VOYAGE_DIM, "similarity": "cosine"},
                        {"type": "filter", "path": "scope"},
                        {"type": "filter", "path": "meta.department"},
                        {"type": "filter", "path": "meta.availability"},
                    ]
                },
                name=self.index_name,
                type="vectorSearch",
            )
            self.col.create_search_index(model)

        deadline = time.time() + timeout
        while time.time() < deadline:
            for idx in self.col.list_search_indexes():
                if idx["name"] == self.index_name and idx.get("queryable"):
                    return True
            time.sleep(3)
        return False

    def recall_semantic(
        self, query: str, k: int = 4, *, department: str | None = None
    ) -> list[dict[str, Any]]:
        vs_filter: dict[str, Any] = {"scope": DIRECTORY_SCOPE}
        if department:
            vs_filter["meta.department"] = department
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.index_name,
                    "path": "embedding",
                    "queryVector": embed_text(query, input_type="query"),
                    "numCandidates": max(k * 20, 100),
                    "limit": k,
                    "filter": vs_filter,
                }
            },
            {"$project": {"embedding": 0, "score": {"$meta": "vectorSearchScore"}}},
        ]
        return list(self.col.aggregate(pipeline))

    def close(self) -> None:
        self.client.close()
