# agency-swarm-mongodb

MongoDB Atlas–backed thread/session persistence for [VRSEN Agency Swarm](https://github.com/VRSEN/agency-swarm).

Drop-in `load_threads_callback` / `save_threads_callback` for the `Agency` class — persist
entire conversations to MongoDB and restore them across application restarts.

## Why

Agency Swarm persists conversations through two `Agency` hooks but ships no database
backend (only a file-based example). `MongoThreadStore` is that backend: one document per
`chat_id`, idempotent upserts, optional TTL expiry — backed by MongoDB or Atlas.

## Install

```bash
pip install agency-swarm-mongodb
```

## Usage

```python
from agency_swarm import Agency, Agent
from agency_swarm_mongodb import MongoThreadStore

store = MongoThreadStore("mongodb+srv://...", database_name="agency_swarm")
load_cb, save_cb = store.as_callbacks("user-123")   # chat_id captured in the closures

agency = Agency(
    Agent(name="Assistant", instructions="You are helpful."),
    load_threads_callback=load_cb,
    save_threads_callback=save_cb,
)
```

The store matches the Agency Swarm callback signatures exactly:
`load_threads_callback() -> list[dict]` and `save_threads_callback(messages: list[dict]) -> None`.

### Options

| Arg | Default | Purpose |
|---|---|---|
| `connection_string` | — | MongoDB / Atlas URI (required unless `client` given) |
| `database_name` | `agency_swarm` | Database name |
| `collection_name` | `threads` | Collection name |
| `ttl_seconds` | `None` | If set, TTL index on `updated_at` auto-expires idle chats |
| `client` | `None` | Bring your own `MongoClient` (then appName is not overwritten) |

### Document shape

```jsonc
{
  "_id": "user-123",          // chat_id
  "messages": [ /* full flat list, exactly as Agency Swarm emits */ ],
  "message_count": 12,
  "updated_at": { "$date": "..." }
}
```

## Demos

- **`demo/custom_persistence_mongo.py`** — Mongo-backed mirror of Agency Swarm's
  `custom_persistence.py`: run a turn, simulate a restart, verify recall.
- **`demo/agent_demo.py`** — a Gemini agent whose threads persist to **Atlas**, plus an
  **Atlas Vector Search** staffing tool over a team directory (Voyage 3.5 embeddings).

```bash
pip install -e ".[demo]"
pip install "openai-agents[litellm]" "litellm[proxy]"
# demo/.env: ATLAS_URI, VOYAGE_API_KEY, GEMINI_API_KEY
python demo/agent_demo.py
```

## Conventions

- Connection `appName`: `devrel-integ-agencyswarm-python` (server-side attribution).
- Driver handshake metadata: `agency-swarm-mongodb` (distinct from appName).

## Tests

```bash
pip install -e ".[dev]"
pytest -q          # 9 tests, mongomock — no infra required
```

## License

MIT
