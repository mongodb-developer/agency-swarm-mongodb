# agency-swarm-mongodb

> ⚠️ **ALPHA — NOT AN OFFICIAL MONGODB PRODUCT.** This integration is in **Alpha** and is **not** a supported or official MongoDB product. **Use at your own risk.**

MongoDB Atlas–backed persistence **and vector memory** for [VRSEN Agency Swarm](https://github.com/VRSEN/agency-swarm).

- **`MongoThreadStore`** — drop-in `load_threads_callback` / `save_threads_callback` for the
  `Agency` class: persist entire conversations to MongoDB and restore them across restarts.
- **`MongoMemoryStore`** *(new in 0.1.1)* — semantic / episodic long-term memory with Atlas
  Vector Search recall. Embedding source-agnostic: **bring your own query vector** (default)
  or enable **Atlas Automated Embedding** (server-side embeddings, no client code).

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

## Vector memory (`MongoMemoryStore`, new in 0.1.1)

Semantic / episodic long-term memory with Atlas Vector Search recall. The package never
calls an embedding provider itself — you choose one of **two first-class paths**:

**1. Bring your own vector (default).** Embed with whatever provider you already use
(OpenAI, Voyage SDK, Cohere, your agency's embedding client, …) and pass the vectors:

```python
from agency_swarm_mongodb import MongoMemoryStore

mem = MongoMemoryStore("mongodb+srv://...")
mem.ensure_vector_index(num_dimensions=1024)        # one-time, on Atlas

mem.add_memory("user-123", "user", "Prefers window seats.",
               kind="semantic", embedding=my_provider.embed(text))

hits = mem.recall_semantic("user-123",
                           query_vector=my_provider.embed("seating?"), k=5)
```

**2. Atlas Automated Embedding.** Atlas generates embeddings server-side (no client
embedding code); recall sends query *text*:

```python
mem = MongoMemoryStore("mongodb+srv://...", auto_embed=True)   # default model: voyage-4
mem.ensure_vector_index()                            # builds an `autoEmbed` index

mem.add_memory("user-123", "user", "Prefers window seats.", kind="semantic")
hits = mem.recall_semantic("user-123", query="seating preferences", k=5)
```

There is **no silent fallback**: if you neither pass a `query_vector` nor enable
`auto_embed`, recall raises `ValueError`. Episodic recency recall needs no vectors:

```python
mem.add_memory("user-123", "assistant", "Booked the morning flight.", kind="episodic")
recent = mem.get_recent("user-123", n=10, kind="episodic")   # chronological
```

`as_save_hook(scope)` returns a `save_threads_callback`-compatible hook to mirror turns
into episodic memory from an `Agency`.

### Memory options

| Arg | Default | Purpose |
|---|---|---|
| `connection_string` | — | MongoDB / Atlas URI (required) |
| `database_name` | `agency_swarm` | Database name |
| `collection_name` | `memories` | Collection name |
| `vector_search_index` | `idx_agent_memory` | Atlas Vector Search index name |
| `auto_embed` | `False` | Enable Atlas Automated Embedding (recall by query text) |
| `auto_embed_model` | `voyage-4` | Voyage model used by Automated Embedding |
| `ttl_seconds` | `None` | If set, TTL index on `ts` auto-expires old memories |

### Memory document shape

```jsonc
{
  "scope": "user-123",        // tenant / conversation / agent scope
  "kind": "semantic",         // "episodic" | "semantic"
  "role": "user",
  "content": "Prefers window seats.",
  "embedding": [ /* present only on the bring-your-own-vector path */ ],
  "ts": { "$date": "..." },
  "meta": {}
}
```

## Demos

- **`demo/custom_persistence_mongo.py`** — Mongo-backed mirror of Agency Swarm's
  `custom_persistence.py`: run a turn, simulate a restart, verify recall.
- **`demo/agent_demo.py`** — a Gemini agent whose threads persist to **Atlas**, plus an
  **Atlas Vector Search** staffing tool over a team directory (Voyage 3.5 embeddings).
- **`demo/memory_demo.py`** — `MongoMemoryStore` semantic + episodic recall on **Atlas**,
  runnable in both modes: bring-your-own Voyage vectors (default) or
  `MEMORY_MODE=auto` for Atlas Automated Embedding.

```bash
pip install -e ".[demo]"
pip install "openai-agents[litellm]" "litellm[proxy]"
# demo/.env: ATLAS_URI, VOYAGE_API_KEY, GEMINI_API_KEY
python demo/agent_demo.py
python demo/memory_demo.py                 # bring-your-own vectors
MEMORY_MODE=auto python demo/memory_demo.py  # Atlas Automated Embedding
```


## Conventions

- Connection `appName`: `devrel-integ-agencyswarm-python` (server-side attribution).
- Driver handshake metadata: `agency-swarm-mongodb` (distinct from appName).

## Tests

```bash
pip install -e ".[dev]"
pytest -q          # 25 tests, mongomock — no infra required

```

## License

MIT
