# AGENTS.md — guide for AI coding agents

A structured guide for AI agents working in `agency-swarm-mongodb`: how to build and test,
where key files live, and the MongoDB-specific rules to follow.

## Build and test commands

```bash
# Install (editable) + dev deps
pip install -e ".[dev]"

# Run the test suite (mongomock — no infra required)
pytest -q                # 25 tests

# Demos (need Atlas + keys; see demo/.env: ATLAS_URI, VOYAGE_API_KEY, GEMINI_API_KEY)
pip install -e ".[demo]"
pip install "openai-agents[litellm]" "litellm[proxy]"
python demo/custom_persistence_mongo.py
python demo/agent_demo.py
python demo/memory_demo.py                  # bring-your-own Voyage vectors
MEMORY_MODE=auto python demo/memory_demo.py # Atlas Automated Embedding
```

## Project structure

- `src/agency_swarm_mongodb/store.py` — `MongoThreadStore` (thread/session persistence).
- `src/agency_swarm_mongodb/memory.py` — `MongoMemoryStore` (semantic/episodic vector memory).
- `src/agency_swarm_mongodb/__init__.py` — public exports + `__version__`.
- `tests/` — acceptance tests (`test_acceptance.py`, `test_memory.py`).
- `demo/` — runnable demos over Atlas + the team-member dataset.
- `EDD.md` — the MongoDB data model (source of truth for schema).
- `PLAN.md` — the 7-phase integration plan.

## Environment variables and configuration

| Name | Required | Description |
|---|---|---|
| `ATLAS_URI` | demos / vector tests | Atlas connection string |
| `VOYAGE_API_KEY` | bring-your-own embedding demos | Voyage AI key for `voyage-3.5` |
| `GEMINI_API_KEY` | agent demos | Gemini model key |
| `MEMORY_MODE` | optional | `auto` to use Atlas Automated Embedding in `memory_demo.py` |

## Conventions (do not break)

- The package **owns its `MongoClient`** — built from a connection string. `appName`
  (`devrel-integ-agencyswarm-python`) and the `agency-swarm-mongodb` driver-info handshake
  are always set and **non-overridable** (caller `appname`/`appName`/`driver` are stripped).
- Embeddings use **Voyage AI 3.5** (`voyage-3.5`, 1024-dim) on the bring-your-own path.
- No silent embedding fallback: `recall_semantic` raises if neither `query_vector` nor
  `auto_embed` is provided.

## MongoDB Skills

Use the official MongoDB agent skills from https://github.com/mongodb/agent-skills
whenever the task is MongoDB-specific and a matching skill exists.

## When To Use EDD.md

Use [EDD.md](./EDD.md) as the source of truth for the MongoDB data model in this repository.

Consult [EDD.md](./EDD.md) before making changes that touch:

- The `threads` or `memories` collections, document structure, or field names
- Code paths that read or write database records (`store.py`, `memory.py`)
- Index definitions (TTL on `updated_at`/`ts`, the Atlas Vector Search index)
- Validation, payloads, or anything that depends on persisted data
- Schema documentation, Mermaid diagrams, or entity modeling discussions
