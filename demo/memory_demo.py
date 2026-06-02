"""Agent demo: an Agency Swarm (Gemini) agent with MongoDB Atlas vector memory.

Shows long-term agent memory (added in 0.1.1) wired into a real LLM agent via two
function tools backed by :class:`MongoMemoryStore`:

- ``remember`` — the agent stores a durable semantic fact about the user.
- ``recall``   — the agent runs Atlas ``$vectorSearch`` to retrieve relevant facts.

The point: memory survives across *separate* Agency sessions because it lives in Atlas,
not in the conversation thread. Session 1 teaches facts; Session 2 is a brand-new Agency
(no shared thread) that answers only by calling ``recall``.

Embedding paths (the package never embeds for you):
- **default (bring-your-own):** queries/facts are embedded here with the Voyage SDK and
  passed to the store as vectors.
- **MEMORY_MODE=auto:** ``MongoMemoryStore(auto_embed=True)`` — Atlas embeds server-side;
  the tools pass query *text* instead of vectors (no VOYAGE_API_KEY needed client-side).

Setup (demo/.env is auto-loaded):
    ATLAS_URI=mongodb+srv://...
    VOYAGE_API_KEY=...        # only for the default bring-your-own-vector mode
    GEMINI_API_KEY=...        # powers the Gemini model (litellm gemini provider)

Install:
    pip install -e ".[demo]" voyageai
    pip install "openai-agents[litellm]" "litellm[proxy]"

Run:
    python demo/memory_demo.py                    # bring-your-own Voyage vectors
    MEMORY_MODE=auto python demo/memory_demo.py   # Atlas Automated Embedding
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# litellm gemini provider reads GEMINI_API_KEY / GOOGLE_API_KEY.
if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

warnings.filterwarnings("ignore", category=DeprecationWarning)

from agency_swarm import Agency, Agent, ModelSettings, function_tool  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from agency_swarm_mongodb import MongoMemoryStore  # noqa: E402

DEMO_DB = "agencyswarm_mem_demo"
SCOPE = "user-memory-1"
MODE = os.environ.get("MEMORY_MODE", "byo").lower()  # "byo" | "auto"
AUTO = MODE == "auto"
MODEL = os.environ.get("GEMINI_MODEL", "litellm/gemini/gemini-2.5-flash")
VOYAGE_MODEL = "voyage-3.5"  # bring-your-own path; 1024-dim

# Module-level store handle so the function tools can reach it.
_MEM: MongoMemoryStore | None = None


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def embed(text: str, *, input_type: str) -> list[float]:
    """Embed with the Voyage SDK in *your* code (bring-your-own-vector path)."""
    import voyageai

    client = voyageai.Client()  # reads VOYAGE_API_KEY
    return client.embed([text], model=VOYAGE_MODEL, input_type=input_type).embeddings[0]


class RememberArgs(BaseModel):
    fact: str = Field(..., description="A durable fact/preference about the user to store.")


class RecallArgs(BaseModel):
    query: str = Field(..., description="What to look up in the user's long-term memory.")


@function_tool
def remember(args: RememberArgs) -> str:
    """Store a durable semantic fact about the user in MongoDB Atlas memory."""
    assert _MEM is not None
    if AUTO:
        _MEM.add_memory(SCOPE, "user", args.fact, kind="semantic")
    else:
        _MEM.add_memory(SCOPE, "user", args.fact, kind="semantic",
                        embedding=embed(args.fact, input_type="document"))
    return f"Stored: {args.fact}"


@function_tool
def recall(args: RecallArgs) -> str:
    """Retrieve relevant user facts from MongoDB Atlas via $vectorSearch."""
    assert _MEM is not None
    if AUTO:
        hits = _MEM.recall_semantic(SCOPE, query=args.query, k=3)
    else:
        hits = _MEM.recall_semantic(SCOPE, query_vector=embed(args.query, input_type="query"), k=3)
    if not hits:
        return "No relevant memories found."
    return "\n".join(f"- {h['content']} (score={h.get('score', 0):.3f})" for h in hits)


def build_agent() -> Agent:
    return Agent(
        name="MemoryAssistant",
        instructions=(
            "You are a personal assistant with long-term memory backed by MongoDB. "
            "When the user shares a durable preference or fact about themselves, call "
            "`remember` to store it. When the user asks something that may depend on "
            "what you know about them, call `recall` first and answer from the results. "
            "Keep answers to 1-3 sentences."
        ),
        tools=[remember, recall],
        model=MODEL,
        model_settings=ModelSettings(temperature=0.0),
    )


async def main() -> None:
    global _MEM

    uri = os.environ.get("ATLAS_URI")
    needs = [uri, os.environ.get("GEMINI_API_KEY")] + ([] if AUTO else [os.environ.get("VOYAGE_API_KEY")])
    if not all(needs):
        extra = "" if AUTO else " + VOYAGE_API_KEY"
        print(f"This demo needs ATLAS_URI + GEMINI_API_KEY{extra} (see demo/.env).")
        sys.exit(1)

    print(f"=== Agency Swarm × MongoDB Vector Memory (agentic demo, mode={MODE}) ===")
    _MEM = MongoMemoryStore(uri, database_name=DEMO_DB, auto_embed=AUTO)
    _MEM.clear_scope(SCOPE)  # clean slate for a repeatable demo

    print("Ensuring vector index (first build can take ~1 min)...")
    ok = _MEM.ensure_vector_index() if AUTO else _MEM.ensure_vector_index(num_dimensions=1024)
    if not ok:
        print("Vector index did not become queryable in time.")
        sys.exit(1)
    print("Index queryable.")

    # --- Session 1: the agent learns and stores facts ---
    banner("SESSION 1 — agent stores durable facts in Atlas memory (via `remember`)")
    agency = Agency(build_agent(), shared_instructions="Be helpful and concise.")
    for msg in [
        "Remember that I'm vegetarian and I avoid dairy.",
        "Also, I always prefer window seats on flights.",
    ]:
        print(f"User: {msg}")
        r = await agency.get_response(message=msg)
        print(f"Agent: {r.final_output}")

    # Atlas indexing is async — wait until the stored facts are searchable.
    print("\nWaiting for Atlas to index the new memories...")
    for _ in range(20):
        probe = (
            _MEM.recall_semantic(SCOPE, query="food", k=1)
            if AUTO
            else _MEM.recall_semantic(SCOPE, query_vector=embed("food", input_type="query"), k=1)
        )
        if probe:
            break
        time.sleep(2)
    print("Memories searchable.")

    # --- Session 2: a brand-new Agency (no shared thread) recalls from Atlas ---
    banner("SESSION 2 — fresh Agency (no shared thread) answers via `recall` from Atlas")
    agency2 = Agency(build_agent(), shared_instructions="Be helpful and concise.")
    q = "I'm booking a long flight and ordering a meal. What should I pick for me?"
    print(f"User: {q}")
    r2 = await agency2.get_response(message=q)
    print(f"Agent (new session): {r2.final_output}")

    banner("Proof: direct $vectorSearch recall over stored memories")
    hits = (
        _MEM.recall_semantic(SCOPE, query="dietary and seating preferences", k=3)
        if AUTO
        else _MEM.recall_semantic(
            SCOPE, query_vector=embed("dietary and seating preferences", input_type="query"), k=3
        )
    )
    for h in hits:
        print(f"  • [{h.get('score', 0):.3f}] {h['content']}")

    _MEM.clear_scope(SCOPE)
    _MEM.close()
    print("\nMemory demo complete — Gemini stored + recalled facts via MongoDB Atlas Vector Search.")


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
