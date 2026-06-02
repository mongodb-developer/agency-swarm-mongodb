"""Agent demo: an Agency Swarm (Gemini) agent with MongoDB Atlas persistence + Vector Search.

Shows two things MongoDB gives an Agency Swarm agent:

1. **Persistent threads across sessions** — ``MongoThreadStore`` wires the
   ``load_threads_callback`` / ``save_threads_callback`` hooks so the *entire* conversation
   survives an application restart, stored under a ``chat_id`` in Atlas.
2. **Atlas Vector Search as an agent tool** — a ``find_team_members`` function tool runs
   ``$vectorSearch`` over the 17-person employee directory (Voyage 3.5 embeddings), with an
   optional department prefilter, so Gemini staffs projects from real data.

Flow:
  Session 1 — user states a durable constraint ("staff from Engineering only"). The whole
              thread is persisted to MongoDB.
  Session 2 — a brand-new Agency (same chat_id) restores the thread from MongoDB and answers
              a staffing ask by calling the Vector Search tool.

Setup (demo/.env is auto-loaded):
    ATLAS_URI=mongodb+srv://...
    VOYAGE_API_KEY=...                 # embeds queries with voyage-3.5
    GEMINI_API_KEY=...                 # powers the Gemini model (litellm gemini provider)

Install:
    pip install -e ".[demo]"
    pip install "openai-agents[litellm]" "litellm[proxy]"

Run:
    python demo/agent_demo.py
"""

from __future__ import annotations

import asyncio
import os
import sys
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

from agency_swarm_mongodb import MongoThreadStore  # noqa: E402
from _atlas_directory import AtlasDirectory  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET = REPO_ROOT / "data" / "embeddings.json"
DEMO_DB = "agencyswarm_demo"
CHAT_ID = "staffing-manager-7"
MODEL = os.environ.get("GEMINI_MODEL", "litellm/gemini/gemini-2.5-flash")

# Module-level directory handle so the function tool can reach it.
_DIRECTORY: AtlasDirectory | None = None


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


class FindArgs(BaseModel):
    need: str = Field(..., description="Natural-language description of the skills/role required.")
    department: str | None = Field(
        None, description="Optional department to prefilter to, e.g. 'Engineering'."
    )


@function_tool
def find_team_members(args: FindArgs) -> str:
    """Find suitable team members via Atlas Vector Search (with optional prefilter)."""
    assert _DIRECTORY is not None
    hits = _DIRECTORY.recall_semantic(args.need, k=4, department=args.department)
    note = f" [prefiltered: department={args.department}]" if args.department else " [no prefilter]"
    if not hits:
        return f"No matching team members found.{note}"
    body = "\n".join(
        f"{h['meta']['name']} — {h['meta']['title']} "
        f"({h['meta']['department']}, score={h.get('score', 0):.3f})"
        for h in hits
    )
    return body + note


def build_agent() -> Agent:
    return Agent(
        name="StaffingConcierge",
        instructions=(
            "You are a staffing concierge. Remember durable user preferences (like a required "
            "department) across the conversation. When asked to staff a project, ALWAYS call "
            "find_team_members, passing the user's preferred department if they stated one. "
            "Keep answers to 2-3 sentences and name the people."
        ),
        tools=[find_team_members],
        model=MODEL,
        model_settings=ModelSettings(temperature=0.0),
    )


async def main() -> None:
    global _DIRECTORY

    uri = os.environ.get("ATLAS_URI")
    if not (uri and os.environ.get("VOYAGE_API_KEY") and os.environ.get("GEMINI_API_KEY")):
        print("This demo needs ATLAS_URI + VOYAGE_API_KEY + GEMINI_API_KEY (see demo/.env).")
        sys.exit(1)

    # Atlas Vector Search directory (Voyage 3.5 vectors, prefilterable index).
    _DIRECTORY = AtlasDirectory(uri, database_name=DEMO_DB)
    print("Seeding employee directory + ensuring prefilterable vector index...")
    _DIRECTORY.seed(DATASET)
    if not _DIRECTORY.ensure_index():
        print("Vector index did not become queryable in time.")
        sys.exit(1)
    # Atlas indexing is async — poll until the new docs are searchable.
    import time as _t
    for _ in range(20):
        if _DIRECTORY.recall_semantic("engineer", k=1):
            break
        _t.sleep(2)
    print("Directory indexed and queryable.")

    # MongoDB thread store — the integration under test.
    store = MongoThreadStore(uri, database_name=DEMO_DB)
    store.clear(CHAT_ID)  # clean slate for a repeatable demo
    load_cb, save_cb = store.as_callbacks(CHAT_ID)

    banner("SESSION 1 — user sets a durable staffing constraint (persisted to MongoDB)")
    agency = Agency(
        build_agent(),
        shared_instructions="Be helpful and concise.",
        load_threads_callback=load_cb,
        save_threads_callback=save_cb,
    )
    q1 = "Going forward, only staff my projects with people from the Engineering department."
    print(f"User: {q1}")
    r1 = await agency.get_response(message=q1)
    print(f"Agent: {r1.final_output}")
    print(f"  [MongoDB] persisted {len(store.load_threads(CHAT_ID))} messages under chat_id={CHAT_ID!r}")

    banner("SESSION 2 — fresh Agency restores the thread from MongoDB + runs Vector Search")
    load_cb2, save_cb2 = store.as_callbacks(CHAT_ID)
    agency_reloaded = Agency(
        build_agent(),
        shared_instructions="Be helpful and concise.",
        load_threads_callback=load_cb2,
        save_threads_callback=save_cb2,
    )
    q2 = "I need someone who can lead a React frontend project. Who do you suggest?"
    print(f"User: {q2}")
    r2 = await agency_reloaded.get_response(message=q2)
    print(f"Agent (reloaded): {r2.final_output}")

    banner("Proof: vector search WITH vs WITHOUT the department prefilter")
    print("  Unfiltered top-4:")
    for h in _DIRECTORY.recall_semantic("lead a React frontend project", k=4):
        print(f"    • {h['meta']['name']} ({h['meta']['department']})")
    print("  Engineering-prefiltered top-4:")
    for h in _DIRECTORY.recall_semantic("lead a React frontend project", k=4, department="Engineering"):
        print(f"    • {h['meta']['name']} ({h['meta']['department']})")

    store.close()
    _DIRECTORY.close()
    print("\nAgent demo complete — Gemini used MongoDB-persisted threads + Atlas Vector Search.")


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
