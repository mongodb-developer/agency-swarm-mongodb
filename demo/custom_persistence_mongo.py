"""Agency Swarm persistence with MongoDB — runnable demo.

Mongo-backed mirror of Agency Swarm's ``examples/custom_persistence.py``. It runs a
conversation, simulates an application restart by building a *new* ``Agency`` that shares
the same ``chat_id``, and verifies the agent recalls earlier facts from MongoDB.

Prereqs:
    pip install -e ".[demo]"               # installs agency-swarm + pymongo
    export OPENAI_API_KEY=...              # required by agency-swarm
    export MONGODB_URI=mongodb://localhost:27017   # or your Atlas URI

Run:
    python demo/custom_persistence_mongo.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Make the local src/ importable without an install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agency_swarm import Agency, Agent, ModelSettings  # noqa: E402

from agency_swarm_mongodb import MongoThreadStore  # noqa: E402

MONGODB_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
CHAT_ID = "demo_session"
TEST_INFO = "blue and lucky number is 77"


def build_agent() -> Agent:
    return Agent(
        name="AssistantAgent",
        instructions="You are a helpful assistant. Answer questions and help users with their tasks.",
        tools=[],
        model_settings=ModelSettings(temperature=0.0),
    )


async def main() -> None:
    print("\n=== Agency Swarm × MongoDB Persistence Demo ===")
    store = MongoThreadStore(MONGODB_URI, database_name="agency_swarm_demo")

    # Start clean for a repeatable demo.
    store.clear(CHAT_ID)

    load_cb, save_cb = store.as_callbacks(CHAT_ID)

    # --- Turn 1: tell the agent something to remember ---
    agency = Agency(
        build_agent(),
        shared_instructions="Be helpful and concise in your responses.",
        load_threads_callback=load_cb,
        save_threads_callback=save_cb,
    )

    msg1 = f"Hello. Please remember that my favorite color is {TEST_INFO}. I'll ask later."
    print(f"\n--- Turn 1 ---\nUser: {msg1}")
    resp1 = await agency.get_response(message=msg1)
    print(f"Assistant: {resp1.final_output}")

    stored = store.load_threads(CHAT_ID)
    print(f"\n[MongoDB] persisted {len(stored)} messages under chat_id={CHAT_ID!r}")

    # --- Simulate application restart: brand-new Agency, same chat_id ---
    print("\n--- Simulating application restart (new Agency, same Mongo chat_id) ---")
    load_cb2, save_cb2 = store.as_callbacks(CHAT_ID)
    agency_reloaded = Agency(
        build_agent(),
        shared_instructions="Be helpful and concise in your responses.",
        load_threads_callback=load_cb2,
        save_threads_callback=save_cb2,
    )

    msg2 = "What was my favorite color and lucky number I told you earlier?"
    print(f"\n--- Turn 2 ---\nUser: {msg2}")
    resp2 = await agency_reloaded.get_response(message=msg2)
    print(f"Assistant (reloaded): {resp2.final_output}")

    out = (resp2.final_output or "").lower()
    if "blue" in out and "77" in out:
        print(f"\n✅ SUCCESS: agent recalled '{TEST_INFO}' from MongoDB after restart.")
    else:
        print(f"\n❌ FAILURE: agent did not recall '{TEST_INFO}'.")

    store.close()


if __name__ == "__main__":
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
