"""Repro script for the LTM key-routing alias fix: FULL live-chat pipeline check (comprehension ->
consensus -> LTM retrieval) via `ChatBrain.gate`, the exact call `webapp/server.py`'s `/api/brain-chat`
handler uses (through `TieredFactStore` -> `ShardedPhasorStore`, unmocked, real shipped bundle).

Run (light path, numpy backend):
  SIM_BACKEND=numpy .venv/bin/python research/findings/raw/_ltm_key_routing_alias_fix/verify_e2e.py
"""
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_LTM_SHIP_DEFAULT", "1")

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, _REPO)

from research.runners.brain_chat_tui import ChatBrain, StubRenderer, _build_tiny_demo  # noqa: E402
from research.runners.developed_brain_io import _inner_agent  # noqa: E402
from research.runners.tiered_fact_store import TieredFactStore  # noqa: E402
from research.runners.sharded_phasor_store import ShardedPhasorStore  # noqa: E402

BUNDLE = os.path.expanduser("~/Projects/sim-data/knowledge_bundles/wikidata_core_15k")


def main():
    agent, aliases, _ = _build_tiny_demo(42, use_multiturn=True, enable_neural_render=False, composer_kind="onebrain")
    ltm = ShardedPhasorStore.load(BUNDLE)
    inner = _inner_agent(agent)
    inner.composer = TieredFactStore(inner.composer, ltm)
    chat = ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())

    print()
    print("=== FULL chat.gate() PIPELINE (comprehension + consensus + LTM retrieval) ===")

    q1 = "what country is berlin in"
    r1 = chat.gate(q1)
    print(f"chat.gate({q1!r}) = {r1!r}   (expect ['berlin', 'country', 'federal_republic_of_germany'])")

    q2 = "what country is chelsea fc from"
    r2 = chat.gate(q2)
    print(f"chat.gate({q2!r}) = {r2!r}   (expect ['chelsea_fc', 'country', 'united_kingom'], unaffected)")

    q3 = "what country is definitely not real xyz in"
    r3 = chat.gate(q3)
    print(f"chat.gate({q3!r}) = {r3!r}   (expect None -> abstain)")

    assert r1 == ["berlin", "country", "federal_republic_of_germany"], f"FAIL berlin e2e: {r1}"
    assert r2 == ["chelsea_fc", "country", "united_kingom"], f"FAIL chelsea_fc regression: {r2}"
    assert r3 is None, f"MOAT BREACH e2e: {r3}"
    print()
    print("ALL FULL-PIPELINE CHECKS PASSED")


if __name__ == "__main__":
    main()
