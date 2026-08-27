"""Builds `verify_verdict.json` for the LTM key-routing alias fix via `tools.verdict.Verdict` (an EARNED
verdict, not a hand-typed one) -- runs the same three checks as `verify_store_level.py` / `verify_e2e.py` /
`verify_moat_synthetic.py` live, through the real `ShardedPhasorStore` / `ChatBrain.gate` path, and records
every precondition that earned the GO.

Run:
  SIM_BACKEND=numpy .venv/bin/python research/findings/raw/_ltm_key_routing_alias_fix/build_verdict.py
"""
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_LTM_SHIP_DEFAULT", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))))
sys.path.insert(0, _REPO)

from tools.verdict import Verdict  # noqa: E402
from research.runners.sharded_phasor_store import ShardedPhasorStore  # noqa: E402
from research.runners.brain_chat_tui import ChatBrain, StubRenderer, _build_tiny_demo  # noqa: E402
from research.runners.developed_brain_io import _inner_agent  # noqa: E402
from research.runners.tiered_fact_store import TieredFactStore  # noqa: E402

BUNDLE = os.path.expanduser("~/Projects/sim-data/knowledge_bundles/wikidata_core_15k")


def _require_abstain(v, name, actual):
    """`Verdict.require`'s OWN sentinel for 'never measured' is a measured value of `None` (see
    tools/verdict.py's `_UNMEASURED` handling) -- which collides with an ABSTAIN check, where `None` is the
    CORRECT, actually-measured outcome we are asserting. Route around the collision: measure the boolean
    `actual is None` instead, so an abstain that genuinely happened records `ok=True`, not 'never measured'."""
    v.require(name, actual is None, expect=True, note="raw return value was %r" % (actual,))


def main():
    v = Verdict("LTM key-routing alias fallback (_portal/_core residual)")

    ltm = ShardedPhasorStore.load(BUNDLE)

    # -- bare-form retrieval now succeeds for every known-suffixed entity family --
    v.require("bare 'canada' retrieves canada_portal's member_of fact",
              ltm.query_patient("canada", "member_of"), expect="united_nations")
    v.require("bare 'berlin' retrieves berlin_portal's country fact",
              ltm.query_patient("berlin", "country"), expect="federal_republic_of_germany")
    v.require("bare 'dorset' retrieves dorset_portal's country fact",
              ltm.query_patient("dorset", "country"), expect="united_kingom")
    v.require("bare 'ska' retrieves ska_core's instance_of fact (_core suffix)",
              ltm.query_patient("ska", "instance_of"), expect="genre_of_music")

    # -- regression: an already-working, non-suffixed lookup is untouched --
    v.require("chelsea_fc|country unaffected (regression)",
              ltm.query_patient("chelsea_fc", "country"), expect="united_kingom")
    v.require("direct canada_portal|member_of unaffected (regression)",
              ltm.query_patient("canada_portal", "member_of"), expect="united_nations")

    # -- moat: nonexistent entity / nonexistent relation still abstain --
    _require_abstain(v, "nonexistent entity still abstains",
                     ltm.query_patient("definitely_not_real_xyz", "country"))
    _require_abstain(v, "nonexistent relation on a real suffix-keyed entity still abstains",
                     ltm.query_patient("canada", "definitely_not_a_real_relation_xyz"))

    # -- adversarial synthetic moat: ambiguous strip + shadowing (not present in the real bundle) --
    syn = ShardedPhasorStore(n_shards=4, seed=1, D=32,
                             vocab=["foo", "foo_portal", "foo_core", "bar", "bar_portal", "x", "y", "z",
                                    "rel1", "rel2"])
    syn.store("foo_portal", "rel1", "x")
    syn.store("foo_core", "rel1", "y")
    idx_a = syn.build_alias_index()
    v.require("ambiguous strip ('foo_portal'+'foo_core' -> 'foo') left unresolved",
              "foo" in idx_a, expect=False)
    _require_abstain(v, "ambiguous bare form abstains rather than guessing",
                     syn.query_patient("foo", "rel1"))

    syn.store("bar", "rel2", "z")
    syn.store("bar_portal", "rel2", "y")
    idx_b = syn.build_alias_index(force=True)
    v.require("real entity 'bar' never shadowed by 'bar_portal' in the alias index",
              "bar" in idx_b, expect=False)
    v.require("real entity 'bar' resolves to its OWN fact, not the shadowed one",
              syn.query_patient("bar", "rel2"), expect="z")

    # -- full live-chat pipeline (comprehension -> consensus -> LTM retrieval), the real webapp call graph --
    agent, aliases, _ = _build_tiny_demo(42, use_multiturn=True, enable_neural_render=False,
                                         composer_kind="onebrain")
    inner = _inner_agent(agent)
    inner.composer = TieredFactStore(inner.composer, ltm)
    chat = ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())
    v.require("full pipeline: 'what country is berlin in' commits",
              chat.gate("what country is berlin in"),
              expect=["berlin", "country", "federal_republic_of_germany"])
    v.require("full pipeline: chelsea_fc question unaffected (regression)",
              chat.gate("what country is chelsea fc from"),
              expect=["chelsea_fc", "country", "united_kingom"])
    _require_abstain(v, "full pipeline: nonexistent entity still abstains",
                     chat.gate("what country is definitely not real xyz in"))

    decided = v.decide(go=(len(v.unmet) == 0 and len(v.unmeasured) == 0))
    out_path = os.path.join(_HERE, "verify_verdict.json")
    with open(out_path, "w") as f:
        json.dump({"runner": "research/findings/raw/_ltm_key_routing_alias_fix/build_verdict.py",
                   "mechanism": "sharded-ltm-key-routing-alias-fallback",
                   "bundle": BUNDLE, "sim_backend": "numpy", **decided}, f, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
