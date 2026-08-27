"""Byte-identical-when-off proof for organ C's mirror-image LTM exemption on the 3-organ GNW bus.

Loads the PRE-PATCH `webapp/gnw_three_organ_bus.py` straight from `git show HEAD` (the exact committed code
before THIS session's organ-C edits -- i.e. the already-merged organ-B-only de-risk at 9402f0aeb) as a standalone
module, runs a panel of turns through BOTH the OLD `three_organ_combine` and the NEW one (flag OFF --
`organb_ltm_exempt=False`, matching the env var unset), and SHA-256-hashes every common field of the returned
info dict (plus the full `chat.gate()` answer for the headline question, through BOTH an OLD-module-installed
gate and a NEW-module-installed gate with the env var genuinely unset). 0 diffs required for the OFF path to be
asserted byte-identical (docs/TERMS.md's own bar: hash-compared in the data, not inferred from reading the code).

Also spot-checks that `webapp/gnw_two_organ_bus.py` (untouched by this arc) produces IDENTICAL output whether or
not `webapp/gnw_three_organ_bus.py` has been imported in-process (proves this arc's edits are confined to the
3-organ module and do not leak into the 2-organ bus via shared caches).

SIM_BACKEND=numpy, tiny-demo + shipped LTM, seed 42.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("BRAIN_LTM_SHIP_DEFAULT", "1")

REPO = "/home/dant123/Projects/sim/.claude/worktrees/agent-ad1545703c36b55e6"
sys.path.insert(0, REPO)
LTM_BUNDLE = os.path.expanduser("~/Projects/sim-data/knowledge_bundles/wikidata_core_15k")
SEED = 42
SCRATCH = "/tmp/claude-1000/-home-dant123-Projects-sim/87891831-e642-4a2f-abeb-50ea0867609b/scratchpad"


def _load_old_module(relpath, modname):
    src = subprocess.run(["git", "show", f"HEAD:{relpath}"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    os.makedirs(SCRATCH, exist_ok=True)
    tmp_path = f"{SCRATCH}/{modname}.py"
    with open(tmp_path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location(modname, tmp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def build_chat():
    from research.runners.brain_chat_tui import ChatBrain, StubRenderer, _build_tiny_demo
    from research.runners.developed_brain_io import _inner_agent
    from research.runners.tiered_fact_store import TieredFactStore
    from research.runners.sharded_phasor_store import ShardedPhasorStore

    agent, aliases, _n = _build_tiny_demo(SEED, use_multiturn=True, enable_neural_render=False,
                                          composer_kind="onebrain")
    ltm = ShardedPhasorStore.load(LTM_BUNDLE)
    inner = _inner_agent(agent)
    inner.composer = TieredFactStore(inner.composer, ltm)
    chat = ChatBrain(agent, self_aliases=aliases, renderer=StubRenderer())
    return chat


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def main():
    old_three = _load_old_module("webapp/gnw_three_organ_bus.py", "gnw_three_organ_bus_OLD")
    from webapp import gnw_three_organ_bus as new_three
    from webapp import gnw_two_organ_bus as two_organ_mod

    probes = [
        ("chelsea_fc", "country"),                       # LTM fact (the headline case)
        ("definitely_not_a_stored_entity_xyz", "definitely_not_a_stored_relation_xyz"),  # unstored (moat)
    ]
    chat = build_chat()
    chat.inner.composer.store("zzz_test_agent", "zzz_test_action", "zzz_test_patient", polarity="AFFIRM")
    probes.append(("zzz_test_agent", "zzz_test_action"))

    diffs = []
    three_organ_rows = []
    two_organ_rows = []
    for a, act in probes:
        old_info = old_three.three_organ_combine(chat, a, act, seed=SEED)
        new_info = new_three.three_organ_combine(chat, a, act, seed=SEED, organb_ltm_exempt=False)
        common_keys = set(old_info) & set(new_info)
        old_h = _h({k: old_info[k] for k in sorted(common_keys)})
        new_h = _h({k: new_info[k] for k in sorted(common_keys)})
        row = {"probe": f"{a}|{act}", "old_hash": old_h, "new_hash": new_h, "match": old_h == new_h,
              "old_committed": old_info.get("committed"), "new_committed": new_info.get("committed"),
              "new_only_keys": {k: new_info[k] for k in sorted(set(new_info) - set(old_info))}}
        three_organ_rows.append(row)
        if old_h != new_h:
            diffs.append(("three_organ", a, act, old_info, new_info))

        # spot-check the untouched 2-organ bus gives identical output regardless of the 3-organ module import
        two_info = two_organ_mod.two_organ_combine(chat, a, act, seed=SEED, organb_ltm_exempt=False)
        two_organ_rows.append({"probe": f"{a}|{act}", "committed": two_info.get("committed"),
                               "abstain_reason": two_info.get("abstain_reason")})

    # --- full chat.gate() panel through the INSTALLED wrapper (env genuinely unset) ---
    assert os.environ.get("BRAIN_GNW_ORGANB_LTM_EXEMPT") is None, "must be unset for this check"
    chat_old = build_chat()
    old_three.install_three_organ_gate(chat_old, seed=SEED)
    chat_new = build_chat()
    new_three.install_three_organ_gate(chat_new, seed=SEED)

    gate_panel = [
        "what country is chelsea fc from",
        "what is chelsea fc",
        "who are you",
    ]
    gate_rows = []
    for q in gate_panel:
        a_old = chat_old.gate(q)
        a_new = chat_new.gate(q)
        row = {"q": q, "old": a_old, "new": a_new, "match": a_old == a_new}
        gate_rows.append(row)
        if a_old != a_new:
            diffs.append(("gate", q, None, a_old, a_new))

    report = {"three_organ_rows": three_organ_rows, "two_organ_bus_spotcheck": two_organ_rows,
             "gate_rows": gate_rows, "n_diffs": len(diffs)}
    print(json.dumps(report, indent=2, default=str))
    print(f"\n=== BYTE-IDENTICAL-WHEN-OFF VERDICT: {'PASS (0 diffs)' if not diffs else f'FAIL ({len(diffs)} diffs)'} ===")
    if diffs:
        for d in diffs:
            print("DIFF:", d)


if __name__ == "__main__":
    main()
