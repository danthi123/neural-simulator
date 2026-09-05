"""RANK-13 PRODUCTION-FLIP: a supplementary sanity check against the ACTUAL deployed brain bundle (not the
tiny-demo fixture both the de-risk and `_rank13_selfid_anaphora_prodflip_verify.py`'s formal 6-seed battery use).

WHY THIS EXISTS. `research/coordination/scaffold_retirement_backlog.md`'s rank-1 entry names
`bridges/developed/scale787/day_33` as "the deployed... bundle... the live composer for every recall/store/abstain"
-- i.e. the brain the owner actually converses with, distinct from `webapp.server._build_chat_brain('tiny-demo',
...)`'s GPU-free demo fixture. Its manifest (`brain.json`) confirms `composer_kind: 'rf'`, `D: 128`, `n_facts: 404`
-- small enough for numpy-CPU, no GPU load needed. This script builds THAT bundle (once; see SCOPE) and re-checks
requirement 1 (no regression) directly on it, plus an honest look at whether requirement 2's positive claim
(self-id resolves correctly) even APPLIES to its current content.

SCOPE, stated plainly (this is NOT a 6-seed battery and is not meant to be):
  - A "developed brain" bundle is a FIXED, already-persisted artifact (`facts.json`/`grounded_codes.npz`/
    `brain.json`, seed=42 baked in at training time) -- there is no meaningful "6 seeds" for re-loading ONE
    specific trained brain; genuine substrate diversity here would need 6 DIFFERENTLY-TRAINED bundles, out of
    scope for a flag-flip verification. This script is ONE build, reported as a supplementary check, not
    formal 6-seed evidence -- the formal 6-seed requirement is carried entirely by
    `_rank13_selfid_anaphora_prodflip_verify.py`'s tiny-demo battery, where genuine seed variation is possible
    and was applied (see that script's SEED-PROPAGATION FIX).
  - `facts.json` was inspected directly (404 facts, TinyStories-style agent/action/patient triples): it has
    ZERO 'brain'-agent facts (`'brain' in agents` is False) -- the bundle has never been taught a self-fact. So
    self_factual/self_identity ("what do you use?"/"what are you") have NOTHING to recall here regardless of the
    flag; this script measures that HONESTLY (both flag states abstain) rather than manufacturing a positive
    self-id result the current bundle's content cannot support. The mechanism's OWN positive correctness/
    retirement evidence comes from the tiny-demo panel (both the de-risk finding and the formal 6-seed battery),
    which HAS self-facts by construction.
  - The anaphora-miss class does NOT depend on self-facts (it resolves an ordinary agent referent, not a self-
    alias), so it IS meaningfully testable here: 'cat meet dog' is the one (agent,action) pair in this bundle
    whose patient ('dog') is itself a known agent AND whose key is unique (`key_counts[('cat','meet')] == 1`,
    checked directly against facts.json, not assumed) -- 'what does cat meet?' establishes the referent 'dog';
    'dog' has no 'swim' fact in this bundle (checked directly), so 'what does it swim?' is a genuine, real-corpus
    anaphora-miss probe.
  - A plain STORED regression probe uses 'what does dog play?' (dog has 3 'play' facts in this corpus -- which
    one comes back is not asserted as "correct", only that flag-on and flag-off return the SAME one).

STATUS (2026-09-05, honest): attempted during the rank-13 production-flip verification and NOT COMPLETED --
this bundle's build allocates a 112,640-neuron `dlpfc_wm`/`cortex_ctx` pair (vocab-scaled PFC/working-memory
regions, unlike tiny-demo's few-hundred-neuron sub-organs) whose connection generation did not finish inside
~5 minutes of wall-clock on a heavily-loaded shared CPU host and was killed rather than left to consume
compute the (more important, tractable, and sufficient on its own) formal 6-seed tiny-demo battery in
`_rank13_selfid_anaphora_prodflip_verify.py` needed. This script is left BANKED, not deleted, as a
correctly-scoped starting point for whoever next has GPU/queued or dedicated-CPU headroom to spend on it --
the panel/fixture design (the 'cat meet dog' -> 'it swim' real-corpus anaphora-miss probe) is sound and
untouched by the non-completion; only the run itself did not finish.

Run (numpy-CPU; empirically NOT "minutes" on a loaded host -- budget accordingly, or queue on a quieter one):
  SIM_BACKEND=numpy python -u -m research.runners._rank13_selfid_anaphora_realbundle_sanity
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

# NOTE: `bridges/` (trained brain checkpoints) is large binary data, not git-tracked, so it exists only in the
# MAIN checkout, not in a git-worktree-isolated agent's own worktree (`_REPO` here) -- confirmed empirically (a
# worktree-relative path 404s). Reading it is a plain filesystem read, not a git operation, so the absolute path
# is used directly; nothing is written there.
BUNDLE = "/home/dant123/Projects/sim/bridges/developed/scale787/day_33"


def _svo_eq(x, y):
    if x is None and y is None:
        return True
    if x is None or y is None:
        return False
    return list(x) == list(y)


def _set_flags(on: bool):
    if on:
        os.environ.pop("BRAIN_NEURAL_SELFID", None)
        os.environ.pop("BRAIN_NEURAL_ANAPHORA_ABSTAIN", None)
    else:
        os.environ["BRAIN_NEURAL_SELFID"] = "0"
        os.environ["BRAIN_NEURAL_ANAPHORA_ABSTAIN"] = "0"


def _build(on: bool):
    os.environ["BRAIN_LTM_SHIP_DEFAULT"] = "off"
    _set_flags(on)
    from webapp.server import _build_chat_brain
    from webapp import gnw_bus_shadow as gbs
    chat, src = _build_chat_brain(BUNDLE, "stub")
    gbs.install_bus_gate(chat)   # the real production combiner
    return chat, src


def _ask(chat, questions):
    out = {}
    for q in questions:
        try:
            out[q] = chat.gate(q)
        except Exception as e:
            out[q] = f"<error: {type(e).__name__}: {e}>"
    return out


def main():
    d = json.load(open(os.path.join(BUNDLE, "facts.json")))
    facts = d["facts"]
    agents = {f["agent"] for f in facts}
    assert "brain" not in agents, "expected no self-facts in this bundle (see module docstring) -- re-check facts.json"

    panel = [
        "what does cat meet?",       # establishes referent='dog' (unique key, patient is itself a known agent)
        "what does dog play?",       # plain STORED regression probe (ambiguous patient OK -- consistency is the check)
        "what does zzznonexistent do?",   # moat / UNSTORED probe
        "what do you use?",          # self_factual -- expected to ABSTAIN on this bundle's current content (no
                                      # 'brain' fact taught yet), both flag states -- see module docstring
        "what are you",              # self_identity -- same expectation
        "what does it swim?",        # anaphora-miss: 'it'->'dog' (from turn 1), dog has no 'swim' fact here
    ]

    chat_off, src_off = _build(False)
    off = _ask(chat_off, panel)
    chat_on, src_on = _build(True)
    on = _ask(chat_on, panel)

    rows = []
    for q in panel:
        rows.append({"q": q, "off": off[q], "on": on[q], "identical": _svo_eq(off[q], on[q])})

    no_regression = all(r["identical"] for r in rows if r["q"] != "what does it swim?")
    anaphora_off_confab = off["what does it swim?"] is not None
    anaphora_on_abstains = on["what does it swim?"] is None
    self_id_has_no_fact_either_way = (off["what do you use?"] is None and on["what do you use?"] is None
                                      and off["what are you"] is None and on["what are you"] is None)

    out = {"runner": "_rank13_selfid_anaphora_realbundle_sanity", "bundle": BUNDLE, "src": src_on,
           "scope": "SUPPLEMENTARY single-build sanity check on the REAL deployed bundle -- NOT part of the "
                    "formal 6-seed requirement (see module docstring for why 6-seed does not apply to one "
                    "already-persisted trained artifact)",
           "rows": rows,
           "no_regression_on_non_anaphora_classes": no_regression,
           "anaphora_off_confabulated": anaphora_off_confab,
           "anaphora_on_abstains": anaphora_on_abstains,
           "self_id_no_fact_present_either_way": self_id_has_no_fact_either_way,
           "n_facts_in_bundle": len(facts), "brain_agent_facts_in_bundle": 0}

    print("\n" + "=" * 100)
    print("  RANK-13 PRODUCTION-FLIP -- REAL DEPLOYED BUNDLE sanity check (scale787/day_33, supplementary)")
    print("=" * 100)
    for r in rows:
        print(f"  {r['q']:32s} off={r['off']!r:40s} on={r['on']!r:40s} identical={r['identical']}")
    print(f"\n  no_regression (non-anaphora classes): {no_regression}")
    print(f"  anaphora-miss: off_confabulated={anaphora_off_confab} on_abstains={anaphora_on_abstains}")
    print(f"  self-id: no fact present in this bundle either way (honest, not a positive claim): "
          f"{self_id_has_no_fact_either_way}")
    op = "research/findings/raw/_rank13_selfid_anaphora_prodflip/realbundle_sanity.json"
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  [saved] {op}")
    ok = no_regression and anaphora_on_abstains
    print(f"\n  SUPPLEMENTARY CHECK: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
