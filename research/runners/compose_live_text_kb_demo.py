"""Owner-facing END-TO-END demo: a queryable knowledge base from LIVE TEXT, in spiking.

The culmination of the 2026-05-31 composition arc. Drives words LIVE through the trained
concept-pool bridge (text -> concept-pool activity), stores subject/verb/object facts via the
validated spiking compositional bind, and answers relational queries -- all in the spiking
substrate, with NO cached codes (the concept codes are captured live from the text-driven
substrate).

Pipeline: type a word -> lang_input drive -> concept-pool population activity (live) -> spiking
bind (agent/action/patient coincidence) -> relational query via spiking unbind + cleanup -> answer.

Validated (research/findings/2026-05-31-in-substrate-spiking-bind-unbind-VALIDATED.md): the spiking
bind/unbind RESOLVES multi-seed to K=6; the relational fact-memory is multi-seed 3/3; end-to-end on
LIVE codes matches the cached baseline (single/relational/control 1.000, seed 42). Honest scope:
roles supplied (parser is the next arc); 16-word vocab; canonical SVO; cue-based retrieval.

Run:  python -m research.runners.compose_live_text_kb_demo  (builds the concept-pool bridge, ~3-5 min)
"""
from __future__ import annotations
import argparse
import numpy as np

import research.findings.raw.activity_level_integration as A
import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as R
from sim.backend import get_backend


def _center(v):
    v = v.astype(np.float64); v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--m-obs", type=int, default=16)
    a = ap.parse_args()
    P.RUN_STEPS = 150; P.COINC_BIAS = -500.0
    xp, backend = get_backend()
    print(f"=== END-TO-END live-text knowledge-base demo (backend={backend}, seed={a.seed}) ===")
    print("building trained concept-pool bridge + capturing live concept codes (~3-5 min)...")
    cp_bridge = A.build_substrate(a.seed)
    all_idx, slices, all_pools = A.pool_layout(cp_bridge)
    recipe = A._phase1_recipe(False)
    all_words, w2i = A._all_words_word_to_idx()
    n_orth = max(A._N_WORDS_ORTHOGONAL, len(all_words))
    live = {}
    for w in all_words:
        try:
            A._direct_pool_target(w)
        except KeyError:
            continue
        rows = [A.capture_activity(cp_bridge, w, all_idx, recipe, w2i, n_orth) for _ in range(a.m_obs)]
        live[w] = _center(np.mean(rows, axis=0))
    words = list(live.keys()); D = live[words[0]].shape[0]
    print(f"vocabulary ({len(words)}, captured LIVE from text): {words}\n")

    bridge, idx = P.build(a.seed, D, xp)
    rng = np.random.default_rng(a.seed)
    roles = {r: rng.choice([-1.0, 1.0], size=D) for r in R.ROLES}
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}

    def pick(*c):
        for w in c:
            if w in live:
                return w
        return words[0]

    facts = [{"agent": pick("dog"), "action": pick("go", "come"), "patient": pick("north", "river")},
             {"agent": pick("cat"), "action": pick("come", "stop"), "patient": pick("south", "apple")}]
    bound = [R.bind_fact_spiking(bridge, idx, f, live, roles, D, xp) for f in facts]
    print("--- Knowledge base (each word captured live from text, bound in spiking) ---")
    for f in facts:
        print(f"    {f['agent']} {f['action']} {f['patient']}")

    def ask(cue, role):
        for fi in range(len(facts)):
            if R.unbind_spiking(bridge, idx, bound[fi], "agent", roles, live, words, D, xp) == cue:
                return R.unbind_spiking(bridge, idx, bound[fi], role, roles, live, words, D, xp)
        return "(no fact found)"

    print("\n--- Relational queries (live text -> spiking bind/unbind -> answer) ---")
    for f in facts:
        p = ask(f["agent"], "patient"); v = ask(f["agent"], "action")
        ok = (p == f["patient"]) and (v == f["action"])
        print(f"    '{f['agent']}': object? {p}   action? {v}   [{'OK' if ok else 'MISS'}]")
    absent = next((w for w in words if w not in [f["agent"] for f in facts]), words[0])
    print(f"    (control) '{absent}' (not in KB): {ask(absent, 'patient')}")
    print("\nEnd-to-end: words captured LIVE from the text-driven substrate, bound + queried entirely "
          "in spiking. Honest scope: roles supplied; 16-word vocab; cue-based retrieval (parser is next).")


if __name__ == "__main__":
    main()
