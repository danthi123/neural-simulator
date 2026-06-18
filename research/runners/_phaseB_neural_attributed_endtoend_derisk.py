"""CYCLE 201 — richer-syntax #1 END-TO-END: the NEURAL attributed parse -> the composer, fully brain-based.

Combines the two validated halves into the end-to-end attributed-entity capability:
  - CYCLE 200: the attributed PARSE is brain-based (AttributedBridgeParser reads each word's role off the bridge in
    SPIKES via the from-start x from-END x voice conjunction; GO 6/6, role read-out 1.000).
  - CYCLE 199: the composer back end is READY (RFPhasorComposer.store accepts a (adjs, noun) patient -> attribute/
    attribute2 roles; query_patient renders 'big red apple') -- the bind/bundle/unbind is spiking RF.
This wires them: NEURAL parse of 'dog eat big red apple' -> {agent,action,attribute,attribute2,patient} (per-word,
from the bridge in spikes) -> reconstruct the (adjs, noun) patient -> composer.store -> query_patient -> 'big red
apple'. Both the comprehension (parser firing selects the role) AND the storage/retrieval (RF bind/bundle/unbind +
cleanup) are spiking. The ONLY host steps are the environment (the token string) + zipping words to the
spike-read-out roles -- the cognition is neural.

GATE (multi-seed): end-to-end round-trip 'agent action [adjs] noun' == the host oracle truth >= 0.90, >= 5/6 seeds,
AND flat-SVO end-to-end un-regressed, AND the no-confab moat holds (an unstored cue abstains). GO => richer-syntax #1
(attributed entities) is fully brain-based END-TO-END -- the first richer-than-flat-SVO capability realized in
spikes from comprehension through retrieval. NEGATIVE => localize (the parse is GO standalone, so a NEGATIVE would
be the composer's attribute bundle at the round-trip operating point).

Reuse-by-import: AttributedBridgeParser (CYCLE 200, the neural parse) + RFPhasorComposer (store/query, UNCHANGED).
GPU for real (the parser trains on the bridge). NO sim/ edit.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_neural_attributed_endtoend_derisk
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._phaseB_neural_attributed_parser_derisk import AttributedBridgeParser  # noqa: E402
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402

NOUNS = ["dog", "cat", "bird", "apple", "river"]
VERBS = ["eat", "see", "chase", "hold", "find"]
ADJS = ["big", "red", "small", "hot", "cold", "wet"]
VOCAB = NOUNS + VERBS + ADJS


def _parse_words(parser, words):
    """Map the bridge's per-position SPIKING role read-out onto the actual words -> {role: word}."""
    roles = parser.parse_roles(len(words), voice=0)
    return {role: w for role, w in zip(roles, words)}


def run_seed(seed):
    parser = AttributedBridgeParser(seed=seed)                  # the NEURAL parse (trained on the bridge, CYCLE 200)
    comp = RFPhasorComposer(seed=seed, D=64, vocab=VOCAB)       # the READY composer back end
    rng = np.random.default_rng(seed)
    attr_ok = attr_n = 0
    for _ in range(24):
        agent = NOUNS[int(rng.integers(len(NOUNS)))]
        action = VERBS[int(rng.integers(len(VERBS)))]
        noun = NOUNS[int(rng.integers(len(NOUNS)))]
        k = int(rng.integers(1, 3))
        adjs = list(rng.choice(ADJS, size=k, replace=False))
        words = [agent, action] + adjs + [noun]
        roles = _parse_words(parser, words)                    # COMPREHEND in spikes
        p_noun = roles.get("patient")
        p_adjs = [roles[r] for r in ("attribute", "attribute2") if r in roles]
        patient = (p_adjs, p_noun) if p_adjs else p_noun
        comp.kb = []                                           # isolate the round-trip (no (a,v) collision)
        comp.store(roles.get("agent"), roles.get("action"), patient)   # STORE in spikes (RF bind/bundle)
        truth = " ".join(adjs + [noun])
        got = comp.query_patient(roles.get("agent"), roles.get("action"))   # RETRIEVE in spikes (RF unbind+cleanup)
        attr_ok += int(got == truth); attr_n += 1
    # flat-SVO end-to-end non-regression
    flat_ok = flat_n = 0
    for _ in range(12):
        a = NOUNS[int(rng.integers(len(NOUNS)))]; v = VERBS[int(rng.integers(len(VERBS)))]
        p = NOUNS[int(rng.integers(len(NOUNS)))]
        roles = _parse_words(parser, [a, v, p])
        comp.kb = []
        comp.store(roles.get("agent"), roles.get("action"), roles.get("patient"))
        flat_ok += int(comp.query_patient(roles.get("agent"), roles.get("action")) == p); flat_n += 1
    comp.kb = []
    moat_ok = comp.query_patient("river", "find") is None      # the no-confab moat
    return {"seed": seed, "attr_e2e": attr_ok / attr_n, "flat_e2e": flat_ok / flat_n, "moat_ok": bool(moat_ok)}


def main():
    os.environ.setdefault("SIM_BACKEND", "cupy")
    t0 = time.time()
    print("[neural attributed END-TO-END] NEURAL parse (spikes) -> reconstruct (adjs,noun) -> composer store/query "
          "(spikes): is attributed-entity comprehension->storage->retrieval fully brain-based?\n", flush=True)
    seeds = (42, 43, 44, 45, 46, 47)
    rows = [run_seed(s) for s in seeds]
    for r in rows:
        print(f"  [seed {r['seed']}] attributed end-to-end {r['attr_e2e']:.3f} | flat-SVO {r['flat_e2e']:.3f} | "
              f"moat {r['moat_ok']}", flush=True)

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    attr, flat = m("attr_e2e"), m("flat_e2e")
    n_go = sum(1 for r in rows if r["attr_e2e"] >= 0.90 and r["flat_e2e"] >= 0.90 and r["moat_ok"])
    print(f"\n{'='*98}\n  MEAN (6 seeds): attributed end-to-end {attr:.3f} | flat-SVO {flat:.3f} | seeds GO {n_go}/6",
          flush=True)
    print(f"{'='*98}", flush=True)
    go = n_go >= 5 and attr >= 0.90 and flat >= 0.90
    if go:
        print(f"  GO: richer-syntax #1 (attributed entities) is fully BRAIN-BASED END-TO-END -- neural parse (spikes) "
              f"-> RF store/query (spikes) round-trips 'agent action adj* noun' at {attr:.3f} ({n_go}/6 seeds), "
              f"flat-SVO un-regressed {flat:.3f}, the moat holds. The first richer-than-flat-SVO conversational "
              f"capability realized in spikes from comprehension through retrieval. ==> richer-syntax #2 "
              f"(multi-frame comprehension).", flush=True)
    else:
        print(f"  NEGATIVE/PARTIAL: attributed {attr:.3f} / flat {flat:.3f} / GO {n_go}/6 -- the neural parse is GO "
              f"standalone (CYCLE 200), so localize the composer's attribute bundle at the round-trip operating "
              f"point (D, the 5-role superposition).", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    out = {"attr_e2e": attr, "flat_e2e": flat, "seeds_go": n_go, "go": bool(go), "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_neural_attributed_endtoend.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
