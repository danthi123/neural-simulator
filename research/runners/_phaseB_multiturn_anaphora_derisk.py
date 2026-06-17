"""Multi-turn dialogue cheap-first DE-RISK — does a PERSISTENT spiking working-memory buffer carry a referent
across a TURN BOUNDARY so an anaphor ("it") resolves and feeds the composer's answer?

Per the scoping (2026-06-17-multihop-reasoning-multiturn-dialogue-scoping.md, Option 3): multi-turn dialogue
needs discourse state held ACROSS turns. The spiking cortico-PFC loop `SpikingLoopContextBuffer`
(content_selection_spiking.py) is a VALIDATED working memory (holds a driven concept via its attractor, ~220x
specificity, >=3-concept span). The NEW thing this de-risk tests is the INTEGRATION: hold the salient referent
from turn 1 in the loop, do NOT reset between turns, and on turn 2 resolve "it" by READING the held attractor,
then answer with the production composer.

THE 2-TURN DIALOGUE.
  Turn 1 (user): "dog chase cat"     -> agent stores the fact AND writes the object referent (cat) into the WM loop.
  Turn 2 (user): "what does it eat?" -> "it" = read the WM loop (the held referent) -> cat -> composer.query_patient(cat, eat) -> fish.

GATE (>=3 seeds): GO = the anaphor resolves to the RIGHT referent (cat dominates the WM read) AND the answer is
correct (fish), across the turn boundary WITHOUT a reset, AND the controls collapse:
  * RESET control: reset the WM between turns -> the referent is gone -> resolution fails (the persistence is load-bearing).
  * LESION control: zero the attractor loop weights -> no persistence -> the referent decays -> resolution fails.
  * MOAT: turn 2 with NO turn-1 referent (empty WM) -> no dominant concept -> abstain (no confabulated "it").
  * SPECIFICITY: the held referent (cat) must dominate the read, not just edge out chance.
NEGATIVE = the persistent read does not beat the reset/lesion controls (the loop does not actually carry the
referent across the turn).

Run (CPU; small ~600-neuron WM bridge, a few seeds):
  SIM_BACKEND=numpy python -m research.runners._phaseB_multiturn_anaphora_derisk --seeds 42 43 44
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.rf_phasor_composer import RFPhasorComposer
from research.runners.content_selection_spiking import SpikingLoopContextBuffer

# a tiny discourse world: facts + the concepts the WM loop holds (referent NOUNS only -- the composer also needs
# the ACTION words in its vocabulary, since it binds the action filler).
FACTS = [("dog", "chase", "cat"), ("cat", "eat", "fish"), ("bird", "eat", "worm"), ("fox", "chase", "hare")]
CONCEPTS = ["dog", "cat", "fish", "bird", "worm", "fox", "hare", "ball"]   # the WM-held referent nouns
ACTIONS = ["chase", "eat"]


def resolve_referent(wm, window=20):
    """Read the WM loop; the resolved referent is the concept whose attractor dominates (with its specificity =
    top firing / mean of the rest). Returns (referent_or_None, specificity)."""
    rates = wm.read(window=window)
    items = sorted(rates.items(), key=lambda kv: kv[1], reverse=True)
    top, top_r = items[0]
    rest = np.mean([r for _, r in items[1:]]) if len(items) > 1 else 0.0
    if top_r <= 1e-6:
        return None, 0.0                              # empty WM -> no antecedent -> abstain
    spec = top_r / (rest + 1e-9)
    return (top if spec > 1.5 else None), spec        # require a clear winner, else abstain


def run_seed(seed):
    composer = RFPhasorComposer(seed=seed, D=128, vocab=CONCEPTS + ACTIONS)
    for a, v, o in FACTS:
        composer.store(a, v, o)

    def fresh_wm(lesion=False):
        wm = SpikingLoopContextBuffer(CONCEPTS, n=600, pattern_size=40, seed=seed, enable_ou=False)
        if lesion:                                    # sever the attractor loop -> no persistence
            for pname in ("c2d", "d2c"):
                idx = wm.bridge.region_manager  # noqa: F841 (kept for clarity)
            # zero the loop by re-installing zero weights on each concept's attractor
            for c in wm.concepts:
                cpat = np.asarray(wm.B.to_host(wm._cpat[c])); dpat = np.asarray(wm.B.to_host(wm._dpat[c]))
                ps = wm._psize
                pre1 = np.repeat(cpat, ps).astype(np.int64); post1 = np.tile(dpat, ps).astype(np.int64)
                pre2 = np.repeat(dpat, ps).astype(np.int64); post2 = np.tile(cpat, ps).astype(np.int64)
                zz = np.zeros(ps * ps, np.float32)
                wm.bridge.set_pathway_weights("c2d", pre_indices=pre1, post_indices=post1, weights=zz, add_missing=True)
                wm.bridge.set_pathway_weights("d2c", pre_indices=pre2, post_indices=post2, weights=zz, add_missing=True)
        return wm

    def dialogue(reset_between_turns=False, lesion=False, empty=False):
        """Turn 1 establishes the referent (cat); turn 2 resolves 'it' and answers. Returns (referent, answer, spec)."""
        wm = fresh_wm(lesion=lesion)
        # TURN 1: "dog chase cat" -> the object 'cat' is the salient discourse referent, written to the loop.
        if not empty:
            wm.update(["cat"])
        # --- turn boundary --- (persistent WM: do nothing; reset control: rebuild the loop, losing the held state)
        if reset_between_turns:
            wm = fresh_wm(lesion=lesion)               # a reset wipes the held referent
        # TURN 2: "what does IT eat?" -> resolve 'it' from the WM, then answer via the composer.
        ref, spec = resolve_referent(wm)
        ans = composer.query_patient(ref, "eat") if ref is not None else None
        return ref, ans, spec

    ref_p, ans_p, spec_p = dialogue()                                  # persistent (the real path)
    ref_r, ans_r, _ = dialogue(reset_between_turns=True)               # reset control
    ref_l, ans_l, _ = dialogue(lesion=True)                           # attractor-lesion control
    ref_e, ans_e, _ = dialogue(empty=True)                            # moat: no antecedent

    out = {
        "seed": seed,
        "persistent": {"referent": ref_p, "answer": ans_p, "specificity": round(spec_p, 2)},
        "reset": {"referent": ref_r, "answer": ans_r},
        "lesion": {"referent": ref_l, "answer": ans_l},
        "empty_moat": {"referent": ref_e, "answer": ans_e},
        "go": (ref_p == "cat" and ans_p == "fish" and ans_r != "fish" and ans_l != "fish" and ref_e is None),
        "moat_ok": ref_e is None,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="research/findings/raw/_phaseB_multiturn_anaphora.json")
    a = ap.parse_args()

    print("[multi-turn anaphora de-risk] does a persistent spiking WM carry the referent across a turn boundary?\n"
          "  GATE: turn2 'it'->cat (held), answer fish; reset/lesion break it; empty WM abstains.\n", flush=True)
    results = []
    for seed in a.seeds:
        r = run_seed(seed)
        results.append(r)
        p = r["persistent"]
        print(f"  [seed {seed}] persistent: it->{p['referent']!r} answer {p['answer']!r} (spec {p['specificity']}) | "
              f"reset it->{r['reset']['referent']!r} ans {r['reset']['answer']!r} | "
              f"lesion it->{r['lesion']['referent']!r} ans {r['lesion']['answer']!r} | "
              f"empty-moat {r['empty_moat']['referent']!r}  ==> {'GO' if r['go'] else 'NO'}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"results": results}, fh, indent=2, default=str)

    n_go = sum(r["go"] for r in results)
    moat_ok = all(r["moat_ok"] for r in results)
    print(f"\n{'='*100}", flush=True)
    if not moat_ok:
        print("  MOAT_BREACH (HARD STOP): an empty WM resolved a referent (confabulated 'it') — investigate.", flush=True)
    elif n_go == len(results):
        print(f"  GO ({n_go}/{len(results)} seeds): a PERSISTENT spiking working-memory loop carries the discourse "
              "referent across the turn boundary — turn-2 'it' resolves to the held concept (cat) and the composer "
              "answers correctly (fish). RESETTING the loop or LESIONING its attractor breaks resolution, and an "
              "empty WM abstains (no confabulated antecedent). Multi-turn anaphora is spiking-native + moat-safe.",
              flush=True)
    else:
        print(f"  PARTIAL/NEGATIVE ({n_go}/{len(results)} seeds): the persistent read does not robustly beat the "
              "reset/lesion controls — the loop is not reliably carrying the referent across the turn at this "
              "config (raise pattern_size / drive / hold window, or check attractor specificity).", flush=True)
    print(f"  [saved] {a.out}\n{'='*100}", flush=True)


if __name__ == "__main__":
    main()
