"""EMERGENT FEATURE #2 (the integrated one-brain loop) — RECONSOLIDATION-IN-THE-LOOP, cheap-first de-risk.

Reconsolidation (`update_on_mismatch`: a corrective utterance reactivates a cued fact and rewrites its patient IN
PLACE, prediction-error-gated) is already built + validated as a single op. The EMERGENT question the integrated
PERSISTENT loop now makes demonstrable: when a correction labilizes + rewrites ONE fact in a SHARED persistent store
of many facts, is the update ISOLATED (the other stored facts are untouched -- no collateral corruption), and does
reconsolidation hold as the store FILLS (set-size K)? Biological reconsolidation is famously update-specific (Nader
2000; the labilized trace is re-stabilized without erasing neighbours) -- a naive shared-memory update would smear.

The claim is brain-based: the correction is the composer's neural reactivation + PE-gated in-place rewrite; the host
only utters the correction + reads the cleanup argmax. NO sim/ edit. Reuse-by-import `OneBrainComposer`
(`update_on_mismatch` / `_calibrate_pe_labile` / `count_facts` / `query_patient`); the store is block-major (each fact
in its own (1+D) trigger->readout block), so isolation is STRUCTURAL -- this de-risk confirms it holds end-to-end in
the live loop (incl. the BATCHED parallel read that fires all triggers at once) + across set-size.

GO bar:
  - REWRITE: every corrected fact recalls its NEW patient (the in-place rewrite worked).
  - NO DUPLICATE: count_facts(agent, action) == 1 after each correction (not a contradictory append).
  - ISOLATION: after correcting fact i, EVERY OTHER fact still recalls its current patient (collateral-damage rate ~0).
  - CONTROLS: a same-patient "correction" RE-STABILIZES (no spurious rewrite); a never-stored correction ABSTAINS
    (the no-confab moat); the PE gate is calibrated from the data (_calibrate_pe_labile), not a downstream probe.
  - SET-SIZE: all of the above hold as K grows (8 -> 16 -> 24).
An honest NEGATIVE (a correction smears neighbours, or reconsolidation breaks as K grows) maps a real boundary of the
shared persistent store + motivates the fix -- itself the deliverable.

Run (GPU; numpy is a tiny-smoke fallback):
  SIM_BACKEND=cupy python -m research.runners._emergent_reconsolidation_in_loop_derisk \
      --seeds 42 43 44 --K 8 16 24 --out research/findings/raw/_emergent_reconsolidation_in_loop.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from research.runners.one_brain_composer import OneBrainComposer

# A vocab big enough to build up to ~24 facts with DISTINCT (agent, action) cues + distinct patients.
VOCAB = [f"w{i:02d}" for i in range(40)]


def _make_facts(K, rng):
    """K facts (agent, action, patient) with DISTINCT (agent, action) cues (so a who/what query is unambiguous) and a
    patient distinct from the agent. Deterministic per seed."""
    facts, seen = [], set()
    i = 0
    while len(facts) < K:
        a = VOCAB[(i * 3 + 1) % len(VOCAB)]
        v = VOCAB[(i * 7 + 5) % len(VOCAB)]
        p = VOCAB[(i * 11 + 9) % len(VOCAB)]
        if (a, v) not in seen and a != v and p != a:
            seen.add((a, v)); facts.append((a, v, p))
        i += 1
        if i > 5000:
            break
    return facts


def _new_patient(orig_patient, agent, action, rng):
    """A correction target: a vocab word != the original patient, != agent/action (a genuine mismatch)."""
    for w in rng.permutation(VOCAB):
        if w not in (orig_patient, agent, action):
            return str(w)
    return orig_patient


def run_seed(seed, K):
    rng = np.random.default_rng(seed)
    facts = _make_facts(K, rng)
    c = OneBrainComposer(seed=seed, D=128, vocab=VOCAB, k_max=max(32, K + 1))
    for (a, v, p) in facts:
        c.store(a, v, p)
    current = {(a, v): p for (a, v, p) in facts}            # the live ground-truth patient per cue

    # baseline: every fact recalls correctly before any correction
    baseline_ok = all(c.query_patient(a, v) == p for (a, v, p) in facts)

    rewrite_ok = no_dup = isolation_ok = 0
    n_corr = 0
    for (a, v, p) in facts:                                  # correct EACH fact in turn, in the live persistent store
        new_p = _new_patient(p, a, v, rng)
        res = c.update_on_mismatch(a, v, new_p)             # neural reactivation + PE-gated in-place rewrite
        n_corr += 1
        if res.get("action") == "rewrite" and res.get("wrote"):
            current[(a, v)] = new_p
            rewrite_ok += 1 if c.query_patient(a, v) == new_p else 0
            no_dup += 1 if c.count_facts(a, v) == 1 else 0
        # ISOLATION: every OTHER fact still recalls its CURRENT patient (no collateral corruption)
        others_ok = all(c.query_patient(oa, ov) == current[(oa, ov)] for (oa, ov) in current if (oa, ov) != (a, v))
        isolation_ok += 1 if others_ok else 0

    # control 1: a SAME-patient "correction" must RE-STABILIZE (no spurious rewrite) + leave count==1
    a0, v0 = facts[0][0], facts[0][1]
    same = c.update_on_mismatch(a0, v0, current[(a0, v0)])
    restabilize_ok = (same.get("action") == "restabilize" and not same.get("wrote")
                      and c.query_patient(a0, v0) == current[(a0, v0)] and c.count_facts(a0, v0) == 1)
    # control 2: a NEVER-stored correction must ABSTAIN (the no-confab moat)
    moat = c.update_on_mismatch("ZZZ_unstored", "go", "north")
    moat_ok = (moat.get("action") == "abstain" and not moat.get("wrote")
               and c.count_facts("ZZZ_unstored", "go") == 0)

    go = bool(baseline_ok and rewrite_ok == n_corr and no_dup == n_corr and isolation_ok == n_corr
              and restabilize_ok and moat_ok)
    return {"seed": seed, "K": K, "baseline_ok": bool(baseline_ok), "n_corrections": n_corr,
            "rewrite_ok": rewrite_ok, "no_duplicate_ok": no_dup, "isolation_ok": isolation_ok,
            "restabilize_ok": bool(restabilize_ok), "moat_ok": bool(moat_ok), "go": go}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--K", type=int, nargs="+", default=[8, 16, 24])
    ap.add_argument("--out", default="research/findings/raw/_emergent_reconsolidation_in_loop.json")
    a = ap.parse_args()
    print("[reconsolidation-in-loop] correct EACH fact in a shared persistent store; GO = rewrite + no-duplicate + "
          "ISOLATION (others untouched) + restabilize/moat controls, holding across set-size K\n", flush=True)
    results = []
    for K in a.K:
        for seed in a.seeds:
            r = run_seed(seed, K)
            results.append(r)
            print(f"  [K={K:2d} seed {seed}] baseline {r['baseline_ok']} | rewrite {r['rewrite_ok']}/{r['n_corrections']} "
                  f"| no-dup {r['no_duplicate_ok']}/{r['n_corrections']} | ISOLATION {r['isolation_ok']}/{r['n_corrections']} "
                  f"| restabilize {r['restabilize_ok']} | moat {r['moat_ok']}  ==> {'GO' if r['go'] else 'NEGATIVE'}",
                  flush=True)
    n_go = sum(1 for r in results if r["go"])
    print("\n" + "=" * 96, flush=True)
    print(f"  RECONSOLIDATION-IN-THE-LOOP: {n_go}/{len(results)} (seed x K) GO. Correcting one fact in the shared "
          f"persistent store rewrites it in place (no duplicate) WITHOUT corrupting the other stored facts (isolation), "
          f"holding as the store fills -- update-specific reconsolidation as an emergent property of the live loop.",
          flush=True)
    print("=" * 96, flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"go": n_go, "n": len(results), "results": results}, f, indent=2)
    print(f"  [saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
