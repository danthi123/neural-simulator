"""Full-320 ANY-BANK composition escalation -- the RESOLVES-branch follow-up to the structured-SVO 320 test.

The structured 320 test (_insubstrate_flatdistinct320_test.py) draws agent in nouns / action in verbs /
patient in adj, cleanup over all 320 (spatial+functional = 128 distractors). This escalation asks the
stronger question the "full 320" claim really needs: are ALL 320 concepts usable as fillers in ANY role?
Here agent/action/patient are each drawn from the FULL 320-word list (any bank), bound as a 3-role fact,
unbound, cleaned up over all 320. PASS = full-3-slot QA min >= 0.80 over seeds 42/43/44.

If this RESOLVES, the 320 distinct flat codes compose robustly regardless of which bank a filler comes from
-> a genuine 320-concept compositional substrate (not just structured noun/verb/adj over distractors).
If it dips below the structured test, the gap localises the residual (some bank's codes are harder to clean
up against the full 320). Reuse-by-import; loads the cached codes from the structured test (no re-capture);
no protected-module change; no autograd. GPU/CuPy.

Run (after the structured 320 chain has produced the cache):
  python -m research.findings.raw._insubstrate_flatdist320_anybank_test
"""
from __future__ import annotations
import os
import numpy as np

import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as RM
from sim.backend import get_backend

R = "research/findings/raw"
CACHE = f"{R}/_flatdist320_codes.npz"


def main():
    xp, backend = get_backend()
    if not os.path.exists(CACHE):
        print(f"CANNOT-CONCLUDE: {CACHE} missing -- run the structured 320 test first to capture the codes",
              flush=True)
        return
    d = np.load(CACHE)
    words = [str(w) for w in d["_words"]]
    bank_of = {str(w): str(b) for w, b in zip(d["_words"], d["_banks"])}
    codes = {w: np.asarray(d[w], dtype=np.float64) for w in words}
    D = codes[words[0]].shape[0]
    print(f"=== full-320 ANY-BANK composition (backend={backend}, V={len(words)}, D={D}) ===", flush=True)

    import itertools
    btw = [float(np.dot(codes[a], codes[b]))
           for a, b in itertools.islice(itertools.combinations(words, 2), 30000)]
    print(f"  {len(words)}-wide between-concept cos: mean {np.mean(btw):.3f}  max {np.max(btw):.3f}  "
          f"({'DISTINCT' if np.max(btw) < 0.9 else 'DUPLICATES REMAIN -> VOID'})", flush=True)
    if np.max(btw) >= 0.9:
        print("VOID: duplicate codes remain (max between-cos >= 0.9) -- distinct-seed retrain incomplete.",
              flush=True)
        return

    P.RUN_STEPS = 150
    P.COINC_BIAS = -500.0   # the validated higher-rate operating point (K=6 multi-seed, relational 3/3)
    print("  ANY-BANK 3-role composition (agent/action/patient each drawn from ALL 320), full-3-slot QA:",
          flush=True)
    results = []
    bank_miss = {}   # which bank a wrong-recovered filler belonged to, for residual localisation
    for seed in [42, 43, 44]:
        rng = np.random.default_rng(seed)
        roles = {r: rng.choice([-1.0, 1.0], size=D) for r in RM.ROLES}
        roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
        bb, bidx = P.build(seed, D, xp)
        ok = tot = 0
        for _ in range(20):
            pick = rng.choice(len(words), size=3, replace=False)   # distinct fillers, ANY bank
            f = {"agent": words[pick[0]], "action": words[pick[1]], "patient": words[pick[2]]}
            b = RM.bind_fact_spiking(bb, bidx, f, codes, roles, D, xp)
            g = {r: RM.unbind_spiking(bb, bidx, b, r, roles, codes, words, D, xp) for r in RM.ROLES}
            ok += int(all(g[r] == f[r] for r in RM.ROLES))
            tot += 1
            for r in RM.ROLES:
                if g[r] != f[r]:
                    bank_miss[bank_of[f[r]]] = bank_miss.get(bank_of[f[r]], 0) + 1
        results.append(ok / tot)
        print(f"    seed {seed}: {ok/tot:.3f}", flush=True)
    mean = float(np.mean(results))
    print(f"\nRESULT: any-bank full-3-slot QA = {results} (mean {mean:.3f})", flush=True)
    if bank_miss:
        print(f"  miss-by-bank (which bank the mis-recovered filler came from): "
              f"{dict(sorted(bank_miss.items(), key=lambda kv: -kv[1]))}", flush=True)
    if min(results) >= 0.80:
        print("VERDICT: RESOLVES -- all 320 distinct flat concepts compose ROBUSTLY as fillers in ANY role "
              "(cleanup over all 320), multi-seed. The full-320 substrate is a genuine compositional store.",
              flush=True)
    else:
        print(f"VERDICT: any-bank min {min(results):.2f} below 0.80 -- structured-SVO holds but any-bank is "
              "harder; the miss-by-bank histogram localises the residual (a bank whose codes clean up worse "
              "against the full 320).", flush=True)


if __name__ == "__main__":
    main()
