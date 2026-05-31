"""THROWAWAY cheap-first (CPU/numpy): does the bind support FLAT NESTED composition -- a phrase
as a filler inside a larger structure ("big dog goes north" where the agent is itself a bound
"big dog")? This tests compositional DEPTH (hierarchy = language structure).

NEGATIVE 2026-05-31 (seeds 42/43/44): flat nesting FAILS at depth-2.
  outer single-level (action+patient) = 1.000
  depth-2 descend (unbind agent -> phrase, then unbind noun/modifier): noun 0.025-0.050,
    modifier 0.050-0.100 -- AT CHANCE (1/16=0.062).
Unbinding the outer role leaves the phrase buried under full-magnitude cross-terms, so the
depth-2 signal is ~1-of-5 terms and cleanup cannot find it (the superposition/multi-hop wall).

IMPLICATION: the bind is a FLAT slot-filler (one level). Hierarchical structure (nesting,
modification) must use SEPARATE storage + cue retrieval (the validated relational-memory pattern:
store "big dog" as its own {head: dog, modifier: big} fact, reference dog as the agent, recover
the modifier by cue), NOT flat nested binding. Separate-fact storage is the universal architecture
for structure in this substrate (multi-fact AND hierarchy) -- flat superposition/nesting hits the
SNR wall. stdlib+numpy + cache; no protected import.
"""
from __future__ import annotations
import os
import numpy as np

CACHE = "research/findings/raw/activity_level_integration_cache/denoise64_seed%d.npz"
OUT = ["agent", "action", "patient"]
IN = ["modifier", "noun"]


def _c(v):
    v = v.astype(np.float64); v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-12)


def load(s):
    d = np.load(CACHE % s); ws = [k[5:] for k in d.files if k.startswith("obs__")]
    return ws, {w: _c(d["obs__" + w].mean(axis=0)) for w in ws}


def main():
    print("=== nested (flat) composition cheap-first: phrase-as-filler descent ===")
    for s in [42, 43, 44]:
        if not os.path.exists(CACHE % s):
            continue
        ws, con = load(s); V = len(ws); D = con[ws[0]].shape[0]; rng = np.random.default_rng(s)
        R = {r: rng.choice([-1.0, 1.0], D) for r in OUT + IN}
        R = {r: v / np.linalg.norm(v) for r, v in R.items()}

        def clean(est):
            sims = np.array([con[w] @ est for w in ws])
            return ws[int(np.argmax(sims))]

        outer = noun = mod = tot = 0
        for _ in range(40):
            pk = list(rng.choice(V, 4, replace=False))
            m, n, a, p = (ws[pk[0]], ws[pk[1]], ws[pk[2]], ws[pk[3]])
            phrase = R["modifier"] * con[m] + R["noun"] * con[n]
            S = R["agent"] * phrase + R["action"] * con[a] + R["patient"] * con[p]
            outer += int(clean(S * R["action"]) == a and clean(S * R["patient"]) == p)
            ph = S * R["agent"]
            noun += int(clean(ph * R["noun"]) == n)
            mod += int(clean(ph * R["modifier"]) == m)
            tot += 1
        print(f"  seed {s}: outer={outer/tot:.3f}  depth2-noun={noun/tot:.3f}  "
              f"depth2-modifier={mod/tot:.3f}  (chance={1/V:.3f})")
    print("VERDICT: flat nesting FAILS at depth-2 -> use separate storage for hierarchy "
          "(the relational-memory pattern), not flat nested binding.")


if __name__ == "__main__":
    main()
