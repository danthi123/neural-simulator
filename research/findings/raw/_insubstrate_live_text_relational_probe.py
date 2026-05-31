"""End-to-end LIVE-TEXT relational fact-memory: instead of cached denoise64 concept codes, drive
each word LIVE through the trained concept-pool bridge (build_substrate + capture_activity) and feed
that live activity into the spiking relational fact-memory. Quantifies how much the recognition
front-end's noise/fragility costs vs the cached-code baseline (single/relational = 1.000 at bias-500).

This is the concrete first step of the live-text-input arc: text(word) -> concept-pool activity (live)
-> spiking bind -> relational query -> answer, fully in the substrate.

Reuse-by-import: activity_level_integration (capture) + _insubstrate_bind_unbind_probe (bind) +
_insubstrate_relational_memory_probe (relational logic). No protected-module modification.

RESULT 2026-05-31 seed 42 (m_obs=16, bias-500): END-TO-END LIVE TEXT WORKS at the cached baseline.
  concept-pool recognition 15/16 (the front-end's own v16 Phase-1 accuracy at M=16 denoising);
  LIVE-code spiking relational memory single-fact=1.000 relational=1.000 control=1.000 -- IDENTICAL
  to the cached-code baseline. The bind uses the full DISTRIBUTED activity vector, so it is robust
  to the 1 recognition mislabel: text -> concept-pool activity (live) -> spiking bind -> relational
  query -> answer works at 1.000, all in the substrate. Multi-seed confirmation in flight.
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
    ap.add_argument("--m-obs", type=int, default=16, help="captures averaged per word (denoise)")
    ap.add_argument("--run-steps", type=int, default=150)
    ap.add_argument("--coinc-bias", type=float, default=-500.0)
    ap.add_argument("--n-trials", type=int, default=12)
    ap.add_argument("--n-facts", type=int, default=2)
    a = ap.parse_args()
    P.RUN_STEPS = a.run_steps; P.COINC_BIAS = a.coinc_bias
    xp, backend = get_backend()

    print(f"=== END-TO-END live-text relational fact-memory (backend={backend}, seed={a.seed}, "
          f"m_obs={a.m_obs}) ===", flush=True)
    print("building trained concept-pool bridge + capturing LIVE concept codes...", flush=True)
    cp_bridge = A.build_substrate(a.seed)
    all_idx, slices, all_pools = A.pool_layout(cp_bridge)
    recipe = A._phase1_recipe(False)
    all_words, w2i = A._all_words_word_to_idx()
    n_orth = max(A._N_WORDS_ORTHOGONAL, len(all_words))

    live = {}
    recog_ok = 0
    for w in all_words:
        try:
            target = A._direct_pool_target(w)
        except KeyError:
            continue
        rows = [A.capture_activity(cp_bridge, w, all_idx, recipe, w2i, n_orth) for _ in range(a.m_obs)]
        mean_a = np.mean(rows, axis=0)
        live[w] = _center(mean_a)
        recog_ok += int(A.recognized_pool(mean_a, slices, all_pools) == target)
    words = list(live.keys()); D = live[words[0]].shape[0]
    print(f"  captured {len(words)} live codes (D={D}); concept-pool recognition "
          f"{recog_ok}/{len(words)} correct (the front-end's own accuracy)", flush=True)

    # build the coincidence bind bridge at the live-code dimension
    bridge, idx = P.build(a.seed, D, xp)
    rng = np.random.default_rng(a.seed)
    roles = {r: rng.choice([-1.0, 1.0], size=D) for r in R.ROLES}
    roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
    concepts = live

    s_ok = rel_ok = ctrl_ok = tot = 0
    for _ in range(a.n_trials):
        picks = rng.choice(len(words), size=3 * a.n_facts, replace=False)
        facts = [{"agent": words[picks[3*f]], "action": words[picks[3*f+1]], "patient": words[picks[3*f+2]]}
                 for f in range(a.n_facts)]
        bound = [R.bind_fact_spiking(bridge, idx, fc, concepts, roles, D, xp) for fc in facts]
        qf = rng.integers(a.n_facts); qrole = R.ROLES[rng.integers(3)]
        s_ok += int(R.unbind_spiking(bridge, idx, bound[qf], qrole, roles, concepts, words, D, xp)
                    == facts[qf][qrole])
        tf = rng.integers(a.n_facts); cue = facts[tf]["agent"]; best = None
        for f in range(a.n_facts):
            if R.unbind_spiking(bridge, idx, bound[f], "agent", roles, concepts, words, D, xp) == cue:
                best = f; break
        ans = (R.unbind_spiking(bridge, idx, bound[best], "patient", roles, concepts, words, D, xp)
               if best is not None else None)
        rel_ok += int(ans == facts[tf]["patient"])
        non = [w for w in words if w not in [fc["agent"] for fc in facts]]
        cb = str(rng.choice(non)); bestc = None
        for f in range(a.n_facts):
            if R.unbind_spiking(bridge, idx, bound[f], "agent", roles, concepts, words, D, xp) == cb:
                bestc = f; break
        ctrl_ok += int(bestc is None); tot += 1

    print(f"  LIVE-code spiking: single-fact={s_ok/tot:.3f}  relational={rel_ok/tot:.3f}  "
          f"control={ctrl_ok/tot:.3f}  (cached-code baseline at bias-500 = 1.000/1.000/1.000)", flush=True)
    print("READ: gap from the 1.000 cached baseline = the recognition front-end's noise/fragility cost. "
          "The bind is validated; end-to-end conversation is bottlenecked by text->concept recognition.",
          flush=True)


if __name__ == "__main__":
    main()
