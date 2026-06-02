"""Knowledge-base capacity on the full-320 flat-distinct substrate: how many separately-stored facts can the
320-concept biological composition hold with reliable relational retrieval?

The KB-scaling finding (2026-05-31) capped relational retrieval at ~12 facts (numpy) / ~5 (spiking) on a
SMALL-vocab denoise64 substrate. Now that robust composition is validated at the full 320-concept scale
(structured 1.000x3, any-bank 0.992 6-seed), this asks the scaling question on the REAL 320 substrate: store N
facts (each a K=3 separate spiking bind, agent/action/patient drawn from all 320), then for every stored fact
run the relational query (find-by-agent cue -> read-patient) + a who/what role query, plus an absent-cue
control (a never-stored agent must return no match). Vary N in {5, 10, 20}, multi-seed.

Separate-fact storage means each fact is an independent bound vector (no superposition interference), so the
capacity question is really cleanup reliability as the fact set grows -- does the 320-wide distinct-code
cleanup keep the right agent/patient on top when many facts compete?

PASS per N = relational + role query >= 0.80 AND control clean (no false match), multi-seed. Reuse-by-import
(RM bind/unbind, the cached 320 codes); no protected-module change; no autograd. GPU/CuPy.

Run (after _insubstrate_flatdistinct320_test has cached the codes):
  python -m research.findings.raw._insubstrate_flatdist320_kb_capacity_test
"""
from __future__ import annotations
import os
import numpy as np

import research.findings.raw._insubstrate_bind_unbind_probe as P
import research.findings.raw._insubstrate_relational_memory_probe as RM
from sim.backend import get_backend

CACHE = "research/findings/raw/_flatdist320_codes.npz"
N_FACTS_GRID = [5, 10, 15]   # brackets the prior ~5 (spiking) / ~12 (numpy) cap on the small-vocab substrate
SEEDS = [42, 43, 44]
TRIALS_PER = 6   # independent KB draws per (N, seed); find-by-agent + control are linear scans (~1.2s/unbind)


def main():
    xp, backend = get_backend()
    if not os.path.exists(CACHE):
        print(f"CANNOT-CONCLUDE: {CACHE} missing -- run _insubstrate_flatdistinct320_test first.", flush=True)
        return
    d = np.load(CACHE)
    words = [str(w) for w in d["_words"]]
    codes = {w: np.asarray(d[w], dtype=np.float64) for w in words}
    D = codes[words[0]].shape[0]
    P.RUN_STEPS = 150
    P.COINC_BIAS = -500.0   # validated higher-rate operating point (relational fact-memory 3/3 multi-seed)
    print(f"=== full-320 KB capacity (backend={backend}, V={len(words)}, D={D}, "
          f"bias={P.COINC_BIAS}, run_steps={P.RUN_STEPS}) ===", flush=True)

    overall = {}
    for n_facts in N_FACTS_GRID:
        per_seed_rel = []
        per_seed_role = []
        per_seed_ctrl = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed * 1000 + n_facts)
            roles = {r: rng.choice([-1.0, 1.0], size=D) for r in RM.ROLES}
            roles = {r: v / np.linalg.norm(v) for r, v in roles.items()}
            bb, bidx = P.build(seed, D, xp)
            rel_ok = role_ok = ctrl_ok = tot = 0
            for _ in range(TRIALS_PER):
                pick = rng.choice(len(words), size=3 * n_facts, replace=False)
                facts = [{"agent": words[pick[3 * f]], "action": words[pick[3 * f + 1]],
                          "patient": words[pick[3 * f + 2]]} for f in range(n_facts)]
                bound = [RM.bind_fact_spiking(bb, bidx, fc, codes, roles, D, xp) for fc in facts]

                # relational: pick a target fact, cue its agent, find it, read its patient
                tf = int(rng.integers(n_facts)); cue = facts[tf]["agent"]
                best = None
                for f in range(n_facts):
                    if RM.unbind_spiking(bb, bidx, bound[f], "agent", roles, codes, words, D, xp) == cue:
                        best = f; break
                ans = (RM.unbind_spiking(bb, bidx, bound[best], "patient", roles, codes, words, D, xp)
                       if best is not None else None)
                rel_ok += int(ans == facts[tf]["patient"])

                # role query (who/what) on a random stored fact
                qf = int(rng.integers(n_facts)); qrole = RM.ROLES[int(rng.integers(3))]
                role_ok += int(RM.unbind_spiking(bb, bidx, bound[qf], qrole, roles, codes, words, D, xp)
                               == facts[qf][qrole])

                # control: a never-stored agent must find no fact
                stored_agents = {fc["agent"] for fc in facts}
                non = [w for w in words if w not in stored_agents]
                cue_bad = str(rng.choice(non)); bestc = None
                for f in range(n_facts):
                    if RM.unbind_spiking(bb, bidx, bound[f], "agent", roles, codes, words, D, xp) == cue_bad:
                        bestc = f; break
                ctrl_ok += int(bestc is None)
                tot += 1
            per_seed_rel.append(rel_ok / tot)
            per_seed_role.append(role_ok / tot)
            per_seed_ctrl.append(ctrl_ok / tot)
            print(f"  N={n_facts:>2} seed {seed}: relational={rel_ok/tot:.3f}  role={role_ok/tot:.3f}  "
                  f"control={ctrl_ok/tot:.3f}", flush=True)
        mr, mo, mc = np.mean(per_seed_rel), np.mean(per_seed_role), np.mean(per_seed_ctrl)
        passed = (min(per_seed_rel) >= 0.80) and (min(per_seed_role) >= 0.80) and (min(per_seed_ctrl) >= 0.80)
        overall[n_facts] = (mr, mo, mc, passed)
        print(f"  -> N={n_facts}: relational {mr:.3f}  role {mo:.3f}  control {mc:.3f}  "
              f"[{'PASS' if passed else 'below-bar'}] (min-across-seed bar 0.80)", flush=True)

    print("\nRESULT: KB capacity on the 320 substrate (mean relational / role / control, multi-seed):", flush=True)
    for n_facts, (mr, mo, mc, passed) in overall.items():
        print(f"  N={n_facts:>2} facts: rel {mr:.3f}  role {mo:.3f}  ctrl {mc:.3f}  {'PASS' if passed else 'BELOW'}",
              flush=True)
    max_pass = max([n for n, v in overall.items() if v[3]], default=0)
    print(f"VERDICT: reliable relational KB holds to N={max_pass} facts (multi-seed >= 0.80) on the full-320 "
          f"biological substrate." if max_pass else
          "VERDICT: even N=5 below bar -- characterise the cleanup limit.", flush=True)


if __name__ == "__main__":
    main()
