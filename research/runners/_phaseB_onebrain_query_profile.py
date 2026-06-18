"""A5 step 0.5 -- PROFILE the OneBrainComposer query to ATTRIBUTE the residual 1.5x vs rf (onebrain 605 vs rf 413
ms/query). Breaks one batched `_read_all_blocks` into its components and times each: (a) the per-query CONNS-BUILD (the
Python list comprehensions for the unbind + cleanup conns), (b) the CSR-INSTALL (`rf_set_complex_weights` = np.fromiter
+ cupy csr_matrix), (c) the RESONATE (`rf_resonate_steps`). The verdict tells which optimization is worth it:
  - if (a)+(b) dominate -> PRECOMPUTE the fixed unbind/cleanup conns/CSRs (cheap, NO sim/ edit) closes it.
  - if (c) dominates -> the masked-MEGAKERNEL (lever 3, the deep sim/ edit) is the real lever.
Diagnose before fixing (systematic-debugging). GPU.
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_onebrain_query_profile --k 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "cupy")

from sim.backend import to_host, synchronize  # noqa: E402
from research.runners.one_brain_composer import OneBrainComposer, ROLES3  # noqa: E402

AG = ["dog", "cat", "bird", "river", "apple", "tree", "sun", "moon"]
AC = ["go", "come", "look", "stop", "swim", "walk", "run", "jump"]
PA = ["north", "east", "south", "west", "home", "hill", "lake", "sky"]


def _profiled_read(c, n_rep):
    """Replicate OneBrainComposer._read_all_blocks with per-component timing (summed over n_rep reps)."""
    comp, b, D, Pd, V, NP = c.comp, c.b, c.D, c.period, c.V, c.NP
    n = len(c.kb)
    t = {"conns_build": 0.0, "csr_install": 0.0, "resonate": 0.0, "kick_reset": 0.0, "read": 0.0}

    for _ in range(n_rep):
        # --- reconstruct window ---
        t0 = time.time()
        b.cp_membrane_potential_v[:] = 0.0; b.cp_recovery_variable_u[:] = 0.0
        kick = np.zeros(c.n_total, dtype=np.complex128)
        for i in range(n):
            kick[c.store_base + i * c.block] = 1.0
        synchronize(); t["kick_reset"] += time.time() - t0
        t0 = time.time(); b.rf_set_complex_weights(c.store_conns); synchronize(); t["csr_install"] += time.time() - t0
        t0 = time.time(); b.rf_kick(kick, period=Pd, lam=0.0, neuron_mask=c.rf_mask); synchronize(); t["kick_reset"] += time.time() - t0
        t0 = time.time(); b.rf_resonate_steps(Pd + 8); synchronize(); t["resonate"] += time.time() - t0
        # --- unbind window ---
        t0 = time.time()
        roles = ROLES3 + ["polarity"]
        unbind = []
        for i in range(n):
            trig = c.store_base + i * c.block
            for ri, role in enumerate(roles):
                zc = np.conj(comp._to_phasor(comp.roles[role]))
                qreg = c.bat_q_base + (i * 4 + ri) * D
                unbind += [(qreg + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
        t["conns_build"] += time.time() - t0
        t0 = time.time(); b.rf_set_complex_weights(unbind); synchronize(); t["csr_install"] += time.time() - t0
        t0 = time.time(); b.rf_resonate_steps(Pd + 8); synchronize(); t["resonate"] += time.time() - t0
        # --- cleanup window ---
        t0 = time.time()
        clean = []
        for i in range(n):
            cblk = c.bat_c_base + i * c.cb
            for ri in range(3):
                qreg = c.bat_q_base + (i * 4 + ri) * D
                for j in range(V):
                    cc = np.conj(comp._to_phasor(comp.concepts[c.words[j]]))
                    clean += [(cblk + ri * V + j, qreg + k, complex(cc[k])) for k in range(D)]
            qreg_p = c.bat_q_base + (i * 4 + 3) * D
            for j in range(NP):
                cc = np.conj(comp._to_phasor(comp.concepts[c.pol_words[j]]))
                clean += [(cblk + 3 * V + j, qreg_p + k, complex(cc[k])) for k in range(D)]
        t["conns_build"] += time.time() - t0
        t0 = time.time(); b.rf_set_complex_weights(clean); synchronize(); t["csr_install"] += time.time() - t0
        t0 = time.time(); b.rf_resonate_steps(1); synchronize(); t["resonate"] += time.time() - t0
        t0 = time.time(); _ = np.asarray(to_host(b.cp_membrane_potential_v)).astype(float); t["read"] += time.time() - t0
    return {k: 1000.0 * v / n_rep for k, v in t.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-rep", type=int, default=3)
    args = ap.parse_args()
    k = min(args.k, len(AG))
    vocab = AG + AC + PA
    print(f"[onebrain query profile] K={k}, attributing the residual vs rf\n", flush=True)
    c = OneBrainComposer(seed=args.seed, D=128, vocab=vocab)
    for i in range(k):
        c.store(AG[i], AC[i], PA[i])
    t = _profiled_read(c, args.n_rep)
    total = sum(t.values())
    print(f"  per-query breakdown (ms, mean of {args.n_rep}):", flush=True)
    for kk in ("conns_build", "csr_install", "resonate", "kick_reset", "read"):
        print(f"    {kk:14s} {t[kk]:7.1f} ms  ({100*t[kk]/max(total,1e-9):4.1f}%)", flush=True)
    print(f"    {'TOTAL':14s} {total:7.1f} ms", flush=True)
    host_side = t["conns_build"] + t["csr_install"]
    print(f"\n  HOST-side (conns_build + csr_install) = {host_side:.1f} ms ({100*host_side/max(total,1e-9):.0f}%); "
          f"RESONATE = {t['resonate']:.1f} ms ({100*t['resonate']/max(total,1e-9):.0f}%)", flush=True)
    if host_side > t["resonate"]:
        print(f"  ==> the HOST-side conns/CSR build DOMINATES -> the CHEAP fix (precompute the fixed unbind+cleanup "
              f"CSRs once, NO sim/ edit) closes most of the residual; lever 3 (megakernel) is secondary.", flush=True)
    else:
        print(f"  ==> the RESONATE DOMINATES -> the masked-MEGAKERNEL (lever 3, the deep sim/ edit) is the real lever; "
              f"the precompute is marginal.", flush=True)


if __name__ == "__main__":
    main()
