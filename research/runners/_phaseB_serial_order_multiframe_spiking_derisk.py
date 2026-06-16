"""CYCLE 106 — multi-frame serial order ON THE SPIKING SUBSTRATE: does frame-conditioned word order survive on
real spikes (the seed of syntax, on-substrate)?

The numpy multi-frame de-risk was GO (the CQ generator learns DISTINCT orders for F0=SVO vs F1, cross-frame 0.000).
Phase-B (spiking) confirmed single-frame ordering on the substrate (graded current -> rate ranking = order). This
runner combines them: per frame, the frame's primacy gradient is realized as GRADED CURRENT into the fact's
concept pools (the role at the frame's position 0 gets the most current), the per-pool spiking RATE ranking is the
emission order, and the SAME fact is driven under BOTH frames -> a DIFFERENT order each (the cross-frame control,
on spikes). Reuses the validated driven-pool bridge + the pre-registered anti-cheat harness.

GATE (>=6 seeds, FIXED g1_verdict bars 0.10/0.5): GO if, per frame, the emitted order clears floor 0.5 AND beats
the permuted-order control by >=10% AND beats the CROSS-FRAME control (the other frame's order on the same fact),
all >=5/6 seeds. GO => the spiking substrate produces frame-CONDITIONED serial order (syntax seed, on-substrate).
GPU (tiny bridge).
Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_serial_order_multiframe_spiking_derisk
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.song_g1_core import score_order, permuted_order_controls, g1_verdict  # noqa: E402
from research.runners._phaseB_serial_order_spiking_derisk import (  # noqa: E402  (reuse the de-risked bridge + read)
    build_pool_bridge, pool_rates, build_facts, N_ROLES, PRIMACY_pA, N_PERM)

FRAMES = {0: [0, 1, 2], 1: [2, 0, 1]}            # F0 = SVO ; F1 = a DISJOINT frame (patient, agent, action)


def emit_frame(bridge, pool_idx, trip, frame_order):
    """Drive the fact's fillers with the FRAME's primacy gradient (position 0 of the frame = highest current),
    read per-pool rates, emit fillers by rate DESC = the frame's serial order."""
    drive = {int(trip[role]): PRIMACY_pA[pos] for pos, role in enumerate(frame_order)}
    rate = pool_rates(bridge, pool_idx, drive)
    return [int(trip[role]) for role in sorted(frame_order, key=lambda role: -rate[int(trip[role])])]


def run_seed(seed):
    bridge, pool_idx = build_pool_bridge(seed)
    held = build_facts(seed)
    rng = np.random.default_rng(seed * 71 + 3)
    trues, perms, crosses = [], [], []
    for trip in held:
        for frame, order in FRAMES.items():
            intended = [trip[r] for r in order]
            cross_intended = [trip[r] for r in FRAMES[1 - frame]]
            emitted = emit_frame(bridge, pool_idx, trip, order)
            trues.append(score_order(emitted, intended))
            perms.append(max((score_order(emitted, c) for c in permuted_order_controls(intended, rng, N_PERM)),
                             default=0.0))
            crosses.append(score_order(emitted, cross_intended))
    t_true, t_perm, t_cross = float(np.mean(trues)), float(np.mean(perms)), float(np.mean(crosses))
    v = g1_verdict(t_true, t_perm, gate_cleared=True)
    gate = bool(v["gate"] and t_true >= t_cross * 1.10)
    print(f"  [seed {seed}] SPIKING frame-CQ true {t_true:.3f} vs perm {t_perm:.3f} vs CROSS-frame {t_cross:.3f} -> "
          f"{'PASS' if gate else 'FAIL'}", flush=True)
    return {"seed": seed, "true": t_true, "perm": t_perm, "cross": t_cross, "gate": gate}


def main():
    os.environ.setdefault("SIM_BACKEND", "cupy")
    t0 = time.time()
    print(f"[multi-frame serial-order SPIKING de-risk] does the spiking substrate produce FRAME-CONDITIONED order "
          f"(F0=SVO vs F1=[2,0,1]), beating permuted + cross-frame controls?", flush=True)
    rows = [run_seed(s) for s in (42, 43, 44, 45, 46, 47)]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    t_true, t_perm, t_cross = m("true"), m("perm"), m("cross")
    n_pass = sum(1 for r in rows if r["gate"])
    agg = g1_verdict(t_true, t_perm, gate_cleared=True)
    cross_ok = t_true >= t_cross * 1.10
    print(f"\n{'='*100}\n  MEAN (6 seeds): SPIKING frame-CQ true {t_true:.3f} vs perm {t_perm:.3f} vs CROSS-frame "
          f"{t_cross:.3f} ({n_pass}/6 PASS) | aggregate {agg['GATE']} ({agg['pct_over_permuted']:.0f}% over perm)",
          flush=True)
    print(f"{'='*100}", flush=True)
    if agg["gate"] and cross_ok and n_pass >= 5:
        print(f"  GO: the SPIKING substrate produces FRAME-CONDITIONED serial order -- per frame, true {t_true:.3f} "
              f">> permuted {t_perm:.3f} AND >> cross-frame {t_cross:.3f} (the SAME fact ordered DIFFERENTLY by "
              f"frame, on real spikes), {n_pass}/6 seeds. ==> the syntax seed (frame-dependent serial order) is "
              f"realized on the substrate, not just in numpy.", flush=True)
    elif agg["gate"] and not cross_ok:
        print(f"  PARTIAL: orders beat permuted but not cross-frame ({t_true:.3f} vs {t_cross:.3f}) on spikes -- "
              f"frame-conditioning weak on the substrate.", flush=True)
    else:
        print(f"  NEGATIVE: frame-conditioned order doesn't hold on spikes ({t_true:.3f} vs perm {t_perm:.3f}). "
              f"Honest negative.", flush=True)
    print(f"  Total elapsed: {time.time()-t0:.1f}s\n", flush=True)
    import json
    out = {"true": t_true, "perm": t_perm, "cross": t_cross, "n_pass": n_pass, "aggregate_gate": agg,
           "per_seed": rows}
    path = os.path.join(_REPO, "research", "findings", "raw", "_phaseB_serial_order_multiframe_spiking.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
