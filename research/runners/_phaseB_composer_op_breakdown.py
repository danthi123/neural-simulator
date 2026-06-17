"""Latency diagnostic (owner-prioritized 2026-06-17): WHERE does the ~160 ms per composer op go? Break one
RFPhasorComposer bind/unbind into its sub-steps so the orchestration-overhead fix targets the real culprit.

Each composer op (_resonate) does: build the 512-entry conns list (host Python) -> rf_set_complex_weights (rebuild
the sparse complex synapses on GPU, FRESH each op) -> rf_kick -> rf_resonate_steps(period+8 = 208 sequential GPU
step launches) -> rf_read_phases (GPU->CPU sync). This times each, GPU-synced, to attribute the ~160 ms.

Run:  SIM_BACKEND=cupy python -u -m research.runners._phaseB_composer_op_breakdown
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

os.environ.setdefault("SIM_BACKEND", "cupy")

from sim.backend import synchronize  # noqa: E402
from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402


def _time(fn, n, warm=3):
    for _ in range(warm):
        fn()
    synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    synchronize()
    return (time.time() - t) / n * 1000.0


def main():
    D = 512
    N = 40
    comp = RFPhasorComposer(seed=42, D=D, vocab=[f"c{i}" for i in range(64)])
    n = 2 * D
    period = comp.period

    zf = comp._to_phasor(comp.concepts["c0"])
    zr = comp._to_phasor(comp.roles["agent"])
    kick = np.zeros(2 * D, dtype=np.complex128)
    kick[:D] = zf

    # warm the cache + kernels with a full op
    comp._bind(comp.roles["agent"], comp.concepts["c0"])
    synchronize()
    b = comp._bridge_cache[n]

    # host-side conns construction (done inside _bind EVERY call)
    t_conns = _time(lambda: [(D + k, k, zr[k]) for k in range(D)], N)
    conns = [(D + k, k, zr[k]) for k in range(D)]

    t_setw = _time(lambda: b.rf_set_complex_weights(conns), N)
    t_kick = _time(lambda: b.rf_kick(kick, period=period, lam=0.0), N)
    t_res = _time(lambda: b.rf_resonate_steps(period + 8), N)
    t_read = _time(lambda: b.rf_read_phases(), N)
    t_full = _time(lambda: comp._bind(comp.roles["agent"], comp.concepts["c0"]), N)

    total = t_conns + t_setw + t_kick + t_res + t_read
    print(f"\n[composer op breakdown] D={D}, resonate steps={period + 8}, n_neurons={n}", flush=True)
    print(f"  {'conns build (host Python list)':36s} {t_conns:8.2f} ms  ({100*t_conns/total:4.1f}%)", flush=True)
    print(f"  {'rf_set_complex_weights (GPU rebuild)':36s} {t_setw:8.2f} ms  ({100*t_setw/total:4.1f}%)", flush=True)
    print(f"  {'rf_kick':36s} {t_kick:8.2f} ms  ({100*t_kick/total:4.1f}%)", flush=True)
    print(f"  {'rf_resonate_steps (208 launches)':36s} {t_res:8.2f} ms  ({100*t_res/total:4.1f}%)  "
          f"= {t_res/(period+8)*1000:6.1f} us/step", flush=True)
    print(f"  {'rf_read_phases (GPU->CPU sync)':36s} {t_read:8.2f} ms  ({100*t_read/total:4.1f}%)", flush=True)
    print(f"  {'-'*36} {'-'*8}", flush=True)
    print(f"  {'sum of parts':36s} {total:8.2f} ms", flush=True)
    print(f"  {'full _bind (measured)':36s} {t_full:8.2f} ms", flush=True)

    # The dominant term tells the fix:
    parts = {"conns_build": t_conns, "rf_set_complex_weights": t_setw, "rf_kick": t_kick,
             "rf_resonate_steps": t_res, "rf_read_phases": t_read}
    dom = max(parts, key=parts.get)
    fixes = {
        "rf_resonate_steps": "fuse the 208-step loop into ONE kernel/CUDA-graph launch (or shorten the period); "
                             "this is the per-op launch-bound killer.",
        "rf_set_complex_weights": "PERSIST the synapse weights instead of rebuilding them every op (cache per "
                                  "role/codebook); rebuild only when the codebook changes.",
        "conns_build": "build the conns ONCE (vectorized arrays), not a 512-tuple Python list per op.",
        "rf_read_phases": "keep the pipeline on-GPU and sync ONCE per turn, not once per op.",
        "rf_kick": "fold the kick into the resonate launch.",
    }
    print(f"\n  DOMINANT: {dom} ({parts[dom]:.1f} ms, {100*parts[dom]/total:.0f}%) -> FIX: {fixes[dom]}", flush=True)
    print(f"  Plus: a query batches over the KB -> stack ALL facts' unbinds into ONE resonate (KB ops -> 1 op).",
          flush=True)

    import json
    out = {"D": D, "resonate_steps": period + 8, "n_neurons": n, "parts_ms": parts,
           "sum_ms": total, "full_bind_ms": t_full, "dominant": dom}
    path = os.path.join(_REPO, "research", "findings", "raw", "_composer_op_breakdown.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"  [saved] {path}", flush=True)


if __name__ == "__main__":
    main()
