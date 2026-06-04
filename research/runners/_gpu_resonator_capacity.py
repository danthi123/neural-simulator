"""GPU resonator-capacity test: does raising the dimension D recover the F=3 two-attribute resonator as the
vocabulary grows? The capacity curve (_capacity_curve_probe) showed two-attribute is the LONE category that
degrades with vocabulary (everything else holds at fixed D), and CPU can't practically run the resonator at
D>=8192 -- exactly where a GPU helps. This isolates the resonator's matmul core on the GPU (CuPy) to map its
D-requirement at a fixed vocabulary, answering: is two-attribute scaling a (GPU-able) DIMENSION requirement or a
deeper CAPACITY-ALGORITHM limit?

This is the legitimate GPU use the scaling question motivates (the large-D regime; not the 320-scale agent,
which runs fine on CPU). Reuse-by-import of the agent's codebooks; the resonator is re-expressed in CuPy.

  python -m research.runners._gpu_resonator_capacity
"""
from __future__ import annotations
import json

import numpy as np

from research.runners.spiking_unified_agent import SpikingUnifiedAgent, _to_phasor
from research.runners.unified_agent_benchmark import build_vocab, FACTS_2ATTR


def _gpu_resonator3(cp, p, AMAT, NMAT, n_restarts, n_iter, D):
    """The F=3 resonator (adj1 ⊗ adj2 ⊗ noun) in CuPy. Returns (sorted-adj-indices, noun-index, residual)."""
    def unit(v):
        return v / (cp.abs(v) + 1e-12)
    Ma = AMAT.shape[1]
    cbs = [AMAT, AMAT, NMAT]
    rng = np.random.default_rng(53)
    best = None
    for _ in range(n_restarts):
        est = [unit(AMAT.sum(1) + 0.7 * AMAT[:, int(rng.integers(Ma))]),
               unit(AMAT.sum(1) + 0.7 * AMAT[:, int(rng.integers(Ma))]),
               unit(NMAT.sum(1))]
        for _ in range(n_iter):
            new = []
            for i in range(3):
                o = cp.ones(D, dtype=cp.complex128)
                for j in range(3):
                    if j != i:
                        o = o * est[j]
                new.append(unit(cbs[i] @ (cbs[i].conj().T @ (p * cp.conj(o)))))
            est = new
        resid = float(cp.abs(cp.vdot(est[0] * est[1] * est[2], p)) / D)
        a1 = int(cp.argmax(cp.abs(AMAT.conj().T @ est[0])))
        a2 = int(cp.argmax(cp.abs(AMAT.conj().T @ est[1])))
        n = int(cp.argmax(cp.abs(NMAT.conj().T @ est[2])))
        if best is None or resid > best[0]:
            best = (resid, tuple(sorted({a1, a2})), n)
    return best[1], best[2], best[0]


def main():
    import time
    import cupy as cp
    print("=== GPU resonator capacity vs D (vocab 640: 120 adj / 400 noun) ===", flush=True)
    print(f"  GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}\n", flush=True)
    nouns, verbs, adjs = build_vocab(400, 120, 120)
    out = {}
    for D in (2048, 4096, 8192, 16384, 32768):
        a = SpikingUnifiedAgent(nouns, verbs, adjs, n_dim=D, seed=42)   # CPU build of the codebooks
        AMAT = cp.asarray(a.AMAT)
        NMAT = cp.asarray(a.NMAT)
        ok, t0 = 0, time.perf_counter()
        for _, _, pa in FACTS_2ATTR:
            (a1, a2), nn = pa
            true = cp.asarray(_to_phasor(a.adj_sym[a1]) * _to_phasor(a.adj_sym[a2]) * _to_phasor(a.noun_sym[nn]))
            ai, ni, _ = _gpu_resonator3(cp, true, AMAT, NMAT, n_restarts=16, n_iter=150, D=D)
            ok += int(set(adjs[k] for k in ai) == {a1, a2} and nouns[ni] == nn)
        cp.cuda.Stream.null.synchronize()
        dt = time.perf_counter() - t0
        out[D] = {"clean_recover": ok, "secs": round(dt, 1)}
        print(f"  D={D:6}: resonator on 5 clean products -> {ok}/5   ({dt:.1f}s GPU)", flush=True)
        with open("research/findings/raw/gpu_resonator_capacity.json", "w") as f:
            json.dump(out, f, indent=2)

    print("\n  --- interpretation ---", flush=True)
    recov = [D for D, r in out.items() if r["clean_recover"] >= 4]
    if recov:
        print(f"    two-attribute RECOVERS at D>={min(recov)} -> it IS a dimension requirement; the D^2 cost "
              f"makes GPU the path for two-attribute composition at scale.", flush=True)
    else:
        print(f"    two-attribute does NOT recover even at D={max(out)} -> a deeper resonator CAPACITY limit; "
              f"the fix is algorithmic (sparse block codes), not more dimension/GPU.", flush=True)
    print("\n  wrote research/findings/raw/gpu_resonator_capacity.json", flush=True)


if __name__ == "__main__":
    main()
