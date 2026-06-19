"""SCOPING PROFILER (2026-06-19): per-section latency breakdown of a production OneBrainComposer query WITH the
A5 megakernel ON, to find the RESIDUAL bottleneck after the resonate loop was collapsed. Read-only / measurement.

Run:  SIM_BACKEND=cupy python -u -m research.runners._latency_profile_onebrain
"""
from __future__ import annotations
import time, sys
import numpy as np
import cupy as cp

from research.runners.one_brain_composer import OneBrainComposer
from sim.backend import is_gpu_backend


def sync():
    cp.cuda.Stream.null.synchronize()


def timeit(fn, n=5, warmup=2):
    for _ in range(warmup):
        fn(); sync()
    ts = []
    for _ in range(n):
        sync(); t0 = time.perf_counter()
        fn(); sync()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), float(np.min(ts)), float(np.max(ts))


# ---- instrument rf_set_complex_weights to separate Python-tuple-build vs CSR-build vs H2D ----
def patch_set_weights_timing(bridge):
    """Wrap rf_set_complex_weights to record (n_conns, build_ms) into bridge._sw_log."""
    import sim.bridge as B
    csp = B.csp
    orig = bridge.rf_set_complex_weights
    bridge._sw_log = []

    def timed(connections):
        t0 = time.perf_counter()
        n = bridge.core_config.num_neurons
        m = len(connections)
        rows = np.fromiter((int(post) for (post, pre, w) in connections), dtype=np.int32, count=m)
        cols = np.fromiter((int(pre) for (post, pre, w) in connections), dtype=np.int32, count=m)
        w_re = np.fromiter((float(complex(w).real) for (post, pre, w) in connections), dtype=np.float64, count=m)
        w_im = np.fromiter((float(complex(w).imag) for (post, pre, w) in connections), dtype=np.float64, count=m)
        t1 = time.perf_counter()                       # Python tuple-iter (host)
        r = cp.asarray(rows); c = cp.asarray(cols)
        wre_d = cp.asarray(w_re); wim_d = cp.asarray(w_im)
        bridge.cp_rf_w_re = csp.csr_matrix((wre_d, (r, c)), shape=(n, n))
        bridge.cp_rf_w_im = csp.csr_matrix((wim_d, (r, c)), shape=(n, n))
        sync()
        t2 = time.perf_counter()                       # H2D + CSR build (gpu)
        bridge._sw_log.append((m, (t1 - t0) * 1e3, (t2 - t1) * 1e3))
    bridge.rf_set_complex_weights = timed


def main():
    assert is_gpu_backend(), "need SIM_BACKEND=cupy"
    print("=" * 78)
    print("OneBrainComposer per-op latency profile — megakernel ON (production default)")
    print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
    print("=" * 78)

    SEED = 42
    for (D, K, label) in [(128, 8, "D=128 K=8 (Phase-2 validated)"),
                          (128, 32, "D=128 K=32 (store full)")]:
        print(f"\n##### {label} #####")
        comp = OneBrainComposer(seed=SEED, D=D, k_max=max(K, 8), enable_rf_cudagraph=True, enable_batched=True)
        # store K facts (cycle a small fact set)
        agents = ["dog", "cat", "apple", "river"]; actions = ["go", "run", "come", "stop"]
        patients = ["north", "south", "east", "west"]
        t_store0 = time.perf_counter()
        for i in range(K):
            comp.store(agents[i % 4], actions[(i // 4) % 4], patients[i % 4])
        sync()
        store_ms = (time.perf_counter() - t_store0) * 1e3
        print(f"  store {K} facts: {store_ms:7.1f} ms total ({store_ms / K:6.1f} ms/fact)")
        print(f"  bridge n_total={comp.n_total}  V={comp.V}  period={comp.period}")

        # ---- full query: query_patient (the hot path: batched read of all K blocks + cue match) ----
        def q():
            return comp.query_patient(agents[0], actions[0])
        med, mn, mx = timeit(q, n=7, warmup=3)
        print(f"  query_patient (full): median {med:7.1f} ms  (min {mn:.1f}, max {mx:.1f})")

        # ---- section breakdown of the batched read (_read_all_blocks) ----
        # Instrument: time the three phases of _read_all_blocks separately by monkeypatching set-weights timing.
        patch_set_weights_timing(comp.b)
        comp.b._sw_log.clear()
        # also time the whole _read_all_blocks + count resonate kernel launches
        sync(); t0 = time.perf_counter()
        _ = comp._read_all_blocks()
        sync(); read_ms = (time.perf_counter() - t0) * 1e3

        sw = comp.b._sw_log     # [(n_conns, host_tuple_ms, gpu_csr_ms), ...] one per rf_set_complex_weights
        n_sw = len(sw)
        host_build = sum(s[1] for s in sw); gpu_csr = sum(s[2] for s in sw)
        tot_conns = sum(s[0] for s in sw)
        print(f"  _read_all_blocks (batched): {read_ms:7.1f} ms")
        print(f"    set_complex_weights calls: {n_sw}  (total {tot_conns:,} conns built)")
        print(f"    -- host Python tuple-iter: {host_build:7.1f} ms  ({100*host_build/read_ms:4.1f}% of read)")
        print(f"    -- GPU H2D + CSR build:    {gpu_csr:7.1f} ms  ({100*gpu_csr/read_ms:4.1f}% of read)")
        per = [f"{s[0]}c:{s[1]:.0f}+{s[2]:.0f}ms" for s in sw]
        print(f"    per-call [{', '.join(per)}]")

        # ---- resonate-only cost (megakernel ON vs OFF) at this scale ----
        # Build a representative cleanup-sized weight set is already installed; time pure resonate_steps.
        nsteps = comp.period + 8
        def res_on():
            comp.b.core_config.enable_rf_cudagraph = True
            comp.b.rf_resonate_steps(nsteps)
        def res_off():
            comp.b.core_config.enable_rf_cudagraph = False
            comp.b.rf_resonate_steps(nsteps)
        # need a prior rf_kick so cp_rf_prev_im exists; do a trivial kick
        comp.b.rf_kick(np.zeros(comp.n_total, dtype=np.complex128), period=comp.period, lam=0.0,
                       neuron_mask=comp.rf_mask)
        mon, _, _ = timeit(res_on, n=5, warmup=2)
        moff, _, _ = timeit(res_off, n=5, warmup=2)
        comp.b.core_config.enable_rf_cudagraph = True
        print(f"  resonate {nsteps} steps: megakernel ON {mon:7.1f} ms | loop OFF {moff:7.1f} ms | speedup {moff/mon:.1f}x")

    print("\n" + "=" * 78)
    print("done")


if __name__ == "__main__":
    main()
