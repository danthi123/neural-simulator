"""Finer-grained section profiler (2026-06-19): break a OneBrainComposer batched read into its actual
phases (weight-build vs kick vs resonate vs cleanup readout), and profile the megakernel per-call overhead
(the double-buffer astype/copy-back) separately from the per-step kernel cost.

Run:  SIM_BACKEND=cupy python -u -m research.runners._latency_profile_sections
"""
from __future__ import annotations
import time
import numpy as np
import cupy as cp

from research.runners.one_brain_composer import OneBrainComposer
from research.runners.rf_phasor_composer import RFPhasorComposer
from sim.backend import to_host, is_gpu_backend


def sync():
    cp.cuda.Stream.null.synchronize()


def med(fn, n=7, warmup=3):
    for _ in range(warmup):
        fn(); sync()
    ts = []
    for _ in range(n):
        sync(); t0 = time.perf_counter(); fn(); sync(); ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main():
    assert is_gpu_backend()
    print("device:", cp.cuda.runtime.getDeviceProperties(0)["name"].decode())
    SEED, D, K = 42, 128, 16
    comp = OneBrainComposer(seed=SEED, D=D, k_max=32, enable_rf_cudagraph=True, enable_batched=True)
    agents = ["dog", "cat", "apple", "river"]; actions = ["go", "run", "come", "stop"]; pats = ["north", "south", "east", "west"]
    for i in range(K):
        comp.store(agents[i % 4], actions[(i // 4) % 4], pats[i % 4])
    sync()
    b = comp.b; D = comp.D; Pd = comp.period; V = comp.V; NP = comp.NP; n = len(comp.kb)
    print(f"K={K} D={D} n_total={comp.n_total} V={V}")

    # ----- megakernel per-call overhead: a no-op resonate (0 steps surrogate) vs 1 step vs N steps -----
    comp.b.rf_kick(np.zeros(comp.n_total, dtype=np.complex128), period=Pd, lam=0.0, neuron_mask=comp.rf_mask)
    sync()
    for ns in (1, 8, 50, 208):
        t = med(lambda ns=ns: comp.b.rf_resonate_steps(ns), n=5, warmup=2)
        print(f"  megakernel resonate {ns:3d} steps: {t:7.2f} ms  ({t/ns:.3f} ms/step incl. fixed setup)")
    # fixed per-call overhead estimate from the slope:
    t1 = med(lambda: comp.b.rf_resonate_steps(1), n=5, warmup=2)
    t208 = med(lambda: comp.b.rf_resonate_steps(208), n=5, warmup=2)
    per_step = (t208 - t1) / 207.0
    fixed = t1 - per_step
    print(f"  -> per-step {per_step*1000:.1f} us, fixed per-call setup ~{fixed:.2f} ms (astype copy + copy-back)")

    # ----- decompose _read_all_blocks into build / kick / resonate / readout phases -----
    from sim.backend import to_host as _toh
    ROLES4 = ["agent", "action", "patient", "polarity"]

    # phase A: kick+settle (1 weight set = store_conns; already-built list is reused)
    def build_store_conns():
        return list(comp.store_conns)
    # We time the three rf_set_complex_weights connection LISTS being BUILT (Python) separately:
    # 1) store_conns is prebuilt (held in self.store_conns) -> tiny
    # 2) unbind list  : n*4*D tuples
    # 3) clean list   : n*(3V+NP)*D tuples
    comp_ = comp
    def build_unbind():
        unbind = []
        for i in range(n):
            trig = comp_.store_base + i * comp_.block
            for ri, role in enumerate(ROLES4):
                zc = np.conj(comp_.comp._to_phasor(comp_.comp.roles[role]))
                qreg = comp_.bat_q_base + (i * 4 + ri) * D
                unbind += [(qreg + k, trig + 1 + k, complex(zc[k])) for k in range(D)]
        return unbind
    def build_clean():
        clean = []
        for i in range(n):
            cblk = comp_.bat_c_base + i * comp_.cb
            for ri in range(3):
                qreg = comp_.bat_q_base + (i * 4 + ri) * D
                for j in range(V):
                    cc = np.conj(comp_.comp._to_phasor(comp_.comp.concepts[comp_.words[j]]))
                    clean += [(cblk + ri * V + j, qreg + k, complex(cc[k])) for k in range(D)]
            qreg_p = comp_.bat_q_base + (i * 4 + 3) * D
            for j in range(NP):
                cc = np.conj(comp_.comp._to_phasor(comp_.comp.concepts[comp_.pol_words[j]]))
                clean += [(cblk + 3 * V + j, qreg_p + k, complex(cc[k])) for k in range(D)]
        return clean
    tb_store = med(build_store_conns, n=5, warmup=2)
    tb_unbind = med(build_unbind, n=5, warmup=2)
    tb_clean = med(build_clean, n=5, warmup=2)
    nu = len(build_unbind()); nc = len(build_clean())
    print(f"\n  CONNECTION-LIST BUILD (pure Python host, the tuple comprehensions):")
    print(f"    store_conns (reuse): {tb_store:6.2f} ms ({len(comp.store_conns)} conns)")
    print(f"    unbind list:         {tb_unbind:6.2f} ms ({nu:,} conns)")
    print(f"    clean list:          {tb_clean:6.2f} ms ({nc:,} conns)   <-- the cleanup explosion")

    # phase: rf_set_complex_weights on the clean list (the biggest)
    clean = build_clean()
    t_setclean = med(lambda: comp.b.rf_set_complex_weights(clean), n=5, warmup=2)
    print(f"  rf_set_complex_weights(clean): {t_setclean:6.2f} ms (H2D + 2x CSR build for {nc:,} conns)")

    # full batched read for reference
    t_read = med(lambda: comp._read_all_blocks(), n=5, warmup=3)
    print(f"\n  _read_all_blocks total: {t_read:6.1f} ms")

    # ----- RFPhasorComposer reference (numpy-CPU oracle is the default; here the GPU rf composer) -----
    print("\n  --- RFPhasorComposer (GPU, megakernel ON) reference query ---")
    ref = RFPhasorComposer(seed=SEED, D=D, enable_rf_cudagraph=True, enable_substrate_store=False)
    for i in range(K):
        ref.store(agents[i % 4], actions[(i // 4) % 4], pats[i % 4])
    t_ref = med(lambda: ref.query_patient(agents[0], actions[0]), n=5, warmup=3)
    print(f"  RFPhasorComposer query_patient: {t_ref:6.1f} ms")


if __name__ == "__main__":
    main()
