"""Cheap-first de-risk: a BLOCK-DIAGONAL BATCHED reslm reservoir state-collection — the mission-critical SCALE enabler.

WHY: the emergent-generation ladder's decisive question is SCALE (does the fixed-reservoir + shallow-read-out generator's
CE margin over the bigram GROW with co-scaled data?). The blocker is that the on-bridge reslm reservoir
(`ReservoirStates(OnBridgeLSM)`) collects each sentence's state per-token through `_run_one_simulation_step` (T_STEP steps
x tokens x sentences) — a heavy CPU job that caps the data scale. The SYNTHESIS (2026-07-11) named "sentence batching" as
the missing infra; THIS is it, reusing this session's validated block-diagonal batched-forward (M disjoint reservoir
copies on ONE bridge, stepped in LOCKSTEP -> the per-call overhead amortizes M x).

Each reservoir copy must be the SAME reservoir (same recurrent wiring + W_in + init), so we build a reference 1-copy
reservoir and TILE its wiring/W_in/init into M block-diagonal copies (a lone brain-region-per-copy would draw DIFFERENT
recurrent wiring). The reslm reservoir is already reproducible (OnBridgeLSM sets heterogeneity_seed; state Δ=0 across
builds), so no num_traits fix is needed here.

GATE: (1) CORRECTNESS -- batched copy m's spike-count read == the serial `final_state(U[m])` read (to tolerance, OU off);
(2) SPEEDUP -- the M-sentence batched collection is faster than M serial. If GO -> wire into
`_emerge_reservoir_lm_realcorpus_derisk.py` and run the decisive larger-data co-scale.

Run (numpy): E:/.../python.exe -m research.runners._reslm_batched_reservoir_derisk [--M 8] [--n 60] [--seed 42]
NO `sim/` edit -- pure reuse of SimulationBridge + inject_explicit_wiring + cp_firing_states.
"""
import argparse, time
import numpy as np


def build_reference(seed, n, in_dim, dt=0.5):
    """The canonical 1-copy reslm reservoir (reuse the shipped builder) -> (bridge, res_idx, W_in, snap)."""
    from research.runners._emerge82_onbridge_lsm_derisk import _build_reservoir_bridge
    return _build_reservoir_bridge(seed, n, in_dim, dt=dt)


def _extract_recurrent(bridge, res_idx):
    """Read the reference reservoir's recurrent connectivity as (pre_local, post_local, weight) in [0,n) local indices."""
    from sim.backend import to_host
    import scipy.sparse as sp
    conn = bridge.cp_connections
    csr = conn.get() if hasattr(conn, "get") else conn
    csr = sp.csr_matrix(csr)
    lo, hi = int(res_idx.min()), int(res_idx.max()) + 1
    sub = csr[lo:hi, lo:hi].tocoo()
    return sub.row.astype(int), sub.col.astype(int), sub.data.astype(float)


def build_batched(seed, n, in_dim, M, dt=0.5):
    """M block-diagonal copies of the REFERENCE reservoir on ONE bridge (identical wiring/W_in/init per copy)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.backend import get_backend, to_host
    xp, _ = get_backend()
    ref_b, ref_res, W_in, ref_snap = build_reference(seed, n, in_dim, dt=dt)
    pre, post, wv = _extract_recurrent(ref_b, ref_res)              # the reference recurrent wiring (local [0,n))

    cfg = CoreSimConfig()
    cfg.dt = float(dt)
    cfg.seed = cfg.ou_seed = cfg.heterogeneity_seed = seed
    cfg.enable_ou_process = False; cfg.enable_stdp = False; cfg.enable_hebbian_learning = False
    cfg.num_neurons = n * M
    rt = RuntimeState(); rt.actual_seed_used = seed
    b = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(), runtime_state=rt, gpu_config=GPUConfig())
    b._initialize_simulation_data()
    # block-diagonal recurrent wiring: copy c's neurons are [c*n, (c+1)*n)
    plan = {}
    for c in range(M):
        base = c * n
        plan[f"res{c}"] = dict(pre_indices=[int(base + p) for p in pre],
                               post_indices=[int(base + q) for q in post],
                               initial_weights=[float(w) for w in wv], plastic=False, conn_type="rec")
    b.inject_explicit_wiring(plan)
    # CLONE the reference's per-neuron init state (v/u/etc, first n) into every copy so all copies == the reference net.
    ref_host = {a: np.asarray(to_host(getattr(ref_b, a))).copy() for a in dir(ref_b)
                if a.startswith("cp_") and getattr(ref_b, a, None) is not None
                and hasattr(getattr(ref_b, a), "shape") and getattr(ref_b, a).ndim == 1
                and int(getattr(ref_b, a).shape[0]) == n}
    for a, refblock in ref_host.items():
        arr = getattr(b, a, None)
        if arr is not None and hasattr(arr, "shape") and arr.ndim == 1 and int(arr.shape[0]) == n * M:
            host = np.asarray(to_host(arr)).copy()
            for c in range(M):
                host[c * n:(c + 1) * n] = refblock
            setattr(b, a, xp.asarray(host).astype(arr.dtype))
    copy_res = [np.arange(c * n, (c + 1) * n) for c in range(M)]
    snap = {a: np.asarray(to_host(getattr(b, a))).copy() for a in dir(b)
            if a.startswith("cp_") and getattr(b, a, None) is not None and hasattr(getattr(b, a), "shape")}
    return b, copy_res, W_in, snap


def _restore(b, snap):
    from sim.backend import get_backend
    xp, _ = get_backend()
    for a, v in snap.items():
        arr = getattr(b, a, None)
        if arr is not None and hasattr(arr, "shape") and tuple(arr.shape) == tuple(v.shape):
            setattr(b, a, xp.asarray(v).astype(arr.dtype) if hasattr(arr, "dtype") else xp.asarray(v))


def serial_states(seed, n, in_dim, U_list, dt=0.5):
    """Reference: M sentences one-at-a-time through the shipped OnBridgeLSM.final_state."""
    from research.runners._emerge82_onbridge_lsm_derisk import OnBridgeLSM
    lsm = OnBridgeLSM(in_dim, seed=seed, n=n, dt=dt)
    return [lsm.final_state(U) for U in U_list]


def batched_states(b, copy_res, W_in, snap, U_list):
    """M sentences at once: drive each copy with its sentence (zero-pad after it ends), step lockstep, read per-copy."""
    from sim.backend import get_backend, to_host
    from research.runners._emerge82_onbridge_lsm_derisk import _T_STEP, _BIAS
    xp, _ = get_backend()
    M = len(copy_res); n = len(copy_res[0]); num = n * M
    _restore(b, snap)
    Lmax = max(len(U) for U in U_list)
    counts = np.zeros(num, np.float64); lens = np.array([len(U) for U in U_list])
    for t in range(Lmax):
        drive = np.zeros(num, np.float32)
        for c in range(M):
            if t < len(U_list[c]):
                drive[copy_res[c]] = (W_in @ U_list[c][t] + _BIAS).astype(np.float32)
        b.cp_external_input_current[:] = 0.0
        b.cp_external_input_current[:] = xp.asarray(drive) if xp is not None else drive
        for _ in range(_T_STEP):
            b._run_one_simulation_step()
            counts += np.asarray(to_host(b.cp_firing_states)).astype(np.float64)
    b.cp_external_input_current[:] = 0.0
    return [counts[copy_res[c]] / max(1, lens[c] * _T_STEP) for c in range(M)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=8); ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--in-dim", type=int, default=12); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    # M random "sentences" (variable length), one-hot-ish input vectors of in_dim
    U_list = []
    for _ in range(a.M):
        L = int(rng.integers(3, 8))
        U_list.append([np.eye(a.in_dim)[int(rng.integers(0, a.in_dim))] for _ in range(L)])

    t0 = time.time(); ser = serial_states(a.seed, a.n, a.in_dim, U_list); t_ser = time.time() - t0
    b, copy_res, W_in, snap = build_batched(a.seed, a.n, a.in_dim, a.M)
    t1 = time.time(); bat = batched_states(b, copy_res, W_in, snap, U_list); t_bat = time.time() - t1

    maxd = max(float(np.max(np.abs(np.asarray(bat[m]) - np.asarray(ser[m])))) for m in range(a.M))
    ok = maxd < 1e-6
    print(f"=== batched reslm reservoir de-risk (M={a.M}, n={a.n}) ===")
    print(f"  CORRECTNESS max|batched - serial| = {maxd:.2e}  -> {'MATCH (GO)' if ok else 'MISMATCH'}")
    print(f"  serial {t_ser:.2f}s  batched {t_bat:.2f}s  speedup {t_ser/max(t_bat,1e-9):.1f}x")


if __name__ == "__main__":
    main()
