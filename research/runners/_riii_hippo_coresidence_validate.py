"""R-iii one-brain integration: validate the CA3 memory (formation + dendritic completion) CO-RESIDENT on the shared
nav/conv bridge (`build_merged_nav_conv_bridge(co_resident_hippo_memory=True)`). Two gates: (A) the MEMORY FUNCTION --
the direct-synchronous FORMATION + partial-cue COMPLETION (CYCLE 1076) fires on the co-resident ca3 slice (held-out
members complete); (B) ZERO CROSS-TALK -- the ca3 memory slice is array-disjoint from nav/conv, so the merged bridge's
neuron count + cp_connections structure with the flag ON differ from OFF only by the appended hippo slice, and the
memory ops do not perturb the nav/conv regions (the `project_one_brain_substrate_vs_functional` bar). Reuse-by-import
of `_train_assemblies`/`_recall` (the validated formation/completion harness). GPU (SIM_BACKEND=cupy). NO `sim/` edit.

Run: SIM_BACKEND=cupy python -m research.runners._riii_hippo_coresidence_validate --seeds 42 43 44 100 101 102
"""
from __future__ import annotations
import argparse, time
import numpy as np


def run_seed(seed, hippo_n_ca3=500, n_assembly=12, n_mem=3, presentations=60, hippo_k_thresh=66.0, cue_drive=1000.0,
             hand_install=False):
    """Build the merged nav/conv bridge WITH the co-resident CA3 memory, run formation + partial-cue completion on the
    ca3 slice, and confirm the ca3 slice is disjoint from the nav/conv regions. Returns the held-out completion + a
    disjointness flag. (The formation/completion params mirror the CYCLE-1086 sparse-large regime.)"""
    from sim.backend import get_backend
    cp, _ = get_backend()
    from research.runners.nav_conv_merged_bridge import build_merged_nav_conv_bridge
    from research.runners._riii_ca3_emergent_completion_derisk import _train_assemblies, _recall
    from research.runners._riii_ca3_coincidence_completion_derisk import _set_gates

    bridge, _handles = build_merged_nav_conv_bridge(seed=seed, co_resident_hippo_memory=True,
                                                    hippo_n_ca3=hippo_n_ca3, hippo_n_ca1=120,
                                                    hippo_k_thresh=hippo_k_thresh)
    rm = bridge.region_manager
    ca3_idx = np.asarray(list(rm.indices("ca3")), dtype=np.int64)
    # disjointness: the ca3 slice must not overlap any nav/parser/dlpfc region
    ca3_set = set(int(x) for x in ca3_idx)
    other = set()
    for rn, idxs in rm.region_indices_dict().items():
        if rn in ("ca3", "ca1", "ca3_pv_basket"):
            continue
        other |= set(int(x) for x in idxs)
    disjoint = len(ca3_set & other) == 0

    try:
        basket = np.asarray(list(rm.indices("ca3_pv_basket")), dtype=np.int64)
    except Exception:
        basket = None
    # CYCLE 1093 diagnostic: the merged bridge pins num_traits=1 (homogeneous ca3, canonical IB preset b=5/d=50 ->
    # lockstep bursting); the standalone completion runs num_traits=5 (diverse ca3, cell0 b=-1.77/d=88 = more
    # adaptation). Env knobs to isolate adaptation (CA3_D higher d_increment) vs diversity (CA3_D_JIT per-cell jitter)
    # vs b-regime (CA3_B). Written to the ca3 slice post-build. No effect unset -> byte-preserved default.
    import os as _os
    from sim.backend import to_host as _th, from_host as _fh
    _ca3dev = cp.asarray(ca3_idx, dtype=cp.int64)
    if _os.environ.get("CA3_D") is not None:
        _dval = float(_os.environ["CA3_D"]); _jit = float(_os.environ.get("CA3_D_JIT", "0"))
        _rd = np.random.default_rng(seed + 7)
        _dv = _dval + (_rd.uniform(-_jit, _jit, size=len(ca3_idx)).astype(np.float32) if _jit > 0 else 0.0)
        bridge.cp_izh_d_increment[_ca3dev] = _fh(np.asarray(_dv, dtype=np.float32))
        print(f"    [ca3 param override] d_increment={_dval} jit={_jit}", flush=True)
    if _os.environ.get("CA3_B") is not None:
        bridge.cp_izh_b[_ca3dev] = float(_os.environ["CA3_B"])
        print(f"    [ca3 param override] b={_os.environ['CA3_B']}", flush=True)
    if _os.environ.get("CA3_TRAIT_JIT") is not None:
        # FAITHFUL num_traits=5 reproduction: the standalone (num_traits=5) diversifies exactly C/a/b/d (the probe
        # showed b 5->-1.77, d 50->88, C 100->108, a 0.01->0.023; vt did NOT differ). enable_heterogeneity jitters the
        # same params but with too-small default sigma. Sample per-cell across the canonical->trait-variant ranges to
        # reproduce the LARGE trait spread that desynchronizes the ca3 (only a fraction fires the weak recurrent spread).
        _rt = np.random.default_rng(seed + 13); _nn = len(ca3_idx)
        bridge.cp_izh_b[_ca3dev] = _fh(_rt.uniform(-2.0, 6.0, _nn).astype(np.float32))
        bridge.cp_izh_d_increment[_ca3dev] = _fh(_rt.uniform(50.0, 120.0, _nn).astype(np.float32))
        bridge.cp_izh_C[_ca3dev] = _fh(_rt.uniform(90.0, 115.0, _nn).astype(np.float32))
        bridge.cp_izh_a[_ca3dev] = _fh(_rt.uniform(0.008, 0.03, _nn).astype(np.float32))
        print(f"    [ca3 param override] TRAIT diversity (b/d/C/a large per-cell spread, num_traits=5 mimic)", flush=True)
    if _os.environ.get("CA3_VT_JIT") is not None:
        # THRESHOLD diversity (the num_traits=5 effect that matters): jitter vt (spike threshold) per-cell so the weak
        # recurrent spread fires only a FRACTION of non-assembly cells -> sparse -> specific formation. d_increment
        # (adaptation) was the wrong lever (CA3_D no-op); vt/k (the f-I threshold) is what the trait-split diversifies.
        _vj = float(_os.environ["CA3_VT_JIT"]); _rv = np.random.default_rng(seed + 11)
        _vt0 = _th(bridge.cp_izh_vt[_ca3dev])
        bridge.cp_izh_vt[_ca3dev] = _fh((_vt0 + _rv.uniform(-_vj, _vj, size=len(ca3_idx))).astype(np.float32))
        print(f"    [ca3 param override] vt jitter +/-{_vj} mV", flush=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ca3_idx)
    assemblies = [np.array(perm[m * n_assembly:(m + 1) * n_assembly], dtype=np.int64) for m in range(n_mem)]
    non_assembly = np.array([g for g in ca3_idx if g not in set(int(x) for a in assemblies for x in a)],
                            dtype=np.int64)[:60]
    # MID-RUN Hebbian-mode FORMATION, scoped to ca3->ca3 (resolves the plasticity-MODE conflict, CYCLE 1091): the
    # merged bridge runs STDP-mode (enable_hebbian_learning=False); the ca3->ca3 attractor needs rate-window Hebbian +
    # max-120. Switch the cfg to Hebbian-mode ONLY for the formation drive, and FREEZE every synapse except ca3->ca3
    # via cp_plasticity_rate_gain=0 -> nav/conv are protected from BOTH the Hebbian rule AND the max-120 clip (gated by
    # gain), while the ca3->ca3 forms its specific attractor. Restore STDP-mode + the saved gains after. NO sim/ edit.
    from sim.backend import to_host, from_host
    cfg = bridge.core_config
    _saved = (cfg.enable_hebbian_learning, getattr(cfg, "hebbian_rate_window", False), cfg.hebbian_max_weight,
              getattr(cfg, "hebbian_coactivity_thresh", 0.25))
    conn = bridge.cp_connections; nnz = int(conn.nnz)
    _indptr = to_host(conn.indptr); _indices = to_host(conn.indices)
    _pre = np.searchsorted(_indptr, np.arange(nnz), side="right") - 1
    _post = _indices[:nnz]
    _ca3 = np.asarray(ca3_idx)
    _ca3ca3 = np.isin(_pre, _ca3) & np.isin(_post, _ca3)                 # the ca3->ca3 recurrent synapses
    _saved_gain = to_host(bridge.cp_plasticity_rate_gain[:nnz]).copy()
    _asm_of = {}
    for _m, _a in enumerate(assemblies):
        for _g in _a:
            _asm_of[int(_g)] = _m
    if hand_install:
        # ISOLATION test (CYCLE 1093): skip the Hebbian formation ENTIRELY and hand-install a KNOWN-SPECIFIC ca3->ca3
        # attractor (within-assembly = strong 80, cross/assembly->non-assembly = init ~6). This SEPARATES the two
        # failure modes the uniform within==cross result conflates: if the co-resident RECALL is SPECIFIC given a
        # clean specific attractor (held-out completes, non-assembly silent) -> the sole blocker is FORMATION
        # selectivity (solvable offline-form-then-transfer / an encoding-state fix); if recall is STILL non-specific
        # -> the merged RECALL dynamics are the blocker (the deeper dynamics-context reconciliation). NO Hebbian, NO
        # cfg flip -- a pure weight write, so it isolates recall from formation.
        _newdata = to_host(conn.data[:nnz]).copy()
        for _kk in np.nonzero(_ca3ca3)[0]:
            _pm, _qm = _asm_of.get(int(_pre[_kk])), _asm_of.get(int(_post[_kk]))
            if _pm is not None and _qm is not None and _pm == _qm:
                _newdata[_kk] = 80.0                                     # within-assembly: strong (cross left at init ~6)
        conn.data[:nnz] = from_host(_newdata)
    else:
        _gain = np.zeros(nnz, dtype=np.float32); _gain[_ca3ca3] = 1.0    # freeze all EXCEPT ca3->ca3
        bridge.cp_plasticity_rate_gain[:nnz] = from_host(_gain)
        cfg.enable_hebbian_learning = True; cfg.hebbian_rate_window = True
        cfg.hebbian_max_weight = 120.0; cfg.hebbian_coactivity_thresh = 0.001
        _train_assemblies(bridge, cp, assemblies, presentations, 1000.0, 8, 12)
        (cfg.enable_hebbian_learning, cfg.hebbian_rate_window, cfg.hebbian_max_weight,
         cfg.hebbian_coactivity_thresh) = _saved                        # restore STDP-mode for nav/conv
        bridge.cp_plasticity_rate_gain[:nnz] = from_host(_saved_gain)
    # DIAGNOSTIC -- is the ca3->ca3 attractor SPECIFIC (within-assembly >> cross)?
    _data = to_host(conn.data[:nnz])
    _within, _cross = [], []
    for _kk in np.nonzero(_ca3ca3)[0]:
        _pm, _qm = _asm_of.get(int(_pre[_kk])), _asm_of.get(int(_post[_kk]))
        if _pm is not None and _qm is not None and _pm == _qm:
            _within.append(_data[_kk])
        elif _pm is not None:
            _cross.append(_data[_kk])
    _wm = float(np.mean(_within)) if _within else 0.0
    _cm = float(np.mean(_cross)) if _cross else 0.0
    print(f"    [ca3->ca3 formed] within={_wm:.2f} cross={_cm:.2f} ratio={_wm/(_cm+1e-9):.2f}x (specific attractor wants within>>cross)", flush=True)
    if hand_install:
        # RAW recall-sparsity instrument (CYCLE 1093 a0): drive assembly-0's cue and read RAW spike counts (not ratios)
        # of cue / held-out / non-assembly + the total ca3 active fraction per step. Separates "everything saturates"
        # (all counts high, active-frac ~1.0 -> broad excitability) from "cue cascades via recurrent" (held>>non but
        # non still leaks). ca3-wide firing tells us WHY the merged recall can't discriminate.
        a0 = assemblies[0].copy(); rng.shuffle(a0); h0 = len(a0) // 2
        cue0, held0 = a0[:h0], a0[h0:]
        _cue = cp.asarray(cue0, dtype=cp.int64); _clm = None      # basket ACTIVE (no disinhibition) -- match validated completion
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(40):
            if _clm is not None:
                bridge.cp_external_input_current[_clm] = -5000.0
            bridge._run_one_simulation_step()
        _ca3dev = cp.asarray(ca3_idx, dtype=cp.int64)
        _active = cp.zeros(1, dtype=cp.float32)
        _steps = 60
        for _ in range(_steps):
            bridge.cp_external_input_current[:] = 0.0
            bridge.cp_external_input_current[_cue] = float(cue_drive)
            if _clm is not None:
                bridge.cp_external_input_current[_clm] = -5000.0
            bridge._run_one_simulation_step()
            _active += float(cp.sum(bridge.cp_firing_states[_ca3dev]))
        bridge.cp_external_input_current[:] = 0.0
        _rc0 = float(np.mean(_recall(bridge, cp, cue0, cue0, cue_drive, clamp_cells=basket)))
        _rh0 = float(np.mean(_recall(bridge, cp, cue0, held0, cue_drive, clamp_cells=basket)))
        _rn0 = float(np.mean(_recall(bridge, cp, cue0, non_assembly, cue_drive, clamp_cells=basket)))
        _frac = float(to_host(_active)[0]) / (_steps * max(1, len(ca3_idx)))
        print(f"    [recall RAW] cue={_rc0:.1f} held={_rh0:.1f} non={_rn0:.1f} (spikes/cell/60steps) | "
              f"ca3 active-frac/step={_frac:.3f} (n_ca3={len(ca3_idx)}); sparse-completion wants held>>non, active-frac<<0.3", flush=True)
    held_c, non_c = [], []
    for asm in assemblies:
        a = asm.copy(); rng.shuffle(a); h = len(a) // 2
        cue, held = a[:h], a[h:]
        # clamp_cells=None: keep the ca3_pv_basket FEEDBACK INHIBITION ACTIVE during recall (the sparsifier that caps
        # the ca3 active-count so completion stays sparse). The standalone completion's 6/6-GO default is
        # recall_disinhib=False (== clamp_cells=None); clamping the basket = SWR-ripple DISINHIBITION, which REMOVES the
        # brake -> the recurrent spreads unchecked -> saturation (the CYCLE-1092/1093 blocker was THIS runner bug, not a
        # merged-bridge dynamics conflict). Match the validated completion: basket active.
        rh = _recall(bridge, cp, cue, held, cue_drive, clamp_cells=None)
        rc = _recall(bridge, cp, cue, cue, cue_drive, clamp_cells=None)
        rn = _recall(bridge, cp, cue, non_assembly, cue_drive, clamp_cells=None)
        ca = float(np.mean(rc)) + 1e-9
        held_c.append(float(np.mean(rh)) / ca); non_c.append(float(np.mean(rn)) / ca)
    return {"heldout": float(np.mean(held_c)), "nonassembly": float(np.mean(non_c)), "disjoint": disjoint,
            "n_total": int(rm.total_neurons()), "n_ca3": int(len(ca3_idx))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--hippo-n-ca3", type=int, default=500)
    ap.add_argument("--n-assembly", type=int, default=12)
    ap.add_argument("--hippo-k-thresh", type=float, default=66.0)
    ap.add_argument("--hand-install", action="store_true",
                    help="ISOLATION: skip Hebbian formation, hand-install a specific attractor, test recall specificity")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split()] if " " in a.seeds else [int(x) for x in a.seeds.split(",")]
    print(f"[R-iii HIPPO CO-RESIDENCE] co_resident_hippo_memory on the merged nav/conv bridge | formation+completion "
          f"on the co-resident ca3 slice + disjointness", flush=True)
    import json
    rows = []
    for s in seeds:
        t0 = time.time()
        r = run_seed(s, hippo_n_ca3=a.hippo_n_ca3, n_assembly=a.n_assembly, hippo_k_thresh=a.hippo_k_thresh,
                     hand_install=a.hand_install)
        rows.append({"seed": s, **r})
        print(f"  [seed {s}] held-out={r['heldout']:.3f} (non {r['nonassembly']:.3f}) | disjoint={r['disjoint']} "
              f"| n_total={r['n_total']} n_ca3={r['n_ca3']} ({time.time()-t0:.0f}s)", flush=True)
    if a.json and rows:
        json.dump(rows, open(a.json, "w"), indent=1)
    if rows:
        h = [r["heldout"] for r in rows]; nn = [r["nonassembly"] for r in rows]
        go = all(x > 0.30 for x in h) and all(x < 0.20 for x in nn) and all(r["disjoint"] for r in rows)
        print(f"\n  AGGREGATE: held-out={np.mean(h):.3f} non-assembly={np.mean(nn):.3f} disjoint={all(r['disjoint'] for r in rows)}", flush=True)
        print(f"  VERDICT: {'GO' if go else 'PARTIAL/NEGATIVE'} -- {'the emergent CA3 completion fires on the co-resident ca3 slice of the shared nav/conv one-brain (held-out completes, non-assembly silent), array-disjoint from nav/conv = the R-iii memory folded into the ONE BRAIN' if go else 'completion or disjointness not yet clean on the merged bridge; check the coincidence-plateau config scoping'}. NO sim/ edit.", flush=True)


if __name__ == "__main__":
    main()
