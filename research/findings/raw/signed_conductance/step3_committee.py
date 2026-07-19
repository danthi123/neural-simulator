"""Feature-conditioning attack (Design-D RANK-5): an on-substrate RESERVOIR COMMITTEE. M independent reservoirs (distinct
draws), features CONCATENATED, one Ws fit on the concatenation, the committed POSITIVE read (Ws_shifted, floor 150) reads all
M*RES_N neurons -> more spike samples + draw-diversity resolve the sub-1%-margin the single degraded draw under-resolves.
VALIDATED 6-SEED FROM THE START (no tuning on a subset -- the lesson from the overfit read-out attempt): committee-seeds
42/43/44 (were the read-out's tuned set) + 100/101/102 (unseen). A generalizing improvement over the single-reservoir
positive read (42:18 43:18 44:11; 100/101/102 poor) would be a real feature-conditioning surpass. Per-seed scale sweep."""
import os, sys, time
os.environ.setdefault("SIM_BACKEND", "numpy")
import numpy as np
import research.runners._rungB1c_spiking_reservoir_synaptic_readout_derisk as C
from research.runners._emerge78_reservoir_form_to_role_derisk import Encoder, _gen, _TRAIN_KINDS
import step1_onoff_opponent as S

seeds = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["42", "43", "44", "100", "101", "102"])]
M = int(sys.argv[2]) if len(sys.argv) > 2 else 3
FLOOR = float(sys.argv[3]) if len(sys.argv) > 3 else 150.0
SCALES_C = [40., 60., 90., 130., 180., 240., 320.]
corpus = S.setup_corpus(seed=42); test = corpus["test"]
C.WS_BIAS_SCALE_C2 = 0.0
C.WS_ENS_FLOOR_C2 = FLOOR


def wire_sub_reservoir(ub, lo, n, in_dim, rseed):
    """Wire ONE reservoir's fixed-random recurrence in bridge indices [lo, lo+n); return its W_in (n x in_dim). Mirrors
    C.wire_reservoir but on an explicit sub-range + its own rseed (so M reservoirs are independent draws)."""
    idx = np.arange(lo, lo + n, dtype=np.int64)
    rng = np.random.default_rng(rseed * 7919 + 3)
    n_inh = int(round((1.0 - C.RES_EXC_FRACTION) * n))
    inh_local = np.sort(rng.choice(n, size=n_inh, replace=False))
    inh_set = set(int(x) for x in idx[inh_local])
    ub.bridge.cp_traits[idx[inh_local]] = 1
    ub.bridge._cached_inhibitory_mask = None
    pre, post, w = [], [], []
    rmat = rng.random((n, n))
    for a in range(n):
        pa = int(idx[a]); base_w = C.RES_INH_W if pa in inh_set else C.RES_EXC_W
        row = rmat[a]
        for bb in range(n):
            if a == bb:
                continue
            if row[bb] < C.RES_INTERNAL_DENSITY:
                jit = rng.standard_normal() * C.RES_JITTER
                pre.append(pa); post.append(int(idx[bb])); w.append(max(0.01, base_w * (1.0 + jit)))
    ub.bridge.set_pathway_weights("reservoir_rec", pre, post, np.asarray(w, np.float32), add_missing=True)
    W_in = (rng.random((n, in_dim)) * 2 - 1) * C.RES_IN_SCALE
    return idx, W_in


def build_committee(seed, enc):
    """Build a bridge with an M*RES_N reservoir slice holding M independent reservoirs; return (ub, res_all, res_idx_all)
    where res_all is a UBReservoir over the FULL concatenated slice (its W_in stacks the M W_ins)."""
    RESN = C.RES_N
    ub = C.UnifiedBrainBridge if False else None  # placeholder; use the c2 builder for parser/composer/role_wta context
    from research.runners.unified_brain_bridge import UnifiedBrainBridge
    from research.runners._rungB1b_neural_role_wta_derisk import PROJ_DIM
    ub = UnifiedBrainBridge(seed=seed, proj_dim=PROJ_DIM, concepts=corpus["concepts"],
                            enable_synaptic_route=True, role_wta_n=C.ROLE_WTA_N_C2, reservoir_n=M * RESN)
    ens, inh = C.wire_wta_c2(ub)
    base = int(ub.reservoir_base)
    W_ins = []
    for m in range(M):
        _idx, W_in = wire_sub_reservoir(ub, base + m * RESN, RESN, enc.dim, seed + 1000 * m)
        W_ins.append(W_in)
    res_idx_all = np.arange(base, base + M * RESN, dtype=np.int64)
    W_in_all = np.vstack(W_ins)                                    # (M*RESN x in_dim); each reservoir driven by its own rows
    res_all = C.UBReservoir(ub, res_idx_all, W_in_all)
    return ub, ens, res_all, res_idx_all


for seed in seeds:
    t0 = time.time()
    enc = Encoder(corpus["discovered"])
    rng = np.random.default_rng(seed * 101 + 5)
    train = _gen(_TRAIN_KINDS, C.N_TRAIN_PER, rng, corpus["subj"], corpus["verb"], corpus["obj"])
    ub, ens, res, res_idx = build_committee(seed, enc)
    res.snapshot_after_wiring()
    Ws = C._fit_Ws_spiking(res, enc, train)                       # fit on the CONCATENATED M*RES_N feature
    Ws_shift = {k: (Ws[k][:, :3] - Ws[k][:, :3].min()) for k in Ws}
    host = []
    for toks, *_ in test:
        f = np.concatenate([res.final_state(enc.encode(toks)), [1.0]])
        host.append([int(np.argmax((f @ Ws[k])[[0, 1, 2]])) for k in (0, 1, 2)])
    # fresh route bridge with res2ens edges pre-allocated
    ub2, ens2, res2, res_idx2 = build_committee(seed, enc)
    C.wire_ws_synapses(ub2, res_idx2, ens2, Ws_shift[0], 1.0, add_missing=True)
    res2.snapshot_after_wiring()
    f_ref = res2.final_state(enc.encode(test[0][0]))
    proj = max(1e-9, float((f_ref[:len(res_idx2)] @ np.maximum(Ws_shift[0][:len(res_idx2), :3], 0)).max()))
    best = 0
    for c in SCALES_C:
        sc = c / proj
        sr = C.SlotReadout(ub2, res2, ens2, Ws_shift, sc)
        ok = 0
        for (toks, *_), hs in zip(test, host):
            for k in (0, 1, 2):
                rb = sr.set_slot(k)
                _f, es = res2.run_with_ens(enc.encode(toks), ens2, role_bias=rb)
                ok += int(int(np.argmax(es)) == hs[k])
        best = max(best, ok)
    print(f"seed {seed} M={M} floor {FLOOR:.0f}: COMMITTEE best {best}/18   [{time.time()-t0:.0f}s]", flush=True)
