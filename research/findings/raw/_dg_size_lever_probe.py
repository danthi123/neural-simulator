"""P5 lever: does a LARGER DG improve WITHIN-concept reliability (the
DG-composition NULL's failure) while keeping BETWEEN-concept separation?

The DG-composition NULL: at the real spiking DG (n_dg=800, ~16-40 active at
sparsity 0.02-0.05) a concept's storage-half vs query-half DG codes were
near-disjoint (within-concept cosine ~0.0-0.3) -- effectively binary, so the
few active neurons differ between halves. A CPU graded model predicted that a
LARGER DG (more active neurons even at low sparsity fraction) is materially
more stable within-concept (~0.6-0.7) while still separating. This probe
tests that on the REAL spiking DG: build the gate's hippocampus bridge at
n_dg in {800 (control), 4000, 8000}, drive each concept's STORAGE-half mean
and QUERY-half mean activity through DG (sparse k-WTA via the real FFi), and
measure BOTH between-concept (store_dg of A vs B) AND within-concept (store_dg
vs query_dg of the same concept) DG-rate cosine, sweeping (drive_scale,
ffi_scale) to bracket the sparse band.

Reuse-by-import / copy of the gate probe's BYTE-UNCHANGED machinery
(make_sparse_projection, the build, the drive). No protected/sim edit. No
autograd. CuPy/GPU. ASCII. The controller reads the table + forms the verdict:
a LARGER DG HELPS if, at a sparsity where between-concept stays <= ~0.5,
within-concept rises materially over the n_dg=800 control (toward >= 0.6).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from itertools import combinations
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# reuse the gate's byte-unchanged helpers
from research.findings.raw._dg_separation_gate import (
    cos, between_concept_cosines, make_sparse_projection,
)

CACHE_DIR = "research/findings/raw/activity_level_integration_cache"


def build_bridge(n_dg, seed):
    from sim.config import (CoreSimConfig, VisualizationConfig,
                            RuntimeState, GPUConfig)
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions)
    n_pv = max(60, int(0.30 * n_dg))   # keep the P1 FFi ratio (~30% of DG)
    regions, pathways = build_biological_brain_regions(
        n_lang_input=64, n_motor_per_action=8, n_motor_fs_per_action=2,
        enable_motor_fs=True, enable_language_output=False,
        enable_hippocampus_consolidation=True,
        n_ec=200, n_dg=int(n_dg), n_dg_pv_basket=int(n_pv),
        n_ca3=400, n_ca1=200,
    )
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = int(seed)
    cfg.enable_nmda = True
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.stdp_w_max = 10.0
    cfg.fast_spike_reset = True
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    return bridge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dg", type=int, nargs="+", default=[800, 4000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--cache-tag", default="denoise64")
    ap.add_argument("--density", type=float, default=0.40)
    ap.add_argument("--weight", type=float, default=5.0)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--reset-steps", type=int, default=40)
    # (drive, ffi) pairs to bracket the sparse band; tuned at run time per n_dg
    ap.add_argument("--sweep", type=float, nargs="+",
                    default=[10.0, 0.16, 9.0, 0.18, 8.0, 0.22, 6.0, 0.30])
    ap.add_argument("--out", default=os.path.join(_HERE, "_dg_size_lever_probe.json"))
    a = ap.parse_args()

    from sim.backend import get_backend, to_host
    cp, backend = get_backend()
    print("backend", backend, flush=True)
    from research.runners.unified_per_regime_monitor_runner import (
        _all_words_word_to_idx, _direct_pool_target)
    all_words, _ = _all_words_word_to_idx()
    concept_words = [w for w in all_words if _direct_pool_target(w).startswith(
        ("noun_pool_", "verb_pool_", "adjective_pool_"))]
    pairs = [(a.sweep[i], a.sweep[i + 1]) for i in range(0, len(a.sweep), 2)]

    results = []
    for n_dg in a.n_dg:
        for seed in a.seeds:
            cache = os.path.join(CACHE_DIR, "%s_seed%d.npz" % (a.cache_tag, seed))
            if not os.path.exists(cache):
                print("[skip] no cache", cache, flush=True); continue
            data = np.load(cache)
            obs = {w: data["obs__" + w] for w in all_words if "obs__" + w in data.files}
            cw = [w for w in concept_words if w in obs]
            d_act = obs[cw[0]].shape[1]
            t0 = time.time()
            bridge = build_bridge(n_dg, seed)
            rm = bridge.region_manager
            dg_arr = cp.asarray(list(rm.indices("dg")), dtype=cp.int64)
            pv_arr = cp.asarray(list(rm.indices("dg_pv_basket")), dtype=cp.int64)
            ec_arr = cp.asarray(list(rm.indices("ec")), dtype=cp.int64)
            ndg = int(dg_arr.shape[0])
            print("[build] n_dg=%d (PV=%d) seed=%d in %.1fs"
                  % (ndg, pv_arr.shape[0], seed, time.time() - t0), flush=True)
            w_dg = cp.asarray(make_sparse_projection(ndg, d_act, a.density, a.weight, 1000 + seed), dtype=cp.float32)
            w_pv = cp.asarray(make_sparse_projection(int(pv_arr.shape[0]), d_act, a.density, a.weight, 2000 + seed), dtype=cp.float32)

            def drive(act, ds, fs):
                a_ = np.maximum(act.astype(np.float64), 0.0)
                a_ = a_ / (np.linalg.norm(a_) + 1e-9)
                ag = cp.asarray(a_, dtype=cp.float32)
                bridge.cp_external_input_current[:] = 0.0
                for _ in range(a.reset_steps):
                    bridge._run_one_simulation_step()
                i_dg = (w_dg @ ag) * float(ds)
                i_pv = (w_pv @ ag) * float(ds) * float(fs)
                bridge.cp_external_input_current[:] = 0.0
                bridge.cp_external_input_current[dg_arr] = i_dg.astype(cp.float32)
                bridge.cp_external_input_current[pv_arr] = i_pv.astype(cp.float32)
                c = cp.zeros(ndg, dtype=cp.float64)
                for _ in range(a.n_steps):
                    bridge._run_one_simulation_step()
                    c += bridge.cp_firing_states[dg_arr].astype(cp.float64)
                bridge.cp_external_input_current[:] = 0.0
                return to_host(c) / float(a.n_steps)

            store_act = {w: obs[w][:32].mean(axis=0) for w in cw}
            query_act = {w: obs[w][32:].mean(axis=0) for w in cw}
            for (ds, fs) in pairs:
                store_dg = {w: drive(store_act[w], ds, fs) for w in cw}
                query_dg = {w: drive(query_act[w], ds, fs) for w in cw}
                spars = float(np.mean([(v > 0).mean() for v in store_dg.values()]))
                bmean, bmax, _ = between_concept_cosines(store_dg)
                within = [cos(store_dg[w], query_dg[w]) for w in cw]
                wmean = float(np.mean(within))
                n_silent = int(sum(1 for w in cw if (store_dg[w] > 0).sum() < 2 or (query_dg[w] > 0).sum() < 2))
                row = {"n_dg": ndg, "seed": seed, "drive": ds, "ffi": fs,
                       "sparsity": spars, "between_mean": bmean, "between_max": bmax,
                       "within_mean": wmean, "n_silent": n_silent}
                results.append(row)
                print("  n_dg=%-5d drive=%-4.1f ffi=%-4.2f spars=%.3f  BETWEEN=%.3f  WITHIN=%.3f  silent=%d/%d"
                      % (ndg, ds, fs, spars, bmean, wmean, n_silent, len(cw)), flush=True)
            del bridge
            if backend == "cupy":
                cp.get_default_memory_pool().free_all_blocks()

    json.dump({"results": results}, open(a.out, "w"), indent=2)
    print("\nwrote", a.out, flush=True)
    print("READ: compare WITHIN at matched BETWEEN/sparsity across n_dg. Larger DG HELPS if "
          "WITHIN rises materially (toward >=0.6) at a sparsity where BETWEEN stays <=~0.5.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
