"""THROWAWAY GATE PROBE (do not commit, do not import elsewhere).

Falsifiable prediction under test
---------------------------------
The concept-pool per-neuron MEAN activity (from the TRAINED v16+hippo
substrate, 64-observation cache) has BETWEEN-concept cosine ~0.82
(within-concept split-half ~0.90) -- concepts are barely separable. That
~0.82 is the SAME regime as P1's validated DG pattern-separation INPUT
(cosine 0.800), where the bridge's DG produced OUTPUT cosine 0.218 (a 58pp
orthogonalization). PREDICTION: routing the concept-pool activity THROUGH
the bridge's spiking DG should separate it from ~0.82 toward ~0.22.

This probe measures that on the REAL spiking DG of a hippocampus-enabled
bridge.

The genuine 0.82 regime + why we route the activity (not re-fire pools)
----------------------------------------------------------------------
The ~0.82 figure (reproduced exactly: fillers 0.818/0.829/0.815, cues
~0.80, within ~0.90 for seeds 42/43/44) is the between-concept cosine of
the 64-OBSERVATION-AVERAGED per-neuron activity over the 3200-neuron
concept-pool population (16 pools x 200) of the 800-event-TRAINED
substrate. It lives in the disk caches
  research/findings/raw/activity_level_integration_cache/denoise64_seed{N}.npz
captured by activity_level_integration.capture_activity. An UNTRAINED
bridge does NOT reproduce 0.82 (its pools fire word-specific random
subsets -> already ~0.24 separated); the overlap is a property of the
TRAINED weights.

load_checkpoint REPLACES core_config + num_neurons + the entire CSR with
the checkpoint's saved data (verified in sim/bridge.py:6124-6182), so a
concept->dg pathway ADDED to the architecture before loading the 800ev
checkpoint would be WIPED. Therefore the faithful test is: take the
genuine 0.82-regime activity from the cache, and DRIVE the bridge's REAL
spiking DG with it via a fixed concept->dg afferent projection, letting
the bridge's REAL dg_pv_basket -> dg feed-forward inhibition (the k-WTA
that does pattern separation) sparsify the DG output. The DG region itself
(95% exc, no recurrence) + its PV-basket FFi are STRUCTURAL/untrained in
the 800ev checkpoint too -- P1 validated DG separation on exactly this
structural DG -- so an untrained hippocampus bridge has the same DG.

Isolation (DG fed the CONCEPT activity, NOT the orthogonal lang_input)
----------------------------------------------------------------------
There is NO lang_input drive in this probe. DG is driven PURELY by the
concept-pool activity vector, through a fixed sparse random concept->dg
(and concept->dg_pv_basket FFi) projection. ec is never driven. So DG
cannot be separating orthogonal lang_input codes -- it only ever sees the
0.82-overlapping concept activity. (We verify ec is silent as a guard.)

Pre-registered verdict (frozen by the controller):
* SEPARATES  : DG between-concept cosine <= 0.50 (ideally ~0.22), DG active.
* DOES-NOT   : DG between-concept cosine > ~0.65.
* INCONCLUSIVE: DG silent / not driven / scale too small.

Reports numbers only; controller forms the official verdict + scrutinizes.

NOTE on cache loading: we read ONLY the numeric ``obs__<word>`` arrays
(plain float32) -- never the object-typed ``__words__`` key -- so no
pickle is needed. The word list comes from the known v16 vocabulary
(unified_per_regime_monitor_runner._all_words_word_to_idx), which is
exactly the order capture wrote.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def cos(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def between_concept_cosines(vecs: dict) -> tuple:
    words = list(vecs.keys())
    pc = [cos(vecs[a], vecs[b]) for a, b in combinations(words, 2)]
    arr = np.asarray(pc, dtype=np.float64)
    return float(arr.mean()), float(arr.max()), pc


def make_sparse_projection(d_out, d_in, density, weight_mean, seed):
    """Fixed sparse random projection (the concept->dg afferent weight
    matrix): each (out, in) edge present w.p. `density`, weight
    ~ |N(weight_mean, (0.2*weight_mean)^2)| (positive = excitatory,
    matching the pathway). Dense (800 x 3200) is fine."""
    rng = np.random.default_rng(seed)
    mask = rng.random((d_out, d_in)) < density
    w = np.abs(rng.normal(weight_mean, 0.2 * weight_mean, size=(d_out, d_in)))
    return (mask * w).astype(np.float64)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def run(seeds, cache_tag, concept_to_dg_density, concept_to_dg_weight,
        drive_scale, ffi_scale, n_steps, reset_steps, out_path):
    from sim.backend import get_backend, to_host
    cp, backend_name = get_backend()
    print(f"backend = {backend_name}", flush=True)

    from sim.config import (CoreSimConfig, VisualizationConfig,
                            RuntimeState, GPUConfig)
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    from research.runners.unified_per_regime_monitor_runner import (
        _all_words_word_to_idx, _direct_pool_target,
    )

    CACHE_DIR = ("research/findings/raw/"
                 "activity_level_integration_cache")

    # ---- build ONE hippocampus-only bridge (DG/ec/pv_basket/ca3/ca1) --
    # P1-scale DG. No concept pools / motor / lang needed in the bridge:
    # we drive DG directly with the concept-activity projection. (Keep a
    # tiny language_input so the builder is happy; it is never driven.)
    t0 = time.time()
    regions, pathways = build_biological_brain_regions(
        n_lang_input=64,                # unused, minimal
        n_motor_per_action=8, n_motor_fs_per_action=2, enable_motor_fs=True,
        enable_language_output=False,
        enable_hippocampus_consolidation=True,
        n_ec=200, n_dg=800, n_dg_pv_basket=240, n_ca3=400, n_ca1=200,
    )
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = int(seeds[0])
    cfg.enable_nmda = True
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.stdp_w_max = 10.0
    cfg.fast_spike_reset = True

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    rm = bridge.region_manager
    dg_idx = list(rm.indices("dg"))
    pv_idx = list(rm.indices("dg_pv_basket"))
    ec_idx = list(rm.indices("ec"))
    dg_arr = cp.asarray(dg_idx, dtype=cp.int64)
    pv_arr = cp.asarray(pv_idx, dtype=cp.int64)
    ec_arr = cp.asarray(ec_idx, dtype=cp.int64)
    n_dg = len(dg_idx)
    n_pv = len(pv_idx)
    build_sec = time.time() - t0
    print(f"[BUILD] hippocampus bridge: {cfg.num_neurons} neurons, "
          f"{int(bridge.cp_connections.nnz)} synapses, "
          f"DG={n_dg} PV={n_pv} ec={len(ec_idx)} in {build_sec:.1f}s",
          flush=True)
    has_ffi = any(p.from_region == "dg_pv_basket" and p.to_region == "dg"
                  for p in cfg.region_pathways)
    has_ec_dg = any(p.from_region == "ec" and p.to_region == "dg"
                    for p in cfg.region_pathways)
    print(f"[BUILD] real dg_pv_basket->dg FFi present: {has_ffi}; "
          f"ec->dg present (NOT used, ec undriven): {has_ec_dg}", flush=True)

    all_words, word_to_idx = _all_words_word_to_idx()
    concept_words = [w for w in all_words
                     if _direct_pool_target(w).startswith(
                         ("noun_pool_", "verb_pool_", "adjective_pool_"))]

    def drive_dg_with_activity(act_norm_gpu, w_dg_gpu, w_pv_gpu):
        """Inject I_dg = drive_scale*(W_dg @ act_norm) into DG and
        I_pv = drive_scale*(W_pv @ act_norm) into dg_pv_basket, run the
        bridge's spiking step n_steps, accumulate DG spike counts. The
        bridge's REAL dg_pv_basket->dg synapses provide FFi sparsity.
        Returns (dg_rate_vec, ec_rate_during)."""
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(reset_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        i_dg = (w_dg_gpu @ act_norm_gpu) * float(drive_scale)
        i_pv = (w_pv_gpu @ act_norm_gpu) * float(drive_scale) * float(ffi_scale)
        counts = cp.zeros(n_dg, dtype=cp.float64)
        ec_counts = cp.zeros(len(ec_idx), dtype=cp.float64)
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[dg_arr] = i_dg.astype(cp.float32)
        bridge.cp_external_input_current[pv_arr] = i_pv.astype(cp.float32)
        for _ in range(n_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            counts += bridge.cp_firing_states[dg_arr].astype(cp.float64)
            ec_counts += bridge.cp_firing_states[ec_arr].astype(cp.float64)
        bridge.cp_external_input_current[:] = 0.0
        return (to_host(counts) / float(n_steps),
                float(to_host(ec_counts).mean()) / float(n_steps))

    per_seed = []
    for seed in seeds:
        cache_path = os.path.join(CACHE_DIR, f"{cache_tag}_seed{seed}.npz")
        if not os.path.exists(cache_path):
            print(f"[SKIP] no cache {cache_path}", flush=True)
            continue
        # read ONLY numeric obs__<word> arrays (no pickle); words from vocab
        data = np.load(cache_path)
        obs = {}
        for w in all_words:
            key = "obs__" + w
            if key in data.files:
                obs[w] = data[key]
        present_words = [w for w in all_words if w in obs]
        present_concepts = [w for w in concept_words if w in obs]
        nobs = obs[present_words[0]].shape[0]
        d_act = obs[present_words[0]].shape[1]
        # 64-obs-averaged consolidated activity = the 0.82-regime vector
        cons = {w: obs[w].mean(axis=0) for w in present_words}

        # baseline between-concept cosine (concept words)
        pool_vecs = {w: cons[w] for w in present_concepts}
        pool_mean, pool_max, pool_list = between_concept_cosines(pool_vecs)

        # fixed concept->dg + concept->pv afferent projections (per seed)
        w_dg = make_sparse_projection(
            n_dg, d_act, concept_to_dg_density, concept_to_dg_weight,
            seed=1000 + seed)
        w_pv = make_sparse_projection(
            n_pv, d_act, concept_to_dg_density, concept_to_dg_weight,
            seed=2000 + seed)
        w_dg_gpu = cp.asarray(w_dg, dtype=cp.float32)
        w_pv_gpu = cp.asarray(w_pv, dtype=cp.float32)

        print(f"\n[seed {seed}] cache={cache_tag} nobs={nobs} d_act={d_act}; "
              f"baseline pool between-concept cos mean {pool_mean:.3f} "
              f"max {pool_max:.3f} (expect ~0.82)", flush=True)

        dg_vecs = {}
        dg_sparsities = []
        dg_means = []
        ec_means = []
        for w in present_concepts:
            a = np.maximum(cons[w].astype(np.float64), 0.0)
            nrm = np.linalg.norm(a)
            a_hat = a / (nrm + 1e-9)
            a_gpu = cp.asarray(a_hat, dtype=cp.float32)
            dg_vec, ec_rate = drive_dg_with_activity(a_gpu, w_dg_gpu, w_pv_gpu)
            dg_vecs[w] = dg_vec
            spars = float(np.mean(dg_vec > 0))
            dg_sparsities.append(spars)
            dg_means.append(float(dg_vec.mean()))
            ec_means.append(ec_rate)
            print(f"  {w:>6} | DG_active={spars:.3f} "
                  f"DG_meanHz~={dg_vec.mean():.4f} ec_meanHz~={ec_rate:.4f}",
                  flush=True)

        dg_mean_cos, dg_max_cos, dg_list = between_concept_cosines(dg_vecs)
        dg_overall_active = float(np.mean(dg_sparsities))
        dg_overall_hz = float(np.mean(dg_means))
        ec_overall_hz = float(np.mean(ec_means))
        dg_silent = (dg_overall_active < 0.01) or (dg_overall_hz <= 1e-4)

        print(f"  [seed {seed}] DG activity: mean sparsity "
              f"{dg_overall_active:.3f}, mean rate ~{dg_overall_hz:.4f}; "
              f"silent={dg_silent}; ec(undriven) rate ~{ec_overall_hz:.4f}")
        print(f"  [seed {seed}] BETWEEN-concept cosine: "
              f"POOL {pool_mean:.3f} (max {pool_max:.3f})  ->  "
              f"DG {dg_mean_cos:.3f} (max {dg_max_cos:.3f})", flush=True)

        per_seed.append({
            "seed": seed, "cache": cache_tag, "nobs": nobs,
            "d_act": d_act, "n_concept_words": len(present_concepts),
            "pool_between_mean": pool_mean, "pool_between_max": pool_max,
            "dg_between_mean": dg_mean_cos, "dg_between_max": dg_max_cos,
            "dg_mean_sparsity": dg_overall_active,
            "dg_mean_rate": dg_overall_hz,
            "ec_undriven_rate": ec_overall_hz,
            "dg_silent": bool(dg_silent),
            "per_word_dg_sparsity": dict(zip(present_concepts, dg_sparsities)),
            "pool_pairwise": pool_list, "dg_pairwise": dg_list,
        })

    if not per_seed:
        print("[ERROR] no seeds processed (no caches found)", flush=True)
        return {}

    # ---- aggregate + pre-registered verdict --------------------------
    pool_m = float(np.mean([r["pool_between_mean"] for r in per_seed]))
    dg_m = float(np.mean([r["dg_between_mean"] for r in per_seed]))
    dg_active_m = float(np.mean([r["dg_mean_sparsity"] for r in per_seed]))
    dg_rate_m = float(np.mean([r["dg_mean_rate"] for r in per_seed]))
    any_silent = any(r["dg_silent"] for r in per_seed)

    print("\n" + "=" * 66)
    print("AGGREGATE (seeds %s)" % [r["seed"] for r in per_seed])
    print("=" * 66)
    print(f"  DG activity : mean sparsity {dg_active_m:.3f}, "
          f"mean rate ~{dg_rate_m:.4f}; any_silent={any_silent}")
    print(f"  BETWEEN-concept cosine  POOL (baseline): {pool_m:.3f} "
          f"(expect ~0.82)")
    print(f"  BETWEEN-concept cosine  DG   (the test): {dg_m:.3f}")

    if any_silent or dg_active_m < 0.01:
        verdict = "INCONCLUSIVE"
        decider = f"DG silent/under-driven (mean sparsity {dg_active_m:.3f})"
    elif dg_m <= 0.50:
        verdict = "SEPARATES"
        decider = f"DG between-concept mean cosine {dg_m:.3f} <= 0.50"
    elif dg_m > 0.65:
        verdict = "DOES-NOT-SEPARATE"
        decider = f"DG between-concept mean cosine {dg_m:.3f} > 0.65"
    else:
        verdict = "AMBIGUOUS(0.50-0.65)"
        decider = f"DG between-concept mean cosine {dg_m:.3f} in (0.50,0.65]"

    print(f"\n  PRE-REGISTERED VERDICT: {verdict}")
    print(f"  deciding number: {decider}")
    print("=" * 66, flush=True)

    out = {
        "backend": backend_name, "cache_tag": cache_tag,
        "concept_to_dg": {
            "density": concept_to_dg_density,
            "weight_mean": concept_to_dg_weight,
            "drive_scale": drive_scale, "ffi_scale": ffi_scale,
        },
        "n_steps": n_steps, "reset_steps": reset_steps,
        "bridge": {
            "n_neurons": int(cfg.num_neurons),
            "n_synapses": int(bridge.cp_connections.nnz),
            "n_dg": n_dg, "n_pv_basket": n_pv,
            "real_dg_pv_basket_to_dg_ffi": bool(has_ffi),
        },
        "concept_words": concept_words,
        "per_seed": per_seed,
        "aggregate": {
            "pool_between_mean": pool_m, "dg_between_mean": dg_m,
            "dg_mean_sparsity": dg_active_m, "dg_mean_rate": dg_rate_m,
            "any_silent": bool(any_silent),
        },
        "verdict": verdict, "deciding_number": decider,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str),
                        encoding="utf-8")
    print(f"[OUT] {out_path}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--cache-tag", type=str, default="denoise64",
                    help="cache prefix in activity_level_integration_cache "
                         "(denoise64 = 64-obs 0.82 regime; full = 16-obs)")
    ap.add_argument("--concept-to-dg-density", type=float, default=0.40)
    ap.add_argument("--concept-to-dg-weight", type=float, default=5.0)
    ap.add_argument("--drive-scale", type=float, default=1.0)
    ap.add_argument("--ffi-scale", type=float, default=1.0,
                    help="multiplier on concept->dg_pv_basket FFi drive "
                         "(<1 weakens FFi so DG fires; 1.0 = balanced)")
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--reset-steps", type=int, default=40)
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_dg_separation_gate.json")
    args = ap.parse_args()
    run(seeds=args.seeds, cache_tag=args.cache_tag,
        concept_to_dg_density=args.concept_to_dg_density,
        concept_to_dg_weight=args.concept_to_dg_weight,
        drive_scale=args.drive_scale, ffi_scale=args.ffi_scale,
        n_steps=args.n_steps, reset_steps=args.reset_steps,
        out_path=Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
