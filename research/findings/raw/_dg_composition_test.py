"""THROWAWAY DG-COMPOSITION PROBE (do not commit, do not import elsewhere).

Decisive question (this probe)
------------------------------
The activity-grounded composition symbols derived from the TRAINED
substrate's concept-pool MEAN activity are SEPARABILITY-limited:
between-concept cosine ~0.82, and FHRR compose (bind/bundle/unbind +
argmax cleanup) plateaus below the 0.80 bar at higher load -- on the
concept-pool activity (distinct, k=32) it got L2 0.834, L3 0.694,
L5 0.575 (THE BASELINE TO BEAT).

A separate DG-separation GATE then PASSED: routing that same concept
activity through the bridge's REAL spiking DG (sparse k-WTA via the
dg_pv_basket->dg FFi) orthogonalizes it from between-concept 0.806 down
to ~0.296 (headline sparse regime) / ~0.169 (anchor). DO composition
symbols DERIVED FROM THE DG-SEPARATED activity clear the 0.80 bar at
loads {2,3,5}, where the raw-pool-activity symbols FAILED?

What this probe does (per seed)
-------------------------------
1. Build ONE hippocampus bridge (the GATE's bridge, byte-faithful: DG=800,
   ec=200, dg_pv_basket=240, ca3=400, ca1=200; structural untrained DG +
   real FFi). Reuse the GATE's sparse concept->dg / concept->dg_pv_basket
   afferent projection + the GATE's drive-DG-and-capture step at the
   SAME tuned sparse regime that gave 0.296 (headline) / 0.169 (anchor).
2. For each concept WORD, split its 64 cached observations into TWO
   DISJOINT halves: storage-half = mean(obs[:32]), query-half =
   mean(obs[32:64]). Drive DG with each half's normalized concept
   activity -> capture the DG-separated per-neuron firing for storage
   and for query. Storage and query DG symbols therefore differ by
   GENUINE trial variation (mirrors the denoiser distinct-halves test;
   no leakage).
3. deriver_dg = ali.make_deriver(N_DIM=512, dg_dim=800, DERIV_SEED) --
   sized to the DG vector (NOT the 3200 concept-pool dim).
   storage DG symbol = deriver_dg(DG firing from storage-half);
   query DG symbol   = deriver_dg(DG firing from query-half);
   filler-vocab symbol = deriver_dg(DG firing from storage-half) (the
   registration subset), per filler word.
4. Run the SAME composition structure as _denoiser_cheap_probe (distinct
   halves): per load {2,3,5}, sample `load` cues + `load` fillers,
   FHRR-encode the (cue_dg, filler_dg) bound pairs, bundle, query each
   cue's DG symbol, recover, argmax cleanup vs the DG filler vocab;
   score composition-only accuracy (restricted to facts whose underlying
   concept half-activity recognizes its target pool -- the apples-to-
   apples filter the denoiser uses). ~60 trials/load.
5. ALSO report: between-concept cosine of the DG SYMBOLS (sanity, should
   be ~0.30 headline / ~0.17 anchor) AND how many words have SILENT DG
   (sparsity 0 -> degenerate symbol -> that word's composition fails),
   with composition accuracy computed with and without the silent words.

PRE-REGISTERED VERDICT (frozen; baseline pool L2 0.834 / L3 0.694 / L5 0.575):
  - FIXES   : DG-symbol composition L=3 AND L=5 both >= 0.80.
  - PARTIAL : DG-symbol L=3/L=5 materially above the pool baseline
              (>= +0.10 each) but not both >= 0.80.
  - NULL    : DG-symbol composition NOT better than the pool baseline
              (DG separation necessary-but-not-sufficient, OR silent-word
              degeneracy dominates).
Reports NUMBERS; the controller forms the official verdict + scrutinizes.

Reuse-by-import (byte-unchanged): activity_level_integration (make_deriver,
_direct_pool_target, recognized_pool, N_DIM, LOADS, BAR, SEEDS, K_VOCAB),
spiking_phasor_fhrr (SpikingPhasorFHRR, phase_similarity), the bridge
builder build_biological_brain_regions, and the GATE's vocab order. The
GATE's drive-DG closure + sparse-projection are COPIED here verbatim
(they are inner functions, not importable) -- the GATE .py is NOT edited.

GPU bridge (real run). The numpy FHRR composition is CPU. Plain ASCII.
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

import research.findings.raw.activity_level_integration as ali
from research.runners.spiking_phasor_fhrr import (
    SpikingPhasorFHRR, phase_similarity,
)

CACHE_DIR = "research/findings/raw/activity_level_integration_cache"
N_TRIALS = 60  # per load per seed


# ----------------------------------------------------------------------
# helpers (cos / between-concept cosine COPIED from the gate verbatim)
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
    """COPIED VERBATIM from _dg_separation_gate.make_sparse_projection.
    Fixed sparse random concept->dg afferent weight matrix; each (out,in)
    edge present w.p. `density`, weight ~ |N(weight_mean, (0.2*wm)^2)|."""
    rng = np.random.default_rng(seed)
    mask = rng.random((d_out, d_in)) < density
    w = np.abs(rng.normal(weight_mean, 0.2 * weight_mean, size=(d_out, d_in)))
    return (mask * w).astype(np.float64)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def run(seeds, cache_tag, concept_to_dg_density, concept_to_dg_weight,
        drive_scale, ffi_scale, n_steps, reset_steps, regime_label, out_path):
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

    # ---- build ONE hippocampus-only bridge (GATE's build, byte-faithful) --
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
    has_ffi = any(p.from_region == "dg_pv_basket" and p.to_region == "dg"
                  for p in cfg.region_pathways)
    print(f"[BUILD] hippocampus bridge: {cfg.num_neurons} neurons, "
          f"{int(bridge.cp_connections.nnz)} synapses, DG={n_dg} PV={n_pv} "
          f"ec={len(ec_idx)} FFi={has_ffi} in {build_sec:.1f}s "
          f"(regime={regime_label}: drive={drive_scale} ffi={ffi_scale})",
          flush=True)

    all_words, word_to_idx = _all_words_word_to_idx()
    concept_words = [w for w in all_words
                     if _direct_pool_target(w).startswith(
                         ("noun_pool_", "verb_pool_", "adjective_pool_"))]

    def drive_dg_with_activity(act_norm_gpu, w_dg_gpu, w_pv_gpu):
        """COPIED (logic) from _dg_separation_gate.drive_dg_with_activity.
        Inject I_dg = drive_scale*(W_dg @ act_norm) into DG and
        I_pv = drive_scale*ffi_scale*(W_pv @ act_norm) into dg_pv_basket,
        run the bridge's spiking step n_steps, accumulate DG spike counts.
        The bridge's REAL dg_pv_basket->dg synapses provide FFi sparsity.
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

    def dg_symbol_from_half(half_act, deriver_dg, w_dg_gpu, w_pv_gpu):
        """Drive DG with a normalized concept half-activity vector;
        return (dg_phasor_symbol_spikes, dg_rate_vec, ec_rate)."""
        a = np.maximum(np.asarray(half_act, dtype=np.float64), 0.0)
        nrm = np.linalg.norm(a)
        a_hat = a / (nrm + 1e-9)
        a_gpu = cp.asarray(a_hat, dtype=cp.float32)
        dg_vec, ec_rate = drive_dg_with_activity(a_gpu, w_dg_gpu, w_pv_gpu)
        return deriver_dg(dg_vec), dg_vec, ec_rate

    deriver_dg = ali.make_deriver(ali.N_DIM, n_dg, ali.DERIV_SEED)

    def tp(w):
        return _direct_pool_target(w)

    per_seed = []
    for seed in seeds:
        cache_path = os.path.join(CACHE_DIR, f"{cache_tag}_seed{seed}.npz")
        if not os.path.exists(cache_path):
            print(f"[SKIP] no cache {cache_path}", flush=True)
            continue
        # read ONLY numeric obs__<word> + pools/slices (no object-arrays /
        # never touch the object-typed __words__ key); words from vocab.
        data = np.load(cache_path)
        obs = {}
        for w in all_words:
            key = "obs__" + w
            if key in data.files:
                obs[w] = data[key]
        present_concepts = [w for w in concept_words if w in obs]
        pools = [str(p) for p in data["__pools__"]]
        slices = {p: tuple(int(x) for x in data["slice__" + p]) for p in pools}
        nobs = obs[present_concepts[0]].shape[0]
        d_act = obs[present_concepts[0]].shape[1]
        half = nobs // 2
        # storage-half / query-half disjoint mean concept activities
        store_act = {w: obs[w][:half].mean(axis=0) for w in present_concepts}
        query_act = {w: obs[w][half:nobs].mean(axis=0)
                     for w in present_concepts}

        # ---- per-concept-pool baseline cosine (the documented ~0.82) -----
        pool_vecs = {w: obs[w].mean(axis=0) for w in present_concepts}
        pool_mean, pool_max, _ = between_concept_cosines(pool_vecs)

        # fixed concept->dg + concept->pv afferent projections (per seed) --
        # SAME seeds as the gate: 1000+seed (dg), 2000+seed (pv).
        w_dg = make_sparse_projection(
            n_dg, d_act, concept_to_dg_density, concept_to_dg_weight,
            seed=1000 + seed)
        w_pv = make_sparse_projection(
            n_pv, d_act, concept_to_dg_density, concept_to_dg_weight,
            seed=2000 + seed)
        w_dg_gpu = cp.asarray(w_dg, dtype=cp.float32)
        w_pv_gpu = cp.asarray(w_pv, dtype=cp.float32)

        print(f"\n[seed {seed}] cache={cache_tag} nobs={nobs} d_act={d_act} "
              f"half={half}; baseline pool between-concept cos {pool_mean:.3f} "
              f"(max {pool_max:.3f}, expect ~0.82)", flush=True)

        # ---- derive DG symbols (storage + query) per concept word --------
        store_sym = {}
        query_sym = {}
        store_dg_vec = {}     # for between-concept cosine + silence check
        store_spars = {}
        query_spars = {}
        ec_rates = []
        for w in present_concepts:
            ss, sv, ser = dg_symbol_from_half(store_act[w], deriver_dg,
                                              w_dg_gpu, w_pv_gpu)
            qs, qv, qer = dg_symbol_from_half(query_act[w], deriver_dg,
                                              w_dg_gpu, w_pv_gpu)
            store_sym[w] = ss
            query_sym[w] = qs
            store_dg_vec[w] = sv
            store_spars[w] = float(np.mean(sv > 0))
            query_spars[w] = float(np.mean(qv > 0))
            ec_rates.append(0.5 * (ser + qer))
            print(f"  {w:>6} | store_DG_active={store_spars[w]:.3f} "
                  f"query_DG_active={query_spars[w]:.3f} "
                  f"store_DG_Hz~={sv.mean():.4f} ec~={0.5*(ser+qer):.4f}",
                  flush=True)

        # filler-vocab = DG symbol from the storage-half (registration)
        filler_words = [w for w in present_concepts
                        if tp(w).startswith("adjective_pool_")]
        cue_words = [w for w in present_concepts
                     if tp(w).startswith(("noun_pool_", "verb_pool_"))]
        vocab = {fw: store_sym[fw] for fw in filler_words}

        # ---- sanity: between-concept cosine of the DG SYMBOLS ------------
        # (use the underlying store DG firing vectors, like the gate)
        dg_between_mean, dg_between_max, _ = between_concept_cosines(
            {w: store_dg_vec[w] for w in present_concepts})

        # silent-DG words: store OR query half produced zero-sparsity DG
        silent_words = [w for w in present_concepts
                        if store_spars[w] <= 0.0 or query_spars[w] <= 0.0]
        n_silent = len(silent_words)
        silent_set = set(silent_words)

        # ---- recognition flags for composition-only filter ---------------
        # apples-to-apples with the denoiser: the underlying concept
        # half-activity must recognize its target pool (argmax over pools).
        store_recog = {w: (ali.recognized_pool(store_act[w], slices, pools)
                           == tp(w)) for w in present_concepts}
        query_recog = {w: (ali.recognized_pool(query_act[w], slices, pools)
                           == tp(w)) for w in present_concepts}

        # ---- composition scoring (denoiser distinct structure) -----------
        net = SpikingPhasorFHRR(ali.N_DIM, np.random.default_rng(seed))
        qrng = np.random.default_rng(seed + 1)

        def score_composition(exclude_silent):
            cw = [w for w in cue_words if not (exclude_silent
                                               and w in silent_set)]
            fw = [w for w in filler_words if not (exclude_silent
                                                  and w in silent_set)]
            res = {}
            for load in ali.LOADS:
                if len(cw) < load or len(fw) < 1:
                    res[load] = {"composition_only": float("nan"), "n_comp": 0}
                    continue
                ncc = nct = 0
                for _ in range(N_TRIALS):
                    cues = list(qrng.choice(cw, size=load, replace=False))
                    fills = list(qrng.choice(fw, size=load, replace=True))
                    enc_syms = [(store_sym[c], store_sym[f])
                                for (c, f) in zip(cues, fills)]
                    composite = net.encode(enc_syms)
                    for (c, f) in zip(cues, fills):
                        recovered = net.query(composite, query_sym[c])
                        sims = {x: phase_similarity(recovered, vocab[x])
                                for x in fw}
                        best = max(sims, key=sims.get)
                        hit = (tp(best) == tp(f))
                        # composition-only: storage cue+filler recognized
                        # AND query cue recognized (matches the denoiser).
                        if (store_recog[c] and store_recog[f]
                                and query_recog[c]):
                            ncc += int(hit)
                            nct += 1
                res[load] = {
                    "composition_only": (ncc / nct) if nct else float("nan"),
                    "n_comp": nct,
                }
            return res

        comp_all = score_composition(exclude_silent=False)
        comp_nosil = score_composition(exclude_silent=True) if n_silent \
            else comp_all

        ec_overall = float(np.mean(ec_rates))
        print(f"  [seed {seed}] DG-symbol between-concept cosine "
              f"{dg_between_mean:.3f} (max {dg_between_max:.3f}); "
              f"silent words={n_silent} {silent_words}; "
              f"ec(undriven)~={ec_overall:.4f}", flush=True)
        for load in ali.LOADS:
            ca = comp_all[load]["composition_only"]
            cn = comp_nosil[load]["composition_only"]
            print(f"    L={load}: composition-only(all)={ca:.3f} "
                  f"(n={comp_all[load]['n_comp']})  "
                  f"composition-only(no-silent)={cn:.3f} "
                  f"(n={comp_nosil[load]['n_comp']})", flush=True)

        per_seed.append({
            "seed": seed, "cache": cache_tag, "nobs": nobs, "d_act": d_act,
            "n_dg": n_dg, "regime": regime_label,
            "pool_between_mean": pool_mean, "pool_between_max": pool_max,
            "dg_between_mean": dg_between_mean, "dg_between_max": dg_between_max,
            "store_dg_sparsity": store_spars, "query_dg_sparsity": query_spars,
            "mean_store_dg_sparsity": float(
                np.mean(list(store_spars.values()))),
            "n_silent_words": n_silent, "silent_words": silent_words,
            "ec_undriven_rate": ec_overall,
            "comp_all": {str(l): comp_all[l] for l in ali.LOADS},
            "comp_no_silent": {str(l): comp_nosil[l] for l in ali.LOADS},
        })

    if not per_seed:
        print("[ERROR] no seeds processed (no caches found)", flush=True)
        return {}

    # ---- aggregate + pre-registered verdict ------------------------------
    POOL_BASE = {2: 0.834, 3: 0.694, 5: 0.575}

    def agg_load(key, load):
        vals = [r[key][str(load)]["composition_only"] for r in per_seed
                if r[key][str(load)]["composition_only"] ==
                r[key][str(load)]["composition_only"]]
        return float(np.mean(vals)) if vals else float("nan")

    dg_all = {l: agg_load("comp_all", l) for l in ali.LOADS}
    dg_nosil = {l: agg_load("comp_no_silent", l) for l in ali.LOADS}
    pool_between_m = float(np.mean([r["pool_between_mean"] for r in per_seed]))
    dg_between_m = float(np.mean([r["dg_between_mean"] for r in per_seed]))
    dg_spars_m = float(np.mean([r["mean_store_dg_sparsity"] for r in per_seed]))
    n_silent_total = [r["n_silent_words"] for r in per_seed]

    print("\n" + "=" * 70)
    print("AGGREGATE (seeds %s) regime=%s" % ([r["seed"] for r in per_seed],
                                              regime_label))
    print("=" * 70)
    print(f"  DG-symbol between-concept cosine: POOL(base) {pool_between_m:.3f} "
          f"-> DG {dg_between_m:.3f}  (DG mean sparsity {dg_spars_m:.3f})")
    print(f"  silent-DG words per seed: {n_silent_total}")
    print(f"  {'load':>5} | {'pool_base':>9} | {'DG(all)':>8} | "
          f"{'DG(no-silent)':>13}")
    for l in ali.LOADS:
        print(f"  {l:>5} | {POOL_BASE[l]:>9.3f} | {dg_all[l]:>8.3f} | "
              f"{dg_nosil[l]:>13.3f}")

    # verdict uses the no-silent DG numbers as the cleanest DG-composition
    # estimate (silent words are degenerate symbols, reported separately);
    # also report the all-words verdict.
    def verdict_of(dg):
        l3, l5 = dg[3], dg[5]
        if l3 != l3 or l5 != l5:
            return "INCONCLUSIVE", "NaN at L3/L5"
        if l3 >= 0.80 and l5 >= 0.80:
            return "FIXES", f"L3 {l3:.3f}>=0.80 AND L5 {l5:.3f}>=0.80"
        d3 = l3 - POOL_BASE[3]
        d5 = l5 - POOL_BASE[5]
        if d3 >= 0.10 and d5 >= 0.10:
            return "PARTIAL", (f"L3 +{d3:.3f} & L5 +{d5:.3f} over base "
                               f"(both>=+0.10) but not both>=0.80")
        return "NULL", (f"L3 {l3:.3f} (d{d3:+.3f}) / L5 {l5:.3f} (d{d5:+.3f}) "
                        f"not materially over pool base")

    v_nosil, dec_nosil = verdict_of(dg_nosil)
    v_all, dec_all = verdict_of(dg_all)
    print(f"\n  PRE-REGISTERED VERDICT (no-silent): {v_nosil}  [{dec_nosil}]")
    print(f"  PRE-REGISTERED VERDICT (all words):  {v_all}  [{dec_all}]")
    print("=" * 70, flush=True)

    out = {
        "probe": "dg_composition_test", "backend": backend_name,
        "cache_tag": cache_tag, "regime": regime_label,
        "concept_to_dg": {
            "density": concept_to_dg_density,
            "weight_mean": concept_to_dg_weight,
            "drive_scale": drive_scale, "ffi_scale": ffi_scale,
        },
        "n_steps": n_steps, "reset_steps": reset_steps, "n_trials": N_TRIALS,
        "bridge": {
            "n_neurons": int(cfg.num_neurons),
            "n_synapses": int(bridge.cp_connections.nnz),
            "n_dg": n_dg, "n_pv_basket": n_pv,
            "real_dg_pv_basket_to_dg_ffi": bool(has_ffi),
        },
        "pool_baseline": {str(k): v for k, v in POOL_BASE.items()},
        "loads": list(ali.LOADS), "bar": ali.BAR,
        "per_seed": per_seed,
        "aggregate": {
            "pool_between_mean": pool_between_m,
            "dg_between_mean": dg_between_m,
            "dg_mean_sparsity": dg_spars_m,
            "n_silent_words_per_seed": n_silent_total,
            "dg_comp_all": {str(k): v for k, v in dg_all.items()},
            "dg_comp_no_silent": {str(k): v for k, v in dg_nosil.items()},
        },
        "verdict_no_silent": v_nosil, "deciding_no_silent": dec_nosil,
        "verdict_all_words": v_all, "deciding_all_words": dec_all,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str),
                        encoding="utf-8")
    print(f"[OUT] {out_path}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--cache-tag", type=str, default="denoise64")
    ap.add_argument("--concept-to-dg-density", type=float, default=0.40)
    ap.add_argument("--concept-to-dg-weight", type=float, default=5.0)
    # HEADLINE sparse regime (gave DG between-concept ~0.296, sparsity ~0.044):
    ap.add_argument("--drive-scale", type=float, default=10.0)
    ap.add_argument("--ffi-scale", type=float, default=0.16)
    ap.add_argument("--n-steps", type=int, default=100)
    ap.add_argument("--reset-steps", type=int, default=40)
    ap.add_argument("--regime-label", type=str, default="headline")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/_dg_composition_test.json")
    args = ap.parse_args()
    run(seeds=args.seeds, cache_tag=args.cache_tag,
        concept_to_dg_density=args.concept_to_dg_density,
        concept_to_dg_weight=args.concept_to_dg_weight,
        drive_scale=args.drive_scale, ffi_scale=args.ffi_scale,
        n_steps=args.n_steps, reset_steps=args.reset_steps,
        regime_label=args.regime_label, out_path=Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
