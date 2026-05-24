"""DLPFC_WM-extension of HIPPO-OPTION3: does parallel-matching
biologized mode-unification still PASS when the NMDA bistable
dlpfc_wm region + lang_input -> dlpfc_wm pathway are ALSO present?

CONTEXT
- HIPPO-OPTION3 (pillar n=97) validated parallel-matching with
  hippocampus + Phase 1.3 SWR consolidation pathways PRESENT.
- Pillar n=97 honest scope correction noted: dlpfc_wm region is
  built ONLY in g11_bg_runner.py via explicit BrainRegion, NOT by
  enable_hippocampus_consolidation=True.
- This probe adds dlpfc_wm using g11_bg_runner's pattern, plus the
  lang_input -> dlpfc_wm pathway (so the (c) loop can inject the
  encoded PFC frame into dlpfc_wm via lang_input).

PRE-REGISTERED reading (fixed; never tuned):
- DLPFC_PASS: multi-seed-mean >= 0.80 every cell on BOTH readouts.
  All FIVE load-bearing components of (c) (concept pools + parallel-
  matching + hippocampus + SWR consolidation + dlpfc_wm) validated
  on a single coherent substrate. (c) loop-controller TDD build is
  the next pre-registered direction.
- DLPFC_NEGATIVE: either readout misses. The dlpfc_wm presence
  perturbs the substrate enough to break grounded-symbol pipeline.
  Biology-translatable: NMDA bistable PFC region modulates cortex
  baseline enough to break parallel-matching at this scale; (c)
  needs a different integration path (different PFC region design
  / different mode-unification grounding / etc).

Mirrors HIPPO-OPTION3 exactly EXCEPT the bridge adds dlpfc_wm
BrainRegion (g11_bg_runner pattern at line 412+) + a lang_input ->
dlpfc_wm plastic pathway. Concept-pool dynamics + topographic prior
+ training + capture + decoder ALL byte-unchanged.

Reuses HIPPO-OPTION3's capture/grounding/pipeline functions byte-
unchanged via import; no protected/frozen/moat module modified;
no autograd; no-confab moat must stay 7/7 green.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.findings.raw.vocabulary_scaling_run import (
    BAR, LOADS, SEEDS, N_DIM, N_TRIALS,
)
from research.findings.raw.biologized_spiking_mode_unification_parallel_matching_runner import (
    K_VOCAB_TARGET, DERIV_SEED,
)
from research.findings.raw.pattern_separation_grounding_probe import (
    make_deriver,
)
from research.findings.raw.biologized_spiking_mode_unification_helpers import (
    gamma_slot_positions,
)
from research.findings.raw.cross_bridge_mode_unification_probe import (
    batched_phase_similarity, verify_batched_equivalent_to_scalar,
)
from research.findings.raw.mode_unification_on_bio_brain_regions_probe import (
    _capture_concept_pool_activity, _ground_symbols,
    _load_activity_cache,
    DEFAULT_N_LANG_INPUT, DEFAULT_N_PER_POOL, DEFAULT_N_FS_PER_POOL,
    DEFAULT_TOPOGRAPHIC_FACTOR, DEFAULT_OFF_TARGET_FACTOR,
    DEFAULT_SPARSITY, DEFAULT_N_TRAIN_EVENTS, M_OBS_FULL,
    SMOKE_VOCAB, SMOKE_N_TRAIN_EVENTS, SMOKE_M_OBS, SMOKE_LOADS,
)
from research.runners.resonate_fire_fhrr import ResonateFireFHRR
from research.runners.spiking_phasor_fhrr import phases_to_spikes
from research.runners.concept_pool_demo import (
    apply_concept_topographic_bias, train_word_to_pool,
    DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB, ADJECTIVE_VOCAB,
    NOUN_NAMES, VERB_NAMES, ADJECTIVE_NAMES,
)
from research.runners.text_minimal_isolation import (
    build_biological_brain_regions,
)
from sim.regions import BrainRegion, RegionPathway
from sim.enums import NeuronType
from sim.backend import get_backend, is_gpu_backend, to_host

CACHE_DIR = os.path.join(
    _HERE, "mode_unification_with_hippo_dlpfc_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# dlpfc_wm region config (mirrors g11_bg_runner.py:412-427 exactly).
DLPFC_N_NEURONS = 60
DLPFC_EXC_FRACTION = 0.8
DLPFC_INTERNAL_DENSITY = 0.10
DLPFC_EXC_WEIGHT = 2.0
DLPFC_INH_WEIGHT = 4.0
DLPFC_WEIGHT_JITTER = 0.2

# lang_input -> dlpfc_wm pathway (the (c) loop will use this to
# inject encoded PFC frames; for this probe, the pathway is PRESENT
# but plasticity is gated off during the standard training).
LANG_TO_DLPFC_DENSITY = 0.10
LANG_TO_DLPFC_WEIGHT = 1.0
LANG_TO_DLPFC_JITTER = 0.3


def _bridge_save_path(seed, smoke):
    tag = "smoke" if smoke else "full"
    return os.path.join(CACHE_DIR, f"bridge_{tag}_seed{seed}.simstate.h5")


def _activity_cache_path(seed, smoke):
    tag = "smoke" if smoke else "full"
    return os.path.join(CACHE_DIR, f"activity_{tag}_seed{seed}.npz")


def _build_bridge_with_hippo_and_dlpfc(seed, enable_adjective,
                                         n_lang_input=DEFAULT_N_LANG_INPUT,
                                         n_per_pool=DEFAULT_N_PER_POOL,
                                         n_fs_per_pool=DEFAULT_N_FS_PER_POOL,
                                         verbose=True):
    """v14/v16 16-pool concept bridge + hippocampus + dlpfc_wm region
    + lang_input -> dlpfc_wm pathway. Replicates HIPPO-OPTION3 bridge-
    build inline then appends dlpfc_wm BrainRegion + lang_input ->
    dlpfc_wm pathway (g11_bg_runner pattern; declarative; ~30 lines).
    """
    from sim.config import (CoreSimConfig, VisualizationConfig,
                              RuntimeState, GPUConfig)
    from sim.bridge import SimulationBridge

    concept_internal_density = 0.05
    concept_exc_weight = 0.3
    concept_inh_weight = 0.8
    motor_internal_density = 0.10
    motor_exc_weight = 2.0
    motor_inh_weight = 4.0

    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=n_per_pool,
        motor_internal_density=motor_internal_density,
        motor_exc_weight_mean=motor_exc_weight,
        motor_inh_weight_mean=motor_inh_weight,
        text_input_to_motor_density=0.30,
        text_input_to_motor_weight=3.0,
        text_input_to_motor_jitter=0.5,
        enable_motor_fs=True,
        n_motor_fs_per_action=n_fs_per_pool,
        enable_language_output=True,
        n_lang_output=n_lang_input,
        motor_to_language_output_weight=2.0,
        enable_noun_pools=True,
        noun_pool_names=NOUN_NAMES,
        n_noun_per_pool=n_per_pool,
        n_noun_fs_per_pool=n_fs_per_pool,
        enable_verb_pools=True,
        verb_pool_names=VERB_NAMES,
        n_verb_per_pool=n_per_pool,
        n_verb_fs_per_pool=n_fs_per_pool,
        enable_adjective_pools=enable_adjective,
        adjective_pool_names=ADJECTIVE_NAMES if enable_adjective else None,
        n_adjective_per_pool=n_per_pool,
        n_adjective_fs_per_pool=n_fs_per_pool,
        concept_pool_internal_density=concept_internal_density,
        concept_pool_exc_weight_mean=concept_exc_weight,
        concept_pool_inh_weight_mean=concept_inh_weight,
        enable_hippocampus_consolidation=True,
    )

    # ADD: dlpfc_wm region (g11_bg_runner.py:411-427 pattern verbatim
    # modulo enable_nmda which we set True for the (c) NMDA bistable
    # frame-holding role; matches Cluster G v2 design).
    regions.append(BrainRegion(
        name="dlpfc_wm",
        n_neurons=DLPFC_N_NEURONS,
        exc_fraction=DLPFC_EXC_FRACTION,
        internal_density=DLPFC_INTERNAL_DENSITY,
        exc_weight_mean=DLPFC_EXC_WEIGHT,
        inh_weight_mean=DLPFC_INH_WEIGHT,
        weight_jitter=DLPFC_WEIGHT_JITTER,
        plastic_internal=True,
        izh_neuron_type=NeuronType.IZH2007_HIPPO_PYRAMIDAL.name,
        enable_nmda=True,
    ))
    # ADD: lang_input -> dlpfc_wm pathway (so the (c) loop can inject
    # the encoded PFC frame via lang_input). Plasticity gate "lang_to_
    # dlpfc_wm" so the (c) training can train it; for THIS probe, the
    # gate is closed during standard training so dlpfc_wm only sees
    # lang_input passively.
    pathways.append(RegionPathway(
        from_region="language_input", to_region="dlpfc_wm",
        density=LANG_TO_DLPFC_DENSITY,
        weight_mean=LANG_TO_DLPFC_WEIGHT,
        weight_jitter=LANG_TO_DLPFC_JITTER,
        plastic=True,
        plasticity_gate="lang_to_dlpfc_wm",
    ))

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 0.5
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.nmda_tau_decay = 100.0
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = False
    cfg.enable_short_term_plasticity = False
    cfg.stdp_w_max = 8.0
    cfg.fast_spike_reset = True

    bridge = SimulationBridge(
        core_config=cfg,
        viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(),
        gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    # Close the lang_to_dlpfc_wm gate for standard training (probe
    # tests dlpfc_wm region PRESENCE; (c) loop will open this gate
    # later when injecting the frame).
    try:
        bridge.set_plasticity_gate("lang_to_dlpfc_wm", 0.0)
    except Exception:
        pass

    if verbose:
        n_concept_pools = (4 + 4 + 4 + (4 if enable_adjective else 0))
        total_neurons = sum(r.n_neurons for r in regions)
        print(f"[BUILD] hippo+dlpfc concept-pool bridge: "
              f"{total_neurons} neurons total, "
              f"{n_concept_pools} concept pools "
              f"({n_concept_pools * n_per_pool} pool neurons); "
              f"hippocampus (EC/DG/CA3/CA1) + Phase 1.3 SWR "
              f"consolidation pathways + dlpfc_wm "
              f"({DLPFC_N_NEURONS} neurons, NMDA bistable; "
              f"lang_input -> dlpfc_wm pathway plasticity-gated OFF "
              f"for this probe) PRESENT", flush=True)
    return bridge


def _build_and_train(seed, smoke, verbose):
    bridge_p = _bridge_save_path(seed, smoke)
    enable_adjective = not smoke

    if smoke:
        words = (list(DIRECTION_VOCAB) + list(NOUN_VOCAB) +
                 list(VERB_VOCAB))
        n_train_events = SMOKE_N_TRAIN_EVENTS
    else:
        words = (list(DIRECTION_VOCAB) + list(NOUN_VOCAB) +
                 list(VERB_VOCAB) + list(ADJECTIVE_VOCAB))
        n_train_events = DEFAULT_N_TRAIN_EVENTS
    word_to_idx = {w: i for i, w in enumerate(words)}
    n_words = len(words)

    t0 = time.time()
    bridge = _build_bridge_with_hippo_and_dlpfc(
        seed=seed, enable_adjective=enable_adjective, verbose=verbose)
    if os.path.exists(bridge_p):
        if verbose:
            print(f"  [seed {seed}] loading cached trained bridge "
                  f"({bridge_p})", flush=True)
        bridge.load_checkpoint(bridge_p)
        for g in ("language_input_to_motor",
                  "language_input_to_noun_pool",
                  "language_input_to_verb_pool",
                  "language_input_to_adjective_pool",
                  "motor_to_language_output",
                  "noun_pool_to_language_output",
                  "verb_pool_to_language_output",
                  "adjective_pool_to_language_output",
                  "lang_to_dlpfc_wm"):
            try:
                bridge.set_plasticity_gate(g, 0.0)
            except Exception:
                pass
        return bridge, words, word_to_idx

    if verbose:
        print(f"  [seed {seed}] training v16 substrate WITH "
              f"hippocampus + dlpfc_wm ({n_words} words x "
              f"{n_train_events} events)", flush=True)
    apply_concept_topographic_bias(
        bridge,
        n_lang_input=DEFAULT_N_LANG_INPUT,
        topographic_factor=DEFAULT_TOPOGRAPHIC_FACTOR,
        off_target_factor=DEFAULT_OFF_TARGET_FACTOR,
        sparsity=DEFAULT_SPARSITY,
        orthogonal_codes=True,
        n_words_for_orthogonal=n_words,
        word_to_idx=word_to_idx,
        verbose=verbose,
    )

    rng = np.random.default_rng(seed)
    target_pool = {}
    for w in DIRECTION_VOCAB:
        if w == "north": target_pool[w] = "motor_N"
        elif w == "east": target_pool[w] = "motor_E"
        elif w == "south": target_pool[w] = "motor_S"
        elif w == "west": target_pool[w] = "motor_W"
    for w in NOUN_VOCAB:
        target_pool[w] = f"noun_pool_{w.upper()}"
    for w in VERB_VOCAB:
        target_pool[w] = f"verb_pool_{w.upper()}"
    if enable_adjective:
        for w in ADJECTIVE_VOCAB:
            target_pool[w] = f"adjective_pool_{w.upper()}"

    schedule = []
    for w in words:
        for _ in range(n_train_events):
            schedule.append(w)
    rng.shuffle(schedule)
    if verbose:
        print(f"  [seed {seed}] interleaved schedule: "
              f"{len(schedule)} events", flush=True)

    for ei, w in enumerate(schedule):
        train_word_to_pool(
            bridge, word=w, target_pool_region=target_pool[w],
            n_events=1, n_lang_input=DEFAULT_N_LANG_INPUT,
            n_lang_output=DEFAULT_N_LANG_INPUT,
            sparsity=DEFAULT_SPARSITY,
            orthogonal_codes=True,
            n_words_for_orthogonal=n_words,
            word_to_idx=word_to_idx,
            verbose=False,
        )
        if verbose and (ei + 1) % max(1, len(schedule) // 10) == 0:
            print(f"    [seed {seed}] {ei+1}/{len(schedule)} events "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
    bridge.save_checkpoint(bridge_p)
    if verbose:
        print(f"  [seed {seed}] trained + saved in "
              f"{(time.time()-t0)/60:.1f} min", flush=True)
    return bridge, words, word_to_idx


def run_one_seed(seed, smoke, xp, verbose):
    print(f"\n--- seed {seed} ---", flush=True)
    enable_adjective = not smoke
    bridge, words, word_to_idx = _build_and_train(seed, smoke, verbose)

    cache_p = _activity_cache_path(seed, smoke)
    if os.path.exists(cache_p):
        if verbose:
            print(f"  [seed {seed}] loading cached activity "
                  f"({cache_p})", flush=True)
        acts = _load_activity_cache(cache_p, words)
        n_pool_union = acts[words[0]].shape[1]
    else:
        acts, n_pool_union = _capture_concept_pool_activity(
            bridge, words, word_to_idx, smoke, enable_adjective, verbose)
        np.savez_compressed(cache_p,
                            **{str(w): acts[w] for w in words})
    d_act = n_pool_union

    k_vocab = SMOKE_M_OBS if smoke else K_VOCAB_TARGET
    consolidated = {w: acts[w][:k_vocab].mean(axis=0) for w in words}
    grounded = _ground_symbols(consolidated, words, d_act)
    max_diff, vocab_phase_matrix = verify_batched_equivalent_to_scalar(
        grounded, list(words), xp, rng_seed=seed)
    if verbose:
        print(f"  [seed {seed}] V={len(words)} d_act={d_act}; "
              f"batched-vs-scalar max-diff={max_diff:.2e}", flush=True)

    positions = gamma_slot_positions(seed, 7, N_DIM)
    net = ResonateFireFHRR(N_DIM, np.random.default_rng(seed))
    qrng = np.random.default_rng(seed + 1)
    loads = SMOKE_LOADS if smoke else LOADS
    n_trials = (max(1, N_TRIALS // 4) if smoke else N_TRIALS)

    per_load = {}
    V = len(words)
    for load in loads:
        ob_ok = oi_ok = 0
        for _ in range(n_trials):
            items_idx = tuple(int(i) for i in
                              qrng.choice(V, size=load, replace=False))
            items = [words[i] for i in items_idx]
            C = net.encode([(grounded[items[k]], positions[k])
                            for k in range(load)])
            unbinds = [net.query(C, positions[k]) for k in range(load)]
            recovered = []
            scores_oi_gpu = xp.zeros(V)
            for k in range(load):
                sims_k = batched_phase_similarity(
                    unbinds[k], vocab_phase_matrix, xp)
                recovered.append(int(xp.argmax(sims_k)))
                scores_oi_gpu = scores_oi_gpu + sims_k
            if tuple(recovered) == items_idx:
                ob_ok += 1
            scores_oi_host = to_host(scores_oi_gpu)
            topK = sorted(
                int(i) for i in np.argsort(scores_oi_host)[-load:])
            if tuple(topK) == tuple(sorted(items_idx)):
                oi_ok += 1
        per_load[load] = {
            "order_bearing_accuracy": ob_ok / n_trials,
            "order_invariant_accuracy": oi_ok / n_trials,
            "n_trials": n_trials,
        }
        print(f"    L={load}: OB={per_load[load]['order_bearing_accuracy']:.3f} "
              f"OI={per_load[load]['order_invariant_accuracy']:.3f}",
              flush=True)
    return per_load, V, d_act, max_diff


def main():
    ap = argparse.ArgumentParser(
        description="DLPFC_WM-extension of HIPPO-OPTION3")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    smoke = bool(args.smoke)

    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print("=== DLPFC-extension of HIPPO-OPTION3: parallel-matching "
          "mode-unification on bio_brain_regions WITH hippocampus + "
          "dlpfc_wm ===", flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    if smoke:
        print("  *** SMOKE MODE ***", flush=True)
    print(f"  bar={BAR}; K_VOCAB={K_VOCAB_TARGET if not smoke else SMOKE_M_OBS}; "
          f"decoder=parallel_population_matching (batched)", flush=True)

    seeds = [42] if smoke else list(SEEDS)
    loads = SMOKE_LOADS if smoke else LOADS

    seed_results = []
    t0 = time.time()
    for seed in seeds:
        t_seed = time.time()
        per_load, V, d_act, max_diff = run_one_seed(
            seed, smoke, xp, verbose=True)
        seed_results.append({
            "seed": seed, "V": V, "d_act": d_act,
            "batched_vs_scalar_max_diff": max_diff,
            "per_load": {str(l): v for l, v in per_load.items()},
        })
        print(f"  [seed {seed} done in {(time.time()-t_seed)/60:.1f} min]",
              flush=True)
    total_time = time.time() - t0
    print(f"\nTotal wall-clock: {total_time/60:.1f} min "
          f"(backend={backend_name})", flush=True)

    agg = {}
    ob_all_pass = oi_all_pass = True
    cells = []
    for load in loads:
        obs = [r["per_load"][str(load)]["order_bearing_accuracy"]
               for r in seed_results]
        ois = [r["per_load"][str(load)]["order_invariant_accuracy"]
               for r in seed_results]
        ob_m = float(np.mean(obs)); oi_m = float(np.mean(ois))
        agg[load] = {"order_bearing_mean": ob_m,
                     "order_bearing_per_seed": obs,
                     "order_invariant_mean": oi_m,
                     "order_invariant_per_seed": ois}
        cells.append(f"{ob_m:.3f} {oi_m:.3f}")
        if ob_m < BAR:
            ob_all_pass = False
        if oi_m < BAR:
            oi_all_pass = False
    print(f"\n=== MULTI-SEED AGGREGATE ===", flush=True)
    print(f"  multi-seed:  {'   '.join(cells)}", flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    if smoke:
        verdict = "SMOKE"
        print("  SMOKE -- numbers NOT propagated.", flush=True)
    elif ob_all_pass and oi_all_pass:
        verdict = "DLPFC_PASS"
        print("  Parallel-matching mode-unification PASSES on the "
              "bio_brain_regions substrate WITH hippocampus + "
              "dlpfc_wm PRESENT. All FIVE load-bearing components of"
              " (c) generative-replay validated on a single coherent "
              "substrate. The (c) TDD plan + loop-controller build "
              "is the next pre-registered direction. NOT yet a "
              "capability claim -- pending fresh adversarial review.",
              flush=True)
    else:
        verdict = "DLPFC_NEGATIVE"
        print("  dlpfc_wm presence breaks the substrate-grounding. "
              "Biology-translatable: the NMDA bistable PFC region "
              "modulates cortex baseline enough to break parallel-"
              "matching at this scale; (c) needs a different "
              "integration path. Honest characterisation.", flush=True)

    out = {
        "backend": backend_name, "gpu": gpu,
        "smoke": smoke,
        "seeds": seeds, "loads": loads, "bar": BAR,
        "k_vocab": SMOKE_M_OBS if smoke else K_VOCAB_TARGET,
        "decoder_order_bearing": "parallel_population_matching_batched",
        "decoder_order_invariant": "marginal_sum_phase_similarity_batched",
        "substrate": ("build_biological_brain_regions_v16_recipe_"
                       "WITH_HIPPO_AND_DLPFC"),
        "per_seed": seed_results,
        "aggregate": {str(l): v for l, v in agg.items()},
        "verdict": verdict,
        "wall_clock_minutes": total_time / 60,
    }
    tag = "smoke" if smoke else "full"
    out_path = os.path.join(
        _HERE,
        f"mode_unification_with_hippo_dlpfc_probe_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
