"""OPTION 3 cheap-first probe: does parallel-matching biologized
mode-unification PASS on the build_biological_brain_regions
concept-pool substrate's activity?

CONTEXT (autonomous: owner gave standing autonomy + said "go with
whatever you think is most effective to reach our goals").
- (b) parallel-matching mode-unification VALIDATED on G.20 sparse
  bridgeA_nouns (pillar n=93).
- (e) per-bridge across all 5 bridges of the 160-concept G.20 sparse
  ensemble ENSEMBLE-PASS (pillar n=94).
- (n=95) cross-bridge 160-concept union BOUNDARY (OB perfect; OI L=5
  ceilings).
- (c) generative-replay design surfaced with 3 substrate options;
  OPTION 3 (use build_biological_brain_regions with hippocampus +
  dlpfc_wm + Phase 1.3 SWR consolidation already validated) is the
  cleanest biology match for the conversational arc -- IF parallel-
  matching mode-unification PASSes on this substrate's concept-pool
  activity. This probe ANSWERS that question empirically (vs further
  deliberation).

WHAT THIS PROBE TESTS
- Substrate: build_concept_bridge from concept_pool_demo (the
  validated v14/v16 16-pool architecture: 4 motor + 4 noun + 4 verb
  + 4 adjective = 16 distinct concept pools). The SAME architecture
  with v14's 88.75% W->A multi-seed binding capability. NO hippocampus
  for this cheap-first probe (the hippocampus addition is a separate
  follow-up; basic mode-unification grounding must PASS first).
- Capture: per-neuron firing across the union of all 16 concept-pool
  neurons (16 pools x 200 neurons = 3200 neurons total) when each
  word's lang_input drive is applied. M_OBS=16 observations per
  concept; mean-centred GLOBALLY across the 16 concepts.
- Grounded symbols: mean-centred activity -> fixed-seed deriver ->
  spike-phase representation (same DERIV_SEED=90909 as (b)/(e)/n=95).
- Decoder: parallel-population matching (per-slot argmax over the 16
  concepts; biology-grounded: dendritic-integration + lateral-
  inhibition WTA). The vocabulary at the decoder = the 16 substrate-
  derived grounded symbols (NOT a hand-supplied table).
- Reuses the (b)/(e)/n=95 mode-unification pipeline byte-unchanged
  via import.

PRE-REGISTERED reading (fixed; never tuned):
- OPTION3_BASIC_PASS: multi-seed-mean >= 0.80 at every load {2, 3, 5}
  on BOTH order-bearing AND order-invariant readouts. The basic
  substrate-grounding works on build_biological_brain_regions concept
  pools; (c) generative-replay can build on this substrate (with
  hippocampus addition as a separate follow-up step).
- OPTION3_BASIC_NEGATIVE: either readout misses at some load; honest
  characterisation. Biology-translatable: the build_biological_brain_-
  regions concept-pool activity geometry doesn't ground-symbol cleanly
  for parallel-matching mode-unification; OPTION 1 substrate-merge
  may be required for (c).

WALL-CLOCK ESTIMATE: per seed ~17 min train (v16 production recipe) +
~3 min capture + ~3 min pipeline = ~25 min/seed. Smoke mode is much
cheaper (~5 min total).

KILL-SAFE via per-seed save_bridge cache; reuses substrate across
seeds where possible.

Reuses every primitive byte-unchanged; no protected/frozen/moat
module modified; no autograd; no-confab moat must stay 7/7 green.
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

# Reuse-by-import only.
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
from research.runners.resonate_fire_fhrr import ResonateFireFHRR
from research.runners.spiking_phasor_fhrr import (
    phases_to_spikes,
)
from research.runners.concept_pool_demo import (
    build_concept_bridge, apply_concept_topographic_bias,
    train_word_to_pool, DIRECTION_VOCAB, NOUN_VOCAB, VERB_VOCAB,
    ADJECTIVE_VOCAB,
)
from sim.backend import get_backend, is_gpu_backend, to_host

DEFAULT_N_LANG_INPUT = 2048
DEFAULT_N_PER_POOL = 200
DEFAULT_N_FS_PER_POOL = 24
DEFAULT_TOPOGRAPHIC_FACTOR = 3.0
DEFAULT_OFF_TARGET_FACTOR = 0.3
DEFAULT_SPARSITY = 0.05
DEFAULT_N_TRAIN_EVENTS = 200
M_OBS_FULL = K_VOCAB_TARGET  # 16

SMOKE_VOCAB = 12  # 4 motor + 4 noun + 4 verb (bridge always builds verb pools)
SMOKE_N_TRAIN_EVENTS = 30
SMOKE_M_OBS = 4
SMOKE_LOADS = [2, 3]

CACHE_DIR = os.path.join(
    _HERE, "mode_unification_on_bio_brain_regions_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _bridge_save_path(seed, smoke):
    tag = "smoke" if smoke else "full"
    return os.path.join(CACHE_DIR, f"bridge_{tag}_seed{seed}.simstate.h5")


def _activity_cache_path(seed, smoke):
    tag = "smoke" if smoke else "full"
    return os.path.join(CACHE_DIR, f"activity_{tag}_seed{seed}.npz")


def _build_and_train(seed, smoke, verbose):
    """v16 production recipe build + train; kill-safe via save."""
    bridge_p = _bridge_save_path(seed, smoke)
    enable_adjective = not smoke

    if smoke:
        # Bridge always builds noun+verb pools; include verb vocab so
        # apply_concept_topographic_bias finds every pool's word.
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
    bridge = build_concept_bridge(
        seed=seed,
        n_lang_input=DEFAULT_N_LANG_INPUT,
        n_per_pool=DEFAULT_N_PER_POOL,
        n_fs_per_pool=DEFAULT_N_FS_PER_POOL,
        enable_adjective=enable_adjective,
        weak_dynamics=True,
        verbose=verbose,
    )
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
                  "adjective_pool_to_language_output"):
            try:
                bridge.set_plasticity_gate(g, 0.0)
            except Exception:
                pass
        return bridge, words, word_to_idx

    if verbose:
        print(f"  [seed {seed}] training v16 substrate "
              f"({n_words} words x {n_train_events} events)", flush=True)
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


def _capture_concept_pool_activity(bridge, words, word_to_idx, smoke,
                                    enable_adjective, verbose):
    """Per-neuron activity vectors across the union of concept-pool
    neurons. Per word: drive lang_input(word); record per-neuron
    firing over stim window; M_OBS observations."""
    from sim.backend import get_backend
    cp, _ = get_backend()
    from sim.text_embeddings import orthogonal_drive_pattern

    rm = bridge.region_manager
    n_words = len(words)
    m_obs = SMOKE_M_OBS if smoke else M_OBS_FULL

    pool_names = [f"motor_{d}" for d in ("N", "E", "S", "W")]
    pool_names += [f"noun_pool_{w.upper()}" for w in NOUN_VOCAB]
    pool_names += [f"verb_pool_{w.upper()}" for w in VERB_VOCAB]
    if enable_adjective:
        pool_names += [f"adjective_pool_{w.upper()}" for w in
                        ADJECTIVE_VOCAB]
    pool_idx_lists = []
    for p in pool_names:
        pool_idx_lists.extend(list(rm.indices(p)))
    pool_idx_arr_host = np.asarray(pool_idx_lists, dtype=np.int64)
    pool_idx_arr = cp.asarray(pool_idx_arr_host)
    n_pool_union = pool_idx_arr.shape[0]
    if verbose:
        print(f"  [capture] {len(pool_names)} pools, "
              f"{n_pool_union} pool-union neurons", flush=True)

    lang_input_idx = list(rm.indices("language_input"))
    lang_input_arr_host = np.asarray(lang_input_idx, dtype=np.int64)
    lang_input_arr = cp.asarray(lang_input_arr_host)

    stim_steps = 50
    reset_steps = 25

    acts = {}
    for w in words:
        observations = np.zeros((m_obs, n_pool_union), dtype=np.float32)
        for obs in range(m_obs):
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(reset_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
            drive_in = orthogonal_drive_pattern(
                cue_idx=word_to_idx[w], n_cues=n_words,
                n_neurons=DEFAULT_N_LANG_INPUT,
                drive_max_pA=200.0, sparsity=DEFAULT_SPARSITY)
            drive_in_gpu = cp.asarray(drive_in, dtype=cp.float32)
            bridge.cp_external_input_current[lang_input_arr] = drive_in_gpu
            counts = cp.zeros(n_pool_union, dtype=cp.float32)
            for _ in range(stim_steps):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
                fired = bridge.cp_firing_states[pool_idx_arr]
                counts = counts + fired.astype(cp.float32)
            observations[obs] = cp.asnumpy(counts)
        acts[w] = observations
        if verbose:
            density = float(np.mean(observations > 0.0))
            mean_rate = float(np.mean(observations))
            print(f"    [capture] '{w}': mean_rate {mean_rate:.4f} "
                  f"density {density:.4f}", flush=True)
    return acts, n_pool_union


def _ground_symbols(consolidated, words, d_act):
    """Same pattern as (b)/(e)/n=95."""
    common = np.mean([consolidated[w] for w in words], axis=0)
    deriver = make_deriver(N_DIM, d_act, DERIV_SEED)
    return {w: phases_to_spikes(deriver(consolidated[w] - common))
            for w in words}


def _load_activity_cache(cache_p, words):
    """Plain npz loader; numeric arrays only; no pickle."""
    data = np.load(cache_p)
    return {w: data[str(w)] for w in words}


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
        if verbose:
            print(f"  [seed {seed}] cached activity ({cache_p})",
                  flush=True)
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
        description="OPTION 3 cheap-first probe: parallel-matching "
                    "mode-unification on build_biological_brain_regions "
                    "concept-pool activity")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny build + few obs/trials; numbers NOT "
                         "propagated as a result")
    args = ap.parse_args()
    smoke = bool(args.smoke)

    xp, backend_name = get_backend()
    gpu = is_gpu_backend()
    print("=== OPTION 3 cheap-first probe: parallel-matching mode-"
          "unification on build_biological_brain_regions ===",
          flush=True)
    print(f"  backend={backend_name} (GPU={gpu})", flush=True)
    print(f"  Reuses (b)/(e)/n=95 mode-unification primitives byte-"
          f"unchanged; build_concept_bridge + train via v16 recipe "
          f"byte-unchanged", flush=True)
    if smoke:
        print("  *** SMOKE MODE: tiny bridge + few obs/trials; NOT a "
              "result ***", flush=True)
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

    print(f"\n=== MULTI-SEED AGGREGATE ===", flush=True)
    print(f"            L=2 OB   OI    L=3 OB   OI    L=5 OB   OI",
          flush=True)
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
    print(f"  multi-seed:  {'   '.join(cells)}", flush=True)

    print(f"\n=== VERDICT ===", flush=True)
    if smoke:
        verdict = "SMOKE"
        print(f"  SMOKE -- numbers NOT propagated.", flush=True)
    elif ob_all_pass and oi_all_pass:
        verdict = "OPTION3_BASIC_PASS"
        print(f"  Parallel-matching mode-unification PASSES on the "
              f"build_biological_brain_regions concept-pool substrate. "
              f"(c) generative-replay can build on this substrate; "
              f"hippocampus + dlpfc_wm + Phase 1.3 SWR consolidation "
              f"addition is a separate follow-up step. NOT yet a "
              f"capability claim -- pending fresh dedicated adversarial "
              f"review.", flush=True)
    else:
        verdict = "OPTION3_BASIC_NEGATIVE"
        print(f"  Either readout misses the bar. The build_biological"
              f"_brain_regions concept-pool activity geometry doesn't "
              f"ground-symbol cleanly for parallel-matching mode-"
              f"unification at this scale; OPTION 1 substrate-merge "
              f"may be required for (c). Honest characterisation.",
              flush=True)

    out = {
        "backend": backend_name, "gpu": gpu,
        "smoke": smoke,
        "seeds": seeds, "loads": loads, "bar": BAR,
        "k_vocab": SMOKE_M_OBS if smoke else K_VOCAB_TARGET,
        "decoder_order_bearing": "parallel_population_matching_batched",
        "decoder_order_invariant": "marginal_sum_phase_similarity_batched",
        "substrate": "build_biological_brain_regions_v16_recipe",
        "per_seed": seed_results,
        "aggregate": {str(l): v for l, v in agg.items()},
        "verdict": verdict,
        "wall_clock_minutes": total_time / 60,
    }
    tag = "smoke" if smoke else "full"
    out_path = os.path.join(
        _HERE,
        f"mode_unification_on_bio_brain_regions_probe_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
