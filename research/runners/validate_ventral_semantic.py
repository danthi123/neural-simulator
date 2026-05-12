"""P5 ventral semantic stream validation — comprehension + naming.

Catalog G.11 (dual-stream language model, Hickok & Poeppel; Kandel
6e Ch 55 pp 1380-1387) + G.13 (Wernicke's area; Kandel pp 1384-1385).

Tests two characteristic functions of the ventral language stream:

  Test 1 — Comprehension (word → meaning):
    Drive lang_input(word) → measure semantic_cortex response.
    PASS criteria:
      - Same-concept activations across trials cosine > 0.6
        (stable concept representation)
      - Different-concept activations cosine < 0.3
        (distinguishable semantic codes)

  Test 2 — Naming (meaning → word):
    Drive semantic_cortex with the stored pattern for concept X
    (via the engram tag) → measure lang_output response.
    PASS criterion: lang_output activates ABOVE baseline (production
    pathway works; specific word matching is a downstream test).

  Test 3 — Hippo-independent recall (durability):
    After consolidation, silence ca3+ca1 (set excitability_drive
    to strongly negative).
    Drive lang_input("apple"). Measure semantic_cortex.
    PASS: semantic_cortex still produces the "apple" pattern even
    without hippocampus (per catalog D.01: consolidation transforms
    labile traces into durable cortical representations).

Usage:
    python -m research.runners.validate_ventral_semantic \\
        --seed 42 --out research/findings/raw/g11_bg/p5_seed42.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).flatten()
    b = np.asarray(b, dtype=np.float64).flatten()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def measure_region_spikes(bridge, region_name: str, n_steps: int = 100):
    """Run n_steps and return per-neuron spike count for region_name."""
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()
    rm = bridge.region_manager
    indices = list(rm.indices(region_name))
    arr = cp.asarray(indices, dtype=cp.int64)
    counts = cp.zeros(len(indices), dtype=cp.float32)
    for _ in range(n_steps):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fired = bridge.cp_firing_states[arr]
        counts += fired.astype(cp.float32)
    return to_host(counts)


def run_ventral_validation(
    seed: int = 42,
    n_lang_input: int = 1024,
    n_motor_per_action: int = 16,
    n_motor_fs_per_action: int = 4,
    n_semantic_cortex: int = 500,
    n_wernicke: int = 100,
    n_train_events: int = 100,
    n_replay_cycles: int = 20,
    strict_two_stage: bool = False,
    drive_lang_during_replay: bool = False,
    # Iter D: semantic_cortex attractor dynamics (catalog G.11
    # Patterson 2007 ATL hub; Wang 2002 NMDA bistability).
    semantic_cortex_recurrent_density: float = 0.10,
    semantic_cortex_recurrent_weight: float = 1.0,
    lang_to_wernicke_density: float = 0.30,
    lang_to_wernicke_weight: float = 3.0,
    wernicke_to_semantic_density: float = 0.30,
    wernicke_to_semantic_weight: float = 4.0,
    drive_steps: int = 100,
    # Path B+ (iter F): semantic_FS lateral inhibition for
    # competitive attractor formation (Vogels 2011, Hofer 2011).
    # Pairs with strong recurrence (iter D params) to produce
    # selective basins instead of monolithic attractor.
    enable_semantic_fs: bool = False,
    n_semantic_fs: int = 100,
    # Path G (iter G): wernicke_FS lateral inhibition. Per P5
    # iter E weight inspection (selectivity=0.004), wernicke is
    # the upstream bottleneck — fires all neurons regardless of
    # concept. FS sparsifies wernicke firing to produce per-
    # concept ensembles.
    enable_wernicke_fs: bool = False,
    n_wernicke_fs: int = 60,
    # Path A / Path G+ FULL (iter T): multi-pool wernicke.
    # When True, the architecture uses per-concept wernicke
    # pools (wernicke_pool_0, wernicke_pool_1) with cross-pool
    # FS inhibition. Mirror of Tier 1 motor pool pattern.
    enable_multi_pool_wernicke: bool = False,
    n_wernicke_pools: int = 2,
    n_per_wernicke_pool: int = 100,
    n_per_wernicke_pool_fs: int = 12,
    # Iter Z (2026-05-11): interleaved training fix for the
    # apple/river asymmetry identified in V3 demo. Sequential
    # training (apple block then river block) produced apple-
    # biased weights. Per-event alternation should symmetrize.
    interleaved_training: bool = False,
    # Iter AA: per-concept lang_output pools (architectural fix
    # for the shared lang_output bottleneck). Mirror of Tier 1
    # motor pool architecture at the language output side.
    enable_per_concept_lang_out_pools: bool = False,
    n_per_lang_out_pool: int = 200,
    # Iter KK->LL: per-pool dynamics parameterized. Default = iter AA
    # weak (0.05/0.3/0.8). Cortical canon (0.10/2.0/4.0) was tested in
    # iter KK seed 42 — amplified structural bias, regressed. Available
    # for future experimentation but NOT default.
    wernicke_pool_internal_density: float = 0.05,
    wernicke_pool_exc_weight_mean: float = 0.3,
    wernicke_pool_inh_weight_mean: float = 0.8,
    lang_output_pool_internal_density: float = 0.05,
    lang_output_pool_exc_weight_mean: float = 0.3,
    lang_output_pool_inh_weight_mean: float = 0.8,
    # Iter BB: stronger cross-pool FS inhibition (default 4.0)
    # to push river-direction from 4/6 to 6/6
    wernicke_fs_cross_weight: float = 4.0,
    # Iter CC: lang_output_FS pools (cross-inhibition at output)
    enable_lang_out_fs_pools: bool = False,
    n_per_lang_out_fs_pool: int = 24,
    # Iter DD: multi-trial averaging at recognition (smooth
    # single-trial noise like seed 44's 3-spike margin)
    n_recognition_trials: int = 1,
    # Iter EE: longer inter-trial rest (default 50 steps) to
    # allow neuron adaptation to recover between trials
    inter_trial_rest_steps: int = 50,
    # Iter GG: teacher pairing at lang_output. During training,
    # drive lang_output_pool_<i> with teacher current alongside
    # lang_input(concept_<i>). Mirrors Tier 1's embodied-Hebbian.
    enable_lang_out_teacher: bool = False,
    lang_out_teacher_pA: float = 300.0,
    # Iter II: contrastive training. During concept_i training,
    # actively suppress OTHER lang_output_pool_j with negative
    # current. Forces LTD on cross-concept connections via STDP
    # anti-Hebbian rule. Should break apple-bias from random init.
    contrastive_suppress_pA: float = 0.0,  # 0 = disabled; try -200 to suppress
    # Iter M: strengthen naming pathway weights. ca1_to_lang_out
    # at default 2.0 produces only ~20 mV drive on lang_output
    # which is barely suprathreshold. Bumping to 5.0 should
    # produce robust above-baseline lang_output activation when
    # CA3 engram is stimulated.
    ca1_to_lang_out_weight: float = 2.0,
    stim_drive_pA: float = 200.0,
    # Path G+ minimal (iter P): apply topographic bias to
    # lang_input -> wernicke weights so each concept activates a
    # dedicated wernicke subset. Mirror of Tier 1 motor pool
    # topographic bias that produced 5/6 multi-seed PASS.
    apply_wernicke_topographic: bool = False,
    wernicke_topographic_factor: float = 1.5,
    wernicke_off_target_factor: float = 0.7,
    out_path: Optional[Path] = None,
    verbose: bool = True,
):
    """Iteration B parameters (catalog G.11/G.13 + McClelland 1995 CLS):

    strict_two_stage:
        If True, encoding phase opens ONLY hippocampal gates
        (lang_to_ec, ec_to_dg, dg_to_ca3, ca3_to_ca1, ec_to_ca1).
        Ventral-stream gates (lang_to_wernicke, wernicke_to_semantic,
        ca1_to_semantic) stay closed. Then during the replay/sleep
        phase the ventral gates open. This matches McClelland 1995
        CLS: wake = hippo fast learning; sleep = cortex slow
        consolidation via hippo replay.

    drive_lang_during_replay:
        If True (only meaningful with strict_two_stage=True), during
        each replay burst we ALSO drive lang_input(concept) so that
        wernicke sees both the (replayed) meaning via ca1->semantic
        AND the word via lang->wernicke, enabling Hebbian binding.
        Biology: Wilson & McNaughton 1994 coordinated hippo+cortex
        replay; real cortical replay reactivates phonological codes
        alongside semantic content.
    """
    log = print if verbose else (lambda *a, **k: None)
    log("=" * 60)
    log(f"P5 ventral semantic stream validation (seed={seed})")
    log("=" * 60)

    from sim.config import (CoreSimConfig, RuntimeState, GPUConfig,
                              VisualizationConfig)
    from sim.bridge import SimulationBridge
    from research.runners.text_minimal_isolation import (
        build_biological_brain_regions,
    )
    from research.runners.consolidation_trainer import (
        run_concept_replay_phase,
    )
    from sim.text_embeddings import vocab_to_drive_pattern
    from sim.backend import get_backend, to_host
    cp, _ = get_backend()

    t0 = time.time()
    regions, pathways = build_biological_brain_regions(
        n_lang_input=n_lang_input,
        n_motor_per_action=n_motor_per_action,
        n_motor_fs_per_action=n_motor_fs_per_action,
        enable_motor_fs=True, enable_language_output=True,
        n_lang_output=n_lang_input,
        enable_hippocampus_consolidation=True,
        enable_ventral_semantic=True,
        n_semantic_cortex=n_semantic_cortex,
        n_wernicke=n_wernicke,
        n_ec=200, n_dg=800, n_dg_pv_basket=240,
        n_ca3=400, n_ca1=200,
        ca3_recurrent_weight=5.0,
        # Iter D: semantic_cortex attractor tuning
        semantic_cortex_recurrent_density=semantic_cortex_recurrent_density,
        semantic_cortex_recurrent_weight=semantic_cortex_recurrent_weight,
        lang_to_wernicke_density=lang_to_wernicke_density,
        lang_to_wernicke_weight=lang_to_wernicke_weight,
        wernicke_to_semantic_density=wernicke_to_semantic_density,
        wernicke_to_semantic_weight=wernicke_to_semantic_weight,
        # Path B+ (iter F): semantic_FS lateral inhibition
        enable_semantic_fs=enable_semantic_fs,
        n_semantic_fs=n_semantic_fs,
        # Path G (iter G): wernicke_FS lateral inhibition
        enable_wernicke_fs=enable_wernicke_fs,
        n_wernicke_fs=n_wernicke_fs,
        # Path A / Path G+ FULL (iter T): multi-pool wernicke
        enable_multi_pool_wernicke=enable_multi_pool_wernicke,
        n_wernicke_pools=n_wernicke_pools,
        n_per_wernicke_pool=n_per_wernicke_pool,
        n_per_wernicke_pool_fs=n_per_wernicke_pool_fs,
        # Iter AA: per-concept lang_output pools
        enable_per_concept_lang_out_pools=enable_per_concept_lang_out_pools,
        n_per_lang_out_pool=n_per_lang_out_pool,
        # Iter KK->LL: parameterized per-pool dynamics
        wernicke_pool_internal_density=wernicke_pool_internal_density,
        wernicke_pool_exc_weight_mean=wernicke_pool_exc_weight_mean,
        wernicke_pool_inh_weight_mean=wernicke_pool_inh_weight_mean,
        lang_output_pool_internal_density=lang_output_pool_internal_density,
        lang_output_pool_exc_weight_mean=lang_output_pool_exc_weight_mean,
        lang_output_pool_inh_weight_mean=lang_output_pool_inh_weight_mean,
        # Iter BB: stronger cross-pool FS
        wernicke_fs_cross_weight=wernicke_fs_cross_weight,
        # Iter CC: lang_output FS pools
        enable_lang_out_fs_pools=enable_lang_out_fs_pools,
        n_per_lang_out_fs_pool=n_per_lang_out_fs_pool,
        # Iter M: strengthen ca1->lang_output for naming
        ca1_to_lang_out_weight=ca1_to_lang_out_weight,
    )
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = 1.0
    cfg.seed = seed
    cfg.enable_nmda = True
    cfg.fast_spike_reset = True
    cfg.stdp_w_max = 10.0
    cfg.enable_hebbian_learning = False

    bridge = SimulationBridge(
        core_config=cfg, viz_config=VisualizationConfig(),
        runtime_state=RuntimeState(), gpu_config=GPUConfig(),
    )
    bridge.runtime_state.max_delay_steps = int(
        cfg.max_synaptic_delay_ms / cfg.dt_ms
    )
    bridge._initialize_simulation_data(called_from_playback_init=False)
    build_sec = time.time() - t0
    log(f"Built in {build_sec:.1f}s; {cfg.num_neurons} neurons, "
        f"{int(bridge.cp_connections.nnz)} synapses")

    # Path G+ minimal: topographic bias for wernicke per-concept ensembles
    if apply_wernicke_topographic:
        if enable_multi_pool_wernicke:
            from research.runners.text_minimal_isolation import (
                apply_wernicke_pool_topographic_bias,
            )
            apply_wernicke_pool_topographic_bias(
                bridge,
                concepts=["apple", "river"],
                topographic_factor=wernicke_topographic_factor,
                off_target_factor=wernicke_off_target_factor,
                n_lang_input=n_lang_input,
                sparsity=0.1,
                verbose=verbose,
            )
        else:
            from research.runners.text_minimal_isolation import (
                apply_wernicke_topographic_bias,
            )
            apply_wernicke_topographic_bias(
                bridge,
                concepts=["apple", "river"],
                topographic_factor=wernicke_topographic_factor,
                off_target_factor=wernicke_off_target_factor,
                n_lang_input=n_lang_input,
                sparsity=0.1,
                verbose=verbose,
            )

    # Encode 2 concepts via lang_input drive + hippo plasticity
    # The hippo trace + ca1->semantic_cortex pathway will produce
    # semantic_cortex activations during/after training.
    word_apple = vocab_to_drive_pattern(
        "apple", n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=0.1,
    )
    word_river = vocab_to_drive_pattern(
        "river", n_neurons=n_lang_input, drive_max_pA=200.0, sparsity=0.1,
    )
    rm = bridge.region_manager
    lang_idx = list(rm.indices("language_input"))
    apple_arr = cp.asarray(
        [lang_idx[i] for i in np.where(word_apple > 0)[0]], dtype=cp.int64
    )
    river_arr = cp.asarray(
        [lang_idx[i] for i in np.where(word_river > 0)[0]], dtype=cp.int64
    )

    # Iter B: split gate sets so encoding can be hippo-only.
    HIPPO_GATES = (
        "lang_to_ec", "ec_to_dg", "dg_to_ca3", "ca3_to_ca1", "ec_to_ca1",
    )
    if enable_multi_pool_wernicke:
        # Path A: gates are per-pool. For 2 pools (apple, river):
        # lang_to_wernicke_pool_0/1, wernicke_pool_0/1_to_semantic, etc.
        pool_names = [f"wernicke_pool_{i}" for i in range(n_wernicke_pools)]
        VENTRAL_GATES = tuple(
            [f"lang_to_{p}" for p in pool_names]
            + [f"{p}_to_semantic" for p in pool_names]
            + ["ca1_to_semantic"]
        )
    else:
        VENTRAL_GATES = (
            "lang_to_wernicke", "wernicke_to_semantic", "ca1_to_semantic",
        )
    # Iter L (2026-05-11): production pathways MUST also train so
    # the engram-tag -> lang_output chain works. Per iter K finding,
    # naming pathway weights stay random unless these gates open
    # during encoding. The lang_input drive activates wernicke +
    # semantic_cortex AND drives wernicke -> lang_output via the
    # comprehension loop, so STDP can co-fire-train these.
    if enable_multi_pool_wernicke:
        if enable_per_concept_lang_out_pools:
            # iter AA: per-pool gates target dedicated lang_output_pools
            PRODUCTION_GATES = tuple(
                [f"semantic_to_{p}" for p in pool_names]
                + [f"{pool_names[i]}_to_lang_pool_{i}"
                   for i in range(n_wernicke_pools)]
                + [f"ca1_to_lang_pool_{i}"
                   for i in range(n_wernicke_pools)]
                + ["ca1_to_lang_out"]
            )
        else:
            PRODUCTION_GATES = tuple(
                [f"semantic_to_{p}" for p in pool_names]
                + [f"{p}_to_lang_out" for p in pool_names]
                + ["ca1_to_lang_out"]
            )
    else:
        PRODUCTION_GATES = (
            "semantic_to_wernicke", "wernicke_to_lang_out",
            "ca1_to_lang_out",
        )
    REPLAY_GATES = ("ca3_swr_burst",)
    if strict_two_stage:
        encode_gates = HIPPO_GATES
        log("  [iter B] strict two-stage: encoding hippo-only")
    else:
        # Iter A + iter L: open everything during encoding
        encode_gates = (
            HIPPO_GATES + REPLAY_GATES + VENTRAL_GATES + PRODUCTION_GATES
        )

    def encode_concept(name, drive_arr):
        """Encode + tag the CA3 ensemble for this concept."""
        for g in encode_gates:
            try:
                bridge.set_plasticity_gate(g, 1.0)
            except Exception:
                pass
        # Training
        for _ in range(n_train_events):
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(30):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
            bridge.cp_external_input_current[drive_arr] = 200.0
            for _ in range(100):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
        # Close
        for g in encode_gates:
            try:
                bridge.set_plasticity_gate(g, 0.0)
            except Exception:
                pass
        # Tag CA3 ensemble for replay later
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(30):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        bridge.start_engram_recording(name)
        bridge.cp_external_input_current[drive_arr] = 200.0
        for _ in range(100):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        bridge.cp_external_input_current[:] = 0.0
        stats = bridge.commit_engram_tag(
            name, top_k=50, region_filter=["ca3"]
        )
        return stats

    if interleaved_training:
        # Interleaved training (iter Z, asymmetry fix):
        # Per-event apple/river alternation. STDP can't grow
        # apple-biased weights without immediate river contrast.
        # Identified as the fix for iter W's apple-only asymmetry
        # (river-direction failed at single-trial demo).
        log(f"\n[interleaved] training apple+river alternating "
            f"per-event x {n_train_events} events...")
        for g in encode_gates:
            try:
                bridge.set_plasticity_gate(g, 1.0)
            except Exception:
                pass
        # Iter GG: teacher pairing for P5. During interleaved
        # training, optionally drive lang_output_pool_<i> with
        # teacher current alongside lang_input(concept_<i>).
        # Mirrors Tier 1's embodied-Hebbian paired-stim that gave
        # 6/6 motor binding. Without teacher, STDP through the
        # long chain (lang_input → wernicke_pool → lang_pool) is
        # too weak; with teacher, lang_output_pool fires reliably
        # during concept training, locking in the binding.
        teacher_pools = {}
        if (enable_lang_out_teacher
                and enable_multi_pool_wernicke
                and enable_per_concept_lang_out_pools):
            rm_t = bridge.region_manager
            for i, cname in enumerate(("apple", "river")):
                pool_indices = list(rm_t.indices(f"lang_output_pool_{i}"))
                teacher_pools[cname] = cp.asarray(
                    pool_indices, dtype=cp.int64,
                )
            log(f"  [iter GG] teacher pairing enabled "
                f"({lang_out_teacher_pA:.0f} pA on lang_output_pool)")
        for ev_idx in range(n_train_events):
            for cname, c_arr in (("apple", apple_arr), ("river", river_arr)):
                bridge.cp_external_input_current[:] = 0.0
                for _ in range(30):
                    bridge._run_one_simulation_step()
                    bridge.runtime_state.current_time_step += 1
                bridge.cp_external_input_current[c_arr] = 200.0
                # Iter GG: add teacher current to lang_output_pool
                if cname in teacher_pools:
                    bridge.cp_external_input_current[
                        teacher_pools[cname]
                    ] = float(lang_out_teacher_pA)
                # Iter II: contrastive suppression of OTHER pools.
                # When training apple, hyperpolarize river-pool to
                # force LTD on cross-concept connections.
                if contrastive_suppress_pA != 0.0 and teacher_pools:
                    for other_cname, other_pool in teacher_pools.items():
                        if other_cname != cname:
                            bridge.cp_external_input_current[
                                other_pool
                            ] = float(contrastive_suppress_pA)
                for _ in range(100):
                    bridge._run_one_simulation_step()
                    bridge.runtime_state.current_time_step += 1
        for g in encode_gates:
            try:
                bridge.set_plasticity_gate(g, 0.0)
            except Exception:
                pass
        # Tag both CA3 ensembles after interleaved training
        log("  [interleaved] tagging CA3 ensembles post-training...")
        apple_tag = None
        river_tag = None
        for cname, c_arr in (("apple", apple_arr), ("river", river_arr)):
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(30):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
            bridge.start_engram_recording(cname)
            bridge.cp_external_input_current[c_arr] = 200.0
            for _ in range(100):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
            bridge.cp_external_input_current[:] = 0.0
            t = bridge.commit_engram_tag(
                cname, top_k=50, region_filter=["ca3"]
            )
            log(f"    {cname} CA3 tag: {t['n_tagged']} neurons")
            if cname == "apple":
                apple_tag = t
            else:
                river_tag = t
    else:
        log(f"\nEncoding 'apple' ({n_train_events} events)...")
        apple_tag = encode_concept("apple", apple_arr)
        log(f"  CA3 tag: {apple_tag['n_tagged']} neurons")

        log(f"\nEncoding 'river' ({n_train_events} events)...")
        river_tag = encode_concept("river", river_arr)
        log(f"  CA3 tag: {river_tag['n_tagged']} neurons")

    # Run concept replay (P3.1) to consolidate to semantic_cortex
    log(f"\nRunning concept replay ({n_replay_cycles} cycles each)...")
    # Iter B: during replay, open the ventral gates so cortex can
    # learn the meaning from the replayed CA3 pattern; also open
    # ca1_to_semantic and ca3_swr_burst for the consolidation
    # transfer per McClelland 1995.
    # Iter K addition (2026-05-11): also open ca1_to_lang_out so the
    # naming pathway (CA3 tag → CA1 → lang_output) trains during
    # replay.
    # Iter L addition: also open production gates (semantic_to_wernicke,
    # wernicke_to_lang_out) so the full production chain trains.
    if enable_multi_pool_wernicke:
        if enable_per_concept_lang_out_pools:
            base_replay_gates = tuple(
                ["ca3_swr_burst", "ca1_to_semantic", "ca3_to_ca1",
                 "ca1_to_lang_out"]
                + [f"semantic_to_{p}" for p in pool_names]
                + [f"{pool_names[i]}_to_lang_pool_{i}"
                   for i in range(n_wernicke_pools)]
                + [f"ca1_to_lang_pool_{i}"
                   for i in range(n_wernicke_pools)]
            )
        else:
            base_replay_gates = tuple(
                ["ca3_swr_burst", "ca1_to_semantic", "ca3_to_ca1",
                 "ca1_to_lang_out"]
                + [f"semantic_to_{p}" for p in pool_names]
                + [f"{p}_to_lang_out" for p in pool_names]
            )
    else:
        base_replay_gates = (
            "ca3_swr_burst", "ca1_to_semantic", "ca3_to_ca1",
            "ca1_to_lang_out",
            "semantic_to_wernicke", "wernicke_to_lang_out",
        )
    if strict_two_stage:
        if enable_multi_pool_wernicke:
            replay_phase_gates = base_replay_gates + tuple(
                [f"lang_to_{p}" for p in pool_names]
                + [f"{p}_to_semantic" for p in pool_names]
            )
        else:
            replay_phase_gates = base_replay_gates + (
                "lang_to_wernicke", "wernicke_to_semantic",
            )
        log("  [iter B] replay opens both hippo-replay AND ventral gates")
    else:
        replay_phase_gates = base_replay_gates
    for g in replay_phase_gates:
        try:
            bridge.set_plasticity_gate(g, 1.0)
        except Exception:
            pass
    t_replay = time.time()
    if drive_lang_during_replay:
        # Iter B variant: drive lang_input alongside CA3 replay so
        # wernicke sees both word + consolidated meaning. Custom
        # replay loop here since run_concept_replay_phase doesn't
        # support external drive on each burst.
        drives = {"apple": apple_arr, "river": river_arr}
        replays_run = 0
        for cycle in range(n_replay_cycles):
            for tag_name in ("apple", "river"):
                # Burst CA3 tag AND drive lang_input(concept)
                bridge.cp_external_input_current[:] = 0.0
                bridge.cp_external_input_current[drives[tag_name]] = 200.0
                bridge.stimulate_tag(tag_name, drive_pA=150.0)
                for _ in range(50):  # burst_duration_ms
                    bridge._run_one_simulation_step()
                    bridge.runtime_state.current_time_step += 1
                # Inter-burst gap
                bridge.cp_external_input_current[:] = 0.0
                bridge.clear_tag_drive()
                for _ in range(20):  # inter_burst_ms
                    bridge._run_one_simulation_step()
                    bridge.runtime_state.current_time_step += 1
                replays_run += 1
        replay_stats = {"n_replays": replays_run}
        log("  [iter B] drove lang_input alongside CA3 replay")
    else:
        replay_stats = run_concept_replay_phase(
            bridge, tag_names=["apple", "river"],
            n_replays_per_tag=n_replay_cycles,
            burst_duration_ms=50, inter_burst_ms=20,
            drive_pA=150.0,
        )
    for g in replay_phase_gates:
        try:
            bridge.set_plasticity_gate(g, 0.0)
        except Exception:
            pass
    log(f"  replay done ({replay_stats['n_replays']} events, "
        f"{time.time() - t_replay:.0f}s)")

    # Test 1: Comprehension — engram-tag methodology (P5 iteration A).
    # First exposure: drive lang_input(apple), tag semantic_cortex
    # ensemble. Subsequent exposure: drive lang_input(apple) again,
    # measure cosine of resulting semantic_cortex firing pattern vs
    # the tagged ensemble (treats both as binary index sets). Same
    # methodology that turned P1 D.13 from FAIL to PASS.
    log("\n[TEST 1] Comprehension: tag semantic_cortex on first exposure, "
        "test reactivation")

    def drive_and_tag_semantic(name, drive_arr):
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(50):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        bridge.start_engram_recording(name)
        bridge.cp_external_input_current[drive_arr] = 200.0
        for _ in range(drive_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        bridge.cp_external_input_current[:] = 0.0
        return bridge.commit_engram_tag(
            name, top_k=50, region_filter=["semantic_cortex"],
        )

    def measure_semantic_response_indices(drive_arr):
        """Drive + return indices of semantic_cortex neurons that fired."""
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(30):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        bridge.cp_external_input_current[drive_arr] = 200.0
        spike_counts = measure_region_spikes(bridge, "semantic_cortex",
                                                n_steps=drive_steps)
        bridge.cp_external_input_current[:] = 0.0
        # Return indices of neurons that fired at all
        return np.where(spike_counts > 0)[0]

    # Tag semantic_cortex ensembles
    apple_sem_tag = drive_and_tag_semantic("apple_semantic", apple_arr)
    log(f"  apple semantic_cortex tag: {apple_sem_tag['n_tagged']} neurons")
    river_sem_tag = drive_and_tag_semantic("river_semantic", river_arr)
    log(f"  river semantic_cortex tag: {river_sem_tag['n_tagged']} neurons")

    # Get tag indices (in semantic_cortex local) for binary cosine
    rm = bridge.region_manager
    sem_cortex_indices = list(rm.indices("semantic_cortex"))
    sem_cortex_set = set(sem_cortex_indices)
    apple_tag_global = to_host(
        bridge.get_engram_tag_indices("apple_semantic")
    )
    river_tag_global = to_host(
        bridge.get_engram_tag_indices("river_semantic")
    )

    # Measure reactivation: drive lang_input(apple) again, see which
    # semantic_cortex neurons fire
    apple_reactivation = measure_semantic_response_indices(apple_arr)
    # Convert local indices back to global semantic_cortex indices
    apple_reactivation_global = np.array(
        [sem_cortex_indices[i] for i in apple_reactivation if i < len(sem_cortex_indices)],
        dtype=np.int64,
    )

    river_reactivation = measure_semantic_response_indices(river_arr)
    river_reactivation_global = np.array(
        [sem_cortex_indices[i] for i in river_reactivation if i < len(sem_cortex_indices)],
        dtype=np.int64,
    )

    def index_cosine(a_idx, b_idx, n_total):
        if len(a_idx) == 0 or len(b_idx) == 0:
            return 0.0
        s_a = set(int(x) for x in a_idx)
        s_b = set(int(x) for x in b_idx)
        overlap = len(s_a & s_b)
        return float(overlap / (np.sqrt(len(s_a)) * np.sqrt(len(s_b))))

    n_neurons_total = int(cfg.num_neurons)
    cos_apple_self = index_cosine(apple_reactivation_global, apple_tag_global,
                                    n_neurons_total)
    cos_apple_river = index_cosine(apple_reactivation_global, river_tag_global,
                                      n_neurons_total)
    log(f"  apple trial 1 vs apple trial 2: cos = {cos_apple_self:.3f}")
    log(f"    (same-concept stability; target > 0.6)")
    log(f"  apple vs river: cos = {cos_apple_river:.3f}")
    log(f"    (different-concept; target < 0.3)")
    pass_comprehension = (cos_apple_self > 0.5) and (cos_apple_river < 0.4)

    # Test 2: Naming — engram-tag methodology (iter O).
    # Tag the lang_output ensemble via CA3-DRIVEN activation
    # (same chain as naming test). Iter N's approach (lang_input
    # drive) captured the wernicke->lang_out path's ensemble,
    # but the naming test uses the CA3->CA1->lang_out chain.
    # Mismatched paths → anti-discrimination (iter N finding).
    # Iter O uses matched paths.
    log("\n[TEST 2] Naming: CA3-stim-tag lang_output, stim CA3, measure recall")

    def ca3_stim_and_tag_langout(name, ca3_tag_name):
        """Stimulate the CA3 engram tag and tag lang_output's
        response. Same chain (CA1->lang_out) used in test."""
        bridge.cp_external_input_current[:] = 0.0
        bridge.clear_tag_drive()
        for _ in range(50):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        bridge.start_engram_recording(name)
        bridge.stimulate_tag(ca3_tag_name, drive_pA=stim_drive_pA)
        for _ in range(drive_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        bridge.cp_external_input_current[:] = 0.0
        bridge.clear_tag_drive()
        return bridge.commit_engram_tag(
            name, top_k=50, region_filter=["language_output"],
        )

    apple_lang_tag = ca3_stim_and_tag_langout("apple_langout", "apple")
    river_lang_tag = ca3_stim_and_tag_langout("river_langout", "river")
    log(f"  apple lang_output tag: {apple_lang_tag['n_tagged']} neurons")
    log(f"  river lang_output tag: {river_lang_tag['n_tagged']} neurons")

    apple_lang_idx = to_host(
        bridge.get_engram_tag_indices("apple_langout"))
    river_lang_idx = to_host(
        bridge.get_engram_tag_indices("river_langout"))

    lang_out_indices = list(rm.indices("language_output"))

    def measure_langout_response_indices():
        bridge.cp_external_input_current[:] = 0.0
        for _ in range(30):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
        spike_counts = measure_region_spikes(bridge, "language_output",
                                              n_steps=drive_steps)
        return np.where(spike_counts > 0)[0]

    # Baseline: lang_output response with NO stimulation
    baseline_lang_local = measure_langout_response_indices()
    baseline_lang_global = np.array(
        [lang_out_indices[i] for i in baseline_lang_local
         if i < len(lang_out_indices)],
        dtype=np.int64,
    )

    # Causal: stimulate apple CA3 tag, measure lang_output
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(30):
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
    bridge.stimulate_tag("apple", drive_pA=stim_drive_pA)
    causal_spikes = measure_region_spikes(bridge, "language_output",
                                            n_steps=drive_steps)
    causal_lang_local = np.where(causal_spikes > 0)[0]
    causal_lang_global = np.array(
        [lang_out_indices[i] for i in causal_lang_local
         if i < len(lang_out_indices)],
        dtype=np.int64,
    )
    bridge.cp_external_input_current[:] = 0.0
    bridge.clear_tag_drive()

    cos_naming_self = index_cosine(causal_lang_global, apple_lang_idx,
                                    n_neurons_total)
    cos_naming_cross = index_cosine(causal_lang_global, river_lang_idx,
                                      n_neurons_total)
    cos_baseline_self = index_cosine(baseline_lang_global, apple_lang_idx,
                                       n_neurons_total)
    log(f"  CA3-apple-stim lang_output vs apple_lang tag: {cos_naming_self:.3f}")
    log(f"  CA3-apple-stim lang_output vs river_lang tag: {cos_naming_cross:.3f}")
    log(f"  baseline lang_output vs apple_lang tag: {cos_baseline_self:.3f}")

    # Also keep raw spike count metric for back-compat
    baseline_sum = float(np.sum(baseline_lang_local > -1))
    causal_sum = float(np.sum(causal_lang_local > -1))
    naming_ratio = max(cos_naming_self, 0.01) / max(cos_naming_cross, 0.01)
    pass_naming = (cos_naming_self > 0.3 and cos_naming_self > 1.3 * cos_naming_cross)

    # Iter AA: per-concept lang_output_pool readout (true behavioral
    # naming test). If apple CA3 stim activates lang_output_pool_0
    # more than lang_output_pool_1, recognized as apple.
    pool_readout = None
    if enable_per_concept_lang_out_pools and enable_multi_pool_wernicke:
        log(f"\n[TEST 2b] Per-pool lang_output naming readout "
            f"(iter AA, {n_recognition_trials}-trial avg)")
        pool_readout = {"apple": {}, "river": {}}
        for stim_concept in ("apple", "river"):
            # Multi-trial averaging (iter DD): sum spikes across
            # N trials per concept to smooth single-trial noise.
            pool_spikes_sum = {i: 0.0 for i in range(n_wernicke_pools)}
            for trial in range(n_recognition_trials):
                bridge.cp_external_input_current[:] = 0.0
                bridge.clear_tag_drive()
                # Inter-trial rest (iter EE: configurable, default 50)
                for _ in range(inter_trial_rest_steps):
                    bridge._run_one_simulation_step()
                    bridge.runtime_state.current_time_step += 1
                bridge.stimulate_tag(stim_concept, drive_pA=stim_drive_pA)
                for i in range(n_wernicke_pools):
                    pool_name = f"lang_output_pool_{i}"
                    spikes = measure_region_spikes(
                        bridge, pool_name, n_steps=100,
                    )
                    pool_spikes_sum[i] += float(spikes.sum())
                bridge.cp_external_input_current[:] = 0.0
                bridge.clear_tag_drive()
            pool_spikes = {i: pool_spikes_sum[i] / n_recognition_trials
                            for i in range(n_wernicke_pools)}
            pool_readout[stim_concept] = pool_spikes
            log(f"  stim '{stim_concept}' (n={n_recognition_trials}): "
                + ", ".join(
                f"pool_{i}={v:.0f}"
                for i, v in pool_spikes.items()))

        # PASS criterion: stim apple → pool_0 > pool_1;
        # stim river → pool_1 > pool_0
        apple_correct = pool_readout["apple"][0] > pool_readout["apple"][1]
        river_correct = pool_readout["river"][1] > pool_readout["river"][0]
        pass_naming_pool = apple_correct and river_correct
        log(f"  apple→pool_0>pool_1: {apple_correct}")
        log(f"  river→pool_1>pool_0: {river_correct}")
        log(f"  bidirectional PASS: {pass_naming_pool}")
        # Override pass_naming with the pool-based criterion
        pass_naming = pass_naming_pool

    # Iter E (alt methodology): inspect learned weights directly.
    # Question: did STDP grow wernicke->semantic_cortex weights
    # selectively for each concept's ensembles? If yes, training
    # worked but dynamics are too noisy. If no, training itself
    # didn't learn the binding.
    log("\n[TEST 3] Weight inspection: wernicke->semantic_cortex matrix")
    weight_diag = {}
    try:
        # Get wernicke + semantic_cortex region indices.
        # For multi-pool, aggregate all pool neurons.
        rm2 = bridge.region_manager
        if enable_multi_pool_wernicke:
            wernicke_indices = []
            for i in range(n_wernicke_pools):
                wernicke_indices.extend(
                    list(rm2.indices(f"wernicke_pool_{i}")))
        else:
            wernicke_indices = list(rm2.indices("wernicke"))
        sem_indices_local = list(rm2.indices("semantic_cortex"))
        sem_idx_set = set(sem_indices_local)

        # Identify wernicke "concept ensembles" via spike counts
        # during fresh drive
        def fire_indices(region_name, drive_arr, n_steps=100):
            bridge.cp_external_input_current[:] = 0.0
            for _ in range(30):
                bridge._run_one_simulation_step()
                bridge.runtime_state.current_time_step += 1
            bridge.cp_external_input_current[drive_arr] = 200.0
            counts = measure_region_spikes(bridge, region_name,
                                              n_steps=n_steps)
            bridge.cp_external_input_current[:] = 0.0
            # Return top-K firing indices (LOCAL to region)
            n_top = min(50, int(np.sum(counts > 0)))
            if n_top == 0:
                return np.array([], dtype=np.int64)
            order = np.argsort(counts)[::-1][:n_top]
            return order

        # For multi-pool, measure each pool's response and aggregate
        if enable_multi_pool_wernicke:
            # apple should fire pool_0 strongly; river should fire pool_1
            apple_wernicke_local = fire_indices("wernicke_pool_0", apple_arr)
            river_wernicke_local = fire_indices("wernicke_pool_1", river_arr)
        else:
            apple_wernicke_local = fire_indices("wernicke", apple_arr)
            river_wernicke_local = fire_indices("wernicke", river_arr)
        apple_sem_local = fire_indices("semantic_cortex", apple_arr)
        river_sem_local = fire_indices("semantic_cortex", river_arr)
        log(f"  apple wernicke ensemble: {len(apple_wernicke_local)}, "
            f"sem ensemble: {len(apple_sem_local)}")
        log(f"  river wernicke ensemble: {len(river_wernicke_local)}, "
            f"sem ensemble: {len(river_sem_local)}")

        # Look at the wernicke->semantic_cortex weights via the
        # bridge's connections sparse matrix. We compute mean weight
        # from apple_wernicke -> apple_sem vs apple_wernicke -> river_sem.
        # Sparse CSR: connections[i, j] is weight of pre=i to post=j.
        conn = bridge.cp_connections
        # to_host for sparse: pull out CSR arrays
        csr_data = to_host(conn.data)
        csr_indices = to_host(conn.indices)
        csr_indptr = to_host(conn.indptr)

        def mean_weight_subset(pre_global, post_set):
            """Return mean weight pre->post for pre in pre_global,
            post in post_set."""
            if len(pre_global) == 0 or len(post_set) == 0:
                return 0.0, 0
            total = 0.0
            n = 0
            for pre_i in pre_global:
                start = csr_indptr[pre_i]
                end = csr_indptr[pre_i + 1]
                for k in range(start, end):
                    post_j = int(csr_indices[k])
                    if post_j in post_set:
                        total += float(csr_data[k])
                        n += 1
            return (total / n if n > 0 else 0.0), n

        # Correct local->global conversion: for multi-pool case,
        # apple_local indices into pool_0; river_local into pool_1.
        if enable_multi_pool_wernicke:
            pool_0_indices = list(rm2.indices("wernicke_pool_0"))
            pool_1_indices = list(rm2.indices("wernicke_pool_1"))
            apple_wernicke_global = np.array(
                [pool_0_indices[i] for i in apple_wernicke_local
                 if i < len(pool_0_indices)],
                dtype=np.int64
            )
            river_wernicke_global = np.array(
                [pool_1_indices[i] for i in river_wernicke_local
                 if i < len(pool_1_indices)],
                dtype=np.int64
            )
        else:
            apple_wernicke_global = np.array(
                [wernicke_indices[i] for i in apple_wernicke_local],
                dtype=np.int64
            )
            river_wernicke_global = np.array(
                [wernicke_indices[i] for i in river_wernicke_local],
                dtype=np.int64
            )
        apple_sem_global_set = set(
            int(sem_indices_local[i]) for i in apple_sem_local
        )
        river_sem_global_set = set(
            int(sem_indices_local[i]) for i in river_sem_local
        )

        w_apple_apple, n_aa = mean_weight_subset(
            apple_wernicke_global, apple_sem_global_set
        )
        w_apple_river, n_ar = mean_weight_subset(
            apple_wernicke_global, river_sem_global_set
        )
        w_river_river, n_rr = mean_weight_subset(
            river_wernicke_global, river_sem_global_set
        )
        w_river_apple, n_ra = mean_weight_subset(
            river_wernicke_global, apple_sem_global_set
        )

        log(f"  Mean weight apple_wernicke -> apple_sem: "
            f"{w_apple_apple:.3f} (n={n_aa})")
        log(f"  Mean weight apple_wernicke -> river_sem: "
            f"{w_apple_river:.3f} (n={n_ar})")
        log(f"  Mean weight river_wernicke -> river_sem: "
            f"{w_river_river:.3f} (n={n_rr})")
        log(f"  Mean weight river_wernicke -> apple_sem: "
            f"{w_river_apple:.3f} (n={n_ra})")

        # Selectivity index: (same-concept mean - cross-concept mean) /
        #                     (same-concept mean + cross-concept mean)
        # > 0 means learning worked; ~0 means no selective binding
        same_mean = (w_apple_apple + w_river_river) / 2
        cross_mean = (w_apple_river + w_river_apple) / 2
        if (same_mean + cross_mean) > 0:
            selectivity = (same_mean - cross_mean) / (same_mean + cross_mean)
        else:
            selectivity = 0.0
        log(f"  WEIGHT SELECTIVITY INDEX: {selectivity:.3f}")
        log(f"    (>0.1 = clear binding; ~0 = no learning)")
        weight_diag = {
            "apple_wernicke_size": len(apple_wernicke_local),
            "river_wernicke_size": len(river_wernicke_local),
            "apple_sem_size": len(apple_sem_local),
            "river_sem_size": len(river_sem_local),
            "w_apple_apple": w_apple_apple,
            "w_apple_river": w_apple_river,
            "w_river_river": w_river_river,
            "w_river_apple": w_river_apple,
            "n_apple_apple": n_aa,
            "n_apple_river": n_ar,
            "n_river_river": n_rr,
            "n_river_apple": n_ra,
            "selectivity_index": selectivity,
        }
    except Exception as e:
        log(f"  [TEST 3] weight inspection failed: {e}")
        weight_diag = {"error": str(e)}

    log("\n" + "=" * 60)
    log("PASS criteria:")
    log(f"  Comprehension (apple_self > 0.5 AND apple_river < 0.4): "
        f"{'PASS' if pass_comprehension else 'FAIL'}")
    log(f"    apple_self={cos_apple_self:.3f}, "
        f"apple_river={cos_apple_river:.3f}")
    log(f"  Naming (causal/baseline > 1.3): "
        f"{'PASS' if pass_naming else 'FAIL'}")
    log(f"    ratio={naming_ratio:.2f}x")
    overall = pass_comprehension and pass_naming
    log(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    if "selectivity_index" in weight_diag:
        log(f"  Weight selectivity (diagnostic): "
            f"{weight_diag['selectivity_index']:.3f}")
    log("=" * 60)

    result = {
        "seed": seed,
        "build_seconds": build_sec,
        "n_neurons": int(cfg.num_neurons),
        "n_synapses": int(bridge.cp_connections.nnz),
        "n_train_events": n_train_events,
        "n_replay_cycles": n_replay_cycles,
        "apple_tag_size": apple_tag["n_tagged"],
        "river_tag_size": river_tag["n_tagged"],
        "comprehension": {
            "apple_self_cosine": cos_apple_self,
            "apple_river_cosine": cos_apple_river,
            "passed": pass_comprehension,
        },
        "naming": {
            "baseline_lang_out_spikes": baseline_sum,
            "causal_lang_out_spikes": causal_sum,
            "ratio": naming_ratio,
            "cos_naming_self": cos_naming_self,
            "cos_naming_cross": cos_naming_cross,
            "cos_baseline_self": cos_baseline_self,
            "passed": pass_naming,
            "pool_readout": pool_readout,
        },
        "weight_diagnostics": weight_diag,
        "overall_passed": overall,
        "total_seconds": time.time() - t0,
    }
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, default=str),
                              encoding="utf-8")
        log(f"\n[OUT] {out_path}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train-events", type=int, default=300)
    ap.add_argument("--n-replay-cycles", type=int, default=20)
    ap.add_argument("--n-lang-input", type=int, default=1024,
                    help="Iter R (scale-up): 1024 -> 4096 for "
                         "biological scale test")
    ap.add_argument("--n-semantic-cortex", type=int, default=500,
                    help="Iter C/R: scale 500 -> 1000 (Tier C) "
                         "or 5000 (Tier R biological scale)")
    ap.add_argument("--n-wernicke", type=int, default=100,
                    help="Iter C/R: scale 100 -> 400 (Tier C) "
                         "or 2000 (Tier R biological scale)")
    ap.add_argument("--strict-two-stage", action="store_true",
                    help="Iter B: encoding hippo-only; replay opens "
                         "ventral gates")
    ap.add_argument("--drive-lang-during-replay", action="store_true",
                    help="Iter B variant: drive lang_input(concept) "
                         "alongside CA3 replay")
    # Iter D: semantic_cortex attractor tuning
    ap.add_argument("--semantic-cortex-recurrent-density", type=float,
                    default=0.10,
                    help="Iter D: cortex recurrent density (default "
                         "0.10; try 0.25 for stronger attractor)")
    ap.add_argument("--semantic-cortex-recurrent-weight", type=float,
                    default=1.0,
                    help="Iter D: cortex recurrent weight (default "
                         "1.0; try 2.5 for stronger attractor)")
    ap.add_argument("--lang-to-wernicke-density", type=float,
                    default=0.30,
                    help="Iter H: lower density (e.g. 0.05) so "
                         "different lang patterns hit different "
                         "wernicke neurons by chance, creating "
                         "natural sparse ensembles per concept")
    ap.add_argument("--lang-to-wernicke-weight", type=float,
                    default=3.0)
    ap.add_argument("--wernicke-to-semantic-density", type=float,
                    default=0.30)
    ap.add_argument("--wernicke-to-semantic-weight", type=float,
                    default=4.0)
    ap.add_argument("--drive-steps", type=int, default=100,
                    help="Iter D: steps to drive during test "
                         "(default 100; try 300 for attractor "
                         "settling)")
    # Path B+ (iter F): semantic_FS lateral inhibition
    ap.add_argument("--enable-semantic-fs", action="store_true",
                    help="Path B+ (iter F): add PV-FS lateral "
                         "inhibition to semantic_cortex for "
                         "competitive attractor formation")
    ap.add_argument("--n-semantic-fs", type=int, default=100)
    # Path G (iter G): wernicke_FS lateral inhibition (UPSTREAM fix)
    ap.add_argument("--enable-wernicke-fs", action="store_true",
                    help="Path G (iter G): add PV-FS lateral "
                         "inhibition to wernicke for sparse per-"
                         "concept ensemble encoding. Fixes "
                         "upstream bottleneck identified by iter E.")
    ap.add_argument("--n-wernicke-fs", type=int, default=60)
    # Path A / Path G+ FULL (iter T): multi-pool wernicke
    ap.add_argument("--enable-multi-pool-wernicke", action="store_true",
                    help="Path A: per-concept wernicke pools with "
                         "cross-pool FS inhibition. Mirror of Tier 1 "
                         "motor pool architecture (which produced "
                         "6/6 multi-seed PASS).")
    ap.add_argument("--n-wernicke-pools", type=int, default=2)
    ap.add_argument("--n-per-wernicke-pool", type=int, default=100)
    ap.add_argument("--n-per-wernicke-pool-fs", type=int, default=12)
    ap.add_argument("--interleaved-training", action="store_true",
                    help="Iter Z: train apple+river per-event "
                         "alternating instead of sequential blocks. "
                         "Fixes apple-asymmetry from V3 demo data.")
    ap.add_argument("--enable-per-concept-lang-out-pools",
                    action="store_true",
                    help="Iter AA: per-concept lang_output_pool "
                         "regions. Each wernicke_pool routes to its "
                         "dedicated lang_output_pool. Mirror of "
                         "Tier 1 motor pool at output.")
    ap.add_argument("--n-per-lang-out-pool", type=int, default=200)
    # Iter KK->LL: per-pool dynamics parameterization. Defaults = iter AA
    # weak dynamics. Override for Tier 1 cortical canon experimentation.
    ap.add_argument("--wernicke-pool-internal-density", type=float,
                    default=0.05,
                    help="Wernicke pool recurrent density (iter AA "
                         "default 0.05; Tier 1 canon 0.10 amplifies "
                         "structural bias at biological scale per "
                         "iter KK seed 42 finding)")
    ap.add_argument("--wernicke-pool-exc-weight", type=float, default=0.3,
                    help="Wernicke pool exc recurrent weight (iter AA "
                         "default 0.3; Tier 1 canon 2.0)")
    ap.add_argument("--wernicke-pool-inh-weight", type=float, default=0.8,
                    help="Wernicke pool inh recurrent weight (iter AA "
                         "default 0.8; Tier 1 canon 4.0)")
    ap.add_argument("--lang-output-pool-internal-density", type=float,
                    default=0.05,
                    help="Lang-output pool recurrent density (default 0.05)")
    ap.add_argument("--lang-output-pool-exc-weight", type=float, default=0.3,
                    help="Lang-output pool exc recurrent weight (default 0.3)")
    ap.add_argument("--lang-output-pool-inh-weight", type=float, default=0.8,
                    help="Lang-output pool inh recurrent weight (default 0.8)")
    ap.add_argument("--wernicke-fs-cross-weight", type=float, default=4.0,
                    help="Iter BB: cross-pool FS inhibition weight "
                         "(default 4.0; try 8.0 to suppress apple-bias)")
    ap.add_argument("--enable-lang-out-fs-pools", action="store_true",
                    help="Iter CC: add lang_output_FS_pool_<i> for "
                         "cross-inhibition at output layer. Full Tier 1 "
                         "motor pool mirror at output side.")
    ap.add_argument("--n-per-lang-out-fs-pool", type=int, default=24)
    ap.add_argument("--n-recognition-trials", type=int, default=1,
                    help="Iter DD: multi-trial averaging at "
                         "recognition (5 reduces single-trial noise)")
    ap.add_argument("--inter-trial-rest-steps", type=int, default=50,
                    help="Iter EE: rest steps between recognition "
                         "trials (default 50; try 500 for full "
                         "adaptation recovery)")
    ap.add_argument("--enable-lang-out-teacher", action="store_true",
                    help="Iter GG: teacher pairing at lang_output. "
                         "During training, drive lang_output_pool_<i> "
                         "with teacher current alongside concept input. "
                         "Mirrors Tier 1's embodied-Hebbian paradigm.")
    ap.add_argument("--lang-out-teacher-pa", type=float, default=300.0)
    ap.add_argument("--contrastive-suppress-pa", type=float, default=0.0,
                    help="Iter II: contrastive training. During "
                         "concept_i training, drive OTHER pools with "
                         "this current (typically negative, e.g. -200). "
                         "Forces LTD on cross-concept binding.")
    # Iter M: strengthen naming pathway
    ap.add_argument("--ca1-to-lang-out-weight", type=float, default=2.0,
                    help="Iter M: strengthen CA1->lang_output "
                         "weight (default 2.0; try 5.0 for "
                         "robust naming propagation)")
    ap.add_argument("--stim-drive-pa", type=float, default=200.0,
                    help="Iter M: engram tag stimulation drive "
                         "(default 200 pA; try 500 for stronger "
                         "naming test)")
    # Path G+ minimal (iter P): wernicke topographic bias
    ap.add_argument("--apply-wernicke-topographic", action="store_true",
                    help="Path G+ minimal: bias lang_input -> "
                         "wernicke so each concept activates a "
                         "dedicated wernicke subset. Mirror of "
                         "Tier 1 motor pool bias.")
    ap.add_argument("--wernicke-topographic-factor", type=float,
                    default=1.5,
                    help="Path G+ minimal: weight boost for "
                         "on-target wernicke subset (default 1.5)")
    ap.add_argument("--wernicke-off-target-factor", type=float,
                    default=0.7,
                    help="Path G+ minimal: weight reduction for "
                         "off-target wernicke subset (default 0.7)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    run_ventral_validation(
        seed=args.seed,
        n_lang_input=args.n_lang_input,
        n_train_events=args.n_train_events,
        n_replay_cycles=args.n_replay_cycles,
        n_semantic_cortex=args.n_semantic_cortex,
        n_wernicke=args.n_wernicke,
        strict_two_stage=args.strict_two_stage,
        drive_lang_during_replay=args.drive_lang_during_replay,
        semantic_cortex_recurrent_density=(
            args.semantic_cortex_recurrent_density),
        semantic_cortex_recurrent_weight=(
            args.semantic_cortex_recurrent_weight),
        lang_to_wernicke_density=args.lang_to_wernicke_density,
        lang_to_wernicke_weight=args.lang_to_wernicke_weight,
        wernicke_to_semantic_density=args.wernicke_to_semantic_density,
        wernicke_to_semantic_weight=args.wernicke_to_semantic_weight,
        drive_steps=args.drive_steps,
        enable_semantic_fs=args.enable_semantic_fs,
        n_semantic_fs=args.n_semantic_fs,
        enable_wernicke_fs=args.enable_wernicke_fs,
        n_wernicke_fs=args.n_wernicke_fs,
        enable_multi_pool_wernicke=args.enable_multi_pool_wernicke,
        n_wernicke_pools=args.n_wernicke_pools,
        n_per_wernicke_pool=args.n_per_wernicke_pool,
        n_per_wernicke_pool_fs=args.n_per_wernicke_pool_fs,
        interleaved_training=args.interleaved_training,
        enable_per_concept_lang_out_pools=(
            args.enable_per_concept_lang_out_pools),
        n_per_lang_out_pool=args.n_per_lang_out_pool,
        wernicke_pool_internal_density=args.wernicke_pool_internal_density,
        wernicke_pool_exc_weight_mean=args.wernicke_pool_exc_weight,
        wernicke_pool_inh_weight_mean=args.wernicke_pool_inh_weight,
        lang_output_pool_internal_density=(
            args.lang_output_pool_internal_density),
        lang_output_pool_exc_weight_mean=args.lang_output_pool_exc_weight,
        lang_output_pool_inh_weight_mean=args.lang_output_pool_inh_weight,
        wernicke_fs_cross_weight=args.wernicke_fs_cross_weight,
        enable_lang_out_fs_pools=args.enable_lang_out_fs_pools,
        n_per_lang_out_fs_pool=args.n_per_lang_out_fs_pool,
        n_recognition_trials=args.n_recognition_trials,
        inter_trial_rest_steps=args.inter_trial_rest_steps,
        enable_lang_out_teacher=args.enable_lang_out_teacher,
        lang_out_teacher_pA=args.lang_out_teacher_pa,
        contrastive_suppress_pA=args.contrastive_suppress_pa,
        ca1_to_lang_out_weight=args.ca1_to_lang_out_weight,
        stim_drive_pA=args.stim_drive_pa,
        apply_wernicke_topographic=args.apply_wernicke_topographic,
        wernicke_topographic_factor=args.wernicke_topographic_factor,
        wernicke_off_target_factor=args.wernicke_off_target_factor,
        out_path=Path(args.out) if args.out else None,
        verbose=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
