"""
Minimal language->motor isolation experiment.

Tests the decisive question: can the architecture learn word-action
mapping AT ALL when stripped of cascade interference?

Prior data (2026-05-03 autonomous overnight) shows 0/39 aligned across
all v2-architecture conditions. Pattern analysis shows misalignment is
seed-dependent (each random init creates its own private misalignment),
with a mild cascade-driven motor_E bias of ~3pp.

If THIS minimal architecture (NO cascade, NO PFC, NO retina, NO
visuomotor — just language_input -> motor_X with paired-stim training)
achieves aligned >= 4/6, the cascade IS the dominant interference.

If THIS also fails, the fundamental issue is deeper (plasticity dose,
soft-bound STDP, sparse-code overlap, or eval methodology).

Architecture:
  - language_input: 256 neurons (same as v2 baseline for fair compare)
  - motor_N, motor_E, motor_S, motor_W: 25 each (slightly larger than
    v2's 10 to reduce SNR noise; doesn't affect alignment if test
    works)
  - language_input -> motor_X pathways (4 plastic, all4 actions)
  - NO cluster_a, NO cluster_e, NO cortex_X cascade
  - NO retina, NO visual cortex, NO PFC
  - NO visuomotor pathways

Training:
  - paired-stim only (same _run_swr_replay_phase mechanism as H4)
  - synthetic balanced buffer: N events per direction, +1 reward
  - directly tests STDP's ability to differentiate words on a clean
    pathway

Eval:
  - same evaluate_word_to_action that everything else uses
  - 25 trials per word, interleaved
  - aligned ratio is the headline metric

Usage:
    python -m research.runners.text_minimal_isolation \\
        --seed 42 --n-events-per-direction 1000 \\
        --out-stats research/findings/raw/g11_bg/text_eval_minimal_iso_seed42.json
"""

import argparse
import json
import time
import numpy as np


def build_minimal_brain_regions(
    n_lang_input: int = 256,
    n_motor_per_action: int = 25,
    text_input_to_motor_density: float = 0.30,
    text_input_to_motor_weight: float = 3.0,
    text_input_to_motor_jitter: float = 0.5,
    enable_motor_fs: bool = False,
    n_motor_fs_per_action: int = 3,
    motor_to_fs_weight: float = 2.0,
    fs_to_motor_weight: float = 2.0,
):
    """Build a minimal language->motor architecture for isolation tests.

    Returns (regions, pathways) tuple compatible with the brain-region
    framework.

    Args:
        enable_motor_fs: add motor_FS_X interneuron pools providing
            cross-pool lateral inhibition (PV-FS in real motor cortex).
            Each motor_X drives its own motor_FS_X (excitatory), which
            inhibits the OTHER 3 motor pools (no self-inhibition). This
            is biology-grounded: real PV-FS interneurons provide
            ~10-15% of cortical population, mediating winner-takes-most
            competition without absolute veto. See Vogels et al 2011,
            Hofer et al 2011.
        n_motor_fs_per_action: FS interneurons per pool (default 3 ~12%
            of 25-neuron motor pool; biology range 10-15%).
        motor_to_fs_weight: excitatory drive from motor pyramidal to FS.
        fs_to_motor_weight: inhibitory weight from FS to other motor
            pools. Equal to motor_to_fs by default (graded competition,
            not absolute WTA).
    """
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronType

    ACTION_NAMES = ["N", "E", "S", "W"]

    regions = []
    pathways = []

    # Language input region (sparse code substrate)
    regions.append(BrainRegion(
        name="language_input",
        n_neurons=n_lang_input,
        exc_fraction=0.8,
        internal_density=0.05,
        exc_weight_mean=2.0, inh_weight_mean=4.0,
        weight_jitter=0.2, plastic_internal=True,
        izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
    ))

    # Motor pools — separate region per action
    for action in ACTION_NAMES:
        regions.append(BrainRegion(
            name=f"motor_{action}",
            n_neurons=n_motor_per_action,
            exc_fraction=1.0,  # purely excitatory motor pool
            internal_density=0.0,  # no internal recurrence
            exc_weight_mean=0.0, inh_weight_mean=0.0,
            weight_jitter=0.0, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))

    # language_input -> motor_X pathways (the ONE pathway being tested)
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region="language_input", to_region=f"motor_{action}",
            density=text_input_to_motor_density,
            weight_mean=text_input_to_motor_weight,
            weight_jitter=text_input_to_motor_jitter,
            plastic=True,
            plasticity_gate="language_input_to_motor",
        ))

    # Motor lateral inhibition via PV-FS interneurons (biology-grounded)
    if enable_motor_fs:
        for action in ACTION_NAMES:
            regions.append(BrainRegion(
                name=f"motor_FS_{action}",
                n_neurons=n_motor_fs_per_action,
                exc_fraction=0.0,  # purely inhibitory
                internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
            ))
        for action in ACTION_NAMES:
            # motor_X excites its own FS pool (not language_input -> FS:
            # FS recruitment must come from motor activity itself, not
            # language input directly, to prevent "language drive
            # directly suppresses wrong motor pools" shortcut)
            pathways.append(RegionPathway(
                from_region=f"motor_{action}", to_region=f"motor_FS_{action}",
                density=0.5,
                weight_mean=motor_to_fs_weight,
                weight_jitter=0.3,
                plastic=False,  # static recruitment (genetic-spec, not learned)
            ))
            # motor_FS_X inhibits the OTHER motor pools (no self-inhibition)
            for target_action in ACTION_NAMES:
                if target_action == action:
                    continue
                pathways.append(RegionPathway(
                    from_region=f"motor_FS_{action}",
                    to_region=f"motor_{target_action}",
                    density=0.5,
                    weight_mean=fs_to_motor_weight,
                    weight_jitter=0.3,
                    plastic=False,  # static inhibitory specification
                ))

    return regions, pathways


def build_biological_brain_regions(
    n_lang_input: int = 2048,
    n_motor_per_action: int = 500,
    text_input_to_motor_density: float = 0.30,
    text_input_to_motor_weight: float = 3.0,
    text_input_to_motor_jitter: float = 0.5,
    enable_motor_fs: bool = False,
    n_motor_fs_per_action: int = 60,
    motor_to_fs_weight: float = 2.0,
    fs_to_motor_weight: float = 2.0,
    motor_internal_density: float = 0.10,
    motor_exc_fraction: float = 0.8,
    motor_exc_weight_mean: float = 2.0,
    motor_inh_weight_mean: float = 4.0,
    lang_internal_density: float = 0.05,
    lang_exc_fraction: float = 0.8,
):
    """Biological-scale architecture with cortical canon ENABLED.

    vs build_minimal_brain_regions: motor pools have recurrent
    excitation + E/I balance + larger N. Specifically:
      - n_motor_per_action: 500 (vs 25). Schieber 2001 / Rathelot 2009
        estimate motor cortex sub-pools at 100-500 neurons per action.
      - motor exc_fraction: 0.8 (vs 1.0 pure-exc). Real cortex is 80E/20I.
      - motor internal_density: 0.10 (vs 0.0). Lefort 2009 estimates
        cortical recurrent connectivity at 10-20%.
      - motor exc_weight: 2.0, inh_weight: 4.0 (vs 0.0). Recurrent E
        amplifies signal; local I prevents runaway.
      - n_lang_input: 2048 (vs 256). Wernicke-area scale.

    Combined with cfg.enable_nmda=True (Wang 2002 NMDA bistability),
    these motor pools should produce attractor dynamics — transient
    differential drive locks into sustained differential firing.

    Memory budget at default sizes:
      - 2048 + 4*500 = 4048 neurons
      - lang->motor synapses: 2048 * 500 * 0.30 * 4 = 1.23M
      - motor recurrence: 500 * 500 * 0.10 * 4 = 100K (E/E + E/I + I/E)
      - lang internal recurrence: 2048 * 2048 * 0.05 = 209K
      - Total: ~1.55M synapses. Estimated ~1-2 GB GPU at peak.
        Single-process fit comfortably in RTX 3090 24 GB.

    Returns (regions, pathways) tuple.
    """
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronType

    ACTION_NAMES = ["N", "E", "S", "W"]
    regions = []
    pathways = []

    # Language input region — biological-scale Wernicke-like
    regions.append(BrainRegion(
        name="language_input",
        n_neurons=n_lang_input,
        exc_fraction=lang_exc_fraction,
        internal_density=lang_internal_density,
        exc_weight_mean=2.0, inh_weight_mean=4.0,
        weight_jitter=0.2, plastic_internal=False,  # frozen for clean test
        izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
    ))

    # Motor pools — cortical canon: recurrent excitation + E/I balance
    for action in ACTION_NAMES:
        regions.append(BrainRegion(
            name=f"motor_{action}",
            n_neurons=n_motor_per_action,
            exc_fraction=motor_exc_fraction,
            internal_density=motor_internal_density,
            exc_weight_mean=motor_exc_weight_mean,
            inh_weight_mean=motor_inh_weight_mean,
            weight_jitter=0.2, plastic_internal=False,
            izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name,
        ))

    # language_input -> motor_X (the pathway being tested)
    for action in ACTION_NAMES:
        pathways.append(RegionPathway(
            from_region="language_input", to_region=f"motor_{action}",
            density=text_input_to_motor_density,
            weight_mean=text_input_to_motor_weight,
            weight_jitter=text_input_to_motor_jitter,
            plastic=True,
            plasticity_gate="language_input_to_motor",
        ))

    # Motor lateral inhibition via PV-FSI (Vogels 2011 / Hofer 2011) —
    # biological 12% of motor pool size.
    if enable_motor_fs:
        for action in ACTION_NAMES:
            regions.append(BrainRegion(
                name=f"motor_FS_{action}",
                n_neurons=n_motor_fs_per_action,
                exc_fraction=0.0,
                internal_density=0.0,
                exc_weight_mean=0.0, inh_weight_mean=0.0,
                weight_jitter=0.0, plastic_internal=False,
                izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
            ))
        for action in ACTION_NAMES:
            pathways.append(RegionPathway(
                from_region=f"motor_{action}", to_region=f"motor_FS_{action}",
                density=0.5,
                weight_mean=motor_to_fs_weight,
                weight_jitter=0.3, plastic=False,
            ))
            for target_action in ACTION_NAMES:
                if target_action == action:
                    continue
                pathways.append(RegionPathway(
                    from_region=f"motor_FS_{action}",
                    to_region=f"motor_{target_action}",
                    density=0.5,
                    weight_mean=fs_to_motor_weight,
                    weight_jitter=0.3, plastic=False,
                ))

    return regions, pathways


def apply_topographic_bias(
    bridge,
    topographic_factor: float = 1.5,
    off_target_factor: float = 0.7,
    n_lang_input: int = 256,
    sparsity: float = 0.1,
    verbose: bool = True,
):
    """Apply biology-grounded topographic bias to language_input -> motor_X
    weights. Models the somatotopic Wernicke->motor projection that real
    cortex develops via early Hebbian co-firing (Pulvermüller 2001-2003,
    Hauk et al 2004).

    For each word w with active neuron set A_w (the same set that
    vocab_to_drive_pattern returns):
        weights[A_w -> motor_target(w)] *= topographic_factor
        weights[A_w -> motor_other]     *= off_target_factor

    With default 1.5 / 0.7, the ratio between target and off-target is
    ~2.1x — squarely within Pulvermüller's reported biology range
    (2-3x).

    With baseline weight=3.0, topographic_factor=1.5 gives target init
    of 4.5 — well below stdp_w_max=5.0, leaving STDP room to grow OR
    shrink. Off-target init of 2.1 has even more headroom.

    Args:
        bridge: initialized SimulationBridge (after _initialize_simulation_data)
        topographic_factor: multiplier for weights from active neurons
            of word w to motor_target(w). Default 1.5 (mid-biology).
        off_target_factor: multiplier for weights from active neurons
            of word w to other motor pools. Default 0.7.
        n_lang_input: language_input region size. Used to compute the
            same drive pattern as the eval/training pipeline.
        sparsity: same sparsity used for token drive elsewhere.

    Returns:
        dict with applied edge counts per pathway, for verification.
    """
    import numpy as np
    from sim.text_embeddings import vocab_to_drive_pattern

    if bridge.region_manager is None:
        raise RuntimeError("apply_topographic_bias: region_manager is None")

    rm = bridge.region_manager
    lang_input_indices = list(rm.indices("language_input"))
    n_lang = len(lang_input_indices)
    if n_lang != n_lang_input:
        raise ValueError(
            f"apply_topographic_bias: bridge has {n_lang} language_input "
            f"neurons but caller specified {n_lang_input}"
        )

    word_to_action = {"north": "N", "east": "E", "south": "S", "west": "W"}
    actions = ["N", "E", "S", "W"]

    # Extract current CSR weights once (avoids per-pathway pull)
    indptr = bridge.cp_connections.indptr.get()
    indices = bridge.cp_connections.indices.get()
    data = bridge.cp_connections.data.get()

    # Pre-compute (pre, post) -> data index for fast lookup
    pair_to_idx = {}
    n_rows = int(bridge.cp_connections.shape[0])
    for r in range(n_rows):
        start = int(indptr[r])
        end = int(indptr[r + 1])
        for off in range(start, end):
            pair_to_idx[(r, int(indices[off]))] = off

    summary = {}
    for word, target_action in word_to_action.items():
        # Active language_input neurons for this word
        drive = vocab_to_drive_pattern(word, n_neurons=n_lang_input,
                                        sparsity=sparsity)
        local_active = np.where(drive > 0)[0]
        global_active = [lang_input_indices[i] for i in local_active]

        for action in actions:
            motor_indices = list(rm.indices(f"motor_{action}"))
            factor = (topographic_factor if action == target_action
                      else off_target_factor)

            n_changed = 0
            for src in global_active:
                for dst in motor_indices:
                    key = (src, dst)
                    if key in pair_to_idx:
                        idx = pair_to_idx[key]
                        data[idx] = float(data[idx]) * factor
                        n_changed += 1
            summary[f"{word}->motor_{action}"] = {
                "factor": factor,
                "edges_modified": n_changed,
            }

    # Push back to GPU
    import cupy as cp
    bridge.cp_connections.data = cp.asarray(data, dtype=cp.float32)

    if verbose:
        print(f"[topographic-bias] Applied factor={topographic_factor:.2f}/"
              f"{off_target_factor:.2f} to language_input -> motor_X")
        for k, v in summary.items():
            print(f"  {k}: x{v['factor']:.2f} on {v['edges_modified']} edges")

    return summary


def run_minimal_isolation(
    seed: int = 42,
    n_events_per_direction: int = 1000,
    stim_steps_per_step: int = 100,
    reset_steps: int = 50,
    lang_input_drive_pA: float = 200.0,
    lang_output_coactive_pA: float = 0.0,  # no language_output region
    motor_replay_drive_pA: float = 50.0,
    n_motor_per_action: int = 25,
    n_lang_input: int = 256,
    token_sparsity: float = 0.1,
    dt_ms: float = 1.0,
    text_input_to_motor_weight: float = 3.0,
    text_input_to_motor_jitter: float = 0.5,
    stdp_w_max: float = 5.0,
    enable_hebbian: bool = False,
    # Biology-grounded additions (2026-05-03)
    enable_motor_fs: bool = False,
    n_motor_fs_per_action: int = 3,
    topographic_bias_factor: float = 1.0,  # 1.0 = off (uniform random)
    off_target_bias_factor: float = 1.0,   # 1.0 = off (uniform random)
    freeze_stdp: bool = False,             # anti-cheat control: skip STDP
    # Performance: fast-path spike reset (no GPU-CPU sync). 1.29x on
    # minimal arch under 4-way contention. Numerical equivalence verified
    # at tests/test_fast_spike_reset.py.
    fast_spike_reset: bool = True,
    # Biological-scale architecture (2026-05-04). When True, uses
    # build_biological_brain_regions: cortical canon (recurrence + E/I +
    # NMDA) + larger N. See function docstring for parameter details.
    biological: bool = False,
    enable_nmda: bool = False,
    ou_tau_ms: float = 15.0,
    ou_std_current_pA: float = 100.0,
    verbose: bool = True,
):
    """Run the minimal isolation experiment. Returns (bridge, stats)."""
    import cupy as cp
    from sim.config import (
        CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig,
    )
    from sim.bridge import SimulationBridge

    rng = np.random.default_rng(seed)

    if verbose:
        print("=" * 60)
        print(f"MINIMAL LANGUAGE->MOTOR ISOLATION (seed={seed})")
        print(f"  n_lang_input={n_lang_input}, motor_per_action={n_motor_per_action}")
        total_neurons = (n_lang_input + 4 * n_motor_per_action +
                         (4 * n_motor_fs_per_action if enable_motor_fs else 0))
        print(f"  Total: {total_neurons} neurons")
        print(f"  {n_events_per_direction} paired-stim events per direction")
        print(f"  dt={dt_ms}ms, stim={stim_steps_per_step}, reset={reset_steps}")
        print(f"  enable_hebbian={enable_hebbian}, stdp_w_max={stdp_w_max}")
        print(f"  enable_motor_fs={enable_motor_fs} (n_fs_per_action="
              f"{n_motor_fs_per_action})")
        print(f"  topographic_bias: target={topographic_bias_factor:.2f}, "
              f"off={off_target_bias_factor:.2f} "
              f"(1.0/1.0 = no topography)")
        print(f"  freeze_stdp={freeze_stdp} (anti-cheat control)")
        if biological:
            print(f"  BIOLOGICAL ARCH: cortical canon (recurrence + E/I + NMDA)")
            print(f"  enable_nmda={enable_nmda}, ou_tau_ms={ou_tau_ms}, "
                  f"ou_std_current_pA={ou_std_current_pA}")
        print("=" * 60, flush=True)

    if biological:
        regions, pathways = build_biological_brain_regions(
            n_lang_input=n_lang_input,
            n_motor_per_action=n_motor_per_action,
            text_input_to_motor_weight=text_input_to_motor_weight,
            text_input_to_motor_jitter=text_input_to_motor_jitter,
            enable_motor_fs=enable_motor_fs,
            n_motor_fs_per_action=n_motor_fs_per_action,
        )
    else:
        regions, pathways = build_minimal_brain_regions(
            n_lang_input=n_lang_input,
            n_motor_per_action=n_motor_per_action,
            text_input_to_motor_weight=text_input_to_motor_weight,
            text_input_to_motor_jitter=text_input_to_motor_jitter,
            enable_motor_fs=enable_motor_fs,
            n_motor_fs_per_action=n_motor_fs_per_action,
        )

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = list(regions)
    cfg.region_pathways = list(pathways)
    cfg.dt_ms = dt_ms
    cfg.seed = seed
    cfg.enable_nmda = enable_nmda
    cfg.ou_tau_ms = ou_tau_ms
    cfg.ou_std_current_pA = ou_std_current_pA
    cfg.enable_structural_plasticity = False
    cfg.enable_per_type_stp = False
    cfg.enable_hebbian_learning = enable_hebbian
    cfg.stdp_w_max = stdp_w_max
    cfg.fast_spike_reset = fast_spike_reset

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

    # Apply topographic bias if requested (must come AFTER init, before
    # training, so STDP can refine from the biased starting point).
    if topographic_bias_factor != 1.0 or off_target_bias_factor != 1.0:
        apply_topographic_bias(
            bridge,
            topographic_factor=topographic_bias_factor,
            off_target_factor=off_target_bias_factor,
            n_lang_input=n_lang_input,
            sparsity=token_sparsity,
            verbose=verbose,
        )

    # Anti-cheat control: freeze STDP via plasticity gate. Tests whether
    # topographic bias alone (without learning) solves the task.
    if freeze_stdp:
        try:
            bridge.set_plasticity_gate("language_input_to_motor", 0.0)
            if verbose:
                print("[freeze_stdp] STDP frozen on language_input_to_motor "
                      "(anti-cheat control)", flush=True)
        except Exception as e:
            print(f"[freeze_stdp] WARNING: could not freeze gate: {e}",
                  flush=True)

    # Build synthetic balanced experience buffer
    DIRECTIONS = ["north", "east", "south", "west"]
    DIRECTION_TO_ACTION = {"north": "N", "east": "E", "south": "S", "west": "W"}
    synthetic_buffer = []
    for direction in DIRECTIONS:
        action = DIRECTION_TO_ACTION[direction]
        for _ in range(n_events_per_direction):
            synthetic_buffer.append({
                "token": direction,
                "action": action,
                "reward": 1.0,
                "correct_move": True,
            })
    rng.shuffle(synthetic_buffer)

    if verbose:
        print(f"\n[minimal-iso] Synthetic buffer: {len(synthetic_buffer)} events "
              f"({n_events_per_direction}/dir, shuffled)", flush=True)

    # Training: paired-stim using same mechanism as H4 SWR replay.
    # Inline since we don't have language_output region (curriculum's
    # _run_swr_replay_phase requires it).
    from sim.text_embeddings import vocab_to_drive_pattern

    rm = bridge.region_manager
    lang_input_idx = cp.asarray(list(rm.indices("language_input")), dtype=cp.int64)
    motor_idx = {
        a: cp.asarray(list(rm.indices(f"motor_{a}")), dtype=cp.int64)
        for a in ["N", "E", "S", "W"]
    }
    n_lang = int(lang_input_idx.size)

    t_start = time.time()
    n_replays = 0
    for event_idx, event in enumerate(synthetic_buffer):
        token = event["token"]
        action = event["action"]
        reward = event["reward"]

        # Inter-trial reset
        bridge.cp_external_input_current[:] = 0.0
        bridge.core_config.current_reward_signal = 0.0
        for _ in range(reset_steps):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        # Drive language_input only (no language_output in minimal arch)
        in_drive = vocab_to_drive_pattern(
            token, n_neurons=n_lang,
            drive_max_pA=lang_input_drive_pA, sparsity=token_sparsity,
        )
        bridge.cp_external_input_current[lang_input_idx] = cp.asarray(
            in_drive, dtype=cp.float32,
        )
        # Drive motor pool (the "nudge" toward correct action)
        bridge.cp_external_input_current[motor_idx[action]] += motor_replay_drive_pA

        # Stim window
        for _ in range(stim_steps_per_step):
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        # Apply reward
        bridge.core_config.current_reward_signal = float(reward)
        for _ in range(20):  # reward window
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1

        n_replays += 1

        if verbose and (event_idx + 1) % 250 == 0:
            elapsed = time.time() - t_start
            print(f"  [minimal-iso] {event_idx+1}/{len(synthetic_buffer)} events "
                  f"({elapsed:.0f}s)", flush=True)
            # Tier-1 universal progress event for webapp
            from sim.progress import emit_progress
            emit_progress(
                "replay", event_idx + 1, len(synthetic_buffer),
                phase="paired-stim", unit="events",
                label="minimal-isolation",
                elapsed_seconds=elapsed,
            )

    elapsed = time.time() - t_start
    if verbose:
        print(f"\n[minimal-iso] Training complete: {n_replays} events "
              f"({elapsed:.0f}s)", flush=True)

    training_stats = [{
        "phase": 1,
        "regime": "minimal_language_motor_isolation",
        "n_total_events": n_replays,
        "n_per_direction": n_events_per_direction,
        "elapsed_seconds": elapsed,
    }]

    return bridge, training_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-events-per-direction", type=int, default=1000,
                    help="Paired-stim events per direction (default 1000)")
    ap.add_argument("--n-eval-word-action", type=int, default=25)
    ap.add_argument("--out-stats", type=str, default=None)
    ap.add_argument("--lang-input-drive-pA", type=float, default=200.0)
    ap.add_argument("--motor-replay-drive-pA", type=float, default=50.0)
    ap.add_argument("--n-motor-per-action", type=int, default=25)
    ap.add_argument("--n-lang-input", type=int, default=256)
    ap.add_argument("--stim-steps-per-step", type=int, default=100)
    ap.add_argument("--reset-steps", type=int, default=50)
    ap.add_argument("--token-sparsity", type=float, default=0.1)
    ap.add_argument("--dt-ms", type=float, default=1.0)
    ap.add_argument("--text-input-to-motor-weight", type=float, default=3.0)
    ap.add_argument("--text-input-to-motor-jitter", type=float, default=0.5)
    ap.add_argument("--stdp-w-max", type=float, default=5.0)
    ap.add_argument("--enable-hebbian", action="store_true", default=False)
    # Biology-grounded additions (2026-05-03)
    ap.add_argument("--enable-motor-fs", action="store_true", default=False,
                    help="add motor PV-FS interneurons providing cross-pool "
                    "lateral inhibition (3 FS neurons per pool by default)")
    ap.add_argument("--n-motor-fs-per-action", type=int, default=3,
                    help="FS interneurons per motor pool (default 3 ~12%% "
                    "of 25-neuron pool, biology range 10-15%%)")
    ap.add_argument("--topographic-bias-factor", type=float, default=1.0,
                    help="multiplier for weights from word's active neurons "
                    "to its target motor pool. 1.0 = no topography (random). "
                    "1.5 = mid-biology range (Pulvermuller 2001-2003 ratio "
                    "~2-3x). Pair with --off-target-bias-factor < 1.0.")
    ap.add_argument("--off-target-bias-factor", type=float, default=1.0,
                    help="multiplier for weights from word's active neurons "
                    "to NON-target motor pools. 1.0 = no topography. 0.7 "
                    "with topographic-bias-factor=1.5 gives ratio ~2.1x.")
    ap.add_argument("--freeze-stdp", action="store_true", default=False,
                    help="anti-cheat control: freeze STDP on the language_"
                    "input_to_motor pathway. Combined with topographic bias, "
                    "tests whether the prior alone solves the task.")
    ap.add_argument("--no-fast-spike-reset", dest="fast_spike_reset",
                    action="store_false", default=True,
                    help="disable the fast spike-reset optimization "
                    "(cp.where masked-update, no GPU-CPU sync). Default "
                    "is enabled for ~1.3x speedup on minimal arch.")
    # Biological-scale architecture (2026-05-04). When --biological is set,
    # the runner uses build_biological_brain_regions and bumps default sizes:
    # n_lang_input=2048, n_motor_per_action=500, n_motor_fs_per_action=60.
    # Override individual sizes by passing the relevant flags after --biological.
    ap.add_argument("--biological", action="store_true", default=False,
                    help="use biological-scale architecture (cortical canon: "
                    "recurrent excitation + E/I balance + larger N). "
                    "Auto-bumps lang/motor/FS sizes; enables NMDA. See "
                    "build_biological_brain_regions docstring.")
    ap.add_argument("--enable-nmda", action="store_true", default=False,
                    help="enable NMDA synapses globally (Wang 2002 "
                    "bistability). Auto-set when --biological. Defaults off.")
    ap.add_argument("--ou-tau-ms", type=float, default=15.0,
                    help="OU noise correlation time. Default 15ms (synaptic-"
                    "timescale). Set 50-100ms for slower biological cortical "
                    "noise (alpha/beta-scale).")
    ap.add_argument("--ou-std-current-pA", type=float, default=100.0,
                    help="OU noise amplitude. Default 100pA (CoreSimConfig "
                    "default).")
    args = ap.parse_args()

    # --biological auto-bumps sizes if user didn't override them
    if args.biological:
        if args.n_lang_input == 256:
            args.n_lang_input = 2048
        if args.n_motor_per_action == 25:
            args.n_motor_per_action = 500
        if args.n_motor_fs_per_action == 3:
            args.n_motor_fs_per_action = 60
        # NMDA is integral to biological motor pool dynamics (Wang 2002).
        # Auto-enable unless user explicitly opts out (no opt-out flag yet,
        # so just force on when --biological).
        args.enable_nmda = True

    bridge, train_stats = run_minimal_isolation(
        seed=args.seed,
        n_events_per_direction=args.n_events_per_direction,
        stim_steps_per_step=args.stim_steps_per_step,
        reset_steps=args.reset_steps,
        lang_input_drive_pA=args.lang_input_drive_pA,
        motor_replay_drive_pA=args.motor_replay_drive_pA,
        n_motor_per_action=args.n_motor_per_action,
        n_lang_input=args.n_lang_input,
        token_sparsity=args.token_sparsity,
        dt_ms=args.dt_ms,
        text_input_to_motor_weight=args.text_input_to_motor_weight,
        text_input_to_motor_jitter=args.text_input_to_motor_jitter,
        stdp_w_max=args.stdp_w_max,
        enable_hebbian=args.enable_hebbian,
        enable_motor_fs=args.enable_motor_fs,
        n_motor_fs_per_action=args.n_motor_fs_per_action,
        topographic_bias_factor=args.topographic_bias_factor,
        off_target_bias_factor=args.off_target_bias_factor,
        freeze_stdp=args.freeze_stdp,
        fast_spike_reset=args.fast_spike_reset,
        biological=args.biological,
        enable_nmda=args.enable_nmda,
        ou_tau_ms=args.ou_tau_ms,
        ou_std_current_pA=args.ou_std_current_pA,
        verbose=True,
    )

    # Eval W->A only (no I->W since no visual cortex)
    from research.runners.text_eval import evaluate_word_to_action
    print("\n" + "=" * 60)
    print(f"EVAL: word -> action ({args.n_eval_word_action} per word, "
          f"token_sparsity={args.token_sparsity})")
    print("=" * 60, flush=True)
    wa_result = evaluate_word_to_action(
        bridge, n_trials_per_word=args.n_eval_word_action,
        stim_steps_per_trial=args.stim_steps_per_step,
        n_reset_steps=args.reset_steps,
        token_sparsity=args.token_sparsity,
    )
    print(f"\n  Accuracy: {wa_result['correct']}/{wa_result['n_trials']} "
          f"= {wa_result['accuracy']:.1%}")
    print(f"  Confusion: {wa_result['confusion_matrix']}", flush=True)

    if args.out_stats:
        out = {
            "regime": "minimal_language_motor_isolation",
            "seed": args.seed,
            "n_events_per_direction": args.n_events_per_direction,
            "n_total_events": 4 * args.n_events_per_direction,
            "training_stats": train_stats,
            "word_to_action_eval": wa_result,
            "config": {
                "n_lang_input": args.n_lang_input,
                "n_motor_per_action": args.n_motor_per_action,
                "lang_input_drive_pA": args.lang_input_drive_pA,
                "motor_replay_drive_pA": args.motor_replay_drive_pA,
                "stim_steps_per_step": args.stim_steps_per_step,
                "reset_steps": args.reset_steps,
                "token_sparsity": args.token_sparsity,
                "dt_ms": args.dt_ms,
                "text_input_to_motor_weight": args.text_input_to_motor_weight,
                "stdp_w_max": args.stdp_w_max,
                "enable_hebbian": args.enable_hebbian,
                "enable_motor_fs": args.enable_motor_fs,
                "n_motor_fs_per_action": args.n_motor_fs_per_action,
                "topographic_bias_factor": args.topographic_bias_factor,
                "off_target_bias_factor": args.off_target_bias_factor,
                "freeze_stdp": args.freeze_stdp,
            },
        }
        from pathlib import Path
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_stats).write_text(json.dumps(out, indent=2))
        print(f"\n  Saved: {args.out_stats}", flush=True)


if __name__ == "__main__":
    main()
