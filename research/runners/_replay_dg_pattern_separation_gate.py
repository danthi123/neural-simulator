"""Replay pattern-separation gate: a DG-style sparse-expansive separator on the
consolidation stream keeps SIMILAR memories from blurring in the cortical store.

Board #43 / "keep similar memories from blurring during sleep-like replay".

Prior record (read before extending):
  * research/findings/2026-08-03-replay-cortical-consolidation-v2-calibration-NO-GO.md
    -- the replay->cortex path is CAUSAL but its cortical write alternates
       between diffuse FALSE recall (broad coactivity strengthens the wrong
       target) and near-inert learning. The named residual: "make CA1->cortex
       reinstatement reliable without increasing false recall, using a local
       competitive/inhibitory mechanism." That false-recall == the BLUR.
  * research/findings/2026-05-31-DG-biologization-FUNDAMENTAL-BOUNDARY-...md
    -- DG cannot cheaply produce a NEAR-ORTHOGONAL (cos ~ 0) VSA symbol on this
       substrate (separation-vs-reliability tradeoff). BUT its own refinement:
       the substrate is ALREADY id-separable; the unmet bar was near-orthogonal
       VSA binding, NOT discriminability. This gate's bar is discriminability /
       no-confusion -- a much lower bar than that boundary.

Mechanism under test (biology): the dentate gyrus recodes overlapping entorhinal
inputs into a SPARSE, EXPANSIVE code via a random perforant projection plus
strong feedforward/feedback PV-basket inhibition (winner-take-few). Similar
inputs -> orthogonalized DG engrams. If replay re-emits the SEPARATED engrams,
the offline Hebbian cortical write binds each memory's answer to distinct cells
and the two memories stay discriminable. Remove the DG competition (dense DG) and
the engrams overlap -> the write cross-contaminates -> the blur returns.

Circuit (one persistent spiking bridge, Izhikevich, rate-window Hebbian):
    input (EC)  --fixed random expansive-->  dg (excitatory)
    dg  <--I_TO_E feedback (gate: dg_competition)-->  dg_fs (PV basket)
    dg  --PLASTIC (gate: dg_answer_write, sleep-only)-->  answer (cortex)
    answer  <--opponent inhibition-->  answer_fs
Consolidation happens OFFLINE (replay): each event reinstates a memory's input
(-> its dg engram) together with its answer assembly (the hippocampal index),
and the coincidence writes dg_engram -> answer. Retrieval drives ONLY the input
(index/teacher OFF, plasticity OFF) and reads which answer assembly wins.

The single manipulated variable across the dissociation is the DG competition
gate; every drive, pattern, and schedule is identical. So a difference is
attributable to the separator, not to the scaffold.

Acknowledged scaffolds (tracked, not hidden): host-defined input patterns
(sensory world), host reinstatement of the memory's input AND answer during
replay (the hippocampal index / SWR trigger), scheduled down-states, and a
host argmax over answer-assembly spike counts for MEASUREMENT only.

CPU smoke (fast, structural):
    OMP_NUM_THREADS=2 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_gate --seeds 42 --smoke

6-seed run:
    OMP_NUM_THREADS=2 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_gate \
        --seeds 42 43 44 100 101 102 --out research/findings/raw/replay_dg_sep/run.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.runners._replay_cortical_consolidation_gate import (  # noqa: E402
    _all_to_all,
    _path_weights,
    _zero_current,
)


def _reset_dynamics(bridge) -> None:
    """Return activity to a true down-state WITHOUT changing learned weights.

    Beyond membrane/conductance, this drains the synaptic delay ring buffer
    (``cp_synapse_pulse_timers``/``progress``) and the Hebbian coactivity trace,
    so an in-flight spike from one replay/probe event cannot leak into the next.
    (Not draining the buffer silently contaminated every second read -- a
    memory's engram vanished because the prior memory's spikes were still in
    flight.)

    THE READ-ISOLATION FIX (2026-09-02, board #150's ~29-runner follow-up audit — see
    `research/findings/2026-09-02-read-isolation-audit-C2-bug-class-across-14-runners.md`): this reset already
    covered `cp_prev_firing_states` and drained the pulse-timer ring buffer, but never zeroed
    `cp_refractory_timers` (a HARD firing gate independent of membrane potential) — `enable_homeostasis=False`
    in `build_bridge` makes the other 2 C2 arrays (`cp_neuron_activity_ema` / `cp_neuron_firing_thresholds`)
    config-inert here, so they are not added. VERIFIED CLEAN both before and after this hygiene port: a
    repeat-probe / order-dependence diagnostic (`_probe` on the same target twice, and with an intervening
    different-target probe) is BITWISE IDENTICAL on this file's `smoke_config()` — the `replay_settle_steps`
    window already washes out the missing refractory residue before any scored step. This port is defense-in-
    depth only; it does not change (and cannot change, per that diagnostic) the file's own banked NO-GO verdict
    (`2026-08-03-replay-cortical-consolidation-v2-calibration-NO-GO.md`).
    """
    if getattr(bridge, "cp_izh_c_reset", None) is not None:
        bridge.cp_membrane_potential_v[:] = bridge.cp_izh_c_reset
    else:
        bridge.cp_membrane_potential_v[:] = -65.0
    if getattr(bridge, "cp_recovery_variable_u", None) is not None:
        bridge.cp_recovery_variable_u[:] = 0.0
    for name in (
        "cp_firing_states",
        "cp_prev_firing_states",
        "cp_refractory_timers",
        "cp_conductance_g_e",
        "cp_conductance_g_i",
        "cp_conductance_g_nmda",
        "cp_conductance_g_nmda_rise",
        "cp_conductance_g_nmda_recurrent",
        "cp_conductance_g_nmda_recurrent_rise",
        "cp_synapse_pulse_timers",
        "cp_synapse_pulse_progress",
        "cp_hebb_coactivity_trace",
    ):
        arr = getattr(bridge, name, None)
        if arr is not None:
            arr[:] = 0
    _zero_current(bridge)

SEEDS = (42, 43, 44, 100, 101, 102)

DG_WRITE_GATE = "dg_answer_write"
DG_COMPETITION_GATE = "dg_competition"
ANSWER_INHIBITION_GATE = "answer_opponent"

# Memory battery. Each is (name, {"pair": (m0, m1), "overlap": jaccard-ish}).
# "similar" -> high input overlap (the memories that blur); "dissimilar" -> low.
CONDITIONS = (
    "similar_separator_on",
    "similar_separator_off",   # DG competition lesion -> the NULL / blur baseline
    "dissimilar_separator_on",
    "dissimilar_separator_off",
    "single_separator_on",     # single-memory recall guard (no interference)
)


@dataclass(frozen=True)
class GateConfig:
    # populations
    n_input: int = 48
    n_dg: int = 200
    n_dg_fs: int = 32
    n_answer: int = 60
    n_answer_fs: int = 12
    answer_assembly: int = 16      # cells per memory's answer (disjoint across the 2 memories)
    # memory input patterns (indices into n_input)
    input_assembly: int = 24
    similar_overlap: int = 18      # shared cells for the SIMILAR pair (Jaccard 18/30 = 0.60)
    dissimilar_overlap: int = 2    # shared cells for the DISSIMILAR pair (Jaccard ~0.04)
    # perforant path input->dg (fixed random EXPANSIVE projection: dense baseline
    # so that WITHOUT inhibition DG fires densely and inherits the input overlap)
    dg_fan_in: int = 20            # afferents per dg cell
    input_to_dg_weight: float = 40.0
    # DG feedforward + feedback PV-basket inhibition (the k-WTA separator; the
    # single manipulated variable is the fs_to_dg transmission gate)
    input_to_fs_weight: float = 60.0    # feedforward (perforant -> interneuron)
    dg_to_fs_weight: float = 30.0       # feedback (granule -> interneuron)
    fs_to_dg_weight: float = 6.0        # basket -> granule (gated: dg_competition); moderate to avoid post-inhibitory rebound
    # answer opponent competition (winner-take-most read-out)
    answer_to_fs_weight: float = 120.0
    fs_to_answer_weight: float = 60.0
    # plastic consolidation write dg->answer (Hebbian, sleep-only)
    dg_answer_init_weight: float = 0.05
    # short local delays so the microcircuit's inhibition can gate within a
    # readout window (hippocampal local synapses are ~1-2 ms; the 20 ms default
    # is a long-range value that makes any fast competition impossible).
    max_synaptic_delay_ms: float = 2.0
    # drives
    replay_input_drive_pA: float = 1300.0
    replay_answer_drive_pA: float = 1300.0   # the reinstated answer (hippocampal index)
    probe_input_drive_pA: float = 1300.0
    # schedule
    replay_events_per_memory: int = 14
    replay_on_steps: int = 25
    replay_settle_steps: int = 12
    probe_steps: int = 40
    probe_cue_fraction: float = 1.0          # full input cue at retrieval
    # plasticity
    hebbian_learning_rate: float = 0.0       # wake write disabled (write is sleep-only)
    replay_learning_rate: float = 2.0
    hebbian_max_weight: float = 90.0
    hebbian_coactivity_decay: float = 0.90
    hebbian_coactivity_thresh: float = 0.01
    propagation_strength: float = 0.12


def smoke_config() -> GateConfig:
    return GateConfig(
        n_input=48,
        n_dg=160,
        n_dg_fs=28,
        n_answer=40,
        n_answer_fs=8,
        answer_assembly=12,
        input_assembly=24,
        similar_overlap=18,
        dissimilar_overlap=2,
        replay_events_per_memory=6,
        replay_on_steps=20,
        replay_settle_steps=10,
        probe_steps=25,
    )


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = set(int(x) for x in a), set(int(x) for x in b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _input_patterns(seed: int, cfg: GateConfig, kind: str) -> dict[str, np.ndarray]:
    """Two memory input patterns over n_input with controlled overlap.

    'similar' -> high overlap (the memories that blur); 'dissimilar' -> low;
    'single' -> only m0 is used downstream.
    """
    rng = np.random.default_rng(seed * 97 + (7 if kind == "similar" else 19 if kind == "dissimilar" else 3))
    pool = np.arange(cfg.n_input)
    overlap = cfg.similar_overlap if kind == "similar" else cfg.dissimilar_overlap
    size = cfg.input_assembly
    draw = rng.choice(pool, 2 * size - overlap, replace=False)
    shared = draw[:overlap]
    m0 = np.sort(np.concatenate([shared, draw[overlap:size]]))
    m1 = np.sort(np.concatenate([shared, draw[size:]]))
    return {"m0": m0, "m1": m1}


def _answer_assemblies(seed: int, cfg: GateConfig, answer_idx: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed * 61 + 29)
    draw = rng.choice(answer_idx, 2 * cfg.answer_assembly, replace=False)
    a0, a1 = np.split(np.sort(draw), 2)
    return {"m0": np.sort(a0), "m1": np.sort(a1)}


def _perforant_edges(seed: int, cfg: GateConfig, input_idx: np.ndarray, dg_idx: np.ndarray):
    """Fixed random expansive projection: each DG cell samples dg_fan_in inputs."""
    rng = np.random.default_rng(seed * 131 + 41)
    pre, post = [], []
    for dg in dg_idx:
        aff = rng.choice(input_idx, cfg.dg_fan_in, replace=False)
        pre.extend(int(x) for x in aff)
        post.extend([int(dg)] * cfg.dg_fan_in)
    return np.asarray(pre, dtype=np.int64), np.asarray(post, dtype=np.int64)


def build_bridge(seed: int, cfg: GateConfig):
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, GPUConfig, RuntimeState, VisualizationConfig
    from sim.enums import NeuronModel, NeuronType
    from sim.regions import BrainRegion, RegionPathway

    exc = dict(exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
               weight_jitter=0.0, plastic_internal=False,
               izh_neuron_type=NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name)
    inh = dict(exc_fraction=0.0, internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
               weight_jitter=0.0, plastic_internal=False,
               izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name)

    core = CoreSimConfig()
    core.enable_brain_region_framework = True
    core.brain_regions = [
        BrainRegion("input", cfg.n_input, **exc),
        BrainRegion("dg", cfg.n_dg, **exc),
        BrainRegion("dg_fs", cfg.n_dg_fs, **inh),
        BrainRegion("answer", cfg.n_answer, **exc),
        BrainRegion("answer_fs", cfg.n_answer_fs, **inh),
    ]
    core.region_pathways = [RegionPathway("input", "dg", density=0.01, weight_mean=0.01, plastic=False)]
    core.num_neurons = 0
    core.connections_per_neuron = 0
    core.neuron_model_type = NeuronModel.IZHIKEVICH.name
    core.neural_profile_name = "GENERIC_UNSTRUCTURED"
    core.dt_ms = 1.0
    core.seed = core.heterogeneity_seed = core.ou_seed = int(seed)
    core.enable_stdp = False
    core.enable_hebbian_learning = True
    core.hebbian_rate_window = True
    core.hebbian_learning_rate = float(cfg.hebbian_learning_rate)
    core.hebbian_max_weight = float(cfg.hebbian_max_weight)
    core.hebbian_min_weight = 0.0
    core.hebbian_weight_decay = 0.0
    core.hebbian_coactivity_decay = float(cfg.hebbian_coactivity_decay)
    core.hebbian_coactivity_thresh = float(cfg.hebbian_coactivity_thresh)
    core.enable_reward_modulation = False
    core.enable_homeostasis = False
    core.enable_short_term_plasticity = False
    core.enable_structural_plasticity = False
    core.enable_ou_process = False
    core.ou_std_current_pA = 0.0
    core.fast_spike_reset = True
    core.propagation_strength = float(cfg.propagation_strength)
    core.max_synaptic_delay_ms = float(cfg.max_synaptic_delay_ms)

    runtime = RuntimeState()
    runtime.actual_seed_used = int(seed)
    bridge = SimulationBridge(core_config=core, viz_config=VisualizationConfig(),
                              runtime_state=runtime, gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(core.max_synaptic_delay_ms / core.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)

    regions = {name: np.asarray(bridge.region_manager.indices(name), dtype=np.int64)
               for name in ("input", "dg", "dg_fs", "answer", "answer_fs")}

    pp_pre, pp_post = _perforant_edges(seed, cfg, regions["input"], regions["dg"])
    input_to_fs_pre, input_to_fs_post = _all_to_all(regions["input"], regions["dg_fs"])
    dg_to_fs_pre, dg_to_fs_post = _all_to_all(regions["dg"], regions["dg_fs"])
    fs_to_dg_pre, fs_to_dg_post = _all_to_all(regions["dg_fs"], regions["dg"])
    ans_to_fs_pre, ans_to_fs_post = _all_to_all(regions["answer"], regions["answer_fs"])
    fs_to_ans_pre, fs_to_ans_post = _all_to_all(regions["answer_fs"], regions["answer"])
    dg_ans_pre, dg_ans_post = _all_to_all(regions["dg"], regions["answer"])

    def group(pre, post, weight, *, plastic, conn_type, plasticity_gate=None, transmission_gate=None):
        row = {"pre_indices": pre.tolist(), "post_indices": post.tolist(),
               "initial_weights": np.full(pre.size, weight, dtype=np.float32),
               "plastic": bool(plastic), "conn_type": conn_type, "count": int(pre.size)}
        if plasticity_gate:
            row["plasticity_gate"] = plasticity_gate
        if transmission_gate:
            row["transmission_gate"] = transmission_gate
        return row

    wiring = {
        "input_to_dg": group(pp_pre, pp_post, cfg.input_to_dg_weight, plastic=False, conn_type="E_TO_E"),
        "input_to_fs": group(input_to_fs_pre, input_to_fs_post, cfg.input_to_fs_weight, plastic=False, conn_type="E_TO_I"),
        "dg_to_fs": group(dg_to_fs_pre, dg_to_fs_post, cfg.dg_to_fs_weight, plastic=False, conn_type="E_TO_I"),
        "fs_to_dg": group(fs_to_dg_pre, fs_to_dg_post, cfg.fs_to_dg_weight, plastic=False,
                          conn_type="I_TO_E", transmission_gate=DG_COMPETITION_GATE),
        "answer_to_fs": group(ans_to_fs_pre, ans_to_fs_post, cfg.answer_to_fs_weight, plastic=False, conn_type="E_TO_I"),
        "fs_to_answer": group(fs_to_ans_pre, fs_to_ans_post, cfg.fs_to_answer_weight, plastic=False,
                             conn_type="I_TO_E", transmission_gate=ANSWER_INHIBITION_GATE),
        "dg_to_answer": group(dg_ans_pre, dg_ans_post, cfg.dg_answer_init_weight, plastic=True,
                             conn_type="E_TO_E", plasticity_gate=DG_WRITE_GATE),
    }
    inh_indices = np.concatenate([regions["dg_fs"], regions["answer_fs"]]).tolist()
    bridge.inject_explicit_wiring(wiring, output_inhibitory_indices=inh_indices)
    bridge.set_transmission_gate(DG_COMPETITION_GATE, 1.0)
    bridge.set_transmission_gate(ANSWER_INHIBITION_GATE, 1.0)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)

    handles = {"regions": regions, "wiring_counts": {k: v["count"] for k, v in wiring.items()},
               "bridge_identity": id(bridge)}
    return bridge, handles


def _dg_engram(bridge, cfg: GateConfig, regions: dict, input_pat: np.ndarray, competition: bool) -> np.ndarray:
    """Read the DG engram evoked by an input pattern (no plasticity, measurement)."""
    bridge.set_transmission_gate(DG_COMPETITION_GATE, 1.0 if competition else 0.0)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)
    _reset_dynamics(bridge)
    dg_idx = regions["dg"]
    counts = np.zeros(dg_idx.size, dtype=np.float64)
    from sim.backend import to_host
    for step in range(cfg.replay_on_steps + cfg.replay_settle_steps):
        _zero_current(bridge)
        if step < cfg.replay_on_steps:
            bridge.cp_external_input_current[input_pat] = cfg.replay_input_drive_pA
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        counts += np.asarray(to_host(bridge.cp_firing_states[dg_idx]), dtype=np.float64)
    _zero_current(bridge)
    return dg_idx[counts > 0]


def _replay_consolidate(bridge, cfg: GateConfig, regions: dict, memories: dict, competition: bool, seed: int) -> dict:
    """Offline replay: interleave reinstatement of each memory's (input, answer)
    and let the sleep-gated Hebbian rule write dg_engram -> answer."""
    bridge.set_transmission_gate(DG_COMPETITION_GATE, 1.0 if competition else 0.0)
    bridge.set_transmission_gate(ANSWER_INHIBITION_GATE, 1.0)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 1.0)
    bridge.core_config.hebbian_learning_rate = float(cfg.replay_learning_rate)

    names = list(memories.keys())
    # deterministic interleaved replay order (A, B, A, B, ...)
    order = []
    for _ in range(cfg.replay_events_per_memory):
        order.extend(names)
    dg_spikes = {name: 0 for name in names}
    answer_spikes = {name: 0 for name in names}
    from sim.backend import to_host
    for name in order:
        _reset_dynamics(bridge)            # down-state between replay events
        inp = memories[name]["input"]
        ans = memories[name]["answer"]
        for step in range(cfg.replay_on_steps + cfg.replay_settle_steps):
            _zero_current(bridge)
            if step < cfg.replay_on_steps:
                bridge.cp_external_input_current[inp] = cfg.replay_input_drive_pA
                bridge.cp_external_input_current[ans] = cfg.replay_answer_drive_pA
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
            firing = bridge.cp_firing_states
            dg_spikes[name] += int(firing[regions["dg"]].sum())
            answer_spikes[name] += int(firing[ans].sum())
    _zero_current(bridge)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)
    bridge.core_config.hebbian_learning_rate = float(cfg.hebbian_learning_rate)
    return {"replay_events": len(order), "dg_spikes": dg_spikes, "answer_spikes": answer_spikes}


def _probe(bridge, cfg: GateConfig, regions: dict, memories: dict, target_name: str, competition: bool, seed: int) -> dict:
    """Hippocampus-independent retrieval: drive ONLY the memory's input (index /
    teacher off, plasticity off) and read which answer assembly wins."""
    from sim.backend import to_host
    bridge.set_transmission_gate(DG_COMPETITION_GATE, 1.0 if competition else 0.0)
    bridge.set_transmission_gate(ANSWER_INHIBITION_GATE, 1.0)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)
    _reset_dynamics(bridge)

    inp = memories[target_name]["input"].copy()
    if cfg.probe_cue_fraction < 1.0:
        rng = np.random.default_rng(50_000 + seed * 7 + hash(target_name) % 97)
        rng.shuffle(inp)
        n_partial = max(2, int(round(cfg.probe_cue_fraction * inp.size)))
        inp = np.sort(inp[:n_partial])

    answer_idx = regions["answer"]
    counts = np.zeros(answer_idx.size, dtype=np.float64)
    for _ in range(cfg.probe_steps):
        _zero_current(bridge)
        bridge.cp_external_input_current[inp] = cfg.probe_input_drive_pA
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        counts += np.asarray(to_host(bridge.cp_firing_states[answer_idx]), dtype=np.float64)
    _zero_current(bridge)

    local = {int(g): i for i, g in enumerate(answer_idx)}
    names = list(memories.keys())
    other = [n for n in names if n != target_name][0]
    correct_pos = np.asarray([local[int(i)] for i in memories[target_name]["answer"]], dtype=np.int64)
    wrong_pos = np.asarray([local[int(i)] for i in memories[other]["answer"]], dtype=np.int64)
    occupied = set(correct_pos.tolist()) | set(wrong_pos.tolist())
    bg_pos = np.asarray([i for i in range(answer_idx.size) if i not in occupied], dtype=np.int64)

    correct = float(counts[correct_pos].mean() / cfg.probe_steps)
    wrong = float(counts[wrong_pos].mean() / cfg.probe_steps)
    background = float(counts[bg_pos].mean() / cfg.probe_steps) if bg_pos.size else 0.0
    total = float(counts.sum())
    false_spikes = float(counts[wrong_pos].sum()) + (float(counts[bg_pos].sum()) if bg_pos.size else 0.0)
    return {
        "target": target_name,
        "correct_rate": correct,
        "wrong_rate": wrong,
        "background_rate": background,
        "margin": correct - max(wrong, background),
        "selectivity": (correct - wrong) / (correct + wrong + 1e-9),
        "false_recall_fraction": false_spikes / total if total > 0 else 0.0,
        "target_assembly_wins": bool(correct > wrong),
        "total_answer_spikes": int(total),
    }


def run_condition(seed: int, condition: str, cfg: GateConfig) -> dict:
    kind = "similar" if condition.startswith("similar") else "dissimilar" if condition.startswith("dissimilar") else "single"
    competition = condition.endswith("separator_on")
    bridge, handles = build_bridge(seed, cfg)
    regions = handles["regions"]
    inputs = _input_patterns(seed, cfg, kind)
    answers = _answer_assemblies(seed, cfg, regions["answer"])

    if kind == "single":
        memories = {"m0": {"input": inputs["m0"], "answer": answers["m0"]},
                    "m1": {"input": inputs["m1"], "answer": answers["m1"]}}
        # single-memory guard: consolidate ONLY m0, then confirm m0 recalls and
        # nothing is spuriously bound to m1's answer.
        replay_mems = {"m0": memories["m0"]}
    else:
        memories = {"m0": {"input": inputs["m0"], "answer": answers["m0"]},
                    "m1": {"input": inputs["m1"], "answer": answers["m1"]}}
        replay_mems = memories

    # DG separation measured directly (mechanistic precondition).
    eng0 = _dg_engram(bridge, cfg, regions, inputs["m0"], competition)
    eng1 = _dg_engram(bridge, cfg, regions, inputs["m1"], competition)
    dg_sep = {
        "input_jaccard": _jaccard(inputs["m0"], inputs["m1"]),
        "dg_jaccard": _jaccard(eng0, eng1),
        "dg_active_frac_m0": eng0.size / regions["dg"].size,
        "dg_active_frac_m1": eng1.size / regions["dg"].size,
    }

    w_before = _path_weights(bridge, DG_WRITE_GATE)
    replay = _replay_consolidate(bridge, cfg, regions, replay_mems, competition, seed)
    w_after = _path_weights(bridge, DG_WRITE_GATE)
    replay["dg_answer_weight_delta"] = float(np.mean(w_after - w_before))

    probes = {name: _probe(bridge, cfg, regions, memories, name, competition, seed)
              for name in memories}

    return {
        "seed": int(seed),
        "condition": condition,
        "kind": kind,
        "competition": bool(competition),
        "single_bridge": handles["bridge_identity"] == id(bridge),
        "wiring_counts": handles["wiring_counts"],
        "dg_separation": dg_sep,
        "replay": replay,
        "probes": probes,
        "mean_selectivity": float(np.mean([p["selectivity"] for p in probes.values()])),
        "mean_correct": float(np.mean([p["correct_rate"] for p in probes.values()])),
        "mean_wrong": float(np.mean([p["wrong_rate"] for p in probes.values()])),
        "mean_false_recall": float(np.mean([p["false_recall_fraction"] for p in probes.values()])),
        "both_win": all(p["target_assembly_wins"] for p in probes.values()),
    }


def run_seed(seed: int, cfg: GateConfig) -> dict:
    from tools.lab import attributable_to
    rows = {c: run_condition(seed, c, cfg) for c in CONDITIONS}
    on = rows["similar_separator_on"]
    off = rows["similar_separator_off"]
    # Attribute the DG-separation difference to the competition (treatment ON) vs
    # its lesion control (OFF): a pair measured is not a pair attributed.
    separator_attribution = attributable_to(
        "DG competition on engram separation",
        off["dg_separation"]["dg_jaccard"], on["dg_separation"]["dg_jaccard"])
    dg_drop_on = on["dg_separation"]["input_jaccard"] - on["dg_separation"]["dg_jaccard"]
    verdict = {
        # anti-cheat 1: similar memories discriminable after consolidation, and
        #               the NULL (separator off) shows the blur.
        "similar_on_discriminable": on["both_win"] and on["mean_selectivity"] >= 0.30
        and on["mean_correct"] >= 0.02,
        "null_shows_blur": off["mean_selectivity"] < on["mean_selectivity"] - 0.20,
        # anti-cheat 2: the separator is load-bearing (lesion -> blur returns).
        "separator_dissociation": (on["mean_selectivity"] - off["mean_selectivity"]) >= 0.20,
        "dg_actually_separates": on["dg_separation"]["dg_jaccard"] <= off["dg_separation"]["dg_jaccard"] - 0.15
        and on["dg_separation"]["dg_jaccard"] < on["dg_separation"]["input_jaccard"] - 0.15,
        # anti-cheat 3: no catastrophic cost on dissimilar / single memory.
        "dissimilar_preserved": rows["dissimilar_separator_on"]["both_win"]
        and rows["dissimilar_separator_on"]["mean_selectivity"] >= 0.30,
        "single_memory_recall": rows["single_separator_on"]["probes"]["m0"]["target_assembly_wins"]
        and rows["single_separator_on"]["probes"]["m0"]["correct_rate"] >= 0.02,
    }
    return {
        "seed": int(seed),
        "conditions": rows,
        "summary": {
            "similar_on_selectivity": on["mean_selectivity"],
            "similar_off_selectivity": off["mean_selectivity"],
            "selectivity_dissociation": on["mean_selectivity"] - off["mean_selectivity"],
            "similar_on_false_recall": on["mean_false_recall"],
            "similar_off_false_recall": off["mean_false_recall"],
            "dg_jaccard_on": on["dg_separation"]["dg_jaccard"],
            "dg_jaccard_off": off["dg_separation"]["dg_jaccard"],
            "input_jaccard": on["dg_separation"]["input_jaccard"],
            "dg_separation_gain": dg_drop_on,
            "dissimilar_on_selectivity": rows["dissimilar_separator_on"]["mean_selectivity"],
            "single_correct": rows["single_separator_on"]["probes"]["m0"]["correct_rate"],
            "separator_attribution": separator_attribution,
        },
        "checks": verdict,
        "seed_go": all(verdict.values()),
    }


def run(seeds: Iterable[int], cfg: GateConfig) -> dict:
    started = time.time()
    rows = [run_seed(int(s), cfg) for s in seeds]
    n = len(rows)

    def pooled(key_fn):
        return float(np.mean([key_fn(r) for r in rows]))

    pooled_summary = {
        "similar_on_selectivity": pooled(lambda r: r["summary"]["similar_on_selectivity"]),
        "similar_off_selectivity": pooled(lambda r: r["summary"]["similar_off_selectivity"]),
        "selectivity_dissociation": pooled(lambda r: r["summary"]["selectivity_dissociation"]),
        "similar_on_false_recall": pooled(lambda r: r["summary"]["similar_on_false_recall"]),
        "similar_off_false_recall": pooled(lambda r: r["summary"]["similar_off_false_recall"]),
        "dg_jaccard_on": pooled(lambda r: r["summary"]["dg_jaccard_on"]),
        "dg_jaccard_off": pooled(lambda r: r["summary"]["dg_jaccard_off"]),
        "input_jaccard": pooled(lambda r: r["summary"]["input_jaccard"]),
        "dissimilar_on_selectivity": pooled(lambda r: r["summary"]["dissimilar_on_selectivity"]),
        "single_correct": pooled(lambda r: r["summary"]["single_correct"]),
    }
    check_names = list(rows[0]["checks"].keys())
    pooled_checks = {name: int(sum(r["checks"][name] for r in rows)) for name in check_names}
    all_go = all(r["seed_go"] for r in rows)
    majority = {name: pooled_checks[name] >= (n // 2 + 1) for name in check_names}
    return {
        "gate": "replay_dg_pattern_separation",
        "seeds": [int(s) for s in seeds],
        "n_seeds": n,
        "aggregate_status": "GO" if all_go else "NO-GO",
        "seeds_go": [r["seed"] for r in rows if r["seed_go"]],
        "pooled_summary": pooled_summary,
        "pooled_check_counts": pooled_checks,
        "pooled_majority": majority,
        "per_seed": rows,
        "scaffolds": [
            "host-defined input (sensory) patterns and answer assemblies",
            "host reinstatement of each memory's input AND answer during replay (hippocampal index / SWR trigger)",
            "scheduled down-states between replay events",
            "host argmax over answer-assembly spike counts for measurement only",
            "fixed random perforant projection and fixed FS anatomy (not developed)",
        ],
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cfg = smoke_config() if args.smoke else GateConfig()
    print(f"[replay-dg-pattern-separation] backend={os.environ.get('SIM_BACKEND','default')} "
          f"seeds={args.seeds} smoke={args.smoke}", flush=True)
    payload = run(args.seeds, cfg)
    for r in payload["per_seed"]:
        s = r["summary"]
        print(f"  seed {r['seed']}: GO={r['seed_go']} sel_on={s['similar_on_selectivity']:.3f} "
              f"sel_off={s['similar_off_selectivity']:.3f} dissoc={s['selectivity_dissociation']:+.3f} "
              f"dgJ_on={s['dg_jaccard_on']:.3f} dgJ_off={s['dg_jaccard_off']:.3f} inputJ={s['input_jaccard']:.3f} "
              f"checks={r['checks']}", flush=True)
    print(f"  AGGREGATE: {payload['aggregate_status']} seeds_go={payload['seeds_go']}", flush=True)
    print(f"  pooled: {json.dumps(payload['pooled_summary'], indent=None)}", flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
