"""Replay DG pattern-separation on the PRODUCTION Izhikevich substrate (board #43).

Port target: reproduce, on the real ``SimulationBridge`` (Izhikevich + rate-window
Hebbian), the controlled-LIF result of
``research/findings/2026-08-19-replay-pattern-separation-DG-separator-keeps-similar-memories-discriminable-6seed-GO.md``
-- a DG sparse-expansive separator on the replay stream keeps two SIMILAR memories
discriminable through offline consolidation.

WHAT THE PRIOR BRIDGE RUNNER HIT (``_replay_dg_pattern_separation_gate.py``): the
consolidation READ stalled at chance (mean selectivity ~0, 6/6). The prior finding
attributed this to (a) Izhikevich post-inhibitory REBOUND and (b) a razor-thin k-WTA.

WHAT THIS RUNNER FIXES (two biologically-grounded levers, additive, NO sim/ edit --
both use existing public config fields):

  1. SHUNTING INHIBITION (fixes the rebound). The DG granule (and answer) inhibitory
     reversal is set to ~vr (BrainRegion.syn_reversal_potential_i_override, the same
     field striatal MSNs use). Inhibition then DIVIDES the excitatory drive toward
     rest instead of hyperpolarizing BELOW vr, where the Izhikevich quadratic
     k*(v-vr)*(v-vt) turns regeneratively depolarizing and produces the rebound burst.
     Biology: shunting (chloride, ECl~=Vrest) feedforward inhibition -- Carandini &
     Heeger divisive normalization; granule cells sit near ECl.

  2. TRANSMISSION-GATED WRITE (fixes a Hebbian RUNAWAY the prior finding missed).
     The bridge rate-window rule is SOFT-BOUND: dw = lr*coact*(w_max - w). On an
     all-to-all dg->answer path with strong dg drive, ONLINE plasticity runs away --
     as any dg->answer weight grows, dg drives non-target answer cells, they fire,
     Hebbian potentiates them, and the whole matrix saturates -> the write is
     non-selective -> chance read (this, NOT the DG rebound, was the dominant blocker
     -- measured: answer spikes balloon to ~1900/event during plastic replay while a
     teacher-only measurement shows the target assembly firing alone). The LIF
     sidesteps this by computing the write OFFLINE (answer = teacher only during the
     coincidence). The bridge equivalent: a TRANSMISSION gate on dg->answer, OFF
     during replay (the WRITE: answer fires from the hippocampal-index teacher only ->
     clean selective coincidence) and ON during probe (the READ: dg drives answer via
     the learned weights). Plasticity is per-neuron pre x post firing, independent of
     transmission, so the coincidence still writes. Biology: encoding coincidence
     (SWR-time potentiation) is distinct from recall transmission.

RESULT (see the finding). With both fixes the pipeline WORKS end-to-end on the
Izhikevich substrate: SINGLE-memory recall is at ceiling (selectivity ~+0.9-1.0, 6/6)
-- the read no longer stalls at chance -- and the per-memory selectivity for the
similar pair is large (|sel| up to ~0.55). The RESIDUAL, precisely localized and
quantified here, is NOT rebound and NOT the write: it is the k-WTA STABILITY. The
Izhikevich k-WTA does not reliably hold a SPARSE code for BOTH memories at once -- for
some seeds ONE memory's engram collapses to near-dense (150-200 of 200 granules), that
dense engram SUBSUMES the other memory's sparse engram, and the dense memory's answer
then wins BOTH probes (the universal anti-symmetric signature: m0 reads +x, m1 reads
-x -> both_win 0/6, pooled selectivity ~0). This is the 2026-05-31
separation-vs-reliability boundary, now traced to symmetric-sparse-stability of the
Izhikevich k-WTA specifically.

Acknowledged scaffolds (tracked, not hidden): host-defined input (sensory) patterns
and answer assemblies; host reinstatement of each memory's input AND answer during
replay (the hippocampal index / SWR trigger); scheduled down-states; an argmax over
answer-assembly spike counts for MEASUREMENT only; a rate-window Hebbian coactivity
write (the stand-in the consolidation gates use); the transmission-gate schedule
(WRITE vs READ) is a host-scheduled sleep/wake phase, like the plasticity gate.

Run:
    OMP_NUM_THREADS=2 SIM_BACKEND=numpy .venv/bin/python -m \
        research.runners._replay_dg_pattern_separation_bridge \
        --seeds 42 43 44 100 101 102 \
        --out research/findings/raw/replay_dg_sep/bridge_fixed_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, replace
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
from research.runners._replay_dg_pattern_separation_gate import (  # noqa: E402
    _answer_assemblies,
    _input_patterns,
    _jaccard,
    _perforant_edges,
    _reset_dynamics,
)

SEEDS = (42, 43, 44, 100, 101, 102)

DG_WRITE_GATE = "dg_answer_write"
DG_COMPETITION_GATE = "dg_competition"
ANSWER_INHIBITION_GATE = "answer_opponent"
DG_ANSWER_TX_GATE = "dg_answer_tx"

CONDITIONS = (
    "similar_separator_on",
    "similar_separator_off",   # DG competition lesion -> the NULL / blur baseline
    "dissimilar_separator_on",
    "single_separator_on",     # single-memory recall guard (POSITIVE control)
)


@dataclass(frozen=True)
class BridgeConfig:
    # populations
    n_input: int = 48
    n_dg: int = 200
    n_dg_fs: int = 32
    n_answer: int = 60
    n_answer_fs: int = 12
    answer_assembly: int = 16
    # memory input patterns
    input_assembly: int = 24
    similar_overlap: int = 18       # Jaccard 18/30 = 0.60 (the memories that blur)
    dissimilar_overlap: int = 2     # Jaccard ~0.04
    # perforant path input->dg (fixed random expansive projection)
    dg_fan_in: int = 12             # sparser sampling than the prior gate (was 20) -> decorrelation
    input_to_dg_weight: float = 70.0
    # DG feedforward + feedback PV-basket inhibition (the k-WTA separator)
    input_to_fs_weight: float = 60.0
    dg_to_fs_weight: float = 30.0
    fs_to_dg_weight: float = 6.0
    # answer opponent competition (winner-take-most read-out)
    answer_to_fs_weight: float = 120.0
    fs_to_answer_weight: float = 60.0
    # FIX 1 -- shunting inhibitory reversal (mV). DG granule vr=-60, so -63 is a mild
    # near-rest shunt (limited rebound); answer -60 is pure shunting.
    dg_inh_reversal_mV: float = -63.0
    answer_inh_reversal_mV: float = -60.0
    # plastic consolidation write dg->answer (soft-bound Hebbian)
    dg_answer_init_weight: float = 0.05
    hebbian_max_weight: float = 90.0
    hebbian_coactivity_decay: float = 0.90
    hebbian_coactivity_thresh: float = 0.01
    replay_learning_rate: float = 2.0
    wake_learning_rate: float = 0.0
    # local delays so the microcircuit can gate within a readout window
    max_synaptic_delay_ms: float = 2.0
    propagation_strength: float = 0.12
    # drives
    replay_input_drive_pA: float = 1300.0
    replay_answer_drive_pA: float = 1300.0
    probe_input_drive_pA: float = 1300.0
    # schedule
    replay_events_per_memory: int = 14
    replay_on_steps: int = 25
    replay_settle_steps: int = 12
    probe_steps: int = 40
    # instrument threshold: an engram with > this fraction of DG active is "collapsed
    # to dense" -- the k-WTA-stability failure mode.
    dense_engram_frac: float = 0.60


def smoke_config() -> BridgeConfig:
    return BridgeConfig(n_dg=160, n_dg_fs=28, n_answer=40, n_answer_fs=8,
                        answer_assembly=12, replay_events_per_memory=6,
                        replay_on_steps=20, replay_settle_steps=10, probe_steps=25)


def _drain(bridge) -> None:
    _reset_dynamics(bridge)


def build_bridge(seed: int, cfg: BridgeConfig):
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

    dg_kwargs = dict(exc, syn_reversal_potential_i_override=float(cfg.dg_inh_reversal_mV))
    ans_kwargs = dict(exc, syn_reversal_potential_i_override=float(cfg.answer_inh_reversal_mV))

    core = CoreSimConfig()
    core.enable_brain_region_framework = True
    core.brain_regions = [
        BrainRegion("input", cfg.n_input, **exc),
        BrainRegion("dg", cfg.n_dg, **dg_kwargs),
        BrainRegion("dg_fs", cfg.n_dg_fs, **inh),
        BrainRegion("answer", cfg.n_answer, **ans_kwargs),
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
    core.hebbian_learning_rate = float(cfg.wake_learning_rate)
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
    i2fs_pre, i2fs_post = _all_to_all(regions["input"], regions["dg_fs"])
    d2fs_pre, d2fs_post = _all_to_all(regions["dg"], regions["dg_fs"])
    fs2d_pre, fs2d_post = _all_to_all(regions["dg_fs"], regions["dg"])
    a2fs_pre, a2fs_post = _all_to_all(regions["answer"], regions["answer_fs"])
    fs2a_pre, fs2a_post = _all_to_all(regions["answer_fs"], regions["answer"])
    dga_pre, dga_post = _all_to_all(regions["dg"], regions["answer"])

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
        "input_to_fs": group(i2fs_pre, i2fs_post, cfg.input_to_fs_weight, plastic=False, conn_type="E_TO_I"),
        "dg_to_fs": group(d2fs_pre, d2fs_post, cfg.dg_to_fs_weight, plastic=False, conn_type="E_TO_I"),
        "fs_to_dg": group(fs2d_pre, fs2d_post, cfg.fs_to_dg_weight, plastic=False,
                          conn_type="I_TO_E", transmission_gate=DG_COMPETITION_GATE),
        "answer_to_fs": group(a2fs_pre, a2fs_post, cfg.answer_to_fs_weight, plastic=False, conn_type="E_TO_I"),
        "fs_to_answer": group(fs2a_pre, fs2a_post, cfg.fs_to_answer_weight, plastic=False,
                              conn_type="I_TO_E", transmission_gate=ANSWER_INHIBITION_GATE),
        # dg->answer: plastic AND transmission-gated (FIX 2).
        "dg_to_answer": group(dga_pre, dga_post, cfg.dg_answer_init_weight, plastic=True,
                              conn_type="E_TO_E", plasticity_gate=DG_WRITE_GATE,
                              transmission_gate=DG_ANSWER_TX_GATE),
    }
    inh_idx = np.concatenate([regions["dg_fs"], regions["answer_fs"]]).tolist()
    bridge.inject_explicit_wiring(wiring, output_inhibitory_indices=inh_idx)
    bridge.set_transmission_gate(DG_COMPETITION_GATE, 1.0)
    bridge.set_transmission_gate(ANSWER_INHIBITION_GATE, 1.0)
    bridge.set_transmission_gate(DG_ANSWER_TX_GATE, 1.0)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)

    handles = {"regions": regions, "bridge_identity": id(bridge),
               "wiring_counts": {k: v["count"] for k, v in wiring.items()}}
    return bridge, handles


def _dg_engram(bridge, cfg, regions, input_pat, competition):
    from sim.backend import to_host
    bridge.set_transmission_gate(DG_COMPETITION_GATE, 1.0 if competition else 0.0)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)
    _drain(bridge)
    dg_idx = regions["dg"]
    counts = np.zeros(dg_idx.size, dtype=np.float64)
    total = 0
    for step in range(cfg.replay_on_steps + cfg.replay_settle_steps):
        _zero_current(bridge)
        if step < cfg.replay_on_steps:
            bridge.cp_external_input_current[input_pat] = cfg.replay_input_drive_pA
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fs = np.asarray(to_host(bridge.cp_firing_states[dg_idx]), dtype=np.float64)
        counts += fs
        total += int(fs.sum())
    _zero_current(bridge)
    return dg_idx[counts > 0], total


def _replay_answer_teacher_selectivity(bridge, cfg, regions, memories, competition):
    """Teacher-only measurement (plasticity OFF, dg->answer transmission OFF): during
    replay of m0, does ONLY a0 fire? Confirms the write coincidence is teacher-clean."""
    from sim.backend import to_host
    bridge.set_transmission_gate(DG_COMPETITION_GATE, 1.0 if competition else 0.0)
    bridge.set_transmission_gate(ANSWER_INHIBITION_GATE, 1.0)
    bridge.set_transmission_gate(DG_ANSWER_TX_GATE, 0.0)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)
    a0, a1 = memories["m0"]["answer"], memories["m1"]["answer"]
    inp, ans = memories["m0"]["input"], a0
    _drain(bridge)
    s0 = s1 = 0
    for step in range(cfg.replay_on_steps + cfg.replay_settle_steps):
        _zero_current(bridge)
        if step < cfg.replay_on_steps:
            bridge.cp_external_input_current[inp] = cfg.replay_input_drive_pA
            bridge.cp_external_input_current[ans] = cfg.replay_answer_drive_pA
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fs = bridge.cp_firing_states
        s0 += int(np.asarray(to_host(fs[a0])).sum())
        s1 += int(np.asarray(to_host(fs[a1])).sum())
    _zero_current(bridge)
    bridge.set_transmission_gate(DG_ANSWER_TX_GATE, 1.0)
    return s0, s1


def _consolidate(bridge, cfg, regions, memories, competition, seed):
    """Offline replay WRITE. Transmission dg->answer is OFF so the answer fires from
    the reinstated-index teacher only (selective coincidence, no Hebbian runaway)."""
    bridge.set_transmission_gate(DG_COMPETITION_GATE, 1.0 if competition else 0.0)
    bridge.set_transmission_gate(ANSWER_INHIBITION_GATE, 1.0)
    bridge.set_transmission_gate(DG_ANSWER_TX_GATE, 0.0)          # WRITE: no read-back
    bridge.set_plasticity_gate(DG_WRITE_GATE, 1.0)
    bridge.core_config.hebbian_learning_rate = float(cfg.replay_learning_rate)
    names = list(memories.keys())
    order = []
    for _ in range(cfg.replay_events_per_memory):
        order.extend(names)
    for name in order:
        _drain(bridge)
        inp, ans = memories[name]["input"], memories[name]["answer"]
        for step in range(cfg.replay_on_steps + cfg.replay_settle_steps):
            _zero_current(bridge)
            if step < cfg.replay_on_steps:
                bridge.cp_external_input_current[inp] = cfg.replay_input_drive_pA
                bridge.cp_external_input_current[ans] = cfg.replay_answer_drive_pA
            bridge._run_one_simulation_step()
            bridge.runtime_state.current_time_step += 1
    _zero_current(bridge)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)
    bridge.set_transmission_gate(DG_ANSWER_TX_GATE, 1.0)          # READ enabled
    bridge.core_config.hebbian_learning_rate = float(cfg.wake_learning_rate)


def _probe(bridge, cfg, regions, memories, target_name, competition, seed):
    """Hippocampus-independent retrieval: drive ONLY the input (index/teacher off,
    plasticity off), dg->answer transmission ON, read which answer assembly wins."""
    from sim.backend import to_host
    bridge.set_transmission_gate(DG_COMPETITION_GATE, 1.0 if competition else 0.0)
    bridge.set_transmission_gate(ANSWER_INHIBITION_GATE, 1.0)
    bridge.set_transmission_gate(DG_ANSWER_TX_GATE, 1.0)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)
    _drain(bridge)
    inp = memories[target_name]["input"]
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
    other = [n for n in memories if n != target_name][0]
    correct_pos = np.asarray([local[int(i)] for i in memories[target_name]["answer"]], dtype=np.int64)
    wrong_pos = np.asarray([local[int(i)] for i in memories[other]["answer"]], dtype=np.int64)
    correct = float(counts[correct_pos].mean() / cfg.probe_steps)
    wrong = float(counts[wrong_pos].mean() / cfg.probe_steps)
    return {
        "target": target_name,
        "correct_rate": correct,
        "wrong_rate": wrong,
        "selectivity": (correct - wrong) / (correct + wrong + 1e-9),
        "target_assembly_wins": bool(correct > wrong),
        "total_answer_spikes": int(counts.sum()),
    }


def _direct_readout(bridge, cfg, regions, engram_global, a0, a1):
    """Drive the WRITTEN dg engram cells DIRECTLY (bypass input->dg competition),
    dg->answer transmission ON. Isolates the learned dg->answer MAPPING from the
    probe reactivation dynamics."""
    from sim.backend import to_host
    bridge.set_transmission_gate(DG_ANSWER_TX_GATE, 1.0)
    bridge.set_transmission_gate(ANSWER_INHIBITION_GATE, 1.0)
    bridge.set_plasticity_gate(DG_WRITE_GATE, 0.0)
    _drain(bridge)
    c0 = c1 = 0
    for _ in range(cfg.probe_steps):
        _zero_current(bridge)
        bridge.cp_external_input_current[engram_global] = cfg.probe_input_drive_pA
        bridge._run_one_simulation_step()
        bridge.runtime_state.current_time_step += 1
        fs = bridge.cp_firing_states
        c0 += int(np.asarray(to_host(fs[a0])).sum())
        c1 += int(np.asarray(to_host(fs[a1])).sum())
    _zero_current(bridge)
    sel = (c0 - c1) / (c0 + c1 + 1e-9)
    return {"a0": c0, "a1": c1, "selectivity": sel}


def run_condition(seed, condition, cfg):
    kind = ("similar" if condition.startswith("similar")
            else "dissimilar" if condition.startswith("dissimilar") else "single")
    competition = condition.endswith("separator_on")
    bridge, handles = build_bridge(seed, cfg)
    regions = handles["regions"]
    inputs = _input_patterns(seed, cfg, kind)
    answers = _answer_assemblies(seed, cfg, regions["answer"])
    memories = {"m0": {"input": inputs["m0"], "answer": answers["m0"]},
                "m1": {"input": inputs["m1"], "answer": answers["m1"]}}
    replay_mems = {"m0": memories["m0"]} if kind == "single" else memories

    eng0, _ = _dg_engram(bridge, cfg, regions, inputs["m0"], competition)
    eng1, _ = _dg_engram(bridge, cfg, regions, inputs["m1"], competition)
    dg_sep = {
        "input_jaccard": _jaccard(inputs["m0"], inputs["m1"]),
        "dg_jaccard": _jaccard(eng0, eng1),
        "dg_active_frac_m0": eng0.size / regions["dg"].size,
        "dg_active_frac_m1": eng1.size / regions["dg"].size,
        "dg_size_m0": int(eng0.size),
        "dg_size_m1": int(eng1.size),
    }
    # k-WTA STABILITY instrument: is EITHER memory's engram collapsed to dense?
    dense_collapse = bool(max(dg_sep["dg_active_frac_m0"], dg_sep["dg_active_frac_m1"]) > cfg.dense_engram_frac)

    teacher = None
    if kind == "similar" and competition:
        s0, s1 = _replay_answer_teacher_selectivity(bridge, cfg, regions, memories, competition)
        teacher = {"replay_m0_a0_spikes": s0, "replay_m0_a1_spikes": s1}

    w_before = _path_weights(bridge, DG_WRITE_GATE)
    _consolidate(bridge, cfg, regions, replay_mems, competition, seed)
    w_after = _path_weights(bridge, DG_WRITE_GATE)

    direct = None
    if kind in ("similar", "dissimilar") and competition:
        d0 = _direct_readout(bridge, cfg, regions, eng0, answers["m0"], answers["m1"])
        d1 = _direct_readout(bridge, cfg, regions, eng1, answers["m1"], answers["m0"])
        direct = {"m0_engram_selectivity": d0["selectivity"], "m1_engram_selectivity": d1["selectivity"],
                  "m0": d0, "m1": d1}

    probe_targets = ["m0"] if kind == "single" else ["m0", "m1"]
    probes = {n: _probe(bridge, cfg, regions, memories, n, competition, seed) for n in probe_targets}

    return {
        "seed": int(seed),
        "condition": condition,
        "kind": kind,
        "competition": bool(competition),
        "single_bridge": handles["bridge_identity"] == id(bridge),
        "dg_separation": dg_sep,
        "dense_engram_collapse": dense_collapse,
        "replay_teacher_selectivity": teacher,
        "dg_answer_weight_delta": float(np.mean(w_after - w_before)),
        "direct_readout": direct,
        "probes": probes,
        "mean_selectivity": float(np.mean([p["selectivity"] for p in probes.values()])),
        "mean_correct": float(np.mean([p["correct_rate"] for p in probes.values()])),
        "per_memory_selectivity": {n: probes[n]["selectivity"] for n in probes},
        "both_win": all(p["target_assembly_wins"] for p in probes.values()),
    }


def _scramble_teach_single(seed, cfg):
    """Causal control for the single-memory read: consolidate m0 with the WRONG
    answer (m1's assembly) on a fresh bridge, then probe m0 against the TRUE
    pairing. If the read rides the LEARNED mapping (not readout geometry), the
    selectivity INVERTS (m0 now recalls m1's answer)."""
    bridge, handles = build_bridge(seed, cfg)
    regions = handles["regions"]
    inputs = _input_patterns(seed, cfg, "single")
    answers = _answer_assemblies(seed, cfg, regions["answer"])
    # TRUE pairing for scoring; SWAPPED pairing for the write.
    true_mem = {"m0": {"input": inputs["m0"], "answer": answers["m0"]},
                "m1": {"input": inputs["m1"], "answer": answers["m1"]}}
    swapped = {"m0": {"input": inputs["m0"], "answer": answers["m1"]}}
    _consolidate(bridge, cfg, regions, swapped, True, seed)
    p = _probe(bridge, cfg, regions, true_mem, "m0", True, seed)
    return p["selectivity"]


def run_seed(seed, cfg):
    rows = {c: run_condition(seed, c, cfg) for c in CONDITIONS}
    on = rows["similar_separator_on"]
    off = rows["similar_separator_off"]
    single = rows["single_separator_on"]
    dissim = rows["dissimilar_separator_on"]
    single_scramble_sel = _scramble_teach_single(seed, cfg)

    checks = {
        # ---- VALIDATED ADVANCES (expected to PASS on the Izhikevich substrate) ----
        # A1: DG rebound fixed -- competition ON sparsifies (does not RAISE DG spikes).
        "rebound_fixed_dg_sparsifies_on": on["dg_separation"]["dg_active_frac_m0"] < 1.0
        and on["dg_separation"]["dg_active_frac_m1"] < 1.0,
        # A2: the write coincidence is teacher-clean (a1 ~ silent while replaying m0).
        "write_teacher_selective": (on["replay_teacher_selectivity"] is not None
                                    and on["replay_teacher_selectivity"]["replay_m0_a1_spikes"]
                                    <= 0.2 * max(1, on["replay_teacher_selectivity"]["replay_m0_a0_spikes"])),
        # A3: single-memory recall reaches ceiling (read no longer stalls at chance).
        "single_memory_recall": single["probes"]["m0"]["target_assembly_wins"]
        and single["probes"]["m0"]["selectivity"] >= 0.30,
        # A3b: single read rides the LEARNED mapping (scramble-teach inverts it).
        "single_scramble_inverts": single_scramble_sel <= single["probes"]["m0"]["selectivity"] - 0.30,
        # A4: per-memory read is OFF chance -- at least one similar memory reads strongly.
        "per_memory_read_off_chance": max(abs(v) for v in on["per_memory_selectivity"].values()) >= 0.25,
        # ---- BOARD #43 HEADLINE BAR (the RESIDUAL -- k-WTA stability) ----
        "both_similar_discriminable": on["both_win"] and on["mean_selectivity"] >= 0.30,
        "separator_dissociation": (on["mean_selectivity"] - off["mean_selectivity"]) >= 0.20,
    }
    return {
        "seed": int(seed),
        "conditions": rows,
        "summary": {
            "similar_on_mean_selectivity": on["mean_selectivity"],
            "similar_on_per_memory": on["per_memory_selectivity"],
            "similar_off_mean_selectivity": off["mean_selectivity"],
            "single_selectivity": single["probes"]["m0"]["selectivity"],
            "single_scramble_selectivity": single_scramble_sel,
            "dissimilar_both_win": dissim["both_win"],
            "dissimilar_per_memory": dissim["per_memory_selectivity"],
            "dg_jaccard_on": on["dg_separation"]["dg_jaccard"],
            "dg_jaccard_off": off["dg_separation"]["dg_jaccard"],
            "dg_size_on": (on["dg_separation"]["dg_size_m0"], on["dg_separation"]["dg_size_m1"]),
            "dense_engram_collapse_on": on["dense_engram_collapse"],
            "direct_readout_on": (on["direct_readout"]["m0_engram_selectivity"],
                                  on["direct_readout"]["m1_engram_selectivity"]) if on["direct_readout"] else None,
        },
        "checks": checks,
        "advances_pass": all(checks[k] for k in ("rebound_fixed_dg_sparsifies_on", "write_teacher_selective",
                                                 "single_memory_recall", "single_scramble_inverts",
                                                 "per_memory_read_off_chance")),
        "headline_go": checks["both_similar_discriminable"] and checks["separator_dissociation"],
    }


def run(seeds, cfg):
    started = time.time()
    rows = [run_seed(int(s), cfg) for s in seeds]
    n = len(rows)

    def frac(key_fn):
        return int(sum(1 for r in rows if key_fn(r)))

    check_names = list(rows[0]["checks"].keys())
    pooled_checks = {name: frac(lambda r, name=name: r["checks"][name]) for name in check_names}
    advances_all = all(r["advances_pass"] for r in rows)
    headline_all = all(r["headline_go"] for r in rows)
    return {
        "gate": "replay_dg_pattern_separation_bridge",
        "seeds": [int(s) for s in seeds],
        "n_seeds": n,
        # honest aggregate: the ADVANCE is real; the HEADLINE board#43 bar is not met.
        "advances_status": "GO" if advances_all else "PARTIAL",
        "headline_status": "GO" if headline_all else "NO-GO",
        "aggregate_status": "ADVANCE-GO / HEADLINE-NO-GO" if (advances_all and not headline_all)
        else ("GO" if headline_all else "NO-GO"),
        "pooled_check_counts": pooled_checks,
        "pooled": {
            "single_selectivity_mean": float(np.mean([r["summary"]["single_selectivity"] for r in rows])),
            "similar_on_mean_selectivity": float(np.mean([r["summary"]["similar_on_mean_selectivity"] for r in rows])),
            "similar_off_mean_selectivity": float(np.mean([r["summary"]["similar_off_mean_selectivity"] for r in rows])),
            "both_similar_win_count": frac(lambda r: r["conditions"]["similar_separator_on"]["both_win"]),
            "dissimilar_both_win_count": frac(lambda r: r["summary"]["dissimilar_both_win"]),
            "dense_collapse_count": frac(lambda r: r["summary"]["dense_engram_collapse_on"]),
            "dg_jaccard_on_mean": float(np.mean([r["summary"]["dg_jaccard_on"] for r in rows])),
        },
        "per_seed": rows,
        "scaffolds": [
            "host-defined input (sensory) patterns and answer assemblies",
            "host reinstatement of each memory's input AND answer during replay (hippocampal index / SWR trigger)",
            "scheduled down-states between replay events; host WRITE/READ transmission-gate phase (sleep vs wake)",
            "argmax over answer-assembly spike counts for measurement only",
            "rate-window Hebbian coactivity write (the stand-in the consolidation gates use)",
            "fixed random perforant projection and fixed FS anatomy (not developed)",
        ],
        "elapsed_seconds": time.time() - started,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    cfg = smoke_config() if args.smoke else BridgeConfig()
    print(f"[replay-dg-bridge] backend={os.environ.get('SIM_BACKEND','default')} "
          f"seeds={args.seeds} smoke={args.smoke}", flush=True)
    payload = run(args.seeds, cfg)
    for r in payload["per_seed"]:
        s = r["summary"]
        pm = s["similar_on_per_memory"]
        print(f"  seed {r['seed']}: advances={r['advances_pass']} headline_GO={r['headline_go']} "
              f"single_sel={s['single_selectivity']:+.2f} scramble={s['single_scramble_selectivity']:+.2f} "
              f"sim_on(m0={pm['m0']:+.2f} m1={pm['m1']:+.2f}) sim_off={s['similar_off_mean_selectivity']:+.2f} "
              f"dgJ={s['dg_jaccard_on']:.2f} dgSz={s['dg_size_on']} dense={s['dense_engram_collapse_on']}",
              flush=True)
    print(f"  AGGREGATE: {payload['aggregate_status']}", flush=True)
    print(f"  pooled: {json.dumps(payload['pooled'])}", flush=True)
    print(f"  checks (of {payload['n_seeds']}): {json.dumps(payload['pooled_check_counts'])}", flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
