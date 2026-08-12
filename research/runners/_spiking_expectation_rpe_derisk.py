"""De-risk a GENUINE SPIKING expectation / prediction-error (surprise) faculty that is
READABLE AT CONVERSATION TIME — the owner's "understanding of consequences / expectation."

THE FACULTY (conversational, NOT reward-magnitude)
--------------------------------------------------
When an incoming assertion VIOLATES the brain's stored expectation — the owner says
"the dog eats grass" while the brain holds "(dog, eats) -> meat", or asserts a novel /
contradictory fact — a genuine SPIKING surprise signal should FIRE, readable at
conversation time, so the brain can NOTICE ("that's not what I expected") and gate
learning by surprise. This is expectation-violation as a SPIKING signal, NOT a host
string comparison of the recalled vs asserted patient.

WHY THIS IS A REAL GAP (the honest boundary this de-risk sits at)
----------------------------------------------------------------
The project ALREADY has a genuine spiking reward-prediction-error: the limbic core / SNc
fires delta = r - V entirely in spikes (2026-06-18-limbic-core-rpe-battery-GO), and the
multicue learner replaced its host `err` with SNc firing (2026-06-19). But EVERY existing
spiking RPE is over a SCALAR REWARD MAGNITUDE (Schultz delta). NONE reads the brain's
stored (agent,action)->patient association, recalls the EXPECTED patient, and fires on a
SEMANTIC CONTENT contradiction. That comparison, done today in the conversational pipeline,
would be a host `recalled_patient == asserted_patient` compare in Python — a SHORTCUT under
the brain-based-only bar. This runner de-risks the genuinely SPIKING replacement.

THE MECHANISM UNDER TEST — a spiking predictive-coding MISMATCH (error) unit
----------------------------------------------------------------------------
Predictive coding (Rao & Ballard 1999; Bastos et al. 2012): an error unit fires the part
of the FEED-FORWARD input NOT explained by the TOP-DOWN prediction, i.e. error =
[actual - prediction]_+, with the prediction delivered as SUBTRACTIVE inhibition by an
interneuron. Realized here in the patient-concept space (the direct analogue of the SNc's
delta = r - V, but over CONTENT rather than scalar reward):

    cue (agent,action)  --PLASTIC (Hebbian co-fire, topographic)-->  patient_expected (FS,
      the recalled expectation: "(dog,eats)" recalls "meat")           PV-like interneuron;
                                                              |  GABA_A perisomatic:  the PREDICTION
                                                              |  the subtractive prediction
                                                              v
    patient_asserted  --EXC (topographic concept c -> block c)-->  surprise  (RS pyramidal)
      (the incoming asserted patient, delivered as sensory drive          the ERROR / SURPRISE
       — the legitimate teacher/environment boundary)                     unit; its FIRING RATE
                                                                          IS the surprise signal

The cue->patient_expected mapping is TOPOGRAPHIC (cue block i -> prediction block i) with the
association STRENGTH learned by Hebbian co-fire (untrained -> weak recall). A fully-LEARNED
all-to-all mapping (where untrained/permute would be decisive) needs the CA3 pattern-separation
/ competition companion process (2026-06-05-D-cue-recall-RESOLVED) — the characterized next rung.

Topographic error-unit wiring (the standard PC alignment): concept c's ASSERTED excitation
and concept c's EXPECTED (predicted) inhibition BOTH target surprise block c. So:
  - CONFIRM  (assert == expected): block c gets excitation AND matching inhibition -> cancel
             -> surprise ~ 0.
  - CONTRADICT (assert = j != expected i): block j is excited (asserted) but NOT inhibited
             (the prediction inhibits block i) -> surprise block j FIRES -> high surprise.
  - NOVEL    (assert an out-of-repertoire patient, no cue recalls it): un-inhibited -> FIRES.
The surprise pool's TOTAL windowed rate is the conversation-time readout ("am I surprised?").

WHAT IS NEURAL vs THE LEGITIMATE BOUNDARY
-----------------------------------------
- The EXPECTATION is neural + LEARNED: cue->patient_expected is Hebbian co-fire (the
  validated sparse heteroassociative memory, 2026-06-05-D-cue-recall-RESOLVED). Which
  patient is expected is RECALLED by firing, not a host lookup.
- The MISMATCH is neural: surprise = asserted excitation - expected GABA_A inhibition,
  computed at the surprise membrane; the signal is a cp_firing_states READ, never a host
  subtraction of the codes.
- The legitimate host boundary: the gold (agent,action,patient) tokens delivered as sensory
  DRIVE (exactly as the nav reward_us rides on the perceived reward; the environment renders
  the input the brain then processes).

GO-GATE (pre-registered)
------------------------
 (1) SEPARATION: surprise(contradict) >= 3x surprise(confirm) AND surprise(novel) >= 3x
     surprise(confirm) AND surprise(contradict) >= 5 Hz (a real signal, not noise), 6-seed
     >= 5/6.
 (2) LESION-PREDICTOR (load-bearing, decisive): zero the patient_expected->surprise edges ->
     no prediction inhibition -> surprise fires HIGH on CONFIRM too -> the contradict/confirm
     separation COLLAPSES (ratio <= 1.5) AND confirm RISES to the contradict level. Proves the
     spiking prediction is load-bearing (not a fixed input-driven artifact).
 (3) UNTRAINED (learning contributes): skip Hebbian encoding -> weaker recall -> confirm rises.
     At a low structural prior this collapses the separation; at the robust operating point the
     TOPOGRAPHIC PRIOR itself predicts, so untrained stays low (the honest scope: the MAPPING is
     structural here, the STRENGTH learned).
 (4) BRAIN-BASED: the surprise = cp_firing_states[surprise] READ; current_reward_signal == 0
     and no host compare of the asserted vs expected codes produces the signal.
 (5) READS AT CONVERSATION TIME: the signal is a windowed rate available WITHIN the assertion
     window (measured live, no offline decode).

WALL DISCIPLINE (the companion process): a predictive-coding error unit's operating point is
the GAIN MATCH between the asserted excitation and the recalled-prediction inhibition — i.e.
PRECISION / divisive normalization, set biologically by inhibitory gain control (PV/SST) and
neuromodulation (NE/ACh). If we proxy that precision with a fixed constant, the separation is
brittle (weak recall -> false surprise on confirm; over-broad inhibition -> misses
contradictions). This runner MEASURES the operating point (recall rate + surprise f-I) and
places it with headroom BOTH ways; if a fixed gain is not robust 6-seed, the honest boundary
is "needs a homeostatic precision companion", reported as the deliverable.

CPU-friendly (~600-neuron bridge). Run under SIM_BACKEND=numpy for a deterministic regime.

Usage
-----
    SIM_BACKEND=numpy python -m research.runners._spiking_expectation_rpe_derisk \
        --seeds 42,43,44,100,101,102 --out research/findings/raw/_spiking_expectation_rpe_6seed.json
    SIM_BACKEND=numpy python -m research.runners._spiking_expectation_rpe_derisk --opsearch
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as _st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _host(a):
    from sim.backend import to_host
    try:
        return to_host(a)
    except Exception:
        return a


def build_expectation_circuit(seed, *, n_trained=5, n_novel=2, blk=24, cue_blk=24,
                              cue_to_expected_weight=0.8, asserted_to_surprise_weight=5.0,
                              expected_to_surprise_weight=14.0, gabab_prop=0.22,
                              gabab_tau_decay=150.0, hebbian_learning_rate=0.06,
                              hebbian_max_weight=45.0, enable_heterogeneity=False):
    """Build cue -> patient_expected(FS, GABA_A) -> surprise <- patient_asserted(exc).

    cue->patient_expected is TOPOGRAPHIC + PLASTIC (Hebbian co-fire strengthens the recall).
    patient_asserted->surprise (exc) and patient_expected->surprise (inh, GABA_A) are FIXED and
    TOPOGRAPHIC (concept c -> surprise block c), installed block-diagonal after build. The
    surprise pool's firing IS the mismatch/surprise signal.
    """
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType

    n_concepts = n_trained + n_novel
    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    # The expectation is LEARNED by Hebbian co-fire (the sparse heteroassoc pattern).
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = float(hebbian_learning_rate)
    cfg.hebbian_min_weight = 0.0            # non-co-fired edges must be free to stay ~0 (selectivity)
    cfg.hebbian_max_weight = float(hebbian_max_weight)  # above the assoc working range (soft-bound gotcha)
    cfg.hebbian_weight_decay = 0.0
    # The COMPETITION / normalization companion process (the wall-discipline answer): a plain
    # Hebbian rule has an input-INDEPENDENT fixed point (w -> w_max for every co-fired synapse),
    # so cue_i's afferents to a prediction neuron ALL run away -> non-selective recall. The
    # rate-window Hebbian + Miller-MacKay SUBTRACTIVE normalization (sum_j dw_ij = 0 per post
    # neuron) makes afferents COMPETE -> only the strongest (truly co-fired) association survives
    # -> selective recall. This is the built-in engine mechanism for the CA3 autoassociator.
    cfg.hebbian_rate_window = True
    cfg.hebbian_coactivity_decay = 0.85
    cfg.hebbian_coactivity_thresh = 0.20
    cfg.hebbian_mean_subtract = 1.0
    cfg.enable_reward_modulation = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_parameter_heterogeneity = bool(enable_heterogeneity)
    # Deterministic regime (the navfaithful / limbic read protocol): OU background + channel
    # noise OFF so the surprise operating point is CONTROLLABLE (silent at rest; driven only by
    # the asserted excitation vs the predicted inhibition). Noise-on is a separate robustness test.
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.current_reward_signal = 0.0        # BRAIN-BASED: no host reward scalar anywhere
    cfg.reward_baseline = 0.0

    # GABA_B/GIRK slow K+ subtractive inhibition (the prediction; the already-shipped edit).
    cfg.enable_gabab = True
    cfg.gabab_reversal_potential = -90.0
    cfg.gabab_tau_decay = float(gabab_tau_decay)
    cfg.gabab_propagation_strength = float(gabab_prop)
    cfg.gabab_conductance_max = 0.0

    RS = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
    # The PREDICTION pool: an FS (PV-like) inhibitory interneuron delivering fast SUBTRACTIVE
    # (perisomatic) inhibition — the biologically-faithful predictive-coding prediction carrier.
    # Sustained across the assertion window because the cue keeps the recall firing. NOTE
    # (verified 2026-08-12 on this substrate): FS + GABA_A inhibits cleanly (drive 24 -> 0 Hz),
    # and FS has a LOW rheobase so the learned recall fires it robustly. An FS + GABA_B combo
    # produced a WRONG-SIGN (net excitatory) effect here, and MSN-D1's high rheobase left the
    # learned recall near-silent (2 Hz) -> no effective prediction. FS + GABA_A is the choice.
    cfg.brain_regions = [
        BrainRegion(name="cue", n_neurons=n_trained * cue_blk, exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False, izh_neuron_type=RS),
        # The PREDICTION pool (inhibitory FS; delivers the subtractive GABA_A prediction).
        BrainRegion(name="patient_expected", n_neurons=n_concepts * blk, exc_fraction=0.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
                    syn_reversal_potential_i_override=-70.0),
        BrainRegion(name="patient_asserted", n_neurons=n_concepts * blk, exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False, izh_neuron_type=RS),
        # The ERROR / SURPRISE unit; its total firing rate IS the surprise signal.
        BrainRegion(name="surprise", n_neurons=n_concepts * blk, exc_fraction=1.0,
                    internal_density=0.0, exc_weight_mean=0.0, inh_weight_mean=0.0,
                    weight_jitter=0.0, plastic_internal=False, izh_neuron_type=RS),
    ]
    cfg.region_pathways = [
        # The LEARNED prediction: cue (state) -> patient_expected. PLASTIC, all-to-all so
        # Hebbian co-fire SELECTS cue_i -> patient_i (others stay ~0).
        RegionPathway(from_region="cue", to_region="patient_expected",
                      density=1.0, weight_mean=float(cue_to_expected_weight),
                      weight_jitter=0.0, plastic=True),
        # The asserted feed-forward drive: patient_asserted -> surprise (exc). Built full,
        # masked block-diagonal after build (concept c -> surprise block c).
        RegionPathway(from_region="patient_asserted", to_region="surprise",
                      density=1.0, weight_mean=float(asserted_to_surprise_weight),
                      weight_jitter=0.0, plastic=False),
        # The subtractive prediction: patient_expected -> surprise (inh, GABA_A). Built full,
        # masked block-diagonal after build.
        RegionPathway(from_region="patient_expected", to_region="surprise",
                      density=1.0, weight_mean=float(expected_to_surprise_weight),
                      weight_jitter=0.0, plastic=False),
    ]
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)

    meta = dict(n_trained=n_trained, n_novel=n_novel, n_concepts=n_concepts, blk=blk,
                cue_blk=cue_blk, W_exc=float(asserted_to_surprise_weight),
                W_inh=float(expected_to_surprise_weight))
    _install_block_diagonal(bridge, "patient_asserted", "surprise", blk,
                            float(asserted_to_surprise_weight))
    _install_block_diagonal(bridge, "patient_expected", "surprise", blk,
                            float(expected_to_surprise_weight))
    # cue -> patient_expected is TOPOGRAPHIC (cue block i -> prediction block i) and PLASTIC:
    # the association STRENGTH is learned by Hebbian co-fire (untrained -> no recall), but the
    # WHICH-patient mapping is a topographic prior (selective by construction, sidestepping the
    # separate CA3 pattern-separation / competition problem that a fully-learned ALL-TO-ALL
    # mapping needs — 2026-06-05-D-cue-recall-RESOLVED; that is the characterized next rung).
    _install_block_diagonal(bridge, "cue", "patient_expected", blk,
                            float(cue_to_expected_weight))
    # Snapshot the resting state (v=vr, u=0, all conductances/firing 0) for hard resets between
    # trials — a 20-step settle cannot fully quiesce a 500 Hz FS pool, so residual firing of one
    # fact's prediction pool leaks into the next fact's Hebbian window (a recency contamination
    # that made recall non-selective). A hard reset removes it.
    bridge._rest_v = bridge.cp_membrane_potential_v.copy()
    bridge._rest_u = bridge.cp_recovery_variable_u.copy()
    return bridge, cfg, meta


def _hard_reset(bridge):
    """Restore the network to its resting snapshot: membrane/recovery to rest, all conductances
    + firing + refractory + external current to zero. Removes cross-trial state carryover."""
    bridge.cp_membrane_potential_v[:] = bridge._rest_v
    bridge.cp_recovery_variable_u[:] = bridge._rest_u
    for name in ("cp_conductance_g_e", "cp_conductance_g_i", "cp_conductance_g_gabab",
                 "cp_conductance_g_nmda", "cp_firing_states", "cp_refractory"):
        arr = getattr(bridge, name, None)
        if arr is not None:
            arr[:] = 0
    bridge.cp_external_input_current[:] = 0.0


def _idx(bridge, name):
    import numpy as np
    return np.asarray(bridge.region_manager.indices(name), dtype=np.int64)


def _install_block_diagonal(bridge, src, dst, blk, weight):
    """Make src->dst TOPOGRAPHIC: keep only same-concept edges (concept c of src -> concept c
    of dst), zero all cross-concept edges. Operates directly on the CSR weight matrix so it is
    robust to the pre/post orientation convention (determined empirically here)."""
    import numpy as np
    src_idx = set(int(i) for i in _idx(bridge, src))
    dst_idx = set(int(i) for i in _idx(bridge, dst))
    src_base = min(src_idx); dst_base = min(dst_idx)
    M = bridge.cp_connections.tocsr()
    indptr = np.asarray(_host(M.indptr)); indices = np.asarray(_host(M.indices))
    data = np.asarray(_host(M.data)).astype(np.float32)
    n_rows = M.shape[0]
    # Determine orientation: CSR row is post or pre? Count edges under each hypothesis.
    row_is_dst = 0; row_is_src = 0
    for r in range(n_rows):
        r_in_dst = r in dst_idx; r_in_src = r in src_idx
        if not (r_in_dst or r_in_src):
            continue
        for off in range(int(indptr[r]), int(indptr[r + 1])):
            c = int(indices[off])
            if r_in_dst and c in src_idx:
                row_is_dst += 1
            if r_in_src and c in dst_idx:
                row_is_src += 1
    row_is_post = row_is_dst >= row_is_src   # standard W[post,pre]: row=post(dst), col=pre(src)
    n_kept = n_zeroed = 0
    for r in range(n_rows):
        for off in range(int(indptr[r]), int(indptr[r + 1])):
            c = int(indices[off])
            if row_is_post:
                post, pre = r, c
            else:
                post, pre = c, r
            if pre in src_idx and post in dst_idx:
                src_concept = (pre - src_base) // blk
                dst_concept = (post - dst_base) // blk
                if src_concept == dst_concept:
                    data[off] = float(weight); n_kept += 1
                else:
                    data[off] = 0.0; n_zeroed += 1
    import scipy.sparse as sp
    newM = sp.csr_matrix((data, indices, indptr), shape=M.shape)
    bridge.cp_connections = newM
    return n_kept, n_zeroed


def _step(bridge):
    bridge._run_one_simulation_step()
    bridge.runtime_state.current_time_step += 1
    bridge.runtime_state.current_time_ms = (
        bridge.runtime_state.current_time_step * bridge.core_config.dt_ms)


def _settle(bridge, n_steps=80):
    """Clean-reset inter-trial gap (the nav `_n9_reset_critic_read_state` protocol): zero all
    external current + the slow GABA_B conductance so a prior window's subtraction does not
    carry into the next measurement (the order artifact)."""
    bridge.cp_external_input_current[:] = 0.0
    if getattr(bridge, "cp_conductance_g_gabab", None) is not None:
        bridge.cp_conductance_g_gabab[:] = 0.0
    for _ in range(n_steps):
        _step(bridge)


def _set_drives(bridge, idx_map, drives, xp):
    bridge.cp_external_input_current[:] = 0.0
    for region, (concept, pA) in drives.items():
        idx = idx_map[region]
        if concept is None:
            bridge.cp_external_input_current[idx] = xp.float32(pA)
        else:
            blk = bridge._blk
            sub = idx[concept * blk:(concept + 1) * blk]
            bridge.cp_external_input_current[sub] = xp.float32(pA)


def _drive_read(bridge, idx_map, drives, n_steps, xp, read_regions, pre_drives=None,
                pre_steps=0):
    """Optionally run a PREDICTION phase (pre_drives, pre_steps: establish the top-down
    expectation so the slow GABA_B is already present), then the measured phase (drives,
    n_steps). Predictive coding / mismatch-negativity: the expectation precedes the deviant.
    Returns {region: rate_hz} measured in the second phase only. concept=None drives the whole
    region; else only that concept's block."""
    if pre_drives is not None and pre_steps > 0:
        _set_drives(bridge, idx_map, pre_drives, xp)
        for _ in range(pre_steps):
            _step(bridge)
    _set_drives(bridge, idx_map, drives, xp)
    counts = {r: 0 for r in read_regions}
    for _ in range(n_steps):
        _step(bridge)
        fs = bridge.cp_firing_states
        for r in read_regions:
            counts[r] += int(fs[idx_map[r]].sum())
    dur_s = n_steps * 1e-3
    return {r: counts[r] / max(len(_host(idx_map[r])), 1) / dur_s for r in read_regions}


def train_expectation(bridge, cfg, idx_map, meta, xp, *, n_reps=12, cue_pa=600.0,
                      teach_pa=600.0, hold=40, permute=False):
    """Encode cue_i -> patient_i (Hebbian co-fire). permute -> cue_i -> patient_pi(i)."""
    n_trained = meta["n_trained"]
    targets = list(range(n_trained))
    if permute:
        targets = [(i + 1) % n_trained for i in range(n_trained)]  # a fixed derangement
    for _ in range(n_reps):
        for i in range(n_trained):
            _hard_reset(bridge)
            _drive_read(bridge, idx_map,
                        {"cue": (i, cue_pa), "patient_expected": (targets[i], teach_pa)},
                        hold, xp, [])
    return targets


def measure_conditions(bridge, cfg, idx_map, meta, xp, *, cue_pa=600.0, assert_pa=600.0,
                       hold=60, pre_steps=60, targets=None):
    """For each trained fact i, measure surprise rate under CONFIRM / CONTRADICT / NOVEL.
    Each condition: (1) a PREDICTION phase (cue alone -> recall the expected patient -> the
    slow GABA_B prediction settles onto surprise), then (2) the ASSERTION phase (cue + asserted
    patient) where the surprise rate is read. The expectation precedes the deviant (predictive
    coding / mismatch-negativity). Returns per-condition mean surprise (Hz) + the recall rate of
    patient_expected during the prediction phase."""
    n_trained = meta["n_trained"]; n_novel = meta["n_novel"]
    cfg.enable_hebbian_learning = False   # FREEZE learning during the read
    confirm, contra, novel, recall = [], [], [], []
    for i in range(n_trained):
        j = (i + 1) % n_trained                      # another STORED patient (contradiction)
        nov = n_trained + (i % max(n_novel, 1))      # an out-of-repertoire patient (novel)
        cue_only = {"cue": (i, cue_pa)}
        # CONFIRM: assert the TRUE patient i (what a faithful listener stored for fact i).
        _hard_reset(bridge)
        r = _drive_read(bridge, idx_map,
                        {"cue": (i, cue_pa), "patient_asserted": (i, assert_pa)},
                        hold, xp, ["surprise", "patient_expected"],
                        pre_drives=cue_only, pre_steps=pre_steps)
        confirm.append(r["surprise"]); recall.append(r["patient_expected"])
        # CONTRADICT: assert a different stored patient j.
        _hard_reset(bridge)
        r = _drive_read(bridge, idx_map,
                        {"cue": (i, cue_pa), "patient_asserted": (j, assert_pa)},
                        hold, xp, ["surprise"], pre_drives=cue_only, pre_steps=pre_steps)
        contra.append(r["surprise"])
        # NOVEL: assert an out-of-repertoire patient.
        _hard_reset(bridge)
        r = _drive_read(bridge, idx_map,
                        {"cue": (i, cue_pa), "patient_asserted": (nov, assert_pa)},
                        hold, xp, ["surprise"], pre_drives=cue_only, pre_steps=pre_steps)
        novel.append(r["surprise"])
    return {"confirm_hz": _st.mean(confirm), "contradict_hz": _st.mean(contra),
            "novel_hz": _st.mean(novel), "recall_hz": _st.mean(recall),
            "confirm_per": confirm, "contradict_per": contra, "novel_per": novel}


def _lesion_prediction(bridge, meta):
    """Anti-cheat: zero the patient_expected->surprise edges (the prediction pathway) -> NO
    prediction inhibition -> surprise must then fire HIGH on confirm too (the separation
    collapses). Weight-based so it is receptor-agnostic."""
    import numpy as np
    return _install_block_diagonal(bridge, "patient_expected", "surprise", meta["blk"], 0.0)


def run_seed(seed, *, mode="intact", verbose=True, n_reps=20, **build_kw):
    """mode: intact | lesion | untrained."""
    from sim.backend import get_backend
    xp, _ = get_backend()
    bridge, cfg, meta = build_expectation_circuit(seed, **build_kw)
    bridge._blk = meta["blk"]
    regions = ("cue", "patient_expected", "patient_asserted", "surprise")
    idx_map = {n: xp.asarray(_idx(bridge, n)) for n in regions}

    targets = None
    if mode != "untrained":
        targets = train_expectation(bridge, cfg, idx_map, meta, xp, n_reps=n_reps)
    if mode == "lesion":
        nk, nz = _lesion_prediction(bridge, meta)
        if verbose:
            print(f"  [lesion] zeroed the patient_expected->surprise prediction pathway "
                  f"({nk} same-concept edges set to 0)")

    # BRAIN-BASED anti-cheat: no host reward/comparison scalar anywhere in the loop. The surprise
    # signal is read ONLY from cp_firing_states[surprise] (see measure_conditions); nothing here
    # subtracts the asserted vs expected codes in Python.
    assert float(cfg.current_reward_signal) == 0.0, "brain-based violated: host reward scalar set"

    res = measure_conditions(bridge, cfg, idx_map, meta, xp, targets=targets)
    conf = max(res["confirm_hz"], 1e-6)
    res["contradict_ratio"] = res["contradict_hz"] / conf
    res["novel_ratio"] = res["novel_hz"] / conf
    res["mode"] = mode; res["seed"] = seed
    # GO gate (intact): contradict & novel each >= 3x confirm AND contradict is a real signal.
    res["go"] = bool(res["contradict_ratio"] >= 3.0 and res["novel_ratio"] >= 3.0
                     and res["contradict_hz"] >= 5.0)
    if verbose:
        print(f"  [{mode:9s} seed {seed}] recall(expected)={res['recall_hz']:5.1f}Hz | "
              f"surprise: confirm={res['confirm_hz']:5.2f}  contradict={res['contradict_hz']:5.2f} "
              f"({res['contradict_ratio']:4.1f}x)  novel={res['novel_hz']:5.2f} "
              f"({res['novel_ratio']:4.1f}x) | GO={res['go']}")
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--n-reps", type=int, default=22)
    ap.add_argument("--cue-to-expected-weight", type=float, default=0.8,
                    help="prediction gain (recall strength). 0.8 = robust 6/6 GO operating point; "
                         "0.4 = low-prior, learning-sensitive (3/6, the precision boundary).")
    ap.add_argument("--asserted-to-surprise-weight", type=float, default=5.0)
    ap.add_argument("--expected-to-surprise-weight", type=float, default=14.0)
    ap.add_argument("--gabab-prop", type=float, default=0.22)
    ap.add_argument("--hebbian-learning-rate", type=float, default=0.06)
    ap.add_argument("--opsearch", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    build_kw = dict(cue_to_expected_weight=args.cue_to_expected_weight,
                    asserted_to_surprise_weight=args.asserted_to_surprise_weight,
                    expected_to_surprise_weight=args.expected_to_surprise_weight,
                    gabab_prop=args.gabab_prop,
                    hebbian_learning_rate=args.hebbian_learning_rate)

    if args.opsearch:
        print("[expectation-RPE OPSEARCH seed=42] asserted->surprise(exc) x exp->surprise(gaba_b) x gabab_prop")
        for xw in (0.4, 0.8, 1.5, 3.0):
            for ew in (6.0, 12.0, 20.0):
                for gp in (0.22, 0.35):
                    bk = dict(build_kw); bk.update(asserted_to_surprise_weight=xw,
                                                   expected_to_surprise_weight=ew, gabab_prop=gp)
                    r = run_seed(42, verbose=False, n_reps=args.n_reps, **bk)
                    print(f"  xw={xw:.1f} ew={ew:4.1f} gp={gp:.2f} | recall={r['recall_hz']:5.1f} "
                          f"conf={r['confirm_hz']:6.2f} contra={r['contradict_hz']:6.2f} "
                          f"({r['contradict_ratio']:5.1f}x) nov={r['novel_hz']:6.2f} "
                          f"({r['novel_ratio']:5.1f}x) GO={r['go']}")
        return

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    print("=== INTACT (the spiking surprise signal) ===")
    intact = [run_seed(s, mode="intact", n_reps=args.n_reps, **build_kw) for s in seeds]

    les_seeds = seeds[:3]
    print("\n=== ANTI-CHEATS (mechanistic; 3 seeds) ===")
    lesion = [run_seed(s, mode="lesion", n_reps=args.n_reps, **build_kw) for s in les_seeds]
    untrained = [run_seed(s, mode="untrained", n_reps=args.n_reps, **build_kw) for s in les_seeds]

    n_go = sum(1 for r in intact if r["go"])
    verdict = "GO" if (len(intact) >= 6 and n_go >= 5) or (len(intact) < 6 and n_go == len(intact)) else "BOUNDARY"
    print(f"\n=== VERDICT ===")
    print(f"  INTACT GO: {n_go}/{len(intact)} seeds (>= 5/6 required)  ->  {verdict}")
    # (1) LESION (decisive): removing the prediction collapses the separation (ratio -> ~1) AND
    #     confirm RISES to the contradict level (the surprise on a CONFIRMED assertion is the
    #     part the prediction cancelled). (2) UNTRAINED: no learned recall -> confirm RISES vs
    #     intact (the LEARNED association strength is load-bearing).
    les_collapse = sum(1 for r in lesion if r["contradict_ratio"] <= 1.5)
    intact_conf = _st.mean([r["confirm_hz"] for r in intact[:3]]) if intact else 0.0
    les_conf = _st.mean([r["confirm_hz"] for r in lesion]) if lesion else 0.0
    unt_conf = _st.mean([r["confirm_hz"] for r in untrained]) if untrained else 0.0
    unt_rise = sum(1 for r in untrained if r["confirm_hz"] >= 2.0 * max(intact_conf, 1e-6))
    print(f"  lesion-predictor collapses separation:   {les_collapse}/{len(lesion)} (ratio<=1.5); "
          f"confirm {intact_conf:.2f} -> {les_conf:.2f} Hz (prediction gone -> confirm fires)")
    print(f"  untrained (no learned recall) raises conf: {unt_rise}/{len(untrained)} "
          f"(intact confirm {intact_conf:.2f} -> untrained {unt_conf:.2f} Hz)")

    # ATTRIBUTION (tools.lab): whose is the confirm/contradict SEPARATION? Treatment = the intact
    # separation (contradict-confirm), control = the lesioned separation. lesion ~0 -> the whole
    # separation is owned by the SPIKING PREDICTION, not a fixed input artifact.
    from tools.lab import attributable_to
    from tools.verdict import Verdict
    intact_sep = _st.mean([r["contradict_hz"] - r["confirm_hz"] for r in intact[:len(les_seeds)]])
    lesion_sep = _st.mean([r["contradict_hz"] - r["confirm_hz"] for r in lesion])
    frac = attributable_to("surprise separation @ spiking prediction", intact_sep, lesion_sep)

    # VERDICT with carried PRECONDITIONS (tools.verdict): a GO must travel with what earned it.
    contra_min = min(r["contradict_hz"] for r in intact)
    ratio_min = min(min(r["contradict_ratio"], r["novel_ratio"]) for r in intact)
    v = (Verdict("spiking expectation-violation (surprise) — mismatch mechanism")
         .require("intact GO on >=5/6 seeds", n_go, expect=lambda k: k >= max(5, len(intact) - 1)
                  if len(intact) >= 6 else k == len(intact))
         .floor("contradict is a real signal (min Hz)", contra_min, floor=5.0)
         .require("separation ratio >= 3x (min over seeds)", ratio_min, expect=lambda x: x >= 3.0)
         .control("prediction lesion changes the separation", intact_sep, lesion_sep, min_separation=2.0)
         .require("lesion collapses ratio to ~1 (3/3)", les_collapse, expect=lambda k: k == len(lesion))
         .reaches("lesion raises the confirm-surprise", intact_conf, les_conf)
         .require("separation attributable to prediction (>=0.8)", frac,
                  expect=lambda x: x is not None and x >= 0.8)
         .disabled("OU background process", "deterministic regime for a controllable operating point")
         .disabled("conductance noise", "deterministic regime")
         .disabled("fully-LEARNED all-to-all mapping",
                   "the which-patient mapping is a TOPOGRAPHIC prior; strength is Hebbian-learned "
                   "(untrained collapses only at low prior). Fully-learned recall = the next rung."))
    decided = v.decide(go=(verdict == "GO"))

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"mode": "spiking_expectation_rpe", "intact": intact,
                       "lesion": lesion, "untrained": untrained,
                       "n_go": n_go, "n_seeds": len(intact),
                       # The EARNED verdict (GO/NO-GO/UNDEFINED) from the carried preconditions.
                       # A run whose GO preconditions fail (e.g. the low-prior precision point) is
                       # UNDEFINED, never a negative — the human-readable label is separate.
                       "verdict": decided["status"], "verdict_label": verdict,
                       "intact_confirm_hz": intact_conf, "lesion_confirm_hz": les_conf,
                       "untrained_confirm_hz": unt_conf, "lesion_collapse": les_collapse,
                       "intact_separation_hz": intact_sep, "lesion_separation_hz": lesion_sep,
                       "separation_attributable_to_prediction": frac,
                       "preconditions": decided["preconditions"],
                       "disabled_processes": decided["disabled_processes"],
                       "verdict_status": decided["status"]}, f, indent=2)
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
