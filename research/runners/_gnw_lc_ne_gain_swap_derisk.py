"""GNW LC-NE PHASIC-GAIN on the neural thought-swap EVICTION — WAVE-1 one-brain integration de-risk.

WHAT THIS TESTS. The GNW thought-swap (board #77/#85, `2026-08-19-gnw-neural-swap-intention-GO.md`, wired to
production in `webapp/gnw_thought_swap.py`) already EVICTS the incumbent coalition neurally: a spiking
mismatch/salience detector (mm) fires on a salient topic-change, and its rate sets a fixed-gain readout
(`eff_boost = min(MAX_BOOST, BOOST_GAIN * mm_rate)`) that raises the release probability U on the incumbent's OWN
recurrent E->E loop (Mongillo-Barak-Tsodyks short-term depression, `MultiLoopSTD`) — the incumbent's own sustained
firing then depletes its loop below the sustain knee and it SELF-EVICTS (an emergent one-thought-at-a-time,
winner-take-all workspace: `n_ignited` never exceeds 1; the "suppression" of the loser IS this depression-driven
collapse). `BOOST_GAIN` has been a FIXED host constant (1.0) since that finding. This runner does not rebuild any
of that: `build()`/`MultiLoopSTD`/the vacancy gate/the mismatch-pred detector are reused UNCHANGED (imported, not
copied) exactly as `_gnw_neural_swap_intention_derisk.py` and `webapp/gnw_thought_swap.py` already do. The ONE
new mechanism is a locus-coeruleus-like population (`lc`) whose graded, lesionable spiking rate REPLACES that
fixed constant with an adaptive gain — i.e. the existing eviction/suppression substrate is REUSED as the
eviction effector; only the GAIN feeding it is now neuromodulator-set rather than hard-coded.

THE MECHANISM (Bouret & Sara 2005; Aston-Jones & Cohen 2005):
  * Bouret, C. & Sara, S.J. (2005), Trends Neurosci 28(11):574-582, "Network reset: a simplified overarching
    theory of LC-NE function" — a phasic LC-NE burst, triggered by a behaviorally significant (salient/novel)
    event, interrupts the current network configuration and permits a NEW one to form — i.e. NE bursts AT THE
    MOMENT a switch is warranted. Realized here as a NEW synaptic pathway, mm_ALL -> lc (dense, E_TO_E): lc
    receives convergent phasic drive from the SAME mismatch/salience population whose firing already triggers the
    swap, so LC bursts precisely when (and because) a salient mismatch is being detected.
  * Aston-Jones, G. & Cohen, J.D. (2005), Annu Rev Neurosci 28:403-450, "An integrative theory of locus coeruleus-
    norepinephrine function: adaptive gain and optimal performance" — NE does not carry new information; it sets
    the GAIN on the circuit that is already processing a signal (a multiplicative/adaptive readout, not a new
    input). Realized here exactly at the ALREADY-EXISTING gain slot: `BOOST_GAIN` in the base finding is a fixed
    multiplier on mm's rate; this runner replaces it with `boost_gain_eff = GAIN_FLOOR + NE_GAIN_SPAN * ne_level`,
    where `ne_level` is `lc`'s own windowed spiking rate, NORMALIZED by its own empirically-measured dynamic range
    (calibrated below, mirroring how every other rung in this arc calibrates against ITS substrate rather than
    asserting a target rate). A TONIC external current (`ne_tonic_pa`, the sweep/lesion variable — Aston-Jones &
    Cohen's TONIC mode, the animal's baseline arousal state) sets lc's baseline drive; the mm->lc synapse (Bouret &
    Sara's PHASIC mode) adds a burst on top exactly when the swap-relevant mismatch fires.
  * The GAIN_FLOOR is deliberately calibrated BELOW the production BOOST_GAIN=1.0 operating point, not merely equal
    to it: Devauges & Sara 1990 (Behav Brain Res 39(1):19-28, PMID 2167690, "Activation of the noradrenergic system
    facilitates an attentional shift in the rat" — verified via PubMed, not quoted from memory) show
    PHARMACOLOGICALLY RAISING LC-NE firing (idazoxan) makes rats reach criterion in FEWER trials specifically during
    the shift phase of a task (no effect on either component learned before the shift). Read the complementary
    direction: if boosting NE speeds a switch, a circuit whose adaptive-gain source is silent is the slow end of the
    SAME dimension, not merely "back to an ungated default" — consistent with Bouret & Sara's own account of
    LC-NE loss as impaired network reconfiguration.

REUSE-BY-IMPORT (NO `sim/` edit; NONE of the suppression/eviction substrate is rebuilt): `_pattern_geometry`,
`_rec_population_split`, `_ws_step`, `_drive`, `_read_private_rates`, `_instant_private_rate`, `_margin`, the
substrate constants, `_dense_pop`, the ignition/vacancy-gate/mismatch-pred constants, `_full_snapshot`/
`_full_restore`, and — the actual eviction/suppression EFFECTOR — `MultiLoopSTD` (Mongillo-Barak-Tsodyks STD on
each coalition's own recurrent loop) come straight from the prior GO rungs. `build()` here is a FORK of
`_gnw_neural_swap_intention_derisk.build()` (itself a fork of the vacancy-gate/active-overwrite builds) that adds
EXACTLY one new region (`lc`) and one new synaptic pathway (`mm_ALL -> lc`) on top — the same "fork the build,
reuse everything else by import" pattern every prior rung in this arc used. `run_intention_swap_ne()` is the same
fork of `run_intention_swap()`: identical stepping loop, with `lc` driven + read each step and `boost_gain`
replaced by the NE-dependent `boost_gain_eff`. Nothing about the workspace geometry, the STD effector, the
vacancy gate, or the mismatch/pred detector is altered.

DE-RISK QUESTION (board WAVE-1, this runner): is the LC-NE gain LOAD-BEARING (a graded NE level produces a graded
swap SPEED/CLEANLINESS effect) and LESIONABLE (NE removed -> a sluggish/sticky swap, verified via `lc`'s own
rate staying at floor)? Cost-routed to CPU/numpy only (cheap; no GPU). 6 seeds: 42/43/44/100/101/102.

HONEST BRAIN-BASED CHECK (named up front, not buried): `lc`'s RATE is genuine spiking activity — real Izhikevich
neurons on a real `SimulationBridge`, driven by a real synapse from `mm`'s spikes (not a host-read scalar). The
READOUT from that rate into `boost_gain_eff` (`GAIN_FLOOR + NE_GAIN_SPAN * ne_level`) is HOST ARITHMETIC — a
neuromodulator-like linear read-out with no engine primitive for "one population's firing sets another synapse
population's release-probability gain". This is NOT a new gap: it is the SAME already-disclosed residual #2 in
`webapp/gnw_thought_swap.py` ("the mm->boost COUPLING is host arithmetic... a functional correlate only"), now
extended one link further upstream (mm-rate -> lc-rate is a real synapse; lc-rate -> boost_gain is the same kind
of host read-out mm-rate -> boost_gain already was). Documented as a residual to burn down later (an engine
primitive coupling one population's rate to another synapse population's STP release-probability), not claimed
closed.

Usage (CPU cheap-first; EXPORT OMP/OPENBLAS/MKL=2):
  SIM_BACKEND=numpy python -u -m research.runners._gnw_lc_ne_gain_swap_derisk --calibrate --seed 42
  SIM_BACKEND=numpy python -u -m research.runners._gnw_lc_ne_gain_swap_derisk --smoke --seed 42 \
      --json research/findings/raw/_gnw_lc_ne_gain_swap_smoke.json
  SIM_BACKEND=numpy python -u -m research.runners._gnw_lc_ne_gain_swap_derisk --six-seed \
      --json research/findings/raw/_gnw_lc_ne_gain_swap_6seed.json
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
from sim.config import CoreSimConfig
from sim.enums import NeuronModel
from sim.regions import BrainRegion
from sim.backend import get_backend, to_host
from tools.verdict import Verdict
from tools.lab import attributable_to

# ── reuse-by-import: the ENTIRE swap/eviction substrate, unmodified — the WTA-style single-content workspace
# competition (disjoint supra-critical recurrence + divisive-norm pool + tonic thal), the neural vacancy-gate
# admission (occ/gate), the neural mismatch/salience swap trigger (mm/pred), and the STD eviction EFFECTOR
# (MultiLoopSTD) that performs the actual suppression. NO sim/ edit; none of this is rebuilt. ────────────────────
from research.runners._gnw_active_overwrite_derisk import (
    _pattern_geometry, _rec_population_split, _ws_step, _drive,
    _read_private_rates, _instant_private_rate, _margin,
    N_PATTERNS, PATTERN_SIZE, WORKSPACE_N, NORM_N, THAL_N,
    WS_NORM_W, NORM_WS_W, THAL_TONIC_PA, THAL_WS_W, OU_NOISE_PA,
)
from research.runners._gnw_rung2c_salience_disinhibition_derisk import _dense_pop
from research.runners._gnw_rung1_ignition_curve_derisk import DRIVE_STEPS, SETTLE_STEPS, WS_LOOP_GATE
from research.runners._gnw_rung2_competitive_access_derisk import _ignited
from research.runners._p1_2_workspace_deliberation_loop_derisk import _full_snapshot, _full_restore
from research.runners._gnw_recurrence_weaken_swap_derisk import MultiLoopSTD
from research.runners._gnw_neural_vacancy_gate_derisk import OCC_N, GATE_PER, W_WS_OCC, W_OCC_GATE, W_GATE_WS
from research.runners._gnw_neural_swap_intention_derisk import (
    MM_PER, PRED_PER, W_PAT_PRED, W_PRED_MM, W_MM_REC,
    SALIENT_PA, BOOST_GAIN, MAX_BOOST, BOOST_WINDOW,
    ESTABLISH_PA, EVICT_STEPS, REIGNITE_HOLD, W_REC, TIMING_WINDOW, IGNITE_THRESH,
    _pop_rate, _izh_hash,
)

_RESTORE_CALLS = {"n": 0}


def _counted_restore(bridge, snap):
    _RESTORE_CALLS["n"] += 1
    _full_restore(bridge, snap)


# ── the ONE new mechanism: a locus-coeruleus-like population (Bouret & Sara 2005; Aston-Jones & Cohen 2005) ────────
LC_N = 60                      # matches MM_PER: a comparable population-code granularity for the gain readout.
W_MM_LC = 1.2                  # mm_ALL -> lc (E_TO_E, dense): the PHASIC "network reset" trigger — lc is driven
                               # by convergent input from every pattern's mismatch/salience detector, so LC bursts
                               # to salience regardless of WHICH content is mismatching (Aston-Jones & Cohen's
                               # many-input salience integrator). Calibrated below (--calibrate).

# Tonic drive levels (pA) swept as the independent variable — Aston-Jones & Cohen's TONIC mode (the animal's
# baseline arousal STATE). NE_TONIC_LESION=0.0 combined with a lesioned build (w_mm_lc=0.0, see `build()`) fully
# ablates the LC circuit (both tonic and phasic capacity) — a clean ("verified to still hold") lesion, not merely
# a zeroed input on an intact circuit. Calibrated below against this substrate's own lc rate-current response.
NE_TONIC_LESION = 0.0
NE_TONIC_LOW = 250.0
NE_TONIC_BASE = 550.0
NE_TONIC_HIGH = 1400.0

# The adaptive-gain readout: boost_gain_eff = GAIN_FLOOR + NE_GAIN_SPAN * ne_level, where ne_level = lc's own
# windowed rate normalized by LC_RATE_REF (its rate at a strong, near-saturating tonic drive — measured, not
# asserted). GAIN_FLOOR sits BELOW the production BOOST_GAIN=1.0 default (an LC-lesioned circuit is the SLOW end of
# the dimension Devauges & Sara 1990 measured raising NE speeds up, not merely "today's shipped default").
# NE_GAIN_SPAN is calibrated so NE_TONIC_BASE lands near ~BOOST_GAIN and NE_TONIC_HIGH exceeds it with margin
# before MAX_BOOST saturates the readout (both frozen from the --calibrate measurement on seed 42 below).
GAIN_FLOOR = 0.30
NE_GAIN_SPAN = 0.45
LC_RATE_REF = 0.1761           # lc's OWN rate at a strong reference tonic drive (LC_CAL_PA) — MEASURED via
                               # --calibrate on seed 42 (tonic-only, mm silent), frozen here (same convention as
                               # the base finding's MM_PER/W_PAT_PRED calibration: "calibrated on seed 42, frozen")
                               # so ne_level = lc_rate / LC_RATE_REF is stable run-to-run, not re-derived per seed.
LC_CAL_PA = 1400.0             # == NE_TONIC_HIGH: the reference tonic drive LC_RATE_REF was measured at.


# ── build: the neural-swap-intention substrate (UNCHANGED) + the NEW lc region + the mm->lc synapse ───────────────
def build(seed=42, w_rec=W_REC, heterogeneity=True, ou_noise_pA=OU_NOISE_PA,
          w_pat_pred=W_PAT_PRED, w_pred_mm=W_PRED_MM, w_mm_rec=W_MM_REC, w_mm_lc=W_MM_LC, pred_lesion=False):
    """`_gnw_neural_swap_intention_derisk.build()` PLUS ONE new region `lc` (LC_N exc, feedforward — matches mm's
    own no-intra-recurrence design, Aston-Jones & Cohen's tonic rate is smoothly graded, not bistable) and ONE new
    dense E_TO_E pathway `mm_ALL -> lc`. `w_mm_lc=0.0` ablates the phasic pathway (paired with `ne_tonic_pa=0.0`
    at call time, this is the full LC lesion — see the module docstring). Everything else byte-for-byte the
    neural-swap-intention substrate: same regions, same wiring, same constants, same settle."""
    xp, _ = get_backend()
    regions = [
        BrainRegion(name="workspace", n_neurons=WORKSPACE_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="norm_pool", n_neurons=NORM_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="thal", n_neurons=THAL_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="occ", n_neurons=OCC_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False),
        BrainRegion(name="gate", n_neurons=GATE_PER * N_PATTERNS, exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=False),
        BrainRegion(name="mm", n_neurons=MM_PER * N_PATTERNS, exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=False),
        BrainRegion(name="pred", n_neurons=PRED_PER * N_PATTERNS, exc_fraction=0.0, internal_density=0.0,
                    enable_nmda=False),
        BrainRegion(name="lc", n_neurons=LC_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False),
    ]
    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = []
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)                # the substrate seed (het/threshold RNG) — NOT actual_seed_used
    cfg.heterogeneity_seed = int(seed)
    cfg.ou_seed = int(seed)
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.enable_stdp = False
    cfg.enable_reward_modulation = False
    cfg.enable_hebbian_learning = False
    cfg.enable_homeostasis = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.stdp_w_max = max(400.0, float(w_rec) * 4.0)
    cfg.hebbian_max_weight = max(400.0, float(w_rec) * 4.0)
    cfg.enable_parameter_heterogeneity = bool(heterogeneity)
    if ou_noise_pA > 0.0:
        cfg.enable_ou_process = True
        cfg.ou_mean_current_pA = 0.0
        cfg.ou_std_current_pA = float(ou_noise_pA)
    else:
        cfg.enable_ou_process = False

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    assert cfg.enable_homeostasis is False

    rm = bridge.region_manager
    ws = np.asarray(rm.indices("workspace"), dtype=np.int64)
    patterns, privates = _pattern_geometry(ws, N_PATTERNS, PATTERN_SIZE, 0)   # overlap=0 -> disjoint supra-critical
    ws_used = np.unique(np.concatenate(patterns)).astype(np.int64)
    norm_idx = np.asarray(rm.indices("norm_pool"), dtype=np.int64)
    thal_idx = np.asarray(rm.indices("thal"), dtype=np.int64)
    occ_idx = np.asarray(rm.indices("occ"), dtype=np.int64)
    gate_idx = np.asarray(rm.indices("gate"), dtype=np.int64)
    mm_idx = np.asarray(rm.indices("mm"), dtype=np.int64)
    pred_idx = np.asarray(rm.indices("pred"), dtype=np.int64)
    lc_idx = np.asarray(rm.indices("lc"), dtype=np.int64)
    gate_slices = [gate_idx[k * GATE_PER:(k + 1) * GATE_PER] for k in range(N_PATTERNS)]
    mm_slices = [mm_idx[k * MM_PER:(k + 1) * MM_PER] for k in range(N_PATTERNS)]
    pred_slices = [pred_idx[k * PRED_PER:(k + 1) * PRED_PER] for k in range(N_PATTERNS)]

    union_plan = dict(rm.build_wiring_plan(seed=int(seed)))
    union_plan["workspace_rec"] = _rec_population_split(patterns, privates, float(w_rec), float(w_rec))
    union_plan["ws2norm"] = _dense_pop(ws_used, norm_idx, float(WS_NORM_W), "E_TO_I")
    union_plan["norm2ws"] = _dense_pop(norm_idx, ws_used, float(NORM_WS_W), "I_TO_E")
    union_plan["thal2ws"] = _dense_pop(thal_idx, ws_used, float(THAL_WS_W), "E_TO_E")
    union_plan["ws2occ"] = _dense_pop(ws_used, occ_idx, float(W_WS_OCC), "E_TO_I")
    for k in range(N_PATTERNS):
        union_plan[f"occ2gate{k}"] = _dense_pop(occ_idx, gate_slices[k], float(W_OCC_GATE), "I_TO_E")
        union_plan[f"gate{k}2ws"] = _dense_pop(gate_slices[k], patterns[k], float(W_GATE_WS), "E_TO_E")
    wpm = 0.0 if pred_lesion else float(w_pred_mm)
    for k in range(N_PATTERNS):
        if float(w_mm_rec) > 0.0:
            union_plan[f"mm_rec{k}"] = _dense_pop(mm_slices[k], mm_slices[k], float(w_mm_rec), "E_TO_E")
        union_plan[f"pat2pred{k}"] = _dense_pop(patterns[k], pred_slices[k], float(w_pat_pred), "E_TO_I")
        union_plan[f"pred2mm{k}"] = _dense_pop(pred_slices[k], mm_slices[k], wpm, "I_TO_E")
    # ── the NEW pathway: every mm_k (all K patterns' mismatch detectors) -> lc, dense E_TO_E. w_mm_lc=0.0 -> the
    # LC LESION (the phasic "network reset" pathway is anatomically absent; paired with ne_tonic_pa=0.0 at call
    # time this is a full ablation of both LC modes, verified via lc's own measured rate staying at floor).
    if float(w_mm_lc) > 0.0:
        union_plan["mm2lc"] = _dense_pop(mm_idx, lc_idx, float(w_mm_lc), "E_TO_E")

    inh = list(norm_idx) + list(occ_idx) + list(pred_idx)   # occ + pred are inhibitory; lc is excitatory
    bridge.inject_explicit_wiring(union_plan, output_inhibitory_indices=inh)
    bridge.set_plasticity_gate(WS_LOOP_GATE, 0.0)

    thal_dev = xp.asarray(thal_idx)
    bridge.cp_external_input_current[:] = 0.0
    bridge.cp_external_input_current[thal_dev] = xp.float32(THAL_TONIC_PA)
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    snap = _full_snapshot(bridge)

    handles = {"seed": int(seed), "w_rec": float(w_rec), "w_mm_lc": float(w_mm_lc), "lc_n": int(LC_N)}
    return {
        "bridge": bridge, "xp": xp,
        "patterns": [xp.asarray(p) for p in patterns], "privates": [xp.asarray(p) for p in privates],
        "patterns_host": [p.astype(np.int64) for p in patterns], "ws_used": ws_used,
        "thal": thal_dev, "occ": xp.asarray(occ_idx),
        "gate_slices": [xp.asarray(g) for g in gate_slices],
        "mm_slices": [xp.asarray(m) for m in mm_slices], "mm_all": xp.asarray(mm_idx),
        "pred_slices": [xp.asarray(p) for p in pred_slices],
        "lc": xp.asarray(lc_idx),
        "snap": snap, "handles": handles,
    }


# ── one single-move swap: the mismatch/salience trigger is UNCHANGED; the STD boost gain is now NE-dependent ──────
def run_intention_swap_ne(S, std, *, incumbent=0, proposed=1, proposal_pa=SALIENT_PA, boost_gain_base=BOOST_GAIN,
                          gain_floor=GAIN_FLOOR, ne_gain_span=NE_GAIN_SPAN, lc_rate_ref=LC_RATE_REF,
                          ne_tonic_pa=NE_TONIC_BASE, evict_steps=EVICT_STEPS, reignite_hold=REIGNITE_HOLD,
                          trigger_lesion=False, isolate=True):
    """Byte-for-byte `run_intention_swap` (the mismatch/salience TRIGGER, the STD EFFECTOR, and the vacancy-gate
    ADMISSION are unchanged), with lc additionally driven each step by a TONIC current `ne_tonic_pa` (the
    sweep/lesion variable) plus whatever phasic drive the (already-existing) mm->lc synapse delivers, and with the
    STD boost gain now `boost_gain_eff = gain_floor + ne_gain_span * ne_level` (ne_level = lc's own windowed rate
    / lc_rate_ref) IN PLACE OF the fixed `boost_gain` constant. `boost_gain_base` is accepted for API parity with
    the base runner but unused (the NE readout REPLACES it, it does not multiply it) — kept so a caller can still
    request the byte-identical fixed-gain behavior via `ne_gain_span=0, gain_floor=boost_gain_base`."""
    bridge, xp, thal = S["bridge"], S["xp"], S["thal"]
    patterns, privates = S["patterns"], S["privates"]
    lc_dev = S["lc"]
    if isolate:
        _counted_restore(bridge, S["snap"])
        std.reset()

    # (1) establish the incumbent -> it holds on its supra-critical recurrent loop (mm/lc silent: no proposal yet).
    _drive(bridge, xp, thal, THAL_TONIC_PA, std, [(patterns[incumbent], ESTABLISH_PA)], n=DRIVE_STEPS)
    pre = _read_private_rates(bridge, xp, thal, THAL_TONIC_PA, privates, std)
    win_pre, _m_pre, n_pre = _margin(pre)

    # (2) present the proposal; mm's rate is the swap TRIGGER (unchanged); lc's rate (tonic + mm's phasic drive)
    # sets the STD boost GAIN (the new adaptive-gain readout).
    gate_dev = S["gate_slices"][proposed]
    mm_dev = S["mm_slices"][proposed]
    mm_all = S["mm_all"]
    mm_drive = 0.0 if trigger_lesion else float(proposal_pa)
    mm_hist, lc_hist, a_hist, b_hist = [], [], [], []
    xA_min, boost_max, mm_peak, lc_peak, gain_max = 1.0, 0.0, 0.0, 0.0, 0.0
    a_vacate_step, b_ignite_step, coign_steps, trigger_step = -1, -1, 0, -1
    for t in range(int(evict_steps)):
        mm_rate = _pop_rate(S, mm_all)                                  # the SALIENCE/MISMATCH read (unchanged)
        lc_rate = _pop_rate(S, lc_dev)                                  # the NEW LC-NE read (spiking, genuine)
        mm_hist.append(mm_rate); lc_hist.append(lc_rate)
        mm_win = float(np.mean(mm_hist[-BOOST_WINDOW:]))
        lc_win = float(np.mean(lc_hist[-BOOST_WINDOW:]))
        ne_level = lc_win / float(lc_rate_ref) if lc_rate_ref > 0 else 0.0
        boost_gain_eff = float(gain_floor) + float(ne_gain_span) * ne_level      # the ADAPTIVE GAIN (host readout)
        eff_boost = min(MAX_BOOST, boost_gain_eff * mm_win)              # SAME formula shape as the base finding
        for k in range(N_PATTERNS):
            std.set_boost(k, float(eff_boost))
        _ws_step(bridge, xp, thal, THAL_TONIC_PA, std,
                 drive_map=[(gate_dev, float(proposal_pa)), (mm_dev, mm_drive), (lc_dev, float(ne_tonic_pa))])
        a_hist.append(_instant_private_rate(bridge, xp, privates, incumbent))
        b_hist.append(_instant_private_rate(bridge, xp, privates, proposed))
        aw = float(np.mean(a_hist[-TIMING_WINDOW:])); bw = float(np.mean(b_hist[-TIMING_WINDOW:]))
        xA_min = min(xA_min, std.x_mean(incumbent))
        boost_max = max(boost_max, eff_boost); mm_peak = max(mm_peak, mm_rate); lc_peak = max(lc_peak, lc_rate)
        gain_max = max(gain_max, boost_gain_eff)
        if trigger_step < 0 and eff_boost >= 0.10:
            trigger_step = t
        if a_vacate_step < 0 and t >= TIMING_WINDOW and aw < IGNITE_THRESH:
            a_vacate_step = t
        if b_ignite_step < 0 and bw > IGNITE_THRESH and proposed != incumbent:
            b_ignite_step = t
        if proposed != incumbent and _ignited(aw) and _ignited(bw):
            coign_steps += 1
    for k in range(N_PATTERNS):
        std.set_boost(k, 0.0)

    # (3) identity read (free-run, no proposal/NE drive -> the held coalition sustains on its loop alone).
    post = _read_private_rates(bridge, xp, thal, THAL_TONIC_PA, privates, std)
    win_post, _m_post, n_post = _margin(post)
    hold = _read_private_rates(bridge, xp, thal, THAL_TONIC_PA, privates, std, n_free=int(reignite_hold))
    win_hold, _m_hold, n_hold = _margin(hold)

    old_res = float(post[incumbent]); new_rate = float(post[proposed])
    old_res_hold = float(hold[incumbent]); new_hold = float(hold[proposed])
    swapped = bool(proposed != incumbent and win_pre == incumbent and n_pre == 1 and (not _ignited(old_res))
                   and _ignited(new_rate) and win_post == proposed and n_post == 1)
    reignite_ok = bool(_ignited(new_hold) and win_hold == proposed and n_hold == 1 and (not _ignited(old_res_hold)))
    held = bool(win_post == incumbent and n_post == 1 and _ignited(old_res)
                and (proposed == incumbent or not _ignited(new_rate)))
    timing_ok = bool(swapped and a_vacate_step >= 0 and b_ignite_step > a_vacate_step and coign_steps == 0)
    return {
        "pre_rates": [float(r) for r in pre], "post_rates": [float(r) for r in post],
        "hold_rates": [float(r) for r in hold],
        "winner_pre": int(win_pre), "winner_post": int(win_post), "winner_hold": int(win_hold),
        "n_ignited_pre": int(n_pre), "n_ignited_post": int(n_post), "n_ignited_hold": int(n_hold),
        "old_residual_post": old_res, "old_residual_hold": old_res_hold,
        "new_rate_post": new_rate, "new_rate_hold": new_hold, "xA_min": float(xA_min),
        "mm_peak": float(mm_peak), "lc_peak": float(lc_peak), "boost_max": float(boost_max),
        "gain_max": float(gain_max), "ne_tonic_pa": float(ne_tonic_pa),
        "trigger_step": int(trigger_step), "a_vacate_step": int(a_vacate_step), "b_ignite_step": int(b_ignite_step),
        "coign_steps": int(coign_steps), "swapped": swapped, "reignite_ok": reignite_ok, "held": held,
        "timing_ok": timing_ok, "trigger_lesion": bool(trigger_lesion), "proposed": int(proposed),
        "incumbent": int(incumbent),
    }


# ── calibration: lc's own rate-current response (tonic-only, and phasic-only via a salient mismatch) ───────────────
def run_calibrate(seed, args):
    print(f"[lc-ne-gain calibrate] seed={seed} — lc rate-current response + boost_gain_eff preview", flush=True)
    S = build(seed=seed, w_mm_lc=args.w_mm_lc)
    b, xp = S["bridge"], S["xp"]
    lc_dev = S["lc"]

    def _lc_rate_tonic(tonic_pa, read=60):
        _full_restore(b, S["snap"])
        rates = []
        for _ in range(read):
            _ws_step(b, xp, S["thal"], THAL_TONIC_PA, None, drive_map=[(lc_dev, float(tonic_pa))])
            rates.append(_pop_rate(S, lc_dev))
        return float(np.mean(rates[-read // 2:]))

    def _lc_rate_phasic_only(read=60):
        """A held on the substrate + a SALIENT mismatch proposal for B (mm fires) with NO lc tonic drive -> the
        mm->lc synapse ALONE (Bouret & Sara's phasic pathway) drives lc. Read lc's rate over the proposal window."""
        _full_restore(b, S["snap"])
        _drive(b, xp, S["thal"], THAL_TONIC_PA, None, [(S["patterns"][0], ESTABLISH_PA)], n=DRIVE_STEPS)
        rates = []
        for _ in range(read):
            _ws_step(b, xp, S["thal"], THAL_TONIC_PA, None,
                     drive_map=[(S["gate_slices"][1], SALIENT_PA), (S["mm_slices"][1], SALIENT_PA), (lc_dev, 0.0)])
            rates.append(_pop_rate(S, lc_dev))
        return float(np.mean(rates[-read // 2:]))

    levels = [0.0, NE_TONIC_LOW, NE_TONIC_BASE, NE_TONIC_HIGH, LC_CAL_PA, 2500.0, 4000.0]
    print("  TONIC-ONLY lc rate-current curve (mm silent):", flush=True)
    tonic_rates = {}
    for pa in levels:
        r = _lc_rate_tonic(pa)
        tonic_rates[pa] = r
        eff = GAIN_FLOOR + NE_GAIN_SPAN * (r / LC_RATE_REF if LC_RATE_REF > 0 else 0.0)
        print(f"    ne_tonic_pa={pa:8.1f} -> lc_rate={r:.4f}  (boost_gain_eff preview={eff:.3f}, "
              f"vs production BOOST_GAIN={BOOST_GAIN:.3f})", flush=True)
    phasic = _lc_rate_phasic_only()
    print(f"  PHASIC-ONLY (salient mismatch, tonic=0): lc_rate={phasic:.4f}  (Bouret & Sara's network-reset burst "
          f"alone, w_mm_lc={args.w_mm_lc})", flush=True)
    ok = bool(tonic_rates[0.0] < 0.01 and tonic_rates[NE_TONIC_HIGH] > tonic_rates[NE_TONIC_BASE] > tonic_rates[NE_TONIC_LOW] > 0.0)
    print(f"  MONOTONIC TONIC RESPONSE (0 < low < base < high) {'HOLDS' if ok else 'FAILS'}", flush=True)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump({"runner": "_gnw_lc_ne_gain_swap_derisk", "mode": "calibrate", "seed": seed,
                       "tonic_rates": {str(k): v for k, v in tonic_rates.items()}, "phasic_only": phasic,
                       "monotonic_ok": ok}, f, indent=2)
    return 0 if ok else 1


# ── one seed: LESION vs LOW vs BASE vs HIGH tonic NE, all on the SAME substrate except the lesion build ────────────
def evaluate_seed(seed, *, proposal_pa=SALIENT_PA, evict_steps=EVICT_STEPS, reignite_hold=REIGNITE_HOLD,
                  w_rec=W_REC, heterogeneity=True, w_mm_lc=W_MM_LC, verbose=True):
    # intact substrate (LOW/BASE/HIGH + the readout-lesion control all share this ONE build, per the Rung-2d/swap
    # bug lesson: construct every MultiLoopSTD instance NOW, on the freshly-built substrate, before any arm runs).
    # ⚠ banked bug (2026-08-19, `_gnw_recurrence_weaken_swap_derisk.py`): `RecurrenceDepression` snapshots its
    # `base` recurrent weights from `cp_connections.data` AT CONSTRUCTION. An STD instance built AFTER a prior arm
    # already ran on the SAME bridge captures DEPRESSED (too-low) base weights -> a silently-weakened incumbent
    # loop in every arm built late. Fix (identical to the banked one): construct EVERY MultiLoopSTD instance NOW,
    # on the freshly-built substrate, BEFORE any arm runs.
    S = build(seed=seed, w_rec=w_rec, heterogeneity=heterogeneity, w_mm_lc=w_mm_lc)
    ws_used, pats_host = S["ws_used"], S["patterns_host"]
    b_, xp = S["bridge"], S["xp"]
    std_low = MultiLoopSTD(b_, xp, ws_used, pats_host)
    std_base = MultiLoopSTD(b_, xp, ws_used, pats_host)
    std_high = MultiLoopSTD(b_, xp, ws_used, pats_host)
    std_roff2 = MultiLoopSTD(b_, xp, ws_used, pats_host)    # readout-lesion control (lc fires, coefficient zeroed)

    # LESIONED substrate: the mm->lc synapse is anatomically absent (w_mm_lc=0) — a SEPARATE build (structural
    # lesion, mirrors wta_lesion/pred_lesion elsewhere in this arc: pools kept, the lesioned WEIGHT is what differs).
    # Verified NOT an RNG-prefix confound: `_izh_hash` is IDENTICAL between w_mm_lc=1.2 and w_mm_lc=0.0 builds at
    # the same seed (the wiring plan is built AFTER `_initialize_simulation_data` already fixed heterogeneity).
    S_les = build(seed=seed, w_rec=w_rec, heterogeneity=heterogeneity, w_mm_lc=0.0)
    std_les = MultiLoopSTD(S_les["bridge"], S_les["xp"], S_les["ws_used"], S_les["patterns_host"])

    common = dict(incumbent=0, proposed=1, proposal_pa=proposal_pa, evict_steps=evict_steps,
                  reignite_hold=reignite_hold, isolate=True)

    lesion = run_intention_swap_ne(S_les, std_les, ne_tonic_pa=NE_TONIC_LESION, **common)
    low = run_intention_swap_ne(S, std_low, ne_tonic_pa=NE_TONIC_LOW, **common)
    base = run_intention_swap_ne(S, std_base, ne_tonic_pa=NE_TONIC_BASE, **common)
    high = run_intention_swap_ne(S, std_high, ne_tonic_pa=NE_TONIC_HIGH, **common)
    # READOUT-LESION control: lc receives the SAME strong tonic drive as HIGH (so it fires just as much) but its
    # contribution to the gain is ZEROED (ne_gain_span=0) -> boost_gain_eff collapses to gain_floor regardless of
    # lc's own activity. If this reproduces the LESION-like (non-swapping/sluggish) outcome even though lc is
    # firing, the GAIN READOUT — not mere lc activity — is what is load-bearing (rules out "lc firing incidentally
    # helps via some other route" e.g. exciting the workspace directly; lc has NO projection to the workspace).
    readout_off = run_intention_swap_ne(S, std_roff2, ne_tonic_pa=NE_TONIC_HIGH, ne_gain_span=0.0, **common)

    # a CONTINUOUS (no host reset) demonstration at the BASE operating point, mirroring the base finding's own
    # no-host-reset anti-cheat: its OWN freshly-built, never-before-touched substrate (isolate=False, the FIRST and
    # ONLY operation run on it) — exactly how the base runner's headline is the first thing done to ITS substrate.
    # The four comparison arms above stay isolate=True on the SHARED substrate for a clean, independent A/B/C/D.
    S_cont = build(seed=seed, w_rec=w_rec, heterogeneity=heterogeneity, w_mm_lc=w_mm_lc)
    std_cont = MultiLoopSTD(S_cont["bridge"], S_cont["xp"], S_cont["ws_used"], S_cont["patterns_host"])
    restore_before = _RESTORE_CALLS["n"]
    continuous = run_intention_swap_ne(S_cont, std_cont, ne_tonic_pa=NE_TONIC_BASE, incumbent=0, proposed=1,
                                       proposal_pa=proposal_pa, evict_steps=evict_steps,
                                       reignite_hold=reignite_hold, isolate=False)
    host_workspace_reset_calls = int(_RESTORE_CALLS["n"] - restore_before)

    # ── anti-cheats ──
    lc_floor_verified = bool(lesion["lc_peak"] < 0.02)          # the lesion's OWN rate stayed at floor (verified,
                                                                # not merely a zeroed input on an intact circuit)
    lesion_sluggish = bool((not lesion["swapped"]) or lesion["a_vacate_step"] < 0
                           or (base["swapped"] and base["a_vacate_step"] >= 0
                               and lesion["a_vacate_step"] > base["a_vacate_step"]))
    lesion_gain_at_floor = bool(abs(lesion["gain_max"] - GAIN_FLOOR) < 1e-6)
    graded_speed = bool(base["swapped"] and high["swapped"] and low is not None
                        and (not low["swapped"] or low["a_vacate_step"] >= base["a_vacate_step"] >= 0)
                        and base["a_vacate_step"] >= 0 and high["a_vacate_step"] >= 0
                        and high["a_vacate_step"] <= base["a_vacate_step"])
    graded_cleanliness = bool(base["swapped"] and high["swapped"]
                              and high["old_residual_post"] <= base["old_residual_post"] + 1e-9)
    readout_load_bearing = bool((not readout_off["swapped"]) or readout_off["a_vacate_step"] < 0
                                or (readout_off["a_vacate_step"] >= high["a_vacate_step"] if high["swapped"] else True))
    readout_off_gain_at_floor = bool(abs(readout_off["gain_max"] - GAIN_FLOOR) < 1e-6)
    base_matches_swap_ok = bool(base["swapped"] and base["reignite_ok"] and base["timing_ok"])
    high_swap_ok = bool(high["swapped"] and high["reignite_ok"] and high["timing_ok"])
    continuous_ok = bool(continuous["swapped"])

    # attributable_to assumes bigger-value == more-effect, so the quantity fed in is SPEEDUP (evict_steps minus the
    # vacate step), not the raw step count (where smaller is faster) — a swap that never completes within the
    # window scores 0 speedup, the floor, not a fabricated maximum.
    high_speedup = float(evict_steps - high["a_vacate_step"]) if high["a_vacate_step"] >= 0 else 0.0
    roff_speedup = float(evict_steps - readout_off["a_vacate_step"]) if readout_off["a_vacate_step"] >= 0 else 0.0
    speedup_attr = attributable_to("NE gain-READOUT vs the readout-zeroed control (vacate speedup, HIGH tonic both)",
                                   high_speedup, roff_speedup, warn_below=0.5)

    # DETERMINISM (substrate-integrity anti-cheat).
    h1 = _izh_hash(b_)
    S2 = build(seed=seed, w_rec=w_rec, heterogeneity=heterogeneity, w_mm_lc=w_mm_lc)
    seed_deterministic = bool(_izh_hash(S2["bridge"]) == h1 and h1 != "")

    seed_go = bool(base_matches_swap_ok and high_swap_ok and lc_floor_verified and lesion_gain_at_floor
                   and lesion_sluggish and graded_speed and graded_cleanliness and readout_load_bearing
                   and readout_off_gain_at_floor and continuous_ok and host_workspace_reset_calls == 0
                   and seed_deterministic)

    v = Verdict("GNW LC-NE gain on the swap eviction (seed %d)" % seed)
    v.require("BASE (calibrated NE) reproduces a clean neural swap (n_post==1, B, reignite, timing)",
              base_matches_swap_ok, expect=True)
    v.require("HIGH NE also swaps cleanly", high_swap_ok, expect=True)
    v.require("LESION: lc's own rate verified at floor (<0.02) — a genuine ablation, not merely a zeroed input",
              lc_floor_verified, expect=True)
    v.require("LESION: boost_gain_eff pinned at GAIN_FLOOR (no residual NE contribution)",
              lesion_gain_at_floor, expect=True)
    v.require("LESIONABLE: LESION swap is SLUGGISH/STICKY vs BASE (fails, or vacates later)",
              lesion_sluggish, expect=True)
    v.require("LOAD-BEARING (speed): a_vacate_step is monotonically <= as NE rises (LOW>=BASE>=HIGH)",
              graded_speed, expect=True)
    v.require("LOAD-BEARING (cleanliness): HIGH NE's old_residual_post <= BASE's",
              graded_cleanliness, expect=True)
    v.require("READOUT load-bearing: zeroing ne_gain_span at HIGH tonic (lc fires, gain does not rise) "
              "reproduces the floor-gain (sluggish) outcome despite lc activity",
              readout_load_bearing, expect=True)
    v.require("READOUT-OFF control: boost_gain_eff pinned at GAIN_FLOOR despite lc firing",
              readout_off_gain_at_floor, expect=True)
    v.require("a CONTINUOUS (no host reset) run at BASE also swaps", continuous_ok, expect=True)
    v.require("no host workspace reset in the continuous demonstration", host_workspace_reset_calls == 0,
              expect=True)
    v.require("determinism: cfg.seed seeds the substrate (build-twice hash)", seed_deterministic, expect=True)
    v.disabled("homeostasis", why="frozen weights; the synaptic-scaling clip is a Rung-1/2 foot-gun")
    v.disabled("native_short_term_plasticity",
               why="engine global STP banked as annihilating; STD targets ONLY the incumbent E->E loop")
    v.disabled("additive_substrate_hash",
               why="N/A: RNG-prefix property does not hold on this engine; determinism (build-twice) is used instead")
    vd = v.decide(go=seed_go, verbose=verbose)

    result = {
        "seed": int(seed), "verdict": vd["status"], "seed_go": bool(seed_go and vd["status"] == "GO"),
        "operating_point": {"proposal_pa": float(proposal_pa), "evict_steps": int(evict_steps),
                            "reignite_hold": int(reignite_hold), "w_rec": float(w_rec), "w_mm_lc": float(w_mm_lc),
                            "gain_floor": float(GAIN_FLOOR), "ne_gain_span": float(NE_GAIN_SPAN),
                            "lc_rate_ref": float(LC_RATE_REF),
                            "ne_tonic": {"lesion": NE_TONIC_LESION, "low": NE_TONIC_LOW, "base": NE_TONIC_BASE,
                                         "high": NE_TONIC_HIGH},
                            "production_boost_gain": float(BOOST_GAIN), "heterogeneity": bool(heterogeneity)},
        "arms": {
            "lesion": {"swapped": lesion["swapped"], "a_vacate_step": lesion["a_vacate_step"],
                      "old_residual_post": lesion["old_residual_post"], "new_rate_post": lesion["new_rate_post"],
                      "lc_peak": lesion["lc_peak"], "gain_max": lesion["gain_max"], "mm_peak": lesion["mm_peak"]},
            "low": {"swapped": low["swapped"], "a_vacate_step": low["a_vacate_step"],
                   "old_residual_post": low["old_residual_post"], "lc_peak": low["lc_peak"],
                   "gain_max": low["gain_max"]},
            "base": {"swapped": base["swapped"], "a_vacate_step": base["a_vacate_step"],
                    "old_residual_post": base["old_residual_post"], "new_rate_post": base["new_rate_post"],
                    "lc_peak": base["lc_peak"], "gain_max": base["gain_max"], "reignite_ok": base["reignite_ok"],
                    "timing_ok": base["timing_ok"]},
            "high": {"swapped": high["swapped"], "a_vacate_step": high["a_vacate_step"],
                    "old_residual_post": high["old_residual_post"], "new_rate_post": high["new_rate_post"],
                    "lc_peak": high["lc_peak"], "gain_max": high["gain_max"], "reignite_ok": high["reignite_ok"],
                    "timing_ok": high["timing_ok"]},
            "readout_off": {"swapped": readout_off["swapped"], "a_vacate_step": readout_off["a_vacate_step"],
                           "old_residual_post": readout_off["old_residual_post"],
                           "lc_peak": readout_off["lc_peak"], "gain_max": readout_off["gain_max"]},
            "continuous_base": {"swapped": continuous["swapped"], "a_vacate_step": continuous["a_vacate_step"]},
        },
        "anti_cheats": {
            "lc_floor_verified": lc_floor_verified, "lesion_gain_at_floor": lesion_gain_at_floor,
            "lesion_sluggish": lesion_sluggish, "graded_speed": graded_speed,
            "graded_cleanliness": graded_cleanliness, "readout_load_bearing": readout_load_bearing,
            "readout_off_gain_at_floor": readout_off_gain_at_floor,
            "no_host_workspace_reset": bool(host_workspace_reset_calls == 0),
            "seed_deterministic": seed_deterministic,
            "speedup_attributable_fraction": speedup_attr,
        },
        "host_workspace_reset_calls": int(host_workspace_reset_calls), "substrate_hash": h1,
        "preconditions": vd["preconditions"], "disabled_processes": vd["disabled_processes"],
        "undefined_reasons": vd["undefined_reasons"],
    }
    if verbose:
        print(f"[lc-ne-gain seed={seed}] verdict={vd['status']} seed_go={result['seed_go']}", flush=True)
        for name, r in (("LESION", lesion), ("LOW", low), ("BASE", base), ("HIGH", high),
                       ("READOUT-OFF", readout_off)):
            print(f"    {name:11s} swapped={r['swapped']} a_vacate@{r['a_vacate_step']:>4d} "
                  f"b_ignite@{r['b_ignite_step']:>4d} coign={r['coign_steps']:>3d} trig@{r['trigger_step']:>4d} "
                  f"old_res={r['old_residual_post']:.4f} new={r['new_rate_post']:.3f} "
                  f"lc_peak={r['lc_peak']:.3f} gain_max={r['gain_max']:.3f} boost_max={r['boost_max']:.4f}",
                  flush=True)
        print(f"    CONTINUOUS(base) swapped={continuous['swapped']} resets={host_workspace_reset_calls} "
              f"det={seed_deterministic}", flush=True)
    return result


def run_smoke(seed, args):
    r = evaluate_seed(seed, proposal_pa=args.proposal_pa, evict_steps=args.evict_steps,
                      reignite_hold=args.reignite_hold, w_rec=args.w_rec, w_mm_lc=args.w_mm_lc,
                      heterogeneity=not args.no_heterogeneity, verbose=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_lc_ne_gain_swap_derisk", "mode": "smoke", "seed": seed, "result": r}, f,
                  indent=2, default=str)
    print(f"\n[lc-ne-gain smoke] wrote {args.json}  seed_go={r['seed_go']}", flush=True)
    return 0 if r["seed_go"] else 1


def run_six_seed(args):
    seeds = [42, 43, 44, 100, 101, 102]
    print(f"[lc-ne-gain six-seed] seeds={seeds} evict={args.evict_steps} w_mm_lc={args.w_mm_lc}", flush=True)
    per_seed = []
    for s in seeds:
        per_seed.append(evaluate_seed(s, proposal_pa=args.proposal_pa, evict_steps=args.evict_steps,
                                      reignite_hold=args.reignite_hold, w_rec=args.w_rec, w_mm_lc=args.w_mm_lc,
                                      heterogeneity=not args.no_heterogeneity, verbose=True))
    n_go = sum(1 for r in per_seed if r["seed_go"])
    n_lc_floor = sum(1 for r in per_seed if r["anti_cheats"]["lc_floor_verified"])
    n_les_gain = sum(1 for r in per_seed if r["anti_cheats"]["lesion_gain_at_floor"])
    n_sluggish = sum(1 for r in per_seed if r["anti_cheats"]["lesion_sluggish"])
    n_gspeed = sum(1 for r in per_seed if r["anti_cheats"]["graded_speed"])
    n_gclean = sum(1 for r in per_seed if r["anti_cheats"]["graded_cleanliness"])
    n_rlb = sum(1 for r in per_seed if r["anti_cheats"]["readout_load_bearing"])
    n_roff_floor = sum(1 for r in per_seed if r["anti_cheats"]["readout_off_gain_at_floor"])
    n_nores = sum(1 for r in per_seed if r["anti_cheats"]["no_host_workspace_reset"])
    n_det = sum(1 for r in per_seed if r["anti_cheats"]["seed_deterministic"])
    n_base_swap = sum(1 for r in per_seed if r["arms"]["base"]["swapped"])
    n_high_swap = sum(1 for r in per_seed if r["arms"]["high"]["swapped"])
    n_lesion_swap = sum(1 for r in per_seed if r["arms"]["lesion"]["swapped"])
    pooled_go = bool(n_go >= 5 and n_lc_floor == 6 and n_les_gain == 6 and n_sluggish >= 5 and n_gspeed >= 5
                     and n_gclean >= 5 and n_rlb >= 5 and n_roff_floor == 6 and n_nores == 6 and n_det == 6
                     and n_base_swap >= 5 and n_high_swap >= 5)
    verdict = "GO" if pooled_go else ("PARTIAL" if (n_base_swap >= 1 or n_high_swap >= 1) else "NO-GO")

    v = Verdict("GNW LC-NE gain on the swap eviction: 6-seed aggregate")
    v.require("seed-level GO on >=5/6", bool(n_go >= 5), expect=True)
    v.require("LC floor verified on 6/6 (a genuine ablation)", bool(n_lc_floor == 6), expect=True)
    v.require("lesion gain pinned at floor on 6/6", bool(n_les_gain == 6), expect=True)
    v.require("lesion sluggish/sticky on >=5/6", bool(n_sluggish >= 5), expect=True)
    v.require("graded speed on >=5/6 (LOW>=BASE>=HIGH vacate step)", bool(n_gspeed >= 5), expect=True)
    v.require("graded cleanliness on >=5/6 (HIGH old_residual <= BASE)", bool(n_gclean >= 5), expect=True)
    v.require("readout load-bearing on >=5/6", bool(n_rlb >= 5), expect=True)
    v.require("readout-off control pinned at floor on 6/6", bool(n_roff_floor == 6), expect=True)
    v.require("no host workspace reset on 6/6", bool(n_nores == 6), expect=True)
    v.require("determinism on 6/6", bool(n_det == 6), expect=True)
    v.disabled("homeostasis", why="frozen base weights; the synaptic-scaling clip is a Rung-1/2 foot-gun")
    v.disabled("native_short_term_plasticity",
               why="engine global STP banked as annihilating; STD targets ONLY the incumbent E->E recurrence")
    v.disabled("additive_substrate_hash",
               why="N/A: RNG-prefix property does not hold on this engine; determinism (build-twice) is used instead")
    vd = v.decide(go=pooled_go)

    summary = {"runner": "_gnw_lc_ne_gain_swap_derisk", "mode": "six_seed", "verdict": verdict,
               "pooled_go": pooled_go, "seeds": seeds, "operating_point": per_seed[0]["operating_point"],
               "verdict_status": vd["status"], "preconditions": vd["preconditions"],
               "disabled_processes": vd["disabled_processes"],
               "swap_rates": {"lesion": n_lesion_swap / len(seeds), "base": n_base_swap / len(seeds),
                              "high": n_high_swap / len(seeds)},
               "counts": {"seed_go": n_go, "lc_floor_verified": n_lc_floor, "lesion_gain_at_floor": n_les_gain,
                          "lesion_sluggish": n_sluggish, "graded_speed": n_gspeed, "graded_cleanliness": n_gclean,
                          "readout_load_bearing": n_rlb, "readout_off_floor": n_roff_floor,
                          "no_host_reset": n_nores, "seed_deterministic": n_det, "n_seeds": len(seeds)},
               "per_seed": per_seed}
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[lc-ne-gain six-seed] verdict={verdict} seed_go {n_go}/6 lc_floor {n_lc_floor}/6 "
          f"les_gain_floor {n_les_gain}/6 sluggish {n_sluggish}/6 graded_speed {n_gspeed}/6 "
          f"graded_clean {n_gclean}/6 readout_lb {n_rlb}/6 roff_floor {n_roff_floor}/6 no_reset {n_nores}/6 "
          f"det {n_det}/6", flush=True)
    print(f"[lc-ne-gain six-seed] SWAP RATES  lesion={n_lesion_swap}/6  base={n_base_swap}/6  "
          f"high={n_high_swap}/6  -> POOLED_GO={pooled_go}", flush=True)
    print(f"[lc-ne-gain six-seed] wrote {args.json}", flush=True)
    return 0 if pooled_go else 1


def main():
    ap = argparse.ArgumentParser(description="GNW LC-NE phasic gain on the neural thought-swap eviction: a "
                                             "locus-coeruleus-like population's graded, lesionable rate sets the "
                                             "STD eviction boost's GAIN (Aston-Jones & Cohen adaptive gain), "
                                             "driven by the SAME mismatch/salience trigger that already exists "
                                             "(Bouret & Sara network reset). REUSES the existing swap/eviction "
                                             "substrate unchanged; NO sim/ edit.")
    ap.add_argument("--calibrate", action="store_true", help="measure lc's own rate-current response")
    ap.add_argument("--smoke", action="store_true", help="full single-seed evaluation")
    ap.add_argument("--six-seed", action="store_true", help="42/43/44/100/101/102 at the frozen operating point")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--proposal-pa", type=float, default=SALIENT_PA)
    ap.add_argument("--evict-steps", type=int, default=EVICT_STEPS)
    ap.add_argument("--reignite-hold", type=int, default=REIGNITE_HOLD)
    ap.add_argument("--w-rec", type=float, default=W_REC)
    ap.add_argument("--w-mm-lc", type=float, default=W_MM_LC)
    ap.add_argument("--no-heterogeneity", action="store_true")
    ap.add_argument("--json", type=str, default="research/findings/raw/_gnw_lc_ne_gain_swap.json")
    args = ap.parse_args()

    if args.calibrate:
        return run_calibrate(args.seed, args)
    if args.six_seed:
        return run_six_seed(args)
    if args.smoke:
        return run_smoke(args.seed, args)
    r = evaluate_seed(args.seed, proposal_pa=args.proposal_pa, evict_steps=args.evict_steps,
                      reignite_hold=args.reignite_hold, w_rec=args.w_rec, w_mm_lc=args.w_mm_lc,
                      heterogeneity=not args.no_heterogeneity, verbose=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
    with open(args.json, "w") as f:
        json.dump({"runner": "_gnw_lc_ne_gain_swap_derisk", "mode": "single", "result": r}, f,
                  indent=2, default=str)
    print(f"[lc-ne-gain] wrote {args.json} seed_go={r['seed_go']}", flush=True)
    return 0 if r["seed_go"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
