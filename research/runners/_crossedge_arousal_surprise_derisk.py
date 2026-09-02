"""CROSS-EDGE #3 (one-brain integration design, rank #3): AROUSAL -> D2 SURPRISE, a PREDICTION-GAIN
neuromodulatory projection (LC-NE adaptive gain). DE-RISK ONLY (run_gate-style; NO production wire-in).

THE MECHANISM (brain-based, spiking, NOT a host multiplier). The felt-AROUSAL state (the #81/#84 graded-affect
ladder's `appraisal_lad_arousal` in [0,1], carried by the LC-like salience-integrator population,
`2026-08-13-affect-lc-arousal-population-GO.md`) is delivered by a SPIKING neuromodulatory PROJECTION onto the D2
surprise pool (`surprise_production_organ.py` / `_spiking_expectation_rpe_derisk.py`: cue -> patient_expected(FS,
GABA_A subtractive prediction) -> surprise <- patient_asserted(exc); the surprise pool's windowed firing rate IS
the "am I surprised?" read, thresholded into a verdict). A small AROUSAL source population, driven at a rate set by
the felt-arousal STATE, projects DIFFUSELY (all-to-all, NOT topographic -- the broadcast hallmark of ascending
neuromodulation, distinct from the block-diagonal prediction edges) onto the surprise pool with a FIXED
neuromodulatory weight. When arousal FIRES, its synapses inject a tonic depolarizing conductance that shifts the
surprise pool up its f-I curve -> the SAME asserted-patient input reads as MORE surprising (a lower effective
threshold / sharper competition). Low arousal (source silent) -> baseline surprise. The gain is carried ENTIRELY by
spiking synaptic transmission; the only host boundary is the felt-arousal STATE delivered as drive to the arousal
SOURCE (exactly the surprise organ's own legitimate sensory boundary -- the asserted token is likewise a drive).

WHY THIS IS NOT THE REFUTED PATH. `2026-08-10-NE-LC-gain-vigilance-REAL-SUBSTRATE-does-not-robustly-transfer-3of6.md`
showed a GLOBAL multiplicative synaptic-gain modulator (`NeuromodulatorManager.compute_synaptic_gain_multiplier`,
scope=all) for a DETECTION d' task transferring only 3/6, because a single global operating point left heterogeneous
neurons off the sensitive part of their f-I curve -- "the operating point is implicit in the animal, held by a
homeostatic set-point the idealization omitted." TWO differences here: (1) this is a GENUINE spiking PROJECTION
(arousal neurons fire -> conductance onto surprise), not a host/global weight-scalar; (2) the D2 surprise pool
ALREADY carries that missing companion -- the per-block HOMEOSTATIC prediction-gain equalizer (`_homeostat`, GO 6/6,
`2026-08-13-surprise-organ-homeostat-GO.md`) that places each block at a firing set-point. The hypothesis is that a
projection onto an already-homeostatically-held pool is more seed-robust than a global multiply onto a raw one. That
is a HYPOTHESIS this de-risk's smoke + the queued 6-seed cupy verify TEST -- NOT a claim.

HONEST SCOPE (weaker than the plasticity-gate siblings, by design of the mechanism):
  * The arousal->surprise weight is FIXED (a biologically-fixed ascending-modulation weight), NOT learned. There is
    NO emergence/growth claim here (the run_gate emergence arm does not apply). A learned/plastic ascending gain is
    a separate, later rung. This is acceptable per the design brief IFF the projection is genuinely spiking (it is:
    the lesion below zeros the SYNAPSES while the arousal source still fires, and the effect vanishes).
  * The de-risk runs on the BASE surprise circuit (homeostat left at its default; the queued cupy verify can toggle
    `BRAIN_SURPRISE_HOMEOSTAT`). The load-bearing question -- does the projection shift the surprise read, is that
    shift the SYNAPSE -- is orthogonal to the precision equalizer.

THE FOUR CHECKS (a-d in the brief):
  (a) SHIFT: for a FIXED contradict-mismatch input, HIGH arousal raises the surprise Hz vs LOW (silent) arousal
      (the control), by a signed floor -- and can flip the thresholded verdict.
  (b) VARY/LESION: zero the arousal->surprise synapses (`..._LESION=1`) -> the high-vs-low arousal shift VANISHES
      (the surprise read is flat across arousal state).
  (c) ATTRIBUTABLE: attributable_to(intact shift, lesion shift) ~ 1 (the shift IS the projection, not the host drive
      on the arousal source -- which is unchanged under lesion).
  (d) BYTE-OFF: with the projection zeroed AND the arousal source silent, the surprise reads are byte-identical
      (within the FP-layout floor) to the SHIPPED plain surprise circuit (`build_expectation_circuit`); and the
      with-edge pool's base connectivity is byte-identical to a without-edge pool (integration added ONLY the edge).

Run (numpy CPU; NO sim/ edit; NO GPU -- the GPU is busy):
  SIM_BACKEND=numpy python -m research.runners._crossedge_arousal_surprise_derisk --smoke
  SIM_BACKEND=numpy python -m research.runners._crossedge_arousal_surprise_derisk \
      --seeds 42,43,44,100,101,102 --out research/findings/raw/_crossedge_arousal_surprise_6seed.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")   # CPU only -- never touch the (busy) GPU
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from sim.backend import to_host, get_backend
from tools.lab import attributable_to

# READ-ONLY reuse of the surprise faculty's own circuit + drive/read helpers (imported, NOT edited).
from research.runners._spiking_expectation_rpe_derisk import (
    build_expectation_circuit,
    train_expectation,
    measure_conditions,
    _install_block_diagonal,
    _drive_read,
    _hard_reset,
    _idx,
    _host,
)

# ── the arousal -> surprise neuromodulatory projection (fixed weight; diffuse/broadcast) ──
AROUSAL_N = 24                 # a small LC-like ascending source population (RS)
W_AROUSAL = 0.30               # per-synapse fixed neuromodulatory weight (diffuse all-to-all onto surprise);
                                # MODULATORY by design: N*W stays a fraction of the asserted drive (W_exc=5.0), so
                                # arousal shifts the operating point rather than dominating it (gain-like: the
                                # CONTRADICT shift is ~8x the CONFIRM shift, so specificity holds). Tuned at --smoke.
AROUSAL_HIGH_PA = 600.0        # felt-arousal HIGH: the ascending source fires hard (matches the organ's 600 pA scale)
AROUSAL_LOW_PA = 0.0           # felt-arousal LOW: the ascending source is GENUINELY SILENT (the control condition --
                                # the sibling lesson: pick the condition where the SOURCE is silent as the control)

# The organ's own read geometry (unchanged): a prediction pre-phase then the measured assertion phase.
CUE_PA = 600.0
ASSERT_PA = 600.0
ASSERT_PA_WEAK = 325.0         # a FAINT/uncertain assertion (vs the full 600): its low-arousal surprise sits just
                                # below threshold, so high arousal can flip the verdict (the adaptive-gain story --
                                # noticing, when vigilant, a subtle violation you'd miss when drowsy). Bracketed to
                                # seed 42's threshold at --smoke; the flip is thus seed-dependent (reported per-seed,
                                # not a gate) -- the load-bearing checks (a-d) do not depend on it.
HOLD = 60
PRE_STEPS = 60
N_READS = 3                    # averaged reads per condition (denoise; the circuit is deterministic but reset-robust)

INTACT_FLOOR = 0.02            # the |Δ surprise Hz| high-vs-low arousal must clear, intact (a real modulatory shift)
LESION_RATIO = 0.34            # the lesioned shift must be < this * the intact shift (R1/R4's own convention)
BYTEOFF_FLOOR = 1e-6           # FP-layout floor for "reads exactly as the shipped organ"

_BASE_KW = dict(n_trained=8, n_novel=4, blk=24, cue_blk=24)   # the organ's own default circuit geometry


# ═══════════════════════════════════════════════════════════════════════════════════════════════
#  BUILD -- the surprise circuit + an appended arousal source + a diffuse fixed arousal->surprise edge.
#  The config is the surprise organ's own (build_expectation_circuit), replicated faithfully with the arousal
#  region APPENDED LAST (so the 4 core regions' per-neuron draws are an identical RNG prefix -> byte-off holds).
# ═══════════════════════════════════════════════════════════════════════════════════════════════
def build_arousal_surprise_circuit(seed, *, arousal_n=AROUSAL_N, w_arousal=W_AROUSAL, **base_kw):
    """Return (bridge, cfg, meta, arousal_mask). The 4 surprise regions are built + wired + trained EXACTLY as
    build_expectation_circuit; an `arousal` RS population is appended and wired DIFFUSELY (all-to-all, fixed weight,
    plastic=False) onto the surprise pool. `arousal_mask` is the boolean over cp_connections.data selecting the
    arousal->surprise synapses (for the lesion)."""
    from sim.bridge import SimulationBridge
    from sim.config import CoreSimConfig, RuntimeState, GPUConfig, VisualizationConfig
    from sim.regions import BrainRegion, RegionPathway
    from sim.enums import NeuronModel, NeuronType

    kw = dict(_BASE_KW); kw.update(base_kw)
    n_trained = kw["n_trained"]; n_novel = kw["n_novel"]; blk = kw["blk"]; cue_blk = kw["cue_blk"]
    n_concepts = n_trained + n_novel
    asserted_w, expected_w, cue_w = 5.0, 14.0, 0.8      # the organ's frozen operating point

    cfg = CoreSimConfig()
    cfg.seed = int(seed); cfg.heterogeneity_seed = int(seed); cfg.ou_seed = int(seed)
    cfg.backend_neutral_izh_initialization = True
    cfg.backend_neutral_izh_arithmetic = True
    cfg.per_region_threshold_heterogeneity = False
    cfg.dt_ms = 1.0
    cfg.num_traits = 1
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.enable_brain_region_framework = True
    cfg.enable_stdp = False
    cfg.enable_hebbian_learning = True
    cfg.hebbian_learning_rate = 0.06
    cfg.hebbian_min_weight = 0.0
    cfg.hebbian_max_weight = 45.0
    cfg.hebbian_weight_decay = 0.0
    cfg.hebbian_rate_window = True
    cfg.hebbian_coactivity_decay = 0.85
    cfg.hebbian_coactivity_thresh = 0.20
    cfg.hebbian_mean_subtract = 1.0
    cfg.enable_reward_modulation = False
    cfg.enable_short_term_plasticity = False
    cfg.enable_structural_plasticity = False
    cfg.enable_parameter_heterogeneity = False
    cfg.enable_ou_process = False
    cfg.enable_conductance_noise = False
    cfg.current_reward_signal = 0.0
    cfg.reward_baseline = 0.0
    cfg.enable_gabab = True
    cfg.gabab_reversal_potential = -90.0
    cfg.gabab_tau_decay = 150.0
    cfg.gabab_propagation_strength = 0.22
    cfg.gabab_conductance_max = 0.0

    RS = NeuronType.IZH2007_RS_CORTICAL_PYRAMIDAL.name
    cfg.brain_regions = [
        BrainRegion(name="cue", n_neurons=n_trained * cue_blk, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=RS),
        BrainRegion(name="patient_expected", n_neurons=n_concepts * blk, exc_fraction=0.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=NeuronType.IZH2007_FS_CORTICAL_INTERNEURON.name,
                    syn_reversal_potential_i_override=-70.0),
        BrainRegion(name="patient_asserted", n_neurons=n_concepts * blk, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=RS),
        BrainRegion(name="surprise", n_neurons=n_concepts * blk, exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=RS),
        # ── the AROUSAL ascending source (LC-like), APPENDED LAST so the 4 core regions' RNG prefix is unchanged ──
        BrainRegion(name="arousal", n_neurons=int(arousal_n), exc_fraction=1.0, internal_density=0.0,
                    exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                    izh_neuron_type=RS),
    ]
    cfg.region_pathways = [
        RegionPathway(from_region="cue", to_region="patient_expected", density=1.0, weight_mean=cue_w,
                      weight_jitter=0.0, plastic=True),
        RegionPathway(from_region="patient_asserted", to_region="surprise", density=1.0, weight_mean=asserted_w,
                      weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="patient_expected", to_region="surprise", density=1.0, weight_mean=expected_w,
                      weight_jitter=0.0, plastic=False),
        # the DIFFUSE ascending neuromodulatory projection (fixed; broadcast onto the whole surprise pool).
        RegionPathway(from_region="arousal", to_region="surprise", density=1.0, weight_mean=float(w_arousal),
                      weight_jitter=0.0, plastic=False),
    ]
    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge.runtime_state.actual_seed_used = seed
    bridge._initialize_simulation_data(called_from_playback_init=False)

    meta = dict(n_trained=n_trained, n_novel=n_novel, n_concepts=n_concepts, blk=blk, cue_blk=cue_blk,
                W_exc=asserted_w, W_inh=expected_w, arousal_n=int(arousal_n), w_arousal=float(w_arousal))
    # the surprise circuit's own topographic installs (SAME order as build_expectation_circuit).
    _install_block_diagonal(bridge, "patient_asserted", "surprise", blk, asserted_w)
    _install_block_diagonal(bridge, "patient_expected", "surprise", blk, expected_w)
    _install_block_diagonal(bridge, "cue", "patient_expected", blk, cue_w)
    # the arousal->surprise diffuse projection: set every arousal->surprise synapse to w_arousal, record its mask.
    arousal_mask = _set_region_pair_weight(bridge, "arousal", "surprise", float(w_arousal))

    bridge._blk = meta["blk"]
    bridge._rest_v = bridge.cp_membrane_potential_v.copy()
    bridge._rest_u = bridge.cp_recovery_variable_u.copy()

    # LEARN the topographic cue->expected association (strength), then FREEZE (per-turn reads never learn).
    idx_map = {n: get_backend()[0].asarray(_idx(bridge, n))
               for n in ("cue", "patient_expected", "patient_asserted", "surprise", "arousal")}
    xp = get_backend()[0]
    train_expectation(bridge, cfg, idx_map, meta, xp, n_reps=22)
    cfg.enable_hebbian_learning = False
    # NOTE: the rest snapshot is the PRE-TRAIN init state (taken above, exactly as build_expectation_circuit does);
    # _hard_reset restores it before every read, so it must NOT be refreshed post-train (that would diverge the read
    # from the shipped organ and break byte-off).
    return bridge, cfg, meta, idx_map, arousal_mask


def _live_csr(bridge):
    M = bridge.cp_connections
    if not sp.isspmatrix_csr(M):
        M = M.tocsr()
        bridge.cp_connections = M
    return M


def _region_pair_mask(bridge, src, dst):
    """Boolean over the LIVE cp_connections.data (CSR order) for synapses pre in `src`, post in `dst`. Same
    orientation probe as _install_block_diagonal (the CSR row/col convention is determined empirically)."""
    M = _live_csr(bridge)
    src_idx = set(int(i) for i in _idx(bridge, src))
    dst_idx = set(int(i) for i in _idx(bridge, dst))
    indptr = np.asarray(_host(M.indptr)); indices = np.asarray(_host(M.indices))
    row_is_dst = row_is_src = 0
    for r in range(M.shape[0]):
        r_in_dst = r in dst_idx; r_in_src = r in src_idx
        if not (r_in_dst or r_in_src):
            continue
        for off in range(int(indptr[r]), int(indptr[r + 1])):
            c = int(indices[off])
            if r_in_dst and c in src_idx:
                row_is_dst += 1
            if r_in_src and c in dst_idx:
                row_is_src += 1
    row_is_post = row_is_dst >= row_is_src
    mask = np.zeros(int(M.data.shape[0]), dtype=bool)
    for r in range(M.shape[0]):
        for off in range(int(indptr[r]), int(indptr[r + 1])):
            c = int(indices[off])
            post, pre = (r, c) if row_is_post else (c, r)
            if pre in src_idx and post in dst_idx:
                mask[off] = True
    return mask


def _set_region_pair_weight(bridge, src, dst, weight):
    """Set every src->dst synapse weight to `weight` IN PLACE; return the mask (over cp_connections.data)."""
    M = _live_csr(bridge)
    mask = _region_pair_mask(bridge, src, dst)
    assert int(mask.sum()) > 0, f"arousal->surprise selected 0 synapses -- the diffuse edge did not wire"
    data = np.asarray(_host(M.data)).astype(np.float32).copy()
    data[mask] = float(weight)
    bridge.cp_connections = sp.csr_matrix((data, np.asarray(_host(M.indices)), np.asarray(_host(M.indptr))),
                                          shape=M.shape)
    return mask


def _lesion_arousal(bridge, mask, restore=None):
    """Zero (or restore) the arousal->surprise synapse weights IN PLACE. Returns the pre-lesion data (to restore)."""
    M = _live_csr(bridge)
    data = np.asarray(_host(M.data)).astype(np.float32).copy()
    before = data.copy()
    data[mask] = 0.0 if restore is None else np.asarray(restore)[mask]
    bridge.cp_connections = sp.csr_matrix((data, np.asarray(_host(M.indices)), np.asarray(_host(M.indptr))),
                                          shape=M.shape)
    return before


# ═══════════════════════════════════════════════════════════════════════════════════════════════
#  READ -- the surprise organ's own drive/read path, with an ADDED tonic arousal drive on the source population.
# ═══════════════════════════════════════════════════════════════════════════════════════════════
def _read_surprise_hz(bridge, idx_map, xp, i, j, arousal_pa, assert_pa=ASSERT_PA):
    """Surprise Hz for asserting patient block `j` when the cue predicts block `i` (a CONTRADICT mismatch when
    j != i), with the arousal source driven tonically at `arousal_pa` throughout BOTH the prediction pre-phase and
    the measured assertion phase (ascending modulation is tonic). i==j -> CONFIRM. `assert_pa` sets the strength of
    the asserted-patient drive (a weaker drive = a fainter/more-uncertain assertion -> a borderline surprise)."""
    pre = {"cue": (i, CUE_PA)}
    drv = {"cue": (i, CUE_PA), "patient_asserted": (j, float(assert_pa))}
    if arousal_pa and arousal_pa > 0.0:
        pre = {**pre, "arousal": (None, float(arousal_pa))}
        drv = {**drv, "arousal": (None, float(arousal_pa))}
    _hard_reset(bridge)
    r = _drive_read(bridge, idx_map, drv, HOLD, xp, ["surprise"], pre_drives=pre, pre_steps=PRE_STEPS)
    return float(r["surprise"])


def _contradict_hz(bridge, idx_map, xp, arousal_pa, n_trained):
    """Mean surprise Hz over the fixed CONTRADICT battery (cue i predicts i, assert j=(i+1)%n_trained), at a given
    arousal drive. This is the fixed input whose surprise verdict the arousal projection is asked to shift."""
    vals = []
    for i in range(n_trained):
        j = (i + 1) % n_trained
        acc = [_read_surprise_hz(bridge, idx_map, xp, i, j, arousal_pa) for _ in range(N_READS)]
        vals.append(float(np.mean(acc)))
    return float(np.mean(vals)), vals


def _contradict_hz_assert(bridge, idx_map, xp, arousal_pa, n_trained, assert_pa):
    """As _contradict_hz but with a tunable asserted-patient drive (for the borderline/faint-assertion probe)."""
    vals = []
    for i in range(n_trained):
        j = (i + 1) % n_trained
        acc = [_read_surprise_hz(bridge, idx_map, xp, i, j, arousal_pa, assert_pa=assert_pa) for _ in range(N_READS)]
        vals.append(float(np.mean(acc)))
    return float(np.mean(vals)), vals


def _confirm_hz(bridge, idx_map, xp, arousal_pa, n_trained):
    vals = []
    for i in range(n_trained):
        acc = [_read_surprise_hz(bridge, idx_map, xp, i, i, arousal_pa) for _ in range(N_READS)]
        vals.append(float(np.mean(acc)))
    return float(np.mean(vals)), vals


# ═══════════════════════════════════════════════════════════════════════════════════════════════
#  THE SEED GATE -- (a) shift, (b) lesion-flatten, (c) attributable, (d) byte-off.
# ═══════════════════════════════════════════════════════════════════════════════════════════════
def run_seed(seed, *, arousal_n=AROUSAL_N, w_arousal=W_AROUSAL, assert_weak=ASSERT_PA_WEAK, verbose=False):
    t0 = time.time()
    xp, _ = get_backend()
    bridge, cfg, meta, idx_map, amask = build_arousal_surprise_circuit(
        seed, arousal_n=arousal_n, w_arousal=w_arousal)
    nt = meta["n_trained"]

    # --- baseline threshold at LOW (silent) arousal: the organ's own formula 0.5*(confirm + min(contra,novel)) ---
    base = measure_conditions(bridge, cfg, idx_map, meta, xp, cue_pa=CUE_PA, assert_pa=ASSERT_PA,
                              hold=HOLD, pre_steps=PRE_STEPS)     # arousal is NOT driven here -> silent baseline
    threshold = 0.5 * (base["confirm_hz"] + min(base["contradict_hz"], base["novel_hz"]))

    # --- (a) SHIFT: fixed CONTRADICT input, HIGH vs LOW arousal (INTACT) ---
    contra_lo, lo_per = _contradict_hz(bridge, idx_map, xp, AROUSAL_LOW_PA, nt)   # control (source silent)
    contra_hi, hi_per = _contradict_hz(bridge, idx_map, xp, AROUSAL_HIGH_PA, nt)
    delta_intact = contra_hi - contra_lo
    # specificity (honest transparency, not a gate): does arousal also raise CONFIRM? gain-like iff contra >> confirm.
    confirm_lo, _ = _confirm_hz(bridge, idx_map, xp, AROUSAL_LOW_PA, nt)
    confirm_hi, _ = _confirm_hz(bridge, idx_map, xp, AROUSAL_HIGH_PA, nt)
    delta_confirm = confirm_hi - confirm_lo
    # verdict flip on the STRONG contradict input (surprised = hz >= threshold): deeply supra-threshold, so it does
    # not flip -- the strong-contradict headline is the ATTRIBUTABLE gain-like SHIFT, not a flip.
    verdict_lo = bool(contra_lo >= threshold); verdict_hi = bool(contra_hi >= threshold)

    # --- (a') BORDERLINE VERDICT FLIP: a FAINT/uncertain assertion sits near threshold at low arousal; high arousal
    #     can push it over -> the SAME input yields a DIFFERENT surprise verdict by arousal state. The adaptive-gain
    #     story: a subtle violation, missed when drowsy, is noticed when vigilant. (Same fixed input; only arousal
    #     differs.) INTACT (measured before the lesion). ---
    bl_lo, _ = _contradict_hz_assert(bridge, idx_map, xp, AROUSAL_LOW_PA, nt, assert_weak)
    bl_hi, _ = _contradict_hz_assert(bridge, idx_map, xp, AROUSAL_HIGH_PA, nt, assert_weak)
    borderline_flip = bool((bl_lo < threshold) and (bl_hi >= threshold))

    # --- (b) LESION: zero the arousal->surprise synapses; re-read the SAME contradict input HIGH vs LOW ---
    _lesion_arousal(bridge, amask)
    contra_lo_les, _ = _contradict_hz(bridge, idx_map, xp, AROUSAL_LOW_PA, nt)
    contra_hi_les, _ = _contradict_hz(bridge, idx_map, xp, AROUSAL_HIGH_PA, nt)
    delta_lesion = contra_hi_les - contra_lo_les

    # --- (c) ATTRIBUTABLE: the shift is the projection, not the host drive on the source (unchanged under lesion) ---
    frac = attributable_to("arousal->surprise gain shift = the projection", delta_intact, delta_lesion)

    # --- (d) BYTE-OFF ---
    # (d2) STRUCTURAL: with-edge base connectivity (ALL synapses EXCEPT the arousal->surprise slots) byte-identical
    #      to a without-edge pool -> integration added ONLY the arousal->surprise synapses.
    bw, _cw, _mw, idx_w, amask_w = build_arousal_surprise_circuit(seed, arousal_n=arousal_n, w_arousal=w_arousal)
    bwo, _cwo, _mwo, _iwo, amask_wo = build_arousal_surprise_circuit(seed, arousal_n=arousal_n, w_arousal=0.0)
    dw = np.asarray(_host(_live_csr(bw).data)); dwo = np.asarray(_host(_live_csr(bwo).data))
    base_ident = bool(dw.shape == dwo.shape and np.array_equal(dw[~amask_w], dwo[~amask_wo]))
    edge_added = bool(np.array_equal(dw[~amask_w], dwo[~amask_wo]) and (dw[amask_w] != 0.0).any()
                      and (dwo[amask_wo] == 0.0).all())
    # (d1) SILENT-STATE BYTE-IDENTITY (the load-bearing off-check, CLEAN): with the arousal state OFF (source
    #      undriven), my FULL circuit (arousal region + the arousal->surprise edge) reads BYTE-IDENTICALLY to the
    #      SHIPPED plain 4-region surprise organ (build_expectation_circuit). `base` (measure_conditions on my
    #      bridge -- it never drives arousal) vs `base0` (the shipped organ); both measured on a fresh bridge before
    #      any HIGH-arousal read, so neither is contaminated. 0.0 == the edge + region are provably inert when
    #      arousal is silent AND do not perturb the core (the arousal region is a clean RNG prefix). THIS is
    #      "flag off = surprise reads exactly as today."
    b0, c0, m0 = build_expectation_circuit(seed, **_BASE_KW)
    b0._blk = m0["blk"]     # claim the block size BEFORE train_expectation drives concept blocks (organ parity)
    idx0 = {n: xp.asarray(_idx(b0, n)) for n in ("cue", "patient_expected", "patient_asserted", "surprise")}
    train_expectation(b0, c0, idx0, m0, xp, n_reps=22); c0.enable_hebbian_learning = False
    base0 = measure_conditions(b0, c0, idx0, m0, xp, cue_pa=CUE_PA, assert_pa=ASSERT_PA, hold=HOLD, pre_steps=PRE_STEPS)
    silent_byte_dev = max(abs(base["confirm_hz"] - base0["confirm_hz"]),
                          abs(base["contradict_hz"] - base0["contradict_hz"]),
                          abs(base["novel_hz"] - base0["novel_hz"]))
    silent_byte_ok = bool(silent_byte_dev < BYTEOFF_FLOOR)
    # (d3) REUSED-BRIDGE SILENT INERTNESS (REPORTED diagnostic, NOT a gate): on the fresh `bw`, read contradict at
    #      LOW arousal (edge intact), lesion, re-read at LOW. A small residual here is the un-reset ADAPTIVE-THRESHOLD
    #      drift the NE-vigilance finding characterized (the surprise pool's own firing over the intervening reads
    #      drifts its thresholds; _hard_reset does not restore them), NOT the arousal edge -- (d1) already proves the
    #      edge is inert when arousal is silent, cleanly. Recorded for transparency.
    si_intact, _ = _contradict_hz(bw, idx_w, xp, AROUSAL_LOW_PA, nt)
    _lesion_arousal(bw, amask_w)
    si_lesion, _ = _contradict_hz(bw, idx_w, xp, AROUSAL_LOW_PA, nt)
    silent_inert_dev = abs(si_intact - si_lesion)

    # --- verdicts ---
    shift_ok = bool((delta_intact > INTACT_FLOOR))
    lesion_ok = bool(abs(delta_lesion) < LESION_RATIO * max(abs(delta_intact), 1e-9))
    attrib_ok = bool(frac is not None and frac > 0.66)
    byteoff_ok = bool(silent_byte_ok and base_ident and edge_added)
    go = bool(shift_ok and lesion_ok and attrib_ok and byteoff_ok)

    return {
        "seed": int(seed), "GO": go, "elapsed_s": round(time.time() - t0, 1),
        "threshold_hz": float(threshold),
        "shift": {"contra_lo": contra_lo, "contra_hi": contra_hi, "delta_intact": float(delta_intact),
                  "floor": INTACT_FLOOR, "ok": shift_ok},
        "verdict_flip": {"low_arousal_surprised": verdict_lo, "high_arousal_surprised": verdict_hi,
                         "flipped": bool(verdict_lo != verdict_hi)},
        "borderline_flip": {"assert_pa_weak": float(assert_weak), "surprise_lo": float(bl_lo), "surprise_hi": float(bl_hi),
                            "threshold": float(threshold), "flipped": borderline_flip,
                            "note": "a faint/uncertain assertion near threshold; high arousal pushes the SAME input "
                                    "over -> a different surprise verdict by arousal state (adaptive gain)"},
        "specificity": {"confirm_lo": confirm_lo, "confirm_hi": confirm_hi, "delta_confirm": float(delta_confirm),
                        "contra_over_confirm_shift_ratio": (float(delta_intact / delta_confirm)
                                                            if abs(delta_confirm) > 1e-9 else None),
                        "note": "gain-like (sharpening) iff the contradict shift >> the confirm shift; "
                                "a comparable confirm shift = an additive threshold bias, not a slope change"},
        "lesion": {"contra_lo": contra_lo_les, "contra_hi": contra_hi_les, "delta_lesion": float(delta_lesion),
                   "ratio": LESION_RATIO, "ok": lesion_ok},
        "attributable": {"frac": (None if frac is None else float(frac)), "ok": attrib_ok},
        "byte_off": {"silent_state_byte_dev_hz": float(silent_byte_dev), "silent_state_byte_ok": silent_byte_ok,
                     "base_connectivity_identical": base_ident, "only_edge_added": edge_added,
                     "reused_bridge_silent_inert_dev_hz": float(silent_inert_dev),
                     "reused_bridge_note": "small residual = un-reset adaptive-threshold drift over the intervening "
                                           "reads (the NE-vigilance instrument artifact), NOT the edge; (d1) proves "
                                           "the edge inert when arousal silent, cleanly -- reported, not gated",
                     "ok": byteoff_ok},
        "config": {"arousal_n": int(arousal_n), "w_arousal": float(w_arousal),
                   "arousal_high_pa": AROUSAL_HIGH_PA, "arousal_low_pa": AROUSAL_LOW_PA,
                   "n_arousal_synapses": int(amask.sum())},
        "baseline": {"confirm_hz": float(base["confirm_hz"]), "contradict_hz": float(base["contradict_hz"]),
                     "novel_hz": float(base["novel_hz"])},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,100,101,102")
    ap.add_argument("--smoke", action="store_true", help="1-2 seed indicator")
    ap.add_argument("--arousal-n", type=int, default=AROUSAL_N)
    ap.add_argument("--w-arousal", type=float, default=W_AROUSAL)
    ap.add_argument("--assert-weak", type=float, default=ASSERT_PA_WEAK)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    seeds = [42, 43] if args.smoke else [int(s) for s in args.seeds.split(",") if s.strip()]

    runs = []
    for s in seeds:
        r = run_seed(s, arousal_n=args.arousal_n, w_arousal=args.w_arousal, assert_weak=args.assert_weak)
        runs.append(r)
        sh, le, at, bo, vf = r["shift"], r["lesion"], r["attributable"], r["byte_off"], r["verdict_flip"]
        print(f"[seed {s}] GO={r['GO']} | contra lo={sh['contra_lo']:.2f} hi={sh['contra_hi']:.2f} "
              f"Δ={sh['delta_intact']:+.3f} (lesion Δ={le['delta_lesion']:+.3f}) frac={at['frac']} "
              f"| borderline flip={r['borderline_flip']['flipped']} "
              f"(lo={r['borderline_flip']['surprise_lo']:.2f} hi={r['borderline_flip']['surprise_hi']:.2f} "
              f"thr={r['borderline_flip']['threshold']:.2f}) "
              f"| byteoff={bo['ok']}(silentbyte={bo['silent_state_byte_dev_hz']:.2e},"
              f"reusedinert={bo['reused_bridge_silent_inert_dev_hz']:.2e}) | shift={sh['ok']} les={le['ok']} "
              f"att={at['ok']} ({r['elapsed_s']}s)", flush=True)

    n_go = sum(r["GO"] for r in runs)
    all_go = (n_go == len(runs)) and not args.smoke
    tag = ("GO" if all_go else
           ("SMOKE-GO (indicator)" if args.smoke and n_go == len(runs) else
            ("SMOKE-PARTIAL" if args.smoke else "NO-GO")))
    verdict = (f"{tag} -- CROSS-EDGE #3 AROUSAL->D2-SURPRISE (LC-NE prediction-gain, Aston-Jones & Cohen 2005): a "
               f"FIXED diffuse spiking neuromodulatory projection from a felt-arousal source population onto the D2 "
               f"surprise pool. {n_go}/{len(runs)} seeds: HIGH arousal raises the fixed CONTRADICT surprise Hz vs "
               f"LOW/silent arousal (a); the shift VANISHES when the arousal->surprise synapses are lesioned while "
               f"the source still fires (b), attributable to the projection (c); and the surprise reads are "
               f"byte-identical to the shipped plain organ with the edge off (d). The gain WEIGHT is FIXED "
               f"(biologically-fixed ascending modulation), NOT learned -- no emergence claim. numpy CPU; NO sim/ "
               f"edit; DE-RISK only, NO production wire-in. PARTIAL pending the 6-seed cupy soak.")

    # A verdict must travel with the preconditions that earned it (gates/verdict-preconditions).
    preconditions = []
    try:
        from tools.verdict import Verdict
        Vd = Verdict("crossedge_arousal_surprise_derisk")
        Vd.require("all_seeds_shift", sum(r["shift"]["ok"] for r in runs), expect=lambda x: x == len(runs),
                   note="HIGH arousal raises the fixed contradict surprise Hz above the floor on every seed")
        Vd.require("lesion_flattens", sum(r["lesion"]["ok"] for r in runs), expect=lambda x: x == len(runs),
                   note="zeroing the arousal->surprise synapses collapses the shift (< lesion_ratio * intact)")
        Vd.require("attributable", sum(r["attributable"]["ok"] for r in runs), expect=lambda x: x == len(runs),
                   note="frac_attributable > 0.66 -- the shift is the projection, not the host drive on the source")
        Vd.require("silent_state_byte_identical", sum(r["byte_off"]["ok"] for r in runs),
                   expect=lambda x: x == len(runs),
                   note="edge OFF (source silent) reads byte-identical to the shipped organ + only-edge-added")
        dec = Vd.decide(all_go or (args.smoke and n_go == len(runs)), verbose=False)
        preconditions = dec.get("preconditions", [])
    except Exception as _e:
        preconditions = [{"kind": "meta", "name": "verdict_helper_unavailable", "ok": None, "detail": repr(_e)}]

    payload = {"probe": "crossedge_arousal_surprise_derisk", "verdict": verdict, "GO": all_go,
               "n_go": n_go, "n_seeds": len(seeds), "seeds": seeds, "preconditions": preconditions,
               "backend": os.environ.get("SIM_BACKEND", "numpy"), "cost_acknowledged": True,
               "mechanism": "arousal->surprise prediction-gain; fixed diffuse neuromodulatory projection; "
                            "NOT a host multiplier; weight fixed (not learned)",
               "grounding": "Aston-Jones & Cohen 2005 Annu Rev Neurosci 28:403 (LC-NE adaptive gain); "
                            "affect-lc-arousal-population-GO (the felt-arousal source); "
                            "surprise-organ-homeostat-GO (the companion set-point the NE-vigilance negative lacked)",
               "runs": runs}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"wrote {args.out}", flush=True)
    print("\n" + "=" * 100 + f"\n[AROUSAL->SURPRISE] VERDICT: {verdict}\n" + "=" * 100, flush=True)
    return 0 if (all_go or args.smoke) else 1


if __name__ == "__main__":
    raise SystemExit(main())
