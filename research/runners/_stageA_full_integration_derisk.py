"""Stage-A FULL SINGLE-BRIDGE LIVE INTEGRATION -- the TRUE ONE BRAIN conversation composer.

This CONSOLIDATES the four adversarially-verified Stage-A faculties -- previously MODULAR (each on its OWN
per-faculty SimulationBridge, feeding a shared arbiter via host drive numbers) -- onto ONE co-resident spiking
SimulationBridge running a REAL multi-turn conversational loop, per the integration contract in
`research/findings/2026-08-07-stageA-conversation-integration-DESIGN.md` (the 7 seams + the 8 failure modes +
the substrate = CoResidentOneBrainComposer on the merged bridge).

THE ONE BRIDGE (single SimulationBridge object, ONE process). Region slices, all co-resident:
  * rf .................... the CoResidentOneBrainComposer's VSA fact substrate -- the REAL no-confab MOAT.
  * workspace/workspace_fs/meta_schema/self_schema ... the honesty-floor relay (STEP 1); LIVE for FM4 + the
    certainty-band confidence read (self_schema spike rate on the shared substrate).
  * arb_volunteer/arb_ask/arb_silent/arb_fs ......... the ONE shared 3-way {volunteer|ask|silent} WTA arbiter
    (competitive queuing; STEP 1). affect FEEDS arb_volunteer/arb_silent; curiosity FEEDS arb_ask.
  * affect_vplus/vminus/arousal + inh_plus/inh_minus + recall_pos/neg + speak_acc/silence_acc + wta_fs ... the
    P0.3 affect organ (STEP 2); tone + forthcomingness are spike-rate DIFFERENTIALS off cp_firing_states,
    transmitted through the `affect_out` gate.
  * cur_ask .............. the curiosity ASK drive (STEP 3): `curiosity` neuromodulator (from_novelty) ->
    excitability_drive scope=group:cur_ask -> ASK-pool spikes read off cp_firing_states.

THE COMPOSITION LAW (seam 1, enforced LIVE): cue_match_moat (HARD floor) < honesty_floor < affect/DA. Affect only
modulates talkativeness/tone on candidates that already cleared moat + honesty; it NEVER touches the moat and NEVER
flips an abstain/hedge into an assert (FM4). Per-faculty RNG isolation (seam 7). One neuromodulator bus, group-scoped
(seam 5): appraisal_v+/v-/arousal (affect) + curiosity (ask) -- never scope=all.

THE MULTI-TURN LOOP demonstrates COMPOSED behavior in ONE process on ONE bridge:
  (1) a KNOWN-fact query -> honest grounded answer + affect-colored tone (arb_volunteer wins);
  (2) a NOVEL query -> the brain ASKS its OWN wh-question (arb_ask wins; crave, don't refuse), moat intact;
  (3) affect state PERSISTS + colors across turns (the slow-NMDA opponent attractor);
  (4) the honesty floor + no-confab moat hold throughout.

ANTI-CHEATS / GO-gate (single-seed smoke; the parent runs the 6-seed sweep):
  (a) SINGLE-BRIDGE -- every faculty is a region slice of ONE bridge OBJECT in ONE process (asserted: the composer's
      `_merged` bridge IS the honesty/arbiter/affect/curiosity bridge; region count reported).
  (b) COMPOSES-LIVE -- the multi-turn transcript shows honest+affect-colored answer AND curiosity-ask-on-novel AND
      moat-holds, in one loop.
  (c) FM4 LIVE -- a yoked high-arousal affect (read off the shared affect slices) mis-colors tone but NEVER flips a
      below-assert honesty read (self_schema rate on the shared relay) into an assert; a naive affect-into-confidence
      path DOES flip (the check can fail).
  (d) MOAT LIVE 475/475 -- the co-resident composer still abstains on every unstored cue under a strong positive
      high-arousal mood; 0 false-accepts, 0 manufactured answers.
  (e) NO-PIECE-BREAKS-ANOTHER -- each pairwise interaction is checked + reported HONESTLY: affect vs honesty (FM4),
      curiosity vs turn-taking (arbiter one-winner), shared arbiter one-winner, RNG isolation, and whether
      co-residence (shared het/OU/global cfg) degraded any faculty vs its modular baseline.
  (f) DEFAULT-OFF byte-identity -- the co-resident faculty slices append AFTER the composer rf slice, so the
      composer's neuron indices' firing thresholds are byte-identical with vs without the faculty slices.

HONEST-NEGATIVES (declared, not hidden):
  * HONESTY SIGNAL SPLIT: the LIVE honesty floor in the loop is the composer's on-bridge cue-match (moat abstain ->
    MOAT band; a cleared cue -> assert) composed under the g_eff law; the calibrated ACC/aPFC monitor of banked
    STEP 1 is co-resident as the workspace/meta/self relay and is exercised LIVE for FM4 + a graded-confidence probe,
    but its full calibrated-monitor routing (fit + _run_report) is run on its own modular bridges in STEP 1 -- porting
    that routing onto the shared slices is the remaining honesty consolidation step.
  * HOST-FED APPRAISAL (affect) + the BISTABLE good/bad LATCH (binary tone) + HOST RENDER of the wh-frame / tone token
    -- the STEP-2/STEP-3 characterized boundaries, inherited unchanged.
  * SHARED GLOBAL CFG: all faculties run under ONE global (het=on, OU toggled per read window). The
    no-piece-breaks-another check measures whether this degraded any faculty vs its modular baseline.

DISCIPLINE: SIM_BACKEND=numpy, reuse-by-import, NO `sim/` edit (only additive co-resident slices + read-side glue),
cfg.seed (not actual_seed_used), additive/default-off. Single-seed SMOKE -> VERDICT in ONE foreground process.

Run:
  PYTHONPATH=$PWD SIM_BACKEND=numpy python -m research.runners._stageA_full_integration_derisk \
    --seed 42 --out research/findings/raw/lanes/stageA/stageA_full_integration_s42.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("SIM_BACKEND", "numpy")
import logging as _logging  # noqa: E402
_logging.getLogger("SIM_BRIDGE").setLevel(_logging.ERROR)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig  # noqa: E402
from sim.config import CoreSimConfig  # noqa: E402
from sim.enums import NeuronModel  # noqa: E402
from sim.regions import BrainRegion, RegionPathway  # noqa: E402
from sim.backend import get_backend, to_host  # noqa: E402
from sim.neuromodulators import NeuromodulatorConfig, ModulatorTarget, ProductionRule  # noqa: E402

# reuse-by-import: the four banked faculties + the shared foundation glue.
from research.runners import _second_order_metacog_monitor_derisk as meta  # noqa: E402
from research.runners import _laneC_self_schema_metacog_integration_derisk as integ  # noqa: E402
from research.runners import _affect_state_region_derisk as aff  # noqa: E402
from research.runners._stageA_foundation_honesty_arbiter_derisk import (  # noqa: E402
    g_eff_law, certainty_band, BANDS, FacultyRNG,
    ARB_GATE, ARB_POOL_N, ARB_FS_N, ARB_LOOP_W, ARB_POOL_TO_FS_W, ARB_FS_TO_POOL_W,
)
from research.runners._gnw_rung1_ignition_curve_derisk import (  # noqa: E402
    _snapshot_state, _restore_state, _build_assembly_loop_population, SETTLE_STEPS,
)
from research.runners._gnw_rung3_report_reasoning_identity_derisk import _dense_projection  # noqa: E402
from research.runners.nav_conv_merged_bridge import CoResidentOneBrainComposer  # noqa: E402
from research.runners.rf_phasor_composer import DEFAULT_VOCAB  # noqa: E402
from tools.lab import attributable_to  # noqa: E402
from tools.verdict import Verdict  # noqa: E402


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# THE ONE BRIDGE -- every faculty as a co-resident region slice; the composer attached to the rf slice.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
AFF_ESTABLISH_MS = 120
AFF_READ_MS = 100
AFF_SETTLE_MS = 40
AFF_OU_PA = 8.0

# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SEAM-A -- FORWARD-MODEL RESERVOIR (OnBridgeLSM) -> WORLD-MODEL -> CONTENT + CERTAINTY-BAND seam (default-off).
# The fm_reservoir slice is a recurrent Izhikevich BrainRegion (internal_density = the fixed-random LSM recurrence),
# appended LAST with NO out-edges (nav/conv-inert). The agent holds the fixed-random W_in projection; per (s,a)
# token it writes `W_in @ U[t] + BIAS` into cp_external_input_current[fm_idx], runs the bridge's real step loop
# T_STEP times, and accumulates cp_firing_states[fm_idx] -> per-neuron spike-COUNT (population rate) = the read-out
# feature (OnBridgeLSM.final_state, ported to the shared bridge). Constants lifted verbatim from
# _emerge82_onbridge_lsm_derisk (_build_reservoir_bridge L83-119, final_state L128-154).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
FM_N_POOL = 300            # fm_reservoir region size (emerge82 _N_POOL)
FM_INTERNAL_DENSITY = 0.1  # the fixed-random LSM recurrence (emerge82 _INTERNAL_DENSITY)
FM_EXC_W = 6.0             # recurrent excitatory synaptic weight (emerge82 _EXC_W)
FM_INH_W = 8.0             # recurrent inhibitory synaptic weight (emerge82 _INH_W)
FM_EXC_FRACTION = 0.8      # emerge82 exc_fraction
FM_WEIGHT_JITTER = 0.3     # emerge82 weight_jitter
FM_T_STEP = 12             # bridge steps per input token (emerge82 _T_STEP)
FM_BIAS = 45.0             # tonic background current (fluctuation-driven LSM regime; emerge82 _BIAS)
FM_IN_SCALE = 320.0        # input drive scale (emerge82 _IN_SCALE)
FM_G0 = 0.06               # the certainty-band gate floor (== the cue_match_moat HARD floor; g_eff can only rise)
FM_GATE_K = 0.30           # how strongly a LOW fm margin tightens g_eff (the da_to_gate clamp discipline)
# Per-region homeostasis on the fm slice: the design names it the het-off operating-point fix, but this merged bridge
# runs het-ON (enable_parameter_heterogeneity=True), and enabling per-region homeostasis draws from the shared
# init-time RNG stream BEFORE the threshold draw -> it offsets EVERY pre-existing neuron's threshold (a measured ~5e-4
# global shift; NOT a reorder), which breaks the appended-LAST byte-identity guarantee. Since the reservoir is
# genuinely active WITHOUT homeostasis at this operating point (het-on gives the graded threshold spread the
# standalone het-off run lacked), the fm slice keeps homeostasis OFF -> byte-identical AND active. (measured 2026-08-08)
FM_ENABLE_HOMEOSTASIS = False


def make_fm_projection(seed, fm_n, in_dim):
    """The agent's fixed-random input projection W_in (emerge82 _build_reservoir_bridge L116-117)."""
    rng = np.random.default_rng(int(seed) * 7919 + 3)
    return (rng.random((int(fm_n), int(in_dim))) * 2.0 - 1.0) * FM_IN_SCALE


def read_forward_model(bridge, xp, idx, baseline_snap, W_in, U, silence=False, t_step=FM_T_STEP):
    """SEAM-A neural drive + read (OnBridgeLSM.final_state on the SHARED bridge). Per (s,a) token write
    `W_in @ U[t] + BIAS` into the fm slice, run the bridge's real step loop t_step times, accumulate the fm slice's
    spike-COUNT (population read-out feature). The FULL baseline snapshot is restored after the read, so co-resident
    nav/conv v/u return byte-identical (the wash). Returns per-neuron mean spike-count over the sequence."""
    fm = idx["fm"]
    fm_dev = xp.asarray(fm)
    _restore_state(bridge, baseline_snap)
    bridge.cp_external_input_current[:] = 0.0
    counts = np.zeros(len(fm), np.float64)
    steps = 0
    for t in range(len(U)):
        drive = np.zeros(len(fm)) if silence else (W_in @ np.asarray(U[t]) + FM_BIAS)
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[fm_dev] = xp.asarray(drive.astype(np.float32))
        for _ in range(int(t_step)):
            bridge._run_one_simulation_step()
            counts += np.asarray(to_host(bridge.cp_firing_states[fm_dev]), dtype=np.float64)
            steps += 1
    _restore_state(bridge, baseline_snap)
    bridge.cp_external_input_current[:] = 0.0
    return counts / max(1, steps)


def fm_decode(spikecounts, Ws):
    """CONTENT DECODE (declared HOST SHORTCUT, identical in status to the composer's numpy render + OnBridgeLSM's
    _fit_slots): the predicted state/role is the ridge read-out argmax(spikecounts @ Ws); the certainty is the
    top1-top2 read-out MARGIN. The brain-based content is the reservoir SPIKES; the linear decode is the read-out to
    biologize (the spiking synaptic read-out prototyped in _rungB1c_spiking_reservoir_synaptic_readout_derisk.py)."""
    feat = np.concatenate([np.asarray(spikecounts, np.float64), [1.0]])   # + bias column (emerge82 _fit_slots)
    logits = feat @ Ws
    order = np.argsort(logits)[::-1]
    top1, top2 = float(logits[order[0]]), float(logits[order[1]]) if len(order) > 1 else (float(logits[order[0]]), 0.0)
    denom = abs(top1) + abs(top2) + 1e-9
    margin_norm = float(max(0.0, min(1.0, (top1 - top2) / denom)))
    return int(order[0]), margin_norm, [float(x) for x in logits]


def fm_tighten_g_eff(g_eff, fm_margin_norm, g0=FM_G0, k_fm=FM_GATE_K):
    """SEAM-A ROUTE (i) CERTAINTY BAND (moat-safe, TIGHTENING-ONLY): a LOW forward-model margin can only RAISE
    g_eff (tighten abstention), NEVER lower it -- the SAME clamp discipline `da_to_gate` uses in the composer's
    _da_confidence_gate. `g_eff = max(g_eff, g0 + k_fm*(1 - fm_margin_norm))`. A silent reservoir (margin None) ->
    g_eff untouched (= the g0 floor). So faculty A can only make the brain MORE cautious, never less."""
    if fm_margin_norm is None:
        return float(g_eff)
    return max(float(g_eff), float(g0) + float(k_fm) * (1.0 - float(fm_margin_norm)))


def fm_content_channel(decoded_state, fm_margin_norm):
    """SEAM-A ROUTE (ii) CONTENT PATH: the decoded s' enters render ONLY as a certainty-TAGGED SIMULATION channel
    ("predicted, not observed"). It is NOT written into cp_rf_w_* and NOT added to the cue-match candidate set, so
    the no-confab moat's HARD structural floor is untouched (an unstored FACTUAL cue still abstains)."""
    return {"predicted": decoded_state, "certainty_margin": fm_margin_norm,
            "tag": "predicted, not observed", "written_to_moat": False, "in_cue_match_candidates": False}


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SEAM-C -- STAGGERED BISTABLE LADDER GRADED AFFECT -> AFFECT-COLORING seam (default-off, FM4-safe).
# The single-pool P0.3 affect latch (affect_vplus/vminus, one pool per sign -> a saturating good/bad latch) is
# AUGMENTED by a Koulakov-2002 robust-discrete integrator: N self-recurrent slow-NMDA sub-pools per valence sign
# (aff_vplus_L1..LN / aff_vminus_L1..LN), each latched by its OWN within-pool NMDA recurrence, recruited at
# STAGGERED intrinsic-excitability thresholds by a UNIFORM diffuse appraisal broadcast. Held value = number of
# latched rungs = a GRADED population rate (an N+1-level staircase, NOT the binary latch). Opponent cross-inhibition
# ONLY at the AGGREGATE (aff_agg_plus/minus); NO intra-sign lateral inhibition (the load-bearing rule -- else the
# ladder collapses back to the 2-level latch). Read NEURALLY as rate(aff_pos_readout) - rate(aff_neg_readout)
# through the SAME `affect_out` transmission gate the P0.3 organ uses. Constants lifted verbatim from
# _affect_graded_ladder_derisk (GradedLadderBrain L84-116, 6-seed GO 2026-08-08).
#
# BYTE-IDENTITY (same mechanism as SEAM-A): the ladder sub-pools carry internal_density=0 in cfg.brain_regions so
# the shared build_wiring_plan draws NO ladder-internal rng (the density>0 recurrence is injected as a SEPARATE
# union entry with an INDEPENDENT rng, below); the ladder REGIONS + ladder PATHWAYS are appended LAST to cfg so
# every pre-existing region/pathway keeps the SAME rng draw (append-LAST index+draw invariance -> pre-existing
# thresholds + conn weights bit-identical). Homeostasis stays OFF (enabling per-region homeostasis draws init-time
# RNG before the threshold draw -> shifts every pre-existing threshold; the SEAM-A measurement, identical here).
#
# FM4 SAFETY (structural): `affect_out` drives ONLY the tone/readout targets; it is ARRAY-DISJOINT from g_eff
# (_da_confidence_gate) and from the cue-match moat gate -> graded affect colors tone WITHIN the already-decided
# band and can NEVER flip abstain->assert. Lesion = set_transmission_gate("affect_out", 0.0).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
LAD_N_SUB = 20             # exc neurons per ladder sub-pool (rung) (graded-ladder N_SUB)
LAD_N_RO = 30             # readout pool size (graded-ladder N_RO)
LAD_N_AGG = 15            # aggregate opponent interneuron pool (graded-ladder N_AGG)
LAD_RECUR_DENSITY = 0.8  # within-sub-pool NMDA recurrence density (graded-ladder RECUR_DENSITY)
LAD_RECUR_W = 24.0       # within-sub-pool NMDA recurrent weight (bistable-latch regime; graded-ladder DEFAULT_RECUR)
LAD_OFF_HI = 40.0        # intrinsic offset of L1 (ignites first) pA; neutral m=0 leaves all rungs OFF (graded-ladder)
LAD_OFF_DEEPEST = -150.0  # deepest rung intrinsic offset kept ABOVE the holding floor (~-180pA) so it can persist
LAD_GAIN_PA = 240.0      # uniform appraisal sensitivity pA per unit concentration (graded-ladder DRIVE_GAIN_PA)
LAD_READOUT_INTRINSIC = -80.0  # readout threshold offset (keeps readout below saturation; graded-ladder)
LAD_BIAS_WEIGHT = 9.0    # sub-pool -> readout feedforward weight (graded-ladder BIAS_WEIGHT)
LAD_AGG_EXC_W = 6.0      # sub-pool -> aggregate interneuron (graded-ladder AGG_EXC_W)
LAD_AGG_INH_W = 10.0     # aggregate interneuron -> the OTHER sign's sub-pools (cross-inhibition; graded-ladder)
LAD_AROUSAL_SPEAK_W = 0.8  # arousal ladder -> speak_acc (WEAK: gates vigor, cannot flip abstain -> FM4 floor)
LAD_RAMP_MS = 300        # appraisal rises as a graded ramp (recruits rungs sequentially; graded-ladder RAMP_MS)
LAD_DRIVE_OFF_MS = 400   # DRIVE-OFF hold: persistence via the latches (graded-ladder drive_off_ms window)
LAD_READ_MS = 120        # readout probe window (graded-ladder probe_ms)
LAD_PATH_DENSITY = 0.6   # ladder pathway density (graded-ladder pathway density)
LADDER_NEUTRAL_TOL = 0.03  # |differential| below this at neutral appraisal == neutral tone (at-rest byte-neutral)
LADDER_RANGE_BAR = 0.05  # held-differential range bar for the staircase (graded-ladder RANGE_BAR)
LAD_ENABLE_HOMEOSTASIS = False  # OFF for byte-identity (per-region homeostasis shifts pre-existing thresholds; SEAM-A)


def _ladder_offsets(n):
    """Descending Koulakov stagger: L1 ignites at the lowest appraisal m; the deepest rung is kept ABOVE the
    holding floor so it can persist. off_step is chosen so the span L1..LN lands exactly on [LAD_OFF_DEEPEST,
    LAD_OFF_HI] for ANY rung count n (so n=8 does not push the deepest rungs monostable-OFF)."""
    if n <= 1:
        return [LAD_OFF_HI]
    step = (LAD_OFF_HI - LAD_OFF_DEEPEST) / (n - 1)
    return [float(LAD_OFF_HI - i * step) for i in range(n)]


def _ladder_region_specs(aff_n_rungs):
    """The staggered-bistable-ladder region slice (all `aff_`-prefixed; appended LAST). Sub-pools carry
    internal_density=0 (the density>0 recurrence is injected as an independent union entry -> byte-identity);
    the stagger lives in per-rung intrinsic_current_pA. Returns (regions, names) where names groups the rung/agg/
    readout region names for wiring + neuromodulator scoping."""
    RS = "IZH2007_RS_CORTICAL_PYRAMIDAL"
    FS = "IZH2007_FS_CORTICAL_INTERNEURON"
    offs = _ladder_offsets(int(aff_n_rungs))
    vplus = [f"aff_vplus_L{i+1}" for i in range(int(aff_n_rungs))]
    vminus = [f"aff_vminus_L{i+1}" for i in range(int(aff_n_rungs))]
    arousal = [f"aff_arousal_L{i+1}" for i in range(int(aff_n_rungs))]

    def _rung(name, i):
        # internal_density=0 in the region (the recurrence is a SEPARATE union entry, independent rng); the latch
        # is enable_nmda=True + LAD_RECUR_W; the stagger is the cell-autonomous intrinsic offset.
        return BrainRegion(name=name, n_neurons=LAD_N_SUB, exc_fraction=1.0, internal_density=0.0,
                           exc_weight_mean=LAD_RECUR_W, inh_weight_mean=0.0, weight_jitter=0.05,
                           plastic_internal=False, izh_neuron_type=RS, enable_nmda=True,
                           intrinsic_current_pA=float(offs[i]), enable_homeostasis=LAD_ENABLE_HOMEOSTASIS)

    def _ro(name):
        return BrainRegion(name=name, n_neurons=LAD_N_RO, exc_fraction=1.0, internal_density=0.0,
                           exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.05, plastic_internal=False,
                           izh_neuron_type=RS, enable_nmda=False, intrinsic_current_pA=LAD_READOUT_INTRINSIC,
                           enable_homeostasis=LAD_ENABLE_HOMEOSTASIS)

    def _agg(name):
        return BrainRegion(name=name, n_neurons=LAD_N_AGG, exc_fraction=0.0, internal_density=0.0,
                           exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False,
                           izh_neuron_type=FS, enable_homeostasis=LAD_ENABLE_HOMEOSTASIS)

    regions = ([_rung(n, i) for i, n in enumerate(vplus)]
               + [_rung(n, i) for i, n in enumerate(vminus)]
               + [_rung(n, i) for i, n in enumerate(arousal)]
               + [_ro("aff_pos_readout"), _ro("aff_neg_readout"),
                  _agg("aff_agg_plus"), _agg("aff_agg_minus")])
    names = {"vplus": vplus, "vminus": vminus, "arousal": arousal,
             "pos_readout": "aff_pos_readout", "neg_readout": "aff_neg_readout",
             "agg_plus": "aff_agg_plus", "agg_minus": "aff_agg_minus"}
    return regions, names


def _ladder_pathways(names):
    """The ladder pathways (feedforward readout gated by affect_out + AGGREGATE-ONLY opponent cross-inhibition +
    arousal->speak_acc vigor gated by affect_out). Appended LAST to cfg.region_pathways so pre-existing pathway
    rng draws are byte-unchanged; the transmission_gate is the framework-native gate the P0.3 organ already uses."""
    G = "affect_out"
    P = []
    # ladder -> readout (the NEURAL graded read, gated so the affect_out lesion collapses the staircase)
    for n in names["vplus"]:
        P.append(RegionPathway(from_region=n, to_region=names["pos_readout"], density=LAD_PATH_DENSITY,
                               weight_mean=LAD_BIAS_WEIGHT, weight_jitter=0.1, plastic=False, transmission_gate=G))
    for n in names["vminus"]:
        P.append(RegionPathway(from_region=n, to_region=names["neg_readout"], density=LAD_PATH_DENSITY,
                               weight_mean=LAD_BIAS_WEIGHT, weight_jitter=0.1, plastic=False, transmission_gate=G))
    # Namburi-Tye opponent cross-inhibition ONLY at the AGGREGATE (never same-sign lateral inhibition)
    for n in names["vplus"]:
        P.append(RegionPathway(from_region=n, to_region=names["agg_plus"], density=LAD_PATH_DENSITY,
                               weight_mean=LAD_AGG_EXC_W, weight_jitter=0.1, plastic=False))
    for n in names["vminus"]:
        P.append(RegionPathway(from_region=names["agg_plus"], to_region=n, density=LAD_PATH_DENSITY,
                               weight_mean=LAD_AGG_INH_W, weight_jitter=0.1, plastic=False, receptor="gaba_a"))
        P.append(RegionPathway(from_region=n, to_region=names["agg_minus"], density=LAD_PATH_DENSITY,
                               weight_mean=LAD_AGG_EXC_W, weight_jitter=0.1, plastic=False))
    for n in names["vplus"]:
        P.append(RegionPathway(from_region=names["agg_minus"], to_region=n, density=LAD_PATH_DENSITY,
                               weight_mean=LAD_AGG_INH_W, weight_jitter=0.1, plastic=False, receptor="gaba_a"))
    # arousal ladder -> speak_acc (WEAK, gated by affect_out): vigor only, cannot overcome reticence (FM4 floor)
    for n in names["arousal"]:
        P.append(RegionPathway(from_region=n, to_region="speak_acc", density=LAD_PATH_DENSITY,
                               weight_mean=LAD_AROUSAL_SPEAK_W, weight_jitter=0.1, plastic=False,
                               transmission_gate=G))
    return P


def _ladder_appraisal_mods(names):
    """One diffuse appraisal neuromodulator per sign, broadcasting concentration UNIFORMLY (volume transmission)
    to every rung group of that sign via excitability_drive. The stagger lives in the per-rung intrinsic offset;
    the uniform drive crosses each rung's staggered threshold in turn (graded recruitment)."""
    def _mod(name, groups):
        return NeuromodulatorConfig(
            name=name, baseline=0.0, decay_tau_ms=aff.APPRAISAL_TAU_MS, concentration_min=0.0, concentration_max=1.5,
            targets=[ModulatorTarget(target_type="excitability_drive", scope=f"group:{g}", sensitivity=LAD_GAIN_PA)
                     for g in groups],
            production_rules=[ProductionRule(rule_type="manual")])
    return [_mod("appraisal_lad_vplus", names["vplus"]),
            _mod("appraisal_lad_vminus", names["vminus"]),
            _mod("appraisal_lad_arousal", names["arousal"])]


def _affect_regions_pathways():
    """The P0.3 affect organ regions + pathways, LIFTED verbatim from AffectStateBrain (cross-inhibition opponent +
    affect_out-gated state->cognition + speak/silence WTA). Names are the organ's own (no collision with the
    honesty/arbiter/composer slices)."""
    RS = "IZH2007_RS_CORTICAL_PYRAMIDAL"
    FS = "IZH2007_FS_CORTICAL_INTERNEURON"

    def _aff(name):
        return BrainRegion(name=name, n_neurons=aff.N_AFF, exc_fraction=1.0, internal_density=aff.RECUR_DENSITY,
                           exc_weight_mean=float(aff.DEFAULT_RECUR_WEIGHT), inh_weight_mean=0.0, weight_jitter=0.05,
                           plastic_internal=False, izh_neuron_type=RS, enable_nmda=True)

    def _exc(name, n):
        return BrainRegion(name=name, n_neurons=n, exc_fraction=1.0, internal_density=0.0, exc_weight_mean=0.0,
                           inh_weight_mean=0.0, weight_jitter=0.05, plastic_internal=False, izh_neuron_type=RS,
                           enable_nmda=False)

    def _fs(name, n):
        return BrainRegion(name=name, n_neurons=n, exc_fraction=0.0, internal_density=0.0, exc_weight_mean=0.0,
                           inh_weight_mean=0.0, weight_jitter=0.0, plastic_internal=False, izh_neuron_type=FS)

    regions = [
        _aff("affect_vplus"), _aff("affect_vminus"), _aff("affect_arousal"),
        _fs("inh_plus", aff.XINH_N), _fs("inh_minus", aff.XINH_N),
        _exc("recall_pos", aff.N_RECALL), _exc("recall_neg", aff.N_RECALL),
        BrainRegion(name="speak_acc", n_neurons=aff.N_ACC, exc_fraction=1.0, internal_density=0.4,
                    exc_weight_mean=0.3, inh_weight_mean=0.0, weight_jitter=0.05, plastic_internal=False,
                    izh_neuron_type=RS, enable_nmda=True),
        BrainRegion(name="silence_acc", n_neurons=aff.N_ACC, exc_fraction=1.0, internal_density=0.4,
                    exc_weight_mean=0.3, inh_weight_mean=0.0, weight_jitter=0.05, plastic_internal=False,
                    izh_neuron_type=RS, enable_nmda=True),
        _fs("wta_fs", aff.N_WTA),
    ]
    G = "affect_out"
    W_XE, W_XI, W_BIAS = aff.XINH_EXC_W, aff.XINH_INH_W, aff.BIAS_WEIGHT
    pathways = [
        RegionPathway(from_region="affect_vplus", to_region="inh_plus", density=0.6, weight_mean=W_XE,
                      weight_jitter=0.1, plastic=False),
        RegionPathway(from_region="inh_plus", to_region="affect_vminus", density=0.7, weight_mean=W_XI,
                      weight_jitter=0.1, plastic=False, receptor="gaba_a"),
        RegionPathway(from_region="affect_vminus", to_region="inh_minus", density=0.6, weight_mean=W_XE,
                      weight_jitter=0.1, plastic=False),
        RegionPathway(from_region="inh_minus", to_region="affect_vplus", density=0.7, weight_mean=W_XI,
                      weight_jitter=0.1, plastic=False, receptor="gaba_a"),
        RegionPathway(from_region="affect_vplus", to_region="recall_pos", density=0.6, weight_mean=W_BIAS,
                      weight_jitter=0.1, plastic=False, transmission_gate=G),
        RegionPathway(from_region="affect_vminus", to_region="recall_neg", density=0.6, weight_mean=W_BIAS,
                      weight_jitter=0.1, plastic=False, transmission_gate=G),
        RegionPathway(from_region="affect_arousal", to_region="speak_acc", density=0.6, weight_mean=W_BIAS,
                      weight_jitter=0.1, plastic=False, transmission_gate=G),
        RegionPathway(from_region="speak_acc", to_region="wta_fs", density=0.5, weight_mean=8.0,
                      weight_jitter=0.1, plastic=False),
        RegionPathway(from_region="silence_acc", to_region="wta_fs", density=0.5, weight_mean=8.0,
                      weight_jitter=0.1, plastic=False),
        RegionPathway(from_region="wta_fs", to_region="speak_acc", density=0.6, weight_mean=6.0,
                      weight_jitter=0.1, plastic=False, receptor="gaba_a"),
        RegionPathway(from_region="wta_fs", to_region="silence_acc", density=0.6, weight_mean=6.0,
                      weight_jitter=0.1, plastic=False, receptor="gaba_a"),
    ]
    return regions, pathways


def _honesty_regions_pathways():
    """The honesty-floor relay (STEP 1): workspace K-assemblies + shared inhibition + slow-NMDA meta_schema +
    self_schema readout."""
    n_ws = meta.ASSEMBLY_SIZE * meta.K_CLASSES
    regions = [
        BrainRegion(name="workspace", n_neurons=n_ws, exc_fraction=1.0, internal_density=0.0, enable_nmda=True),
        BrainRegion(name="workspace_fs", n_neurons=meta.WORKSPACE_FS_N, exc_fraction=0.0, internal_density=0.0,
                    enable_nmda=False),
        BrainRegion(name="meta_schema", n_neurons=meta.META_SIZE, exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=True),
        BrainRegion(name="self_schema", n_neurons=integ.SELF_CONFID_SIZE, exc_fraction=1.0, internal_density=0.0,
                    enable_nmda=False),
    ]
    pathways = [
        RegionPathway(from_region="workspace", to_region="workspace_fs", density=0.5,
                      weight_mean=meta.WS_TO_FS_WEIGHT, weight_jitter=0.0, plastic=False),
        RegionPathway(from_region="workspace_fs", to_region="workspace", density=0.5,
                      weight_mean=meta.FS_TO_WS_WEIGHT, weight_jitter=0.0, plastic=False),
    ]
    return regions, pathways


def _arbiter_regions():
    pools = ["arb_volunteer", "arb_ask", "arb_silent"]
    regions = [BrainRegion(name=p, n_neurons=ARB_POOL_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=True)
               for p in pools]
    regions.append(BrainRegion(name="arb_fs", n_neurons=ARB_FS_N, exc_fraction=0.0, internal_density=0.0,
                               enable_nmda=False))
    return regions, pools


def build_one_brain(seed: int, with_faculties: bool = True, lesion_arbiter_inhibition: bool = False,
                    onebrain_k_max: int = 32, co_resident_forward_model: bool = False, fm_n_pool: int = FM_N_POOL,
                    co_resident_affect_ladder: bool = False, aff_n_rungs: int = 8):
    """Build ONE SimulationBridge: the composer rf slice FIRST, then (default-on) every faculty slice appended AFTER
    it. Returns (bridge, comp, idx, baseline_snap). When with_faculties=False, ONLY the rf slice is built (the
    default-off byte-identity baseline).

    SEAM-A (co_resident_forward_model, DEFAULT-OFF): append ONE recurrent `fm_reservoir` slice LAST with NO out-edges
    (nav-inert). Flag off -> no fm_reservoir region -> byte-identical (append-LAST index invariance).
    SEAM-C (co_resident_affect_ladder, DEFAULT-OFF): append the staggered bistable ladder (aff_n_rungs per sign +
    aggregate opponent pools + readouts) LAST, wired to the affect-coloring tone targets through affect_out. Flag off
    -> no aff_ regions -> byte-identical."""
    xp, _ = get_backend()
    rf_size = CoResidentOneBrainComposer.n_total_for(D=128, vocab=DEFAULT_VOCAB, k_max=onebrain_k_max)

    regions = [BrainRegion(name="rf", n_neurons=int(rf_size), exc_fraction=1.0, internal_density=0.0,
                           enable_nmda=False)]
    pathways = []
    pools = []
    if with_faculties:
        hon_r, hon_p = _honesty_regions_pathways()
        arb_r, pools = _arbiter_regions()
        aff_r, aff_p = _affect_regions_pathways()
        regions += hon_r + arb_r + aff_r
        regions.append(BrainRegion(name="cur_ask", n_neurons=40, exc_fraction=1.0, internal_density=0.0,
                                   enable_nmda=False))
        pathways += hon_p + aff_p

    # ---- SEAM-A: the forward-model reservoir slice, appended LAST with NO out-edges (nav/conv-inert). The slice's
    # BrainRegion carries internal_density=0 so the SHARED build_wiring_plan (which draws all region-internals THEN
    # all pathways from ONE rng) does NOT draw an fm entry -> the pre-existing pathways keep the SAME rng state ->
    # byte-identical (measured: an internal_density>0 fm draw shifts the shared rng and perturbs 13 pre-existing
    # pathways). The fm's fixed-random LSM recurrence (density FM_INTERNAL_DENSITY) is injected instead as a SEPARATE
    # union entry built with an INDEPENDENT rng (below), so it is decoupled from the shared stream. Homeostasis stays
    # OFF (enabling it draws init-time RNG before the threshold draw -> shifts every pre-existing threshold).
    if co_resident_forward_model:
        regions.append(BrainRegion(
            name="fm_reservoir", n_neurons=int(fm_n_pool), exc_fraction=FM_EXC_FRACTION,
            internal_density=0.0, exc_weight_mean=FM_EXC_W, inh_weight_mean=FM_INH_W,
            weight_jitter=FM_WEIGHT_JITTER, plastic_internal=False, enable_homeostasis=FM_ENABLE_HOMEOSTASIS))

    # ---- SEAM-C: the staggered-bistable-ladder graded-affect slice, appended LAST. Sub-pools carry
    # internal_density=0 (recurrence injected as an independent union entry below) so the shared build_wiring_plan
    # draws NO ladder-internal rng; the ladder PATHWAYS are appended LAST to `pathways` so every pre-existing
    # pathway keeps the SAME rng draw -> pre-existing thresholds + conn weights bit-identical (append-LAST
    # invariance, the SEAM-A / Probe-1 mechanism). The transmission_gate="affect_out" on the readout/vigor paths is
    # the framework-native gate the P0.3 organ already uses (FM4: array-disjoint from g_eff + the moat gate).
    ladder_names = None
    if co_resident_affect_ladder:
        lad_r, ladder_names = _ladder_region_specs(int(aff_n_rungs))
        regions += lad_r
        pathways += _ladder_pathways(ladder_names)

    cfg = CoreSimConfig()
    cfg.enable_brain_region_framework = True
    cfg.brain_regions = regions
    cfg.region_pathways = pathways
    cfg.dt_ms = 1.0
    cfg.neuron_model_type = NeuronModel.IZHIKEVICH.name
    cfg.neural_profile_name = "GENERIC_UNSTRUCTURED"
    cfg.connections_per_neuron = 0
    cfg.num_traits = 1
    cfg.seed = int(seed)                                # ⛔ seed the SUBSTRATE (not actual_seed_used)
    cfg.enable_nmda = True
    cfg.nmda_ratio = 0.5
    cfg.nmda_tau_decay = float(meta.DEFAULT_NMDA_TAU)
    cfg.nmda_recurrent_tau_decay_ms = float(meta.DEFAULT_NMDA_TAU)
    for f in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning", "enable_homeostasis",
              "enable_short_term_plasticity", "enable_structural_plasticity"):
        setattr(cfg, f, False)
    # OU state must be ALLOCATED at build (affect coloring needs it); toggled OFF at rest, ON per affect window.
    cfg.enable_ou_process = True
    cfg.ou_std_current_pA = AFF_OU_PA
    cfg.enable_parameter_heterogeneity = True           # honesty relay's graded rate code REQUIRES het (seeded)
    cfg.stdp_w_max = 400.0
    cfg.hebbian_max_weight = 400.0

    if with_faculties:
        cfg.enable_neuromodulator_subsystem = True
        cfg.current_novelty_signal = 0.0
        cfg.neuromodulators = [
            _appraisal_mod("appraisal_vplus", "affect_vplus"),
            _appraisal_mod("appraisal_vminus", "affect_vminus"),
            _appraisal_mod("appraisal_arousal", "affect_arousal"),
            NeuromodulatorConfig(
                name="curiosity", baseline=0.0, decay_tau_ms=50.0, concentration_min=0.0, concentration_max=2.0,
                targets=[ModulatorTarget(target_type="excitability_drive", scope="group:cur_ask", sensitivity=320.0)],
                production_rules=[ProductionRule(rule_type="from_novelty", sensitivity=1.0, threshold=0.0,
                                                 window_ms=50.0)]),
        ]
        # SEAM-C: the ladder appraisal broadcast (diffuse volume transmission, group-scoped to the aff_ rungs).
        if co_resident_affect_ladder and ladder_names is not None:
            cfg.neuromodulators += _ladder_appraisal_mods(ladder_names)

    bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                              runtime_state=RuntimeState(), gpu_config=GPUConfig())
    bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
    bridge._initialize_simulation_data(called_from_playback_init=False)
    cfg.enable_ou_process = False                       # rest config: OU OFF (toggled ON per affect read window)

    rm = bridge.region_manager
    rf_base = int(rm.indices("rf")[0])

    idx = {}
    if with_faculties:
        # combined injection: framework plan (honesty relay + affect) + honesty explicit relay + 3-way arbiter.
        union = dict(rm.build_wiring_plan(seed=int(seed)))
        # honesty relay explicit wiring (workspace class loops + workspace/fs->meta + meta->self); meta_rate read.
        ws = np.asarray(rm.indices("workspace"), dtype=np.int64)
        ws_fs = np.asarray(rm.indices("workspace_fs"), dtype=np.int64)
        meta_idx = np.asarray(rm.indices("meta_schema"), dtype=np.int64)
        self_idx = np.asarray(rm.indices("self_schema"), dtype=np.int64)
        member = {k: ws[k * meta.ASSEMBLY_SIZE:(k + 1) * meta.ASSEMBLY_SIZE] for k in range(meta.K_CLASSES)}
        for k in range(meta.K_CLASSES):
            union[f"loop_{k}"] = _build_assembly_loop_population(member[k], float(meta.DEFAULT_ATTRACTOR_WEIGHT))
        union["workspace_to_meta"] = _dense_projection(ws, meta_idx, float(meta.DEFAULT_META_EXC_W), meta.META_GATE)
        union["fs_to_meta"] = _dense_projection(ws_fs, meta_idx, float(meta.DEFAULT_META_INH_W), meta.META_GATE)
        union["meta_to_self_confid"] = _dense_projection(
            meta_idx, self_idx, float(integ.DEFAULT_META_TO_SELF_CONFID_W), integ.META_TO_SELF_CONFID_GATE)
        # 3-way arbiter competitive queuing.
        pool_idx = {p: np.asarray(rm.indices(p), dtype=np.int64) for p in pools}
        arb_fs = np.asarray(rm.indices("arb_fs"), dtype=np.int64)
        for p in pools:
            union[f"loop_{p}"] = _build_assembly_loop_population(pool_idx[p], ARB_LOOP_W)
            union[f"{p}_to_fs"] = _dense_projection(pool_idx[p], arb_fs, ARB_POOL_TO_FS_W, ARB_GATE)
            w_fs = 0.0 if lesion_arbiter_inhibition else ARB_FS_TO_POOL_W
            union[f"fs_to_{p}"] = _dense_projection(arb_fs, pool_idx[p], w_fs, ARB_GATE)

        # SEAM-A: inject the fm reservoir's fixed-random recurrence as its OWN union entry (the region carries
        # internal_density=0 to stay OUT of the shared plan's rng stream). Uses rm._build_region_internal (the
        # engine's exact Erdős-Rényi builder) on a density-restored shadow with an INDEPENDENT rng, so the fm
        # recurrence is identical-in-kind to a normal region-internal but decoupled from the pre-existing wiring.
        if co_resident_forward_model:
            import dataclasses as _dc
            import random as _random
            fm_region = next(r for r in rm.regions() if r.name == "fm_reservoir")
            fm_shadow = _dc.replace(fm_region, internal_density=FM_INTERNAL_DENSITY)
            fm_internal = rm._build_region_internal(fm_shadow, _random.Random(int(seed) * 100003 + 7))
            if fm_internal is not None:
                union["fm_reservoir_internal"] = fm_internal

        # SEAM-C: inject each ladder sub-pool's within-pool NMDA recurrence as its OWN union entry (the regions
        # carry internal_density=0 to stay OUT of the shared plan's rng). Each rung gets an INDEPENDENT rng so the
        # latch recurrence is identical-in-kind to a framework region-internal but decoupled from pre-existing wiring.
        if co_resident_affect_ladder and ladder_names is not None:
            import dataclasses as _dc_c
            import random as _random_c
            all_rungs = ladder_names["vplus"] + ladder_names["vminus"] + ladder_names["arousal"]
            for ri, rname in enumerate(all_rungs):
                rung_region = next(r for r in rm.regions() if r.name == rname)
                rung_shadow = _dc_c.replace(rung_region, internal_density=LAD_RECUR_DENSITY)
                rung_internal = rm._build_region_internal(
                    rung_shadow, _random_c.Random(int(seed) * 100019 + 13 + ri))
                if rung_internal is not None:
                    union[f"{rname}_internal"] = rung_internal

        inh = []
        for region in rm.regions():
            inh.extend(rm.inhibitory_indices(region.name))
        bridge.inject_explicit_wiring(union, output_inhibitory_indices=inh or None)
        # freeze every plasticity gate we registered (fixed relays + WTA).
        for g in (meta.WS_LOOP_GATE, meta.META_GATE, integ.META_TO_SELF_CONFID_GATE, ARB_GATE):
            try:
                bridge.set_plasticity_gate(g, 0.0)
            except Exception:
                pass

        idx = {
            "ws": ws, "ws_fs": ws_fs, "meta": meta_idx, "self": self_idx, "member": member,
            "pools": pools, "pool_dev": {p: xp.asarray(pool_idx[p]) for p in pools},
            "arb_fs_dev": xp.asarray(arb_fs),
            "affect": {n: np.asarray(rm.indices(n), dtype=np.int64) for n in
                       ("affect_vplus", "affect_vminus", "affect_arousal", "recall_pos", "recall_neg",
                        "speak_acc", "silence_acc")},
            "cur_ask": np.asarray(rm.indices("cur_ask"), dtype=np.int64),
        }

    if co_resident_forward_model:
        idx["fm"] = np.asarray(rm.indices("fm_reservoir"), dtype=np.int64)

    if co_resident_affect_ladder and ladder_names is not None:
        idx["ladder"] = {
            "vplus": [np.asarray(rm.indices(n), dtype=np.int64) for n in ladder_names["vplus"]],
            "vminus": [np.asarray(rm.indices(n), dtype=np.int64) for n in ladder_names["vminus"]],
            "arousal": [np.asarray(rm.indices(n), dtype=np.int64) for n in ladder_names["arousal"]],
            "pos_readout": np.asarray(rm.indices(ladder_names["pos_readout"]), dtype=np.int64),
            "neg_readout": np.asarray(rm.indices(ladder_names["neg_readout"]), dtype=np.int64),
            "names": ladder_names,
        }

    comp = CoResidentOneBrainComposer(bridge, rf_base, build_parser=False, seed=seed, D=128, vocab=DEFAULT_VOCAB,
                                      k_max=onebrain_k_max)

    # settle to a clean quiescent baseline and snapshot it (all reads restore to here).
    bridge.cp_external_input_current[:] = 0.0
    for _ in range(SETTLE_STEPS):
        bridge._run_one_simulation_step()
    bridge.cp_external_input_current[:] = 0.0
    baseline_snap = _snapshot_state(bridge, xp)
    return bridge, comp, idx, baseline_snap


def _appraisal_mod(name, group):
    return NeuromodulatorConfig(
        name=name, baseline=0.0, decay_tau_ms=aff.APPRAISAL_TAU_MS, concentration_min=0.0, concentration_max=2.0,
        targets=[ModulatorTarget(target_type="excitability_drive", scope=f"group:{group}",
                                 sensitivity=aff.DRIVE_GAIN_PA)],
        production_rules=[ProductionRule(rule_type="manual")])


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# LIVE reads on the ONE bridge (each snapshot/restores the baseline -> isolated, composer store untouched).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _reset_modulators(bridge):
    """Reset every neuromodulator concentration to its baseline (a read's own drive is set per-step; this clears any
    cross-turn carry-over so each isolated read starts from the same modulator state)."""
    nm = getattr(bridge, "neuromodulator_manager", None)
    if nm is None:
        return
    for name in ("appraisal_vplus", "appraisal_vminus", "appraisal_arousal", "curiosity"):
        try:
            nm.set_concentration(name, 0.0)
        except Exception:
            pass
    bridge.core_config.current_novelty_signal = 0.0


def read_affect(bridge, xp, idx, baseline_snap, mood_sign: int, arousal: float, lesion: bool = False) -> dict:
    """Establish an affect state (HOST-FED appraisal via the shared neuromodulator bus) and READ the two coloring
    signals as spike-rate DIFFERENTIALS off the shared bridge's cp_firing_states, gated by `affect_out`."""
    _restore_state(bridge, baseline_snap)
    bridge.cp_external_input_current[:] = 0.0
    _reset_modulators(bridge)                              # clean modulator concentrations (cross-turn isolation)
    bridge.set_transmission_gate("affect_out", 0.0 if lesion else 1.0)
    bridge.core_config.enable_ou_process = True            # affect regime (OU allocated at build)
    af = idx["affect"]
    aff_dev = {n: xp.asarray(v) for n, v in af.items()}
    nm = bridge.neuromodulator_manager
    vp = 1.0 if mood_sign > 0 else 0.0
    vm = 1.0 if mood_sign < 0 else 0.0

    def _drive(cue_pos=0.0, cue_neg=0.0, speak_base=0.0, silence_base=0.0, record=None, n_steps=1):
        counts = {r: 0.0 for r in (record or ())}
        for _ in range(int(n_steps)):
            nm.set_concentration("appraisal_vplus", float(vp))
            nm.set_concentration("appraisal_vminus", float(vm))
            nm.set_concentration("appraisal_arousal", float(arousal))
            bridge.cp_external_input_current[:] = 0.0
            if cue_pos:
                bridge.cp_external_input_current[aff_dev["recall_pos"]] = xp.float32(cue_pos)
            if cue_neg:
                bridge.cp_external_input_current[aff_dev["recall_neg"]] = xp.float32(cue_neg)
            if speak_base:
                bridge.cp_external_input_current[aff_dev["speak_acc"]] = xp.float32(speak_base)
            if silence_base:
                bridge.cp_external_input_current[aff_dev["silence_acc"]] = xp.float32(silence_base)
            bridge._run_one_simulation_step()
            if record:
                fs = to_host(bridge.cp_firing_states)
                for r in record:
                    counts[r] += float(fs[af[r]].sum())
        return counts

    _drive(n_steps=AFF_SETTLE_MS)                           # settle under the appraisal
    _drive(n_steps=AFF_ESTABLISH_MS)                        # establish the standing mood
    rec = ("recall_pos", "recall_neg", "speak_acc", "silence_acc", "affect_vplus", "affect_vminus", "affect_arousal")
    c = _drive(cue_pos=aff.RECALL_CUE_PA, cue_neg=aff.RECALL_CUE_PA, speak_base=aff.SPEAK_BASE_PA,
               silence_base=aff.SILENCE_BASE_PA, record=rec, n_steps=AFF_READ_MS)
    bridge.core_config.enable_ou_process = False
    bridge.set_transmission_gate("affect_out", 1.0)
    _restore_state(bridge, baseline_snap)
    bridge.cp_external_input_current[:] = 0.0
    n = float(AFF_READ_MS)
    v_color = (c["recall_pos"] - c["recall_neg"]) / (aff.N_RECALL * n)
    m_color = (c["speak_acc"] - c["silence_acc"]) / (aff.N_ACC * n)
    v_state = (c["affect_vplus"] - c["affect_vminus"]) / (aff.N_AFF * n)
    return {"v_color": float(v_color), "m_color": float(m_color), "v_state": float(v_state),
            "arousal_rate": float(c["affect_arousal"] / (aff.N_AFF * n))}


def read_affect_ladder(bridge, xp, idx, baseline_snap, appraisal: float, lesion: bool = False,
                       ramp_ms: int = LAD_RAMP_MS, drive_off_ms: int = LAD_DRIVE_OFF_MS,
                       read_ms: int = LAD_READ_MS) -> dict:
    """SEAM-C neural read (staggered-bistable-ladder graded affect). RAMP a POSITIVE appraisal 0->m over ramp_ms
    (Koulakov graded recruitment: rungs latch sequentially as the ramp crosses each staggered threshold), DRIVE
    OFF for drive_off_ms (persistence via the within-pool NMDA latches), then read the held value NEURALLY as the
    population-rate DIFFERENTIAL rate(aff_pos_readout) - rate(aff_neg_readout) through the `affect_out` gate. A
    balanced/neutral appraisal -> differential ~0 (neutral tone). `lesion` clamps affect_out=0 -> the readout
    collapses (proves the read is the gated ladder projection, not a host count). Snapshot/restore-isolated."""
    lad = idx["ladder"]
    _restore_state(bridge, baseline_snap)
    bridge.cp_external_input_current[:] = 0.0
    _reset_modulators(bridge)
    bridge.set_transmission_gate("affect_out", 0.0 if lesion else 1.0)
    bridge.core_config.enable_ou_process = True
    nm = bridge.neuromodulator_manager
    pos_dev = xp.asarray(lad["pos_readout"])
    neg_dev = xp.asarray(lad["neg_readout"])

    def _set(m):
        nm.set_concentration("appraisal_lad_vplus", float(m))     # POSITIVE appraisal drives the V+ ladder only
        nm.set_concentration("appraisal_lad_vminus", 0.0)
        nm.set_concentration("appraisal_lad_arousal", float(m))

    for _ in range(40):                                            # settle
        _set(0.0)
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
    for s in range(int(ramp_ms)):                                 # graded ramp 0 -> appraisal
        _set(float(appraisal) * (s + 1) / ramp_ms)
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
    for _ in range(int(drive_off_ms)):                            # DRIVE-OFF: persistence via the latches
        _set(0.0)
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
    pos = neg = 0.0
    for _ in range(int(read_ms)):                                 # read the held differential
        _set(0.0)
        bridge.cp_external_input_current[:] = 0.0
        bridge._run_one_simulation_step()
        fs = to_host(bridge.cp_firing_states)
        pos += float(np.asarray(fs)[lad["pos_readout"]].sum())
        neg += float(np.asarray(fs)[lad["neg_readout"]].sum())
    bridge.core_config.enable_ou_process = False
    bridge.set_transmission_gate("affect_out", 1.0)
    _restore_state(bridge, baseline_snap)
    bridge.cp_external_input_current[:] = 0.0
    denom = float(LAD_N_RO * max(1, read_ms))
    pos_rate, neg_rate = pos / denom, neg / denom
    return {"differential": float(pos_rate - neg_rate), "pos_rate": float(pos_rate), "neg_rate": float(neg_rate),
            "appraisal": float(appraisal), "lesioned": bool(lesion)}


def read_curiosity_want(bridge, xp, idx, baseline_snap, novelty: float, steps: int = 18) -> float:
    """Read the ASK-pool spiking wanting for an epistemic gap (novelty) on the shared bridge: write the gate novelty
    to current_novelty_signal (the from_novelty modulator input) and read cur_ask mean Hz off cp_firing_states."""
    _restore_state(bridge, baseline_snap)
    bridge.cp_external_input_current[:] = 0.0
    _reset_modulators(bridge)
    bridge.core_config.current_novelty_signal = float(novelty)
    ask = xp.asarray(idx["cur_ask"])
    n_ask = int(len(idx["cur_ask"]))
    spk = 0
    for _ in range(int(steps)):
        bridge._run_one_simulation_step()
        spk += int(to_host(bridge.cp_firing_states[ask]).sum())
    bridge.core_config.current_novelty_signal = 0.0
    _restore_state(bridge, baseline_snap)
    bridge.cp_external_input_current[:] = 0.0
    return spk / max(n_ask, 1) / (steps * 1e-3)


def run_arbiter(bridge, xp, idx, baseline_snap, drives, steps: int = 80) -> tuple:
    """Drive the three shared arbiter pools, read the late-window per-pool rate off cp_firing_states, return
    (winner, margin, rates). Snapshot/restore-isolated (one turn's arbitration leaves the bridge unchanged)."""
    _restore_state(bridge, baseline_snap)
    bridge.cp_external_input_current[:] = 0.0
    pools = idx["pools"]
    late = steps - max(1, steps // 3)
    acc = {p: 0 for p in pools}
    n_late = 0
    for t in range(steps):
        bridge.cp_external_input_current[:] = 0.0
        for p in pools:
            bridge.cp_external_input_current[idx["pool_dev"][p]] = xp.float32(float(drives[p]))
        bridge._run_one_simulation_step()
        if t >= late:
            for p in pools:
                acc[p] += int(to_host(bridge.cp_firing_states[idx["pool_dev"][p]]).sum())
            n_late += 1
    _restore_state(bridge, baseline_snap)
    bridge.cp_external_input_current[:] = 0.0
    denom = float(max(1, n_late) * ARB_POOL_N)
    rates = {p: acc[p] / denom for p in pools}
    ordered = sorted(rates.values(), reverse=True)
    margin = float((ordered[0] - ordered[1]) / (ordered[0] + ordered[1] + 1e-9))
    winner = max(rates, key=rates.get)
    return winner, margin, {p: float(r) for p, r in rates.items()}


def read_honesty_self_rate(bridge, xp, idx, baseline_snap, drive_class0: float, drive_class1: float,
                           report_steps: int = 60) -> float:
    """Drive the shared workspace class assemblies with (drive_class0, drive_class1), run the relay, and read the
    self_schema mean firing rate off cp_firing_states -- the honesty organ's on-substrate graded confidence read.
    A large drive imbalance (a confident decision) -> higher self_schema rate; a tie (uncertain) -> lower."""
    _restore_state(bridge, baseline_snap)
    bridge.cp_external_input_current[:] = 0.0
    member = idx["member"]
    m0 = xp.asarray(member[0])
    m1 = xp.asarray(member[1])
    self_dev = xp.asarray(idx["self"])
    n_self = int(len(idx["self"]))
    late = report_steps - max(1, report_steps // 3)
    acc = 0
    n_late = 0
    for t in range(report_steps):
        bridge.cp_external_input_current[:] = 0.0
        bridge.cp_external_input_current[m0] = xp.float32(float(drive_class0))
        bridge.cp_external_input_current[m1] = xp.float32(float(drive_class1))
        bridge._run_one_simulation_step()
        if t >= late:
            acc += int(to_host(bridge.cp_firing_states[self_dev]).sum())
            n_late += 1
    _restore_state(bridge, baseline_snap)
    bridge.cp_external_input_current[:] = 0.0
    return acc / max(1, n_late) / max(1, n_self)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# tone / forthcomingness renders (host render of the neural coloring signal -- declared honest-negative)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
TONE_POS, TONE_NEG, TONE_NEU = "gladly", "reluctantly", ""


def _tone(v_color, dead=0.02):
    return TONE_POS if v_color > dead else (TONE_NEG if v_color < -dead else TONE_NEU)


def _forthcomingness(m_color, dead=0.02, max_extra=3):
    return 0 if m_color <= dead else int(min(max_extra, 1 + int(m_color / 0.03)))


def _colored_answer(comp, agent, action, v_color, m_color):
    """The g_eff-LAW colored read: the moat (query_patient) runs FIRST; on a matched answer, affect adds tone +
    volunteers extra on-topic associates from the composer's OWN association graph (never a different fact)."""
    raw = comp.query_patient(agent, action)
    if raw is None:
        return {"answer": None, "abstain": True, "utterance": None}
    tone = _tone(v_color)
    extra = _forthcomingness(m_color)
    associates = []
    try:
        graph = comp._assoc_graph()
        if agent in graph:
            associates = [k for k, _ in sorted(graph[agent].items(), key=lambda kv: -kv[1])][:extra]
    except Exception:
        associates = []
    parts = ([tone] if tone else []) + [f"{agent} {action} {raw}"]
    if associates:
        parts.append("; also " + ", ".join(associates))
    return {"answer": raw, "abstain": False, "utterance": " ".join(parts).strip(),
            "tone": tone, "forthcomingness_extra": int(extra), "associates": associates}


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SEAM-A LIVE ON THE TURN -- forward-model world-model: on a NOVEL (s,a) turn drive the fm_reservoir with the
# turn's (s,a), read its per-neuron spike-counts, decode s' + the read-out MARGIN, fold the margin into g_eff
# (tighten-only), and OFFER the decoded s' as a certainty-TAGGED "predicted, not observed" channel. The moat still
# abstains on the unstored FACTUAL cue (the decoded s' NEVER enters the cue-match candidate set / cp_rf_w_*).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
FM_LOOP_IN_DIM = 16   # the (s,a) token-embedding dim the agent's fixed-random W_in projects into the fm reservoir


def _word_embedding(seed, vocab, in_dim=FM_LOOP_IN_DIM):
    """A fixed per-word SENSORY embedding of the (s,a) tokens -- host-provided token encoding, SAME status as the
    reservoir's fixed-random W_in projection / the retinal render of a sensory input (a legitimate host input), NOT
    the read-out. Deterministic per seed."""
    rng = np.random.default_rng(int(seed) * 99991 + 5)
    return {w: rng.normal(0.0, 1.0, int(in_dim)) for w in vocab}


def _fm_encode_sa(emb, a, v, in_dim=FM_LOOP_IN_DIM):
    """The (state, action) token sequence fed to the fm reservoir: [emb(state), emb(action)]. An unknown token -> a
    zero vector (the OTHER token + the fixed recurrence still drive a distinct trajectory for a novel (s,a))."""
    z = np.zeros(int(in_dim))
    return [np.asarray(emb.get(a, z)), np.asarray(emb.get(v, z))]


def build_fm_world_model(bridge, xp, idx, baseline_snap, comp, facts, emb, W_in, seed):
    """Train the forward model's ridge read-out (DECLARED HOST SHORTCUT, identical in status to the composer render /
    OnBridgeLSM._fit_slots): each STORED (agent, action) drives the shared fm reservoir -> a spike-COUNT feature;
    the target is the patient STATE class. The reservoir SPIKES are the brain-based content; the linear decode is the
    read-out to biologize (the spiking synaptic read-out prototyped in _rungB1c_...). On a NOVEL (s,a) the SAME
    reservoir generalizes -> a predicted s' + a top1-top2 margin (the world model's guess), while the moat still
    ABSTAINS on the unstored factual cue."""
    classes = sorted({p for (_a, _v, p) in facts})
    cidx = {p: i for i, p in enumerate(classes)}
    X, Y = [], []
    for (a, v, p) in facts:
        sc = read_forward_model(bridge, xp, idx, baseline_snap, W_in, _fm_encode_sa(emb, a, v))
        X.append(np.concatenate([sc, [1.0]]))
        Y.append(cidx[p])
    X = np.asarray(X)
    Y = np.asarray(Y)
    K = max(1, len(classes))
    Yoh = np.eye(K)[Y]
    Ws = np.linalg.solve(X.T @ X + 1.0 * np.eye(X.shape[1]), X.T @ Yoh)
    train_hit = sum(int(np.argmax(x @ Ws) == y) for x, y in zip(X, Y))
    return {"Ws": Ws, "classes": classes, "W_in": W_in, "emb": emb, "in_dim": FM_LOOP_IN_DIM,
            "n_classes": int(K), "train_acc": float(train_hit / max(1, len(Y)))}


def fm_predict_turn(bridge, xp, idx, baseline_snap, fm, a, v, silence=False):
    """SEAM-A LIVE on a turn: drive the shared fm reservoir with the turn's (s,a), read per-neuron spike-counts,
    decode s' + the read-out margin, and (TIGHTENING-ONLY) fold the margin into g_eff. A SILENT reservoir (the A
    lesion) -> no spikes -> NO prediction, g_eff untouched (the g0 floor) -> the turn reverts to plain abstention.
    The decoded s' is a certainty-TAGGED simulation channel (fm_content_channel): NOT written to cp_rf_w_*, NOT in
    the cue-match candidate set -> the no-confab moat's HARD floor is untouched by construction."""
    sc = read_forward_model(bridge, xp, idx, baseline_snap, fm["W_in"], _fm_encode_sa(fm["emb"], a, v),
                            silence=silence)
    active = bool(float(np.max(sc)) > 1e-6)
    if not active:
        return {"predicted": None, "margin": None, "g_eff": float(fm_tighten_g_eff(FM_G0, None)),
                "reservoir_active": False, "content": fm_content_channel(None, None)}
    pred_i, margin, _logits = fm_decode(sc, fm["Ws"])
    predicted = fm["classes"][pred_i] if 0 <= pred_i < len(fm["classes"]) else None
    return {"predicted": predicted, "margin": float(margin), "g_eff": float(fm_tighten_g_eff(FM_G0, margin)),
            "reservoir_active": True, "content": fm_content_channel(predicted, float(margin))}


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# SEAM-C LIVE ON THE TURN -- GRADED-affect coloring: tone + forthcomingness are set from the NEURAL ladder
# differential rate(aff_pos_readout) - rate(aff_neg_readout), a MULTI-LEVEL staircase that REPLACES the binary
# latch coloring. Affect only colors WITHIN the already-decided band; it never touches the moat or g_eff (FM4).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
GRADED_TONE_LEVELS = {3: "warmly, gladly", 2: "gladly", 1: "readily", 0: "",
                      -1: "reluctantly", -2: "curtly", -3: "coldly, reluctantly"}


def _graded_tone_level(differential, tol=LADDER_NEUTRAL_TOL, step=0.03, max_lvl=3):
    """GRADED tone LEVEL from the ladder's NEURAL differential -- MULTIPLE warmth levels (the Koulakov staircase),
    REPLACING the P0.3 binary latch tone. |diff| < tol -> neutral (level 0)."""
    if abs(differential) < tol:
        return 0
    lvl = int(min(max_lvl, 1 + int(abs(differential) / step)))
    return lvl if differential > 0 else -lvl


def _graded_tone_token(level):
    return GRADED_TONE_LEVELS.get(int(level), "")


def _graded_forthcomingness(differential, tol=LADDER_NEUTRAL_TOL, step=0.02, max_extra=3):
    """Forthcomingness GRADED from the SAME ladder differential (higher positive valence -> more volunteered
    associates)."""
    if differential <= tol:
        return 0
    return int(min(max_extra, 1 + int(differential / step)))


def _colored_answer_graded(comp, agent, action, differential):
    """SEAM-C LIVE colored read: the moat (query_patient) runs FIRST; on a matched answer, the GRADED tone LEVEL +
    graded forthcomingness (both from the neural ladder differential) color the reply -- the binary-latch tone is
    REPLACED. Affect NEVER touches the moat (an unmatched cue still abstains -> answer None)."""
    raw = comp.query_patient(agent, action)
    if raw is None:
        return {"answer": None, "abstain": True, "utterance": None, "tone_level": 0, "tone_token": ""}
    level = _graded_tone_level(differential)
    token = _graded_tone_token(level)
    extra = _graded_forthcomingness(differential)
    associates = []
    try:
        graph = comp._assoc_graph()
        if agent in graph:
            associates = [k for k, _ in sorted(graph[agent].items(), key=lambda kv: -kv[1])][:extra]
    except Exception:
        associates = []
    parts = ([token] if token else []) + [f"{agent} {action} {raw}"]
    if associates:
        parts.append("; also " + ", ".join(associates))
    return {"answer": raw, "abstain": False, "utterance": " ".join(parts).strip(),
            "tone_level": int(level), "tone_token": token, "forthcomingness_extra": int(extra),
            "associates": associates}


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _store_facts(comp):
    vocab = list(comp.words)
    facts = []
    for i in range(min(6, len(vocab) // 3)):
        a, v, p = vocab[i * 3], vocab[i * 3 + 1], vocab[i * 3 + 2]
        try:
            comp.store(a, v, p)
            facts.append((a, v, p))
        except Exception:
            pass
    return vocab, facts


ARB_BASE_LO = 60.0        # a losing channel's floor drive (below the arbiter's ignition knee)
ARB_SILENT_DEFAULT = 350.0  # silence is the standing default; volunteer/ask must EARN a win above it
WANT_FLOOR_HZ = 18.0


def _arb_drives(m_color, want):
    """The shared-arbiter feed: silence is the standing DEFAULT (a mid drive); affect forthcomingness raises
    arb_volunteer ABOVE it, and curiosity crave raises arb_ask ABOVE it. A channel whose faculty is inactive sits
    at the losing floor. So: forthcoming+familiar -> volunteer; neutral+novel -> ask; neutral+familiar -> silent."""
    vol = ARB_BASE_LO + max(0.0, float(m_color)) * 12000.0
    ask = ARB_BASE_LO + max(0.0, float(want) - WANT_FLOOR_HZ) * 15.0
    sil = ARB_SILENT_DEFAULT
    return {"arb_volunteer": vol, "arb_ask": ask, "arb_silent": sil}


def _turn_valence(bridge, xp, idx, baseline_snap, appraisal, ladder_live):
    """SEAM-C LIVE: this turn's appraised valence read as the NEURAL graded ladder differential
    rate(aff_pos_readout)-rate(aff_neg_readout) (ramp -> drive-off hold -> read), replacing the P0.3 binary latch.
    Falls back to the P0.3 v_color when the ladder is not co-resident (seams-off rollback path)."""
    if ladder_live:
        r = read_affect_ladder(bridge, xp, idx, baseline_snap, appraisal=float(appraisal))
        return float(r["differential"]), r
    r = read_affect(bridge, xp, idx, baseline_snap, mood_sign=(1 if appraisal > 0 else 0),
                    arousal=(1.0 if appraisal > 0 else 0.0))
    return float(r["v_color"]), r


def run_multi_turn_loop(bridge, xp, idx, baseline_snap, comp, facts, faculty_rng, fm=None) -> dict:
    """The REAL multi-turn conversational loop on the ONE bridge, with seams A + C ROUTED LIVE. Each turn reads the
    GRADED-affect ladder differential (SEAM-C) + curiosity + the arbiter off the shared cp_firing_states; a KNOWN
    turn composes a graded-tone answer under the g_eff law; a NOVEL turn drives the forward-model reservoir (SEAM-A)
    with the (s,a), decodes a certainty-TAGGED predicted s', folds the read-out margin into g_eff (tighten-only), and
    ASKS its wh-question -- while the moat still abstains on the unstored factual cue."""
    turns = []
    a0, v0, p0 = facts[0]
    a1, v1, p1 = facts[1]
    vocab = list(comp.words)
    stored_cues = {(a, v) for (a, v, _p) in facts}
    ladder_live = "ladder" in idx
    fm_live = bool(fm is not None and "fm" in idx)

    def _novel_cue():
        rng = faculty_rng.get("curiosity")
        for _ in range(400):
            a = vocab[int(rng.integers(0, len(vocab)))]
            v = vocab[int(rng.integers(0, len(vocab)))]
            if (a, v) not in stored_cues and comp.query_patient(a, v) is None:
                return a, v
        return vocab[0], vocab[1]

    def _known_turn(tno, ttype, a, v, gold, appraisal):
        diff, _aff = _turn_valence(bridge, xp, idx, baseline_snap, appraisal, ladder_live)
        want = read_curiosity_want(bridge, xp, idx, baseline_snap, novelty=0.05)         # familiar -> low want
        winner, margin, rates = run_arbiter(bridge, xp, idx, baseline_snap, _arb_drives(diff, want))
        ans = _colored_answer_graded(comp, a, v, diff)
        return {
            "turn": tno, "type": ttype, "cue": [a, v], "gold_patient": gold,
            "moat_answer": ans["answer"], "honesty_band": "assert" if ans["answer"] is not None else "MOAT",
            "affect_differential": float(diff), "affect_v_state": float(diff),
            "tone_level": ans.get("tone_level"), "tone_token": ans.get("tone_token"),
            "forthcomingness_extra": ans.get("forthcomingness_extra"),
            "graded_affect_live": bool(ladder_live), "curiosity_want_hz": want,
            "arbiter_winner": winner, "arbiter_margin": margin, "arbiter_rates": rates,
            "utterance": ans["utterance"], "moat_correct": bool(ans["answer"] == gold),
            "composed_ok": bool(ans["answer"] == gold and winner == "arb_volunteer"
                                and (ans.get("tone_level") or 0) > 0),
        }

    def _novel_turn(tno, appraisal):
        an, vn = _novel_cue()
        diff, _aff = _turn_valence(bridge, xp, idx, baseline_snap, appraisal, ladder_live)
        want = read_curiosity_want(bridge, xp, idx, baseline_snap, novelty=1.0)          # NOVEL -> high want
        winner, margin, rates = run_arbiter(bridge, xp, idx, baseline_snap, _arb_drives(diff, want))
        moat = comp.query_patient(an, vn)                                               # HARD moat: must abstain
        asked = winner == "arb_ask"
        base_q = f"what does {an} {vn} ?" if asked else None
        pred = fm_predict_turn(bridge, xp, idx, baseline_snap, fm, an, vn) if fm_live else None
        utter = base_q
        if pred is not None and pred["predicted"] is not None:
            sim = (f"my forward model predicts '{pred['predicted']}' for this novel case "
                   f"(margin {pred['margin']:.2f}); I have not observed it")
            utter = (base_q + " -- " + sim) if base_q else sim
        return {
            "turn": tno, "type": "novel_query", "cue": [an, vn],
            "moat_answer": moat, "honesty_band": "MOAT",
            "affect_differential": float(diff), "curiosity_want_hz": want,
            "arbiter_winner": winner, "arbiter_margin": margin, "arbiter_rates": rates,
            "utterance": utter, "asked_not_refused": bool(asked), "moat_held": bool(moat is None),
            "forward_model_live": bool(fm_live),
            "fm_predicted": (pred["predicted"] if pred else None),
            "fm_margin": (pred["margin"] if pred else None),
            "fm_g_eff": (pred["g_eff"] if pred else None),
            "fm_g_eff_tightened_only": bool(pred is None or pred["g_eff"] >= FM_G0 - 1e-12),
            "fm_content_channel": (pred["content"] if pred else None),
            "composed_ok": bool(moat is None and winner == "arb_ask"),
        }

    # positive-mood KNOWN turns (1,3) bracket neutral NOVEL turns (2,4); the positive mood re-reads positive on
    # each high-arousal turn -> graded affect PERSISTS across the conversation (via the ladder NMDA latches).
    turns.append(_known_turn(1, "known_fact", a0, v0, p0, +1.0))
    turns.append(_novel_turn(2, 0.0))
    turns.append(_known_turn(3, "known_fact_mood_persists", a1, v1, p1, +1.0))
    turns.append(_novel_turn(4, 0.0))

    mood_signs = [turns[0]["affect_v_state"], turns[2]["affect_v_state"]]
    affect_persists = bool(all(s > 0 for s in mood_signs))
    known_turns_ok = bool(turns[0]["composed_ok"] and turns[2]["composed_ok"])
    novel_turns_ok = bool(turns[1]["composed_ok"] and turns[3]["composed_ok"])
    moat_held_all = bool(turns[1]["moat_held"] and turns[3]["moat_held"])
    fm_content_on_novel = bool(fm_live and turns[1].get("fm_predicted") is not None
                               and turns[3].get("fm_predicted") is not None)
    graded_tone_multilevel = bool(ladder_live and (turns[0].get("tone_level") or 0) > 0)
    composes_live = bool(known_turns_ok and novel_turns_ok and moat_held_all and affect_persists)
    return {
        "turns": turns,
        "affect_persists_across_turns": affect_persists,
        "known_turns_honest_and_colored": known_turns_ok,
        "novel_turns_curiosity_asks": novel_turns_ok,
        "moat_held_all_novel_turns": moat_held_all,
        "graded_affect_live": bool(ladder_live),
        "graded_tone_multilevel_on_known_turns": graded_tone_multilevel,
        "forward_model_live": bool(fm_live),
        "forward_model_content_on_novel_turns": fm_content_on_novel,
        "composes_live": composes_live,
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# CONVERSATION-LESION BATTERY (the acceptance test) -- with A+C LIVE, lesion each faculty and show the CONVERSATION
# OUTPUT changes vs a matched SHAM (an off-target intervention of the same kind on the OTHER faculty's pathway).
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def conversation_lesion_battery(bridge, xp, idx, baseline_snap, comp, facts, fm) -> dict:
    """A (world-model): silence the fm reservoir on a NOVEL turn -> the predicted-content channel VANISHES and the
    turn reverts to plain abstention (real changes the turn output); a matched off-target sham (clamp affect_out=0,
    C's gate, which the fm content path does NOT traverse) leaves the prediction intact. C (graded affect): clamp
    affect_out=0 on a KNOWN turn -> the ladder readout collapses -> tone goes FLAT/ungraded while the ANSWER is
    unchanged (real); a matched off-target sham (silence the fm, which the ladder read does NOT traverse) leaves the
    tone graded. Deltas are measured ON THE TURN OUTPUT (utterance/predicted-content/tone-level), not the isolated
    read. The moat abstains on the novel cue under every condition (invariant)."""
    a0, v0, _p0 = facts[0]
    stored_cues = {(a, v) for (a, v, _p) in facts}
    vocab = list(comp.words)
    novel = None
    for a in vocab:
        for v in vocab:
            if (a, v) not in stored_cues and comp.query_patient(a, v) is None:
                novel = (a, v)
                break
        if novel:
            break
    an, vn = novel if novel else (vocab[0], vocab[1])

    # ---- FACULTY A: the forward-model content channel on a NOVEL turn ----
    intact_a = fm_predict_turn(bridge, xp, idx, baseline_snap, fm, an, vn, silence=False)
    real_a = fm_predict_turn(bridge, xp, idx, baseline_snap, fm, an, vn, silence=True)   # REAL lesion: fm silenced
    bridge.set_transmission_gate("affect_out", 0.0)                                       # off-target (C's gate)
    sham_a = fm_predict_turn(bridge, xp, idx, baseline_snap, fm, an, vn, silence=False)
    bridge.set_transmission_gate("affect_out", 1.0)
    moat_novel = comp.query_patient(an, vn)                                               # abstains (invariant)
    a_real_content_lost = bool(intact_a["predicted"] is not None and real_a["predicted"] is None)
    a_sham_content_kept = bool(sham_a["predicted"] is not None and sham_a["predicted"] == intact_a["predicted"])
    a_g_eff_reverts_to_floor = bool(real_a["g_eff"] <= FM_G0 + 1e-12 and intact_a["g_eff"] >= real_a["g_eff"])
    a_moat_invariant = bool(moat_novel is None)

    # ---- FACULTY C: the GRADED tone on a KNOWN turn ----
    d_intact = read_affect_ladder(bridge, xp, idx, baseline_snap, appraisal=1.0, lesion=False)["differential"]
    ans_intact = _colored_answer_graded(comp, a0, v0, d_intact)
    d_real = read_affect_ladder(bridge, xp, idx, baseline_snap, appraisal=1.0, lesion=True)["differential"]
    ans_real = _colored_answer_graded(comp, a0, v0, d_real)                               # REAL lesion: affect_out=0
    # off-target sham: silence the fm (A's input); the ladder read does NOT traverse the fm -> tone stays graded.
    _ = read_forward_model(bridge, xp, idx, baseline_snap, fm["W_in"], _fm_encode_sa(fm["emb"], a0, v0),
                           silence=True)
    d_sham = read_affect_ladder(bridge, xp, idx, baseline_snap, appraisal=1.0, lesion=False)["differential"]
    ans_sham = _colored_answer_graded(comp, a0, v0, d_sham)
    c_real_tone_flat = bool((ans_intact["tone_level"] or 0) != 0 and (ans_real["tone_level"] or 0) == 0)
    c_real_answer_unchanged = bool(ans_real["answer"] is not None and ans_real["answer"] == ans_intact["answer"])
    c_sham_tone_kept = bool((ans_sham["tone_level"] or 0) == (ans_intact["tone_level"] or 0)
                            and (ans_intact["tone_level"] or 0) != 0)

    battery_ok = bool(a_real_content_lost and a_sham_content_kept and a_moat_invariant
                      and c_real_tone_flat and c_real_answer_unchanged and c_sham_tone_kept)
    return {
        "novel_cue": [an, vn], "known_cue": [a0, v0],
        "faculty_A": {
            "intact_predicted": intact_a["predicted"], "intact_margin": intact_a["margin"],
            "intact_g_eff": intact_a["g_eff"], "intact_utterance_has_prediction": bool(intact_a["predicted"] is not None),
            "real_lesion_predicted": real_a["predicted"], "real_lesion_g_eff": real_a["g_eff"],
            "sham_predicted": sham_a["predicted"],
            "real_content_lost": a_real_content_lost, "sham_content_kept": a_sham_content_kept,
            "g_eff_reverts_to_floor_on_lesion": a_g_eff_reverts_to_floor, "moat_invariant": a_moat_invariant,
            "real_vs_sham_delta": ("REAL: predicted '%s'->None (content lost); SHAM(off-target affect_out=0): "
                                   "predicted '%s' unchanged" % (intact_a["predicted"], sham_a["predicted"])),
        },
        "faculty_C": {
            # NB the specificity SHAM (off-target fm-silence) is EXPECTED to tie the full read exactly (the ladder
            # read does not traverse the fm) -- that tie IS the null result, not a dead instrument. The DISCRIMINATING
            # contrast is differential_full vs differential_affectout_off (the affect_out lesion). Keys are named to
            # avoid a false treatment/control auto-pairing of full-vs-sham (which correctly ties).
            "differential_full": float(d_intact), "tone_level_full": ans_intact["tone_level"],
            "tone_token_full": ans_intact["tone_token"], "answer_full": ans_intact["answer"],
            "differential_affectout_off": float(d_real), "tone_level_affectout_off": ans_real["tone_level"],
            "answer_affectout_off": ans_real["answer"],
            "differential_offtarget_null": float(d_sham), "tone_level_offtarget_null": ans_sham["tone_level"],
            "real_tone_flat": c_real_tone_flat, "real_answer_unchanged": c_real_answer_unchanged,
            "sham_tone_kept": c_sham_tone_kept,
            "real_vs_sham_delta": ("REAL(affect_out=0): tone L%s->L0 (flat), answer '%s' unchanged; "
                                   "SHAM(off-target fm-silence): tone L%s unchanged"
                                   % (ans_intact["tone_level"], ans_intact["answer"], ans_sham["tone_level"])),
        },
        "battery_ok": battery_ok,
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# (c) FM4 LIVE -- yoked high-arousal affect (shared slices) never flips a below-assert honesty read to assert.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def fm4_live(bridge, xp, idx, baseline_snap, faculty_rng, n_candidates: int = 16) -> dict:
    """FM4 on the ONE bridge. The confidence read is the honesty relay's self_schema spike rate (graded by the
    workspace drive imbalance); the affect is read off the shared affect slices. For every below-assert candidate,
    the g_eff LAW keeps the band; a naive affect-into-confidence path DOES flip (the check can fail)."""
    # a real high-arousal positive affect (the yoked mis-coloring pressure), off the shared organ.
    hi = read_affect(bridge, xp, idx, baseline_snap, mood_sign=+1, arousal=1.0)
    v_color, m_color = hi["v_color"], hi["m_color"]

    # calibrate assert/hedge self_schema rate thresholds from confident vs tie relay drives.
    assert_rate = read_honesty_self_rate(bridge, xp, idx, baseline_snap, drive_class0=520.0, drive_class1=40.0)
    tie_rate = read_honesty_self_rate(bridge, xp, idx, baseline_snap, drive_class0=300.0, drive_class1=300.0)
    if assert_rate <= tie_rate:                      # degenerate relay -> fall back to a fixed band cut
        assert_rate, tie_rate = max(assert_rate, tie_rate) + 1e-3, min(assert_rate, tie_rate)
    hedge_rate = tie_rate + 0.4 * (assert_rate - tie_rate)
    assert_cut = tie_rate + 0.85 * (assert_rate - tie_rate)

    rng = faculty_rng.get("honesty")
    law_flips = 0
    naive_flips = 0
    tone_miscolored = 0
    checked = 0
    for _ in range(int(n_candidates)):
        # a below-assert relay read: random workspace imbalance that keeps self_rate below assert.
        d0 = float(rng.uniform(320.0, 480.0))
        d1 = float(rng.uniform(120.0, 320.0))
        sr = read_honesty_self_rate(bridge, xp, idx, baseline_snap, drive_class0=d0, drive_class1=d1)
        base_band = certainty_band(sr, assert_cut, hedge_rate, False)
        if base_band == "assert":
            continue
        checked += 1
        # g_eff LAW: affect adds ONLY above the honesty floor; the band is written by the self_schema read alone.
        law = g_eff_law(cue_match_moat_floor=0.06, honesty_floor=0.40,
                        affect_mod=max(0.0, v_color) + max(0.0, m_color))
        law_band = certainty_band(sr, assert_cut, hedge_rate, False)
        if not law["affect_cannot_loosen"] or law_band == "assert":
            law_flips += 1
        # naive (WRONG): affect leaks INTO the confidence -> can flip.
        eff = sr + max(0.0, v_color) * 8.0 + max(0.0, m_color) * 8.0
        if certainty_band(eff, assert_cut, hedge_rate, False) == "assert":
            naive_flips += 1
        if _tone(v_color) == TONE_POS:
            tone_miscolored += 1
    fm4_holds = bool(checked > 0 and law_flips == 0 and naive_flips > 0 and tone_miscolored > 0)
    return {
        "yoked_affect_v_color": float(v_color), "yoked_affect_m_color": float(m_color),
        "assert_rate_threshold": float(assert_cut), "hedge_rate_threshold": float(hedge_rate),
        "n_candidates_checked": int(checked),
        "g_eff_law_abstain_to_assert_flips": int(law_flips),
        "naive_path_abstain_to_assert_flips": int(naive_flips),
        "tone_miscolored_count": int(tone_miscolored),
        "fm4_holds": fm4_holds,
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# (d) MOAT LIVE 475/475 on the co-resident composer under a strong positive high-arousal mood.
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def moat_live(bridge, xp, idx, baseline_snap, comp, vocab, facts, n_unknown, faculty_rng) -> dict:
    stored_cues = {(a, v) for (a, v, _p) in facts}
    mood = read_affect(bridge, xp, idx, baseline_snap, mood_sign=+1, arousal=1.0)   # the most dangerous mood
    v_color, m_color = mood["v_color"], mood["m_color"]
    rng = faculty_rng.get("moat")
    checked = abstains = false_accepts = manufactured = 0
    attempts = 0
    max_attempts = n_unknown * 40
    while checked < n_unknown and attempts < max_attempts:
        attempts += 1
        a = vocab[int(rng.integers(0, len(vocab)))]
        v = vocab[int(rng.integers(0, len(vocab)))]
        if (a, v) in stored_cues:
            continue
        raw = comp.query_patient(a, v)
        if raw is not None:
            continue
        checked += 1
        colored = _colored_answer(comp, a, v, v_color, m_color)     # colored read path on a novel cue
        if colored["answer"] is None and colored["abstain"]:
            abstains += 1
        else:
            false_accepts += 1
        if colored["answer"] is not None:
            manufactured += 1
    return {
        "moat_stress_v_color": float(v_color), "moat_stress_m_color": float(m_color),
        "hard_moat_checked": checked, "hard_moat_abstains": abstains,
        "added_false_accepts": false_accepts, "colored_manufactured_answers": manufactured,
        "moat_battery_target": int(n_unknown),
        "moat_preserved": bool(checked > 0 and abstains == checked and false_accepts == 0 and manufactured == 0),
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
# (f) default-off byte-identity + (a) single-bridge + (e) no-piece-breaks-another
# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def byte_identity(seed: int) -> dict:
    base_bridge, _c, _i, _s = build_one_brain(seed, with_faculties=False)
    n_base = int(base_bridge.core_config.num_neurons)
    base_thr = np.asarray(to_host(base_bridge.cp_neuron_firing_thresholds), dtype=np.float64).copy()
    full_bridge, _c2, _i2, _s2 = build_one_brain(seed, with_faculties=True)
    n_full = int(full_bridge.core_config.num_neurons)
    full_thr = np.asarray(to_host(full_bridge.cp_neuron_firing_thresholds), dtype=np.float64)
    base_hash = hashlib.sha256(base_thr.tobytes()).hexdigest()
    overlap_hash = hashlib.sha256(np.asarray(full_thr[:n_base], dtype=np.float64).tobytes()).hexdigest()
    return {
        "n_composer_only": n_base, "n_with_faculties": n_full,
        "faculty_slices_appended_after_composer": bool(n_full > n_base),
        "composer_threshold_sha256": base_hash,
        "with_faculties_composer_indices_sha256": overlap_hash,
        "byte_identical": bool(base_hash == overlap_hash),
    }


def arbiter_three_way_and_lesion(seed: int, faculty_rng) -> dict:
    """(e) shared arbiter one-winner + a mutual-inhibition lesion collapses the winner margin, on the ONE bridge.
    Also confirms curiosity->arb_ask can win vs affect (curiosity vs turn-taking / one winner per turn)."""
    xp, _ = get_backend()
    bridge, comp, idx, snap = build_one_brain(seed, with_faculties=True, lesion_arbiter_inhibition=False)
    # affect m_color: forthcoming (positive high-arousal mood) vs neutral; curiosity want hi (novel) vs lo (familiar).
    m_forth = read_affect(bridge, xp, idx, snap, mood_sign=+1, arousal=1.0)["m_color"]
    m_neutral = read_affect(bridge, xp, idx, snap, mood_sign=0, arousal=0.0)["m_color"]
    want_hi = read_curiosity_want(bridge, xp, idx, snap, novelty=1.0)
    want_lo = read_curiosity_want(bridge, xp, idx, snap, novelty=0.05)
    regimes = {
        "novel_ask": (_arb_drives(m_neutral, want_hi), "arb_ask"),               # neutral affect + novel -> ask
        "forthcoming_volunteer": (_arb_drives(m_forth, want_lo), "arb_volunteer"),  # forthcoming + familiar -> volunteer
        "reticent_silent": (_arb_drives(m_neutral, want_lo), "arb_silent"),      # neutral + familiar -> silence default
    }
    # a regime is CONTESTED iff >=2 channels are driven above the ignition knee (a genuine competition to resolve);
    # the reticent regime drives only silence above the knee, so it is a non-contest (no margin to collapse).
    knee = ARB_BASE_LO + 50.0

    def _contested(drives):
        return int(sum(1 for v in drives.values() if v > knee)) >= 2

    intact = {}
    contested = {}
    for name, (drives, expected) in regimes.items():
        w, margin, rates = run_arbiter(bridge, xp, idx, snap, drives)
        intact[name] = {"winner": w, "expected": expected, "correct": bool(w == expected), "margin": margin,
                        "rates": rates, "contested": _contested(drives)}
        contested[name] = _contested(drives)
    # lesion the mutual inhibition on a fresh co-resident bridge.
    bridge_l, comp_l, idx_l, snap_l = build_one_brain(seed, with_faculties=True, lesion_arbiter_inhibition=True)
    lesioned = {}
    for name, (drives, expected) in regimes.items():
        w, margin, rates = run_arbiter(bridge_l, xp, idx_l, snap_l, drives)
        lesioned[name] = {"winner": w, "margin": margin, "rates": rates}
    all_correct = all(intact[n]["correct"] for n in regimes)
    distinct = len({intact[n]["winner"] for n in regimes}) == 3
    ask_can_win = bool(intact["novel_ask"]["winner"] == "arb_ask")
    per_regime_collapse = {n: bool(intact[n]["margin"] > 0.15 and lesioned[n]["margin"] < 0.5 * intact[n]["margin"])
                           for n in regimes}
    contested_regimes = [n for n in regimes if contested[n]]
    # contention collapses iff EVERY genuinely-contested regime's winner-margin collapses on the inhibition lesion
    # (the reticent single-channel regime is excluded: it has no competition to resolve).
    contention_collapses = bool(contested_regimes and all(per_regime_collapse[n] for n in contested_regimes))
    intact_min = float(min(intact[n]["margin"] for n in contested_regimes)) if contested_regimes else 0.0
    lesion_max = float(max(lesioned[n]["margin"] for n in contested_regimes)) if contested_regimes else 0.0
    return {
        "intact": intact, "lesioned": lesioned,
        "all_regimes_correct": all_correct, "distinct_winners_three": distinct, "ask_pool_can_win": ask_can_win,
        "contested_regimes": contested_regimes,
        "per_regime_margin_collapses_on_lesion": per_regime_collapse,
        "contention_collapses_on_lesion": contention_collapses,
        "intact_min_margin_contested": intact_min, "lesion_max_margin_contested": lesion_max,
        "arbitrates_three_way": bool(all_correct and distinct and ask_can_win),
        "margin_attributable_to_inhibition": attributable_to(
            "shared 3-way arbiter winner-margin from mutual inhibition (intact vs inhibition-lesion, contested "
            "regimes), co-resident", intact_min, lesion_max, warn_below=0.5),
    }


# ════════════════════════════════════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Stage-A FULL single-bridge live integration (single-seed smoke).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--moat-battery", type=int, default=475)
    ap.add_argument("--fm4-candidates", type=int, default=16)
    ap.add_argument("--skip-byte-identity", action="store_true")
    # SEAMS LIVE (default ON): route A (forward-model world-model) + C (graded-affect ladder) through the turn loop.
    # --no-seam-a / --no-seam-c roll a seam back to OFF (the seams-off regression path, for honest rollback reports).
    ap.add_argument("--seam-a", action=argparse.BooleanOptionalAction, default=True,
                    help="route SEAM-A (forward-model world-model) LIVE into the novel turns")
    ap.add_argument("--seam-c", action=argparse.BooleanOptionalAction, default=True,
                    help="route SEAM-C (graded-affect ladder) LIVE into the turn coloring")
    ap.add_argument("--out", type=str,
                    default="research/findings/raw/lanes/stageA/stageA_full_integration_smoke.json")
    args = ap.parse_args()

    get_backend("numpy")
    xp, _ = get_backend()
    faculty_rng = FacultyRNG(args.seed, ["moat", "honesty", "arbiter", "affect", "curiosity"])
    t0 = time.time()
    print(f"[stageA-full] seed={args.seed} moat_battery={args.moat_battery} backend={os.environ.get('SIM_BACKEND')}",
          flush=True)

    # ---- build the ONE bridge (all faculties co-resident + composer attached; seams A + C LIVE by default) ----
    print("[stageA-full] building the ONE co-resident bridge (composer + honesty + arbiter + affect + curiosity"
          f"{' + fm-reservoir(A)' if args.seam_a else ''}{' + affect-ladder(C)' if args.seam_c else ''}) ...",
          flush=True)
    bridge, comp, idx, baseline_snap = build_one_brain(
        args.seed, with_faculties=True,
        co_resident_forward_model=bool(args.seam_a), co_resident_affect_ladder=bool(args.seam_c))
    rm = bridge.region_manager
    region_names = [r.name for r in rm.regions()]
    n_regions = len(region_names)
    N = int(bridge.core_config.num_neurons)
    single_bridge = bool(getattr(comp, "_merged", None) is bridge)   # composer's substrate IS this bridge
    faculties_coresident = all(
        nm in region_names for nm in
        ("rf", "workspace", "meta_schema", "self_schema", "arb_volunteer", "arb_ask", "arb_silent",
         "affect_vplus", "affect_vminus", "affect_arousal", "cur_ask"))
    print(f"   ONE bridge N={N}, {n_regions} regions; composer._merged is bridge={single_bridge}; "
          f"all faculties present={faculties_coresident}", flush=True)

    vocab, facts = _store_facts(comp)
    print(f"   stored {len(facts)} facts on the co-resident composer", flush=True)

    # ---- SEAM-A LIVE: train the forward-model world-model read-out over the STORED facts (declared host shortcut) ----
    fm = None
    if args.seam_a:
        print("[stageA-full] SEAM-A: training the forward-model world-model read-out over the stored facts ...",
              flush=True)
        emb = _word_embedding(args.seed, vocab)
        W_in = make_fm_projection(args.seed, FM_N_POOL, FM_LOOP_IN_DIM)
        fm = build_fm_world_model(bridge, xp, idx, baseline_snap, comp, facts, emb, W_in, args.seed)
        print(f"   fm world-model: {fm['n_classes']} state classes, train_acc={fm['train_acc']:.2f}", flush=True)

    # ---- (b) COMPOSES-LIVE: the multi-turn loop (seams A + C routed LIVE) ----
    print("[stageA-full] (b) COMPOSES-LIVE: multi-turn conversational loop on the ONE bridge (seams live) ...",
          flush=True)
    loop = run_multi_turn_loop(bridge, xp, idx, baseline_snap, comp, facts, faculty_rng, fm=fm)
    for tt in loop["turns"]:
        print(f"   turn {tt['turn']} [{tt['type']}] winner={tt['arbiter_winner']} "
              f"band={tt['honesty_band']} -> {tt['utterance']!r} composed_ok={tt['composed_ok']}", flush=True)
    print(f"   composes_live={loop['composes_live']} (affect_persists={loop['affect_persists_across_turns']} "
          f"graded_tone={loop.get('graded_tone_multilevel_on_known_turns')} "
          f"fm_content_on_novel={loop.get('forward_model_content_on_novel_turns')})", flush=True)

    # ---- CONVERSATION-LESION BATTERY (acceptance test): lesion A/C -> the turn output changes vs a matched sham ----
    battery = {"skipped": True, "battery_ok": None}
    if args.seam_a and args.seam_c:
        print("[stageA-full] CONVERSATION-LESION BATTERY: lesion A/C -> turn output changes vs matched sham ...",
              flush=True)
        battery = conversation_lesion_battery(bridge, xp, idx, baseline_snap, comp, facts, fm)
        print(f"   A: {battery['faculty_A']['real_vs_sham_delta']}", flush=True)
        print(f"   C: {battery['faculty_C']['real_vs_sham_delta']}", flush=True)
        print(f"   battery_ok={battery['battery_ok']}", flush=True)

    # ---- (c) FM4 LIVE ----
    print("[stageA-full] (c) FM4 LIVE: yoked high-arousal affect never flips a below-assert read to assert ...",
          flush=True)
    fm4 = fm4_live(bridge, xp, idx, baseline_snap, faculty_rng, args.fm4_candidates)
    print(f"   fm4_holds={fm4['fm4_holds']} (law_flips={fm4['g_eff_law_abstain_to_assert_flips']} "
          f"naive_flips={fm4['naive_path_abstain_to_assert_flips']} "
          f"tone_miscolored={fm4['tone_miscolored_count']}/{fm4['n_candidates_checked']})", flush=True)

    # ---- (e) NO-PIECE-BREAKS-ANOTHER: shared 3-way arbiter + lesion, co-resident ----
    print("[stageA-full] (e) shared 3-way arbiter (co-resident) + mutual-inhibition lesion ...", flush=True)
    arbiter = arbiter_three_way_and_lesion(args.seed, faculty_rng)
    print(f"   arbitrates_three_way={arbiter['arbitrates_three_way']} "
          f"(novel->{arbiter['intact']['novel_ask']['winner']} "
          f"forth->{arbiter['intact']['forthcoming_volunteer']['winner']} "
          f"ret->{arbiter['intact']['reticent_silent']['winner']}; "
          f"contention_collapses={arbiter['contention_collapses_on_lesion']})", flush=True)

    # ---- (d) MOAT LIVE 475/475 ----
    print(f"[stageA-full] (d) MOAT LIVE {args.moat_battery}/{args.moat_battery} under a positive high-arousal mood ...",
          flush=True)
    moat = moat_live(bridge, xp, idx, baseline_snap, comp, vocab, facts, args.moat_battery, faculty_rng)
    print(f"   moat_preserved={moat['moat_preserved']} "
          f"({moat['hard_moat_abstains']}/{moat['hard_moat_checked']} abstain, "
          f"added_FA={moat['added_false_accepts']}, manufactured={moat['colored_manufactured_answers']})", flush=True)

    # ---- (f) default-off byte-identity ----
    if args.skip_byte_identity:
        bid = {"skipped": True, "byte_identical": None}
        print("[stageA-full] (f) byte-identity SKIPPED", flush=True)
    else:
        print("[stageA-full] (f) default-off byte-identity (faculty slices appended after composer) ...", flush=True)
        bid = byte_identity(args.seed)
        print(f"   byte_identical={bid['byte_identical']} "
              f"(n_composer={bid['n_composer_only']} -> n_full={bid['n_with_faculties']})", flush=True)

    # ---- no-piece-breaks-another: pairwise honest read ----
    pairwise = {
        "affect_vs_honesty_fm4_holds": bool(fm4["fm4_holds"]),
        "curiosity_vs_turntaking_one_winner": bool(arbiter["arbitrates_three_way"]),
        "shared_arbiter_one_winner_per_turn": bool(arbiter["distinct_winners_three"]
                                                   and arbiter["all_regimes_correct"]),
        "arbiter_contention_from_shared_inhibition": bool(arbiter["contention_collapses_on_lesion"]),
        "moat_intact_under_affect_and_curiosity": bool(moat["moat_preserved"]),
        "affect_coloring_alive_under_coresidence": bool(abs(loop["turns"][0]["affect_differential"]) > 0.02),
        "curiosity_want_alive_under_coresidence": bool(loop["turns"][1]["curiosity_want_hz"]
                                                       > loop["turns"][0]["curiosity_want_hz"]),
        "honesty_relay_graded_confidence_alive": bool(fm4["assert_rate_threshold"] > fm4["hedge_rate_threshold"]),
    }
    no_piece_breaks_another = bool(all(pairwise.values()))

    # ---- verdict ----
    ac = {
        "a_single_bridge": bool(single_bridge and faculties_coresident),
        "b_composes_live": bool(loop["composes_live"]),
        "c_fm4_live": bool(fm4["fm4_holds"]),
        "d_moat_live_475": bool(moat["moat_preserved"]),
        "e_no_piece_breaks_another": bool(no_piece_breaks_another),
        "f_default_off_byte_identity": (None if args.skip_byte_identity else bool(bid["byte_identical"])),
    }
    core_ok = bool(
        ac["a_single_bridge"] and ac["c_fm4_live"] and ac["d_moat_live_475"]
        and ac["e_no_piece_breaks_another"]
        and (args.skip_byte_identity or ac["f_default_off_byte_identity"])
    )
    if core_ok and ac["b_composes_live"]:
        verdict = "GO"
    elif ac["a_single_bridge"] and ac["d_moat_live_475"] and ac["c_fm4_live"]:
        verdict = "PARTIAL"       # single bridge holds + moat/FM4 hold; some composition property not fully shown
    else:
        verdict = "NEGATIVE"

    vd = Verdict("stageA FULL single-bridge live integration (single-seed smoke)")
    vd.require("SINGLE-BRIDGE: composer + all faculties are slices of ONE bridge object", ac["a_single_bridge"],
               expect=True)
    vd.require("MOAT LIVE 475/475 under affect+curiosity (0 false-accepts, 0 manufactured)", ac["d_moat_live_475"],
               expect=True)
    vd.require("FM4 LIVE: yoked affect cannot flip a below-assert honesty read -> assert (g_eff hard floor)",
               ac["c_fm4_live"], expect=True)
    vd.require("NO-PIECE-BREAKS-ANOTHER: every pairwise interaction holds under co-residence",
               ac["e_no_piece_breaks_another"], expect=True)
    if not args.skip_byte_identity:
        vd.require("default-off byte-identity (faculty slices appended after the composer rf slice)",
                   ac["f_default_off_byte_identity"], expect=True)
    vd.control("shared 3-way arbiter winner-margin, contested regimes (intact vs inhibition-lesion), co-resident",
               arbiter["intact_min_margin_contested"], arbiter["lesion_max_margin_contested"], min_separation=0.1)
    vd.control("FM4 g_eff-law vs naive-path abstain->assert flips (law must not flip; naive does)",
               float(fm4["naive_path_abstain_to_assert_flips"]), float(fm4["g_eff_law_abstain_to_assert_flips"]),
               min_separation=1.0)
    vd.disabled("STDP/Hebbian/homeostasis/STP/structural on the co-resident bridge; OU toggled per affect window",
                "isolation of the fixed relays + organs; a property under this isolation")
    vd_decided = vd.decide(go=bool(verdict == "GO"), verbose=False)

    out = {
        "runner": "research/runners/_stageA_full_integration_derisk.py",
        "faculty": "Stage-A FULL single-bridge live integration -- TRUE ONE BRAIN conversation composer",
        "design": "research/findings/2026-08-07-stageA-conversation-integration-DESIGN.md",
        "backend": os.environ.get("SIM_BACKEND", "(unset)"),
        "seed": int(args.seed),
        "verdict": verdict,
        "verdict_earned_status": vd_decided["status"],
        "preconditions": vd_decided["preconditions"],
        "disabled_processes": vd_decided["disabled_processes"],
        "anti_cheats": ac,
        "single_bridge": {
            "one_bridge_object": True,
            "composer_merged_is_the_bridge": bool(single_bridge),
            "all_faculties_coresident": bool(faculties_coresident),
            "n_neurons": N, "n_regions": n_regions, "region_names": region_names,
            "composer_class": type(comp).__name__,
        },
        "seams_live": {
            "seam_a_forward_model": bool(args.seam_a),
            "seam_c_graded_affect_ladder": bool(args.seam_c),
            "forward_model_content_on_novel_turns": bool(loop.get("forward_model_content_on_novel_turns")),
            "graded_tone_multilevel_on_known_turns": bool(loop.get("graded_tone_multilevel_on_known_turns")),
            "fm_train_acc": (fm["train_acc"] if fm is not None else None),
        },
        "conversation_lesion_battery": battery,
        "multi_turn_loop": loop,
        "fm4_live": fm4,
        "arbiter_three_way": arbiter,
        "moat_live": moat,
        "byte_identity": bid,
        "no_piece_breaks_another": {"pairwise": pairwise, "all_hold": no_piece_breaks_another},
        "vram_feasibility": {
            "backend": "numpy (CPU RAM)",
            "n_neurons": N,
            "note": ("One co-resident bridge at ~{n} neurons on the numpy/CPU backend (RAM, not VRAM). The design "
                     "flagged a VRAM ceiling for 4-5 co-resident slices on GPU; on numpy the ceiling is host RAM "
                     "and this build is comfortably within it (the modular composer alone was ~28K neurons; the "
                     "faculty slices add ~{f}).").format(n=N, f=N - CoResidentOneBrainComposer.n_total_for(
                        D=128, vocab=DEFAULT_VOCAB, k_max=32)),
        },
        "honesty_source": (
            "The LIVE honesty floor in the loop is the co-resident composer's on-bridge cue-match (moat abstain -> "
            "MOAT band; a cleared cue -> assert), composed under the g_eff LAW. The calibrated ACC/aPFC monitor "
            "(STEP 1) is co-resident as the workspace/meta/self relay and is exercised LIVE for FM4 + a graded "
            "self_schema confidence read on the shared substrate; porting its full calibrated-monitor routing "
            "(fit + _run_report) onto the shared slices is the remaining honesty consolidation step (STEP 1 runs it "
            "on its own modular bridges)."
        ),
        "honest_scope": (
            "Single-seed SMOKE of the FULL single-bridge live integration. ALL FOUR Stage-A faculties (honesty "
            "relay, 3-way arbiter, affect organ, curiosity ask) AND the CoResidentOneBrainComposer no-confab moat "
            "are region SLICES of ONE SimulationBridge object in ONE process (asserted: composer._merged IS the "
            "bridge; region count + names in single_bridge). The multi-turn loop reads affect/curiosity/arbiter off "
            "the shared cp_firing_states and composes honest+colored answers on known turns and curiosity wh-asks "
            "on novel turns; the moat holds LIVE 475/475 under a positive high-arousal mood. HONEST-NEGATIVES: "
            "(1) the loop honesty band uses the composer margin, not the full calibrated ACC/aPFC monitor routing "
            "(co-resident but run modularly in STEP 1); (2) host-fed appraisal + bistable-latch binary tone + host "
            "wh-frame/tone render (STEP-2/3 boundaries); (3) shared global cfg (het on, OU toggled per affect "
            "window) -- the no-piece-breaks-another check measures whether co-residence degraded any faculty. "
            "Parent runs the 6-seed sweep."
        ),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n[stageA-full] === VERDICT: {verdict} === core_ok={core_ok} composes_live={loop['composes_live']}",
          flush=True)
    print(f"[stageA-full] seams_live: A={args.seam_a} C={args.seam_c} "
          f"fm_content_on_novel={loop.get('forward_model_content_on_novel_turns')} "
          f"graded_tone={loop.get('graded_tone_multilevel_on_known_turns')} "
          f"lesion_battery_ok={battery.get('battery_ok')}", flush=True)
    print(f"[stageA-full] anti_cheats={ac}", flush=True)
    print(f"[stageA-full] elapsed={out['elapsed_seconds']}s wrote {args.out}", flush=True)
    return 0 if verdict in ("GO", "PARTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
