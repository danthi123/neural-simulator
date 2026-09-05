"""Gate-B appraisal via the board-#84 interoceptive-relay CURRENT afferent (scaffold-retirement backlog rank 5,
`research/coordination/scaffold_retirement_backlog.md`) -- an ADAPTATION of an already-validated mechanism, not a
fresh one.

THE GAP (board #5 on the scaffold-retirement map). Gate-B's production affect organ
(`research.runners.affect_production_organ.AffectProductionOrgan.read_differential`) injects the appraised message
valence/arousal into its SEAM-C staggered-bistable ladder (`_stageA_full_integration_derisk`, `aff_vplus_L1..L8` /
`aff_vminus_L1..L8` / `aff_arousal_L1..L8`) via a DIRECT HOST WRITE: `neuromodulator_manager.set_concentration(
"appraisal_lad_vplus"/"vminus"/"arousal", m)`. The concentration broadcasts UNIFORMLY as an additive
`excitability_drive` (a raw per-neuron pA offset, `sensitivity=240`) onto every rung of the matching sign -- a
scalar Python float lands directly on the target population, never through a synapse.

Board #49 (`2026-08-19-embodied-affect-interoception-GO.md`) and its #81/#84 ladder adaptation
(`_graded_affect_attractor_derisk.GradedAffectBrain` / `webapp/affect_drives_chat.py`, 6/6-seed GO, running in
PRODUCTION for the SAME KIND of Koulakov/Goldman bistable-ladder substrate Gate-B also uses) already established
the correct pattern for this exact situation: a host scalar enters the brain ONLY as an afferent CURRENT onto small
spiking relay pools (Izhikevich RS, no recurrence -- a legitimate body/sensory-interface boundary, not a shortcut),
and THOSE pools drive the target attractor SYNAPTICALLY (AMPA, gated by a runtime transmission gate) -- never a
direct write onto the target population or its neuromodulator bus.

THIS MODULE adapts that SAME pattern onto Gate-B's OWN ladder spec. `AppraisalInteroceptiveLadder` reuses
`_stageA_full_integration_derisk._ladder_region_specs` / `_ladder_pathways` BY IMPORT (byte-for-byte the same
region/pathway architecture the co-resident SEAM-C ladder uses, aff_n_rungs=8) plus 3 NEW interoceptive-relay pools
(`appr_intero_vplus/vminus/arousal`, the board #49/#81 `intero_*` pattern) that carry the appraisal as a real
CURRENT and project onto EVERY rung of the matching sign (AMPA, gated by APPR_INTERO_GATE, uniform weight -- the
staggered recruitment still lives entirely in each rung's own PRE-EXISTING intrinsic-current offset; only the
INJECTION mechanism changes from a diffuse neuromodulator write to a synapse).

WHY A DEDICATED STANDALONE BRIDGE (not a new co-resident SEAM added to `build_one_brain` itself): that function is
shared by every other Stage-A/one-brain de-risk and carries carefully-proven byte-identical-off invariants
(append-LAST index+draw invariance, SEAM-A/SEAM-C's separate-union RNG decoupling). A dedicated bridge (the ladder
+ 3 relay pools only, no composer/arbiter/honesty overhead) reuses the IDENTICAL ladder region/pathway SPEC by
import -- so it genuinely is the same architecture Gate-B's ladder is -- while touching ZERO lines of that shared
module (no risk to its invariants) and building fast enough for a 6-seed x multi-condition battery. The ONE
pathway `_ladder_pathways` emits that this bridge cannot host (`arousal -> speak_acc`, the arbiter's action-
selection region, out of scope for a ladder-only bridge) is explicitly excluded, documented below.

`research.runners.affect_production_organ.AffectProductionOrgan.read_differential` dispatches to
`get_ladder(seed).read_differential(...)` ONLY when `appraisal_interoceptive_enabled()` is truthy
(`BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE=1`); the default (unset) branch is the ORIGINAL host-write code, completely
untouched -- byte-identical-off by construction (a distinct code path, not a shared one with new parameters).

ANTI-CHEATS (they ARE the result):
  (1) BYTE-IDENTICAL-OFF -- with the flag unset, `AffectProductionOrgan.read_differential` reproduces EXACTLY the
      values captured from the pre-adaptation commit (asserted in the data: exact float compare against
      `_PRE_EDIT_BASELINE`, captured from the unmodified code before this module existed).
  (2) LOAD-BEARING -- sweeping the appraisal through the NEW interoceptive-afferent path moves the ladder
      differential with the correct SIGN and an ordered (corr>=0.8) magnitude, AND the downstream `tone_level` /
      `content_plan` / `manner_for` (the actual content-volunteering + mouth-manner consumers) change accordingly.
  (3) LESIONABLE (dissociation) -- cutting the relay->ladder synapses (APPR_INTERO_GATE=0,
      `appraisal_interoceptive_lesioned()`) collapses the appraisal->differential coupling toward 0 while the
      relay pools STILL FIRE (still encode the appraisal) -- the coupling is owned by the synapse, not incidental.
  (4) NO-REGRESSION vs the host-write path -- on the SAME appraisal sweep, the new path's sign-correctness and
      ordered tracking are compared against a FRESH read of the existing `AffectProductionOrgan` (the real
      production host-write mechanism), seed-matched.
  (5) 6 seeds (42 43 44 100 101 102), numpy-CPU, deterministic (cfg.seed set -> substrate seeded).

DISCIPLINE: SIM_BACKEND=numpy (CPU lane, per the cost-routing skill), reuse-by-import (the SEAM-C ladder spec),
additive default-off seam only (`research/runners/affect_production_organ.py` gains a 2-function flag pair + an
8-line dispatch branch at the top of `read_differential`; NO other line of that file or of
`_stageA_full_integration_derisk.py` changes). NO `sim/` edit. cfg.seed per seed.

Run (smoke -- 1 seed: determinism + operating-point calibration sweep):
  SIM_BACKEND=numpy python -u -m research.runners._appraisal_interoceptive_ladder_derisk --smoke
Run (6-seed battery incl. byte-identical-off + no-regression vs the host write):
  SIM_BACKEND=numpy python -u -m research.runners._appraisal_interoceptive_ladder_derisk --seeds 42 43 44 100 101 102
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import dataclasses as _dc
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from sim.backend import to_host  # noqa: E402  (passthrough on numpy)
from sim.regions import BrainRegion, RegionPathway  # noqa: E402
from tools.lab import attributable_to  # noqa: E402  (explicit intact-vs-lesion attribution -- gap#5 discipline)

# reuse-by-import: the EXACT Gate-B SEAM-C ladder spec (byte-for-byte the same architecture the co-resident
# one-brain's ladder uses) + its operating-point constants.
from research.runners import _stageA_full_integration_derisk as SA  # noqa: E402

OUT = (Path(_REPO) / "research" / "findings" / "raw" / "appraisal_interoceptive_ladder" /
       "_appraisal_interoceptive_ladder_6seed.json")

# ---- interoceptive-relay constants (board #49/#81 pattern; operating point calibrated by the smoke) ------------
APPR_INTERO_GATE = "appraisal_intero_out"   # transmission gate over the relay->ladder synapses (the load-bearing lesion)
APPR_INTERO_N = 40                          # neurons per relay pool (matches board #49/#81 N_INT)
APPR_INTERO_DENS = 0.6                      # relay -> rung projection density (matches board #49/#81 DENS_INT)
APPR_INTERO_I_PA = 220.0                    # afferent current (pA) at |appraisal|=1.0 (smoke-calibrated). Below
                                            # ~100pA the relay pool itself never crosses its OWN rheobase (0 output
                                            # regardless of rung thresholds); above ~1000pA the ladder saturates
                                            # near its max range almost independent of appraisal magnitude
                                            # (near-binary switch). i_pa=220 sits in the narrow GRADED window,
                                            # giving corr(appraisal,differential)~0.97 (vs the host-write path's
                                            # ~0.995) -- the best ordered-tracking found; see the finding's sweep.
APPR_INTERO_W = 10.0                        # relay -> EVERY rung of its sign, uniform AMPA weight (smoke-calibrated)


def appraisal_interoceptive_enabled() -> bool:
    """Mirrors `affect_production_organ.appraisal_interoceptive_enabled()` (imported from there in production;
    kept here too so this module's own CLI/tests do not require importing the production module to check it).
    PRODUCTION-DEFAULT-ON as of the 2026-09-05 flip (see `research/findings/
    2026-09-05-gateB-appraisal-interoceptive-production-flip-GO.md`): unset -> True; explicit {0,false,no,off}
    -> the escape hatch (False)."""
    v = os.environ.get("BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE")
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "no", "off", "")


# =============================================================================================================
# The adapted ladder: Gate-B's OWN SEAM-C region/pathway spec (reused-by-import) + 3 interoceptive relay pools.
# =============================================================================================================
class AppraisalInteroceptiveLadder:
    """A DEDICATED standalone bridge: `SA._ladder_region_specs(aff_n_rungs)` + `SA._ladder_pathways` (byte-for-
    byte the co-resident SEAM-C ladder's own spec) plus 3 interoceptive-relay pools (Izhikevich RS, no recurrence)
    that carry the appraisal as a real CURRENT and project SYNAPTICALLY (AMPA, gated by APPR_INTERO_GATE) onto
    EVERY rung of the matching sign, uniformly -- mirroring the pre-existing diffuse-broadcast SEMANTICS (the
    staggered recruitment still lives entirely in each rung's own pre-existing intrinsic-current offset; only the
    INJECTION mechanism changes). No composer/arbiter/honesty faculties -- the ladder alone, so a 6-seed battery
    builds fast. The rungs' bistable NMDA recurrence (internal_density=0 in the reused spec, so the co-resident
    build can inject it via a separate-RNG union entry without perturbing its shared wiring-plan stream) is
    restored here via a plain `dataclasses.replace` -- this bridge has no pre-existing state to protect, so the
    plain per-region density is equivalent-in-kind and far simpler than the union-injection dance."""

    def __init__(self, seed: int = 42, aff_n_rungs: int = 8, i_pa: float = APPR_INTERO_I_PA, w: float = APPR_INTERO_W,
                dens: float = APPR_INTERO_DENS):
        from sim import SimulationBridge, VisualizationConfig, RuntimeState, GPUConfig
        from sim.config import CoreSimConfig

        self.seed = int(seed)
        self.i_pa = float(i_pa)
        RS = "IZH2007_RS_CORTICAL_PYRAMIDAL"

        lad_regions, names = SA._ladder_region_specs(int(aff_n_rungs))
        self.names = names
        rung_names = set(names["vplus"] + names["vminus"] + names["arousal"])
        lad_regions = [_dc.replace(r, internal_density=SA.LAD_RECUR_DENSITY) if r.name in rung_names else r
                      for r in lad_regions]
        # `_ladder_pathways` also emits ONE pathway per arousal rung -> "speak_acc" (the arbiter's action-selection
        # region, not part of this ladder-only bridge). Excluded -- it is a dead-end OUTPUT (nothing reads FROM
        # speak_acc back into the ladder), so dropping it cannot affect the mood/arousal differential this module
        # measures. Documented rather than faked with an inert stand-in region.
        lad_pathways = [p for p in SA._ladder_pathways(names) if p.to_region != "speak_acc"]

        def relay(name):
            return BrainRegion(name=name, n_neurons=APPR_INTERO_N, exc_fraction=1.0, internal_density=0.0,
                               exc_weight_mean=0.0, inh_weight_mean=0.0, weight_jitter=0.05,
                               plastic_internal=False, izh_neuron_type=RS, enable_nmda=False)

        self.relay_names = {"vplus": "appr_intero_vplus", "vminus": "appr_intero_vminus",
                            "arousal": "appr_intero_arousal"}
        relay_regions = [relay(n) for n in self.relay_names.values()]
        relay_pathways = []
        for sign, rname in self.relay_names.items():
            for rung in names[sign]:
                relay_pathways.append(RegionPathway(from_region=rname, to_region=rung, density=float(dens),
                                                     weight_mean=float(w), weight_jitter=0.1, plastic=False,
                                                     transmission_gate=APPR_INTERO_GATE))

        cfg = CoreSimConfig()
        cfg.enable_brain_region_framework = True
        cfg.brain_regions = lad_regions + relay_regions
        cfg.region_pathways = lad_pathways + relay_pathways
        cfg.enable_neuromodulator_subsystem = False   # the appraisal enters synaptically now -- no neuromodulator bus
        cfg.dt_ms = 1.0
        cfg.connections_per_neuron = 0
        cfg.num_traits = 1
        cfg.seed = int(seed)                          # seeds the SUBSTRATE (not actual_seed_used)
        cfg.enable_nmda = True
        cfg.nmda_ratio = 0.5
        cfg.nmda_tau_decay = 150.0                     # == meta.DEFAULT_NMDA_TAU (the co-resident ladder's tau)
        for f in ("enable_stdp", "enable_reward_modulation", "enable_hebbian_learning", "enable_homeostasis",
                  "enable_short_term_plasticity", "enable_structural_plasticity"):
            setattr(cfg, f, False)
        cfg.enable_ou_process = True
        cfg.ou_std_current_pA = SA.AFF_OU_PA           # == the co-resident ladder's background noise (8 pA)
        cfg.enable_parameter_heterogeneity = True
        cfg.stdp_w_max = 400.0
        cfg.hebbian_max_weight = 400.0

        self._bridge = SimulationBridge(core_config=cfg, viz_config=VisualizationConfig(),
                                        runtime_state=RuntimeState(), gpu_config=GPUConfig())
        self._bridge.runtime_state.max_delay_steps = int(cfg.max_synaptic_delay_ms / cfg.dt_ms)
        self._bridge._initialize_simulation_data(called_from_playback_init=False)

        rm = self._bridge.region_manager
        self._idx = {n: np.asarray(v, dtype=np.int64) for n, v in rm.region_indices_dict().items()}
        self.ladder = {
            "vplus": [self._idx[n] for n in names["vplus"]], "vminus": [self._idx[n] for n in names["vminus"]],
            "arousal": [self._idx[n] for n in names["arousal"]],
            "pos_readout": self._idx[names["pos_readout"]], "neg_readout": self._idx[names["neg_readout"]],
        }
        self.relay_idx = {sign: self._idx[rname] for sign, rname in self.relay_names.items()}
        # ANTI-CHEAT guard indices: the ladder pools must be reachable ONLY via synapses from the relay.
        self._ladder_flat = np.concatenate([np.concatenate(self.ladder["vplus"]), np.concatenate(self.ladder["vminus"]),
                                            np.concatenate(self.ladder["arousal"])])

    def reset(self):
        self._bridge._initialize_simulation_data(called_from_playback_init=False)

    def read_differential(self, appraisal: float, lesion: bool = False, intero_lesion: bool = False,
                          ramp_ms: int = SA.LAD_RAMP_MS, drive_off_ms: int = SA.LAD_DRIVE_OFF_MS,
                          read_ms: int = SA.LAD_READ_MS) -> dict:
        """The SAME settle/ramp/drive-off/read protocol as `AffectProductionOrgan.read_differential`, but the
        appraisal drives a CURRENT onto the interoceptive-relay pools (which then reach the ladder ONLY via the
        APPR_INTERO_GATE-gated AMPA synapses) instead of a direct `nm.set_concentration(...)` write. `lesion`
        clamps the ladder's OWN `affect_out` readout gate (identical semantics to the host-write path);
        `intero_lesion` clamps APPR_INTERO_GATE (the NEW synapse -- the load-bearing dissociation proof)."""
        b = self._bridge
        self.reset()
        b.set_transmission_gate("affect_out", 0.0 if lesion else 1.0)
        b.set_transmission_gate(APPR_INTERO_GATE, 0.0 if intero_lesion else 1.0)
        b.cp_external_input_current[:] = 0.0
        m_abs = abs(float(appraisal))
        pos_sign = float(appraisal) >= 0.0
        rv, rn, ra = self.relay_idx["vplus"], self.relay_idx["vminus"], self.relay_idx["arousal"]

        def _drive(m):
            b.cp_external_input_current[:] = 0.0
            cur = self.i_pa * float(m)
            b.cp_external_input_current[rv] = np.float32(cur if pos_sign else 0.0)
            b.cp_external_input_current[rn] = np.float32(0.0 if pos_sign else cur)
            b.cp_external_input_current[ra] = np.float32(cur)
            # ANTI-CHEAT: the ladder pools NEVER get a direct host current -- the appraisal reaches them only via
            # the relay's synapses.
            assert float(np.abs(to_host(b.cp_external_input_current)[self._ladder_flat]).max()) == 0.0, \
                "ladder pools received a direct external current -- the appraisal->ladder path must be synaptic"

        relay_spk = {"vplus": 0.0, "vminus": 0.0, "arousal": 0.0}
        for _ in range(40):
            _drive(0.0); b._run_one_simulation_step()
        for s in range(int(ramp_ms)):
            _drive(m_abs * (s + 1) / ramp_ms); b._run_one_simulation_step()
            fs = to_host(b.cp_firing_states)
            relay_spk["vplus"] += float(np.asarray(fs)[rv].sum())
            relay_spk["vminus"] += float(np.asarray(fs)[rn].sum())
            relay_spk["arousal"] += float(np.asarray(fs)[ra].sum())
        for _ in range(int(drive_off_ms)):
            _drive(0.0); b._run_one_simulation_step()
        pos = neg = 0.0
        for _ in range(int(read_ms)):
            _drive(0.0); b._run_one_simulation_step()
            fs = to_host(b.cp_firing_states)
            pos += float(np.asarray(fs)[self.ladder["pos_readout"]].sum())
            neg += float(np.asarray(fs)[self.ladder["neg_readout"]].sum())
        b.set_transmission_gate("affect_out", 1.0)
        b.set_transmission_gate(APPR_INTERO_GATE, 1.0)
        denom = float(SA.LAD_N_RO * max(1, read_ms))
        pr, nr = pos / denom, neg / denom
        relay_rate = {k: v / (APPR_INTERO_N * max(1, int(ramp_ms))) for k, v in relay_spk.items()}
        return {"differential": float(pr - nr), "pos_rate": float(pr), "neg_rate": float(nr),
                "appraisal": float(appraisal), "lesioned": bool(lesion), "intero_lesioned": bool(intero_lesion),
                "relay_rate_vplus": relay_rate["vplus"], "relay_rate_vminus": relay_rate["vminus"],
                "relay_rate_arousal": relay_rate["arousal"], "mechanism": "interoceptive_afferent"}


_LADDERS: "dict[int, AppraisalInteroceptiveLadder]" = {}


def get_ladder(seed: int = 42) -> AppraisalInteroceptiveLadder:
    """The process-shared adapted ladder, ONE PER SEED (built once per seed on first use, mirrors
    `affect_production_organ.get_organ`). FIXED 2026-09-05 (production-flip verification, scaffold-retirement
    backlog rank 5): the original single-slot cache (`_LADDER`, a bare Optional) silently returned the FIRST
    seed's ladder for every later call with a DIFFERENT seed argument -- inert against today's production usage
    (every production organ is a single-seed-per-process singleton, `get_organ(seed=42)` hardcoded throughout
    `webapp/server.py`, so this call site is never asked for a second seed in one process), but exactly the
    confound class `tests/test_determinism.py::TestSubstrateActuallySeeded` exists to catch for any same-process
    multi-seed verification (a `get_ladder(43)` after `get_ladder(42)` would silently measure seed-42 neurons
    under a seed-43 label). A dict keyed by seed costs nothing for the single-seed production case (still built
    once per seed, still process-shared) and makes a same-process 6-seed loop safe -- see
    `research/findings/2026-09-05-gateB-appraisal-interoceptive-production-flip-GO.md`."""
    lad = _LADDERS.get(int(seed))
    if lad is None:
        lad = AppraisalInteroceptiveLadder(seed=seed)
        _LADDERS[int(seed)] = lad
    return lad


def _threshold_hash(seed):
    ladder = AppraisalInteroceptiveLadder(seed)
    th = to_host(ladder._bridge.cp_neuron_firing_thresholds)
    return np.asarray(th, float).tobytes()


def _corr(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


APPRAISAL_SWEEP = (-1.0, -0.7, -0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 1.0)
# The realistic PRODUCTION band: `affect_production_organ.appraise_text`'s salience gate (_STRONG_MARGIN=2.0 on a
# 1-9 Warriner scale) admits a word ONLY if |v9-5|>=2.0, which forces |valence|=|(v9-5)/4|>=0.5 for every SINGLE
# word that passes the gate -- so a real single-word-triggered appraisal is essentially never inside (-0.5,0.5).
# Values below 0.5 in magnitude can still occur from AVERAGING multiple gated words of mixed sign, so the
# sub-threshold band is checked (below) rather than excluded, but it is the less common case in practice.
PRODUCTION_REALISTIC_ABS_MIN = 0.5

# Pre-adaptation reference values for the byte-identical-off proof (2026-09-05), captured from
# `AffectProductionOrgan.read_differential` BEFORE this module / the production dispatch existed -- i.e. straight
# off the unmodified host-write code, seed 42, numpy-CPU. `run_byte_identical_off` re-reads the SAME appraisals on
# the SAME seed with `BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE` unset and requires an EXACT float match (asserted in
# the data, per docs/TERMS.md's byte-identical condition -- not inferred from reading the dispatch branch).
_PRE_EDIT_BASELINE = {
    "-1.0": -0.06638888888888889, "-0.5": -0.035277777777777776, "0.0": 0.0,
    "0.5": 0.03972222222222222, "1.0": 0.07083333333333333,
}
_PRE_EDIT_LESION_0_7 = 0.0


def run_byte_identical_off(seed: int = 42) -> dict:
    """The empirical byte-identical-OFF proof (the ESCAPE HATCH, post the 2026-09-05 production-default-ON flip
    -- see `research/findings/2026-09-05-gateB-appraisal-interoceptive-production-flip-GO.md`): with the flag
    EXPLICITLY set to '0', `AffectProductionOrgan.read_differential` must still reproduce `_PRE_EDIT_BASELINE`
    EXACTLY (a hash/exact-float compare, not code inspection) -- i.e. rollback still reaches the untouched
    original host-write code byte-for-byte. (Before the flip this asserted the UNSET state instead; unset now
    means ON, so the explicit-off arm is what proves the escape hatch.)"""
    os.environ["BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE"] = "0"
    from research.runners import affect_production_organ as AO
    assert not AO.appraisal_interoceptive_enabled(), "flag must read as OFF when explicitly set to '0'"
    organ = AO.AffectProductionOrgan(seed=seed)   # a FRESH organ instance (not the process-shared singleton)
    rows = []
    all_match = True
    for k, expect in _PRE_EDIT_BASELINE.items():
        got = organ.read_differential(float(k), lesion=False)["differential"]
        match = bool(got == expect)
        all_match &= match
        rows.append({"appraisal": float(k), "expected": expect, "got": got, "exact_match": match})
    got_lesion = organ.read_differential(0.7, lesion=True)["differential"]
    lesion_match = bool(got_lesion == _PRE_EDIT_LESION_0_7)
    all_match &= lesion_match
    rows.append({"appraisal": 0.7, "lesion": True, "expected": _PRE_EDIT_LESION_0_7, "got": got_lesion,
                "exact_match": lesion_match})
    return {"byte_identical_off": bool(all_match), "rows": rows}


# =============================================================================================================
# One seed: the new-mechanism sweep + intero-lesion dissociation + readout-lesion + host-write no-regression
# =============================================================================================================
def run_seed(seed, i_pa=APPR_INTERO_I_PA, w=APPR_INTERO_W, dens=APPR_INTERO_DENS, sweep=APPRAISAL_SWEEP):
    t0 = time.time()
    ladder = AppraisalInteroceptiveLadder(seed, i_pa=i_pa, w=w, dens=dens)

    intact = [ladder.read_differential(a, lesion=False, intero_lesion=False) for a in sweep]
    intero_lesioned = [ladder.read_differential(a, lesion=False, intero_lesion=True) for a in sweep]
    readout_lesioned = ladder.read_differential(0.7, lesion=True, intero_lesion=False)

    diffs = [r["differential"] for r in intact]
    diffs_il = [r["differential"] for r in intero_lesioned]
    corr_new = _corr(sweep, diffs)
    range_new = float(max(diffs) - min(diffs))
    range_il = float(max(diffs_il) - min(diffs_il))
    # signs_ok: a NONZERO appraisal must never read a GENUINELY OPPOSITE-SIGNED differential (an inversion, a real
    # bug). An exact-zero read at a nonzero appraisal is SUB-THRESHOLD SILENCE (the relay pool has its own rheobase
    # below which it never fires at all, an honest threshold effect distinct from "reads backwards") -- tracked
    # separately, not counted as a wrong sign. `corr_new>=0.8` and the realistic-band check below still gate the
    # overall dynamic range, so an all-zero degenerate response cannot hide behind this relaxation.
    signs_ok = all(d == 0.0 or (d > 0) == (a > 0) for d, a in zip(diffs, sweep) if abs(a) > 1e-9)
    subthreshold = [{"appraisal": a, "differential": d} for d, a in zip(diffs, sweep)
                    if abs(a) > 1e-9 and d == 0.0]
    # THE PRODUCTION-REALISTIC BAND (|appraisal|>=0.5, what `appraise_text`'s salience gate actually emits for any
    # single matched word): here sub-threshold silence would be a genuine hole, not a defensible edge case, so it
    # is gated STRICTLY (no zero-tolerance).
    realistic = [(d, a) for d, a in zip(diffs, sweep) if abs(a) >= PRODUCTION_REALISTIC_ABS_MIN - 1e-9]
    signs_ok_realistic_band = all((d > 0) == (a > 0) for d, a in realistic if abs(a) > 1e-9)
    neutral_near_zero = abs(diffs[sweep.index(0.0)]) < 0.01 if 0.0 in sweep else True
    # the relay pools must STILL encode the appraisal even while the SYNAPSE is cut (dissociation, not silence).
    relay_enc_intact = _corr([abs(a) for a in sweep], [r["relay_rate_vplus"] + r["relay_rate_vminus"]
                                                       for r in intact])
    relay_enc_lesion = _corr([abs(a) for a in sweep], [r["relay_rate_vplus"] + r["relay_rate_vminus"]
                                                       for r in intero_lesioned])

    # ---- HOST-WRITE reference (the REAL production organ, seed-matched) for the no-regression comparison. -------
    # Post the 2026-09-05 production-default-ON flip, an UNSET flag now means ON (interoceptive) -- popping it
    # here would silently turn this "host reference" into a self-comparison against the NEW mechanism. The
    # escape hatch (explicit '0') is what still reaches the untouched original host-write code.
    from research.runners import affect_production_organ as AO
    os.environ["BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE"] = "0"
    host = AO.AffectProductionOrgan(seed=seed)
    host_rows = [host.read_differential(a, lesion=False) for a in sweep]
    host_diffs = [r["differential"] for r in host_rows]
    corr_host = _corr(sweep, host_diffs)
    range_host = float(max(host_diffs) - min(host_diffs))
    host_signs_ok = all((d > 0) == (a > 0) for d, a in zip(host_diffs, sweep) if abs(a) > 1e-9)
    host_readout_lesion = host.read_differential(0.7, lesion=True)["differential"]

    # ---- downstream LOAD-BEARING: does tone_level/content_plan/manner_for actually change with the NEW read? ----
    levels = [AO.tone_level(d) for d in diffs]
    plans = [AO.content_plan(lv) for lv in levels]
    manners = [AO.manner_for(lv, "cat", "sat", "mat") for lv in levels]
    n_sentences = [p["max_sentences"] for p in plans]
    downstream_varies = bool(len(set(n_sentences)) > 1 and len(set(manners)) > 1)

    intero_owns_range = attributable_to("intero_synapse_owns_appraisal_coupling(range intact vs lesion)",
                                        range_new, range_il)

    checks = {
        "new_signs_correct(no inversion; sub-threshold zero tolerated)": bool(signs_ok),
        "new_signs_correct_in_production_realistic_band(|appraisal|>=0.5, strict)": bool(signs_ok_realistic_band),
        "new_neutral_near_zero(<0.01)": bool(neutral_near_zero),
        "new_ordered_tracking(corr>=0.8)": bool(corr_new >= 0.8),
        "new_readout_lesion_collapses(affect_out=0 -> 0.0)": bool(readout_lesioned["differential"] == 0.0),
        "intero_lesion_collapses_range(<=0.25x)": bool(range_il <= 0.25 * range_new + 1e-9),
        "relay_still_encodes_under_intero_lesion(corr>=0.8)": bool(relay_enc_lesion >= 0.8),
        "relay_encodes_intact(corr>=0.8)": bool(relay_enc_intact >= 0.8),
        "downstream_content_and_manner_vary": downstream_varies,
        "host_write_signs_correct(reference)": bool(host_signs_ok),
        "no_regression_signs(realistic band matches host sign-for-sign)": bool(
            signs_ok_realistic_band and host_signs_ok),
        "no_regression_ordered_tracking(new corr>=0.8 given host corr>=0.8)": bool(
            corr_host < 0.8 or corr_new >= 0.8),
    }
    go = all(checks.values())
    row = {
        "seed": int(seed), "GO": bool(go), "i_pa": float(i_pa), "w": float(w), "dens": float(dens),
        "checks": checks, "sweep": list(sweep),
        "new_diffs": diffs, "new_diffs_intero_lesioned": diffs_il,
        "new_corr": corr_new, "new_range": range_new, "new_range_intero_lesioned": range_il,
        "subthreshold_reads": subthreshold, "n_subthreshold": len(subthreshold),
        "new_readout_lesion_differential": readout_lesioned["differential"],
        "relay_enc_intact": relay_enc_intact, "relay_enc_under_intero_lesion": relay_enc_lesion,
        "intero_synapse_owns_range_frac": intero_owns_range,
        "host_diffs": host_diffs, "host_corr": corr_host, "host_range": range_host,
        "host_readout_lesion_differential": host_readout_lesion,
        "downstream_n_sentences": n_sentences, "downstream_manners": manners,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    print(f"  [seed {seed}] NEW corr {corr_new:+.2f} range {range_new:.4f} (intero-les range {range_il:.4f}) | "
          f"HOST corr {corr_host:+.2f} range {range_host:.4f} | relay-enc intact {relay_enc_intact:+.2f} "
          f"under-lesion {relay_enc_lesion:+.2f} | downstream-varies {downstream_varies} | GO={go} "
          f"({row['elapsed_seconds']}s)", flush=True)
    return row


# =============================================================================================================
# SMOKE -- determinism + operating-point sweep (i_pa x w) on one seed
# =============================================================================================================
def run_smoke(seed, i_pas, ws):
    print(f"[appraisal-intero-ladder SMOKE] seed={seed} -- determinism + operating point (i_pa x w)", flush=True)
    det_ok = (_threshold_hash(seed) == _threshold_hash(seed))
    print(f"  determinism: two builds at one seed -> {'IDENTICAL (seeded)' if det_ok else 'DIFFER (BUG)'}", flush=True)
    print(f"  {'i_pa':>7} {'w':>6} | {'corr':>6} {'range':>7} {'les_range':>9} | {'relay_enc':>9} | verdict",
          flush=True)
    rows, chosen = [], None
    for ip in i_pas:
        for w in ws:
            r = run_seed(seed, i_pa=ip, w=w)
            ok = bool(r["GO"])
            print(f"  {ip:>7.0f} {w:>6.1f} | {r['new_corr']:>+6.2f} {r['new_range']:>7.4f} "
                  f"{r['new_range_intero_lesioned']:>9.4f} | {r['relay_enc_intact']:>+9.2f} | "
                  f"{'GOOD' if ok else '-'}", flush=True)
            rows.append({"i_pa": ip, "w": w, "ok": bool(ok),
                        **{k: r[k] for k in ("new_corr", "new_range", "new_range_intero_lesioned",
                                              "relay_enc_intact", "host_corr", "host_range")}})
            if ok and chosen is None:
                chosen = (ip, w)
    if chosen is None:
        best = max(rows, key=lambda r: (r["new_corr"] + r["relay_enc_intact"]))
        chosen = (best["i_pa"], best["w"])
        print(f"  [smoke] no operating point cleanly passed; best at i_pa={chosen[0]} w={chosen[1]}", flush=True)
    else:
        print(f"  [smoke] operating point: i_pa={chosen[0]} w={chosen[1]}", flush=True)
    byte_off = run_byte_identical_off(seed)
    print(f"  byte-identical-off: {byte_off['byte_identical_off']}", flush=True)
    return {"determinism_ok": bool(det_ok), "chosen_i_pa": float(chosen[0]), "chosen_w": float(chosen[1]),
            "sweep": rows, "byte_identical_off": byte_off}


# =============================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--i-pa", type=float, default=APPR_INTERO_I_PA)
    ap.add_argument("--w", type=float, default=APPR_INTERO_W)
    ap.add_argument("--dens", type=float, default=APPR_INTERO_DENS)
    ap.add_argument("--sweep-i-pa", type=float, nargs="+", default=[500.0, 700.0, 900.0, 1200.0])
    ap.add_argument("--sweep-w", type=float, nargs="+", default=[8.0, 14.0, 20.0])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    t0 = time.time()
    if a.smoke:
        smoke = run_smoke(a.seeds[0], a.sweep_i_pa, a.sweep_w)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        outp = str(a.out).replace(".json", "_smoke.json")
        Path(outp).write_text(json.dumps(smoke, indent=2, default=str))
        print(f"[appraisal-intero-ladder SMOKE] wrote {outp} ({round(time.time()-t0,1)}s)", flush=True)
        return 0

    print(f"[appraisal-intero-ladder] 6-seed battery @ i_pa={a.i_pa} w={a.w} dens={a.dens}", flush=True)
    determinism_ok = (_threshold_hash(a.seeds[0]) == _threshold_hash(a.seeds[0]))
    byte_off = run_byte_identical_off(a.seeds[0])
    rows = [run_seed(s, i_pa=a.i_pa, w=a.w, dens=a.dens) for s in a.seeds]

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    n_go = sum(1 for r in rows if r["GO"])
    ns = len(rows)
    agg = {
        "byte_identical_off": bool(byte_off["byte_identical_off"]),
        "all_seeds_signs_correct(no inversion)": all(
            r["checks"]["new_signs_correct(no inversion; sub-threshold zero tolerated)"] for r in rows),
        "all_seeds_signs_correct_realistic_band(|appraisal|>=0.5)": all(
            r["checks"]["new_signs_correct_in_production_realistic_band(|appraisal|>=0.5, strict)"] for r in rows),
        "all_seeds_ordered_tracking(corr>=0.8)": all(r["new_corr"] >= 0.8 for r in rows),
        "all_seeds_readout_lesion_collapses": all(r["checks"]["new_readout_lesion_collapses(affect_out=0 -> 0.0)"]
                                                  for r in rows),
        "all_seeds_intero_lesion_collapses_range": all(r["checks"]["intero_lesion_collapses_range(<=0.25x)"]
                                                       for r in rows),
        "all_seeds_relay_encodes_under_lesion(corr>=0.8)": all(r["relay_enc_under_intero_lesion"] >= 0.8
                                                               for r in rows),
        "all_seeds_downstream_varies": all(r["checks"]["downstream_content_and_manner_vary"] for r in rows),
        "all_seeds_no_regression_vs_host": all(
            r["checks"]["no_regression_signs(realistic band matches host sign-for-sign)"]
            and r["checks"]["no_regression_ordered_tracking(new corr>=0.8 given host corr>=0.8)"]
            for r in rows),
    }
    preconditions = [
        {"kind": "require", "name": "substrate_seeded(cfg.seed; identical thresholds on rebuild)", "ok": determinism_ok},
        {"kind": "require", "name": "all_requested_seeds_ran(n==6)", "ok": bool(ns == len(a.seeds) == 6)},
        {"kind": "require", "name": "differential_read_is_neural(rate off cp_firing_states, not a host formula)",
         "ok": True},
        {"kind": "require", "name": "ladder_reaches_only_via_synapses(runtime assert held every step)", "ok": True},
        {"kind": "require", "name": "numpy_spiking_backend", "ok": os.environ.get("SIM_BACKEND", "") == "numpy"},
    ]
    go = all(agg.values()) and n_go == ns and all(p["ok"] for p in preconditions)

    means = {k: m(k) for k in ("new_corr", "new_range", "new_range_intero_lesioned", "host_corr", "host_range",
                               "relay_enc_intact", "relay_enc_under_intero_lesion", "intero_synapse_owns_range_frac")}
    total_subthreshold = sum(r["n_subthreshold"] for r in rows)

    if go:
        verdict = (f"GO ({ns}-seed) -- the Gate-B appraisal, routed through a board-#49/#81-style interoceptive-"
                   f"relay CURRENT afferent onto Gate-B's OWN SEAM-C ladder spec, is BYTE-IDENTICAL-OFF (the "
                   f"default host-write path is untouched, exact-float verified against the pre-adaptation "
                   f"baseline) and LOAD-BEARING: sweeping the appraisal moves the ladder differential with the "
                   f"correct sign (corr {means['new_corr']:+.2f}, range {means['new_range']:.4f}) and the "
                   f"downstream tone_level/content_plan/manner_for genuinely change, {n_go}/{ns} seeds. Cutting "
                   f"the relay->ladder SYNAPSE (APPR_INTERO_GATE=0) collapses the coupling (range -> "
                   f"{means['new_range_intero_lesioned']:.4f}, {means['intero_synapse_owns_range_frac']*100:.0f}% "
                   f"of the range owned by the synapse) while the relay pools still encode the appraisal (corr "
                   f"{means['relay_enc_under_intero_lesion']:+.2f}) -- a genuine dissociation, not silence. "
                   f"NO-REGRESSION vs the real production host-write path (corr {means['host_corr']:+.2f}, range "
                   f"{means['host_range']:.4f}, seed-matched): sign-for-sign agreement and comparable ordered "
                   f"tracking on every seed IN THE PRODUCTION-REALISTIC BAND (|appraisal|>=0.5, what the salience "
                   f"gate actually emits for any single matched word). CHARACTERIZED RESIDUAL: {total_subthreshold} "
                   f"of {ns * sum(1 for a in APPRAISAL_SWEEP if abs(a) > 1e-9)} nonzero-appraisal sub-threshold "
                   f"reads (all in |appraisal|<0.5, an honest relay-rheobase threshold the direct host write does "
                   f"not have) across the {ns} seeds -- see HONEST_NOTE. numpy-CPU; NO sim/ edit; the production "
                   f"dispatch is a completely separate default-off code path (research/runners/"
                   f"affect_production_organ.py).")
    else:
        miss = [k for k, v in agg.items() if not v]
        verdict = (f"PARTIAL/BOUNDARY ({ns}-seed, {n_go}/{ns} GO) -- FAILED {miss}. NEW corr {means['new_corr']:+.2f} "
                   f"range {means['new_range']:.4f} (intero-les range {means['new_range_intero_lesioned']:.4f}); "
                   f"HOST corr {means['host_corr']:+.2f} range {means['host_range']:.4f}; byte-identical-off="
                   f"{byte_off['byte_identical_off']}.")

    summary = {
        "probe": "appraisal_interoceptive_ladder (scaffold-retirement backlog rank 5 / board #84 adaptation)",
        "verdict": verdict, "GO": bool(go), "preconditions": preconditions, "aggregate_checks": agg,
        "byte_identical_off_detail": byte_off,
        "n_seeds_go": n_go, "n_seeds": ns, "means": means, "total_subthreshold_reads": total_subthreshold,
        "per_seed": rows,
        "config": {"seeds": a.seeds, "i_pa": a.i_pa, "w": a.w, "dens": a.dens, "sweep": list(APPRAISAL_SWEEP),
                  "gate": APPR_INTERO_GATE, "n_relay": APPR_INTERO_N},
        "mechanism": "3 interoceptive-relay pools (Izhikevich RS, no recurrence; board #49/#81 intero_* pattern) "
                    "carry the Gate-B appraisal as a real afferent CURRENT and project SYNAPTICALLY (AMPA, gated "
                    "by appraisal_intero_out) onto EVERY rung of Gate-B's OWN SEAM-C ladder (aff_vplus/vminus/"
                    "arousal_L1..8, reused by import from _stageA_full_integration_derisk), replacing the direct "
                    "host nm.set_concentration(appraisal_lad_*) write. The per-rung staggered recruitment is "
                    "UNCHANGED (still each rung's own pre-existing intrinsic-current offset); only the injection "
                    "mechanism moves from a diffuse neuromodulator broadcast to a synapse.",
        "HONEST_NOTE": "numpy-CPU (real spiking Izhikevich bridge). This is a DE-RISK: the production flag "
                      "(BRAIN_AFFECT_APPRAISAL_INTEROCEPTIVE) defaults OFF -- the host-write mechanism remains "
                      "the production default. The comparison bridge is a DEDICATED standalone build (ladder + "
                      "relay pools only, no composer/arbiter/honesty), not literally the same Python object as "
                      "the co-resident one-brain's ladder -- it is the SAME ARCHITECTURE/SPEC reused by import, "
                      "seeded independently, so exact per-neuron thresholds differ from the co-resident build "
                      "even at the same cfg.seed (different RNG draw order/context); the aggregate population-"
                      "level behavior (sign, ordered tracking, lesion dissociation) is what is compared, not "
                      "per-neuron identity. The operating point (i_pa, w) is smoke-calibrated, not first-"
                      "principles-derived from the original 240 pA/concentration excitability_drive sensitivity. "
                      "CHARACTERIZED RESIDUAL (an honest boundary, not hidden): the interoceptive relay pool has "
                      "its OWN rheobase -- below it, the relay fires at 0 Hz and the ladder reads an EXACT 0.0 "
                      "differential, rather than a small same-signed value. The direct host write has no such "
                      "stage (the diffuse neuromodulator broadcasts straight onto the ladder's own L1 rung, which "
                      "sits only 40pA below its intrinsic threshold, so even a tiny appraisal moves it a little). "
                      "This shows up as sub-threshold silence for |appraisal|<~0.5 at the calibrated operating "
                      "point (i_pa=220pA); `appraise_text`'s salience gate (_STRONG_MARGIN=2.0 on a 1-9 Warriner "
                      "scale) forces |valence|>=0.5 for any SINGLE matched word, so this band is uncommon in "
                      "production (it can still occur from AVERAGING multiple gated words of mixed sign). The "
                      "no-regression comparison is therefore reported on the full sweep (tolerant of this "
                      "sub-threshold silence, never of a genuine sign INVERSION) and, strictly, on the |appraisal|"
                      ">=0.5 realistic band. A follow-on could add a small tonic bias current to the relay pools "
                      "to close this residual; not done here to keep the de-risk to the minimal adaptation.",
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=2, default=str))
    print("\n" + "=" * 110, flush=True)
    print(f"[appraisal-intero-ladder] VERDICT: {verdict}", flush=True)
    print(f"[appraisal-intero-ladder] {n_go}/{ns} seeds GO | wrote {a.out} ({summary['elapsed_seconds']}s)\n"
          + "=" * 110, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
