"""v5 of the curiosity x metacognition lane: (1) an INSTRUMENT that makes the multiplicative-gain test (G11) defined
on every seed, and (2) the COMPANION PROCESS the v5 substrate measurement located -- on the comparator's read, not
on ASK's operating point.

Pre-registration: `docs/plans/2026-09-24-curiosity-commitment-veto-v5-PREREG.md` (read it first; the gates below are
copied from it and were frozen on DEV seeds 7-12 before any evaluation-seed run of THIS mechanism).

WHY THIS RUNG EXISTS. v4 (`_curiosity_lcne_phasic_gain_derisk.py`) is a 6-seed NO-GO 3/6
(`research/findings/2026-09-24-curiosity-metacog-lcne-phasic-gain-v4-6seed-NOGO-3of6-gain-holds-circuit-residual.md`):
the phasic LC-NE feedback-withdrawal gain held on every seed, seed 42's G11 was UNDEFINED (its reference response
rises across only two points of the coarse drive grid), and seeds 44 (G1) and 100 (G7) failed the same way with the
gain lesioned. The finding's own next-step hypothesis was an ASK OPERATING-POINT problem.

THE MEASUREMENT THAT CHOSE THE MECHANISM (`_curiosity_ask_operating_point_measure.py`, dev seeds 7-12 only; artifacts
`research/findings/raw/_curiosity_ask_op_measure/dev_s*.json`). metacog's margin comparator (`meta_schema`) is two
halves, meta_0 and meta_1. The point-edge sums both onto ASK. Per evidence level, on every dev arm measured:
  * the FAVORED half RISES with evidence (Spearman +0.78..+1.0) and the rival half falls to a floor, so the SUM
    that the edge delivers is U-shaped wherever the favored half's rise outpaces the rival's fall. The failing dev
    arms are exactly the U-shaped ones (seed 8 intact, seed 9 intact, seed 10 class swap);
  * the smaller of the two halves (co-activation, min(meta_0, meta_1)) is monotone decreasing on 11 of the 12 arms
    (Spearman <= -0.94);
  * the ASK OPERATING-POINT route cannot repair a U-shaped input: the best Spearman rho reachable by ANY threshold
    applied to the measured sum is +0.07 (seed 10 swap) and +0.40 (seed 9 intact), and where a threshold does reach
    <= -0.8 it silences 6-8 of the 11 levels;
  * the evaluation seeds' public v4 artifacts show the same shape through lc_add (v4's non-adapting readout of the same
    sum): seed 44 intact 76 -> 111 spikes over evidence 0.7 -> 1.0, seed 100 class swap 58 -> 102.
So the residual is the comparator's summed output, not ASK's operating point. What the real system runs alongside a
choice-channel comparator that this circuit replaced with nothing: an OPPONENT COMMITMENT signal -- neurons that fire
when one channel dominates the other -- and the inhibitory plasticity that holds such a detector at its set-point
when there is no evidence (Vogels et al. 2011 Science 334:1569, inhibitory plasticity balances excitation and
inhibition and sets the neuron's rate to a target). Because each seed's two comparator halves sit at DIFFERENT
no-evidence levels (measured: up to 13.8 vs 8.7 Hz, 19.6 vs 13.3 Hz), a fixed-threshold dominance detector would
fire on some seeds with no evidence at all. The set-point is the companion process: it removes each channel's own
no-evidence asymmetry, per seed, before the evidence sweep.

THE CIRCUIT ADDED IN v5 (every step neurons + synapses; host code drives ONLY metacog's input evidence):
  * `cv_inh{j}` (FS, inhibitory): driven by comparator half meta_j (dense E, W_MI).
  * `cv_veto{k}` (RS, inhibitory output): driven by comparator half meta_k (dense E, W_MV) and inhibited by
    `cv_inh{1-k}` -- the RIVAL half's relay -- through PLASTIC GABA-A synapses (Vogels inhibitory STDP, the engine's
    `enable_inhibitory_stdp` rule; plasticity gate ISTDP_GATE). So veto_k reads "channel k over channel 1-k".
  * `cv_veto{k}` -> `ask` (GABA-A, W_VA, transmission gate VETO_GATE): a committed channel suppresses the ASK pool.
  * CALIBRATION EPOCH (the set-point, once, before any read that is scored): ISTDP_GATE opens; metacog is driven with
    NO evidence differential (its own lesion drive: both assemblies at base) for CAL_READS production reads; the
    inhibitory weights move until each veto half fires at ISTDP_TARGET_HZ; ISTDP_GATE closes and stays closed
    (weights hash-checked frozen to the end of the run). Nothing in the rule sees ASK, the evidence level or the gates.
  Everything v4 built is unchanged (the point-edge, lc_ne's phasic feedback-withdrawal gain onto ASK, the additive
  control). The v4 mechanism is still the thing G3/G11/G12 test.

THE INSTRUMENT CHANGE (G11). v4 sampled the edge-drive sweep at 0.6..1.2 step 0.1; a seed with a steep threshold had
two points on the reference's rising limb and G11 was UNDEFINED. v5 PLACES the grid on each seed's OWN reference
curve, read from the lc-OFF arm only (never the lc-on arm): a placement scan over G11_PLACE_GRID finds the first
drive where the reference ASK >= G11_PLACE_ON_HZ and the drive where it peaks; the scored grid is G11_FINE_N evenly
spaced drives from (onset - G11_FINE_LO_PAD) to (peak + G11_FINE_HI_PAD). The gate logic is v4's, with >=
G11_MIN_DEFINED limb points required on the fine grid. The additive control must still FAIL G11c on that grid.

RESIDUALS (declared): all fixed weights hand-set on dev seeds; the veto's set-point target is a constant (as every
homeostat's is); the calibration epoch is a protocol step (the host drives metacog with no evidence, as production
drives metacog's evidence); the engine's inhibitory rule is a trace-based Vogels rule, not a biophysical model of
GABA-A receptor plasticity; the sAHP relay residuals of v4 stand.

FUNCTIONAL READ-OUTS ONLY -- rates of named spiking populations; no felt state is asserted. Additive research runner:
no `sim/` edit, no production flag, no default flip; nothing in the live chat path imports this file.

Run:
  python -m research.runners._curiosity_commitment_veto_v5_derisk --selftest
  SIM_BACKEND=numpy python -m research.runners._curiosity_commitment_veto_v5_derisk --seeds 7 --dev \\
      --out research/findings/raw/_curiosity_commitment_veto_v5_dev_s7.json
  SIM_BACKEND=numpy python -m research.runners._curiosity_commitment_veto_v5_derisk --seeds 42 \\
      --out research/findings/raw/_curiosity_commitment_veto_v5_s42.json
  python -m research.runners._curiosity_commitment_veto_v5_derisk --combine <six files> \\
      --out research/findings/raw/_curiosity_commitment_veto_v5_6seed_combined.json
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from sim.backend import to_host  # noqa: E402
from sim.regions import BrainRegion  # noqa: E402
from research.runners import _curiosity_lcne_phasic_gain_derisk as V4  # noqa: E402
from research.runners.onebrain_merge_framework import REGISTRY, OrganDescriptor, merge_organs  # noqa: E402
from research.runners.metacog_production_organ import MetacogProductionOrgan, nmda_norm_margin  # noqa: E402
from research.runners._curiosity_metacog_conflict_xedge_derisk import (  # noqa: E402
    build_pool as build_base_pool, METACOG_MARGIN, _metacog_het, _meta_exact, _curiosity_production_threshold,
    perm_null, spearman, level_rho, EVIDENCE_GRID, READ_REPS, STEPS_PER_REP, G7_RHO_MAX,
)

RS, FS = V4.RS, V4.FS

# ── FROZEN operating point (calibrated on DEV seeds 7-12 ONLY; PREREG §2) ─────────────────────────────────────────
CV_N = 20             # veto units per channel (RS, inhibitory output)
CI_N = 20             # rival-relay units per channel (FS, inhibitory)
W_MV = 6.0            # meta_k -> cv_veto{k} (dense E)
W_MI = 6.0            # meta_j -> cv_inh{j} (dense E)
W_IV0 = 4.0           # cv_inh{1-k} -> cv_veto{k} INITIAL weight (GABA-A, plastic under inhibitory STDP)
W_VA = 4.0            # cv_veto{k} -> ask (GABA-A)
ISTDP_TARGET_HZ = 2.0     # the set-point: each veto half's rate with NO evidence differential
ISTDP_ETA = 0.05          # Vogels learning rate
ISTDP_TAU_MS = 20.0       # Vogels trace time constant
ISTDP_W_MAX = 80.0        # inhibitory weight ceiling (conductance magnitude)
CAL_READS = 16            # calibration epoch length, in no-evidence metacog production reads (x READ_REPS x 135 steps)
W_IV0_ALT = 12.0          # the second initial weight of the two-init set-point check (reported, not a gate)

ISTDP_GATE = "cv_istdp"          # plasticity gate of cv_inh -> cv_veto (open only in calibration epochs)
VETO_GATE = "cv_veto_out"        # transmission gate of cv_veto -> ask (the veto lesion switch)
CV_REGIONS = ("cv_veto0", "cv_veto1", "cv_inh0", "cv_inh1")
DEV_SEEDS = frozenset({7, 8, 9, 10, 11, 12})
REQUIRED_SEED_SET = V4.REQUIRED_SEED_SET

# ── pre-registered thresholds (PREREG §3) ────────────────────────────────────────────────────────────────────
SETPOINT_TOL = (0.5, 1.5)        # G13: each veto half's no-evidence rate / ISTDP_TARGET_HZ must lie in this band
G11_PLACE_GRID = tuple(round(0.5 + 0.1 * i, 2) for i in range(12))   # 0.5 .. 1.6: the placement scan (lc-OFF only)
G11_PLACE_ON_HZ = 0.5            # onset: first placement drive where the reference ASK >= this
G11_FINE_N = 11                  # scored grid: this many evenly spaced drives ...
G11_FINE_LO_PAD = 0.1            # ... from onset - this ...
G11_FINE_HI_PAD = 0.05           # ... to the reference peak + this
G11_MIN_DEFINED = 4              # >= this many points on the reference's rising limb (fine grid), else UNDEFINED
G1_RHO_MAX, G1_MIN_RANGE_HZ = V4.G1_RHO_MAX, V4.G1_MIN_RANGE_HZ
G3_GAIN_ATTRIB_MIN = V4.G3_GAIN_ATTRIB_MIN


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  the commitment-veto organ
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _dense(pre, post, w, **kw):
    pre = np.asarray(pre, np.int64)
    post = np.asarray(post, np.int64)
    P = np.repeat(pre, post.size)
    Q = np.tile(post, pre.size)
    d = {"pre_indices": P, "post_indices": Q, "initial_weights": np.full(P.size, float(w), np.float32),
         "plastic": False, "conn_type": "CV_V5", "count": int(P.size)}
    d.update(kw)
    return d


def _cv_spec(seed):
    return ([BrainRegion(name=f"cv_veto{k}", n_neurons=CV_N, exc_fraction=0.0, internal_density=0.0,
                         enable_nmda=False, izh_neuron_type=RS) for k in (0, 1)]
            + [BrainRegion(name=f"cv_inh{k}", n_neurons=CI_N, exc_fraction=0.0, internal_density=0.0,
                           enable_nmda=False, izh_neuron_type=FS) for k in (0, 1)], [], {})


def meta_halves(rm):
    meta = np.asarray(rm.indices("meta_schema"), np.int64)
    h = meta.size // 2
    return {0: meta[:h], 1: meta[h:]}           # the comparator's own class split (_comparator_wiring)


def _cv_rows(rm):
    """(name, pre idx, post idx, weight, extra keys) for every synapse this organ declares."""
    mh = meta_halves(rm)
    ix = {n: np.asarray(rm.indices(n), np.int64) for n in CV_REGIONS + ("ask",)}
    rows = []
    for k in (0, 1):
        rows.append((f"cv_meta{k}_to_veto{k}", mh[k], ix[f"cv_veto{k}"], W_MV, {}))
        rows.append((f"cv_meta{k}_to_inh{k}", mh[k], ix[f"cv_inh{k}"], W_MI, {}))
        rows.append((f"cv_inh{1 - k}_to_veto{k}", ix[f"cv_inh{1 - k}"], ix[f"cv_veto{k}"], W_IV0,
                     {"plastic": True, "plasticity_gate": ISTDP_GATE}))
        rows.append((f"cv_veto{k}_to_ask", ix[f"cv_veto{k}"], ix["ask"], W_VA, {"transmission_gate": VETO_GATE}))
    return rows


def _cv_wiring(bridge, rm):
    return {name: _dense(pre, post, w, **kw) for name, pre, post, w, kw in _cv_rows(rm)}


def _cv_post_inject(bridge):
    bridge.set_plasticity_gate(ISTDP_GATE, 0.0)          # frozen except inside a calibration epoch
    bridge.set_transmission_gate(VETO_GATE, 1.0)


def cv_config():
    return {"enable_inhibitory_stdp": True, "inhibitory_stdp_target_rate_per_step": ISTDP_TARGET_HZ * 1e-3,
            "inhibitory_stdp_eta": ISTDP_ETA, "inhibitory_stdp_tau_ms": ISTDP_TAU_MS,
            "inhibitory_stdp_w_min": 0.0, "inhibitory_stdp_w_max": ISTDP_W_MAX}


def cv_organ():
    return OrganDescriptor(
        key="commitment_veto_organ", regions=CV_REGIONS, spec_fn=_cv_spec, config=cv_config(),
        explicit_wiring_fn=_cv_wiring, post_inject_fn=_cv_post_inject, param_het=True,
        scaffold_residuals=("hand-set fixed weights meta->veto / meta->relay / veto->ask (dev-seed calibration); the "
                            "rival-relay -> veto inhibition is set by inhibitory STDP to a constant target rate in a "
                            "no-evidence calibration epoch, then frozen",))


def build_v5_pool(seed: int):
    """[metacog(het), curiosity, metacog_margin, lcne_gain_organ (v4), commitment_veto_organ] + v4's gated point-edge."""
    pool = merge_organs([_metacog_het(), REGISTRY["curiosity"], METACOG_MARGIN, V4.LCG_ORGAN, cv_organ()],
                        seed=int(seed), wire=True, cross_edges=[V4.XEDGE_GATED])
    pool.ensure_built()
    assert int(pool.bridge.core_config.seed) == int(seed), "cfg.seed must be the substrate seed"
    return pool


MECH_GATES = dict(V4.MECH_GATES, **{VETO_GATE: 1.0})


def _set(b, **gates):
    for k, v in gates.items():
        b.set_transmission_gate(k, float(v))


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  instrument
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
class Recorder(V4.Recorder):
    """v4's Recorder plus the comparator halves and the four v5 populations (v4's `read_level` turns every `idx`
    key into a `<key>_hz` level mean)."""

    def __init__(self, pool):
        super().__init__(pool)
        rm = pool.bridge.region_manager
        mh = meta_halves(rm)
        self.idx.update({"meta_0": mh[0], "meta_1": mh[1]})
        for n in CV_REGIONS:
            self.idx[n] = np.asarray(rm.indices(n), np.int64)
        self.reset()


def _syn_mask(bridge, pre_regions, post_regions):
    rm = bridge.region_manager
    r, c, _ = V4._edge_map(bridge)
    pre = np.concatenate([np.asarray(rm.indices(n), np.int64) for n in pre_regions])
    post = np.concatenate([np.asarray(rm.indices(n), np.int64) for n in post_regions])
    return np.isin(r, pre) & np.isin(c, post)


def _iv_mask(bridge):
    """The plastic rival-relay -> veto synapses (cv_inh* -> cv_veto*)."""
    return _syn_mask(bridge, ("cv_inh0", "cv_inh1"), ("cv_veto0", "cv_veto1"))


def _weights(bridge):
    return np.asarray(to_host(bridge.cp_connections.data), np.float64)


def _sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, np.float64).tobytes()).hexdigest()


def istdp_eligible_pairs(bridge) -> dict:
    """Region pairs of every synapse the engine's inhibitory STDP would update right now (the eligibility the bridge
    applies: inhibitory presynaptic neuron, GABA-A route, plastic, plasticity gain > 0)."""
    cfg = bridge.core_config
    rm = bridge.region_manager
    r, c, _ = V4._edge_map(bridge)
    traits = np.asarray(to_host(bridge.cp_traits))
    inh = np.isin(traits, getattr(cfg, "inhibitory_trait_indices", None) or [cfg.inhibitory_trait_index])
    n = r.size

    def m(name, fill):
        a = getattr(bridge, name, None)
        return np.asarray(to_host(a))[:n] if a is not None else np.full(n, fill)
    elig = inh[r] & m("cp_synapse_plastic_mask", False).astype(bool) & ~m("cp_gabab_synapse_mask", False).astype(bool)
    elig &= m("cp_plasticity_rate_gain", 1.0) > 0.0
    region_of = {}
    for name in rm.region_indices_dict():
        for i in rm.indices(name):
            region_of[int(i)] = name
    out = {}
    for a, b in zip(r[elig], c[elig]):
        k = f"{region_of[int(a)]}->{region_of[int(b)]}"
        out[k] = out.get(k, 0) + 1
    return out


def no_evidence_read(pool, org, rec) -> dict:
    """ONE metacog production read with NO evidence differential (its own lesion drive), recording every population."""
    rec.reset()
    rec.on = True
    nmda_norm_margin(org.bridge, org.xp, org.idx, org.snap, 0.0, lesion=True)
    rec.on = False
    out = {}
    for k, ix in rec.idx.items():
        c = np.asarray(rec.counts[k], np.float64)
        out[f"{k}_hz"] = float(c.sum() / ix.size / (c.size * 1e-3)) if c.size else 0.0
    return out


def calibrate(pool, org, rec, n_reads=CAL_READS) -> dict:
    """The set-point epoch: open ISTDP_GATE, drive metacog with NO evidence differential for n_reads production reads,
    close ISTDP_GATE. Returns the per-read veto-half rates (the convergence trace), the eligibility at the open gate,
    the weight change confined to the plastic rows, and a post-epoch no-evidence read with the gate CLOSED."""
    b = pool.bridge
    w0 = _weights(b)
    iv = _iv_mask(b)
    b.set_plasticity_gate(ISTDP_GATE, 1.0)
    elig = istdp_eligible_pairs(b)
    trace = []
    with pool.sequence_isolation():
        for _ in range(n_reads):
            r = no_evidence_read(pool, org, rec)
            trace.append({"veto0_hz": r["cv_veto0_hz"], "veto1_hz": r["cv_veto1_hz"], "inh0_hz": r["cv_inh0_hz"],
                          "inh1_hz": r["cv_inh1_hz"], "meta_0_hz": r["meta_0_hz"], "meta_1_hz": r["meta_1_hz"]})
    b.set_plasticity_gate(ISTDP_GATE, 0.0)
    w1 = _weights(b)
    with pool.sequence_isolation():
        post = no_evidence_read(pool, org, rec)
    ivw = {}
    for k in (0, 1):
        mk = _syn_mask(b, (f"cv_inh{1 - k}",), (f"cv_veto{k}",))
        ivw[f"to_veto{k}_mean"] = float(w1[mk].mean())
        ivw[f"to_veto{k}_std"] = float(w1[mk].std())
    return {"trace": trace, "eligible_pairs_at_open_gate": elig,
            "weights_changed_outside_plastic_rows": int(np.sum(w0[~iv] != w1[~iv])),
            "plastic_rows_changed": int(np.sum(w0[iv] != w1[iv])), "iv_weights": ivw,
            "iv_weights_sha256": _sha(w1[iv]), "post_epoch_read_gate_closed": post,
            "setpoint_ratio": {k: post[f"cv_veto{k}_hz"] / ISTDP_TARGET_HZ for k in (0, 1)}}


def place_fine_grid(off_place: dict) -> dict:
    """G11's grid, placed on the reference (lc-OFF) curve ONLY. `off_place`: placement drive -> reference ASK Hz."""
    on = [g for g in G11_PLACE_GRID if off_place[g] >= G11_PLACE_ON_HZ]
    if not on:
        return {"grid": None, "undefined": f"reference ASK never reaches {G11_PLACE_ON_HZ} Hz on the placement scan"}
    d_on = on[0]
    after = [g for g in G11_PLACE_GRID if g >= d_on]
    d_pk = max(after, key=lambda g: (off_place[g], -g))            # first maximum
    lo, hi = d_on - G11_FINE_LO_PAD, d_pk + G11_FINE_HI_PAD
    grid = tuple(float(x) for x in np.round(np.linspace(lo, hi, G11_FINE_N), 4))
    return {"grid": grid, "onset": d_on, "peak": d_pk}


def g11_eval(on: dict, off: dict, add: dict, grid, min_defined=G11_MIN_DEFINED) -> dict:
    """v4's G11 logic (V4.g11_eval, same thresholds and quantized trend) on an arbitrary grid with a configurable
    rising-limb minimum. Keys of on/off/add must include 0.0 and every grid value."""
    defined = [g for g in grid if off[g] >= V4.G11_OFF_FLOOR_HZ]
    peak = max(defined, key=lambda g: off[g]) if defined else None
    limb = [g for g in defined if g <= peak] if defined else []
    res = {"grid": list(grid), "defined_points": defined, "n_defined": len(defined), "rising_limb": limb,
           "n_limb": len(limb)}
    if len(limb) < min_defined:
        res.update({"pass": False, "undefined": f"{len(limb)} points on the reference's rising limb "
                                                 f"(need >= {min_defined})"})
        return res
    r_on = [on[g] / off[g] for g in limb]
    r_add = [add[g] / off[g] for g in limb]
    t_on, t_add = V4._trend(limb, r_on), V4._trend(limb, r_add)
    offset = abs(on[0.0] - off[0.0])
    gain_top = on[limb[-1]] / off[limb[-1]]
    add_effect = max(r_add) - 1.0
    instrument_valid = bool(t_add < V4.G11_TREND_MIN and add_effect >= V4.G11_GAIN_MIN)
    parts = {"G11a_no_offset": bool(offset <= V4.G11_OFFSET_MAX_HZ),
             "G11b_gain_at_top_of_rising_limb": bool(gain_top >= 1.0 + V4.G11_GAIN_MIN),
             "G11c_effect_scales_with_drive": bool(t_on >= V4.G11_TREND_MIN)}
    res.update({"ratio_on": r_on, "ratio_add": r_add, "trend_on": t_on, "trend_add": t_add, "offset_hz": offset,
                "gain_at_top_of_rising_limb": gain_top, "additive_effect": add_effect,
                "instrument_valid_additive_control_fails": instrument_valid, "parts": parts,
                "additive_control_parts": {"G11c_effect_scales_with_drive": bool(t_add >= V4.G11_TREND_MIN)},
                "pass": bool(instrument_valid and all(parts.values()))})
    if not instrument_valid:
        res["undefined"] = "the additive control did not fail G11c (or had no effect): instrument cannot discriminate"
    return res


def gate_g13_setpoint(ratios: dict) -> bool:
    """Each veto half's post-calibration no-evidence rate / target lies inside SETPOINT_TOL."""
    return bool(all(SETPOINT_TOL[0] <= float(v) <= SETPOINT_TOL[1] for v in ratios.values()))


def digest(sw) -> str:
    h = hashlib.sha256()
    h.update(V4.digest(sw).encode())
    for l in sw["levels"]:
        h.update(np.asarray([l["cv_veto0_hz"], l["cv_veto1_hz"], l["cv_inh0_hz"], l["cv_inh1_hz"]],
                            np.float64).tobytes())
    return h.hexdigest()


def _vals(sw, key):
    return [l[key] for l in sw["levels"]]


def _rng(sw, key="ask_hz"):
    v = _vals(sw, key)
    return float(max(v) - min(v)), float(max(v))


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  integrity checks
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _byte_off_check(bridge, seed: int) -> dict:
    """The v5 pool's connectivity minus EXACTLY the declared synapses (everything touching v4's and v5's new regions,
    plus the point-edge meta_schema -> ask) equals the conflict_xedge rung's coupled=False pool key for key."""
    rm = bridge.region_manager
    new = np.concatenate([np.asarray(rm.indices(n), np.int64) for n in V4.NEW_REGIONS + CV_REGIONS])
    meta = np.asarray(rm.indices("meta_schema"), np.int64)
    ask = np.asarray(rm.indices("ask"), np.int64)
    r, c, d = V4._edge_map(bridge)
    declared = np.isin(r, new) | np.isin(c, new) | (np.isin(r, meta) & np.isin(c, ask))
    kept = {(int(a), int(b)): float(w) for a, b, w in zip(r[~declared], c[~declared], d[~declared])}
    base = build_base_pool(seed, coupled=False)
    rb, cb, db = V4._edge_map(base.bridge)
    base_map = {(int(a), int(b)): float(w) for a, b, w in zip(rb, cb, db)}
    n_v4 = sum(len(rm.indices(s)) * len(rm.indices(t)) for _, s, t, _, _ in V4._wiring_rows())
    n_v5 = sum(len(pre) * len(post) for _, pre, post, _, _ in _cv_rows(rm))
    n_expected = n_v4 + n_v5 + meta.size * ask.size
    return {"PASS": bool(kept == base_map and int(declared.sum()) == n_expected),
            "base_connectivity_identical": bool(kept == base_map), "n_declared_removed": int(declared.sum()),
            "n_declared_expected": int(n_expected), "n_kept": len(kept), "n_base": len(base_map)}


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  one seed
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _new_session(seed: int):
    pool = build_v5_pool(seed)
    org = MetacogProductionOrgan(seed=seed, shared=pool)
    org.ensure_built()
    rec = Recorder(pool)
    _set(pool.bridge, **MECH_GATES)
    cal = calibrate(pool, org, rec)
    with pool.sequence_isolation():                  # full-sweep prime on the calibrated pool (v3/v4 protocol)
        for ev in EVIDENCE_GRID:
            V4.read_level(pool, org, rec, ev)
    return pool, org, rec, cal


def _combined_intact(pool, org, rec) -> dict:
    V4.prime(pool, org, rec)
    return V4.sweep(pool, org, rec)


def _read_drive(pool, org, rec, gates, drive):
    b = pool.bridge
    _set(b, **MECH_GATES)
    _set(b, **gates)
    _set(b, **{V4.EDGE_GATE: drive})
    V4.prime(pool, org, rec)
    with pool.sequence_isolation():
        lv = V4.read_level(pool, org, rec, V4.EV_G11)
    gm = b._transmission_gate_values
    want = dict(MECH_GATES, **gates, **{V4.EDGE_GATE: drive})
    lv["gates_held"] = bool(all(float(gm[k]) == float(v) for k, v in want.items()))
    return lv


G11_ARMS = (("on", {}), ("off", {V4.LC_GAIN_GATE: 0.0}),
            ("add", {V4.LC_GAIN_GATE: 0.0, V4.ADD_GATE: 1.0}))


def g11_protocol(pool, org, rec) -> dict:
    """Placement scan (lc-OFF only) -> fine grid -> on / off / add at 0.0 + every fine-grid drive."""
    hashes = set()
    place = {}
    for g in G11_PLACE_GRID:
        lv = _read_drive(pool, org, rec, dict(G11_ARMS)["off"], g)
        place[g] = lv
        hashes.add(lv["lc_raster_sha256"])
    placed = place_fine_grid({g: v["ask_hz"] for g, v in place.items()})
    reads = {name: {} for name, _ in G11_ARMS}
    if placed["grid"] is not None:
        for name, gates in G11_ARMS:
            for g in (0.0,) + placed["grid"]:
                lv = _read_drive(pool, org, rec, gates, g)
                reads[name][g] = lv
                hashes.add(lv["lc_raster_sha256"])
        res = g11_eval({g: v["ask_hz"] for g, v in reads["on"].items()},
                       {g: v["ask_hz"] for g, v in reads["off"].items()},
                       {g: v["ask_hz"] for g, v in reads["add"].items()}, placed["grid"])
    else:
        res = {"pass": False, "undefined": placed["undefined"], "grid": None}
    lc_fixed = len(hashes) == 1
    res["lc_raster_identical_across_all_sweep_reads"] = bool(lc_fixed)
    if not lc_fixed:
        res["pass"] = False
        res["undefined"] = "lc_ne raster differed across the drive sweep: the modulator was not held fixed"
    res["placement"] = {"onset": placed.get("onset"), "peak": placed.get("peak"),
                        "reference_ask_hz": {str(g): v["ask_hz"] for g, v in place.items()},
                        "n_limb_on_placement_grid": _coarse_limb(place)}
    _set(pool.bridge, **MECH_GATES)
    return {"g11": res, "reads": reads, "place": place}


def _coarse_limb(place) -> int:
    """How many placement-grid points v4's coarse rule would have had on the rising limb (reported)."""
    off = {g: v["ask_hz"] for g, v in place.items()}
    defined = [g for g in G11_PLACE_GRID if off[g] >= V4.G11_OFF_FLOOR_HZ]
    if not defined:
        return 0
    peak = max(defined, key=lambda g: off[g])
    return len([g for g in defined if g <= peak])


def confident_half_mean(sw) -> float:
    """Mean ASK Hz over the five most-confident levels (evidence 0.6 .. 1.0): where the comparator's U-shaped sum
    produced the v4 tails."""
    return float(np.mean([l["ask_hz"] for l in sw["levels"] if l["evidence"] >= 0.6 - 1e-9]))


def run_seed(seed: int, determinism: bool = True, verbose: bool = True, quick: bool = False) -> dict:
    t0 = time.time()
    pool, org, rec, cal = _new_session(seed)
    b, xp = pool.bridge, pool.xp
    iv_sha_after_cal = cal["iv_weights_sha256"]
    combined = _combined_intact(pool, org, rec)

    def arm(gates=None, swap=False):
        _set(b, **MECH_GATES)
        _set(b, **(gates or {}))
        V4.prime(pool, org, rec)
        sw = V4.sweep(pool, org, rec, swap=swap)
        sw["gates_at_measurement"] = {k: float(v) for k, v in b._transmission_gate_values.items()}
        _set(b, **MECH_GATES)
        return sw

    swap = arm(swap=True)                                                   # G7
    veto_lesion = arm({VETO_GATE: 0.0})                                     # reported: the v4 circuit, same pool
    swap_veto_lesion = arm({VETO_GATE: 0.0}, swap=True)                     # reported
    gain_lesion = arm({V4.LC_GAIN_GATE: 0.0})                               # G3
    swap_gain_lesion = arm({V4.LC_GAIN_GATE: 0.0}, swap=True)               # reported (v4 v1.1 arm)
    if quick:
        rows = {"combined_intact": combined, "class_swap": swap, "veto_lesion": veto_lesion,
                "class_swap_veto_lesion": swap_veto_lesion, "gain_lesion": gain_lesion,
                "class_swap_gain_lesion": swap_gain_lesion}
        res = {"seed": seed, "quick": True, "calibration": cal,
               "rho": level_rho(combined), "rho_swap": level_rho(swap), "rho_veto_lesion": level_rho(veto_lesion),
               "rho_swap_veto_lesion": level_rho(swap_veto_lesion), "rho_gain_lesion": level_rho(gain_lesion),
               "ask_range_hz": {k: _rng(v)[0] for k, v in rows.items()},
               "gain_share": _attrib(seed, combined, gain_lesion), "arms": rows,
               "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
               "elapsed_s": round(time.time() - t0, 1)}
        _print_quick(res)
        return res
    edge_lesion = arm({V4.EDGE_GATE: 0.0})                                  # reported
    both_lesion = arm({V4.EDGE_GATE: 0.0, V4.LC_GAIN_GATE: 0.0})            # G4 (integrity)
    loop_lc_on = arm({V4.FB_LOOP_GATE: 0.0})                                # integrity: lc acts ONLY through the loop
    loop_lc_off = arm({V4.FB_LOOP_GATE: 0.0, V4.LC_GAIN_GATE: 0.0})
    autoinh_lesion = arm({V4.LC_AUTO_GATE: 0.0})                            # G12b

    # G8: lesion metacog's comparator relay (meta_margin_fs -> meta_schema), everything else intact.
    rm = b.region_manager
    r, c, _ = V4._edge_map(b)
    relay_mask = np.isin(r, np.asarray(rm.indices("meta_margin_fs"))) & np.isin(c, np.asarray(rm.indices("meta_schema")))
    data = np.asarray(to_host(b.cp_connections.data)).copy()
    data2 = data.copy()
    data2[relay_mask] = 0.0
    b.cp_connections.data = xp.asarray(data2, dtype=b.cp_connections.data.dtype)
    with pool.sequence_isolation():
        for ev in EVIDENCE_GRID:
            V4.read_level(pool, org, rec, ev)
    relay_lesion = V4.sweep(pool, org, rec)
    relay_lesion["relay_weight_sum_at_measurement"] = float(np.asarray(to_host(b.cp_connections.data))[relay_mask].sum())
    b.cp_connections.data = xp.asarray(data, dtype=b.cp_connections.data.dtype)
    with pool.sequence_isolation():
        for ev in EVIDENCE_GRID:
            V4.read_level(pool, org, rec, ev)

    g11p = g11_protocol(pool, org, rec)
    g11 = g11p["g11"]
    restored = arm()
    iv_sha_end = _sha(_weights(b)[_iv_mask(b)])

    # ── statistics ──
    rng_c, peak_c = _rng(combined)
    rng_g, _ = _rng(gain_lesion)
    rng_e, _ = _rng(edge_lesion)
    rho_raw = level_rho(combined)
    rho_swap = level_rho(swap)
    rho_relay = level_rho(relay_lesion)
    rho_both = level_rho(both_lesion)
    rho_lc = spearman(list(EVIDENCE_GRID), _vals(combined, "lc_ne_hz"))
    rng_lc, peak_lc = _rng(combined, "lc_ne_hz")
    conc = V4.lc_burst_concentration(combined)
    conc_autoinh = V4.lc_burst_concentration(autoinh_lesion)
    from tools.lab import attributable_to
    attrib_gain = attributable_to(f"seed{seed} ASK range = the lc_ne phasic gain pathway", rng_c, rng_g)
    attrib_edge = attributable_to(f"seed{seed} ASK range = the point edge", rng_c, rng_e)
    # whose confident-end suppression is it? (reported; the veto is the v5 companion process)
    veto_tail = {
        "intact": attributable_to(f"seed{seed} confident-half ASK removed by the veto (intact)",
                                  confident_half_mean(veto_lesion), confident_half_mean(combined)),
        "class_swap": attributable_to(f"seed{seed} confident-half ASK removed by the veto (swap)",
                                      confident_half_mean(swap_veto_lesion), confident_half_mean(swap))}
    g9 = perm_null(combined, seed)

    lesion_arms = (("veto_lesion", veto_lesion), ("gain_lesion", gain_lesion), ("edge_lesion", edge_lesion),
                   ("both_lesion", both_lesion), ("loop_lc_on", loop_lc_on), ("loop_lc_off", loop_lc_off),
                   ("autoinh_lesion", autoinh_lesion))
    m_arms = {name: _meta_exact(combined, sw) for name, sw in lesion_arms}
    cmp_equal = all(_vals(combined, "cmp_raster_sha256") == _vals(sw, "cmp_raster_sha256") for _, sw in lesion_arms)
    g5 = bool(cmp_equal and all(m["balance_equal"] and m["metacog_raster_equal"] and m["confident_equal"]
                                for m in m_arms.values()))
    loop_only = bool([l["ask_hz_per_rep"] for l in loop_lc_on["levels"]]
                     == [l["ask_hz_per_rep"] for l in loop_lc_off["levels"]])
    all_sweeps = (combined, swap, veto_lesion, swap_veto_lesion, gain_lesion, swap_gain_lesion, edge_lesion,
                  both_lesion, loop_lc_on, loop_lc_off, autoinh_lesion, relay_lesion, restored)
    bystander_spikes = int(sum(l["bystander_spikes"] for sw in all_sweeps for l in sw["levels"])
                           + sum(v["bystander_spikes"] for rd in g11p["reads"].values() for v in rd.values())
                           + sum(v["bystander_spikes"] for v in g11p["place"].values()))
    dig = digest(combined)
    restore_ok = digest(restored) == dig
    held = {"veto_lesion": veto_lesion["gates_at_measurement"][VETO_GATE] == 0.0,
            "class_swap_veto_lesion": swap_veto_lesion["gates_at_measurement"][VETO_GATE] == 0.0,
            "gain_lesion": gain_lesion["gates_at_measurement"][V4.LC_GAIN_GATE] == 0.0,
            "class_swap_gain_lesion": swap_gain_lesion["gates_at_measurement"][V4.LC_GAIN_GATE] == 0.0,
            "edge_lesion": edge_lesion["gates_at_measurement"][V4.EDGE_GATE] == 0.0,
            "both_lesion": (both_lesion["gates_at_measurement"][V4.EDGE_GATE] == 0.0
                            and both_lesion["gates_at_measurement"][V4.LC_GAIN_GATE] == 0.0),
            "loop_lesion_lc_on": (loop_lc_on["gates_at_measurement"][V4.FB_LOOP_GATE] == 0.0
                                  and loop_lc_on["gates_at_measurement"][V4.LC_GAIN_GATE] == 1.0),
            "loop_lesion_lc_off": (loop_lc_off["gates_at_measurement"][V4.FB_LOOP_GATE] == 0.0
                                   and loop_lc_off["gates_at_measurement"][V4.LC_GAIN_GATE] == 0.0),
            "autoinhibition_lesion": autoinh_lesion["gates_at_measurement"][V4.LC_AUTO_GATE] == 0.0,
            "relay_lesion": relay_lesion["relay_weight_sum_at_measurement"] == 0.0,
            "g11_drive_sweep_gates": all(v["gates_held"] for rd in g11p["reads"].values() for v in rd.values())
            and all(v["gates_held"] for v in g11p["place"].values()),
            "additive_control_closed_in_mechanism_arms": all(
                sw["gates_at_measurement"][V4.ADD_GATE] == 0.0 for sw in all_sweeps if "gates_at_measurement" in sw),
            "veto_open_in_mechanism_arms": all(
                sw["gates_at_measurement"][VETO_GATE] == 1.0 for sw in (combined, swap, gain_lesion, restored)
                if "gates_at_measurement" in sw)}
    held["veto_open_in_mechanism_arms"] = bool(held["veto_open_in_mechanism_arms"]
                                               and b._transmission_gate_values[VETO_GATE] == 1.0)

    det = {"checked": False}
    if determinism:
        cmd = [sys.executable, "-m", "research.runners._curiosity_commitment_veto_v5_derisk", "--digest-only",
               "--seeds", str(seed)] + (["--dev"] if seed in DEV_SEEDS else []) + (["--set"] + _OVERRIDES
                                                                                  if _OVERRIDES else [])
        out = subprocess.run(cmd, cwd=str(_REPO), capture_output=True, text=True, env=dict(os.environ))
        line = [x for x in out.stdout.splitlines() if x.startswith("DIGEST ")]
        other = line[-1].split()[-1] if line else None
        det = {"checked": True, "digest_main": dig, "digest_fresh_process": other, "equal": bool(other == dig),
               "rc": out.returncode}

    prod = _curiosity_production_threshold(seed)
    thr = prod.get("threshold_hz")
    unc = [l for l in combined["levels"] if not l["confident"]]
    con = [l for l in combined["levels"] if l["confident"]]
    s1_undefined = thr is None or not unc or not con
    s1 = (not s1_undefined) and max(l["ask_hz"] for l in unc) >= thr and max(l["ask_hz"] for l in con) < thr

    checks_required = {
        "G1_monotone_rho<=-0.8": V4.gate_g1(rho_raw, rng_c),
        "G3_gain_pathway_load_bearing": bool(attrib_gain is not None and attrib_gain >= G3_GAIN_ATTRIB_MIN),
        "G6_determinism_fresh_process_hash": bool(det.get("equal")) if determinism else None,
        "G7_class_swap_monotone": bool(rho_swap is not None and rho_swap <= G7_RHO_MAX),
        "G8_relay_lesion_abolishes_coupling": V4.gate_g8_relay_lesion(rho_relay),
        "G10_lc_ne_evidence_graded": V4.gate_g10_lc_graded(rho_lc, rng_lc),
        "G11_multiplicative_not_additive": bool(g11["pass"]),
        "G12_lc_ne_phasic": V4.gate_g12_phasic(conc, conc_autoinh),
        "G13_veto_setpoint_reached": gate_g13_setpoint(cal["setpoint_ratio"]),
    }
    checks_integrity = {
        "G4_joint_lesion_breaks_coupling": V4.gate_g4_joint_lesion(rho_both),
        "G5_metacog_unchanged_EXACT_across_arms": g5,
    }
    required = [k for k, v in checks_required.items() if v is not None]
    go = all(checks_required[k] for k in required)

    integrity = {
        "byte_off": _byte_off_check(b, seed),
        "gabab_routing": V4._gabab_routing_check(b),
        "metacog_vs_base_pool": V4._metacog_vs_base(combined, V4._base_pool_metacog(seed)),
        "restore_exact": bool(restore_ok),
        "lesions_held_at_measurement": held,
        "lc_acts_only_through_feedback_loop": loop_only,
        "bystander_spikes_all_arms": bystander_spikes,
        "no_host_novelty_signal": float(getattr(b.core_config, "current_novelty_signal", 0.0) or 0.0) == 0.0,
        "neuromodulator_subsystem_enabled": bool(getattr(b.core_config, "enable_neuromodulator_subsystem", False)),
        "istdp_eligible_only_declared_rows": set(cal["eligible_pairs_at_open_gate"]) == {
            "cv_inh1->cv_veto0", "cv_inh0->cv_veto1"},
        "calibration_changed_only_plastic_rows": bool(cal["weights_changed_outside_plastic_rows"] == 0
                                                      and cal["plastic_rows_changed"] > 0),
        "veto_weights_frozen_after_calibration": bool(iv_sha_end == iv_sha_after_cal),
        "gates_after_run": dict(b._transmission_gate_values),
    }
    # the two-init set-point check (REPORTED): same pool, the plastic rows reset to W_IV0_ALT, re-calibrated.
    two_init = _two_init_check(pool, org, rec)
    res = {
        "seed": seed, "go": bool(go), "dev_seed": seed in DEV_SEEDS, "checks": checks_required,
        "checks_integrity": checks_integrity, "calibration": cal, "two_init_setpoint": two_init,
        "rho": rho_raw, "rho_swap": rho_swap, "rho_veto_lesion_arm": level_rho(veto_lesion),
        "rho_swap_veto_lesion_arm": level_rho(swap_veto_lesion), "rho_gain_lesion_arm": level_rho(gain_lesion),
        "rho_swap_gain_lesion_arm": level_rho(swap_gain_lesion), "rho_relay_lesion": rho_relay,
        "rho_both_lesion": rho_both, "rho_lc_ne": rho_lc,
        "ask_range_hz": {"combined": rng_c, "gain_lesion": rng_g, "edge_lesion": rng_e,
                         "veto_lesion": _rng(veto_lesion)[0]},
        "ask_peak_hz": {"combined": peak_c}, "lc_ne_range_hz": rng_lc, "lc_ne_peak_hz": peak_lc,
        "attributable_frac": {"gain": attrib_gain, "edge": attrib_edge, "veto_confident_half": veto_tail},
        "g11": g11,
        "g11_ask_hz": {name: {str(g): v["ask_hz"] for g, v in rd.items()} for name, rd in g11p["reads"].items()},
        "phasic": {"lc_ne": conc, "lc_ne_autoinhibition_lesioned": conc_autoinh},
        "s1_reaches_production_threshold": bool(s1), "s1_undefined": bool(s1_undefined), "s1_threshold_hz": thr,
        "g9_perm_null": g9, "metacog_exact": m_arms, "comparator_raster_equal_across_arms": bool(cmp_equal),
        "determinism": det, "integrity": integrity, "production_threshold_calibration": prod,
        "arms": {"combined_intact": combined, "class_swap": swap, "veto_lesion": veto_lesion,
                 "class_swap_veto_lesion": swap_veto_lesion, "gain_lesion": gain_lesion,
                 "class_swap_gain_lesion": swap_gain_lesion, "edge_lesion": edge_lesion, "both_lesion": both_lesion,
                 "loop_lesion_lc_on": loop_lc_on, "loop_lesion_lc_off": loop_lc_off,
                 "autoinhibition_lesion": autoinh_lesion, "relay_lesion": relay_lesion},
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if verbose:
        print(f"[seed {seed}] rho={rho_raw} swap={rho_swap} (veto-lesion {res['rho_veto_lesion_arm']} / "
              f"{res['rho_swap_veto_lesion_arm']}) relay={rho_relay} rho_lc={rho_lc} attrib_gain={attrib_gain} "
              f"G11={g11.get('pass')} limb={g11.get('rising_limb')} trend_on={g11.get('trend_on')} "
              f"trend_add={g11.get('trend_add')} setpoint={cal['setpoint_ratio']} det={det.get('equal')} GO={go} "
              f"({res['elapsed_s']}s, {res['peak_rss_mb']:.0f} MB)", flush=True)
        print(f"[seed {seed}] ask combined={[round(x, 2) for x in _vals(combined, 'ask_hz')]} "
              f"swap={[round(x, 2) for x in _vals(swap, 'ask_hz')]}", flush=True)
        print(f"[seed {seed}] checks={checks_required} integrity={checks_integrity} "
              f"{ {k: (v['PASS'] if isinstance(v, dict) and 'PASS' in v else v) for k, v in integrity.items() if k != 'gates_after_run'} }",
              flush=True)
    return res


def _attrib(seed, combined, gain_lesion):
    from tools.lab import attributable_to
    return attributable_to(f"seed{seed} ASK range = lc gain", _rng(combined)[0], _rng(gain_lesion)[0])


def _two_init_check(pool, org, rec) -> dict:
    """REPORTED set-point anti-cheat: reset the plastic rows to W_IV0_ALT and re-run the calibration epoch; a
    set-point process lands both inits in the same rate band. Runs LAST (nothing is scored after it)."""
    b = pool.bridge
    iv = _iv_mask(b)
    data = np.asarray(to_host(b.cp_connections.data)).copy()
    data[iv] = W_IV0_ALT
    b.cp_connections.data = pool.xp.asarray(data, dtype=b.cp_connections.data.dtype)
    cal2 = calibrate(pool, org, rec)
    return {"w_init": W_IV0_ALT, "setpoint_ratio": cal2["setpoint_ratio"], "iv_weights": cal2["iv_weights"],
            "in_band": gate_g13_setpoint(cal2["setpoint_ratio"]),
            "trace_veto_hz": [(t["veto0_hz"], t["veto1_hz"]) for t in cal2["trace"]]}


def _print_quick(res):
    s = res["seed"]
    cal = res["calibration"]
    arms = res["arms"]
    print(f"[seed {s} quick] setpoint={ {k: round(v, 2) for k, v in cal['setpoint_ratio'].items()} } "
          f"iv={ {k: round(v, 2) for k, v in cal['iv_weights'].items()} } elig={cal['eligible_pairs_at_open_gate']} "
          f"outside_changed={cal['weights_changed_outside_plastic_rows']}", flush=True)
    print(f"[seed {s} quick] cal trace veto0={[round(t['veto0_hz'], 2) for t in cal['trace']]}", flush=True)
    print(f"[seed {s} quick] cal trace veto1={[round(t['veto1_hz'], 2) for t in cal['trace']]}", flush=True)
    for k in ("combined_intact", "class_swap", "veto_lesion", "class_swap_veto_lesion"):
        sw = arms[k]
        print(f"[seed {s} quick] {k:24s} ask={[round(x, 2) for x in _vals(sw, 'ask_hz')]}", flush=True)
        print(f"[seed {s} quick] {'':24s} v0={[round(x, 1) for x in _vals(sw, 'cv_veto0_hz')]} "
              f"v1={[round(x, 1) for x in _vals(sw, 'cv_veto1_hz')]}", flush=True)
    print(f"[seed {s} quick] rho={res['rho']} swap={res['rho_swap']} veto_lesion={res['rho_veto_lesion']} "
          f"swap_veto_lesion={res['rho_swap_veto_lesion']} ranges={ {k: round(v, 2) for k, v in res['ask_range_hz'].items()} } "
          f"gain_share={res['gain_share']} ({res['elapsed_s']}s, {res['peak_rss_mb']:.0f} MB)", flush=True)


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  selftest (no simulation)
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _selftest():
    # (1) placement reads the reference only and brackets a steep threshold: seed-42-like (coarse limb = 2 points)
    steep = {g: (0.0 if g < 0.9 else (0.6 if g < 1.0 else (1.74 if g < 1.1 else 1.3))) for g in G11_PLACE_GRID}
    p = place_fine_grid(steep)
    assert p["onset"] == 0.9 and p["peak"] == 1.0 and len(p["grid"]) == G11_FINE_N, p
    assert abs(p["grid"][0] - 0.8) < 1e-9 and abs(p["grid"][-1] - 1.05) < 1e-9, p
    assert place_fine_grid({g: 0.1 for g in G11_PLACE_GRID})["grid"] is None
    # (2) on the placed fine grid, a smooth steep reference (sigmoid) gives >= G11_MIN_DEFINED limb points ...
    grid = p["grid"]

    def curve(f):
        d = {0.0: 0.0}
        d.update({g: float(f(g)) for g in grid})
        return d
    off = curve(lambda g: 2.0 / (1.0 + np.exp(-(g - 0.93) / 0.025)))
    mult = curve(lambda g: 1.4 * off[g])
    grow = curve(lambda g: off[g] * (1.0 + 2.0 * (g - 0.8)))
    add = curve(lambda g: 2.0 / (1.0 + np.exp(-(g + 0.03 - 0.93) / 0.025)))    # additive: the threshold shifts
    r = g11_eval(mult, off, add, grid)
    assert r["n_limb"] >= G11_MIN_DEFINED and r["pass"], ("fine-grid pure gain must pass", r)
    r = g11_eval(grow, off, add, grid)
    assert r["pass"] and r["trend_on"] > 0.0, ("fine-grid growing gain must pass", r)
    # (3) ... and an ADDITIVE shift FAILS G11c on it (the failing direction), and cannot validate the instrument
    r = g11_eval(add, off, add, grid)
    assert not r["pass"] and not r["parts"]["G11c_effect_scales_with_drive"], ("fine-grid additive must fail", r)
    r = g11_eval(mult, off, mult, grid)
    assert not r["pass"] and not r["instrument_valid_additive_control_fails"], ("invalid instrument", r)
    # (4) the v4 coarse points of the same steep reference have < 3 limb points (the UNDEFINED it fixes)
    assert _coarse_limb({g: {"ask_hz": steep[g]} for g in G11_PLACE_GRID}) == 2
    # (5) G13 set-point band, both directions
    assert gate_g13_setpoint({0: 1.0, 1: 0.6}) and not gate_g13_setpoint({0: 1.0, 1: 0.4})
    assert not gate_g13_setpoint({0: 1.6, 1: 1.0})
    # (6) v4's own gate selftest still passes (shared predicates)
    V4._selftest_gate_logic()
    print("[v5 selftest] fine-grid placement + G11 both directions + G13 band OK", flush=True)
    return 0


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  decision + combiner
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
MECHANISM = ("v4 (metacog comparator -> frozen point-edge onto curiosity's ASK + phasic lc_ne feedback-withdrawal "
             "gain) + a per-channel COMMITMENT VETO onto ASK whose rival-relay inhibition is set by inhibitory STDP "
             "to a constant no-evidence rate in a calibration epoch, then frozen; G11 on a reference-placed fine grid")


def operating_point() -> dict:
    op = {k: globals()[k] for k in CIRCUIT_CONSTANTS}
    op.update({"v4": V4.operating_point(), "SETPOINT_TOL": list(SETPOINT_TOL), "G11_PLACE_GRID": list(G11_PLACE_GRID),
               "G11_PLACE_ON_HZ": G11_PLACE_ON_HZ, "G11_FINE_N": G11_FINE_N, "G11_FINE_LO_PAD": G11_FINE_LO_PAD,
               "G11_FINE_HI_PAD": G11_FINE_HI_PAD, "G11_MIN_DEFINED": G11_MIN_DEFINED,
               "homeostasis_target_rate_hz": ISTDP_TARGET_HZ})
    return op


def _decide(rows) -> dict:
    from tools.verdict import Verdict
    seeds = {r["seed"] for r in rows}
    evaluation = seeds == REQUIRED_SEED_SET
    v = Verdict("v4 circuit + no-evidence-calibrated commitment veto -> curiosity ASK (per-seed pre-registered gates)")
    for r in rows:
        s, integ = r["seed"], r["integrity"]
        bal = [l["balance"] for l in r["arms"]["combined_intact"]["levels"]]
        v.require(f"seed{s} metacog balance varies across the evidence grid", (max(bal) - min(bal)) > 0.0)
        v.require(f"seed{s} byte-off: base connectivity identical minus exactly the declared synapses",
                  bool(integ["byte_off"]["PASS"]))
        v.require(f"seed{s} GIRK routing is exactly the declared set", bool(integ["gabab_routing"]["PASS"]))
        v.require(f"seed{s} metacog read EXACTLY equals the conflict_xedge base pool's",
                  bool(integ["metacog_vs_base_pool"]["PASS"]))
        v.require(f"seed{s} metacog unchanged EXACT across every lesion arm (G5)",
                  bool(r["checks_integrity"]["G5_metacog_unchanged_EXACT_across_arms"]))
        v.require(f"seed{s} lesion restore exact", bool(integ["restore_exact"]))
        v.require(f"seed{s} every lesion verified to hold at measurement",
                  all(integ["lesions_held_at_measurement"].values()))
        v.require(f"seed{s} lc_ne acts on ASK only through the feedback loop",
                  bool(integ["lc_acts_only_through_feedback_loop"]))
        v.require(f"seed{s} curiosity's non-ASK regions silent in every read", integ["bystander_spikes_all_arms"] == 0)
        v.require(f"seed{s} no host novelty scalar / neuromodulator subsystem",
                  bool(integ["no_host_novelty_signal"] and not integ["neuromodulator_subsystem_enabled"]))
        v.require(f"seed{s} inhibitory STDP eligible ONLY on the declared rival-relay -> veto rows",
                  bool(integ["istdp_eligible_only_declared_rows"]))
        v.require(f"seed{s} calibration changed only the plastic rows", bool(integ["calibration_changed_only_plastic_rows"]))
        v.require(f"seed{s} veto weights frozen from calibration to the end of the run",
                  bool(integ["veto_weights_frozen_after_calibration"]))
    v.disabled("STDP / Hebbian / homeostatic thresholds / OU / conductance noise",
               "the production metacog pool config; inhibitory STDP runs only in the declared calibration epoch")
    n_go = sum(1 for r in rows if r["go"])
    decided = v.decide(bool(n_go == len(rows)))
    return {
        "mechanism": MECHANISM,
        "prereg": "docs/plans/2026-09-24-curiosity-commitment-veto-v5-PREREG.md",
        "builds_on": "docs/plans/2026-09-24-curiosity-lcne-phasic-gain-PREREG.md",
        "evaluation_set": bool(evaluation),
        "verdict": decided["status"] if evaluation else f"DEV-SMOKE ({decided['status']} on dev seeds; not the "
                                                        f"pre-registered verdict)",
        "preconditions": decided["preconditions"], "undefined_reasons": decided["undefined_reasons"],
        "disabled_processes": decided["disabled_processes"],
        "GO": bool(decided["go"]) if evaluation else None, "n_go": n_go, "n_seeds": len(rows),
        "operating_point": operating_point(), "per_seed": rows,
    }


RUNNER_REL = "research/runners/_curiosity_commitment_veto_v5_derisk.py"
CIRCUIT_CONSTANTS = ("CV_N", "CI_N", "W_MV", "W_MI", "W_IV0", "W_VA", "ISTDP_TARGET_HZ", "ISTDP_ETA",
                     "ISTDP_TAU_MS", "ISTDP_W_MAX", "CAL_READS", "W_IV0_ALT")   # the only names --set may override
_OVERRIDES: list = []


def _combine(paths, out) -> int:
    rows, seen, mech, ops, shas = [], {}, {}, {}, {}
    for fp in paths:
        data = json.loads(Path(fp).read_text())
        mech[fp] = data.get("mechanism")
        ops[fp] = json.dumps(data.get("operating_point"), sort_keys=True)
        prov = Path(str(fp) + ".prov.json")
        shas[fp] = json.loads(prov.read_text()).get("git_sha") if prov.exists() else None
        for r in data["per_seed"]:
            seen.setdefault(r["seed"], []).append(fp)
            rows.append(r)
    dup = {s: f for s, f in seen.items() if len(f) > 1}
    missing = REQUIRED_SEED_SET - set(seen)
    extra = set(seen) - REQUIRED_SEED_SET
    if dup or missing or extra:
        print(f"[combine] REFUSED: dup={dup} missing={missing} extra={extra}", flush=True)
        return 2
    if len(set(mech.values())) > 1 or None in mech.values():
        print(f"[combine] REFUSED: mechanisms differ or missing: {mech}", flush=True)
        return 2
    if len(set(ops.values())) > 1:
        print(f"[combine] REFUSED: operating points differ: {ops}", flush=True)
        return 2
    blobs = {}
    for fp, sha in shas.items():
        if not sha or sha == "unknown":
            print(f"[combine] REFUSED: {fp} has no provenance git_sha", flush=True)
            return 2
        rr = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{sha}:{RUNNER_REL}"], capture_output=True,
                            text=True, cwd=str(_REPO))
        if rr.returncode != 0:
            print(f"[combine] REFUSED: cannot resolve {RUNNER_REL} at {sha}", flush=True)
            return 2
        blobs[fp] = rr.stdout.strip()
    if len(set(blobs.values())) > 1:
        print(f"[combine] REFUSED: inputs ran DIFFERENT runner code: {blobs}", flush=True)
        return 2
    rows.sort(key=lambda r: r["seed"])
    summary = _decide(rows)
    summary["config"] = {"combined_from": list(paths)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=1, default=str))
    print(f"[commitment veto v5] COMBINED VERDICT: {summary['verdict']} ({summary['n_go']}/{summary['n_seeds']}) "
          f"<- {list(paths)} -> {out}", flush=True)
    return 0 if summary["GO"] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[7])
    ap.add_argument("--dev", action="store_true", help="allow dev seeds (7-12); the output is labeled DEV-SMOKE")
    ap.add_argument("--quick", action="store_true", help="DEV ONLY: calibration + intact/swap/veto/gain arms only")
    ap.add_argument("--no-determinism", action="store_true")
    ap.add_argument("--digest-only", action="store_true", help="internal: print the combined-intact digest and exit")
    ap.add_argument("--selftest", action="store_true", help="gate-logic selftest, no simulation")
    ap.add_argument("--combine", nargs="+", default=None, help="the ONLY authoritative 6-seed verdict")
    ap.add_argument("--set", nargs="+", default=[], metavar="NAME=VALUE",
                    help="DEV ONLY: override a circuit constant (never a gate threshold); --combine refuses mixes")
    ap.add_argument("--out", default=str(_REPO / "research" / "findings" / "raw" / "_curiosity_commitment_veto_v5.json"))
    a = ap.parse_args()
    if a.set or a.quick:
        if not a.dev or any(s not in DEV_SEEDS for s in a.seeds):
            print("[commitment veto v5] REFUSED: --set / --quick are for dev-seed calibration only", flush=True)
            return 2
        for kv in a.set:
            k, v = kv.split("=", 1)
            if k not in CIRCUIT_CONSTANTS:
                print(f"[commitment veto v5] REFUSED: {k} is not an overridable circuit constant", flush=True)
                return 2
            globals()[k] = int(float(v)) if isinstance(globals()[k], int) else float(v)
            _OVERRIDES.append(kv)
    if a.selftest:
        return _selftest()
    if a.digest_only:
        pool, org, rec, _cal = _new_session(a.seeds[0])
        print("DIGEST", digest(_combined_intact(pool, org, rec)), flush=True)
        return 0
    if a.combine:
        return _combine(a.combine, a.out)
    dev = [s for s in a.seeds if s in DEV_SEEDS]
    bad = [s for s in a.seeds if s not in DEV_SEEDS and s not in REQUIRED_SEED_SET]
    if bad or (dev and not a.dev) or (dev and len(dev) != len(a.seeds)):
        print(f"[commitment veto v5] REFUSED: seeds {a.seeds} -- dev seeds {sorted(DEV_SEEDS)} need --dev and must "
              f"not mix with evaluation seeds {sorted(REQUIRED_SEED_SET)}", flush=True)
        return 2
    t0 = time.time()
    print(f"[commitment veto v5] seeds={a.seeds} backend={os.environ.get('SIM_BACKEND')} op={operating_point()}",
          flush=True)
    rows = [run_seed(s, determinism=not a.no_determinism, quick=a.quick) for s in a.seeds]
    if a.quick:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"kind": "DEV quick calibration probe (not evidence)",
                                           "operating_point": operating_point(), "per_seed": rows},
                                          indent=1, default=str))
        return 0
    summary = _decide(rows)
    summary["config"] = {"seeds": a.seeds, "dev": a.dev, "backend": os.environ.get("SIM_BACKEND")}
    summary["elapsed_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1, default=str))
    print(f"[commitment veto v5] {summary['verdict']} ({summary['n_go']}/{summary['n_seeds']} seeds pass every "
          f"required gate) -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
