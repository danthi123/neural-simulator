"""v4 of the curiosity x metacognition LC-NE lane: a PHASIC locus-coeruleus analog that raises the GAIN of
curiosity's ASK pool on the metacog edge's drive, by withdrawing an output-proportional negative feedback.

Pre-registration: `docs/plans/2026-09-24-curiosity-lcne-phasic-gain-PREREG.md` (read it first; the gates below are
copied from it and were frozen on dev seeds 7/8/9/10 before any evaluation-seed run of THIS mechanism).

WHY THIS RUNG EXISTS. v3 (`_curiosity_metacog_neuromod_gain_derisk.py`) is a 6-seed NO-GO (1/6, held-out 0/5;
`research/findings/2026-09-24-curiosity-metacog-lcne-modulator-6seed-NOGO-calibration-seed-only.md`): its lc_ne
population was graded by metacog evidence (G10 passed everywhere) but projected an ADDITIVE sub-threshold current
onto ASK, which owned only 8-25% of the ASK dynamic range. The verdict is on the METHOD (tonic, additive); the
capability stays open. Two sources name what the real system does instead:
  * Aston-Jones & Cohen 2005 (Annu Rev Neurosci 28:403-450): LC PHASIC bursts are locked to the decision and
    facilitate the ensuing response; tonic firing is a different mode.
  * Fan, To & Sciolino 2026 (Cell Rep 45(9):117990): PHASIC LC activation expands a cortical population's dynamic
    range, largely via multiplicative gain; tonic activation does not.

THE WALL QUESTION FIRST ("what does the real system run alongside this, that we replaced with a constant?").
A target neuron's output is held down by its OWN output-proportional slow negative feedback -- the Ca2+-activated
slow afterhyperpolarization (sAHP) that produces spike-frequency accommodation. Noradrenaline blocks that sAHP
(beta1 -> cAMP), which removes accommodation and ENHANCES the response to depolarizing input (Madison & Nicoll
1982 Nature 299:636; 1986 J Physiol 372:221, "increase the signal-to-noise ratio"). Slow negative feedback on a
steep F-I curve divides and linearizes its slope, and the details of the feedback mechanism do not matter as long
as it is slow (Ermentrout 1998 Neural Comput 10:1721). v3 had NO such feedback on ASK, so the only thing lc_ne could
do was add current. Withdrawing a feedback that is PROPORTIONAL TO THE OUTPUT is a gain change by construction:
where ASK is silent there is no feedback to withdraw.

Why NOT the balanced-background route (Chance, Abbott & Reyes 2002, Neuron 35:773): its divisive effect needs the
membrane NOISE that balanced background input brings; a pure conductance change is subtractive on firing rate
(Holt & Koch 1997, Neural Comput 9:1001). This pool has no OU / conductance noise (the production metacog config),
so a background-withdrawal circuit here would shift ASK's threshold -- an additive-looking effect.

THE CIRCUIT (every step neurons + synapses; host code drives ONLY metacog's input evidence, as production does):
  * `meta_schema` (metacog's margin comparator, unchanged) -> `lc_ne` (LC_N neurons), dense E, W_CMP_LC.
  * `lc_ne` -> `lc_ne`: alpha2-adrenoceptor AUTOINHIBITION, routed to the slow GIRK K+ conductance (receptor
    "gaba_b", E=-90 mV, tau 150 ms). alpha2 receptors on LC neurons open a K+ conductance (Williams, Henderson &
    North 1985, Neuroscience 14:95). This is what makes lc_ne PHASIC: one burst per event, then a pause.
  * `ask` -> `ask_fb` (N_FB fast-spiking relay neurons, dense E, W_ASK_FB) -> `ask` (slow GIRK K+ conductance,
    W_FB_ASK): ASK's own output-proportional slow negative feedback (the network form of the sAHP; see the
    residual list for why it is a relay population and not an intrinsic current).
  * `lc_ne` -> `ask_fb` (slow GIRK, W_LC_FB): a phasic lc_ne burst silences the feedback relay for ~150 ms, i.e.
    NE withdraws ASK's accommodation. transmission_gate "lc_ne_gain" is the LESION switch only (static 0/1).
  * The frozen point-edge `meta_schema -> ask` (weight 4.0, the conflict_xedge rung) carries transmission_gate
    "edge_drive": 1.0 in every arm except the G11 drive sweep, which scales it to vary the edge's drive while
    lc_ne's own input (from meta_schema) is untouched -- the only way to hold the modulator fixed while the
    modulated drive varies (the lc_ne raster is hash-checked identical across every sweep read).
  * ADDITIVE CONTROL (instrument only; its output gate "lc_add_ctrl" is CLOSED in every mechanism arm): `lc_add`,
    a clone of v3's lc_ne (excitatory, driven by meta_schema at v3's W=5.0) projecting ADDITIVE excitation onto
    ASK. It exists so G11's failing direction is verified on the real substrate, per seed.

ENGINE WORKAROUND (declared). `SimulationBridge.inject_explicit_wiring` keeps a STALE GABA_B routing mask when it
is called a second time (the `if self.cp_gabab_synapse_mask is None` guard added in 9ff3b6353), and the merge
framework injects twice. Measured: with enable_gabab on, 871 metacog `workspace` synapses were routed into GIRK and
metacog's balance jumped from ~0.02-0.09 to ~0.70. This organ's explicit_wiring_fn sets the mask to None before
the second inject so it is rebuilt from the pool's own keyed list; integrity checks verify (a) the routed set is
exactly the declared GIRK synapses plus curiosity's own striosome->snc, and (b) metacog's read is EXACTLY the
conflict_xedge base pool's. No `sim/` edit.

WHAT KIND OF GAIN (honesty). Withdrawing a feedback that only engages above the relay's own threshold is a SLOPE
(response-gain) increase with no threshold shift, whose size GROWS with the response -- not a constant-factor
scaling. G11 accepts a flat or rising ASK_on/ASK_off ratio and rejects a falling one (the additive signature).

RESIDUALS (declared, not closed here): the sAHP is carried by a relay population (`ask_fb`) because the engine has no
spike-triggered intrinsic K+ current that a synapse can modulate; NE's beta1/cAMP block of the sAHP is represented
by a Gi-type GIRK inhibition of that relay; every weight is hand-set on dev seeds, not grown; lc_ne's phasic burst is
driven by the comparator's summed output (a co-activation, conflict-like signal), not by a dedicated ACC/OFC
utility monitor; the point-edge's own drive sits near ASK's threshold on some seeds (dev 9: 0.43 Hz) -- the
operating point that a homeostatic set-point process would normally hold is not modelled.

FUNCTIONAL CORRELATE ONLY -- no phenomenal claim. Additive research runner: no `sim/` edit, no production flag, no
default flip; nothing in the live chat path imports this file.

Run:
  python -m research.runners._curiosity_lcne_phasic_gain_derisk --selftest            # gate logic, no sim
  SIM_BACKEND=numpy python -m research.runners._curiosity_lcne_phasic_gain_derisk --seeds 7 --dev \\
      --out research/findings/raw/_curiosity_lcne_phasic_gain_dev_s7.json             # dev-seed smoke (7/8/9/10)
  SIM_BACKEND=numpy python -m research.runners._curiosity_lcne_phasic_gain_derisk --seeds 42 \\
      --out research/findings/raw/_curiosity_lcne_phasic_gain_s42.json                # one eval seed (pool line)
  python -m research.runners._curiosity_lcne_phasic_gain_derisk --combine <s42.json> ... <s102.json> \\
      --out research/findings/raw/_curiosity_lcne_phasic_gain_6seed_combined.json     # the pre-registered verdict
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
from research.runners.onebrain_merge_framework import REGISTRY, CrossEdge, OrganDescriptor, merge_organs  # noqa: E402
from research.runners.metacog_production_organ import MetacogProductionOrgan, nmda_norm_margin  # noqa: E402
from research.runners._curiosity_metacog_conflict_xedge_derisk import (  # noqa: E402
    build_pool as build_base_pool, Recorder as BaseRecorder, coupled_sweep as base_coupled_sweep,
    METACOG_MARGIN, XEDGE_W, XEDGE_KEY, _metacog_het, _swapped_idx, _meta_exact, _curiosity_production_threshold,
    perm_null, spearman, level_rho, EVIDENCE_GRID, READ_REPS, STEPS_PER_REP, META_REGIONS, CMP_REGIONS,
    G1_RHO_MAX, G1_MIN_RANGE_HZ, G7_RHO_MAX, G8_RHO_MIN,
)

RS = "IZH2007_RS_CORTICAL_PYRAMIDAL"
FS = "IZH2007_FS_CORTICAL_INTERNEURON"

# ── FROZEN operating point (calibrated on DEV seeds 7/8/9/10 ONLY, never on 42/43/44/100/101/102; PREREG §2) ──────
# Criterion: the WORST CASE over dev seeds 7, 8, 10 clears every mechanism-controlled floor (G3, G10, G11, G12) with
# margin. Dev seed 9 is excluded from the criterion and reported: its point-edge alone drives ASK to 0.43 Hz at the
# most uncertain level (< G1's 1.0 Hz range floor), so there is no response for any modulator to scale.
LC_N = 20             # locus-coeruleus-analog population
W_CMP_LC = 5.0        # meta_schema -> lc_ne (dense E). v3's graded value, re-used (dev seed 7: 8.0 makes lc_ne fire on
                      # the comparator's ONSET transient at every evidence level, i.e. not uncertainty-locked)
W_LC_AUTO = 3.0       # lc_ne -> lc_ne alpha2 autoinhibition (GIRK). 8 flattens lc_ne's grading on seed 8
                      # (rho_lc -0.10); 16 silences it on seed 10
N_FB = 20             # ask_fb relay population (fast-spiking; an RS relay gave smaller G3 on every dev seed)
W_ASK_FB = 50.0       # ask -> ask_fb (dense E). The dev-worst G3 rises with it: 20 -> 0.06, 35 -> 0.20, 50 -> 0.29
W_FB_ASK = 16.0       # ask_fb -> ask (slow GIRK K+): ASK's output-proportional slow negative feedback
W_LC_FB = 35.0        # lc_ne -> ask_fb (slow GIRK): the phasic NE withdrawal of the feedback
ADD_N = 20            # ADDITIVE CONTROL population (a clone of v3's lc_ne)
W_CMP_ADD = 5.0       # meta_schema -> lc_add (v3's frozen CMP_TO_LC_W)
W_ADD_ASK = 2.0       # lc_add -> ask (ADDITIVE excitation). Dev seeds: its own ask/off ratio reaches 1.36-2.6 on the
                      # rising limb, i.e. a real effect of the mechanism's size, so its G11c failure is not "too small"

LC_GAIN_GATE = "lc_ne_gain"      # lesion switch for lc_ne -> ask_fb (G3)
FB_LOOP_GATE = "ask_fb_loop"     # lesion switch for ask_fb -> ask (the feedback loop itself)
LC_AUTO_GATE = "lc_ne_autoinh"   # lesion switch for lc_ne's alpha2 autoinhibition (reported diagnostic)
ADD_GATE = "lc_add_ctrl"         # the additive control's output (CLOSED except in the G11 additive-control arm)
EDGE_GATE = "edge_drive"         # the point-edge's current scale (1.0 except in the G11 drive sweep)
NEW_REGIONS = ("lc_ne", "ask_fb", "lc_add")
BYSTANDERS = ("cue", "striosome_value", "reward_us", "snc")   # curiosity's own regions that must stay silent
DEV_SEEDS = frozenset({7, 8, 9, 10})
REQUIRED_SEED_SET = frozenset({42, 43, 44, 100, 101, 102})

# ── pre-registered thresholds (PREREG §3) ────────────────────────────────────────────────────────────────────
G3_GAIN_ATTRIB_MIN = 0.2          # lc_ne pathway owns >= this fraction of the combined ASK dynamic range (as v3)
LC_GRADED_RHO_MAX = -0.3          # G10 (as v3): rho(evidence, lc_ne Hz) <= this
LC_GRADED_MIN_RANGE_HZ = 0.3      # G10 (as v3): lc_ne's own Hz range across the grid >= this
BURST_WIN_STEPS = 25              # G12: a rep's lc_ne burst window, from its first lc_ne spike
PHASIC_CONC_MIN = 0.95            # G12a: pooled per-neuron fraction of lc_ne spikes inside the burst window >= this
PAUSE_RATIO_MIN = 1.5             # G12b: lc_ne spikes with alpha2 autoinhibition lesioned / intact >= this
EV_G11 = 0.0                      # G11: the evidence level of the drive sweep (the most uncertain level)
G11_GRID = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)   # G11: edge_drive values around the operating edge weight (1.0).
                                  # The low end exists for seeds whose reference peaks early (a stronger base
                                  # drive peaks at a lower edge_drive: dev 7 at 1.0, dev 8/10 at 1.1)
G11_OFF_FLOOR_HZ = 0.25           # G11: a grid point is DEFINED when the lc-off ASK rate is >= this
G11_MIN_DEFINED = 3               # G11: >= this many points on the reference's RISING LIMB, else UNDEFINED (fail)
G11_OFFSET_MAX_HZ = 0.05          # G11a: |lc effect| with the edge closed (edge_drive=0) <= this
G11_GAIN_MIN = 0.15               # G11b: ASK_on/ASK_off at the top of the rising limb >= 1 + this
G11_TREND_MIN = 0.0               # G11c: Spearman(drive, ASK_on/ASK_off) over the rising limb >= this
G11_RATIO_RES = 0.05              # G11c: ratios are quantized to this step before ranking (ties within it)


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  the organ
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _dense(pre, post, w, **kw):
    pre = np.asarray(pre, np.int64)
    post = np.asarray(post, np.int64)
    P = np.repeat(pre, post.size)
    Q = np.tile(post, pre.size)
    d = {"pre_indices": P, "post_indices": Q, "initial_weights": np.full(P.size, float(w), np.float32),
         "plastic": False, "conn_type": "LCNE_V4", "count": int(P.size)}
    d.update(kw)
    return d


def _lcg_spec(seed):
    return ([BrainRegion(name="lc_ne", n_neurons=LC_N, exc_fraction=0.0, internal_density=0.0, enable_nmda=False,
                         izh_neuron_type=RS),
             BrainRegion(name="ask_fb", n_neurons=N_FB, exc_fraction=0.0, internal_density=0.0, enable_nmda=False,
                         izh_neuron_type=FS),
             BrainRegion(name="lc_add", n_neurons=ADD_N, exc_fraction=1.0, internal_density=0.0, enable_nmda=False,
                         izh_neuron_type=RS)], [], {})


# (name, source region, target region, weight, extra population keys). The GIRK-routed rows are the ones whose
# receptor is "gaba_b"; `_gabab_routing_check` verifies exactly these (plus curiosity's own striosome->snc) route.
def _wiring_rows():
    return [
        ("v4_cmp_to_lc", "meta_schema", "lc_ne", W_CMP_LC, {}),
        ("v4_lc_autoinh", "lc_ne", "lc_ne", W_LC_AUTO, {"receptor": "gaba_b", "transmission_gate": LC_AUTO_GATE}),
        ("v4_ask_to_fb", "ask", "ask_fb", W_ASK_FB, {}),
        ("v4_fb_to_ask", "ask_fb", "ask", W_FB_ASK, {"receptor": "gaba_b", "transmission_gate": FB_LOOP_GATE}),
        ("v4_lc_to_fb", "lc_ne", "ask_fb", W_LC_FB, {"receptor": "gaba_b", "transmission_gate": LC_GAIN_GATE}),
        ("ctrl_cmp_to_add", "meta_schema", "lc_add", W_CMP_ADD, {}),
        ("ctrl_add_to_ask", "lc_add", "ask", W_ADD_ASK, {"transmission_gate": ADD_GATE}),
    ]


def _lcg_wiring(bridge, rm):
    # ENGINE WORKAROUND (module docstring): drop the stale GABA_B mask from the pool's FIRST inject so the second
    # inject (the one this dict is unioned into) rebuilds it from its own keyed list.
    bridge.cp_gabab_synapse_mask = None
    return {name: _dense(rm.indices(src), rm.indices(dst), w, **kw) for name, src, dst, w, kw in _wiring_rows()}


def _lcg_post_inject(bridge):
    bridge.set_transmission_gate(ADD_GATE, 0.0)          # the additive control is OFF unless an arm opens it


LCG_ORGAN = OrganDescriptor(
    key="lcne_gain_organ", regions=NEW_REGIONS, spec_fn=_lcg_spec, config={"enable_gabab": True},
    explicit_wiring_fn=_lcg_wiring, post_inject_fn=_lcg_post_inject, param_het=True,
    scaffold_residuals=("hand-set weights for meta_schema->lc_ne, the alpha2 autoinhibition, the ask<->ask_fb "
                        "feedback loop and lc_ne->ask_fb (dev-seed 7/8 calibration); not Hebbian-grown; the sAHP is "
                        "carried by a relay population (ask_fb), not an intrinsic Ca2+-activated K+ current; NE's "
                        "beta1 block of the sAHP is represented by a Gi-type GIRK inhibition of that relay",))

XEDGE_GATED = CrossEdge(key=XEDGE_KEY, source_key="metacog", source_region="meta_schema", target_key="curiosity",
                        target_region="ask", init_weight=XEDGE_W, plastic=False, learn_rule="none",
                        freeze_rest=False, transmission_gate=EDGE_GATE)


def build_v4_pool(seed: int):
    """[metacog(het), curiosity, metacog_margin, lcne_gain_organ] + the frozen point-edge (gated for G11)."""
    pool = merge_organs([_metacog_het(), REGISTRY["curiosity"], METACOG_MARGIN, LCG_ORGAN], seed=int(seed),
                        wire=True, cross_edges=[XEDGE_GATED])
    pool.ensure_built()
    return pool


def _set(b, **gates):
    for k, v in gates.items():
        b.set_transmission_gate(k, float(v))


MECH_GATES = {LC_GAIN_GATE: 1.0, FB_LOOP_GATE: 1.0, LC_AUTO_GATE: 1.0, ADD_GATE: 0.0, EDGE_GATE: 1.0}


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  the instrument
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
class Recorder:
    """Wraps the pool's `_run_one_simulation_step` (metacog's production read calls it) and records, per step, the
    ASK / lc_ne / ask_fb / lc_add / bystander spike counts, plus running sha256 hashes of the metacog-side,
    comparator-side and lc_ne spike rasters. Read-only: it runs AFTER the step and never writes into the bridge."""

    def __init__(self, pool):
        b = pool.bridge
        rm = b.region_manager
        self.b = b

        def ix(names):
            return np.concatenate([np.asarray(rm.indices(n), np.int64) for n in names])
        self.idx = {"ask": ix(["ask"]), "lc_ne": ix(["lc_ne"]), "ask_fb": ix(["ask_fb"]), "lc_add": ix(["lc_add"]),
                    "bystanders": ix(BYSTANDERS)}
        self.meta_idx = ix(META_REGIONS)
        self.cmp_idx = ix(CMP_REGIONS)
        self.on = False
        self._orig = b._run_one_simulation_step
        b._run_one_simulation_step = self._step
        self.reset()

    def reset(self):
        self.counts = {k: [] for k in self.idx}
        self.rasters = {"lc_ne": [], "lc_add": []}
        self.h_meta = hashlib.sha256()
        self.h_cmp = hashlib.sha256()
        self.h_lc = hashlib.sha256()

    def _step(self):
        self._orig()
        if not self.on:
            return
        fs = np.asarray(to_host(self.b.cp_firing_states)).astype(bool)
        for k, ix in self.idx.items():
            self.counts[k].append(int(fs[ix].sum()))
        for k in self.rasters:
            self.rasters[k].append(fs[self.idx[k]].copy())
        self.h_meta.update(np.packbits(fs[self.meta_idx]).tobytes())
        self.h_cmp.update(np.packbits(fs[self.cmp_idx]).tobytes())
        self.h_lc.update(np.packbits(fs[self.idx["lc_ne"]]).tobytes())


def burst_stats(raster_by_rep) -> dict:
    """G12's raw counts, PER NEURON (the cellular definition of LC phasic firing: a brief burst, then a pause --
    Aston-Jones & Cohen 2005). `raster_by_rep`: bool array [reps, steps, neurons]. For every (rep, neuron) with >= 1
    spike: its first spike step, and how many of its spikes fall inside [first, first + BURST_WIN_STEPS). Pooled
    over (rep, neuron) pairs, so a neuron weighs by its spikes."""
    ras = np.asarray(raster_by_rep, bool)
    total = in_burst = active = 0
    firsts = []
    for rep in ras:
        n_spk = rep.sum(0)
        for j in np.nonzero(n_spk)[0]:
            s = rep[:, j]
            f = int(np.argmax(s))
            firsts.append(f)
            total += int(n_spk[j])
            in_burst += int(s[f:f + BURST_WIN_STEPS].sum())
            active += 1
    return {"spikes_total": total, "spikes_in_burst": in_burst, "active_neuron_reps": active,
            "active_reps": int(sum(1 for rep in ras if rep.any())),
            "median_first_spike_step": (float(np.median(firsts)) if firsts else None)}


def read_level(pool, org, rec, ev: float, swap: bool = False) -> dict:
    """ONE metacog production read (`judge` -> `nmda_norm_margin`, READ_REPS jittered reps) at evidence `ev`, with
    every recorded population's rate over exactly those simulation steps."""
    rec.reset()
    rec.on = True
    if swap:
        bal = float(nmda_norm_margin(org.bridge, org.xp, _swapped_idx(org.idx), org.snap, ev))
        conf = bool(bal >= org.threshold)
    else:
        j = org.judge(ev)
        bal, conf = float(j["balance"]), bool(j["confident"])
    rec.on = False
    out = {"evidence": ev, "balance": bal, "confident": conf}
    per_rep = {}
    for k, ix in rec.idx.items():
        c = np.asarray(rec.counts[k], np.float64)
        assert c.size == READ_REPS * STEPS_PER_REP, (k, c.size, READ_REPS, STEPS_PER_REP)
        per_rep[k] = c.reshape(READ_REPS, STEPS_PER_REP)
        hz = per_rep[k].sum(1) / ix.size / (STEPS_PER_REP * 1e-3)
        out[f"{k}_hz"] = float(hz.mean())
        if k in ("ask", "lc_ne"):
            out[f"{k}_hz_per_rep"] = [float(x) for x in hz]
    out["bystander_spikes"] = int(per_rep["bystanders"].sum())
    for k in ("lc_ne", "lc_add"):
        ras = np.asarray(rec.rasters[k], bool).reshape(READ_REPS, STEPS_PER_REP, -1)
        out[f"{k}_burst"] = burst_stats(ras)
    out["meta_raster_sha256"] = rec.h_meta.hexdigest()
    out["cmp_raster_sha256"] = rec.h_cmp.hexdigest()
    out["lc_raster_sha256"] = rec.h_lc.hexdigest()
    return out


def sweep(pool, org, rec, swap: bool = False) -> dict:
    org.ensure_built()
    with pool.sequence_isolation():
        levels = [read_level(pool, org, rec, ev, swap=swap) for ev in EVIDENCE_GRID]
    return {"levels": levels, "threshold": float(org.threshold)}


def prime(pool, org, rec):
    """One discarded read after any gate/weight change (v3 isolated a transient on the first reps after a
    `cp_connections.data` write; the full-sweep prime is kept for data writes, this one for gate toggles)."""
    with pool.sequence_isolation():
        read_level(pool, org, rec, 0.0)


def digest(sw) -> str:
    h = hashlib.sha256()
    for l in sw["levels"]:
        h.update(np.asarray(l["ask_hz_per_rep"] + l["lc_ne_hz_per_rep"] + [l["balance"], l["ask_fb_hz"]],
                            np.float64).tobytes())
        for k in ("meta_raster_sha256", "cmp_raster_sha256", "lc_raster_sha256"):
            h.update(l[k].encode())
    return h.hexdigest()


def _vals(sw, key):
    return [l[key] for l in sw["levels"]]


def _rng(sw, key="ask_hz"):
    v = _vals(sw, key)
    return float(max(v) - min(v)), float(max(v))


def floored_rho(values):
    """SECONDARY (reported, NOT a gate; PREREG §3): Spearman rho(evidence, level-mean rate) with every level mean below
    G1_SILENT_HZ set to 0 (tied silence) before ranking. G1 itself stays v3's RAW rho. Measured on the dev seeds, the
    floor is double-edged: it lifts seed 8 (raw -0.61 from a 0.02 -> 0.10 Hz confident-end tail) to -0.86 but drops
    seed 10 (raw -0.98) to -0.79 because the ties cap the attainable |rho| when few levels are supra-floor -- so it
    is reported to show how much of a G1 failure is sub-floor noise, and never used to pass or fail a seed."""
    v = np.asarray(values, np.float64)
    v = np.where(v < G1_SILENT_HZ, 0.0, v)
    return spearman(list(EVIDENCE_GRID), list(v))


def lc_burst_concentration(sw, key="lc_ne_burst") -> dict:
    tot = sum(l[key]["spikes_total"] for l in sw["levels"])
    inb = sum(l[key]["spikes_in_burst"] for l in sw["levels"])
    return {"spikes_total": tot, "spikes_in_burst": inb, "concentration": (inb / tot) if tot > 0 else None,
            "active_reps_at_most_uncertain": sw["levels"][0][key]["active_reps"]}


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  module-level gate predicates (SHARED by run_seed and --selftest, so a regression in either is caught by both)
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
G1_SILENT_HZ = 0.25   # SECONDARY floored rho only: level means below this are tied silence (= G11_OFF_FLOOR_HZ)


def gate_g1(rho, rng_hz):
    """v3's G1, unchanged: Spearman rho(evidence, level-mean ASK Hz) <= G1_RHO_MAX AND range >= G1_MIN_RANGE_HZ;
    a None rho or a sub-floor range fails (UNDEFINED is never a pass)."""
    return bool(rho is not None and rng_hz >= G1_MIN_RANGE_HZ and rho <= G1_RHO_MAX)


def gate_g8_relay_lesion(rho_relay):
    """A None (flat/uninformative) relay-lesion rho NEVER passes (v3 rule, unchanged)."""
    return bool(rho_relay is not None and rho_relay > G8_RHO_MIN)


def gate_g4_joint_lesion(rho_both):
    """INTEGRITY (reported, not required): a None (ASK-silent) both-lesion arm is the expected outcome."""
    return bool(rho_both is None or rho_both > G1_RHO_MAX)


def gate_g10_lc_graded(rho_lc, rng_lc):
    """lc_ne's own rate is evidence-graded (v3 thresholds, unchanged); None or a sub-floor range fails."""
    if rho_lc is None or rng_lc < LC_GRADED_MIN_RANGE_HZ:
        return False
    return bool(rho_lc <= LC_GRADED_RHO_MAX)


def gate_g12_phasic(conc, conc_autoinh_lesioned):
    """PHASIC = a burst, then a pause that the drive alone would not produce (the cellular LC phasic pattern;
    Aston-Jones & Cohen 2005; alpha2 autoinhibition, Williams et al. 1985). Three parts, all required:
      (a) BURST: >= PHASIC_CONC_MIN of lc_ne's spikes (pooled over every level x rep x neuron) fall inside
          BURST_WIN_STEPS of that neuron's first spike in that rep;
      (b) PAUSE IS LOAD-BEARING: lesioning the alpha2 autoinhibition raises lc_ne's total spike count by >=
          PAUSE_RATIO_MIN -- the comparator's drive continues after the burst, and the autoinhibition is what
          silences lc_ne (per-neuron concentration alone cannot tell this from a sparsely driven cell: dev-seed
          measurement, v3's lc clone reads 0.89-0.97 on it);
      (c) lc_ne fires on >= 1 rep at the most uncertain level (a silent lc_ne is UNDEFINED, never phasic)."""
    c = conc.get("concentration")
    n, n_les = conc.get("spikes_total", 0), conc_autoinh_lesioned.get("spikes_total", 0)
    if c is None or n <= 0 or conc.get("active_reps_at_most_uncertain", 0) < 1:
        return False
    return bool(c >= PHASIC_CONC_MIN and (n_les / n) >= PAUSE_RATIO_MIN)


def _trend(grid, ratios):
    """Spearman(drive, ASK_on/ASK_off), with the ratios first QUANTIZED to G11_RATIO_RES so that ratios equal within
    that resolution tie instead of being ranked on noise (selftest: a pure x1.4 gain computed in float64 gave ratios
    1.4 and 1.3999999999999997 and an unquantized rank trend of -0.13). A constant ratio (a pure response-gain
    scaling) then has zero rank variance, `spearman` returns None, and that is a flat trend (0.0) -- exactly what a
    pure multiplicative gain predicts. An additive shift's ratio falls by far more than one resolution step."""
    q = np.round(np.asarray(ratios, np.float64) / G11_RATIO_RES) * G11_RATIO_RES
    r = spearman(list(grid), list(q))
    return 0.0 if r is None else float(r)


def g11_eval(on: dict, off: dict, add: dict, grid=G11_GRID) -> dict:
    """G11 — MULTIPLICATIVE, NOT ADDITIVE (PREREG §3). Inputs map edge_drive -> ASK Hz at EV_G11 for three arms that
    share an IDENTICAL lc_ne raster (the caller verifies the hash): `on` (lc_ne -> ask_fb open), `off` (closed), `add`
    (the v3-style additive control open instead). Keys must include 0.0 (edge closed) and every `grid` value.

      * DEFINED points: grid values where off >= G11_OFF_FLOOR_HZ. The RISING LIMB is the defined points up to the
        drive at which the reference (off) response peaks. Measured on the dev seeds: with ASK's slow negative
        feedback intact, a drive well above the operating point makes ASK fire on the comparator's early ONSET
        transient, recruits the feedback before any phasic lc_ne burst exists, and the reference response FALLS
        with more drive. A ratio over a collapsing denominator would inflate "gain", so only the rising limb is
        scored. Fewer than G11_MIN_DEFINED limb points -> UNDEFINED (fail).
      * G11a no offset: |on(0) - off(0)| <= G11_OFFSET_MAX_HZ (the modulator does nothing without the drive).
      * G11b real gain: on/off at the top of the rising limb >= 1 + G11_GAIN_MIN.
      * G11c scales with the drive: Spearman(drive, on/off) over the rising limb >= G11_TREND_MIN. An ADDITIVE input
        shift gives a ratio that FALLS with drive for any threshold-power-law rate curve ((x+c-t)/(x-t))^n, which is
        also why the Murphy & Miller 2003 expansive-nonlinearity case -- additive input that LOOKS multiplicative in
        slope -- still fails this test; so does an input-gain f(kx) with a threshold. Only a response (output) gain
        change with no threshold shift keeps the ratio flat or rising.
      * INSTRUMENT VALIDITY, measured on the same seed: the additive control must itself FAIL G11c while producing a
        real effect (max over defined points of add/off >= 1 + G11_GAIN_MIN). If it passes G11c, this instrument
        cannot tell additive from multiplicative on this seed -> UNDEFINED (fail), never a pass."""
    defined = [g for g in grid if off[g] >= G11_OFF_FLOOR_HZ]
    peak = max(defined, key=lambda g: off[g]) if defined else None
    limb = [g for g in defined if g <= peak] if defined else []
    res = {"defined_points": defined, "n_defined": len(defined), "rising_limb": limb, "n_limb": len(limb)}
    if len(limb) < G11_MIN_DEFINED:
        res.update({"pass": False, "undefined": f"{len(limb)} points on the reference's rising limb "
                                                 f"(need >= {G11_MIN_DEFINED})"})
        return res
    r_on = [on[g] / off[g] for g in limb]
    r_add = [add[g] / off[g] for g in limb]
    t_on, t_add = _trend(limb, r_on), _trend(limb, r_add)
    offset = abs(on[0.0] - off[0.0])
    gain_top = on[limb[-1]] / off[limb[-1]]
    add_effect = max(r_add) - 1.0
    instrument_valid = bool(t_add < G11_TREND_MIN and add_effect >= G11_GAIN_MIN)
    parts = {"G11a_no_offset": bool(offset <= G11_OFFSET_MAX_HZ),
             "G11b_gain_at_top_of_rising_limb": bool(gain_top >= 1.0 + G11_GAIN_MIN),
             "G11c_effect_scales_with_drive": bool(t_on >= G11_TREND_MIN)}
    res.update({"ratio_on": r_on, "ratio_add": r_add, "trend_on": t_on, "trend_add": t_add, "offset_hz": offset,
                "gain_at_top_of_rising_limb": gain_top, "additive_effect": add_effect,
                "instrument_valid_additive_control_fails": instrument_valid, "parts": parts,
                "additive_control_parts": {"G11c_effect_scales_with_drive": bool(t_add >= G11_TREND_MIN)},
                "pass": bool(instrument_valid and all(parts.values()))})
    if not instrument_valid:
        res["undefined"] = "the additive control did not fail G11c (or had no effect): instrument cannot discriminate"
    return res


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  integrity checks (each a real measurement; all are Verdict preconditions in _decide)
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _edge_map(bridge):
    coo = bridge.cp_connections.tocoo()
    r = np.asarray(to_host(coo.row)); c = np.asarray(to_host(coo.col)); d = np.asarray(to_host(coo.data))
    return r, c, d


def _byte_off_check(v4_bridge, seed: int) -> dict:
    """The v4 pool's connectivity, minus EXACTLY the synapses this rung declares (every synapse touching lc_ne /
    ask_fb / lc_add, plus the point-edge meta_schema -> ask), must equal the conflict_xedge rung's own coupled=False
    pool, key for key and weight for weight (==). Also checks the removed set is exactly the declared set's size."""
    rm = v4_bridge.region_manager
    new = np.concatenate([np.asarray(rm.indices(n), np.int64) for n in NEW_REGIONS])
    meta = np.asarray(rm.indices("meta_schema"), np.int64)
    ask = np.asarray(rm.indices("ask"), np.int64)
    r, c, d = _edge_map(v4_bridge)
    declared = np.isin(r, new) | np.isin(c, new) | (np.isin(r, meta) & np.isin(c, ask))
    kept = {(int(a), int(b)): float(w) for a, b, w in zip(r[~declared], c[~declared], d[~declared])}
    base = build_base_pool(seed, coupled=False)
    rb, cb, db = _edge_map(base.bridge)
    base_map = {(int(a), int(b)): float(w) for a, b, w in zip(rb, cb, db)}
    n_expected = sum(len(rm.indices(s)) * len(rm.indices(t)) for _, s, t, _, _ in _wiring_rows()) + meta.size * ask.size
    return {"PASS": bool(kept == base_map and int(declared.sum()) == n_expected),
            "base_connectivity_identical": bool(kept == base_map), "n_declared_removed": int(declared.sum()),
            "n_declared_expected": int(n_expected), "n_kept": len(kept), "n_base": len(base_map)}


def _gabab_routing_check(bridge) -> dict:
    """The engine workaround's own check: the synapses routed into the GIRK conductance are EXACTLY the declared GIRK
    rows plus curiosity's own striosome_value -> snc pathway (all of it), and nothing else (in particular no metacog
    synapse -- the stale-mask bug routed 871 `workspace` synapses)."""
    rm = bridge.region_manager
    names = list(rm.region_indices_dict())
    region_of = {}
    for n in names:
        for i in rm.indices(n):
            region_of[int(i)] = n
    r, c, _ = _edge_map(bridge)
    m = bridge.cp_gabab_synapse_mask
    if m is None:
        return {"PASS": False, "why": "no GABA_B mask"}
    mask = np.asarray(to_host(m))[:r.size].astype(bool)
    routed = {}
    for a, b in zip(r[mask], c[mask]):
        k = f"{region_of[int(a)]}->{region_of[int(b)]}"
        routed[k] = routed.get(k, 0) + 1
    allowed = {f"{s}->{t}" for _, s, t, _, kw in _wiring_rows() if kw.get("receptor") == "gaba_b"}
    full = {f"{s}->{t}": len(rm.indices(s)) * len(rm.indices(t))
            for _, s, t, _, kw in _wiring_rows() if kw.get("receptor") == "gaba_b"}
    strio = np.isin(r, np.asarray(rm.indices("striosome_value"))) & np.isin(c, np.asarray(rm.indices("snc")))
    full["striosome_value->snc"] = int(strio.sum())
    allowed.add("striosome_value->snc")
    ok = set(routed) <= allowed and all(routed.get(k, 0) == v for k, v in full.items())
    return {"PASS": bool(ok), "routed_by_region_pair": routed, "expected": full}


def _base_pool_metacog(seed: int) -> dict:
    """metacog's read on the conflict_xedge rung's OWN coupled pool (no v4 organ, enable_gabab off), same prime +
    measure sequence -- the reference for 'the v4 organ and the GIRK flag perturb metacog NOT AT ALL'."""
    base = build_base_pool(seed, coupled=True)
    org = MetacogProductionOrgan(seed=seed, shared=base)
    rec = BaseRecorder(base)
    with base.sequence_isolation():
        base_coupled_sweep(base, org, rec)
    with base.sequence_isolation():
        return base_coupled_sweep(base, org, rec)


def _metacog_vs_base(v4_sweep, base_sweep) -> dict:
    m = _meta_exact(v4_sweep, base_sweep)
    cmp_eq = _vals(v4_sweep, "cmp_raster_sha256") == [l["cmp_raster_sha256"] for l in base_sweep["levels"]]
    return dict(m, comparator_raster_equal=bool(cmp_eq),
                PASS=bool(m["balance_equal"] and m["threshold_equal"] and m["confident_equal"]
                          and m["metacog_raster_equal"] and cmp_eq))


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  one seed
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _new_session(seed: int):
    pool = build_v4_pool(seed)
    org = MetacogProductionOrgan(seed=seed, shared=pool)
    org.ensure_built()
    rec = Recorder(pool)
    _set(pool.bridge, **MECH_GATES)
    with pool.sequence_isolation():                  # full-sweep prime on the freshly built pool (v3 protocol)
        for ev in EVIDENCE_GRID:
            read_level(pool, org, rec, ev)
    return pool, org, rec


def _combined_intact(pool, org, rec) -> dict:
    prime(pool, org, rec)
    return sweep(pool, org, rec)


def run_seed(seed: int, determinism: bool = True, verbose: bool = True) -> dict:
    t0 = time.time()
    pool, org, rec = _new_session(seed)
    b, xp = pool.bridge, pool.xp
    combined = _combined_intact(pool, org, rec)

    def arm(gates=None, swap=False):
        _set(b, **MECH_GATES)
        _set(b, **(gates or {}))
        prime(pool, org, rec)
        sw = sweep(pool, org, rec, swap=swap)
        # the lesion is verified to STILL HOLD at measurement (docs/TERMS.md "lesion"): the gate values read back
        # off the bridge right after the sweep, before anything is restored
        sw["gates_at_measurement"] = {k: float(v) for k, v in b._transmission_gate_values.items()}
        _set(b, **MECH_GATES)
        return sw

    swap = arm(swap=True)                                                 # G7
    gain_lesion = arm({LC_GAIN_GATE: 0.0})                                # G3
    edge_lesion = arm({EDGE_GATE: 0.0})                                   # reported (lc_ne alone drives nothing)
    both_lesion = arm({EDGE_GATE: 0.0, LC_GAIN_GATE: 0.0})                # G4 (integrity)
    loop_lc_on = arm({FB_LOOP_GATE: 0.0})                                 # integrity: lc acts ONLY through the loop
    loop_lc_off = arm({FB_LOOP_GATE: 0.0, LC_GAIN_GATE: 0.0})
    autoinh_lesion = arm({LC_AUTO_GATE: 0.0})                             # diagnostic: phasic needs alpha2
    swap_gain_lesion = arm({LC_GAIN_GATE: 0.0}, swap=True)                # REPORTING ONLY (PREREG v1.1): G7's own
    #   statistic with the modulator closed, so a G7 failure can be attributed to the base circuit or to lc_ne

    # G8: lesion metacog's comparator relay (meta_margin_fs -> meta_schema inhibition), everything else intact.
    rm = b.region_manager
    r, c, _ = _edge_map(b)
    relay_mask = np.isin(r, np.asarray(rm.indices("meta_margin_fs"))) & np.isin(c, np.asarray(rm.indices("meta_schema")))
    data = np.asarray(to_host(b.cp_connections.data)).copy()
    data2 = data.copy()
    data2[relay_mask] = 0.0
    b.cp_connections.data = xp.asarray(data2, dtype=b.cp_connections.data.dtype)
    with pool.sequence_isolation():
        for ev in EVIDENCE_GRID:
            read_level(pool, org, rec, ev)                # full prime after a data write (v3 protocol)
    relay_lesion = sweep(pool, org, rec)
    relay_lesion["relay_weight_sum_at_measurement"] = float(
        np.asarray(to_host(b.cp_connections.data))[relay_mask].sum())
    b.cp_connections.data = xp.asarray(data, dtype=b.cp_connections.data.dtype)
    with pool.sequence_isolation():
        for ev in EVIDENCE_GRID:
            read_level(pool, org, rec, ev)

    # G11: drive sweep at EV_G11 -- the edge's drive varies, lc_ne's input (meta_schema) does not.
    g11_reads = {}
    g11_lc_hashes = set()
    for name, gates in (("on", {}), ("off", {LC_GAIN_GATE: 0.0}), ("add", {LC_GAIN_GATE: 0.0, ADD_GATE: 1.0})):
        g11_reads[name] = {}
        for g in (0.0,) + tuple(G11_GRID):
            _set(b, **MECH_GATES)
            _set(b, **gates)
            _set(b, **{EDGE_GATE: g})
            prime(pool, org, rec)
            with pool.sequence_isolation():
                lv = read_level(pool, org, rec, EV_G11)
            gm = b._transmission_gate_values
            want = dict(MECH_GATES, **gates, **{EDGE_GATE: g})
            lv["gates_held"] = bool(all(float(gm[k]) == float(v) for k, v in want.items()))
            g11_reads[name][g] = lv
            g11_lc_hashes.add(lv["lc_raster_sha256"])
    _set(b, **MECH_GATES)
    g11 = g11_eval({g: v["ask_hz"] for g, v in g11_reads["on"].items()},
                   {g: v["ask_hz"] for g, v in g11_reads["off"].items()},
                   {g: v["ask_hz"] for g, v in g11_reads["add"].items()}, grid=tuple(G11_GRID))
    g11_lc_fixed = len(g11_lc_hashes) == 1
    g11["lc_raster_identical_across_all_sweep_reads"] = bool(g11_lc_fixed)
    if not g11_lc_fixed:
        g11["pass"] = False
        g11["undefined"] = "lc_ne raster differed across the drive sweep: the modulator was not held fixed"

    restored = arm()

    # ── statistics ──
    rng_c, peak_c = _rng(combined)
    rng_g, peak_g = _rng(gain_lesion)
    rng_e, peak_e = _rng(edge_lesion)
    rng_b, peak_b = _rng(both_lesion)
    rho_raw = level_rho(combined)
    rho_fl = floored_rho(_vals(combined, "ask_hz"))                      # secondary only
    rho_swap = level_rho(swap)
    rho_swap_fl = floored_rho(_vals(swap, "ask_hz"))                     # secondary only
    rho_lesion = level_rho(gain_lesion)                                   # the base (modulator-closed) arm's own G1 rho
    rho_swap_lesion = level_rho(swap_gain_lesion)                         # ... and its own G7 rho (reporting only)
    rho_relay = level_rho(relay_lesion)
    rho_both = level_rho(both_lesion)
    rho_lc = spearman(list(EVIDENCE_GRID), _vals(combined, "lc_ne_hz"))
    rng_lc, peak_lc = _rng(combined, "lc_ne_hz")
    conc = lc_burst_concentration(combined)
    conc_autoinh = lc_burst_concentration(autoinh_lesion)
    conc_add = lc_burst_concentration(combined, key="lc_add_burst")
    from tools.lab import attributable_to
    attrib_gain = attributable_to(f"seed{seed} ASK range = the lc_ne phasic gain pathway", rng_c, rng_g)
    attrib_edge = attributable_to(f"seed{seed} ASK range = the point edge", rng_c, rng_e)
    g9 = perm_null(combined, seed)

    m_arms = {name: _meta_exact(combined, sw) for name, sw in
              (("gain_lesion", gain_lesion), ("edge_lesion", edge_lesion), ("both_lesion", both_lesion),
               ("loop_lc_on", loop_lc_on), ("loop_lc_off", loop_lc_off), ("autoinh_lesion", autoinh_lesion))}
    cmp_equal = all(_vals(combined, "cmp_raster_sha256") == _vals(sw, "cmp_raster_sha256")
                    for sw in (gain_lesion, edge_lesion, both_lesion, loop_lc_on, loop_lc_off, autoinh_lesion))
    g5 = bool(cmp_equal and all(m["balance_equal"] and m["metacog_raster_equal"] and m["confident_equal"]
                                for m in m_arms.values()))
    loop_only = bool(_vals(loop_lc_on, "ask_hz") == _vals(loop_lc_off, "ask_hz")
                     and [l["ask_hz_per_rep"] for l in loop_lc_on["levels"]]
                     == [l["ask_hz_per_rep"] for l in loop_lc_off["levels"]])
    all_sweeps = (combined, swap, gain_lesion, edge_lesion, both_lesion, loop_lc_on, loop_lc_off, autoinh_lesion,
                  swap_gain_lesion, relay_lesion, restored)
    bystander_spikes = int(sum(l["bystander_spikes"] for sw in all_sweeps for l in sw["levels"])
                           + sum(v["bystander_spikes"] for arm_ in g11_reads.values() for v in arm_.values()))
    dig = digest(combined)
    restore_ok = digest(restored) == dig
    held = {"gain_lesion": gain_lesion["gates_at_measurement"][LC_GAIN_GATE] == 0.0,
            "edge_lesion": edge_lesion["gates_at_measurement"][EDGE_GATE] == 0.0,
            "both_lesion": (both_lesion["gates_at_measurement"][EDGE_GATE] == 0.0
                            and both_lesion["gates_at_measurement"][LC_GAIN_GATE] == 0.0),
            "loop_lesion_lc_on": (loop_lc_on["gates_at_measurement"][FB_LOOP_GATE] == 0.0
                                  and loop_lc_on["gates_at_measurement"][LC_GAIN_GATE] == 1.0),
            "loop_lesion_lc_off": (loop_lc_off["gates_at_measurement"][FB_LOOP_GATE] == 0.0
                                   and loop_lc_off["gates_at_measurement"][LC_GAIN_GATE] == 0.0),
            "autoinhibition_lesion": autoinh_lesion["gates_at_measurement"][LC_AUTO_GATE] == 0.0,
            "class_swap_gain_lesion": swap_gain_lesion["gates_at_measurement"][LC_GAIN_GATE] == 0.0,
            "relay_lesion": relay_lesion["relay_weight_sum_at_measurement"] == 0.0,
            "g11_drive_sweep_gates": all(v["gates_held"] for rd in g11_reads.values() for v in rd.values()),
            "additive_control_closed_in_mechanism_arms": all(
                sw["gates_at_measurement"][ADD_GATE] == 0.0 for sw in (swap, gain_lesion, edge_lesion, both_lesion,
                                                                       loop_lc_on, loop_lc_off, autoinh_lesion,
                                                                       swap_gain_lesion, restored))}

    det = {"checked": False}
    if determinism:
        cmd = [sys.executable, "-m", "research.runners._curiosity_lcne_phasic_gain_derisk", "--digest-only",
               "--seeds", str(seed)] + (["--dev"] if seed in DEV_SEEDS else []) + (["--set"] + _OVERRIDES
                                                                                  if _OVERRIDES else [])
        out = subprocess.run(cmd, cwd=str(_REPO), capture_output=True, text=True, env=dict(os.environ))
        line = [x for x in out.stdout.splitlines() if x.startswith("DIGEST ")]
        other = line[-1].split()[-1] if line else None
        det = {"checked": True, "digest_main": dig, "digest_fresh_process": other, "equal": bool(other == dig),
               "rc": out.returncode}

    cal = _curiosity_production_threshold(seed)
    thr = cal.get("threshold_hz")
    unc = [l for l in combined["levels"] if not l["confident"]]
    con = [l for l in combined["levels"] if l["confident"]]
    s1_undefined = thr is None or not unc or not con
    s1 = (not s1_undefined) and max(l["ask_hz"] for l in unc) >= thr and max(l["ask_hz"] for l in con) < thr

    checks_required = {
        "G1_monotone_rho<=-0.8": gate_g1(rho_raw, rng_c),
        "G3_gain_pathway_load_bearing": bool(attrib_gain is not None and attrib_gain >= G3_GAIN_ATTRIB_MIN),
        "G6_determinism_fresh_process_hash": bool(det.get("equal")) if determinism else None,
        "G7_class_swap_monotone": bool(rho_swap is not None and rho_swap <= G7_RHO_MAX),
        "G8_relay_lesion_abolishes_coupling": gate_g8_relay_lesion(rho_relay),
        "G10_lc_ne_evidence_graded": gate_g10_lc_graded(rho_lc, rng_lc),
        "G11_multiplicative_not_additive": bool(g11["pass"]),
        "G12_lc_ne_phasic": gate_g12_phasic(conc, conc_autoinh),
    }
    checks_integrity = {
        "G4_joint_lesion_breaks_coupling": gate_g4_joint_lesion(rho_both),
        "G5_metacog_unchanged_EXACT_across_arms": g5,
    }
    required = [k for k, v in checks_required.items() if v is not None]
    go = all(checks_required[k] for k in required)

    integrity = {
        "byte_off": _byte_off_check(b, seed),
        "gabab_routing": _gabab_routing_check(b),
        "metacog_vs_base_pool": _metacog_vs_base(combined, _base_pool_metacog(seed)),
        "restore_exact": bool(restore_ok),
        "lesions_held_at_measurement": held,
        "lc_acts_only_through_feedback_loop": loop_only,
        "bystander_spikes_all_arms": bystander_spikes,
        "no_host_novelty_signal": float(getattr(b.core_config, "current_novelty_signal", 0.0) or 0.0) == 0.0,
        "neuromodulator_subsystem_enabled": bool(getattr(b.core_config, "enable_neuromodulator_subsystem", False)),
        "gates_after_run": dict(b._transmission_gate_values),
    }
    res = {
        "seed": seed, "go": bool(go), "dev_seed": seed in DEV_SEEDS, "checks": checks_required,
        "checks_integrity": checks_integrity,
        "rho": rho_raw, "rho_swap": rho_swap, "rho_gain_lesion_arm": rho_lesion,
        "rho_swap_gain_lesion_arm": rho_swap_lesion,
        "secondary_floored_rho": {"combined": rho_fl, "class_swap": rho_swap_fl}, "rho_relay_lesion": rho_relay,
        "rho_both_lesion": rho_both, "rho_lc_ne": rho_lc,
        "ask_range_hz": {"combined": rng_c, "gain_lesion": rng_g, "edge_lesion": rng_e, "both_lesion": rng_b},
        "ask_peak_hz": {"combined": peak_c, "gain_lesion": peak_g, "edge_lesion": peak_e, "both_lesion": peak_b},
        "lc_ne_range_hz": rng_lc, "lc_ne_peak_hz": peak_lc,
        "attributable_frac": {"gain": attrib_gain, "edge": attrib_edge},
        "g11": g11,
        "g11_ask_hz": {name: {str(g): v["ask_hz"] for g, v in rd.items()} for name, rd in g11_reads.items()},
        "g11_ask_fb_hz": {name: {str(g): v["ask_fb_hz"] for g, v in rd.items()} for name, rd in g11_reads.items()},
        "phasic": {"lc_ne": conc, "lc_ne_autoinhibition_lesioned": conc_autoinh, "lc_add_v3_clone": conc_add},
        "s1_reaches_production_threshold": bool(s1), "s1_undefined": bool(s1_undefined), "s1_threshold_hz": thr,
        "g9_perm_null": g9, "metacog_exact": m_arms, "comparator_raster_equal_across_arms": bool(cmp_equal),
        "determinism": det, "integrity": integrity, "production_threshold_calibration": cal,
        "arms": {"combined_intact": combined, "class_swap": swap, "gain_lesion": gain_lesion,
                 "edge_lesion": edge_lesion, "both_lesion": both_lesion, "loop_lesion_lc_on": loop_lc_on,
                 "loop_lesion_lc_off": loop_lc_off, "autoinhibition_lesion": autoinh_lesion,
                 "relay_lesion": relay_lesion, "class_swap_gain_lesion": swap_gain_lesion},
        "elapsed_s": round(time.time() - t0, 1),
    }
    if verbose:
        print(f"[seed {seed}] rho={rho_raw} (floored {rho_fl}; lesion arm {rho_lesion}) swap={rho_swap} "
              f"(lesion arm {rho_swap_lesion}) "
              f"relay={rho_relay} rho_lc={rho_lc} "
              f"attrib_gain={attrib_gain} G11={g11.get('pass')} (trend_on={g11.get('trend_on')} "
              f"trend_add={g11.get('trend_add')} gain_top={g11.get('gain_at_top_of_rising_limb')} "
              f"limb={g11.get('rising_limb')}) "
              f"phasic={conc['concentration']} det={det.get('equal')} GO={go} ({res['elapsed_s']}s)", flush=True)
        print(f"[seed {seed}] ask combined={[round(x, 2) for x in _vals(combined, 'ask_hz')]} "
              f"gain_lesion={[round(x, 2) for x in _vals(gain_lesion, 'ask_hz')]}", flush=True)
        print(f"[seed {seed}] checks={checks_required} integrity={checks_integrity} "
              f"byte_off={integrity['byte_off']['PASS']} gabab={integrity['gabab_routing']['PASS']} "
              f"metacog_vs_base={integrity['metacog_vs_base_pool']['PASS']} restore={restore_ok} "
              f"loop_only={loop_only} bystanders={bystander_spikes}", flush=True)
    return res


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  selftest (no simulation): every gate predicate, in BOTH directions, on the SAME functions run_seed scores with
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _selftest_gate_logic():
    G = G11_GRID

    def curve(f):
        d = {0.0: 0.0}
        d.update({g: float(f(g)) for g in G})
        return d
    # a threshold-linear base response (above floor at every grid point)
    off = curve(lambda g: 4.0 * (g - 0.6))
    # (1) pure response gain x1.4: flat ratio -> trend 0.0 (spearman None) -> passes G11c
    mult = curve(lambda g: 1.4 * off[g])
    # (2) feedback-withdrawal-like: gain grows with the response -> rising ratio -> passes
    grow = curve(lambda g: off[g] * (1.0 + 0.5 * (g - 0.7)))
    # (3) ADDITIVE input shift of 0.2 on the same threshold-linear curve -> falling ratio -> must FAIL G11c
    add = curve(lambda g: 4.0 * (g + 0.2 - 0.6))
    # (4) ADDITIVE shift on an EXPANSIVE (power-law) curve -- the Murphy & Miller case -- must also FAIL G11c
    off_pl = curve(lambda g: 6.0 * (g - 0.6) ** 2)
    add_pl = curve(lambda g: 6.0 * (g + 0.2 - 0.6) ** 2)
    mult_pl = curve(lambda g: 1.4 * off_pl[g])
    r = g11_eval(mult, off, add)
    assert r["pass"] and r["trend_on"] == 0.0 and r["trend_add"] < G11_TREND_MIN, ("G11 pure gain", r)
    r = g11_eval(grow, off, add)
    assert r["pass"] and r["trend_on"] > 0.0, ("G11 growing gain", r)
    r = g11_eval(add, off, add)
    assert not r["pass"] and not r["parts"]["G11c_effect_scales_with_drive"], ("G11 FAILS additive", r)
    r = g11_eval(add_pl, off_pl, add_pl)
    assert not r["pass"] and r["trend_on"] < G11_TREND_MIN, ("G11 FAILS additive on power law", r)
    r = g11_eval(mult_pl, off_pl, add_pl)
    assert r["pass"], ("G11 pure gain on power law", r)
    # an additive control that does NOT fail G11c makes the seed UNDEFINED (never a pass)
    r = g11_eval(mult, off, mult)
    assert not r["pass"] and not r["instrument_valid_additive_control_fails"], ("G11 instrument validity", r)
    # an additive control with no effect is also not a valid instrument
    r = g11_eval(mult, off, off)
    assert not r["pass"] and not r["instrument_valid_additive_control_fails"], ("G11 inert control", r)
    # an offset with the edge closed fails G11a
    shifted = dict(mult)
    shifted[0.0] = 0.5
    r = g11_eval(shifted, off, add)
    assert not r["pass"] and not r["parts"]["G11a_no_offset"], ("G11a offset", r)
    # a COLLAPSING reference must not manufacture gain: no effect on the rising limb, a big ratio only where the
    # reference collapses -> the limb excludes the collapse and G11b fails
    coll = {0.0: 0.0, 0.6: 0.0, 0.7: 0.1, 0.8: 1.0, 0.9: 2.0, 1.0: 3.0, 1.1: 1.5, 1.2: 0.8}
    fake = {0.0: 0.0, 0.6: 0.0, 0.7: 0.1, 0.8: 1.0, 0.9: 2.0, 1.0: 3.0, 1.1: 6.0, 1.2: 3.2}
    r = g11_eval(fake, coll, {g: v + 0.4 * (g > 0) for g, v in coll.items()})
    assert r["rising_limb"] == [0.8, 0.9, 1.0] and not r["pass"], ("G11 collapsing reference", r)
    # too few defined points -> UNDEFINED
    sparse = curve(lambda g: 0.1 if g < 1.1 else 3.0)
    r = g11_eval(mult, sparse, add)
    assert not r["pass"] and "undefined" in r, ("G11 undefined", r)
    # G1 (v3's, unchanged): monotone passes; a supra-floor reversal fails; None / sub-range fail
    mono = [6.5, 4.6, 3.2, 1.6, 0.7, 0.37, 0.22, 0.15, 0.09, 0.08, 0.06]
    assert gate_g1(spearman(list(EVIDENCE_GRID), mono), max(mono) - min(mono)), "G1: monotone must pass"
    reversal = [5.2, 3.9, 2.7, 1.1, 0.6, 0.13, 0.07, 0.08, 0.05, 0.09, 1.10]
    assert not gate_g1(spearman(list(EVIDENCE_GRID), reversal), 5.1), "G1: a supra-floor reversal must fail"
    assert not gate_g1(None, 5.0) and not gate_g1(-0.99, 0.5), "G1: None / sub-range must fail"
    # the SECONDARY floored rho ties sub-floor levels (reported only)
    tail = [3.2, 1.0, 0.5, 0.42, 0.07, 0.06, 0.05, 0.02, 0.07, 0.10, 0.10]
    assert spearman(list(EVIDENCE_GRID), tail) > floored_rho(tail), "floored rho must discount the sub-floor tail"
    # G8 / G4 / G10 (v3 logic, unchanged)
    assert not gate_g8_relay_lesion(None) and not gate_g8_relay_lesion(-0.9) and gate_g8_relay_lesion(0.9)
    assert gate_g4_joint_lesion(None) and not gate_g4_joint_lesion(-0.95) and gate_g4_joint_lesion(-0.2)
    assert not gate_g10_lc_graded(None, 5.0) and not gate_g10_lc_graded(-0.9, 0.1)
    assert gate_g10_lc_graded(-0.9, 1.0) and not gate_g10_lc_graded(0.1, 1.0)
    # G12: concentrated bursts pass; spread (tonic-like) firing fails; a silent lc_ne never passes
    ok = {"concentration": 0.99, "spikes_total": 100, "active_reps_at_most_uncertain": 5}
    assert gate_g12_phasic(ok, {"spikes_total": 300}), "G12: burst + load-bearing pause must pass"
    assert not gate_g12_phasic(dict(ok, concentration=0.9), {"spikes_total": 300}), "G12a: spread spikes must fail"
    assert not gate_g12_phasic(ok, {"spikes_total": 120}), "G12b: a pause the drive alone makes must fail"
    assert not gate_g12_phasic(dict(ok, concentration=None, spikes_total=0), {"spikes_total": 0}), "G12: silent"
    assert not gate_g12_phasic(dict(ok, active_reps_at_most_uncertain=0), {"spikes_total": 300}), "G12c"
    ras = np.zeros((2, 40, 2), bool)
    ras[0, [2, 4], 0] = True          # neuron 0: a 2-spike burst -> both inside the window
    ras[0, [5, 35], 1] = True         # neuron 1: 1 spike in the window, 1 spike 30 steps later (outside)
    fr = burst_stats(ras)
    assert (fr["spikes_total"], fr["spikes_in_burst"], fr["active_neuron_reps"], fr["active_reps"]) == (4, 3, 2, 1), fr
    print("[selftest] gate logic OK in both directions (additive control FAILS G11 on linear and power-law curves; "
          "pure and growing gains PASS; offset / undefined / invalid-instrument cases never pass)")
    return 0


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
#  decision + combiner (PREREG §5)
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
MECHANISM = ("metacog margin-comparator -> [frozen point-edge onto curiosity's ASK pool + a PHASIC spiking lc_ne "
             "(alpha2 GIRK autoinhibition) whose burst silences ask_fb, the relay carrying ASK's output-proportional "
             "slow (GIRK) negative feedback] -- a response-gain change on the edge's drive, not an added current")


def operating_point() -> dict:
    return {"LC_N": LC_N, "W_CMP_LC": W_CMP_LC, "W_LC_AUTO": W_LC_AUTO, "N_FB": N_FB, "W_ASK_FB": W_ASK_FB,
            "W_FB_ASK": W_FB_ASK, "W_LC_FB": W_LC_FB, "ADD_N": ADD_N, "W_CMP_ADD": W_CMP_ADD, "W_ADD_ASK": W_ADD_ASK,
            "G1_SILENT_HZ": G1_SILENT_HZ, "G3_GAIN_ATTRIB_MIN": G3_GAIN_ATTRIB_MIN,
            "LC_GRADED_RHO_MAX": LC_GRADED_RHO_MAX, "LC_GRADED_MIN_RANGE_HZ": LC_GRADED_MIN_RANGE_HZ,
            "BURST_WIN_STEPS": BURST_WIN_STEPS, "PHASIC_CONC_MIN": PHASIC_CONC_MIN,
            "PAUSE_RATIO_MIN": PAUSE_RATIO_MIN, "EV_G11": EV_G11,
            "G11_GRID": list(G11_GRID), "G11_OFF_FLOOR_HZ": G11_OFF_FLOOR_HZ, "G11_MIN_DEFINED": G11_MIN_DEFINED,
            "G11_OFFSET_MAX_HZ": G11_OFFSET_MAX_HZ, "G11_GAIN_MIN": G11_GAIN_MIN, "G11_TREND_MIN": G11_TREND_MIN,
            "G11_RATIO_RES": G11_RATIO_RES}


def _decide(rows) -> dict:
    from tools.verdict import Verdict
    seeds = {r["seed"] for r in rows}
    evaluation = seeds == REQUIRED_SEED_SET
    v = Verdict("metacog comparator -> point-edge + phasic lc_ne feedback-withdrawal gain -> curiosity ASK "
                "(per-seed pre-registered gates)")
    for r in rows:
        s, integ = r["seed"], r["integrity"]
        bal = [l["balance"] for l in r["arms"]["combined_intact"]["levels"]]
        v.require(f"seed{s} metacog balance varies across the evidence grid", (max(bal) - min(bal)) > 0.0)
        v.require(f"seed{s} byte-off: base connectivity identical minus exactly the declared synapses",
                  bool(integ["byte_off"]["PASS"]))
        v.require(f"seed{s} GIRK routing is exactly the declared set (engine stale-mask workaround held)",
                  bool(integ["gabab_routing"]["PASS"]))
        v.require(f"seed{s} metacog read EXACTLY equals the conflict_xedge base pool's",
                  bool(integ["metacog_vs_base_pool"]["PASS"]))
        v.require(f"seed{s} metacog unchanged EXACT across every lesion arm (G5)",
                  bool(r["checks_integrity"]["G5_metacog_unchanged_EXACT_across_arms"]))
        v.require(f"seed{s} lesion restore exact", bool(integ["restore_exact"]))
        v.require(f"seed{s} every lesion verified to hold at measurement",
                  all(integ["lesions_held_at_measurement"].values()))
        v.require(f"seed{s} lc_ne acts on ASK only through the feedback loop", bool(integ["lc_acts_only_through_feedback_loop"]))
        v.require(f"seed{s} curiosity's non-ASK regions silent in every read (GIRK flag inert there)",
                  integ["bystander_spikes_all_arms"] == 0)
        v.require(f"seed{s} no host novelty scalar / neuromodulator subsystem",
                  bool(integ["no_host_novelty_signal"] and not integ["neuromodulator_subsystem_enabled"]))
    v.disabled("STDP / Hebbian / homeostasis / OU / conductance noise",
               "the production metacog pool config; every v4 synapse is fixed-weight by design")
    n_go = sum(1 for r in rows if r["go"])
    decided = v.decide(bool(n_go == len(rows)))
    return {
        "mechanism": MECHANISM,
        "prereg": "docs/plans/2026-09-24-curiosity-lcne-phasic-gain-PREREG.md",
        "builds_on": "docs/plans/2026-09-23-curiosity-metacog-neuromod-gain-PREREG.md",
        "evaluation_set": bool(evaluation),
        "verdict": decided["status"] if evaluation else f"DEV-SMOKE ({decided['status']} on dev seeds; not the "
                                                        f"pre-registered verdict)",
        "preconditions": decided["preconditions"], "undefined_reasons": decided["undefined_reasons"],
        "disabled_processes": decided["disabled_processes"],
        "GO": bool(decided["go"]) if evaluation else None, "n_go": n_go, "n_seeds": len(rows),
        "operating_point": operating_point(), "per_seed": rows,
    }


RUNNER_REL = "research/runners/_curiosity_lcne_phasic_gain_derisk.py"
CIRCUIT_CONSTANTS = ("LC_N", "W_CMP_LC", "W_LC_AUTO", "N_FB", "W_ASK_FB", "W_FB_ASK", "W_LC_FB", "ADD_N", "W_CMP_ADD",
                     "W_ADD_ASK")   # the only names --set may override (never a gate threshold)
_OVERRIDES: list = []     # dev-only --set NAME=VALUE overrides, forwarded to the fresh-process determinism check


def runner_code_mismatch(git_shas):
    """None if every input ran the SAME version of this runner file (blob compare at each input's commit)."""
    blobs = {}
    for fp, sha in git_shas.items():
        if not sha or sha == "unknown":
            return f"{fp} has no provenance git_sha (sidecar missing or unknown)"
        r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{sha}:{RUNNER_REL}"], capture_output=True,
                           text=True, cwd=str(_REPO))
        if r.returncode != 0:
            return f"cannot resolve {RUNNER_REL} at {sha} (for {fp})"
        blobs[fp] = r.stdout.strip()
    if len(set(blobs.values())) > 1:
        return f"inputs ran DIFFERENT runner code: {blobs} (shas {git_shas})"
    return None


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
    why = runner_code_mismatch(shas)
    if why:
        print(f"[combine] REFUSED: {why}", flush=True)
        return 2
    rows.sort(key=lambda r: r["seed"])
    summary = _decide(rows)
    summary["config"] = {"combined_from": list(paths)}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(summary, indent=1, default=str))
    print(f"[lcne phasic gain] COMBINED VERDICT: {summary['verdict']} ({summary['n_go']}/{summary['n_seeds']}) "
          f"<- {list(paths)} -> {out}", flush=True)
    return 0 if summary["GO"] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[7, 8])
    ap.add_argument("--dev", action="store_true", help="allow dev seeds (7/8/9/10); the output is labeled DEV-SMOKE")
    ap.add_argument("--no-determinism", action="store_true")
    ap.add_argument("--digest-only", action="store_true", help="internal: print the combined-intact digest and exit")
    ap.add_argument("--selftest", action="store_true", help="gate-logic selftest, no simulation")
    ap.add_argument("--combine", nargs="+", default=None, help="PREREG §5: the ONLY authoritative 6-seed verdict")
    ap.add_argument("--set", nargs="+", default=[], metavar="NAME=VALUE",
                    help="DEV ONLY (refused on evaluation seeds): override an operating-point constant, e.g. "
                         "W_ASK_FB=25, so the dev-seed calibration grid is reproducible from this file; the "
                         "artifact records the resulting operating_point and --combine refuses mixed points")
    ap.add_argument("--out", default=str(_REPO / "research" / "findings" / "raw" / "_curiosity_lcne_phasic_gain.json"))
    a = ap.parse_args()
    if a.set:
        if not a.dev or any(s not in DEV_SEEDS for s in a.seeds):
            print("[lcne phasic gain] REFUSED: --set is for dev-seed calibration only", flush=True)
            return 2
        for kv in a.set:
            k, v = kv.split("=", 1)
            if k not in CIRCUIT_CONSTANTS:
                print(f"[lcne phasic gain] REFUSED: {k} is not an overridable circuit constant", flush=True)
                return 2
            globals()[k] = type(globals()[k])(float(v)) if not isinstance(globals()[k], int) else int(float(v))
            _OVERRIDES.append(kv)
    if a.selftest:
        return _selftest_gate_logic()
    if a.digest_only:
        pool, org, rec = _new_session(a.seeds[0])
        print("DIGEST", digest(_combined_intact(pool, org, rec)), flush=True)
        return 0
    if a.combine:
        return _combine(a.combine, a.out)
    dev = [s for s in a.seeds if s in DEV_SEEDS]
    bad = [s for s in a.seeds if s not in DEV_SEEDS and s not in REQUIRED_SEED_SET]
    if bad or (dev and not a.dev) or (dev and len(dev) != len(a.seeds)):
        print(f"[lcne phasic gain] REFUSED: seeds {a.seeds} -- dev seeds {sorted(DEV_SEEDS)} need --dev and must not "
              f"mix with evaluation seeds {sorted(REQUIRED_SEED_SET)}; other seeds are not pre-registered", flush=True)
        return 2
    t0 = time.time()
    print(f"[lcne phasic gain] seeds={a.seeds} backend={os.environ.get('SIM_BACKEND')} op={operating_point()}",
          flush=True)
    rows = [run_seed(s, determinism=not a.no_determinism) for s in a.seeds]
    summary = _decide(rows)
    summary["config"] = {"seeds": a.seeds, "dev": a.dev, "backend": os.environ.get("SIM_BACKEND")}
    summary["elapsed_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1, default=str))
    print(f"[lcne phasic gain] {summary['verdict']} ({summary['n_go']}/{summary['n_seeds']} seeds pass every "
          f"required gate) -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
