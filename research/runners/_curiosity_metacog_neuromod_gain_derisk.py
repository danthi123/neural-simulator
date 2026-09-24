"""A broadly-projecting neuromodulatory GAIN on curiosity's ASK pool, driven by metacog's own comparator spike
rate -- the COMPANION PROCESS the point-to-point CrossEdge rung never modeled.

Pre-registration: `docs/plans/2026-09-23-curiosity-metacog-neuromod-gain-PREREG.md` (read it first; the gates
below are copied from it and were frozen before the 6-seed run).

REVIEW HISTORY (v1 -> v2, this file). v1 (commit 7ba619532) built the "gain" as a HOST relay: a `GainRecorder`
read the comparator's firing-fraction each step and wrote it to `core_config.current_novelty_signal`, which the
ALREADY-REGISTERED `curiosity` neuromodulator (an ODE over that host scalar) then turned into current on `ask`.
Adversarial review (v2:fcc2f77) correctly flagged this as a host rate-relay wearing a spiking costume: the
comparator->modulator step was `sum(firing)/len(firing)`, host Python, not a synapse. **v2 (this file) replaces
that relay with an ACTUAL spiking population**: metacog's comparator projects, through real synapses, onto a new
locus-coeruleus-analog population (`lc_ne`), which projects, through real synapses, diffusely onto `ask`. No
host code reads a spike raster and writes a modulator input anywhere in this mechanism any more.

WHY THIS RUNG EXISTS (and what it replaces). `_curiosity_metacog_conflict_xedge_derisk.py` wired metacog's margin
comparator into curiosity's ASK pool through ONE declared point-to-point CrossEdge (fixed weight 4.0). Its 6-seed
verdict is NO-GO (`research/findings/2026-09-23-cpu-lane-harvest-curiosity-metacog-conflict-xedge-6seed-no-go.md`):
4/6 seeds pass, and even on a passing seed the synaptic drive alone sits 3x-14x BELOW production's own 19-24 Hz
curious threshold (its S1, secondary). The review correction on that lane named the next lever: "the missing
companion drive / neuromodulatory gain that the real system runs" -- not another retune of the fixed-weight edge.

THE WALL QUESTION (CLAUDE.md standing rule): what does the real system run ALONGSIDE a single excitatory synapse
that the prior rung replaced with a constant? Externally grounded (recorded
`research/queue/.external_searches.jsonl`, lane=curiosity, 2026-09-23): Aston-Jones & Cohen (2005), Annu Rev
Neurosci 28:403-450 -- the locus-coeruleus norepinephrine system's tonic mode broadcasts a GLOBAL population
excitability GAIN (driven by cortical conflict/utility monitoring, e.g. ACC/OFC), not a point-to-point
glutamatergic increment.

THE CIRCUIT (v2; every step is neurons + synapses ONLY -- host code drives metacog's INPUT, exactly as
production does, and nothing else):
  - The metacog comparator (`meta_schema`, UNCHANGED from the conflict_xedge rung) computes its own margin
    signal, exactly as before.
  - A NEW small population, `lc_ne` (`LC_N` neurons, excitatory-typed, no internal recurrence -- the LC nucleus
    is small and does not need internal dynamics for this rung), receives a FIXED-WEIGHT, uniform, dense
    CrossEdge from `meta_schema` (`x_metacog_meta_to_lc_ne`, weight `CMP_TO_LC_W`). `meta_schema` -- the
    comparator's excitatory principal cells, not its inhibitory relay `meta_margin_fs` -- is the source: this is
    also the more anatomically apt choice (LC's cortical afferents are glutamatergic projection neurons, not
    local interneurons; Aston-Jones & Cohen 2005 Fig 1).
  - `lc_ne` projects, through a SECOND fixed-weight, uniform, dense CrossEdge, DIFFUSELY onto curiosity's `ask`
    pool (`x_lc_ne_to_curiosity_ask`, weight `LC_TO_ASK_W`) -- every `lc_ne` neuron synapses onto every `ask`
    neuron. This CrossEdge carries a `transmission_gate` ("lc_ne_gain") used ONLY as this runner's lesion switch
    (`bridge.set_transmission_gate`, `sim/bridge.py:5281` -- a runtime multiplicative scalar on this ONE
    pathway's current, 0.0=closed/1.0=open); it is never driven continuously from a host readout.
  - The FROZEN point-to-point edge from the conflict_xedge rung (`x_metacog_meta_to_curiosity_ask`, weight 4.0)
    stays wired, unchanged, alongside the new `lc_ne` pathway -- both are independently LESIONABLE (each can be
    switched off on its own).

HONESTY: lc_ne IS A SUB-THRESHOLD MODULATOR OF THE EDGE, NOT A SECOND INDEPENDENT DRIVER (fix round 2, per
adversarial re-review). An earlier version of this docstring, and of the PREREG, described the edge and `lc_ne` as
"two independently lesion-attributable pathways" -- language that implies each one drives `ask` on its own. The
data do not support that: the `edge_lesion` arm (point-edge OFF, `lc_ne` pathway intact) reads 0.0 Hz at every one
of the 11 evidence levels, on every seed measured so far (`attrib_edge`=1.0). `lc_ne` alone drives NOTHING; its
only measured effect is on the COMBINED arm's dynamic range on top of the edge's own drive (G3's floor). Both are
independently LESIONABLE (each switch can be thrown on its own, which is what G3/G4/G8 exploit), but only ONE of
them (the edge) is independently a DRIVER. `lc_ne` is closer in kind to a gain on the edge's response than to a
parallel excitatory pathway -- consistent with the ADDITIVE-vs-MULTIPLICATIVE honesty note below, and worth saying
plainly rather than leaving it to be inferred from the edge_lesion numbers.

HONESTY: ADDITIVE, NOT MULTIPLICATIVE (the residual this rung does NOT close). Aston-Jones & Cohen's LC-NE gain
is MULTIPLICATIVE: it rescales a neuron's RESPONSIVENESS to its other inputs, not just its baseline current. This
substrate's population-to-population synapses (the ONLY mechanism available for a population-to-population
broadcast) deliver ADDITIVE per-spike current, identical in kind to the frozen point-edge -- `lc_ne`'s "diffuse
projection" is architecturally a broadcast (few-to-many) rather than a point edge (one-region-to-one-region), but
it is NOT a gain on `ask`'s excitability to ITS OTHER inputs. Two substrate mechanisms come closer to a true gain
and were considered and rejected for this rung, both honestly:
  (1) `sim/neuromodulators.py`'s `synaptic_gain` target IS multiplicative (`compute_synaptic_gain_multiplier`:
      "Multiplies effective_synaptic_strength"), but `compute_synaptic_gain_multiplier` only supports
      `scope="all"` -- there is no `scope="group:ask"` path. Using `scope="all"` would multiply EVERY synapse in
      the pool, including metacog's own comparator synapses, which G5 requires to stay EXACTLY unchanged; using
      it would confound the very invariance this rung's gates depend on.
  (2) `set_transmission_gate` / `cp_transmission_gain` (`sim/bridge.py:5281`) IS a multiplicative scalar, but it
      scales only the CURRENT of the ONE declared pathway it is attached to (a "volume knob" on `lc_ne -> ask`
      itself) -- it does not rescale `ask`'s response to its OTHER inputs (the frozen point-edge, or any future
      input), so it does not implement gain in the LC-NE sense either. It is used here only as a static lesion
      switch (the role the framework itself declares for it: "the lesion handle for a FIXED (plastic=False)
      neuromodulatory projection"), never as this rung's continuous drive.
The honest next rung, named and not built here: a per-group (not global) `synaptic_gain`-style multiplicative
target on the neuromodulator subsystem, OR a per-neuron background-conductance/threshold shift driven by `lc_ne`
firing rate through the SAME `couple_gate_to_pool`-style in-substrate coupling `sim/bridge.py:5307` already uses
for thalamocortical gating (which would need a continuous, not threshold-open/close, coupling law it does not
yet have).

WHAT IS HOST-DESIGNED (declared, not hidden): `CMP_TO_LC_W` and `LC_TO_ASK_W` are hand-set, calibrated on the
seed-42 smoke only (the 5 other seeds are held out). `LC_N` (population size) is hand-set. Neither the comparator,
the point-edge, nor either new CrossEdge is Hebbian-grown. `lc_ne`'s own internal dynamics (a plain LIF/AdEx
population with `internal_density=0.0`) are NOT a tonic/phasic biophysical LC model -- this rung tests the
POPULATION-RELAY structure (comparator -> a dedicated broadcasting nucleus -> diffuse target), not LC's
intrinsic bistable tonic/phasic firing modes (those are the honest next rung after the gain-vs-additive one).

FUNCTIONAL CORRELATE ONLY -- no phenomenal claim. Additive research runner: no `sim/` edit, no production flag,
no default flip; nothing in the live chat path imports this file.

Run:
  SIM_BACKEND=numpy python -m research.runners._curiosity_metacog_neuromod_gain_derisk --smoke
  SIM_BACKEND=numpy python -m research.runners._curiosity_metacog_neuromod_gain_derisk \
      --seeds 42 43 44 100 101 102 --out research/findings/raw/_curiosity_metacog_neuromod_gain_6seed.json
  python -m research.runners._curiosity_metacog_neuromod_gain_derisk --selftest   # gate-logic selftest, no sim
  python -m research.runners._curiosity_metacog_neuromod_gain_derisk --combine <f1.json> <f2.json> ... --out <out>
      # COMBINER (declared, PREREG Sec 5): the pre-registered 6/6 verdict is split across separate pool-staged
      # artifacts (one per batch). This mode loads each file's `per_seed` rows, unions them (asserting the seed
      # SET is exactly {42,43,44,100,101,102} with no duplicate/missing seed), and re-applies the SAME
      # Verdict/require logic `main()` uses for a single-process 6-seed run, so a split run and a monolithic run
      # would decide identically on the same per-seed data. This is the ONLY authoritative combined verdict.
      # Fix round 2: ALSO refuses if the input files disagree on `mechanism`/`operating_point`/git SHA (the
      # combiner hazard the review flagged -- a stale artifact from a superseded mechanism, or a constant change
      # between batches, could otherwise combine silently instead of crashing on a missing key).

G10 (fix round 2): `lc_ne`'s own per-evidence-level firing rate is now recorded (`RecorderWithLC`,
`coupled_sweep_lc`) and gated -- `CMP_TO_LC_W=30.0` (v2's original calibration, measured only at evidence=1.0)
turned out to SATURATE `lc_ne` (measured: rho(evidence, lc_ne Hz)=+0.93, i.e. `lc_ne` slightly INCREASES with
evidence and sits at 40-44 Hz at every level -- a near-ceiling response, not a graded one). `CMP_TO_LC_W` is
RECALIBRATED in this fix round; see the constant's own comment and the PREREG's v3 amendment for the measured
before/after.
"""
from __future__ import annotations

import os

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import json  # noqa: E402
import numpy as np  # noqa: E402

from sim.backend import to_host  # noqa: E402
from sim.regions import BrainRegion  # noqa: E402
from research.runners.onebrain_merge_framework import REGISTRY, CrossEdge, OrganDescriptor, merge_organs  # noqa: E402
from research.runners.onebrain_crossedge_gate import (  # noqa: E402
    cross_edge_masks, lesion_cross_edges, verify_byte_off,
)
from research.runners.metacog_production_organ import MetacogProductionOrgan  # noqa: E402
from research.runners._curiosity_metacog_conflict_xedge_derisk import (  # noqa: E402
    build_pool as build_base_pool, Recorder, coupled_sweep, level_rho, perm_null, _range, _digest, _meta_exact,
    _curiosity_production_threshold, _metacog_het, METACOG_MARGIN, CMP_REGIONS, XEDGE, XEDGE_KEY,
    G1_RHO_MAX, G1_MIN_RANGE_HZ, G7_RHO_MAX, G8_RHO_MIN,
    EVIDENCE_GRID, READ_REPS, STEPS_PER_REP, spearman, _swapped_idx,
)
from research.runners.metacog_production_organ import nmda_norm_margin  # noqa: E402

# ── FROZEN operating point for THIS rung (seed-42-only calibration, 2026-09-23 v1; RE-CALIBRATED 2026-09-23 v2
#    after the mechanism was rebuilt as spiking synapses -- the v1 constants (CMP_RATE_NORM, GAIN_EXCIT_
#    SENSITIVITY) governed a host modulator ODE that no longer exists and do not transfer) ─────────────────────
LC_N = 20                  # locus-coeruleus-analog population size (small nucleus, no internal recurrence)
CMP_TO_LC_W = 5.0          # meta_schema -> lc_ne, uniform dense CrossEdge weight. RECALIBRATED (fix round 2,
                           # G10): the ORIGINAL v2 value, 30.0, was chosen from a single evidence=1.0 probe (lc_ne
                           # fires 0 times at weight 3.0, ~940 times at weight 30.0) and turned out to SATURATE
                           # lc_ne across the full 11-level grid -- measured 40-44 Hz at EVERY level, rho(evidence,
                           # lc_ne Hz)=+0.93 (slightly INCREASING with evidence, the wrong direction, not flat-by-
                           # chance). A seed-42-only scan of CMP_TO_LC_W in {5,5.5,6,8,10} found the graded regime
                           # is narrow: 5.0 gives rho_lc=-0.963 (peak 7.41 Hz, range 2.78 Hz), 5.5 gives -0.766,
                           # and >=8.0 already flips positive (saturated). 5.0 is used because it clears G10 with
                           # the largest margin found in that scan while also clearing G1/G3/G7/G8 (verified by a
                           # full run_seed at this weight: rho=-0.991, attrib_gain=0.246, rho_swap=-0.991,
                           # rho_relay=0.982, rho_lc=-0.963) -- not an exhaustive search for the supremum, in the
                           # same spirit as v2's original coarse LC_TO_ASK_W search.
LC_TO_ASK_W = 1.0          # lc_ne -> ask, uniform dense DIFFUSE CrossEdge weight. UNCHANGED by the fix-round-2
                           # CMP_TO_LC_W recalibration (this weight governs lc_ne's OUTPUT onto ask, not its own
                           # input drive). Original v2 calibration (at the OLD, since-superseded CMP_TO_LC_W=30.0):
                           # LC_TO_ASK_W=20 SATURATES ask into a high-rate ceiling regime where the response
                           # INCREASES with evidence (rho=+0.96, fails G1/G7 -- the same saturation failure mode
                           # v1's GAIN_EXCIT_SENSITIVITY=500 hit); LC_TO_ASK_W>=2 already compresses the combined
                           # dynamic range below the edge-alone range (frac_gain goes negative). 1.0 was the
                           # largest value that kept G1/G7 monotone and cleared G3's floor at CMP_TO_LC_W=30.0
                           # (frac_gain=0.436). Re-verified at the NEW CMP_TO_LC_W=5.0 (fix round 2): G1
                           # (rho=-0.991), G7 (rho_swap=-0.991) still monotone, G3's floor still cleared
                           # (frac_gain=0.246) -- not re-tuned, since it already worked at the new operating point.
G3_GAIN_ATTRIB_MIN = 0.2   # the gain pathway must own >= this fraction of the combined ASK dynamic range (a floor)
LC_GATE_KEY = "lc_ne_gain"          # transmission_gate name: the LESION switch for lc_ne -> ask (NOT a live drive)
CMP_TO_LC_KEY = "x_metacog_meta_to_lc_ne"
LC_TO_ASK_KEY = "x_lc_ne_to_curiosity_ask"

# G10 (fix round 2, PREREG v3 amendment): lc_ne's OWN firing was never recorded in v2, so an evidence-graded LC
# drive could not be distinguished from a saturated tonic bias (CMP_TO_LC_W=30 was chosen ~10x above lc_ne's
# firing onset, measured only at evidence=1.0). Thresholds chosen BEFORE running the full 11-level grid on the
# rebuilt instrument, by loosening G1's own precedent (rho<=-0.8, range>=1.0 Hz) proportionally to lc_ne's role
# (a sub-threshold MODULATOR of the edge, not the primary driver -- see PREREG Sec 1's fix-round-2 correction) and
# its much smaller population (20 vs 80 neurons, more sampling noise per rep):
LC_GRADED_RHO_MAX = -0.3   # Spearman rho(evidence, lc_ne level-mean Hz) must be <= this (a loose bound)
LC_GRADED_MIN_RANGE_HZ = 0.3   # ...and lc_ne's OWN dynamic range across the 11 levels must be >= this (else FLAT)


def _lc_spec(seed):
    return ([BrainRegion(name="lc_ne", n_neurons=LC_N, exc_fraction=1.0, internal_density=0.0,
                         enable_nmda=False)], [], {})


LC_ORGAN = OrganDescriptor(
    key="lc_ne_organ", regions=("lc_ne",), spec_fn=_lc_spec,
    scaffold_residuals=("hand-set meta_schema->lc_ne and lc_ne->ask CrossEdge weights + LC_N (seed-42 "
                        "calibration); not Hebbian-grown; lc_ne has no internal dynamics of its own (a plain "
                        "relay population, not a tonic/phasic LC biophysical model -- see the module honesty "
                        "note on ADDITIVE vs MULTIPLICATIVE gain).",))

CMP_TO_LC = CrossEdge(key=CMP_TO_LC_KEY, source_key="metacog", source_region="meta_schema",
                      target_key="lc_ne_organ", target_region="lc_ne", init_weight=CMP_TO_LC_W,
                      plastic=False, learn_rule="none", freeze_rest=False)

LC_TO_ASK = CrossEdge(key=LC_TO_ASK_KEY, source_key="lc_ne_organ", source_region="lc_ne",
                      target_key="curiosity", target_region="ask", init_weight=LC_TO_ASK_W,
                      plastic=False, learn_rule="none", freeze_rest=False, transmission_gate=LC_GATE_KEY)

ALL_CROSS_EDGES = [XEDGE, CMP_TO_LC, LC_TO_ASK]


def build_combined_pool(seed: int):
    """[metacog(het), curiosity, metacog_margin, lc_ne_organ] + the frozen point-edge + the two new LC CrossEdges.
    Every synapse in this pool is either a base-organ pathway or one of `ALL_CROSS_EDGES` -- there is no
    explicit_wiring_fn and no host-driven state anywhere in the LC pathway."""
    desc = [_metacog_het(), REGISTRY["curiosity"], METACOG_MARGIN, LC_ORGAN]
    pool = merge_organs(desc, seed=int(seed), wire=True, cross_edges=ALL_CROSS_EDGES)
    pool.ensure_built()
    return pool


class RecorderWithLC(Recorder):
    """Fix round 2 (G10): the base `Recorder` (conflict_xedge module) records `ask` and the metacog/comparator
    rasters only -- `lc_ne`'s OWN firing was never recorded, so an evidence-graded LC drive could not be told apart
    from a saturated tonic bias by looking at the artifact. This subclass ALSO tracks `lc_ne`'s per-step spike
    count, identically to how `ask` is tracked, WITHOUT touching the shared base module (which the conflict_xedge
    lane's own closed-out runner still imports `Recorder` from)."""

    def __init__(self, pool):
        super().__init__(pool)
        rm = pool.bridge.region_manager
        self.lc_idx = np.asarray(rm.indices("lc_ne"), np.int64)

    def reset(self):
        super().reset()
        self.lc_counts = []

    def _step(self):
        self._orig()
        if not self.on:
            return
        fs = np.asarray(to_host(self.b.cp_firing_states)).astype(bool)
        self.ask_counts.append(int(fs[self.ask].sum()))
        self.lc_counts.append(int(fs[self.lc_idx].sum()))
        self.h_meta.update(np.packbits(fs[self.meta_idx]).tobytes())
        if self.cmp_idx is not None:
            self.h_cmp.update(np.packbits(fs[self.cmp_idx]).tobytes())


def coupled_sweep_lc(pool, org, rec, swap=False) -> dict:
    """Identical to the base module's `coupled_sweep`, EXCEPT it also reads `rec.lc_counts` (only present on a
    `RecorderWithLC`) and adds `lc_ne_hz_per_rep`/`lc_ne_hz` to each level -- so `lc_ne`'s own per-evidence-level
    firing rate lands in the artifact (fix round 2, G10). Kept as a separate function (not a base-module edit) so
    the conflict_xedge lane's own runner, which has no `lc_ne` region, is untouched."""
    org.ensure_built()
    n_ask = rec.ask.size
    n_lc = rec.lc_idx.size
    levels = []
    for ev in EVIDENCE_GRID:
        rec.reset()
        rec.on = True
        if swap:
            bal = float(nmda_norm_margin(org.bridge, org.xp, _swapped_idx(org.idx), org.snap, ev))
            conf = bool(bal >= org.threshold)
        else:
            j = org.judge(ev)
            bal, conf = float(j["balance"]), bool(j["confident"])
        rec.on = False
        c = np.asarray(rec.ask_counts, np.float64)
        lc = np.asarray(rec.lc_counts, np.float64)
        assert c.size == READ_REPS * STEPS_PER_REP, (c.size, READ_REPS, STEPS_PER_REP)
        assert lc.size == c.size, (lc.size, c.size)
        per_rep = c.reshape(READ_REPS, STEPS_PER_REP).sum(1) / n_ask / (STEPS_PER_REP * 1e-3)
        per_rep_lc = lc.reshape(READ_REPS, STEPS_PER_REP).sum(1) / n_lc / (STEPS_PER_REP * 1e-3)
        levels.append({"evidence": ev, "balance": bal, "confident": conf,
                       "ask_hz_per_rep": [float(x) for x in per_rep], "ask_hz": float(per_rep.mean()),
                       "lc_ne_hz_per_rep": [float(x) for x in per_rep_lc], "lc_ne_hz": float(per_rep_lc.mean()),
                       "meta_raster_sha256": rec.h_meta.hexdigest(),
                       "cmp_raster_sha256": (rec.h_cmp.hexdigest() if rec.cmp_idx is not None else None)})
    return {"levels": levels, "threshold": float(org.threshold)}


def lc_level_rho(sweep):
    """Spearman rho(evidence, lc_ne level-mean Hz) -- the G10 statistic. `spearman` returns None (UNDEFINED, not a
    score) when either side has zero variance, e.g. a genuinely flat/saturated lc_ne response."""
    return spearman([l["evidence"] for l in sweep["levels"]], [l["lc_ne_hz"] for l in sweep["levels"]])


def _lc_range(sweep):
    v = [l["lc_ne_hz"] for l in sweep["levels"]]
    return float(max(v) - min(v)), float(max(v))


def _byte_off_check(seed: int) -> dict:
    """INTEGRITY (promised, PREREG Sec 3, NOT run in v1 -- fixed here): base connectivity of the combined pool,
    with the point-edge AND both LC CrossEdges (hence all of `lc_ne`'s synapses, since it has no internal
    connectivity of its own) excluded, must be BYTE-IDENTICAL to the conflict_xedge rung's own `coupled=False`
    pool -- i.e. adding the LC organ + its two edges perturbed NOTHING else."""
    combined = build_combined_pool(seed)
    base = build_base_pool(seed, coupled=False)
    spec = type("ByteOffSpec", (), {"cross_edges": ALL_CROSS_EDGES})()
    return verify_byte_off(combined.bridge, base.bridge, spec)


def run_seed(seed: int, determinism: bool = True, verbose: bool = True) -> dict:
    t0 = time.time()
    pool = build_combined_pool(seed)
    org = MetacogProductionOrgan(seed=seed, shared=pool)
    rec = RecorderWithLC(pool)  # ALSO tracks lc_ne's own per-step firing (G10, fix round 2) -- still no host
                                # relay of any kind; only the sim's own bridge runs

    b, xp = pool.bridge, pool.xp
    masks = cross_edge_masks(b, ALL_CROSS_EDGES)
    n_edge = int(masks[XEDGE_KEY].sum())
    n_cmp_lc = int(masks[CMP_TO_LC_KEY].sum())
    n_lc_ask = int(masks[LC_TO_ASK_KEY].sum())

    def _prime():
        # ANY `cp_connections.data` reassignment (a lesion OR a restore) re-triggers a small transient on the
        # FIRST 1-2 reps of the NEXT sweep (isolated 2026-09-23 on the v1 mechanism by bisection; carried
        # forward here defensively for the SAME class of `cp_connections.data` write -- the point-edge lesion).
        # `set_transmission_gate` does NOT reassign `cp_connections.data` (it writes `cp_transmission_gain`
        # instead, sim/bridge.py:5299), so it is not known to need this, but every gain-lesion arm below still
        # primes after toggling it: cheap insurance, and `restore_exact` below would catch it either way if it
        # did turn out to matter.
        with pool.sequence_isolation():
            coupled_sweep(pool, org, rec)

    # ONE throwaway PRIMING sweep against the freshly-built pool (see _prime's note; the v1 cold-start transient
    # was on the FIRST sweep after ANY connectivity write, including pool construction itself).
    with pool.sequence_isolation():
        coupled_sweep(pool, org, rec)

    with pool.sequence_isolation():
        combined_intact = coupled_sweep_lc(pool, org, rec)   # records lc_ne_hz per level too (G10)
    with pool.sequence_isolation():
        combined_swap = coupled_sweep_lc(pool, org, rec, swap=True)

    # G3: gain-only lesion (edge intact, lc_ne -> ask closed via its transmission_gate).
    b.set_transmission_gate(LC_GATE_KEY, 0.0)
    _prime()
    with pool.sequence_isolation():
        gain_lesion = coupled_sweep(pool, org, rec)
    b.set_transmission_gate(LC_GATE_KEY, 1.0)
    _prime()

    # edge-only lesion (gain intact, point-edge off) -- kept for G5's exact-metacog check and reported.
    before_edge = lesion_cross_edges(b, {XEDGE_KEY: masks[XEDGE_KEY]}, xp)
    _prime()
    with pool.sequence_isolation():
        edge_lesion = coupled_sweep(pool, org, rec)
    b.cp_connections.data = xp.asarray(before_edge, dtype=b.cp_connections.data.dtype)
    _prime()

    # G4 (INTEGRITY, not gating -- see the reclassification note below): BOTH lesioned.
    b.set_transmission_gate(LC_GATE_KEY, 0.0)
    before_edge2 = lesion_cross_edges(b, {XEDGE_KEY: masks[XEDGE_KEY]}, xp)
    _prime()
    with pool.sequence_isolation():
        both_lesion = coupled_sweep(pool, org, rec)
    b.cp_connections.data = xp.asarray(before_edge2, dtype=b.cp_connections.data.dtype)
    b.set_transmission_gate(LC_GATE_KEY, 1.0)
    _prime()

    # G8: comparator-relay lesion (meta_margin_fs -> meta_schema inhibition), edge+gain intact.
    rm = b.region_manager
    coo = b.cp_connections.tocoo()
    row = np.asarray(to_host(coo.row)); col = np.asarray(to_host(coo.col))
    relay_mask = np.isin(row, np.asarray(rm.indices("meta_margin_fs"))) & np.isin(col, np.asarray(rm.indices("meta_schema")))
    data = np.asarray(to_host(b.cp_connections.data)).copy()
    data2 = data.copy()
    data2[relay_mask] = 0.0
    b.cp_connections.data = xp.asarray(data2, dtype=b.cp_connections.data.dtype)
    _prime()
    with pool.sequence_isolation():
        relay_lesion = coupled_sweep(pool, org, rec)
    b.cp_connections.data = xp.asarray(data, dtype=b.cp_connections.data.dtype)
    _prime()

    with pool.sequence_isolation():
        restored = coupled_sweep(pool, org, rec)  # integrity: restoration is exact

    # ── statistics ──
    rho = level_rho(combined_intact)
    rng_c, peak_c = _range(combined_intact)
    rng_g, peak_g = _range(gain_lesion)
    rng_e, peak_e = _range(edge_lesion)
    rng_b, peak_b = _range(both_lesion)
    rho_swap = level_rho(combined_swap)
    rho_relay = level_rho(relay_lesion)
    rho_both = level_rho(both_lesion)
    g9 = perm_null(combined_intact, seed)

    # G10 (fix round 2): lc_ne's OWN per-level firing rate, and whether it is evidence-graded or a saturated
    # tonic bias. Measured on the combined-intact arm (the only one where the LC pathway is both driven AND
    # allowed to reach ask; `coupled_sweep_lc` is what fills in `lc_ne_hz`/`lc_ne_hz_per_rep` per level).
    rho_lc = lc_level_rho(combined_intact)
    rng_lc, peak_lc = _lc_range(combined_intact)
    g10 = gate_g10_lc_evidence_graded(rho_lc, rng_lc)

    from tools.lab import attributable_to
    attrib_gain = attributable_to(f"seed{seed} ASK range = the lc_ne gain pathway", rng_c, rng_g)
    attrib_edge = attributable_to(f"seed{seed} ASK range = the point edge", rng_c, rng_e)

    m_gain = _meta_exact(combined_intact, gain_lesion)
    m_edge = _meta_exact(combined_intact, edge_lesion)
    m_both = _meta_exact(combined_intact, both_lesion)
    cmp_equal = ([l["cmp_raster_sha256"] for l in combined_intact["levels"]]
                 == [l["cmp_raster_sha256"] for l in gain_lesion["levels"]]
                 == [l["cmp_raster_sha256"] for l in edge_lesion["levels"]]
                 == [l["cmp_raster_sha256"] for l in both_lesion["levels"]])

    digest = _digest(combined_intact)
    restored_ok = _digest(restored) == digest

    det = {"checked": False}
    if determinism:
        cmd = [sys.executable, "-m", "research.runners._curiosity_metacog_neuromod_gain_derisk",
               "--digest-only", "--seeds", str(seed)]
        out = subprocess.run(cmd, cwd=str(_REPO), capture_output=True, text=True, env=dict(os.environ))
        line = [x for x in out.stdout.splitlines() if x.startswith("DIGEST ")]
        other = line[-1].split()[-1] if line else None
        det = {"checked": True, "digest_main": digest, "digest_fresh_process": other,
               "equal": bool(other == digest), "rc": out.returncode}

    undefined = rho is None or rng_c < G1_MIN_RANGE_HZ

    # S1 (secondary, NOT gating -- amended before ANY 6-seed run in v1, carried forward unchanged in v2): does
    # the combined mechanism reach + separate at the seed's OWN production threshold?
    cal = _curiosity_production_threshold(seed)
    thr = cal.get("threshold_hz")
    uncertain = [l for l in combined_intact["levels"] if not l["confident"]]
    confident = [l for l in combined_intact["levels"] if l["confident"]]
    s1_undefined = thr is None or not uncertain or not confident
    s1_reaches_threshold = (not s1_undefined) and (max(l["ask_hz"] for l in uncertain) >= thr) and \
        (max(l["ask_hz"] for l in confident) < thr)

    g3 = bool(attrib_gain is not None and attrib_gain >= G3_GAIN_ATTRIB_MIN)
    g4 = gate_g4_joint_lesion(rho_both)
    g5 = bool(m_gain["balance_equal"] and m_gain["metacog_raster_equal"]
              and m_edge["balance_equal"] and m_edge["metacog_raster_equal"]
              and m_both["balance_equal"] and m_both["metacog_raster_equal"] and cmp_equal)
    g8 = gate_g8_relay_lesion(rho_relay)

    # REQUIRED gates (evidence FOR the mechanism): G1/G3/G6/G7/G8.
    # INTEGRITY gates (reported, NOT counted toward GO): G4, G5. Review v2:fcc2f77 flagged both as passing BY
    # CONSTRUCTION on the v1 mechanism (no feedback path from ask/gain back into metacog makes G5 vacuous; the
    # v1 edge_lesion arm was already flat, making both_lesion trivially flat too, so G4 could not have failed).
    # v2's spiking rebuild does not remove that structural fact -- there is STILL no path from ask/gain back into
    # metacog (so G5 remains a pure integrity check by the pool's own topology, not evidence), and whether
    # edge_lesion happens to be flat on v2's freshly-calibrated weights is exactly the kind of load-bearing-
    # magnitude question G3's floor already measures -- G4 is demoted alongside it rather than re-litigated
    # per-seed. Both are still SCORED and printed; neither can block or grant a GO.
    checks_required = {
        "G1_monotone_rho<=-0.8": bool((not undefined) and rho <= G1_RHO_MAX),
        "G3_gain_pathway_load_bearing": g3,
        "G6_determinism_fresh_process_hash": bool(det.get("equal")) if determinism else None,
        "G7_class_swap_monotone": bool(rho_swap is not None and rho_swap <= G7_RHO_MAX),
        "G8_relay_lesion_abolishes_coupling": g8,
        "G10_lc_ne_evidence_graded_not_saturated": g10,
    }
    checks_integrity = {
        "G4_joint_lesion_breaks_coupling": g4,
        "G5_metacog_unchanged_EXACT": g5,
    }
    required = [k for k, v in checks_required.items() if v is not None]
    go = all(checks_required[k] for k in required)

    byte_off = _byte_off_check(seed)

    res = {
        "seed": seed, "go": bool(go), "checks": checks_required, "checks_integrity": checks_integrity,
        "calibration_seed": seed == 42,
        "rho": rho, "rho_swap": rho_swap, "rho_relay_lesion": rho_relay, "rho_both_lesion": rho_both,
        "rho_lc_ne": rho_lc,   # G10 (fix round 2): lc_ne's own evidence-graded-ness, combined-intact arm
        "s1_reaches_production_threshold": bool(s1_reaches_threshold), "s1_undefined": bool(s1_undefined),
        "s1_threshold_hz": thr, "g9_perm_null": g9,
        "ask_range_hz": {"combined": rng_c, "gain_lesion": rng_g, "edge_lesion": rng_e, "both_lesion": rng_b},
        "ask_peak_hz": {"combined": peak_c, "gain_lesion": peak_g, "edge_lesion": peak_e, "both_lesion": peak_b},
        "lc_ne_range_hz": {"combined": rng_lc}, "lc_ne_peak_hz": {"combined": peak_lc},
        "lc_ne_hz_per_level": [{"evidence": l["evidence"], "lc_ne_hz": l["lc_ne_hz"]}
                                for l in combined_intact["levels"]],
        "attributable_frac": {"gain": attrib_gain, "edge": attrib_edge},
        "metacog_exact": {"vs_gain_lesion": m_gain, "vs_edge_lesion": m_edge, "vs_both_lesion": m_both},
        "comparator_raster_equal_across_arms": cmp_equal,
        "determinism": det,
        "integrity": {"n_edge_synapses": n_edge, "n_cmp_to_lc_synapses": n_cmp_lc, "n_lc_to_ask_synapses": n_lc_ask,
                      "restore_exact": restored_ok, "byte_off": byte_off,
                      "no_host_novelty_signal": float(getattr(b.core_config, "current_novelty_signal", 0.0) or 0.0) == 0.0,
                      "neuromodulator_subsystem_enabled": bool(getattr(b.core_config, "enable_neuromodulator_subsystem", False))},
        "production_threshold_calibration": cal,
        "arms": {"combined_intact": combined_intact, "class_swap": combined_swap, "gain_lesion": gain_lesion,
                  "edge_lesion": edge_lesion, "both_lesion": both_lesion, "relay_lesion": relay_lesion},
        "elapsed_s": round(time.time() - t0, 1),
    }
    if verbose:
        print(f"[seed {seed}] rho={rho} swap={rho_swap} relay={rho_relay} both={rho_both} rho_lc={rho_lc} "
              f"ask combined={[round(l['ask_hz'], 2) for l in combined_intact['levels']]} "
              f"lc_ne combined={[round(l['lc_ne_hz'], 2) for l in combined_intact['levels']]} "
              f"thr_hz={thr} attrib_gain={attrib_gain} attrib_edge={attrib_edge} "
              f"det={det.get('equal')} GO={go} ({res['elapsed_s']}s)", flush=True)
        print(f"[seed {seed}] checks={checks_required} integrity={checks_integrity}", flush=True)
    return res


# ── module-level gate predicates (2026-09-23 fix: SHARED by run_seed AND _selftest_gate_logic, per the review's
#    "selftest tests a copy, not the scored code" finding. A regression in either would now be caught by BOTH.) ──
def gate_g8_relay_lesion(rho_relay):
    """A None (uninformative/flat) relay-lesion rho NEVER passes -- the v1 bug the review re-flagged risk of."""
    return bool((rho_relay is not None) and (rho_relay > G8_RHO_MIN))


def gate_g4_joint_lesion(rho_both):
    """INTEGRITY, not required (see run_seed's reclassification note). A None (flat, ASK-silent) both-lesion arm
    IS the expected outcome (no third pathway drives ASK); a DEFINED rho must fail G1's own bar."""
    return bool((rho_both is None) or (rho_both > G1_RHO_MAX))


def gate_g10_lc_evidence_graded(rho_lc, rng_lc):
    """G10 (fix round 2): lc_ne's own firing must be evidence-GRADED, not a saturated/flat tonic bias -- the
    instrument gap the review flagged (CMP_TO_LC_W was calibrated ~10x above lc_ne's firing onset, measured only
    at evidence=1.0; a near-ceiling lc_ne could look like it "carries the comparator signal" while actually acting
    as a roughly-constant excitatory bias whose apparent evidence-dependence comes entirely from ask's own
    threshold nonlinearity). A None rho (zero-variance lc_ne response -- exactly the saturation failure mode this
    gate exists to catch) or a sub-floor range NEVER passes; both are the SAME failure, not two UNDEFINED escapes."""
    if rho_lc is None or rng_lc < LC_GRADED_MIN_RANGE_HZ:
        return False
    return bool(rho_lc <= LC_GRADED_RHO_MAX)


def _selftest_gate_logic():
    """Gate-logic selftest -- NO simulation. Calls the SAME `gate_g8_relay_lesion` / `gate_g4_joint_lesion` /
    `gate_g10_lc_evidence_graded` functions `run_seed` scores with, using the SAME imported constants -- so a
    regression in any function's logic, or an accidental import of the wrong constant, fails THIS selftest, not
    just a private copy of it (the review's flagged gap on the v1 file)."""
    # G8: a None (uninformative) relay-lesion arm must NEVER pass.
    assert gate_g8_relay_lesion(None) is False, "G8 selftest FAILED: a None relay-lesion rho passed"
    assert gate_g8_relay_lesion(-0.9) is False, "G8 selftest FAILED: a strongly negative (still-coupled) rho passed"
    assert gate_g8_relay_lesion(G8_RHO_MIN + 0.1) is True, "G8 selftest FAILED: an abolished rho failed"
    # G4 (integrity): a None (flat, ASK-silent) both-lesion arm IS the expected pass; a still-monotone rho fails.
    assert gate_g4_joint_lesion(None) is True, "G4 selftest FAILED: a flat both-lesion arm (expected) failed"
    assert gate_g4_joint_lesion(G1_RHO_MAX - 0.1) is False, "G4 selftest FAILED: a still-monotone rho passed"
    assert gate_g4_joint_lesion(G1_RHO_MAX + 0.1) is True, "G4 selftest FAILED: a broken-coupling rho failed"
    # deliberately wrong direction (the class of bug this selftest exists to catch) must NOT pass this selftest:
    def _bad_g8(rho_relay):
        return rho_relay is None or rho_relay > G8_RHO_MIN
    bad_none_passes = _bad_g8(None)
    assert bad_none_passes is True and gate_g8_relay_lesion(None) is False, (
        "selftest cannot distinguish the fixed logic from the flagged bug -- selftest itself is broken")
    # G10: a None rho (flat/saturated lc_ne) NEVER passes, regardless of range; a sub-floor range never passes
    # either, even with a strongly negative rho; only a defined, sufficiently negative rho AND a real range pass.
    assert gate_g10_lc_evidence_graded(None, 5.0) is False, "G10 selftest FAILED: a None rho (saturated) passed"
    assert gate_g10_lc_evidence_graded(-0.9, LC_GRADED_MIN_RANGE_HZ / 2) is False, (
        "G10 selftest FAILED: a sub-floor range passed despite a strong rho")
    assert gate_g10_lc_evidence_graded(LC_GRADED_RHO_MAX + 0.1, 5.0) is False, (
        "G10 selftest FAILED: an insufficiently-negative rho passed")
    assert gate_g10_lc_evidence_graded(LC_GRADED_RHO_MAX - 0.1, LC_GRADED_MIN_RANGE_HZ + 0.1) is True, (
        "G10 selftest FAILED: a genuinely evidence-graded lc_ne response failed")
    print("[selftest] gate-logic directions OK (module-level functions, shared with run_seed)")
    return 0


REQUIRED_SEED_SET = frozenset({42, 43, 44, 100, 101, 102})


def _decide(rows) -> dict:
    """The ONE authoritative decision procedure over a list of per-seed result dicts -- called identically by a
    monolithic `main()` run and by `--combine` over N separately-staged artifacts (PREREG Sec 5's combiner)."""
    n_go = sum(1 for r in rows if r["go"])
    held_out = [r for r in rows if not r["calibration_seed"]]
    from tools.verdict import Verdict
    v = Verdict("metacog margin comparator -> CrossEdge + spiking lc_ne gain population -> curiosity ASK "
                "(per-seed pre-registered gates)")
    for r in rows:
        s, integ = r["seed"], r["integrity"]
        bal = [l["balance"] for l in r["arms"]["combined_intact"]["levels"]]
        v.require(f"seed{s} metacog balance varies across the evidence grid", (max(bal) - min(bal)) > 0.0)
        v.require(f"seed{s} declared point-edge wired (>0 synapses)", integ["n_edge_synapses"] > 0)
        v.require(f"seed{s} declared lc_ne pathway wired (>0 synapses both legs)",
                  integ["n_cmp_to_lc_synapses"] > 0 and integ["n_lc_to_ask_synapses"] > 0)
        v.require(f"seed{s} byte-off: base connectivity identical minus the point-edge + lc_ne organ",
                  bool(integ["byte_off"]["PASS"]))
        v.require(f"seed{s} lesion restore exact (combined-intact digest re-reads)", bool(integ["restore_exact"]))
        v.require(f"seed{s} no host novelty scalar / neuromodulator subsystem installed",
                  bool(integ["no_host_novelty_signal"] and not integ["neuromodulator_subsystem_enabled"]))
        v.require(f"seed{s} every intact read ran the full rep window",
                  all(len(l["ask_hz_per_rep"]) > 0 for l in r["arms"]["combined_intact"]["levels"]))
    v.disabled("STDP / Hebbian / homeostasis / OU / conductance noise on the comparator, edge, or lc_ne pathway",
               "the production metacog pool config; the comparator, the point-edge and both new lc_ne CrossEdges "
               "are fixed-weight by design")
    decided = v.decide(bool(n_go == len(rows)))
    return {
        "mechanism": "metacog margin-comparator -> [frozen point-edge (drives ask alone) + a SPIKING lc_ne "
                     "population diffusely projecting onto curiosity's ASK pool as a SUB-THRESHOLD MODULATOR of "
                     "the edge's response, not a second independent driver -- edge_lesion reads 0 Hz at every "
                     "level; both fixed-weight CrossEdges] (every step is neurons + synapses; ADDITIVE current, "
                     "not a multiplicative gain -- see the module honesty note)",
        "prereg": "docs/plans/2026-09-23-curiosity-metacog-neuromod-gain-PREREG.md",
        "builds_on": "docs/plans/2026-09-23-curiosity-metacog-conflict-xedge-PREREG.md",
        "verdict": decided["status"], "preconditions": decided["preconditions"],
        "undefined_reasons": decided["undefined_reasons"], "disabled_processes": decided["disabled_processes"],
        "GO": bool(decided["go"]), "n_go": n_go, "n_seeds": len(rows),
        "held_out_n_go": sum(1 for r in held_out if r["go"]), "held_out_n": len(held_out),
        "operating_point": {"LC_N": LC_N, "CMP_TO_LC_W": CMP_TO_LC_W, "LC_TO_ASK_W": LC_TO_ASK_W,
                            "G3_GAIN_ATTRIB_MIN": G3_GAIN_ATTRIB_MIN},
        "per_seed": rows,
    }


def runner_code_mismatch(git_shas):
    """None if every input ran the SAME version of this runner file, else the reason to refuse. Compares the file's
    blob at each input's commit (a short and a full SHA of one commit, or two commits with an identical runner, agree);
    a missing/unknown SHA or one git cannot resolve refuses."""
    import subprocess
    rel = "research/runners/_curiosity_metacog_neuromod_gain_derisk.py"
    root = str(Path(__file__).resolve().parents[2])
    blobs = {}
    for fp, sha in git_shas.items():
        if not sha or sha == "unknown":
            return f"{fp} has no provenance git_sha (sidecar missing or unknown)"
        r = subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{sha}:{rel}"],
                           capture_output=True, text=True, cwd=root)
        if r.returncode != 0:
            return f"cannot resolve {rel} at {sha} (for {fp})"
        blobs[fp] = r.stdout.strip()
    if len(set(blobs.values())) > 1:
        return f"inputs ran DIFFERENT runner code: {blobs} (shas {git_shas})"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true", help="first seed only")
    ap.add_argument("--no-determinism", action="store_true")
    ap.add_argument("--digest-only", action="store_true", help="internal: print the combined-intact digest and exit")
    ap.add_argument("--selftest", action="store_true", help="gate-logic selftest, no simulation")
    ap.add_argument("--combine", nargs="+", default=None,
                    help="COMBINER (PREREG Sec 5): load per_seed rows from these JSON artifacts, union them, "
                         "and re-decide with the SAME logic as a monolithic run. Requires the union to cover "
                         "exactly REQUIRED_SEED_SET with no duplicates, AND (fix round 2) that every input file "
                         "was produced by the SAME mechanism/operating_point/git_sha -- otherwise a future "
                         "constant change (or a stale v1-mechanism artifact) could combine silently with fresh "
                         "v2/v3 rows.")
    ap.add_argument("--out", default=str(_REPO / "research" / "findings" / "raw" /
                                         "_curiosity_metacog_neuromod_gain.json"))
    a = ap.parse_args()
    if a.selftest:
        return _selftest_gate_logic()
    if a.digest_only:
        s = a.seeds[0]
        pool = build_combined_pool(s)
        org = MetacogProductionOrgan(seed=s, shared=pool)
        rec = Recorder(pool)
        with pool.sequence_isolation():
            coupled_sweep(pool, org, rec)  # priming sweep, discarded (see run_seed)
        with pool.sequence_isolation():
            print("DIGEST", _digest(coupled_sweep(pool, org, rec)), flush=True)
        return 0
    if a.combine:
        rows = []
        seen = {}
        mechanisms, operating_points, git_shas = {}, {}, {}
        for fp in a.combine:
            data = json.loads(Path(fp).read_text())
            mechanisms[fp] = data.get("mechanism")
            operating_points[fp] = data.get("operating_point")
            prov_path = Path(str(fp) + ".prov.json")
            git_shas[fp] = json.loads(prov_path.read_text()).get("git_sha") if prov_path.exists() else None
            for r in data["per_seed"]:
                seen.setdefault(r["seed"], []).append(fp)
                rows.append(r)
        dup = {s: fps for s, fps in seen.items() if len(fps) > 1}
        missing = REQUIRED_SEED_SET - set(seen.keys())
        extra = set(seen.keys()) - REQUIRED_SEED_SET
        if dup or missing or extra:
            print(f"[combine] REFUSED: dup={dup} missing={missing} extra={extra}", flush=True)
            return 2
        # Fix round 2 (combiner hazard): every input must share ONE mechanism, ONE operating_point, and (when the
        # provenance sidecar is present for every input) ONE git SHA -- otherwise a future constant change, or a
        # stale artifact from a superseded mechanism (e.g. v1's host-relay rows), could combine silently with
        # fresh rows instead of crashing on a missing key.
        distinct_mech = {v for v in mechanisms.values() if v is not None}
        distinct_op = {json.dumps(v, sort_keys=True) for v in operating_points.values() if v is not None}
        if len(distinct_mech) > 1:
            print(f"[combine] REFUSED: input files report DIFFERENT mechanisms: {mechanisms}", flush=True)
            return 2
        if len(distinct_op) > 1:
            print(f"[combine] REFUSED: input files report DIFFERENT operating_point constants: "
                  f"{operating_points}", flush=True)
            return 2
        # Compare the RUNNER FILE's content at each input's commit, not SHA strings (2026-09-23 review: a short and a
        # full SHA of the same commit read as 'different', and the held-out seeds run from a later merge commit whose
        # runner is byte-identical). A missing sidecar or an unresolvable SHA refuses -- it can no longer be skipped.
        why = runner_code_mismatch(git_shas)
        if why:
            print(f"[combine] REFUSED: {why}", flush=True)
            return 2
        rows.sort(key=lambda r: r["seed"])
        summary = _decide(rows)
        summary["config"] = {"combined_from": list(a.combine)}
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(summary, indent=1, default=str))
        print("=" * 100, flush=True)
        print(f"[metacog->curiosity lc_ne gain] COMBINED VERDICT: {summary['verdict']} "
              f"({summary['n_go']}/{summary['n_seeds']} seeds; held-out {summary['held_out_n_go']}/"
              f"{summary['held_out_n']}) <- {a.combine} -> {a.out}", flush=True)
        return 0 if summary["GO"] else 1
    seeds = [a.seeds[0]] if a.smoke else a.seeds
    t0 = time.time()
    print(f"[metacog->curiosity lc_ne gain] seeds={seeds} backend={os.environ.get('SIM_BACKEND')} "
          f"LC_N={LC_N} CMP_TO_LC_W={CMP_TO_LC_W} LC_TO_ASK_W={LC_TO_ASK_W}", flush=True)
    rows = [run_seed(s, determinism=not a.no_determinism) for s in seeds]
    summary = _decide(rows)
    summary["config"] = {"seeds": seeds, "smoke": a.smoke, "backend": os.environ.get("SIM_BACKEND")}
    summary["elapsed_s"] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1, default=str))
    print("=" * 100, flush=True)
    print(f"[metacog->curiosity lc_ne gain] VERDICT: {summary['verdict']} ({summary['n_go']}/{summary['n_seeds']} "
          f"seeds; held-out {summary['held_out_n_go']}/{summary['held_out_n']}) -> {a.out}", flush=True)
    return 0 if summary["GO"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
