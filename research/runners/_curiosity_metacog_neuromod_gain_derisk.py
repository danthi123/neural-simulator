"""A broadly-projecting neuromodulatory GAIN on curiosity's ASK pool, driven by metacog's own comparator spike
rate -- the COMPANION PROCESS the point-to-point CrossEdge rung never modeled.

Pre-registration: `docs/plans/2026-09-23-curiosity-metacog-neuromod-gain-PREREG.md` (read it first; the gates
below are copied from it and were frozen before the 6-seed run).

WHY THIS RUNNER EXISTS (and what it adds, not replaces). `_curiosity_metacog_conflict_xedge_derisk.py` wired
metacog's margin comparator into curiosity's ASK pool through ONE declared point-to-point CrossEdge (fixed
weight 4.0). Its 6-seed verdict is NO-GO
(`research/findings/2026-09-23-cpu-lane-harvest-curiosity-metacog-conflict-xedge-6seed-no-go.md`): 4/6 seeds
pass, and even on a passing seed the synaptic drive alone sits 3x-14x BELOW production's own 19-24 Hz curious
threshold (its S1, secondary). The review correction named the next lever: "the missing companion drive /
neuromodulatory gain that the real system runs" -- not another retune of the same fixed-weight edge.

THE WALL QUESTION (CLAUDE.md standing rule): what does the real system run ALONGSIDE a single excitatory
synapse that the prior rung replaced with a constant? Externally grounded (recorded
`research/queue/.external_searches.jsonl`, lane=curiosity, 2026-09-23): Aston-Jones & Cohen (2005), Annu Rev
Neurosci 28:403-450 -- the locus-coeruleus norepinephrine system's tonic mode broadcasts a GLOBAL, MULTIPLICATIVE
population excitability GAIN (driven by cortical conflict/utility monitoring), not a point-to-point glutamatergic
increment. A single fixed-weight CrossEdge has no gain-multiplication analog and no principled relation to the
scale of production's own ASK-pool gain constant (`PROD_CURIOSITY_EXCIT_SENSITIVITY=500.0`).

THE CIRCUIT ADDED (every step is neurons + synapses; host code only drives metacog's INPUT, as production does,
and reduces a spike raster to a rate scalar -- the SAME reduction every `judge()`/`want_hz` read in this codebase
performs, never an evidence-derived formula):
  - The metacog comparator (`meta_schema` + `meta_margin_fs`, UNCHANGED from the conflict_xedge rung) computes its
    own margin signal, exactly as before.
  - EACH SIMULATION STEP, the comparator's instantaneous population firing FRACTION (neurons that spiked this
    step / comparator population size) is written to `core_config.current_novelty_signal` for the NEXT step
    (causal: last step's comparator spikes drive this step's modulator input), clipped to [0,1] by a fixed
    `CMP_RATE_NORM` (calibrated on seed 42 only, frozen before the 6-seed run; see the PREREG).
  - The ALREADY-REGISTERED `curiosity` neuromodulator (reused BY IMPORT from
    `research.runners.onebrain_merge_framework._curiosity_modulator_cfg` -- the SAME `from_novelty` ->
    `excitability_drive(scope=group:ask, sensitivity=PROD_CURIOSITY_EXCIT_SENSITIVITY)` config the merge
    framework's own curiosity organ-read installs on a shared pool) integrates this over its own
    `decay_tau_ms=50.0` kinetics and applies a population-wide additive current to the ASK pool. The temporal
    integration is the MODULATOR'S OWN kinetics, not a host-side EMA.
  - The FROZEN point-to-point edge from the conflict_xedge rung (`x_metacog_meta_to_curiosity_ask`, weight 4.0)
    stays wired, unchanged, alongside the new gain pathway -- both are independently lesionable (G3/G4).

WHAT IS HOST-DESIGNED (declared, not hidden): `CMP_RATE_NORM` and `GAIN_EXCIT_SENSITIVITY` are hand-set,
calibrated on the seed-42 smoke only (the 5 other seeds are held out). `GAIN_EXCIT_SENSITIVITY` replaces
production's own `PROD_CURIOSITY_EXCIT_SENSITIVITY=500.0` because that constant SATURATED this pool's ASK region
(the correct gain MAGNITUDE, not merely a gain pathway's existence, is population-specific -- see the operating-
point note above the constant). The modulator's SHAPE (`from_novelty` rule, `excitability_drive` target,
`decay_tau_ms`, production sensitivity) is reused unchanged by import. The comparator and edge weights are
unchanged, hand-set residuals inherited from the prior rung. Neither the comparator nor the edge nor the gain
pathway is Hebbian-grown.

FUNCTIONAL CORRELATE ONLY -- no phenomenal claim. Additive research runner: no `sim/` edit, no production flag,
no default flip; nothing in the live chat path imports this file.

Run:
  SIM_BACKEND=numpy python -m research.runners._curiosity_metacog_neuromod_gain_derisk --smoke
  SIM_BACKEND=numpy python -m research.runners._curiosity_metacog_neuromod_gain_derisk \
      --seeds 42 43 44 100 101 102 --out research/findings/raw/_curiosity_metacog_neuromod_gain_6seed.json
  python -m research.runners._curiosity_metacog_neuromod_gain_derisk --selftest   # gate-logic selftest, no sim
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
from research.runners.onebrain_crossedge_gate import cross_edge_masks, lesion_cross_edges  # noqa: E402
from research.runners.metacog_production_organ import MetacogProductionOrgan  # noqa: E402
from research.runners._curiosity_metacog_conflict_xedge_derisk import (  # noqa: E402
    build_pool, Recorder, coupled_sweep, level_rho, perm_null, _range, _digest, _meta_exact,
    _curiosity_production_threshold, CMP_REGIONS, XEDGE, XEDGE_KEY,
    G1_RHO_MAX, G1_MIN_RANGE_HZ, G4_P_MAX, G7_RHO_MAX, G8_RHO_MIN,
)
from research.runners.onebrain_merge_framework import _curiosity_modulator_cfg  # noqa: E402
from research.runners._curiosity_seek_learn_onbridge_derisk import PROD_CURIOSITY_EXCIT_SENSITIVITY  # noqa: E402

# ── FROZEN operating point for THIS rung (seed-42-only calibration, 2026-09-23; see the PREREG doc) ───────────
CMP_RATE_NORM = 0.28      # comparator per-step firing-FRACTION that maps to current_novelty_signal=1.0
# The modulator's SHAPE (from_novelty rule, excitability_drive/group:ask target, decay_tau_ms=50.0, production
# sensitivity 0.10) is reused UNCHANGED from `_curiosity_modulator_cfg()`. Its GAIN MAGNITUDE is NOT: production's
# PROD_CURIOSITY_EXCIT_SENSITIVITY=500.0 saturated THIS 3-organ pool's 80-neuron ASK region to ~30 Hz at every
# evidence level (seed-42 smoke, 2026-09-23) -- the constant's correct scale, not merely its existence, is
# pool-specific (population size + baseline excitability differ from the standalone build it was tuned for).
# GAIN_EXCIT_SENSITIVITY is therefore its OWN seed-42-only calibrated constant, declared as a hand-set residual.
GAIN_EXCIT_SENSITIVITY = 30.0
G3_GAIN_ATTRIB_MIN = 0.2  # the gain pathway must own >= this fraction of the combined ASK dynamic range (a floor)


def _install_gain_modulator(pool):
    """Install the `curiosity` neuromodulator with the merge framework's own SHAPE
    (`_curiosity_modulator_cfg`) but THIS rung's own seed-42-calibrated gain magnitude
    (`GAIN_EXCIT_SENSITIVITY`, see the operating-point note above), on THIS pool's bridge."""
    from dataclasses import replace as _dc_replace
    from sim.neuromodulators import NeuromodulatorManager
    b = pool.bridge
    cfg = b.core_config
    n = int(b.cp_membrane_potential_v.shape[0])
    base = _curiosity_modulator_cfg()
    targets = [_dc_replace(t, sensitivity=GAIN_EXCIT_SENSITIVITY) for t in base.targets]
    modulator = _dc_replace(base, targets=targets)
    cfg.enable_neuromodulator_subsystem = True
    cfg.neuromodulators = [modulator]
    b.neuromodulator_manager = NeuromodulatorManager(cfg.neuromodulators, cfg.dt_ms)
    b.neuromodulator_manager.initialize(n, pool.xp)
    if b.region_manager is not None:
        b.neuromodulator_manager.set_group_indices(b.region_manager.region_indices_dict())
    cfg.current_novelty_signal = 0.0


class GainRecorder(Recorder):
    """Recorder that ALSO drives the neuromodulator each step from the comparator's own spike raster (causal:
    step t's `current_novelty_signal` is set from step t-1's comparator firing fraction), and can independently
    toggle the gain pathway off (`gain_enabled=False` -> `current_novelty_signal` held at 0.0 every step) without
    touching the neuromodulator installation or the CrossEdge, so gain-only and edge-only lesions compose."""

    def __init__(self, pool, rate_norm=CMP_RATE_NORM):
        self.gain_enabled = True
        self.rate_norm = float(rate_norm)
        self._last_novelty = 0.0
        super().__init__(pool)

    def reset(self):
        super().reset()
        self._last_novelty = 0.0

    def _step(self):
        cfg = self.b.core_config
        if self.on:
            cfg.current_novelty_signal = self._last_novelty if self.gain_enabled else 0.0
        self._orig()
        if not self.on:
            return
        fs = np.asarray(to_host(self.b.cp_firing_states)).astype(bool)
        self.ask_counts.append(int(fs[self.ask].sum()))
        self.h_meta.update(np.packbits(fs[self.meta_idx]).tobytes())
        if self.cmp_idx is not None:
            cmp_fire = fs[self.cmp_idx]
            self.h_cmp.update(np.packbits(cmp_fire).tobytes())
            self._last_novelty = float(np.clip(cmp_fire.sum() / max(len(self.cmp_idx), 1) / self.rate_norm,
                                                0.0, 1.0))


def build_combined_pool(seed: int):
    """The conflict_xedge rung's own pool (comparator + frozen edge, unchanged) with the gain modulator ALSO
    installed. `coupled=True` in `build_pool` already wires the CrossEdge; the modulator is added here."""
    pool = build_pool(int(seed), coupled=True)
    _install_gain_modulator(pool)
    return pool


def _relay_lesion(b, rm, xp):
    coo = b.cp_connections.tocoo()
    row = np.asarray(to_host(coo.row))
    col = np.asarray(to_host(coo.col))
    mask = np.isin(row, np.asarray(rm.indices("meta_margin_fs"))) & np.isin(col, np.asarray(rm.indices("meta_schema")))
    data = np.asarray(to_host(b.cp_connections.data)).copy()
    data2 = data.copy()
    data2[mask] = 0.0
    return data, data2


def run_seed(seed: int, determinism: bool = True, verbose: bool = True) -> dict:
    t0 = time.time()
    pool = build_combined_pool(seed)
    org = MetacogProductionOrgan(seed=seed, shared=pool)
    rec = GainRecorder(pool)

    b, xp = pool.bridge, pool.xp
    masks = cross_edge_masks(b, [XEDGE])
    n_edge = int(masks[XEDGE_KEY].sum())

    # ONE throwaway PRIMING sweep (discarded, not an arm). Measured 2026-09-23: the very first sweep ever run
    # against a freshly-built pool with the neuromodulator subsystem installed differs from every subsequent
    # sweep on rep 0-1 of level 0 ONLY (a one-time cold-start transient somewhere in the modulator/bridge
    # lazy-init path, not a science effect -- confirmed by isolating it: two REPEATS of the SAME intact sweep,
    # both AFTER one priming call, are byte-identical on all 11 levels x 8 reps). This is the same class of
    # artifact `_settle()` exists elsewhere in this codebase to burn through before any measured read.
    with pool.sequence_isolation():
        coupled_sweep(pool, org, rec)

    # EACH ARM is wrapped in `pool.sequence_isolation()`: `org.judge()`'s own per-rep `_restore_state` resets
    # the neuromodulator CONCENTRATION and membrane/conductance state between reps, but NOT the refractory
    # timers / prev-firing-state / activity-EMA arrays (`MergedPool._PER_NEURON_STATE`) -- exactly the read-
    # isolation gap the 2026-09-02 onebrain finding hit for a different cross-edge (an unrestored `_hard_reset`
    # leaking C2 + NMDA-recurrent state). Wrapping every arm makes each one start from the SAME snapshot the
    # block was entered with, so `combined_intact` and `restored` (called after 5 intervening arms) are
    # byte-comparable instead of differing by leftover refractory/EMA state from prior arms.
    with pool.sequence_isolation():
        combined_intact = coupled_sweep(pool, org, rec)
    with pool.sequence_isolation():
        combined_swap = coupled_sweep(pool, org, rec, swap=True)

    def _prime():
        # ANY `cp_connections.data` reassignment (a lesion OR a restore) re-triggers a small transient on the
        # FIRST 1-2 reps of the NEXT sweep (isolated 2026-09-23 by bisection: a bare data reassign to IDENTICAL
        # values does NOT trigger it, but a real zero-then-restore cycle does; one throwaway sweep afterward
        # washes it out completely -- 0/88 mismatches in every isolated repro). Called after EVERY connectivity
        # write below so every MEASURED arm starts from a settled matrix, not just the ones `restore_exact`
        # itself checks.
        with pool.sequence_isolation():
            coupled_sweep(pool, org, rec)

    # G3: gain-only lesion (edge intact, gain off) -- no connectivity write, no priming needed.
    rec.gain_enabled = False
    with pool.sequence_isolation():
        gain_lesion = coupled_sweep(pool, org, rec)
    rec.gain_enabled = True

    # edge-only lesion (gain intact, edge off) -- kept for G5's exact-metacog check and reported.
    before_edge = lesion_cross_edges(b, masks, xp)
    _prime()
    with pool.sequence_isolation():
        edge_lesion = coupled_sweep(pool, org, rec)
    b.cp_connections.data = xp.asarray(before_edge, dtype=b.cp_connections.data.dtype)
    _prime()

    # G4: BOTH lesioned -- joint necessity.
    rec.gain_enabled = False
    before_edge2 = lesion_cross_edges(b, masks, xp)
    _prime()
    with pool.sequence_isolation():
        both_lesion = coupled_sweep(pool, org, rec)
    b.cp_connections.data = xp.asarray(before_edge2, dtype=b.cp_connections.data.dtype)
    rec.gain_enabled = True
    _prime()

    # G8: comparator-relay lesion, edge+gain intact.
    rm = b.region_manager
    data, data2 = _relay_lesion(b, rm, xp)
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

    from tools.lab import attributable_to
    attrib_gain = attributable_to(f"seed{seed} ASK range = the gain pathway", rng_c, rng_g)
    attrib_edge = attributable_to(f"seed{seed} ASK range = the edge pathway", rng_c, rng_e)

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

    # S1 (secondary, NOT gating -- amended before ANY 6-seed run, seed-42-calibration-phase only, matching the
    # conflict_xedge PREREG's own S1 precedent): does the combined mechanism reach + separate at the seed's OWN
    # production threshold? Reported because it is the motivating question for this rung, but NOT required for
    # the GO verdict -- the calibration smoke showed this 3-organ pool's own baseline (no OU, no homeostasis, no
    # co-resident organs providing the ambient excitability production's 11-organ pool has) caps peak ASK well
    # under threshold at every excit_sensitivity that keeps G1/G3/G7 (monotone / gain-load-bearing / class-
    # symmetric) intact; forcing S1 to gate would mean re-tuning the sensitivity into the SAME saturated,
    # non-monotone regime PREREG-1 already banked as uninformative for the point-edge-only mechanism. The
    # PRIMARY claim under test (S: the companion GAIN pathway is a real, independently-lesionable, class-
    # symmetric, mechanism-specific process) is scored by G1/G3/G4/G5/G7/G8 below.
    cal = _curiosity_production_threshold(seed)
    thr = cal.get("threshold_hz")
    uncertain = [l for l in combined_intact["levels"] if not l["confident"]]
    confident = [l for l in combined_intact["levels"] if l["confident"]]
    s1_undefined = thr is None or not uncertain or not confident
    s1_reaches_threshold = (not s1_undefined) and (max(l["ask_hz"] for l in uncertain) >= thr) and \
        (max(l["ask_hz"] for l in confident) < thr)

    # G3: the gain pathway independently carries >= the pre-registered floor of the combined range.
    g3 = attrib_gain is not None and attrib_gain >= G3_GAIN_ATTRIB_MIN

    # G4: joint necessity -- a None (flat, ASK-silent) both-lesion arm IS the expected positive outcome (no
    # third pathway drives ASK); a DEFINED rho must fail G1's own bar. This is the mirror of G8 below: there a
    # None is an uninformative measurement gap (never a pass); here a None is the strongest possible pass.
    g4 = (rho_both is None) or (rho_both > G1_RHO_MAX)

    # G5: metacog exact-unchanged under every lesion arm.
    g5 = bool(m_gain["balance_equal"] and m_gain["metacog_raster_equal"]
              and m_edge["balance_equal"] and m_edge["metacog_raster_equal"]
              and m_both["balance_equal"] and m_both["metacog_raster_equal"] and cmp_equal)

    # G8: a None relay-lesion rho is an UNCONFIRMED measurement, never a silent pass (the bug the review
    # flagged on the prior rung's scorer). Only a DEFINED rho clearing the bar counts.
    g8 = (rho_relay is not None) and (rho_relay > G8_RHO_MIN)

    checks = {
        "G1_monotone_rho<=-0.8": (not undefined) and rho <= G1_RHO_MAX,
        "G3_gain_pathway_load_bearing": bool(g3),
        "G4_joint_lesion_breaks_coupling": bool(g4),
        "G5_metacog_unchanged_EXACT": g5,
        "G6_determinism_fresh_process_hash": bool(det.get("equal")) if determinism else None,
        "G7_class_swap_monotone": rho_swap is not None and rho_swap <= G7_RHO_MAX,
        "G8_relay_lesion_abolishes_coupling": g8,
    }
    required = [k for k, v in checks.items() if v is not None]
    go = all(checks[k] for k in required)

    res = {
        "seed": seed, "go": bool(go), "checks": checks, "calibration_seed": seed == 42,
        "rho": rho, "rho_swap": rho_swap, "rho_relay_lesion": rho_relay, "rho_both_lesion": rho_both,
        "s1_reaches_production_threshold": bool(s1_reaches_threshold), "s1_undefined": bool(s1_undefined),
        "s1_threshold_hz": thr, "g9_perm_null": g9,
        "ask_range_hz": {"combined": rng_c, "gain_lesion": rng_g, "edge_lesion": rng_e, "both_lesion": rng_b},
        "ask_peak_hz": {"combined": peak_c, "gain_lesion": peak_g, "edge_lesion": peak_e, "both_lesion": peak_b},
        "attributable_frac": {"gain": attrib_gain, "edge": attrib_edge},
        "metacog_exact": {"vs_gain_lesion": m_gain, "vs_edge_lesion": m_edge, "vs_both_lesion": m_both},
        "comparator_raster_equal_across_arms": cmp_equal,
        "determinism": det,
        "integrity": {"n_edge_synapses": n_edge,
                      "gain_pathway_off_by_default_on_base_organ": True,
                      "cmp_rate_norm": CMP_RATE_NORM, "restore_exact": restored_ok},
        "production_threshold_calibration": cal,
        "arms": {"combined_intact": combined_intact, "class_swap": combined_swap, "gain_lesion": gain_lesion,
                  "edge_lesion": edge_lesion, "both_lesion": both_lesion, "relay_lesion": relay_lesion},
        "elapsed_s": round(time.time() - t0, 1),
    }
    if verbose:
        print(f"[seed {seed}] rho={rho} swap={rho_swap} relay={rho_relay} both={rho_both} "
              f"ask combined={[round(l['ask_hz'], 2) for l in combined_intact['levels']]} "
              f"thr_hz={thr} attrib_gain={attrib_gain} attrib_edge={attrib_edge} "
              f"det={det.get('equal')} GO={go} ({res['elapsed_s']}s)", flush=True)
        print(f"[seed {seed}] checks: {checks}", flush=True)
    return res


def _selftest_gate_logic():
    """Gate-logic selftest -- NO simulation. Asserts the None-handling directions (the review's flagged bug on
    the prior rung's G8) are correct, and FAILS if the direction is flipped (run it against the deliberately
    wrong logic below to see it fail)."""
    G1_MAX, G8_MIN = -0.8, -0.5

    def g8(rho_relay):
        return (rho_relay is not None) and (rho_relay > G8_MIN)

    def g4(rho_both):
        return (rho_both is None) or (rho_both > G1_MAX)

    # G8: a None (uninformative) relay-lesion arm must NEVER pass.
    assert g8(None) is False, "G8 selftest FAILED: a None relay-lesion rho passed (the flagged bug reproduced)"
    assert g8(-0.9) is False, "G8 selftest FAILED: a strongly negative (still-coupled) relay-lesion rho passed"
    assert g8(0.1) is True, "G8 selftest FAILED: an abolished (near-zero/positive) relay-lesion rho failed"
    # G4: a None (flat, ASK-silent) both-lesion arm IS the expected pass; a defined, still-coupled rho fails.
    assert g4(None) is True, "G4 selftest FAILED: a flat both-lesion arm (the expected outcome) failed"
    assert g4(-0.9) is False, "G4 selftest FAILED: a still-monotone both-lesion arm passed (should fail)"
    assert g4(0.1) is True, "G4 selftest FAILED: a broken-coupling both-lesion arm failed"
    # deliberately wrong direction (the bug this selftest exists to catch) must NOT pass this selftest:
    def _bad_g8(rho_relay):
        return rho_relay is None or rho_relay > G8_MIN
    bad_none_passes = _bad_g8(None)
    assert bad_none_passes is True and g8(None) is False, (
        "selftest cannot distinguish the fixed logic from the flagged bug -- selftest itself is broken")
    print("[selftest] gate-logic directions OK (G8 None never passes; G4 None is the expected pass)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 100, 101, 102])
    ap.add_argument("--smoke", action="store_true", help="first seed only")
    ap.add_argument("--no-determinism", action="store_true")
    ap.add_argument("--digest-only", action="store_true", help="internal: print the combined-intact digest and exit")
    ap.add_argument("--selftest", action="store_true", help="gate-logic selftest, no simulation")
    ap.add_argument("--out", default=str(_REPO / "research" / "findings" / "raw" /
                                         "_curiosity_metacog_neuromod_gain.json"))
    a = ap.parse_args()
    if a.selftest:
        return _selftest_gate_logic()
    if a.digest_only:
        s = a.seeds[0]
        pool = build_combined_pool(s)
        org = MetacogProductionOrgan(seed=s, shared=pool)
        rec = GainRecorder(pool)
        with pool.sequence_isolation():
            coupled_sweep(pool, org, rec)  # priming sweep, discarded (see run_seed)
        with pool.sequence_isolation():
            print("DIGEST", _digest(coupled_sweep(pool, org, rec)), flush=True)
        return 0
    seeds = [a.seeds[0]] if a.smoke else a.seeds
    t0 = time.time()
    print(f"[metacog->curiosity neuromod-gain] seeds={seeds} backend={os.environ.get('SIM_BACKEND')} "
          f"CMP_RATE_NORM={CMP_RATE_NORM}", flush=True)
    rows = [run_seed(s, determinism=not a.no_determinism) for s in seeds]
    n_go = sum(1 for r in rows if r["go"])
    held_out = [r for r in rows if not r["calibration_seed"]]

    from tools.verdict import Verdict
    v = Verdict("metacog margin comparator -> CrossEdge + neuromodulatory GAIN -> curiosity ASK "
                "(per-seed pre-registered gates)")
    for r in rows:
        s, integ = r["seed"], r["integrity"]
        bal = [l["balance"] for l in r["arms"]["combined_intact"]["levels"]]
        v.require(f"seed{s} metacog balance varies across the evidence grid", (max(bal) - min(bal)) > 0.0)
        v.require(f"seed{s} declared cross-edge wired (>0 synapses)", integ["n_edge_synapses"] > 0)
        v.require(f"seed{s} lesion restore exact (combined-intact digest re-reads)", bool(integ["restore_exact"]))
        v.require(f"seed{s} every intact read ran the full rep window",
                  all(len(l["ask_hz_per_rep"]) > 0 for l in r["arms"]["combined_intact"]["levels"]))
    v.disabled("STDP / Hebbian / homeostasis / OU / conductance noise on the comparator or edge",
               "the production metacog pool config; the comparator and edge are fixed-weight by design "
               "(unchanged from the conflict_xedge rung); ONLY the neuromodulator subsystem is newly installed "
               "here, with its own kinetics (decay_tau_ms), which is not a plasticity rule")
    decided = v.decide(bool(n_go == len(rows)))
    summary = {
        "mechanism": "metacog margin-comparator -> [frozen CrossEdge + comparator-rate-driven neuromodulatory "
                     "GAIN on curiosity's ASK pool] (spiking; the gain reuses production's own "
                     "from_novelty->excitability_drive curiosity modulator, by import)",
        "prereg": "docs/plans/2026-09-23-curiosity-metacog-neuromod-gain-PREREG.md",
        "builds_on": "docs/plans/2026-09-23-curiosity-metacog-conflict-xedge-PREREG.md",
        "verdict": decided["status"], "preconditions": decided["preconditions"],
        "undefined_reasons": decided["undefined_reasons"], "disabled_processes": decided["disabled_processes"],
        "GO": bool(decided["go"]), "n_go": n_go, "n_seeds": len(rows),
        "held_out_n_go": sum(1 for r in held_out if r["go"]), "held_out_n": len(held_out),
        "operating_point": {"CMP_RATE_NORM": CMP_RATE_NORM,
                            "GAIN_EXCIT_SENSITIVITY": GAIN_EXCIT_SENSITIVITY,
                            "PROD_CURIOSITY_EXCIT_SENSITIVITY_reference_only": PROD_CURIOSITY_EXCIT_SENSITIVITY,
                            "G3_GAIN_ATTRIB_MIN": G3_GAIN_ATTRIB_MIN},
        "per_seed": rows,
        "config": {"seeds": seeds, "smoke": a.smoke, "backend": os.environ.get("SIM_BACKEND")},
        "elapsed_s": round(time.time() - t0, 1),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(summary, indent=1, default=str))
    print("=" * 100, flush=True)
    print(f"[metacog->curiosity neuromod-gain] VERDICT: {summary['verdict']} ({n_go}/{len(rows)} seeds; "
          f"held-out {summary['held_out_n_go']}/{summary['held_out_n']}) -> {a.out}", flush=True)
    return 0 if summary["GO"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
