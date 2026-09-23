"""D3 ONE-BRAIN — verify the AFFECT organ's migration onto the shared cortical pool (12 organs, ONE bridge) AND the
first cross-region synapse it brings (the ladder's own held arousal -> the D2 surprise pool).

Module under test: `research/runners/onebrain_affect_pool.py` (read its docstring for the readiness inventory and
the declared residuals). Everything is DEFAULT-OFF (`BRAIN_ONEBRAIN_AFFECT_POOL`, `BRAIN_ONEBRAIN_AFFECT_XEDGE`).

PRE-REGISTERED GATE (written 2026-09-23 BEFORE any 12-organ / cross-edge result existed; thresholds are constants
below and are NOT to be edited after a result is seen — a changed threshold is a new, separately-labelled gate).

ARM M — MIGRATION (answer-preservation; the 12-organ pool WITHOUT the cross-edge):
  M1 co-residence byte-identity: the affect read battery on the 12-organ pool == the SAME battery with affect alone
     on the 12-organ SUPERSET config (max |delta| == 0.0 exactly) — the merge adds no non-synaptic coupling.
  M2 carried-organ answer preservation (STRICT): each of the 11 Wave-3 organs' read battery on the 12-organ pool
     is BYTE-IDENTICAL to the SHIPPED `get_wave3_pool(seed)` (the production-default pool), and its rendered answer
     is identical — adding affect perturbs no production organ.
  M3 affect answer preservation vs TODAY'S production path: the graded tone LEVEL (the categorical value
     `content_plan`/`manner_for` consume) at every production appraisal (-1,-0.5,0,+0.5,+1) equals the standalone
     production ladder's (`AppraisalInteroceptiveLadder`, shared=None). Continuous differential reported, not
     gated (different per-neuron het/OU realisation by construction — the same honest residual Waves 1-3 declared
     for their new organs).
  M4 faculty alive on the pool: sign correct at |a|>=0.5, |neutral| < LADDER_NEUTRAL_TOL, affect_out lesion -> 0.0,
     interoceptive-synapse lesion collapses |diff| at +-1 to <= 0.25x intact.
  M5 weights frozen: every ladder synapse weight (both endpoints in affect regions) byte-identical before vs after
     the full 12-organ train+read lifecycle (the pool's global Hebbian cannot move plastic=False edges).
  M6 legacy discriminator: with the name-keyed seams OFF, affect's slice init DIVERGES merged-vs-alone (M1 is not
     vacuous).
  M7 determinism: the affect battery read twice on the same pool is identical.

ARM X — INTEGRATION (the arousal->surprise synapse on the one pool; mechanism level) — AMENDED GATE v2
(`X_INSTRUMENT` below). The v1 arm-X gate (X1..X7 on a mean-Hz shift floor) is SUPERSEDED, see the AMENDMENT LOG.
  Protocol, in ONE continuous sequence on the pool: appraisal ramp into the ladder via the interoceptive relays ->
  drive-off hold (the ladder LATCHES arousal) -> relays silent -> the surprise organ's OWN read drive sequence
  (the production `_drive_read` protocol: prediction phase cue i @600 pA x PRE_STEPS, then cue i + asserted block j
  @ strength S x HOLD, stepped with the production `_step`) from the held state, per trial. CONTRADICT: j=(i+1)%n
  (the organ's own calibration contradiction); CONFIRM: j=i. VERDICT per trial = surprise Hz >= the organ's own
  build-time threshold (exactly `SurpriseProductionOrgan.judge`). OPERATING POINT: the ladder's OU background is
  confined to the affect organ's neurons (`local_ou(scope="affect")`, the engine's `cp_ou_neuron_mask` seam), so the
  surprise pool runs noise-free as in production; Hebbian is OFF in the window (the frozen organ reads frozen).
  Assertion strengths: ASSERT_GRID (600 pA = the production drive, down to 250 pA = weak / low-salience evidence).
  S* (selected on the a=0 read ONLY, before and independent of any arousal read): the LARGEST grid strength at which
  the a=0 brain flags <= MARGINAL_FRAC of the contradictions. No such strength -> X1 is UNDEFINED (= not passed).
  EVIDENTIAL checks (gate v3 -- see the AMENDMENT LOG; exactly TWO, and only X1 is evidence FOR the effect):
  X0 OPERATING POINT: at a=0, at EVERY grid strength, every contradict (and, at 600 pA, confirm) trial's verdict in
     this battery EQUALS the verdict of the organ's literal production read path (`read_isolation` + `_hard_reset`
     + `_drive_read`) for the same (cue, asserted) blocks on the same pool. The instrument must read what production
     reads before any arousal effect is scored.
  X1 FUNCTIONAL VERDICT FLIP -- ONE test per seed: at S*, the held arousal at a=+1 NEWLY flags (not-surprised at
     a=0 -> surprised) at least max(MIN_FLIPS, ceil(FLIP_FRAC * n_trained)) contradict trials. This replaces v1's
     0.10 Hz rate floor: it is a change of the organ's own verdict, not a rate shift. a=-1 is NOT a second test:
     the arousal relay is driven by |appraisal| (`PoolAffectLadder._drive_relays`) and the arousal rungs receive
     no valence-rung input, so the a=-1 read is a CONSTRUCTION DUPLICATE of the a=+1 read. It is REPORTED (its
     flips, and whether its per-trial Hz at S* equal a=+1's exactly), never counted. The only replication of X1
     is ACROSS THE 6 SEEDS (all must pass). The per-seed threshold is unchanged from v2 (see the LOG for why).
  ATTRIBUTION (tools.lab, reported per seed, not a separate verdict): `lever` on the a=+1 verdict vector at S*,
     intact vs edge-lesion (continuous = mean Hz); `attributable_to` of X1's newly-flagged count against each
     control that varies one term -- the edge lesion (the synapse), the intero-null (the ladder's latched arousal)
     and the no-edge pool (the topology). I1-I3 are the gated forms of the same controls.
  INTEGRITY SMOKES (required, but they pass largely BY CONSTRUCTION of the topology -- the edge is the only
  affect->surprise path -- or, for I8, by the physics of a weak additive drive; they are NOT evidence for the effect):
  I1 edge lesion (gate `affect_arousal_to_surprise`=0), a=+1: verdicts at S* and 600 == the a=0 verdicts.
  I2 no-edge pool (same 12 organs, no synapse): a=+1 verdicts == a=0 verdicts at S* and 600, and its a=0 verdicts ==
     the edge pool's a=0 verdicts.
  I3 intero-null (relay->ladder synapse cut; the relays still fire, the ladder never latches): newly-flagged at S*
     <= floor(LESION_RATIO * X1's a=+1 newly-flagged count).
  I4 byte-off: 600 pA a=0 per-trial Hz equal edge-pool vs no-edge pool exactly, AND every one of the 12 organs'
     standard read batteries is byte-identical edge-pool vs no-edge pool.
  I5 held arousal: arousal-rung rate in the surprise window > 0 at a=+-1 and == 0 at a=0; no external current on
     any ladder rung or relay in the surprise window (asserted after every drive change).
  I6 determinism: the a=+1 battery at (S*, 600) read twice -> identical per-trial Hz.
  I7 scope invariance: the ladder's affect read at a=+-1 is identical with OU scope "affect" vs "all" (the mask
     changes only the non-affect neurons' noise, never the ladder).
  I8 production-strength SAFETY (was v2's X2, relabelled in v3): at 600 pA, arousal (a=+-1) adds NO confirm false
     alarm (confirm surprised count == its a=0 count) and loses NO contradict detection. NEAR-GUARANTEED at
     w=0.05: at 600 pA the a=0 brain already flags every contradiction and arousal only ADDS excitation (so a lost
     detection can barely occur), and confirm sits at a ~0 Hz floor that w=0.05 does not lift (the v1 sweep saw
     confirm cross only at w=0.4). A safety check against a regression, not evidence of specificity.
  REPORTED, NOT GATED, NO CLAIM: mean contradict Hz per strength at a=0/+1/-1 (raw f-I data; whether the effect is
  a GAIN or an additive DC drive is NOT tested and NOT claimed), the production-strength (600 pA) contradict
  verdict changes (typically none: the a=0 brain already flags every contradiction there), newly-lost
  detections at S*, the a=-1 construction duplicate, and the attribution fractions.

GO = all of M1..M7 AND X0..X1 AND I1..I8 on every seed. `XEDGE_W` = 0.05 is a HAND-SET constant (see the
provenance note in `onebrain_affect_pool.py`); `--calibrate` is now a DIAGNOSTIC on a NON-gate seed (default 7)
and does not set it. Seeds: 42 43 44 100 101 102.

AMENDMENT LOG
  2026-09-23 11:56 EDT (fix round after the adversarial review of 7b46d761e; committed before any v2 run). Replaced the v1 arm-X gate (X1..X7)
  with v2 (X0..X2 + I1..I7) and changed the arm-X instrument's operating point (OU confined to affect neurons;
  production `_step`/drive sequence). Arm M (M1..M7) is UNCHANGED (thresholds, code path and verdict rule).
  RESULTS SEEN before this amendment: the seed-42 1-seed smoke (`smoke_seed42.json`, M-type checks, 7/7) and the
  seed-42 v1 calibration (`calibrate_seed42.json`: at 600 pA under the v1 noisy operating point, a=0 contradict
  per-block rates 2.66..5.21 Hz vs threshold 2.63 -> all 8 flagged; +0.224 Hz shift at w=0.05). NOT seen: any
  6-seed M or X result (none harvested), any v2-instrument read, any weak-strength curve under v2. The v2 strength
  grid, MARGINAL_FRAC, FLIP_FRAC and MIN_FLIPS were chosen knowing the v1 seed-42 production-strength saturation
  above (that is WHY the gate reads verdicts at a marginal strength), not from any v2 measurement. The v1 X-arm
  jobs dispatched at bfc6978 measure the superseded instrument; `aggregate` ignores their X checks.
  2026-09-23 12:55 EDT (PATH ONLY, no threshold / rule / code change; nothing v2 seen except the 2-organ seed-7
  instrument smoke in a1a5d3e7c). The v2 arm-X pool jobs (revision c6fdf7be7) write to the `xv2/` subdirectory
  (own output dir per arm), so the literal scoring command becomes:
    python -m research.runners._onebrain_affect_pool_verify --aggregate \
      'research/findings/raw/_onebrain_affect_pool/verify_M_seed*.json' \
      'research/findings/raw/_onebrain_affect_pool/xv2/verify_X_seed*.json'
  2026-09-23 ~13:45 EDT -- GATE v3 (SCORING + ATTRIBUTION only; the MEASUREMENT is unchanged). Committed after the
  re-review of 4708ebdee and BEFORE any 6-seed v2 arm-X result was read. What was known at this point: the same
  2-organ seed-7 smoke as above (a=+1 newly flagged 2/8 at S*=350, exactly the minimum). The v1 X101/X102 jobs
  finished on pool41 at 13:10/13:22 (superseded instrument; files NOT opened). The v2 X42/X43 jobs were dispatched
  on pool41 at 13:38 before this entry; their outputs had not been written or read when it was committed.
  Changes: (1) X1 counts a=+1 ONCE -- the "AND at a=-1" half is withdrawn as a construction duplicate (arousal
  relay driven by |appraisal|); a=-1 is reported. The per-seed threshold max(2, ceil(n/4)) is NOT changed: raising
  it now, knowing the only v2 read sits exactly at 2/8, would be choosing a threshold the seen data already
  fails; lowering it is not warranted either. What changes is the claimed evidence -- ONE test per seed,
  replicated only across seeds. (2) v2's X2 is relabelled I8 (safety, near-guaranteed at w=0.05), so the
  evidential set is X0+X1. (3) The tools.lab `lever` + `attributable_to` calls that v1 made and the v2 rewrite
  dropped (a BLOCK-class `attribution-required` regression) are restored on the X1 flips vs the three controls.
  (4) Arm-X scoring now lives in ONE pure function, `score_x_arm`, over the recorded raw batteries; `aggregate`
  RE-SCORES every current-instrument X record with it and ignores the checks the file stored. So the v2 X jobs
  (revision c6fdf7be7, whose battery code is identical -- pinned by a test) are scored by THIS rule, not v2's.
  No threshold, grid, S* rule, weight or battery changed. Arm M is unchanged.

COMPUTE: numpy CPU, ~7.7k-neuron pools (a few GB; the surprise organ's on-pool training dominates, ~11 min per
pool build on one core). Pool nodes: one seed x one arm per queue line.
  SIM_BACKEND=numpy python -u -m research.runners._onebrain_affect_pool_verify --arms M --seeds 42 \
      --json research/findings/raw/_onebrain_affect_pool/verify_M_seed42.json
  SIM_BACKEND=numpy python -u -m research.runners._onebrain_affect_pool_verify --arms X --seeds 42 \
      --json research/findings/raw/_onebrain_affect_pool/verify_X_seed42.json
  SIM_BACKEND=numpy python -u -m research.runners._onebrain_affect_pool_verify --calibrate --seeds 7 \
      --json research/findings/raw/_onebrain_affect_pool/calibrate_v2_seed7.json     # DIAGNOSTIC only
Aggregate (the literal GO-gate command; a seed needs BOTH arms, a missing check is NOT a pass):
  python -m research.runners._onebrain_affect_pool_verify --aggregate \
      'research/findings/raw/_onebrain_affect_pool/verify_M_seed*.json' \
      'research/findings/raw/_onebrain_affect_pool/xv2/verify_X_seed*.json'     # (path amended, see LOG)
"""
from __future__ import annotations

import os
os.environ.setdefault("SIM_BACKEND", "numpy")

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from tools.lab import void_if, undefined_if_empty, lever, attributable_to  # noqa: E402

SEEDS = (42, 43, 44, 100, 101, 102)
X_INSTRUMENT = "v2-production-operating-point-verdict-2026-09-23"   # arm-X files without this are superseded (v1)
X_GATE = "v3-single-count-X1-X2-as-I8-attribution-2026-09-23"      # the SCORING rule (`score_x_arm`); see the LOG
# the raw batteries a current-instrument arm-X record must carry for `score_x_arm` to (re-)score it
_X_RAW_KEYS = ("production_path", "base", "pos", "neg", "pos_again", "lesion", "intero_null", "noedge_base",
               "noedge_pos", "ladder_scope_invariance", "x5_reads")
LESION_RATIO = 0.34          # I3: intero-null newly-flagged count <= floor(this x intact newly-flagged count)
INTERO_COLLAPSE = 0.25       # M4: interoceptive-synapse lesion |diff| <= this x intact at +-1
CAL_WEIGHTS = (0.02, 0.05, 0.1, 0.2, 0.4)   # the diagnostic calibration sweep (effective per-synapse weight)
CAL_BUILD_W = 0.4            # calibration builds at the max weight; lower weights = transmission gate fraction
CAL_SEED_DEFAULT = 7         # the diagnostic calibration runs on a NON-gate seed
PROD_ASSERT_PA = 600.0       # the production assertion drive (`SurpriseProductionOrgan.read_surprise`)
CUE_PA = 600.0               # the production cue drive
ASSERT_GRID = (600.0, 500.0, 450.0, 400.0, 375.0, 350.0, 325.0, 300.0, 275.0, 250.0)
MARGINAL_FRAC = 0.5          # S*: largest grid strength where the a=0 brain flags <= this fraction of contradictions
FLIP_FRAC = 0.25             # X1: newly-flagged contradict trials >= max(MIN_FLIPS, ceil(FLIP_FRAC * n_trained))
MIN_FLIPS = 2
PRE_STEPS = 60               # == the production read's prediction phase
HOLD = 60                    # == the production read's assertion window


# ─────────────────────────────────────────────────────────────────────────────────────────────
def _maxdelta(a, b):
    from research.runners._onebrain_wave1_organread_verify import _maxdelta as md
    return md(a, b)


def _ladder_weights(pool):
    """Every synapse weight with BOTH endpoints in an affect region (the M5 frozen-weights witness)."""
    from research.runners.onebrain_affect_pool import AFFECT_KEY
    b = pool.bridge
    idx = np.concatenate([np.asarray(v) for v in pool.idx(AFFECT_KEY).values()])
    coo = b.cp_connections.tocoo()
    row = np.asarray(coo.row); col = np.asarray(coo.col)
    m = np.isin(row, idx) & np.isin(col, idx)
    return np.asarray(coo.data)[m].astype(np.float64)


def _reads_all(pool, descs, seed):
    from research.runners._onebrain_wave1_organread_verify import _isolated_reads
    return _isolated_reads(pool, descs, seed)


# ─────────────────────────────────────────────────────────────────────────────────────────────
#  ARM X instrument (v2) — the ladder holds arousal, then the surprise organ reads with its OWN production drive
#  sequence, per trial, from the held state. Operating point = production: the surprise pool is noise-free.
# ─────────────────────────────────────────────────────────────────────────────────────────────
_HELD_EXTRA = ("cp_neuron_activity_ema", "cp_neuron_firing_thresholds")   # _hard_reset restores these too


def _snap_held(b, xp):
    from research.runners._gnw_rung1_ignition_curve_derisk import _snapshot_state
    snap = _snapshot_state(b, xp)
    for nm in _HELD_EXTRA:
        arr = getattr(b, nm, None)
        if arr is not None and nm not in snap:
            snap[nm] = arr.copy()
    return snap


def _n_flip_required(nt):
    return int(max(MIN_FLIPS, int(np.ceil(FLIP_FRAC * nt))))


def _surprise_trial(b, idx_map, xp, i, j, strength, guard_idx, sur_idx, ar_idx):
    """ONE surprise read with the production `_drive_read` sequence (prediction phase cue i @CUE_PA x PRE_STEPS, then
    cue i + asserted j @strength x HOLD, `_step`-driven), also counting the held arousal rungs. Returns (Hz, rung Hz)."""
    from research.runners._spiking_expectation_rpe_derisk import _set_drives, _step
    _set_drives(b, idx_map, {"cue": (i, CUE_PA)}, xp)
    assert float(np.abs(np.asarray(b.cp_external_input_current)[guard_idx]).max()) == 0.0, \
        "a ladder rung / relay got external current in the surprise window"
    for _ in range(PRE_STEPS):
        _step(b)
    _set_drives(b, idx_map, {"cue": (i, CUE_PA), "patient_asserted": (j, float(strength))}, xp)
    assert float(np.abs(np.asarray(b.cp_external_input_current)[guard_idx]).max()) == 0.0, \
        "a ladder rung / relay got external current in the surprise window"
    s_cnt = a_cnt = 0
    for _ in range(HOLD):
        _step(b)
        fs = np.asarray(b.cp_firing_states)
        s_cnt += int(fs[sur_idx].sum())
        a_cnt += int(fs[ar_idx].sum())
    return (s_cnt / max(1, sur_idx.size) / (HOLD * 1e-3), a_cnt / max(1, ar_idx.size) / (HOLD * 1e-3))


def _cond_summary(hz, thr):
    v = [bool(h >= thr) for h in hz]
    return {"hz": [float(h) for h in hz], "surprised": v, "n_surprised": int(sum(v)),
            "frac": float(np.mean(v)) if v else None, "mean_hz": float(np.mean(hz)) if hz else None}


def arousal_surprise_battery(pool, ladder, sorg, appraisal, *, strengths=ASSERT_GRID,
                             confirm_strengths=(PROD_ASSERT_PA,), xedge_lesion=False, intero_lesion=False,
                             xedge_gain=1.0):
    """Appraisal -> ladder hold (OU confined to the affect neurons) -> per trial: restore the held state and run the
    surprise organ's production read at each assertion strength. Returns per-strength contradict (and confirm)
    per-trial Hz + verdicts (Hz >= the organ's own threshold) and the held arousal-rung rate."""
    from research.runners._gnw_rung1_ignition_curve_derisk import _restore_state
    from research.runners.onebrain_affect_pool import XEDGE_GATE
    b, xp = pool.bridge, pool.xp
    ladder.ensure_built(); sorg.ensure_built()
    idx_map, meta = sorg.idx_map, sorg.meta
    nt = int(meta["n_trained"])
    sur_idx = np.asarray(idx_map["surprise"])
    ar_idx = np.asarray(ladder.arousal_flat)
    guard_idx = np.concatenate([ladder._ladder_flat] + [np.asarray(v) for v in ladder.relay_idx.values()])
    thr = float(sorg.threshold)
    contra, conf, ar_rate = {}, {}, []
    with pool.sequence_isolation():
        _restore_state(b, pool.snap)
        b.cp_external_input_current[:] = 0.0
        with ladder.local_ou(scope="affect"):
            ladder.set_gates(intero_lesion=intero_lesion, xedge_lesion=xedge_lesion)
            if ladder._xedge and not xedge_lesion:
                b.set_transmission_gate(XEDGE_GATE, float(xedge_gain))
            try:
                ladder.run_appraisal_phase(appraisal)
                held = _snap_held(b, xp)
                # the affect neurons' OU state is part of the held state: restoring it per trial makes every trial a
                # pure function of (held state, i, j, S) -- independent of trial ORDER and of which strengths ran
                # before (so a probe-subset re-read is comparable trial-for-trial with the full-grid read).
                held_ou = (b.cp_ou_current.copy(), int(b._ou_pn_step))
                jobs = [(float(S), "contra") for S in strengths] + [(float(S), "conf") for S in confirm_strengths]
                for S, kind in jobs:
                    hz = []
                    for i in range(nt):
                        j = (i + 1) % nt if kind == "contra" else i
                        _restore_state(b, held)
                        b.cp_ou_current[:] = held_ou[0]
                        b._ou_pn_step = held_ou[1]
                        b._blk = meta["blk"]
                        h, a = _surprise_trial(b, idx_map, xp, i, j, S, guard_idx, sur_idx, ar_idx)
                        hz.append(h); ar_rate.append(a)
                    (contra if kind == "contra" else conf)[S] = _cond_summary(hz, thr)
            finally:
                ladder.restore_gates()
        b.cp_external_input_current[:] = 0.0
    return {"appraisal": float(appraisal), "threshold": thr, "n_trained": nt,
            "contradict": {f"{S:g}": v for S, v in contra.items()},
            "confirm": {f"{S:g}": v for S, v in conf.items()},
            "arousal_rung_hz": float(np.mean(ar_rate)) if ar_rate else None,
            "xedge_lesion": bool(xedge_lesion), "intero_lesion": bool(intero_lesion), "xedge_gain": float(xedge_gain)}


def production_path_battery(pool, sorg, *, strengths=ASSERT_GRID, confirm_strengths=(PROD_ASSERT_PA,)):
    """The organ's LITERAL production read path (surprise_production_organ.read_surprise lines: read_isolation +
    _hard_reset + _drive_read) for the same (cue, asserted) blocks -- the X0 operating-point reference. The pool's
    config is left exactly as production leaves it; the whole battery is sequence-isolated."""
    from research.runners._spiking_expectation_rpe_derisk import _hard_reset, _drive_read
    b, xp = pool.bridge, pool.xp
    sorg.ensure_built()
    idx_map, meta = sorg.idx_map, sorg.meta
    nt = int(meta["n_trained"])
    thr = float(sorg.threshold)
    contra, conf = {}, {}
    with pool.sequence_isolation():
        jobs = [(float(S), "contra") for S in strengths] + [(float(S), "conf") for S in confirm_strengths]
        for S, kind in jobs:
            hz = []
            for i in range(nt):
                j = (i + 1) % nt if kind == "contra" else i
                b._blk = meta["blk"]
                with pool.read_isolation("surprise"):
                    _hard_reset(b)
                    r = _drive_read(b, idx_map, {"cue": (i, CUE_PA), "patient_asserted": (j, S)},
                                    HOLD, xp, ["surprise"], pre_drives={"cue": (i, CUE_PA)}, pre_steps=PRE_STEPS)
                hz.append(float(r["surprise"]))
            (contra if kind == "contra" else conf)[S] = _cond_summary(hz, thr)
        b.cp_external_input_current[:] = 0.0
    return {"threshold": thr, "n_trained": nt, "contradict": {f"{S:g}": v for S, v in contra.items()},
            "confirm": {f"{S:g}": v for S, v in conf.items()}}


def select_marginal_strength(base):
    """S*: the LARGEST grid strength at which the a=0 brain flags <= MARGINAL_FRAC of the contradictions. Reads the
    a=0 battery ONLY (never an arousal read). None -> no marginal strength in the grid (X1 UNDEFINED)."""
    for S in sorted(ASSERT_GRID, reverse=True):
        c = base["contradict"].get(f"{S:g}")
        if c is not None and c["frac"] is not None and c["frac"] <= MARGINAL_FRAC:
            return float(S)
    return None


def _newly(base_cond, cond):
    """(newly flagged, newly lost) trial counts cond vs base_cond (same trials, same order)."""
    b0, b1 = base_cond["surprised"], cond["surprised"]
    return (int(sum((not x) and y for x, y in zip(b0, b1))), int(sum(x and (not y) for x, y in zip(b0, b1))))


def _verdicts_equal(a, b, strengths, kind="contradict"):
    return all(a[kind][f"{S:g}"]["surprised"] == b[kind][f"{S:g}"]["surprised"] for S in strengths)


def score_x_arm(raw):
    """THE arm-X verdict (gate v3), a pure function of the recorded raw batteries (`_X_RAW_KEYS`). Used by
    `verify_seed` at run time AND by `aggregate`, which re-scores every current-instrument record with it, so a
    file produced under an earlier scoring rule is judged by this one. Returns (checks, details)."""
    base, prod, pos, neg = raw["base"], raw["production_path"], raw["pos"], raw["neg"]
    les, inull, nb, npos = raw["lesion"], raw["intero_null"], raw["noedge_base"], raw["noedge_pos"]
    pos_again, scope, x5 = raw["pos_again"], raw["ladder_scope_invariance"], raw["x5_reads"]
    P = PROD_ASSERT_PA
    kP = f"{P:g}"
    s_star = select_marginal_strength(base)
    probe = tuple(dict.fromkeys(s for s in (s_star, P) if s is not None))
    need = _n_flip_required(int(base["n_trained"]))
    checks = {}
    void_x = void_if(pos["arousal_rung_hz"] == 0.0, "the arousal ladder never latched at a=+1 -- the "
                                                      "cross-edge had no presynaptic activity to carry")
    # X0 operating point
    x0 = (_verdicts_equal(base, prod, ASSERT_GRID, "contradict") and _verdicts_equal(base, prod, (P,), "confirm"))
    x0_maxdhz = max(abs(h1 - h2) for kind, grid in (("contradict", ASSERT_GRID), ("confirm", (P,)))
                    for S in grid for h1, h2 in zip(base[kind][f"{S:g}"]["hz"], prod[kind][f"{S:g}"]["hz"]))
    checks["X0_operating_point_verdicts_equal_production_read"] = bool(x0)
    # X1 functional verdict flip at S*, a=+1 ONLY (a=-1 is a construction duplicate: reported, never counted)
    attribution = {"lever_moved": None, "edge_lesion": None, "intero_null": None, "no_edge_pool": None}
    if s_star is None:
        flips = {"pos": None, "lost_pos": None, "neg_reported": None, "lost_neg_reported": None}
        neg_dup = None
        fi = fl = fne = None
        x1 = False
        undefined_reason = ("no grid strength at which the a=0 brain misses >= half of the contradictions -- "
                            "S* undefined, X1 UNDEFINED (not a pass)")
    else:
        k = f"{s_star:g}"
        fp, lp = _newly(base["contradict"][k], pos["contradict"][k])
        fn, ln = _newly(base["contradict"][k], neg["contradict"][k])
        flips = {"pos": fp, "lost_pos": lp, "neg_reported": fn, "lost_neg_reported": ln}
        neg_dup = bool(neg["contradict"][k]["hz"] == pos["contradict"][k]["hz"])
        fl = _newly(base["contradict"][k], les["contradict"][k])[0]
        fi = _newly(base["contradict"][k], inull["contradict"][k])[0]
        fne = _newly(nb["contradict"][k], npos["contradict"][k])[0]
        x1 = bool(fp >= need) and not void_x
        undefined_reason = None
        # ATTRIBUTION (tools.lab): does the synapse move the verdict, and whose is X1's newly-flagged count?
        attribution["lever_moved"] = lever(
            "arousal->surprise synapse (a=+1 verdicts @S*, intact vs edge-lesion)",
            tuple(pos["contradict"][k]["surprised"]), tuple(les["contradict"][k]["surprised"]), required=False,
            continuous=(pos["contradict"][k]["mean_hz"], les["contradict"][k]["mean_hz"]))
        attribution["edge_lesion"] = attributable_to("X1 flips @S*: the arousal->surprise synapse (vs edge lesion)",
                                                     fp, fl)
        attribution["intero_null"] = attributable_to("X1 flips @S*: the ladder's latched arousal (vs intero-null)",
                                                     fp, fi)
        attribution["no_edge_pool"] = attributable_to("X1 flips @S*: the edge topology (vs the no-edge pool)",
                                                       fp, fne)
    checks["X1_functional_verdict_flip_at_marginal_strength"] = bool(x1)
    # integrity smokes
    checks["I1_edge_lesion_verdicts_equal_baseline"] = _verdicts_equal(les, base, probe)
    checks["I2_no_edge_pool_null"] = bool(_verdicts_equal(npos, nb, probe) and _verdicts_equal(nb, base, probe))
    checks["I3_intero_null_collapses"] = bool(s_star is not None and flips["pos"] > 0
                                              and fi <= int(np.floor(LESION_RATIO * flips["pos"])))
    checks["I4_byte_off_inert_at_rest"] = bool(
        base["contradict"][kP]["hz"] == nb["contradict"][kP]["hz"]
        and base["confirm"][kP]["hz"] == nb["confirm"][kP]["hz"]
        and all(v["byte_identical"] and v["same_answer"] for v in x5.values()))
    checks["I5_held_arousal_carries"] = bool(pos["arousal_rung_hz"] > 0 and neg["arousal_rung_hz"] > 0
                                             and base["arousal_rung_hz"] == 0.0)
    checks["I6_deterministic"] = bool(all(pos_again["contradict"][f"{S:g}"]["hz"] == pos["contradict"][f"{S:g}"]["hz"]
                                          for S in probe) and pos_again["confirm"][kP]["hz"] == pos["confirm"][kP]["hz"])
    checks["I7_ladder_scope_invariant"] = all(a == b for a, b in scope.values())
    c0 = base["confirm"][kP]["n_surprised"]
    checks["I8_production_strength_safety"] = bool(
        pos["confirm"][kP]["n_surprised"] == c0 and neg["confirm"][kP]["n_surprised"] == c0
        and _newly(base["contradict"][kP], pos["contradict"][kP])[1] == 0
        and _newly(base["contradict"][kP], neg["contradict"][kP])[1] == 0)
    curve = {lab: {S: r["contradict"][S]["mean_hz"] for S in r["contradict"]}
             for lab, r in (("a0", base), ("a+1", pos), ("a-1", neg))}
    details = {"x_gate": X_GATE, "s_star": s_star, "n_flip_required": need, "flips_at_Sstar": flips,
               "intero_null_flips": fi, "edge_lesion_flips": fl, "no_edge_pool_flips": fne,
               "neg_is_construction_duplicate_of_pos_at_Sstar": neg_dup, "attribution": attribution,
               "void_x": bool(void_x), "undefined_reason": undefined_reason, "x0_max_abs_dhz": float(x0_maxdhz),
               "production_strength_contradict_newly_flagged": {
                   "pos": _newly(base["contradict"][kP], pos["contradict"][kP])[0],
                   "neg": _newly(base["contradict"][kP], neg["contradict"][kP])[0]},
               "reported_contradict_mean_hz_curve_NO_GAIN_CLAIM": curve}
    return checks, details


def _surprise_and_ladder(pool, seed, organs=None):
    from research.runners.onebrain_affect_pool import PoolAffectLadder
    from research.runners.surprise_production_organ import SurpriseProductionOrgan
    sorg = organs["surprise"] if organs and "surprise" in organs else SurpriseProductionOrgan(seed=seed, shared=pool)
    lad = PoolAffectLadder(seed, shared=pool)
    return sorg, lad


# ─────────────────────────────────────────────────────────────────────────────────────────────
def calibrate(seed):
    """DIAGNOSTIC ONLY (does NOT set XEDGE_W, which is a hand-set constant): on a non-gate seed, the v2 instrument's
    newly-flagged count at S* and the 600 pA confirm false alarms, per effective arousal->surprise weight."""
    from research.runners.onebrain_affect_pool import build_affect_pool
    t0 = time.time()
    pool = build_affect_pool(seed, xedge=True, xedge_w=CAL_BUILD_W)
    sorg, lad = _surprise_and_ladder(pool, seed)
    sorg.ensure_built()
    print(f"[calibrate(diagnostic) seed {seed}] pool N={int(pool.bridge.cp_membrane_potential_v.shape[0])} "
          f"surprise thr={sorg.threshold:.3f} built in {time.time() - t0:.0f}s", flush=True)
    base = arousal_surprise_battery(pool, lad, sorg, 0.0)
    s_star = select_marginal_strength(base)
    print(f"  S* = {s_star} (a=0 frac per strength: "
          f"{ {k: v['frac'] for k, v in base['contradict'].items()} })", flush=True)
    probe = tuple(dict.fromkeys(s for s in (s_star, PROD_ASSERT_PA) if s is not None))
    rows = []
    for w in CAL_WEIGHTS:
        g = float(w) / CAL_BUILD_W
        row = {"w": w}
        for a in (1.0, -1.0):
            r = arousal_surprise_battery(pool, lad, sorg, a, strengths=probe, xedge_gain=g)
            key = "pos" if a > 0 else "neg"
            if s_star is not None:
                row[f"newly_{key}_at_Sstar"] = _newly(base["contradict"][f"{s_star:g}"],
                                                      r["contradict"][f"{s_star:g}"])[0]
            row[f"confirm_fa_{key}_600"] = r["confirm"][f"{PROD_ASSERT_PA:g}"]["n_surprised"]
            row[f"contra_meanhz_{key}_600"] = r["contradict"][f"{PROD_ASSERT_PA:g}"]["mean_hz"]
        rows.append(row)
        print(f"  w={w:<5} {row}", flush=True)
    return {"seed": seed, "diagnostic_only": True, "x_instrument": X_INSTRUMENT, "base": base, "s_star": s_star,
            "sweep": rows, "elapsed_s": round(time.time() - t0, 1)}


# ─────────────────────────────────────────────────────────────────────────────────────────────
def verify_seed(seed, arms=("M", "X")):
    from research.runners.onebrain_affect_pool import (
        affect_descriptors, AFFECT_DESCRIPTOR, AFFECT_KEY, PoolAffectLadder, PROD_SWEEP, build_affect_pool,
        XEDGE_W)
    from research.runners.onebrain_merge_framework import merge_organs, substrate_byte_identity
    from research.runners._onebrain_wave1_organread_verify import _isolated_read_one
    from research.runners._onebrain_wave3_organread_verify import _wave3_descriptors
    from research.runners.onebrain_wave3_pool_production import get_wave3_pool
    from research.runners.affect_production_organ import tone_level
    from research.runners._stageA_full_integration_derisk import LADDER_NEUTRAL_TOL
    t0 = time.time()
    descs = affect_descriptors()
    keys = [d.key for d in descs]
    carried = [k for k in keys if k != AFFECT_KEY]
    res = {"seed": int(seed), "xedge_w": float(XEDGE_W)}

    # ── the 12-organ pool WITHOUT the synapse (used by both arms) ──
    merged = build_affect_pool(seed, xedge=False)
    n_all = int(merged.bridge.cp_membrane_potential_v.shape[0])
    w_before = _ladder_weights(merged)
    R_m, A_m, organs_m = _reads_all(merged, descs, seed)
    w_after = _ladder_weights(merged)
    res["n_all_neurons"] = n_all
    print(f"[seed {seed}] 12-organ pool N={n_all} read in {time.time() - t0:.0f}s", flush=True)

    checks = {}
    if "M" in arms:
        # M1 co-residence
        core = merge_organs([AFFECT_DESCRIPTOR], seed, config_descriptors=descs, wire=True)
        c_reads, c_ans = _isolated_read_one(core, AFFECT_DESCRIPTOR, seed)
        d1, wk1, miss1 = _maxdelta(R_m[AFFECT_KEY], c_reads)
        checks["M1_affect_coresidence_byte_identical"] = bool(d1 == 0.0 and not miss1 and c_ans == A_m[AFFECT_KEY])
        res["M1"] = {"maxdelta": d1, "worst_key": wk1, "missing": miss1}
        # M2 carried organs vs the shipped wave-3 pool
        ship = get_wave3_pool(seed)
        R_s, A_s, _ = _reads_all(ship, _wave3_descriptors(), seed)
        m2 = {}
        for k in carried:
            dd, wk, miss = _maxdelta(R_m[k], R_s[k])
            m2[k] = {"maxdelta": dd, "worst_key": wk, "missing": miss,
                     "read_byte_identical": bool(dd == 0.0 and not miss), "answer_same": bool(A_m[k] == A_s[k])}
        checks["M2_carried_11_read_byte_identical_and_answer_same"] = all(
            v["read_byte_identical"] and v["answer_same"] for v in m2.values())
        res["M2"] = m2
        # M3 affect answer vs today's standalone production ladder
        std = PoolAffectLadder(seed, shared=None)
        std_diffs = [std.read_differential(a)["differential"] for a in PROD_SWEEP]
        std_levels = tuple(int(tone_level(d)) for d in std_diffs)
        pool_diffs = [R_m[AFFECT_KEY][f"diff[{a:+.1f}]"] for a in PROD_SWEEP]
        checks["M3_affect_tone_levels_equal_standalone"] = bool(std_levels == tuple(A_m[AFFECT_KEY]))
        res["M3"] = {"sweep": list(PROD_SWEEP), "pool_diffs": pool_diffs, "standalone_diffs": std_diffs,
                     "pool_levels": list(A_m[AFFECT_KEY]), "standalone_levels": list(std_levels)}
        # M4 alive
        r = R_m[AFFECT_KEY]
        signs = all((r[f"diff[{a:+.1f}]"] > 0) == (a > 0) and r[f"diff[{a:+.1f}]"] != 0.0
                    for a in PROD_SWEEP if abs(a) >= 0.5)
        neutral = abs(r["diff[+0.0]"]) < LADDER_NEUTRAL_TOL
        les = r["diff_lesion[+0.7]"] == 0.0
        il = (abs(r["diff_intero_lesion[+1.0]"]) <= INTERO_COLLAPSE * abs(r["diff[+1.0]"])
              and abs(r["diff_intero_lesion[-1.0]"]) <= INTERO_COLLAPSE * abs(r["diff[-1.0]"]))
        checks["M4_affect_alive_on_pool"] = bool(signs and neutral and les and il)
        res["M4"] = {"signs": signs, "neutral": neutral, "readout_lesion_zero": les, "intero_lesion_collapses": il}
        # M5 frozen
        checks["M5_ladder_weights_frozen"] = bool(w_before.shape == w_after.shape and w_before.size > 0
                                                  and float(np.max(np.abs(w_before - w_after))) == 0.0)
        res["M5"] = {"n_ladder_synapses": int(w_before.size)}
        # M6 legacy discriminator
        leg_m = merge_organs(descs, seed, legacy=True)
        leg_c = merge_organs([AFFECT_DESCRIPTOR], seed, config_descriptors=descs, legacy=True)
        lbi = substrate_byte_identity(leg_m, leg_c, list(AFFECT_DESCRIPTOR.regions))
        checks["M6_legacy_discriminator_diverges"] = bool(lbi["maxerr"] > 0.0)
        res["M6"] = {"legacy_maxerr": lbi["maxerr"]}
        # M7 determinism
        lad2 = PoolAffectLadder(seed, shared=merged)
        again = [lad2.read_differential(a)["differential"] for a in PROD_SWEEP]
        checks["M7_affect_read_deterministic"] = bool(again == pool_diffs)
        print(f"[seed {seed}] ARM M: " + " ".join(f"{k.split('_')[0]}={v}" for k, v in checks.items()), flush=True)

    if "X" in arms:
        # X-arm pool: the SAME 12 organs + the arousal->surprise synapse (v2 instrument, see the docstring)
        res["x_instrument"] = X_INSTRUMENT
        xpool = build_affect_pool(seed, xedge=True)
        R_x, A_x, organs_x = _reads_all(xpool, descs, seed)
        x5_reads = {}
        for k in keys:
            dd, wk, miss = _maxdelta(R_x[k], R_m[k])
            x5_reads[k] = {"maxdelta": dd, "worst_key": wk, "same_answer": bool(A_x[k] == A_m[k]),
                           "byte_identical": bool(dd == 0.0 and not miss)}
        sx, lx = _surprise_and_ladder(xpool, seed, organs_x)
        sm, lm = _surprise_and_ladder(merged, seed, organs_m)
        P = PROD_ASSERT_PA
        tx = time.time()
        # X0 reference + the a=0 baseline (S* is selected from the a=0 battery ONLY, before any arousal read)
        prod = production_path_battery(xpool, sx)
        base = arousal_surprise_battery(xpool, lx, sx, 0.0)
        s_star = select_marginal_strength(base)
        print(f"[seed {seed}] X: thr={base['threshold']:.3f} S*={s_star} a=0 frac/strength="
              f"{ {k: v['frac'] for k, v in base['contradict'].items()} } ({time.time() - tx:.0f}s)", flush=True)
        probe = tuple(dict.fromkeys(s for s in (s_star, P) if s is not None))
        pos = arousal_surprise_battery(xpool, lx, sx, 1.0)          # full grid (reported f-I data)
        neg = arousal_surprise_battery(xpool, lx, sx, -1.0)
        pos_again = arousal_surprise_battery(xpool, lx, sx, 1.0, strengths=probe)
        les = arousal_surprise_battery(xpool, lx, sx, 1.0, strengths=probe, xedge_lesion=True)
        inull = arousal_surprise_battery(xpool, lx, sx, 1.0, strengths=probe, intero_lesion=True)
        nb = arousal_surprise_battery(merged, lm, sm, 0.0, strengths=probe)
        npos = arousal_surprise_battery(merged, lm, sm, 1.0, strengths=probe)
        # I7 scope invariance of the ladder itself
        scope = {f"{a:+.1f}": (lx.read_differential(a, ou_scope="affect")["differential"],
                               lx.read_differential(a, ou_scope="all")["differential"]) for a in (1.0, -1.0)}
        raw = {"production_path": prod, "base": base, "pos": pos, "neg": neg, "pos_again": pos_again,
               "lesion": les, "intero_null": inull, "noedge_base": nb, "noedge_pos": npos,
               "ladder_scope_invariance": scope, "x5_reads": x5_reads}
        xchecks, xdet = score_x_arm(raw)
        checks.update(xchecks)
        res["X"] = {**xdet, **raw, "x_elapsed_s": round(time.time() - tx, 1)}
        print(f"[seed {seed}] ARM X(gate {X_GATE}): S*={xdet['s_star']} need={xdet['n_flip_required']} "
              f"flips={xdet['flips_at_Sstar']} intero-null={xdet['intero_null_flips']} "
              f"attribution={xdet['attribution']} x0_max|dHz|={xdet['x0_max_abs_dhz']:.4f} | "
              + " ".join(f"{k.split('_')[0]}={v}" for k, v in checks.items() if k[0] in "XI"), flush=True)

    res["checks"] = checks
    res["GO"] = bool(checks) and all(checks.values())
    res["elapsed_s"] = round(time.time() - t0, 1)
    print(f"[seed {seed}] GO={res['GO']} ({res['elapsed_s']}s)", flush=True)
    return res


_REQUIRED = (tuple(f"M{i}" for i in range(1, 8)) + ("X0", "X1")
             + tuple(f"I{i}" for i in range(1, 9)))


def aggregate(paths):
    """Merge per-arm files (verify_M_seed*.json + verify_X_seed*.json, or combined MX files) PER SEED: a seed is
    GO only when every one of M1..M7, X0..X1 and I1..I8 is present AND true (a missing arm is NOT a pass).
    Arm-X checks from a record WITHOUT the current `x_instrument` (the superseded v1 gate) are DROPPED, not scored.
    A current-instrument arm-X record is RE-SCORED from its raw batteries with `score_x_arm` (gate `X_GATE`); the
    checks the file stored are ignored, so a file written under an earlier scoring rule is judged by this one. A
    current-instrument record missing any raw battery is UNSCORABLE (its X/I checks are then missing = not a pass)."""
    by_seed, superseded, undefined, unscorable, flips = {}, [], {}, [], {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        if d.get("mode") != "verify":
            continue
        for r in d.get("per_seed", []):
            s = int(r["seed"])
            chk = dict(r.get("checks", {}))
            stored_x = [k for k in chk if k[0] in "XI"]
            chk = {k: v for k, v in chk.items() if k[0] not in "XI"}
            if r.get("x_instrument") != X_INSTRUMENT:
                if stored_x:
                    superseded.append((p, s, stored_x))
            else:
                X = r.get("X") or {}
                if all(k in X for k in _X_RAW_KEYS):
                    xchk, xdet = score_x_arm(X)
                    chk.update(xchk)
                    flips[s] = (xdet["flips_at_Sstar"]["pos"], xdet["n_flip_required"])
                    if xdet["undefined_reason"]:
                        undefined[s] = xdet["undefined_reason"]
                else:
                    unscorable.append((p, s, [k for k in _X_RAW_KEYS if k not in X]))
            by_seed.setdefault(s, {}).update(chk)
    for p, s, dropped in superseded:
        print(f"  (superseded v1 arm-X checks IGNORED: {p} seed {s}: {len(dropped)} checks)")
    for p, s, miss in unscorable:
        print(f"  (UNSCORABLE arm-X record, raw batteries missing {miss}: {p} seed {s} -- X/I checks MISSING)")
    n_go = 0
    for s in sorted(by_seed):
        chk = by_seed[s]
        present = {k.split("_")[0] for k in chk}
        missing = [c for c in _REQUIRED if c not in present]
        failed = [k for k, v in chk.items() if not v]
        go = not missing and not failed
        n_go += go
        extra = f" X1 UNDEFINED: {undefined[s]}" if s in undefined else ""
        fl = f" X1 a=+1 flips@S*={flips[s][0]} (need {flips[s][1]})" if s in flips else ""
        print(f"  seed {s}: GO={go} failed={failed} missing={missing}{fl}{extra}")
    missing_seeds = sorted(set(SEEDS) - set(by_seed))
    print(f"affect->one-brain-pool verify (arm-X gate {X_GATE}): {n_go}/{len(by_seed)} seeds GO; "
          f"missing seeds {missing_seeds}")
    undefined_if_empty("affect->pool 6-seed GO", len(by_seed), n_go, len(SEEDS))
    all_go = bool(not missing_seeds and n_go == len(SEEDS))
    print(f"ALL-GO (6/6, every M1-M7 + X0-X1 + I1-I8): {all_go}")
    return all_go


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help=f"default: the 6 gate seeds; with --calibrate: the non-gate seed {CAL_SEED_DEFAULT}")
    ap.add_argument("--arms", default="MX", help="M, X or MX")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--aggregate", nargs="+", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.aggregate:
        paths = []
        for p in a.aggregate:
            paths.extend(sorted(glob.glob(p)) or [p])
        ok = aggregate(paths)
        sys.exit(0 if ok else 1)
    if a.calibrate:
        seeds = a.seeds or [CAL_SEED_DEFAULT]
        out = {"mode": "calibrate", "diagnostic_only": True, "per_seed": [calibrate(s) for s in seeds]}
    else:
        seeds = a.seeds or list(SEEDS)
        out = {"mode": "verify", "arms": a.arms, "per_seed": [verify_seed(s, arms=tuple(a.arms)) for s in seeds],
               "preregistered": {"x_instrument": X_INSTRUMENT, "x_gate": X_GATE, "LESION_RATIO": LESION_RATIO,
                                 "INTERO_COLLAPSE": INTERO_COLLAPSE, "ASSERT_GRID": list(ASSERT_GRID),
                                 "MARGINAL_FRAC": MARGINAL_FRAC, "FLIP_FRAC": FLIP_FRAC, "MIN_FLIPS": MIN_FLIPS}}
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2, default=float))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
