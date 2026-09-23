"""PROSPECTIVE-MEMORY LIVE CLIFF DETECTOR -- v2 (fix round, 2026-09-23).

v1 (commit f30ab4666, artifact `research/findings/raw/_pmem_live_cliff_detector.json`, verdict UNDEFINED 4/6,
discrimination anti-cheat FAILED) was rejected by adversarial review for four reasons, each fixed here:

  1. CIRCULAR READ-OUT. v1 reported, for a climbing seed, the RUNNING BEST `(g_best, rel_best)` -- an arg-max of
     the evaluation metric over the visited gains -- and gated it against the static table (itself an arg-max
     over a coarser subset grid), so "meets-or-beats static" held by construction. v2 reports the CONTROLLER'S
     OWN SETTLED STATE: the last iterate of the closed loop, with the `rel` MEASURED at that iterate. There is no
     running best, no revert-to-remembered value, and no comparison that an arg-max wins by construction. The
     static-table comparison survives only as a DESCRIPTIVE number (the cost of not arg-maxing), never gating.
  2. CONSTANTS FIT IN-SAMPLE. v1's DROP_THRESH / DECLINE_PATIENCE were sized from seed 44's own dip and cliff
     and seed 100's decline, then evaluated on those seeds. v2 replaces both with a textbook one-sided CUSUM
     change detector (Page 1954) whose only data-dependent quantity -- the jitter scale sigma -- is estimated
     robustly (1.4826 x MAD) from lattice scans of CALIBRATION seeds `SEEDS_CALIB` that are disjoint from every
     evaluation seed. The multipliers (slack k = 0.5 sigma, decision interval h = 5 sigma) are the conventional
     control-chart defaults, fixed in the pre-registration, never tuned. Primary evaluation is on HELD-OUT seeds
     `SEEDS_HELDOUT` never looked at before this run; the canonical six are reported as IN-SAMPLE.
  3. NO NULL. v2 tests the detector itself against a permutation null: on each seed's open-loop lattice scan,
     the detector's alarms are scored (+1 an alarm the scan never recovers from, -1 one it does recover from),
     and the same detector is run on 1000 within-seed shuffles of the same scans. The unit of analysis is the
     SEED (one scan per independent substrate), never serially dependent steps of one trajectory.
  4. BYTE-IDENTICAL INFERRED. v2's `--default-off-compare` builds the PRODUCTION organ
     (`ProspectiveMemoryOrgan`) with the flag unset in this tree AND in a `git archive` extraction of the
     PINNED pre-change SHA `PINNED_PRE_CHANGE_SHA`, runs one scripted intention/hold/cue session in each, and
     exact-compares the serialized outputs (plus a negative control that MUST differ, so the compare can fail).

HOST-SHORTCUT DECLARATION (CLAUDE.md BRAIN-BASED ONLY; docs/TERMS.md). The controller and the change detector
are HOST ARITHMETIC reading the task's own coincidence read-out (`rel`) and setting a scalar synaptic gain. That
is a host set-point controller, NOT a brain mechanism, and nothing here is credited to the brain. The biology
it stands in for (slow homeostatic synaptic scaling toward a set point, Turrigiano 2011; a sliding modification
threshold, Lee & Kirkwood 2019) is an ANALOGY for the target, not something this code implements: the real
processes are per-neuron and driven by the neuron's own activity, not by a task-level output metric.

THE LAW (identical for every seed; no seed-keyed branch):
  g_{i+1} = clip(g_i + clip(K_I * (REL_TARGET - rel_i), -MAX_STEP, +MAX_STEP), G_FLOOR, ceiling_i)
  CUSUM on climbing moves only:  S_i = max(0, S_{i-1} + (rel_{i-1} - rel_i) - k)
  when S_i > h: ALARM -> ceiling_{i} = the last gain at which S was 0 (Page's change-point estimate), S := 0,
                and the SAME integral law continues under the lowered ceiling.
  Converged when |g_{i+1} - g_i| < G_TOL for CONVERGE_STREAK consecutive iterations.
  REPORTED = (g_last, rel measured at g_last) -- the settled state a running homeostat would actually sit at.

  SIM_BACKEND=numpy .venv/bin/python -m research.runners._pmem_live_cliff_detector_derisk --selftest
  ... --calibrate --seed 7            # one calibration lattice scan (seeds 7..12 only)
  ... --freeze                        # sigma, k, h from the calibration scans -> frozen_constants.json
  ... --eval --seed 200               # one evaluation seed (held-out 200..205 or canonical six)
  ... --aggregate                     # the pre-registered gate over all per-seed files + the null
  ... --default-off-compare           # organ exact-compare vs the pinned pre-change SHA
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._pmem_intention_latch_derisk import FIRE_THR   # noqa: E402  (imported, never re-typed)
from research.runners._operating_point_stabilizer_derisk import (     # noqa: E402
    FAC_G_DEFAULT, REL_TARGET,
)

V1_ARTIFACT = os.path.join(_REPO, "research", "findings", "raw", "_pmem_live_cliff_detector.json")
V2_DIR = os.path.join(_REPO, "research", "findings", "raw", "_pmem_live_cliff_detector_v2")
FROZEN = os.path.join(V2_DIR, "frozen_constants.json")
DEFAULT_OFF_ARTIFACT = os.path.join(V2_DIR, "default_off_compare.json")
VERDICT_ARTIFACT = os.path.join(V2_DIR, "verdict.json")
STATIC_STABILIZER_ARTIFACT = os.path.join(_REPO, "research", "findings", "raw",
                                          "_pmem_operating_point_stabilizer.json")

# ---- seed roles (frozen in the v2 pre-registration) ----
SEEDS_CANONICAL = (42, 43, 44, 100, 101, 102)   # IN-SAMPLE: the design was informed by these + the v1 run was seen
SEEDS_CALIB = (7, 8, 9, 10, 11, 12)              # detector constants are estimated from THESE ONLY
SEEDS_HELDOUT = (200, 201, 202, 203, 204, 205)   # PRIMARY evaluation; never measured before this run
CONSTANT_SOURCE_SEEDS = SEEDS_CALIB

# ---- the pre-change tree the default-off path is exact-compared against (origin/main when this branch was
# merged up for the fix round; the organ file there has no cliff-detector branch at all) ----
PINNED_PRE_CHANGE_SHA = "4c141b8e8413e39913b2ef72b13f18c3db656dc0"

LIVE_CLIFF_ENV = "BRAIN_PMEM_LIVE_CLIFF_DETECTOR"

# ---- controller constants: reused unchanged from the parent live homeostat (not re-tuned) ----
G_FLOOR = FAC_G_DEFAULT           # 6000.0
G_CEILING_CAP = 11000.0           # DECLARED: the top of the inherited stabilizer grid; it coincides with seed 101's
                                   # static pick, so it is in-sample-adjacent -- the held-out seeds judge it
K_I = 20000.0
MAX_STEP = 500.0
G_TOL = 25.0
CONVERGE_STREAK = 3
MAX_ITERS = 24
SETPOINT_TOL = 750.0
G_INIT_LOW = FAC_G_DEFAULT        # 6000.0
G_INIT_MID = 8000.0

# ---- the open-loop lattice the calibration + null scans use (the controller's own rate-limited step) ----
LATTICE = tuple(float(G_FLOOR + MAX_STEP * i) for i in range(int((G_CEILING_CAP - G_FLOOR) / MAX_STEP) + 1))

# ---- CUSUM multipliers: conventional control-chart defaults (Page 1954; the k = 0.5 sigma, h = 5 sigma design),
# fixed in the pre-registration BEFORE any calibration scan was measured -- never tuned ----
MAD_TO_SIGMA = 1.4826
CUSUM_K_SIGMA = 0.5
CUSUM_H_SIGMA = 5.0

# ---- permutation null ----
NULL_PERMS = 1000
NULL_RNG_SEED = 20260923
NULL_ALPHA = 0.05


# --------------------------------------------------------------------------------------------------------
# the live read (fresh self-consistent build; reused verbatim from the parent live homeostat)
# --------------------------------------------------------------------------------------------------------
def _measure_rel(seed: int, g: float) -> float:
    from research.runners._pmem_live_homeostat_derisk import _measure_rel as _m
    return _m(seed, g)


# --------------------------------------------------------------------------------------------------------
# PURE detector pieces (no brain build; unit-tested)
# --------------------------------------------------------------------------------------------------------
def cusum_step(S: float, drop: float, k: float) -> float:
    """One-sided CUSUM for a DOWNWARD shift of `rel` while gain rises: accumulate each step's drop minus slack."""
    return max(0.0, S + drop - k)


def detect_scan(rels, k: float, h: float) -> dict:
    """Run the CUSUM over an ordered open-loop scan (index = increasing gain). Returns the FIRST alarm and
    Page's change-point estimate (the last index at which S was 0 before the alarm)."""
    S = 0.0
    last_zero = 0
    for i in range(1, len(rels)):
        S = cusum_step(S, rels[i - 1] - rels[i], k)
        if S == 0.0:
            last_zero = i
        if S > h:
            return {"alarm": True, "alarm_idx": i, "onset_idx": last_zero}
    return {"alarm": False, "alarm_idx": None, "onset_idx": None}


def alarm_score(rels, det: dict):
    """+1 if the scan never recovers to the onset level after the alarm (a genuine non-recovering drop),
    -1 if some later point recovers to it (a false alarm), 0 if no alarm or no later point to judge by."""
    if not det["alarm"]:
        return 0
    later = rels[det["alarm_idx"] + 1:]
    if not later:
        return 0
    return 1 if max(later) < rels[det["onset_idx"]] else -1


def detector_statistic(scans, k: float, h: float) -> int:
    return sum(alarm_score(r, detect_scan(r, k, h)) for r in scans)


def null_test(scans, k: float, h: float, n_perm: int = NULL_PERMS, rng_seed: int = NULL_RNG_SEED) -> dict:
    """Permutation null: the SAME detector on within-seed shuffles of the SAME open-loop scans. Each scan is one
    independent substrate (a seed), so the seed is the exchangeable unit; steps within a scan are never treated
    as independent samples. UNDEFINED (p=None) if the detector raises no alarm at all on the real scans -- a
    detector that never fired has shown nothing, which is not the same as passing."""
    scans = [list(map(float, s)) for s in scans]
    dets = [detect_scan(s, k, h) for s in scans]
    n_alarm = sum(1 for d in dets if d["alarm"])
    t_real = sum(alarm_score(s, d) for s, d in zip(scans, dets))
    rng = random.Random(rng_seed)
    t_null = []
    for _ in range(n_perm):
        tot = 0
        for s in scans:
            p = s[:]
            rng.shuffle(p)
            tot += alarm_score(p, detect_scan(p, k, h))
        t_null.append(tot)
    if n_alarm == 0:
        return {"defined": False, "p": None, "t_real": t_real, "n_alarm_real": 0, "n_scans": len(scans),
                "t_null_mean": statistics.fmean(t_null) if t_null else None,
                "note": "the detector raised no alarm on any real scan -- its specificity is UNDEFINED"}
    ge = sum(1 for t in t_null if t >= t_real)
    return {"defined": True, "p": (1 + ge) / (1 + n_perm), "t_real": t_real, "n_alarm_real": n_alarm,
            "n_scans": len(scans), "t_null_mean": round(statistics.fmean(t_null), 4),
            "t_null_q95": sorted(t_null)[int(0.95 * (len(t_null) - 1))], "n_perm": n_perm,
            "rng_seed": rng_seed, "per_scan": [{"alarm": d["alarm"], "alarm_idx": d["alarm_idx"],
                                                "onset_idx": d["onset_idx"], "score": alarm_score(s, d)}
                                               for s, d in zip(scans, dets)]}


def freeze_constants(calib_scans: dict) -> dict:
    """sigma = 1.4826 x MAD of the pooled one-step differences of the CALIBRATION scans (robust: a handful of
    genuine cliffs cannot inflate it); k = CUSUM_K_SIGMA x sigma, h = CUSUM_H_SIGMA x sigma."""
    seeds = sorted(int(s) for s in calib_scans)
    overlap = set(seeds) & (set(SEEDS_CANONICAL) | set(SEEDS_HELDOUT))
    if overlap:
        raise ValueError(f"calibration seeds overlap evaluation seeds: {sorted(overlap)}")
    diffs = []
    for s in seeds:
        r = [float(x) for x in calib_scans[s]]           # ordered by LATTICE (increasing gain)
        diffs += [r[i] - r[i - 1] for i in range(1, len(r))]
    med = statistics.median(diffs)
    mad = statistics.median([abs(d - med) for d in diffs])
    sigma = MAD_TO_SIGMA * mad
    if sigma <= 0:
        raise ValueError("degenerate jitter scale (MAD = 0) -- the calibration scans carry no step variation")
    return {"seeds": seeds, "n_diffs": len(diffs), "median_diff": round(med, 6), "mad": round(mad, 6),
            "sigma": round(sigma, 6), "cusum_k": round(CUSUM_K_SIGMA * sigma, 6),
            "cusum_h": round(CUSUM_H_SIGMA * sigma, 6), "k_sigma": CUSUM_K_SIGMA, "h_sigma": CUSUM_H_SIGMA,
            "mad_to_sigma": MAD_TO_SIGMA, "lattice": list(LATTICE)}


def load_frozen() -> dict:
    if not os.path.exists(FROZEN):
        raise RuntimeError(f"frozen detector constants missing ({os.path.relpath(FROZEN, _REPO)}) -- run --freeze "
                           "on the calibration scans first; the controller refuses to run on unfrozen constants")
    with open(FROZEN) as fh:
        return json.load(fh)


# --------------------------------------------------------------------------------------------------------
# THE CONTROLLER (v2): the reported quantity is the settled state, never an arg-max
# --------------------------------------------------------------------------------------------------------
def run_live_cliff_homeostat(seed: int, g_init: float, cusum_k: float | None = None, cusum_h: float | None = None,
                             measure=None, target: float = REL_TARGET, max_iters: int = MAX_ITERS,
                             k_i: float = K_I, max_step: float = MAX_STEP, g_floor: float = G_FLOOR,
                             g_ceiling_cap: float = G_CEILING_CAP, tol: float = G_TOL,
                             streak_need: int = CONVERGE_STREAK) -> dict:
    if cusum_k is None or cusum_h is None:
        fr = load_frozen()
        cusum_k = fr["cusum_k"] if cusum_k is None else cusum_k
        cusum_h = fr["cusum_h"] if cusum_h is None else cusum_h
    meas = measure if measure is not None else (lambda s, g: _measure_rel(s, g))
    g = float(g_init)
    ceiling = float(g_ceiling_cap)
    S = 0.0
    last_zero_g = g
    prev_g = prev_rel = None
    streak = 0
    traj, events = [], []
    converged = False
    reason = "max_iters exhausted (%d)" % max_iters
    for i in range(max_iters):
        rel = meas(seed, g)
        climbing = prev_g is not None and g > prev_g + 1e-9
        alarm = False
        if climbing:
            S = cusum_step(S, prev_rel - rel, cusum_k)
            if S == 0.0:
                last_zero_g = g
            if S > cusum_h:
                alarm = True
        elif prev_g is None:
            last_zero_g = g
        if prev_g is not None:
            streak = streak + 1 if abs(g - prev_g) < tol else 0
        traj.append({"iter": i, "fac_g": round(g, 1), "rel": rel, "cusum_S": round(S, 6),
                     "ceiling": round(ceiling, 1), "streak": streak, "alarm": alarm})
        if alarm:
            new_ceiling = max(g_floor, min(ceiling, last_zero_g))
            events.append({"iter": i, "at_fac_g": round(g, 1), "at_rel": rel, "cusum_S": round(S, 6),
                           "new_ceiling": round(new_ceiling, 1)})
            ceiling = new_ceiling
            S = 0.0
            streak = 0
        elif prev_g is not None and streak >= streak_need:
            at = "ceiling" if abs(g - ceiling) < 1e-6 else ("floor" if abs(g - g_floor) < 1e-6 else None)
            reason = ("pinned at %s (fac_g=%.0f)" % (at, g)) if at else "settled at interior fac_g=%.0f" % g
            converged = True
            break
        prev_g, prev_rel = g, rel
        delta = max(-max_step, min(max_step, k_i * round(target - rel, 4)))
        g = max(g_floor, min(ceiling, g + delta))
    last = traj[-1]
    return {"seed": seed, "g_init": g_init, "converged": converged, "reason": reason, "n_iters": len(traj),
            "final_fac_g": last["fac_g"], "final_rel": last["rel"], "final_ceiling": round(ceiling, 1),
            "cusum_k": cusum_k, "cusum_h": cusum_h, "alarms": events, "trajectory": traj,
            "read_out": "settled state (last iterate + its own measured rel); no running best"}


# --------------------------------------------------------------------------------------------------------
# the per-seed jobs
# --------------------------------------------------------------------------------------------------------
def _scan(seed: int) -> dict:
    return {str(g): _measure_rel(seed, g) for g in LATTICE}


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    except Exception:  # noqa: BLE001  (pool revision dirs are rsync copies, not checkouts)
        return os.path.basename(os.path.normpath(_REPO))


def _sha256_file(p: str) -> str:
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)


def job_calibrate(seed: int) -> int:
    if seed not in SEEDS_CALIB:
        print(f"⛔ seed {seed} is not a calibration seed {SEEDS_CALIB}", flush=True)
        return 2
    t0 = time.time()
    scan = _scan(seed)
    out = {"seed": seed, "role": "calibration", "lattice": list(LATTICE), "scan": scan,
           "elapsed_s": round(time.time() - t0, 1), "revision": _git_sha()}
    p = os.path.join(V2_DIR, f"calib_s{seed}.json")
    _write_json(p, out)
    print(f"[calibrate] seed {seed}: {scan} -> {p}", flush=True)
    return 0


def job_freeze() -> int:
    scans = {}
    for s in SEEDS_CALIB:
        p = os.path.join(V2_DIR, f"calib_s{s}.json")
        if not os.path.exists(p):
            print(f"⛔ missing calibration scan {p}", flush=True)
            return 2
        with open(p) as fh:
            d = json.load(fh)
        scans[s] = [d["scan"][str(g)] for g in LATTICE]
    fr = freeze_constants(scans)
    fr["calib_file_sha256"] = {str(s): _sha256_file(os.path.join(V2_DIR, f"calib_s{s}.json")) for s in SEEDS_CALIB}
    fr["frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _write_json(FROZEN, fr)
    print(json.dumps(fr, indent=2), flush=True)
    return 0


def job_eval(seed: int, determinism: bool = True) -> int:
    import research.runners._pmem_facilitation_derisk as F
    from research.runners._operating_point_stabilizer_derisk import _frozen_silence
    if seed not in SEEDS_HELDOUT and seed not in SEEDS_CANONICAL:
        print(f"⛔ seed {seed} is neither held-out {SEEDS_HELDOUT} nor canonical {SEEDS_CANONICAL}", flush=True)
        return 2
    fr = load_frozen()
    k, h = fr["cusum_k"], fr["cusum_h"]
    t0 = time.time()
    low = run_live_cliff_homeostat(seed, G_INIT_LOW, k, h)
    mid = run_live_cliff_homeostat(seed, G_INIT_MID, k, h)
    rerun_identical = None
    if determinism:
        again = run_live_cliff_homeostat(seed, G_INIT_LOW, k, h)
        rerun_identical = (again["trajectory"] == low["trajectory"] and again["alarms"] == low["alarms"])
    settled_g = low["final_fac_g"]
    lb = F._n3_load_bearing(seed, fac_on=True, N=3, fac_g=settled_g, fac_U=F.FAC_U,
                            fac_tau_F_steps=F.FAC_TAU_F_STEPS)
    froz = _frozen_silence(seed, settled_g)
    fails = [c for c, v in froz["clauses"].items() if not v]
    scan = _scan(seed)
    default_rel = low["trajectory"][0]["rel"]          # measured at g = G_FLOOR = the shipped constant
    visited = [r["fac_g"] for r in low["trajectory"] + mid["trajectory"]]
    static_margin = None
    if seed in SEEDS_CANONICAL and os.path.exists(STATIC_STABILIZER_ARTIFACT):
        with open(STATIC_STABILIZER_ARTIFACT) as fh:
            static_margin = float(json.load(fh)["per_seed"][str(seed)]["stabilized_margin"])
    out = {
        "seed": seed, "role": "heldout" if seed in SEEDS_HELDOUT else "canonical_in_sample",
        "frozen_constants_sha256": _sha256_file(FROZEN), "cusum_k": k, "cusum_h": h,
        "low_init": low, "mid_init": mid, "rerun_identical": rerun_identical,
        "climber": default_rel < REL_TARGET,
        "settled_fac_g": settled_g, "settled_rel_measured": low["final_rel"],
        "converged_both": bool(low["converged"] and mid["converged"]),
        "same_setpoint": abs(low["final_fac_g"] - mid["final_fac_g"]) <= SETPOINT_TOL,
        "intact_rel": lb["intact_rel"], "intact_fired": lb["intact_fired"], "lesion_rel": lb["lesion_rel"],
        "lesion_fired": lb["lesion_fired"], "load_bearing": lb["load_bearing"],
        "frozen_passed": froz["passed"], "frozen_clause_fails": fails, "frozen_clauses": froz["clauses"],
        "default_rel": default_rel, "margin_default": round(default_rel - FIRE_THR, 4),
        "margin_settled": round(lb["intact_rel"] - FIRE_THR, 4),
        "domain_bounded": bool(visited) and min(visited) >= G_FLOOR and max(visited) <= G_CEILING_CAP,
        "max_fac_g_visited": max(visited), "scan": scan,
        "static_table_margin_DESCRIPTIVE": static_margin,
        "elapsed_s": round(time.time() - t0, 1), "revision": _git_sha(),
    }
    out["margin_positive"] = out["margin_settled"] > 0
    out["margin_ge_default"] = out["margin_settled"] >= out["margin_default"]
    p = os.path.join(V2_DIR, f"eval_s{seed}.json")
    _write_json(p, out)
    print(f"[eval] seed {seed}: settled g={settled_g} rel={low['final_rel']} margin={out['margin_settled']} "
          f"(default {out['margin_default']}) LB={lb['load_bearing']} silence_fails={fails or 'none'} "
          f"alarms low/mid={len(low['alarms'])}/{len(mid['alarms'])} -> {p}", flush=True)
    return 0


# --------------------------------------------------------------------------------------------------------
# the pre-registered gate (pure, unit-tested)
# --------------------------------------------------------------------------------------------------------
SEED_CRITERIA = ("converged_both", "same_setpoint", "margin_positive", "margin_ge_default", "load_bearing",
                 "silence_held", "domain_bounded", "rerun_identical")


def seed_pass(row: dict) -> dict:
    c = {"converged_both": bool(row.get("converged_both")), "same_setpoint": bool(row.get("same_setpoint")),
         "margin_positive": bool(row.get("margin_positive")), "margin_ge_default": bool(row.get("margin_ge_default")),
         "load_bearing": bool(row.get("load_bearing")), "silence_held": not row.get("frozen_clause_fails", ["?"]),
         "domain_bounded": bool(row.get("domain_bounded")), "rerun_identical": row.get("rerun_identical") is True}
    return {"criteria": c, "pass": all(c.values())}


def decide_gate(heldout_rows: dict, canonical_rows: dict, null: dict | None, default_off: dict | None) -> dict:
    """GO iff: every held-out seed passes every criterion; at least one held-out seed is a CLIMBER (else the
    controller was never exercised -> UNDEFINED); every canonical (in-sample) seed passes too; the detector's
    permutation null is defined and p < NULL_ALPHA; the default-off organ exact-compare asserted equality in the
    data AND its negative control differed. Missing inputs are UNDEFINED, never a pass."""
    undefined = []
    if set(heldout_rows) != set(SEEDS_HELDOUT):
        undefined.append(f"held-out results missing for {sorted(set(SEEDS_HELDOUT) - set(heldout_rows))}")
    if set(canonical_rows) != set(SEEDS_CANONICAL):
        undefined.append(f"canonical results missing for {sorted(set(SEEDS_CANONICAL) - set(canonical_rows))}")
    if heldout_rows and not any(r.get("climber") for r in heldout_rows.values()):
        undefined.append("no held-out seed is a climber -- the controller was never exercised on held-out data")
    if null is None or not null.get("defined"):
        undefined.append("detector permutation null UNDEFINED (no alarm on any held-out scan, or not run)")
    if default_off is None:
        undefined.append("default-off organ exact-compare artifact missing")
    elif default_off.get("negative_control_differs") is not True:
        undefined.append("the default-off compare's negative control did not differ -- the compare cannot fail")
    h = {s: seed_pass(r) for s, r in heldout_rows.items()}
    c = {s: seed_pass(r) for s, r in canonical_rows.items()}
    fails = []
    fails += [f"held-out s{s}: " + ",".join(k for k, v in p["criteria"].items() if not v)
              for s, p in sorted(h.items()) if not p["pass"]]
    fails += [f"canonical s{s}: " + ",".join(k for k, v in p["criteria"].items() if not v)
              for s, p in sorted(c.items()) if not p["pass"]]
    if null is not None and null.get("defined") and not (null["p"] < NULL_ALPHA):
        fails.append(f"detector null: p={null['p']:.4f} >= {NULL_ALPHA}")
    if default_off is not None and default_off.get("exact_equal_all") is not True:
        fails.append("default-off organ output NOT byte-identical to the pinned pre-change SHA")
    status = "UNDEFINED" if undefined else ("NO-GO" if fails else "GO")
    return {"status": status, "undefined_reasons": undefined, "fail_reasons": fails,
            "heldout": {str(s): p for s, p in h.items()}, "canonical": {str(s): p for s, p in c.items()},
            "n_heldout_pass": sum(p["pass"] for p in h.values()),
            "n_canonical_pass": sum(p["pass"] for p in c.values())}


def job_aggregate() -> int:
    from tools.verdict import Verdict

    def _load(seeds):
        rows = {}
        for s in seeds:
            p = os.path.join(V2_DIR, f"eval_s{s}.json")
            if os.path.exists(p):
                with open(p) as fh:
                    rows[s] = json.load(fh)
        return rows
    fr = load_frozen()
    frozen_sha = _sha256_file(FROZEN)
    held, canon = _load(SEEDS_HELDOUT), _load(SEEDS_CANONICAL)
    stale = [s for s, r in {**held, **canon}.items() if r.get("frozen_constants_sha256") != frozen_sha]
    null_h = null_test([[r["scan"][str(g)] for g in LATTICE] for _, r in sorted(held.items())],
                       fr["cusum_k"], fr["cusum_h"]) if held else None
    null_c = null_test([[r["scan"][str(g)] for g in LATTICE] for _, r in sorted(canon.items())],
                       fr["cusum_k"], fr["cusum_h"]) if canon else None
    default_off = None
    if os.path.exists(DEFAULT_OFF_ARTIFACT):
        with open(DEFAULT_OFF_ARTIFACT) as fh:
            default_off = json.load(fh)
    dec = decide_gate(held, canon, null_h, default_off)
    vd = Verdict("pmem_live_cliff_detector_v2")
    vd.require("every eval file was produced under the frozen constants now on disk", not stale, expect=True)
    vd.require("all held-out + canonical per-seed results present",
               len(held) == len(SEEDS_HELDOUT) and len(canon) == len(SEEDS_CANONICAL), expect=True)
    vd.require("detector permutation null defined on the held-out scans",
               bool(null_h and null_h.get("defined")), expect=True)
    vd.require("default-off compare present and able to fail (negative control differs)",
               bool(default_off and default_off.get("negative_control_differs") is True), expect=True)
    vd.require("at least one held-out climber (controller exercised)",
               any(r.get("climber") for r in held.values()) if held else None, expect=True)
    vd.disabled("STDP / long-term Hebbian LTP / OU-noise", "identical scope to the parent facilitation + live "
                "homeostat de-risks")
    decided = vd.decide(dec["status"] == "GO")
    status = decided["status"] if decided["status"] == "UNDEFINED" else dec["status"]
    desc = {str(s): {"settled_margin": r["margin_settled"], "static_margin": r["static_table_margin_DESCRIPTIVE"],
                     "settled_minus_static": (round(r["margin_settled"] - r["static_table_margin_DESCRIPTIVE"], 4)
                                              if r.get("static_table_margin_DESCRIPTIVE") is not None else None)}
            for s, r in sorted(canon.items())}
    summary = {"probe": "pmem_live_cliff_detector_v2", "verdict_status": status, "gate": dec,
               "stale_eval_files": stale, "frozen_constants": fr, "frozen_constants_sha256": frozen_sha,
               "null_heldout_PRIMARY": null_h, "null_canonical_in_sample_DESCRIPTIVE": null_c,
               "default_off_compare": default_off,
               "static_table_comparison_DESCRIPTIVE_not_gating": desc,
               "per_seed_summary": {str(s): {k2: r.get(k2) for k2 in (
                   "role", "climber", "settled_fac_g", "settled_rel_measured", "margin_settled", "margin_default",
                   "load_bearing", "frozen_clause_fails", "converged_both", "same_setpoint", "rerun_identical",
                   "domain_bounded", "max_fac_g_visited")} | {"alarms_low": len(r["low_init"]["alarms"]),
                                                             "alarms_mid": len(r["mid_init"]["alarms"])}
                   for s, r in sorted({**held, **canon}.items())},
               "preconditions": decided.get("preconditions"), "disabled_processes": decided.get("disabled_processes"),
               "HOST_SHORTCUT": ("the controller + CUSUM detector are host arithmetic on the task's own coincidence "
                                 "read-out setting a scalar synaptic gain; declared, not credited to the brain"),
               "v1_artifact_sha256": _sha256_file(V1_ARTIFACT) if os.path.exists(V1_ARTIFACT) else None}
    _write_json(VERDICT_ARTIFACT, summary)
    print(json.dumps({k2: summary[k2] for k2 in ("verdict_status",)} | {"fails": dec["fail_reasons"],
                     "undefined": dec["undefined_reasons"] + ([r for r in (decided.get("reasons") or [])])},
                     indent=2), flush=True)
    return 0


# --------------------------------------------------------------------------------------------------------
# default-off byte-identity, asserted in the data against a PINNED pre-change SHA
# --------------------------------------------------------------------------------------------------------
_ORGAN_SCENARIO = r"""
import json, os, sys
sys.path.insert(0, os.getcwd())
from research.runners.prospective_memory_production_organ import ProspectiveMemoryOrgan
o = ProspectiveMemoryOrgan(seed=int(os.environ["_SCEN_SEED"]))
form = o.form_intention("call mom", "the weather", ["weather"])
reads = [o.read_turn("let us talk about dinner plans"), o.read_turn("I painted the fence today"),
         o.read_turn("what is the weather like")]
out = {"form": form, "reads": reads, "calib": o.calib, "fac_g": getattr(o._pm, "_fac_g", None),
       "pm_class": type(o._pm).__name__}
sys.stdout.write("@@SCENARIO@@" + json.dumps(out, sort_keys=True, default=repr) + "\n")
"""


def _run_scenario(tree: str, env_over: dict, seed: int) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("BRAIN_PMEM")}
    env.update({"SIM_BACKEND": "numpy", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1", "SIM_NO_PROVENANCE": "1", "_SCEN_SEED": str(seed)})
    env.update(env_over)
    env["PYTHONPATH"] = tree
    p = subprocess.run([sys.executable, "-c", _ORGAN_SCENARIO], cwd=tree, env=env, capture_output=True, text=True,
                       timeout=3600)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("@@SCENARIO@@")]
    if p.returncode != 0 or not lines:
        raise RuntimeError(f"scenario failed in {tree} (rc={p.returncode}): {p.stderr[-2000:]}")
    return lines[-1][len("@@SCENARIO@@"):]


def job_default_off_compare(seed: int = 44) -> int:
    """Build the production organ with BRAIN_PMEM_LIVE_CLIFF_DETECTOR UNSET in (a) this tree and (b) a git-archive
    extraction of PINNED_PRE_CHANGE_SHA; run one scripted session in each; exact-compare the serialized output.
    Two configs: the shipped default, and BRAIN_PMEM_FACILITATION=1 (the branch whose elif chain this build edited).
    NEGATIVE CONTROL: this tree with BRAIN_PMEM_OP_STABILIZER=1 (a different fac_g at seed 44) MUST differ from the
    pinned facilitation run, or the compare is blind."""
    head = _git_sha()
    tmp = tempfile.mkdtemp(prefix="pmem_pinned_")
    try:
        arch = subprocess.run(["git", "archive", "--format=tar", PINNED_PRE_CHANGE_SHA], cwd=_REPO,
                              capture_output=True, check=True)
        subprocess.run(["tar", "-x", "-C", tmp], input=arch.stdout, check=True)
        organ_rel = "research/runners/prospective_memory_production_organ.py"
        pinned_has_flag = LIVE_CLIFF_ENV in open(os.path.join(tmp, organ_rel)).read()
        head_has_flag = LIVE_CLIFF_ENV in open(os.path.join(_REPO, organ_rel)).read()
        configs = {"shipped_default": {}, "facilitation_on": {"BRAIN_PMEM_FACILITATION": "1"}}
        per = {}
        for name, env_over in configs.items():
            a = _run_scenario(_REPO, env_over, seed)
            b = _run_scenario(tmp, env_over, seed)
            per[name] = {"env": env_over, "head_sha256": hashlib.sha256(a.encode()).hexdigest(),
                         "pinned_sha256": hashlib.sha256(b.encode()).hexdigest(), "exact_equal": a == b,
                         "head_output": json.loads(a)}
            print(f"[default-off] {name}: exact_equal={a == b}", flush=True)
        neg = _run_scenario(_REPO, {"BRAIN_PMEM_FACILITATION": "1", "BRAIN_PMEM_OP_STABILIZER": "1"}, seed)
        neg_differs = hashlib.sha256(neg.encode()).hexdigest() != per["facilitation_on"]["pinned_sha256"]
        out = {"pinned_pre_change_sha": PINNED_PRE_CHANGE_SHA, "head_sha": head, "seed": seed,
               "pinned_organ_has_cliff_flag": pinned_has_flag, "head_organ_has_cliff_flag": head_has_flag,
               "configs": per, "exact_equal_all": all(v["exact_equal"] for v in per.values()),
               "negative_control": {"env": {"BRAIN_PMEM_FACILITATION": "1", "BRAIN_PMEM_OP_STABILIZER": "1"},
                                    "sha256": hashlib.sha256(neg.encode()).hexdigest(),
                                    "fac_g": json.loads(neg).get("fac_g")},
               "negative_control_differs": neg_differs,
               "method": "git archive of the pinned SHA -> separate tree; one scripted organ session per tree in a "
                         "fresh process with every BRAIN_PMEM_* var stripped; sha256 + exact string compare"}
        _write_json(DEFAULT_OFF_ARTIFACT, out)
        print(f"[default-off] exact_equal_all={out['exact_equal_all']} negative_control_differs={neg_differs}",
              flush=True)
        return 0 if (out["exact_equal_all"] and neg_differs) else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------------------------------------
# production entry point (default-OFF; wired into prospective_memory_production_organ.py)
# --------------------------------------------------------------------------------------------------------
def live_cliff_detector_enabled() -> bool:
    v = os.environ.get(LIVE_CLIFF_ENV)
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "on")


_LIVE_CLIFF_CACHE: dict[int, float] = {}


def live_cliff_fac_g_for_seed(seed: int) -> float:
    """Converge this seed's gain live (from G_INIT_LOW, frozen constants) the first time it is needed in this
    process and return the SETTLED iterate. Refuses (raises) if the constants were never frozen."""
    seed = int(seed)
    if seed not in _LIVE_CLIFF_CACHE:
        _LIVE_CLIFF_CACHE[seed] = float(run_live_cliff_homeostat(seed, G_INIT_LOW)["final_fac_g"])
    return _LIVE_CLIFF_CACHE[seed]


# --------------------------------------------------------------------------------------------------------
def selftest() -> bool:
    def plant(table):
        return lambda s, g: table(g)
    k, h = 0.001, 0.01
    # (1) a rise, a mild wobble, then a HARD non-recovering drop at 9500: the controller must alarm, lower its
    # ceiling BELOW the drop, and SETTLE there; the reported rel must be the one measured at the settled gain.
    def cliff(g):
        return 0.20 + 0.000004 * (min(g, 9000) - 6000) - (0.05 if g >= 9500 else 0.0)
    r = run_live_cliff_homeostat(0, 6000.0, k, h, measure=plant(cliff))
    ok_cliff = (bool(r["alarms"]) and r["converged"] and r["final_fac_g"] < 9500
                and r["final_rel"] == cliff(r["final_fac_g"]))
    # (2) no cliff: monotone rise to the cap -> no alarm, pinned at the cap
    r2 = run_live_cliff_homeostat(0, 6000.0, k, h, measure=plant(lambda g: 0.20 + 0.000002 * (g - 6000)))
    ok_nocliff = (not r2["alarms"]) and r2["final_fac_g"] == G_CEILING_CAP
    # (3) the read-out is the settled iterate, never a remembered best
    ok_readout = all(x["final_fac_g"] == x["trajectory"][-1]["fac_g"] and x["final_rel"] == x["trajectory"][-1]["rel"]
                     for x in (r, r2))
    # (4) settled seed descends to the floor
    r3 = run_live_cliff_homeostat(0, 8000.0, k, h, measure=plant(lambda g: 0.34))
    ok_floor = r3["final_fac_g"] == G_FLOOR and not r3["alarms"]
    # (5) detector pure-function sanity
    ok_det = detect_scan([0.2, 0.21, 0.22, 0.15, 0.15], k, h)["alarm"] and not detect_scan(
        [0.2, 0.21, 0.22, 0.23], k, h)["alarm"]
    # (6) freeze refuses evaluation seeds
    try:
        freeze_constants({44: [0.2] * len(LATTICE)})
        ok_refuse = False
    except ValueError:
        ok_refuse = True
    # (7) gate cannot pass with nothing in it
    ok_gate = decide_gate({}, {}, None, None)["status"] == "UNDEFINED"
    checks = {"cliff -> alarm, ceiling below the drop, settled read-out measured there": ok_cliff,
              "no cliff -> no alarm, pinned at the cap": ok_nocliff,
              "reported (g, rel) is the last iterate, never a running best": ok_readout,
              "settled seed descends to the floor": ok_floor, "CUSUM fires on a drop, not on a rise": ok_det,
              "freeze_constants refuses an evaluation seed": ok_refuse,
              "empty gate is UNDEFINED, never GO": ok_gate,
              "calibration seeds disjoint from evaluation seeds": not (
                  set(SEEDS_CALIB) & (set(SEEDS_CANONICAL) | set(SEEDS_HELDOUT))),
              "held-out seeds disjoint from canonical": not (set(SEEDS_HELDOUT) & set(SEEDS_CANONICAL)),
              "default-off: flag unset -> disabled": (os.environ.pop(LIVE_CLIFF_ENV, None) or True)
              and live_cliff_detector_enabled() is False}
    print("=== LIVE CLIFF DETECTOR v2 SELF-TEST ===")
    for kk, v in checks.items():
        print("  [%s] %s" % ("PASS" if v else "FAIL", kk))
    ok = all(checks.values())
    print("VERDICT:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--no-determinism", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--default-off-compare", action="store_true")
    a = ap.parse_args()
    try:
        if a.selftest:
            return 0 if selftest() else 1
        if a.calibrate:
            return job_calibrate(a.seed)
        if a.freeze:
            return job_freeze()
        if a.eval:
            return job_eval(a.seed, determinism=not a.no_determinism)
        if a.aggregate:
            return job_aggregate()
        if a.default_off_compare:
            return job_default_off_compare(a.seed if a.seed is not None else 44)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
