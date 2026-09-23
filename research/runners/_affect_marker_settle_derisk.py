"""Affect-marker SETTLE de-risk (D1 lane "grow the robust core past 23", 2026-09-23).

THE DEFECT (measured, not guessed). `affect-marker-spiking-wta` is outside the robust core: RNG-isolated, it is
lesion-load-bearing on only 1/6 seeds (`research/findings/raw/_lbf_borderline_isolated/op_s*.json`). On the 'emo'
turn the felt mood lands at +0.064..+0.071 -- ON the +2/+3 register boundary (+0.07125) -- and the INTACT WTA reads
"no clean winner" (margin 0.033-0.044 < DEAD_MARGIN 0.05) on 5/6 seeds, so intact and lesion both emit no marker.

THE COMPANION PROCESSES (see `_affect_marker_wta_derisk.py` SETTLE block for the full reasoning + measurements):
  (1) deliberation time -- the 60 ms read stops a near-tie lateral-inhibition race before it resolves;
  (2) inter-turn rest   -- a 40 ms washout leaves the previous read's slow state in the circuit.
`BRAIN_AFFECT_MARKER_SETTLE=1` lets both run (DELIBERATION_MS, INTERTURN_REST_MS). Wiring/weights/drive/DEAD_MARGIN
are untouched. Default OFF -> byte-identical.

MODES
  --calibrate  : fixes DELIBERATION_MS from a PRE-REGISTERED CIRCUIT-LEVEL criterion on CALIBRATION seeds
                 (7,8,9,10,11 -- disjoint from the verification seeds). For each grid window (rest fixed at
                 INTERTURN_REST_MS) and each calibration seed, on warm readers (every read preceded by other reads):
                   C1 boundary commit : at every valence register BOUNDARY (5 midpoints) and the arousal boundary,
                                        a clean winner that is one of the two adjacent registers;
                   C2 center fidelity : at every valence CENTER (6) and arousal CENTER (2), the winner IS that register;
                   C3 spontaneous     : the lesion (baseline-only drive) reads NO clean winner (valence + arousal);
                   C4 repeatable      : reading the same boundary value twice gives the same winner.
                 The chosen window is the SHORTEST grid value passing C1-C4 on ALL calibration seeds. The 'emo' turn,
                 the load-bearing outcome and the verification seeds are never consulted.
  --verify     : the PRE-REGISTERED op-level GO gate on the 6 verification seeds (42 43 44 100 101 102), reading the
                 REAL ladder mood the 'emo' turn produces (`_lbf_borderline_operating_point._affect_marker`, the same
                 read the RNG-isolated diagnosis used), SETTLE OFF vs ON:
                   G1 load-bearing  : SETTLE ON predicted_load_bearing (intact lead != lesion lead, level != 0) on 6/6;
                   G2 lesion silent : SETTLE ON lesion lead == '' on 6/6 (the lesion removes the marker, it does not
                                      merely re-pick one);
                   G3 determinism   : two FRESH readers at the same seed give the identical intact lead, 6/6;
                   G4 shuffle       : mis-routing the drive (shuffle=True) changes the selected register on >= 4/6
                                      (a fixed random 6-permutation leaves the winner in place ~1/6 of the time);
                   G5 OFF baseline  : SETTLE OFF reproduces the committed diagnosis (load-bearing on exactly the same
                                      seeds as op_s*.json) -- the instrument still reads what it read before;
                   G6 attribution   : arms (deliberation only) / (rest only) / (both) reported per seed, so the
                                      verdict names WHICH companion process carries the flip (report-only).
                 GO iff G1..G5 hold. This is the CIRCUIT-level de-risk; the #1-metric verdict is the full-brain
                 `load_bearing_fraction --only affect-marker-spiking-wta --seed S` run with the flag ON, staged on the
                 mini-PC pool (robust core 23 -> 24 iff that reads load-bearing, null-clean and deterministic 6/6).

Brain-based: the SELECTION is the winner of a lateral-inhibition race read off `cp_firing_states`; the host sets
only the clock (how long the circuit runs / rests). No host formula chooses the marker.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")

import numpy as np

CAL_SEEDS = (7, 8, 9, 10, 11)
VERIFY_SEEDS = (42, 43, 44, 100, 101, 102)
DELIB_GRID = (60, 100, 150, 200, 300, 500, 800)   # 60 = today's WARMUP_STEPS (the un-settled baseline)
DIAG_DIR = "research/findings/raw/_lbf_borderline_isolated"


def _M():
    from research.runners import _affect_marker_wta_derisk as M
    return M


# ───────────────────────────────────────────── calibration ─────────────────────────────────────────────
def _calibrate_one(seed: int, delib: int, rest: int) -> dict:
    M = _M()
    C = M.MOOD_CENTERS
    vb = [(C[i] + C[i + 1]) / 2.0 for i in range(len(C) - 1)]
    A = M.AROUSAL_CENTERS
    ab = (A[0] + A[1]) / 2.0
    r = M.AffectMarkerWTA(seed=seed, settle=True, deliberation_ms=delib, rest_ms=rest)
    # warm the reader (a production reader has always read before)
    r.select_valence(0.0); r.select_arousal(0.0)
    c2_v = [r.select_valence(c)[0] == M.LEVEL_ORDER[i] for i, c in enumerate(C)]
    c2_a = [r.select_arousal(A[0])[0] is False, r.select_arousal(A[1])[0] is True]
    c1_v, c4_v = [], []
    for i, b in enumerate(vb):
        l1 = r.select_valence(b)[0]
        l2 = r.select_valence(b)[0]
        c1_v.append(l1 in (M.LEVEL_ORDER[i], M.LEVEL_ORDER[i + 1]))
        c4_v.append(l1 == l2)
    a1 = r.select_arousal(ab)[0]; a2 = r.select_arousal(ab)[0]
    c1_a, c4_a = a1 is not None, a1 == a2
    c3_v = r.select_valence(0.07, lesion=True)[0] is None
    c3_a = r.select_arousal(0.045, lesion=True)[0] is None
    res = {"C1_boundary_commit": bool(all(c1_v) and c1_a), "C2_center_fidelity": bool(all(c2_v) and all(c2_a)),
           "C3_spontaneous_silent": bool(c3_v and c3_a), "C4_repeatable": bool(all(c4_v) and c4_a),
           "detail": {"c1_v": c1_v, "c1_a": c1_a, "c2_v": c2_v, "c2_a": c2_a, "c3_v": c3_v, "c3_a": c3_a,
                      "c4_v": c4_v, "c4_a": c4_a}}
    res["pass"] = all(res[k] for k in ("C1_boundary_commit", "C2_center_fidelity", "C3_spontaneous_silent",
                                       "C4_repeatable"))
    return res


def calibrate(out: str) -> dict:
    M = _M()
    rest = M.INTERTURN_REST_MS
    table = {}
    chosen = None
    for d in DELIB_GRID:
        rows = {s: _calibrate_one(s, d, rest) for s in CAL_SEEDS}
        ok = all(r["pass"] for r in rows.values())
        table[d] = {"all_pass": ok, "per_seed": {str(s): r for s, r in rows.items()}}
        print(f"delib={d:4d} all_pass={ok} " + " ".join(
            f"s{s}:{''.join('1' if rows[s][k] else '0' for k in ('C1_boundary_commit','C2_center_fidelity','C3_spontaneous_silent','C4_repeatable'))}"
            for s in CAL_SEEDS), flush=True)
        if ok and chosen is None:
            chosen = d
    rec = {"probe": "affect_marker_settle_calibration", "calibration_seeds": list(CAL_SEEDS),
           "grid_ms": list(DELIB_GRID), "rest_ms": rest, "chosen_deliberation_ms": chosen,
           "module_constant_DELIBERATION_MS": M.DELIBERATION_MS,
           "constant_matches_calibration": chosen == M.DELIBERATION_MS,
           "criterion": "shortest grid window passing C1 boundary-commit, C2 center-fidelity, C3 spontaneous-silent, "
                        "C4 repeatable on ALL calibration seeds (disjoint from the verification seeds)",
           "table": {str(k): v for k, v in table.items()}}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"CHOSEN deliberation = {chosen} ms (module constant {M.DELIBERATION_MS}) -> {out}")
    return rec


# ───────────────────────────────────────────── verification ─────────────────────────────────────────────
def _ladder(seed: int) -> tuple:
    from research.runners import affect_production_organ as AO
    from research.runners._lbf_borderline_operating_point import EMO_TEXT
    from webapp.affect_drives_chat import AffectDrivesWorkspace
    appr = AO.appraise_text(EMO_TEXT)
    ws = AffectDrivesWorkspace(seed=seed)
    info = ws.observe(float(appr.get("valence", 0.0)), float(appr.get("arousal", 0.0)),
                      int(appr.get("n_hits", 0)), lesion=False)
    return float(info["mood"]), float(info["felt_arousal"]), int(info["level"])


def _lead(reader, mood, felt, *, lesion=False, shuffle=False):
    M = _M()
    lvl, _r, meta = reader.select_valence(mood, lesion=lesion, shuffle=shuffle)
    word = M.marker_from_level(lvl)
    if not word:
        return {"lead": "", "sel_level": lvl, "margin": float(meta["margin"])}
    hi, _r2, _m2 = reader.select_arousal(felt, lesion=lesion, shuffle=shuffle)
    emphatic = bool(hi) if hi is not None else bool(felt > 0.0)
    return {"lead": (word + "! ") if emphatic else (word + " — "), "sel_level": lvl, "margin": float(meta["margin"])}


def _arm(seed, mood, felt, level, *, delib, rest):
    """A production-shaped read: fresh reader, one prior read (the reader is always warm in production), then the
    intact read; an independent fresh+warm reader for the lesion read (RNG-isolated, same seed)."""
    M = _M()

    def warm_reader():
        r = M.AffectMarkerWTA(seed=seed, settle=True, deliberation_ms=delib, rest_ms=rest)
        _lead(r, mood, felt)
        return r
    li = _lead(warm_reader(), mood, felt)
    ll = _lead(warm_reader(), mood, felt, lesion=True)
    return {"intact": li, "lesion": ll, "load_bearing": bool(level != 0 and li["lead"] != ll["lead"])}


def verify(seeds, out: str) -> dict:
    from research.runners import _lbf_borderline_operating_point as B
    from tools.lab import attributable_to, lever, void_if
    from tools.verdict import Verdict
    M = _M()
    per = {}
    for s in seeds:
        mood, felt, level = _ladder(s)
        # production-path reads (get_reader + the diagnosis function itself), flag OFF then ON
        os.environ.pop(M.SETTLE_ENV, None); M.reset_readers()
        off = B._affect_marker(s)
        os.environ[M.SETTLE_ENV] = "1"; M.reset_readers()
        on = B._affect_marker(s)
        os.environ.pop(M.SETTLE_ENV, None); M.reset_readers()
        # determinism: two fresh readers, identical intact lead
        d1 = _arm(s, mood, felt, level, delib=M.DELIBERATION_MS, rest=M.INTERTURN_REST_MS)
        d2 = _arm(s, mood, felt, level, delib=M.DELIBERATION_MS, rest=M.INTERTURN_REST_MS)
        # shuffle anti-cheat under SETTLE
        rsh = M.AffectMarkerWTA(seed=s, settle=True); _lead(rsh, mood, felt)
        sh = _lead(rsh, mood, felt, shuffle=True)
        # attribution arms
        arms = {"neither(60,40)": _arm(s, mood, felt, level, delib=M.WARMUP_STEPS, rest=M.WASHOUT_STEPS),
                "deliberation_only": _arm(s, mood, felt, level, delib=M.DELIBERATION_MS, rest=M.WASHOUT_STEPS),
                "rest_only": _arm(s, mood, felt, level, delib=M.WARMUP_STEPS, rest=M.INTERTURN_REST_MS),
                "both": d1}
        diag = None
        dp = os.path.join(DIAG_DIR, f"op_s{s}.json")
        if os.path.exists(dp):
            with open(dp) as f:
                diag = json.load(f)["per_faculty"]["affect-marker-spiking-wta"]
        per[s] = {"ladder": {"mood": mood, "felt_arousal": felt, "level": level},
                  "off": {k: off[k] for k in ("intact", "lesion", "predicted_load_bearing")},
                  "on": {k: on[k] for k in ("intact", "lesion", "predicted_load_bearing")},
                  "determinism": {"lead_a": d1["intact"]["lead"], "lead_b": d2["intact"]["lead"],
                                  "identical": d1["intact"]["lead"] == d2["intact"]["lead"]},
                  "shuffle": {"intact_level": d1["intact"]["sel_level"], "shuffled_level": sh["sel_level"],
                              "differs": sh["sel_level"] != d1["intact"]["sel_level"]},
                  "attribution_arms": {k: {"intact_lead": v["intact"]["lead"], "lesion_lead": v["lesion"]["lead"],
                                           "intact_margin": v["intact"]["margin"], "load_bearing": v["load_bearing"]}
                                       for k, v in arms.items()},
                  "diagnosis_predicted_load_bearing": None if diag is None else bool(diag["predicted_load_bearing"])}
        print(f"s{s} mood={mood:+.4f} OFF lb={off['predicted_load_bearing']} ({off['intact']['lead']!r}/"
              f"{off['lesion']['lead']!r}, m={off['intact']['margin']:.3f})  ON lb={on['predicted_load_bearing']} "
              f"({on['intact']['lead']!r}/{on['lesion']['lead']!r}, m={on['intact']['margin']:.3f})  "
              f"det={per[s]['determinism']['identical']} shuf_differs={per[s]['shuffle']['differs']}  arms="
              + ",".join(f"{k}:{int(v['load_bearing'])}" for k, v in arms.items()), flush=True)

    n = len(seeds)
    g1 = sum(1 for s in seeds if per[s]["on"]["predicted_load_bearing"])
    g2 = sum(1 for s in seeds if per[s]["on"]["lesion"]["lead"] == "")
    g3 = sum(1 for s in seeds if per[s]["determinism"]["identical"])
    g4 = sum(1 for s in seeds if per[s]["shuffle"]["differs"])
    g5_rows = [(per[s]["off"]["predicted_load_bearing"], per[s]["diagnosis_predicted_load_bearing"]) for s in seeds]
    g5 = sum(1 for a, b in g5_rows if b is not None and a == b)
    n_off = sum(1 for s in seeds if per[s]["off"]["predicted_load_bearing"])
    lever("SETTLE flag: load-bearing seed count", n_off, g1, required=False)
    attributable_to("affect-marker load-bearing count: SETTLE ON vs OFF", float(g1), float(n_off))
    cheat = void_if(g2 < n and g1 == n, "load-bearing only because the lesion RE-PICKS a marker, not removes it")
    full = tuple(seeds) == VERIFY_SEEDS
    go = bool(full and g1 == n and g2 == n and g3 == n and g4 >= 4 and g5 == n and not cheat)
    vd = Verdict("affect_marker_settle_oplevel")
    vd.require("G1 SETTLE-ON load-bearing (count)", g1, expect=lambda x: x == n)
    vd.require("G2 SETTLE-ON lesion removes the marker (count)", g2, expect=lambda x: x == n)
    vd.require("G3 determinism: two fresh readers identical (count)", g3, expect=lambda x: x == n)
    vd.require("G4 shuffle changes the register (count >= 4 of 6)", g4, expect=lambda x: x >= min(4, n))
    vd.require("G5 SETTLE-OFF reproduces the committed diagnosis (count)", g5, expect=lambda x: x == n)
    vd.disabled("OU noise / STDP / homeostasis / STP", "identical to the committed WTA circuit (all OFF there)")
    decided = vd.decide(go)
    rec = {"probe": "affect_marker_settle_oplevel", "seeds": list(seeds), "full_6seed": full, "go": go,
           "counts": {"G1_on_load_bearing": g1, "G2_on_lesion_silent": g2, "G3_deterministic": g3,
                      "G4_shuffle_differs": g4, "G5_off_matches_diagnosis": g5, "off_load_bearing": n_off, "n": n},
           "constants": {"DELIBERATION_MS": M.DELIBERATION_MS, "INTERTURN_REST_MS": M.INTERTURN_REST_MS,
                         "WARMUP_STEPS": M.WARMUP_STEPS, "WASHOUT_STEPS": M.WASHOUT_STEPS,
                         "DEAD_MARGIN": M.DEAD_MARGIN},
           "per_seed": {str(s): v for s, v in per.items()}, "verdict": decided,
           "scope": "CIRCUIT-level de-risk on the real ladder 'emo' mood; the #1-metric verdict is the full-brain "
                    "load_bearing_fraction run with BRAIN_AFFECT_MARKER_SETTLE=1 (staged on the pool)."}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print(f"GO={go} G1={g1}/{n} G2={g2}/{n} G3={g3}/{n} G4={g4}/{n} G5={g5}/{n} (OFF lb {n_off}/{n}) -> {out}")
    return rec


def selftest() -> bool:
    """Byte-identical-OFF: with the flag unset, a reader uses the pre-existing 60/40 windows and the same cache key."""
    M = _M()
    os.environ.pop(M.SETTLE_ENV, None)
    r = M.AffectMarkerWTA(seed=42)
    ok = (r.settle is False and r.warmup == M.WARMUP_STEPS == 60 and r.washout == M.WASHOUT_STEPS == 40)
    M.reset_readers(); M.get_reader(42); ok = ok and (42 in M._READERS)
    os.environ[M.SETTLE_ENV] = "1"
    r2 = M.AffectMarkerWTA(seed=42)
    ok = ok and r2.settle and r2.warmup == M.DELIBERATION_MS and r2.washout == M.INTERTURN_REST_MS
    M.get_reader(42); ok = ok and ((42, "settle") in M._READERS)
    os.environ.pop(M.SETTLE_ENV, None); M.reset_readers()
    print("SELFTEST", "PASS" if ok else "FAIL")
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seeds", default=" ".join(str(s) for s in VERIFY_SEEDS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.calibrate:
        calibrate(a.out or "research/findings/raw/_affect_marker_settle/calibration.json")
    if a.verify:
        seeds = tuple(int(x) for x in a.seeds.replace(",", " ").split())
        verify(seeds, a.out or "research/findings/raw/_affect_marker_settle/oplevel_verify.json")


if __name__ == "__main__":
    main()
