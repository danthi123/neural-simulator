"""Affect-marker SETTLE de-risk (D1 lane "grow the robust core past 23", 2026-09-23).

THE DEFECT (measured, not guessed). `affect-marker-spiking-wta` is outside the robust core: under the diagnosis
read (labelled "RNG-isolated" at the time, but it runs intact then lesion on ONE cached warm reader with OU noise off,
so the per-seed confound is warm-state carry-over, not RNG) it is lesion-load-bearing on only 1/6 seeds (`research/findings/raw/_lbf_borderline_isolated/op_s*.json`). On the 'emo'
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
                 AMENDMENT (post-hoc, disclosed; commit after a4891aa3e): as registered, NO window passed -- C1 failed
                 at the mood=0 midpoint (unreachable: gated to neutral upstream) and at the arousal boundary on one
                 seed. C1' = commit at every REACHABLE valence boundary; arousal boundary commit is report-only.
                 DELIBERATION_MS = the shortest window passing C1'+C2+C3+C4 on all calibration seeds.
  --verify     : the PRE-REGISTERED op-level GO gate on the 6 verification seeds (42 43 44 100 101 102), reading the
                 REAL ladder mood the 'emo' turn produces (`_lbf_borderline_operating_point._affect_marker`, the same
                 read the 2026-09-22 diagnosis used), SETTLE OFF vs ON:
                   G1 load-bearing  : SETTLE ON predicted_load_bearing (intact lead != lesion lead, level != 0) on 6/6;
                   G2 lesion silent : SETTLE ON lesion lead == '' on 6/6 (the lesion removes the marker, it does not
                                      merely re-pick one);
                   G3 determinism   : two FRESH readers at the same seed give the identical intact lead, 6/6;
                   G4 shuffle       : mis-routing the drive (shuffle=True) changes the selected register on >= 4/6
                                      (a fixed random 6-permutation leaves the winner in place ~1/6 of the time);
                   G5 OFF baseline  : SETTLE OFF reproduces the committed diagnosis EXACTLY (intact + lesion read dicts,
                                      float margin included, equal op_s*.json) -- the instrument still reads what it
                                      read before;
                   G6 attribution   : arms (deliberation only) / (rest only) / (both) reported per seed, so the
                                      verdict names WHICH companion process carries the flip (report-only).
                 GO iff G1..G5 hold. This is the CIRCUIT-level de-risk; the #1-metric verdict is the full-brain
                 `load_bearing_fraction --only affect-marker-spiking-wta --seed S` run with the flag ON AND a
                 same-code flag-OFF control, scored by --score-fullbrain (below).
  --score-fullbrain : the PRE-REGISTERED full-brain ON-vs-OFF CONTRAST gate (FIX ROUND 2026-09-23, see the
                 AMENDMENT LOG). SETTLE is CREDITED on a seed only if the ON row is load-bearing (null-clean,
                 lesion-reproduced, deterministic, env=='1') AND the same-seed OFF row is a VALID measurement
                 (env unset, deterministic, null-clean, load_bearing not None) that reads NOT load-bearing. A seed
                 whose OFF row is ALSO load-bearing is load-bearing irrespective of the flag -> no credit to SETTLE.
                   UNDEFINED : any ON or OFF row missing / ambiguous (!=1 match) / non-deterministic / wrong env /
                               load_bearing None, or an OFF null control dirty. NEVER a pass.
                   GO        : ON load-bearing 6/6 AND OFF load-bearing <= 1/6 (credited >= 5/6). The <=1 allowance is
                               the op-level OFF reading known BEFORE this gate was written (s100, oplevel_verify.json).
                   NO-GO/PARTIAL : ON 6/6 but OFF load-bearing on 2..5 seeds -> "load-bearing 6/6 with the flag,
                               SETTLE-attributable on k/6 only" (not credited as the flag's effect).
                   NO-GO     : ON < 6/6, or OFF load-bearing 6/6 (the flag is unattributable).

TERMINOLOGY (docs/TERMS.md). BRAIN_AFFECT_MARKER_SETTLE is DEFAULT-OFF. A GO here means: affect-marker is
lesion-load-bearing under the ADEQUATE probe with an opt-in, default-off flag. It is NOT "on-by-default", NOT
"integrated / production-default", and it does NOT grow the production-default robust core. Report it Option-C
style, as a PAIR: adequate-probe robust core 23/26 at the shipped flag state, and 24/26 with SETTLE opt-in IF the
contrast gate reads GO; the default flip is owner-reserved and would need its own re-verify against flipped code.

NAMED HOST SHORTCUTS on this path (declared, not removed by this build):
  S1 READOUT (`AffectMarkerWTA._select` in `_affect_marker_wta_derisk.py`): the winner is named by `np.argsort`
     over the pools' spike RATES plus a host `DEAD_MARGIN`=0.05 threshold on winner-minus-runner-up. That is an
     argmax-over-spike-counts readout -- a shortcut per CLAUDE.md, pre-existing. After a resolved 500 ms race the
     loser pools sit at rate 0, so the argsort is benign here, but a downstream neural read-out (a motor/premotor
     pool driven by the marker assemblies) is the replacement target.
  S2 DRIVE (`_select` / `_gaussian_drive`): the felt mood float is converted by the HOST into a Gaussian-tuned
     afferent current per pool (`DRIVE_BASE_PA + DRIVE_GAIN_PA*exp(-(v-c)^2/2sigma^2)`). The mood itself is a
     neural read of the #81 ladder, but the population-code projection is a host formula, not synapses from the
     ladder's populations -- pre-existing. Replacement target: a synaptic projection from the ladder's V+/V- and
     arousal pools onto the marker assemblies.
  S3 CLOCK: deliberation / rest durations are host-set (the legitimate clock role; added by this build).
  S4 pre-existing, unchanged: host `mood_to_level` binning upstream; the emphasis fallback `felt > 0` when the
     arousal WTA does not commit; the host renders the winning register's fixed word.
So "no host formula chooses the marker" is too strong: the COMPETITION is spiking, but S1 names the winner and S2
builds the input.

AMENDMENT LOG
  2026-09-23 ~09:00 (build round): C1 -> C1' (post-hoc, disclosed above); DELIBERATION_MS registered 300 in a4891aa3e,
     set to 500 by C1' calibration. SEEN at that time: calibration.json only. Independently rechecked by the review:
     200 and 300 ms also give op-level load-bearing 6/6, so 500 was not outcome-selected.
  2026-09-23 ~12:00 (fix round, BEFORE any full-brain row of the re-staged revision existed): (a) added
     --score-fullbrain with the ON-vs-OFF contrast rule above -- the build round's gate only required OFF env==null
     and so could credit SETTLE even if OFF were load-bearing too; (b) G5 now asserts EXACT dict equality of the OFF
     intact/lesion reads with the committed diagnosis (was a boolean compare); (c) the op-level verify's OUTCOME
     counts (G1, G4) are now the go-condition, not Verdict preconditions, so a failed G1 reads NO-GO instead of
     UNDEFINED; validity checks (G2 lesion-silent, G3 determinism, G5 OFF baseline) stay preconditions. SEEN at that
     time: oplevel_verify.json (ON 6/6, OFF 1/6 = s100), the 1-seed smoke lbf_settle_s42.json (ON s42 load-bearing).
     NOT SEEN: the contents of any pool full-brain row. The superseded revision-56e588d partial rows (ON s42/43/44/101,
     OFF s42/43; their raw arms shared one dir) were copied to lbf_superseded_rev56e588d/ UNREAD, and are scored only
     after this log was committed, as an informational cross-check that does not enter the verdict.
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
    # AMENDED C1' (POST-HOC, disclosed -- written after the as-registered calibration found NO passing window):
    # (a) the valence midpoint between -1 and +1 is mood = 0.0, which production NEVER presents to this circuit
    #     (webapp.affect_drives_chat: |mood| < _MOOD_L1=0.010 -> level 0 -> the neutral gate returns '' BEFORE the
    #     circuit is called), so a commit requirement there tests an unreachable input;
    # (b) the arousal boundary selects only the emphasis punctuation, not whether a marker is emitted (the
    #     load-bearing quantity); it is REPORTED separately, not gated.
    reach = [c1_v[i] for i, b in enumerate(vb) if abs(b) >= 0.010]
    res["C1prime_reachable_valence_boundary_commit"] = bool(all(reach))
    res["pass_amended"] = all(res[k] for k in ("C1prime_reachable_valence_boundary_commit", "C2_center_fidelity",
                                               "C3_spontaneous_silent", "C4_repeatable"))
    return res


def calibrate(out: str) -> dict:
    M = _M()
    rest = M.INTERTURN_REST_MS
    table = {}
    chosen = None
    chosen_amended = None
    for d in DELIB_GRID:
        rows = {s: _calibrate_one(s, d, rest) for s in CAL_SEEDS}
        ok = all(r["pass"] for r in rows.values())
        ok_am = all(r["pass_amended"] for r in rows.values())
        n_arousal = sum(1 for r in rows.values() if r["detail"]["c1_a"])
        table[d] = {"all_pass": ok, "all_pass_amended": ok_am, "arousal_boundary_commit_count": n_arousal,
                    "per_seed": {str(s): r for s, r in rows.items()}}
        if ok_am and chosen_amended is None:
            chosen_amended = d
        print(f"delib={d:4d} all_pass={ok} amended={ok_am} arousal_commit={n_arousal}/{len(rows)} " + " ".join(
            f"s{s}:{''.join('1' if rows[s][k] else '0' for k in ('C1_boundary_commit','C2_center_fidelity','C3_spontaneous_silent','C4_repeatable'))}"
            for s in CAL_SEEDS), flush=True)
        if ok and chosen is None:
            chosen = d
    rec = {"probe": "affect_marker_settle_calibration", "calibration_seeds": list(CAL_SEEDS),
           "grid_ms": list(DELIB_GRID), "rest_ms": rest, "chosen_deliberation_ms": chosen,
           "chosen_deliberation_ms_amended": chosen_amended,
           "module_constant_DELIBERATION_MS": M.DELIBERATION_MS,
           "constant_matches_calibration": chosen_amended == M.DELIBERATION_MS,
           "amendment": "POST-HOC (disclosed): C1 as registered failed at every window; C1' drops the mood=0 midpoint "
                        "(gated to neutral upstream, never reaches the circuit) and moves the arousal boundary to "
                        "report-only (emphasis punctuation, not marker presence). See _calibrate_one.",
           "criterion": "shortest grid window passing C1 boundary-commit, C2 center-fidelity, C3 spontaneous-silent, "
                        "C4 repeatable on ALL calibration seeds (disjoint from the verification seeds)",
           "table": {str(k): v for k, v in table.items()}}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"CHOSEN deliberation: as-registered={chosen} ms, amended C1'={chosen_amended} ms "
          f"(module constant {M.DELIBERATION_MS}) -> {out}")
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
                  "diagnosis_predicted_load_bearing": None if diag is None else bool(diag["predicted_load_bearing"]),
                  "diagnosis_reads": None if diag is None else {"intact": diag["intact"], "lesion": diag["lesion"]}}
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
    # G5 EXACT: the OFF intact AND lesion read dicts (lead, sel_level, float margin) must EQUAL the committed diagnosis
    # dicts, not merely agree on the boolean (fix round 2026-09-23: the finding claimed dict equality, the code
    # checked only the boolean).
    g5 = sum(1 for s in seeds if per[s]["diagnosis_reads"] is not None
             and per[s]["off"]["intact"] == per[s]["diagnosis_reads"]["intact"]
             and per[s]["off"]["lesion"] == per[s]["diagnosis_reads"]["lesion"]
             and per[s]["off"]["predicted_load_bearing"] == per[s]["diagnosis_predicted_load_bearing"])
    n_off = sum(1 for s in seeds if per[s]["off"]["predicted_load_bearing"])
    lever("SETTLE flag: load-bearing seed count", n_off, g1, required=False)
    attributable_to("affect-marker load-bearing count: SETTLE ON vs OFF", float(g1), float(n_off))
    cheat = void_if(g2 < n and g1 == n, "load-bearing only because the lesion RE-PICKS a marker, not removes it")
    full = tuple(seeds) == VERIFY_SEEDS
    # OUTCOMES (G1 load-bearing, G4 shuffle) are the go-condition -> a failure reads NO-GO. VALIDITY checks (full seed
    # set, G2 the lesion removes rather than re-picks, G3 determinism, G5 the OFF instrument still reads what it read
    # before) are preconditions -> a failure reads UNDEFINED (fix round 2026-09-23, AMENDMENT LOG (c)).
    go = bool(g1 == n and g4 >= 4 and not cheat)
    vd = Verdict("affect_marker_settle_oplevel")
    vd.require("full 6-seed verification set", full, expect=True)
    vd.require("G2 SETTLE-ON lesion removes the marker (count)", g2, expect=lambda x: x == n)
    vd.require("G3 determinism: two fresh readers identical (count)", g3, expect=lambda x: x == n)
    vd.require("G5 SETTLE-OFF reproduces the committed diagnosis dicts EXACTLY (count)", g5, expect=lambda x: x == n)
    vd.disabled("OU noise / STDP / homeostasis / STP", "identical to the committed WTA circuit (all OFF there)")
    decided = vd.decide(go)
    go = bool(decided["go"])
    rec = {"probe": "affect_marker_settle_oplevel", "seeds": list(seeds), "full_6seed": full, "go": go,
           "counts": {"G1_on_load_bearing": g1, "G2_on_lesion_silent": g2, "G3_deterministic": g3,
                      "G4_shuffle_differs": g4, "G5_off_matches_diagnosis": g5, "off_load_bearing": n_off, "n": n},
           "constants": {"DELIBERATION_MS": M.DELIBERATION_MS, "INTERTURN_REST_MS": M.INTERTURN_REST_MS,
                         "WARMUP_STEPS": M.WARMUP_STEPS, "WASHOUT_STEPS": M.WASHOUT_STEPS,
                         "DEAD_MARGIN": M.DEAD_MARGIN},
           "per_seed": {str(s): v for s, v in per.items()}, "verdict": decided,
           "preconditions": decided["preconditions"], "status": decided["status"],
           "scope": "CIRCUIT-level de-risk on the real ladder 'emo' mood; the #1-metric verdict is the full-brain "
                    "load_bearing_fraction run with BRAIN_AFFECT_MARKER_SETTLE=1 (staged on the pool)."}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    print(f"GO={go} G1={g1}/{n} G2={g2}/{n} G3={g3}/{n} G4={g4}/{n} G5={g5}/{n} (OFF lb {n_off}/{n}) -> {out}")
    return rec


# ─────────────────────────────────── full-brain ON-vs-OFF contrast gate ───────────────────────────────────
ON_DIR = "research/findings/raw/_affect_marker_settle/lbf_on"
OFF_DIR = "research/findings/raw/_affect_marker_settle/lbf_off"
FACULTY = "affect-marker-spiking-wta"


def _find_row(root: str, arm: str, seed: int):
    """Exactly ONE `lbf_settle_{arm}_s{seed}.json` anywhere under `root` (harvest layout lbf_{arm}/<node>/s<seed>/).
    Zero or >1 matches -> None + reason (ambiguity is UNDEFINED, never silently the first file)."""
    import glob
    hits = sorted(glob.glob(os.path.join(root, "**", f"lbf_settle_{arm}_s{seed}.json"), recursive=True))
    if len(hits) != 1:
        return None, f"{len(hits)} files match lbf_settle_{arm}_s{seed}.json under {root}"
    with open(hits[0]) as f:
        return json.load(f), hits[0]


def _row_facts(j: dict, arm: str) -> dict:
    """The fields the gate reads, straight from the LBF report (read, not derived)."""
    pf = [r for r in j.get("per_faculty", []) if r.get("faculty") == FACULTY]
    r = pf[0] if len(pf) == 1 else {}
    env = j.get("affect_marker_settle_env")
    det = (j.get("determinism") or {}).get("deterministic")
    lb = r.get("load_bearing")
    facts = {"load_bearing": lb, "null_control_clean": r.get("null_control_clean"),
             "lesion_reproduced": r.get("lesion_reproduced"), "deterministic": det, "env": env,
             "diffs": r.get("diffs"), "verdict": r.get("verdict"), "n_faculty_rows": len(pf)}
    why = []
    if len(pf) != 1:
        why.append(f"{len(pf)} {FACULTY} rows")
    if det is not True:
        why.append(f"determinism.deterministic={det}")
    if lb is None:
        why.append("load_bearing=None (noisy/unmeasured)")
    if arm == "on" and env != "1":
        why.append(f"ON env={env!r} (flag did not reach the run)")
    if arm == "off" and env is not None:
        why.append(f"OFF env={env!r} (flag leaked into the control)")
    if arm == "off" and r.get("null_control_clean") is not True:
        why.append("OFF null control not clean (its load_bearing label is not interpretable)")
    facts["valid"] = not why
    facts["invalid_reasons"] = why
    facts["on_credit_ready"] = bool(arm == "on" and facts["valid"] and lb is True
                                    and r.get("null_control_clean") is True and r.get("lesion_reproduced") is True)
    return facts


def score_rows(rows: dict, seeds=VERIFY_SEEDS) -> dict:
    """rows = {seed: {"on": report-or-None, "off": report-or-None, "on_src":..., "off_src":...}} -> verdict record.
    Pure (no I/O), so the selftest can drive it in each failing direction."""
    from tools.verdict import Verdict
    per = {}
    for s in seeds:
        r = rows.get(s, {})
        on = _row_facts(r["on"], "on") if r.get("on") is not None else None
        off = _row_facts(r["off"], "off") if r.get("off") is not None else None
        valid = bool(on and on["valid"] and off and off["valid"])
        credited = bool(valid and on["on_credit_ready"] and off["load_bearing"] is False)
        per[s] = {"on": on, "off": off, "on_src": r.get("on_src"), "off_src": r.get("off_src"),
                  "valid_pair": valid, "on_load_bearing": bool(on and on["on_credit_ready"]),
                  "off_load_bearing": None if not (off and off["valid"]) else bool(off["load_bearing"]),
                  "settle_credited": credited}
    n = len(seeds)
    n_valid = sum(1 for s in seeds if per[s]["valid_pair"])
    n_on = sum(1 for s in seeds if per[s]["on_load_bearing"])
    n_off = sum(1 for s in seeds if per[s]["off_load_bearing"] is True)
    n_cred = sum(1 for s in seeds if per[s]["settle_credited"])
    bad = []
    for s in seeds:
        if not per[s]["valid_pair"]:
            on_why = per[s]["on"]["invalid_reasons"] if per[s]["on"] else rows.get(s, {}).get("on_src")
            off_why = per[s]["off"]["invalid_reasons"] if per[s]["off"] else rows.get(s, {}).get("off_src")
            bad.append(f"s{s}: ON {on_why} OFF {off_why}")
    vd = Verdict("affect_marker_settle_fullbrain_contrast")
    vd.require("every seed has exactly one VALID ON row and one VALID OFF row", n_valid, expect=lambda x: x == n,
               note="; ".join(bad))
    vd.require("full 6-seed verification set", tuple(seeds) == VERIFY_SEEDS, expect=True)
    go = bool(n_on == n and n_off <= 1)
    decided = vd.decide(go)
    if decided["status"] == "UNDEFINED":
        tier = "UNDEFINED"
    elif decided["go"]:
        tier = "GO: load-bearing 6/6 with SETTLE opt-in (default-off), SETTLE-attributable"
    elif n_on == n and 2 <= n_off <= n - 1:
        tier = (f"PARTIAL (NO-GO for the flag): load-bearing {n_on}/{n} with SETTLE, "
                f"SETTLE-attributable on {n_cred}/{n} only")
    elif n_on == n and n_off == n:
        tier = "NO-GO: load-bearing with and without SETTLE on every seed -> the flag is unattributable"
    else:
        tier = f"NO-GO: SETTLE-ON load-bearing on {n_on}/{n} (<{n})"
    return {"probe": "affect_marker_settle_fullbrain_contrast", "seeds": list(seeds),
            "counts": {"n": n, "valid_pairs": n_valid, "on_load_bearing": n_on, "off_load_bearing": n_off,
                       "settle_credited": n_cred},
            "go": bool(decided["go"]), "status": decided["status"], "tier": tier, "verdict": decided,
            "preconditions": decided["preconditions"], "per_seed": {str(s): v for s, v in per.items()},
            "rule": "credit a seed iff ON load-bearing (null-clean, lesion-reproduced, deterministic, env=='1') AND "
                    "the same-seed OFF row is valid and NOT load-bearing; GO iff ON 6/6 and OFF load-bearing <= 1/6",
            "terminology": "GO = load-bearing under the ADEQUATE probe with a DEFAULT-OFF opt-in flag. NOT "
                           "on-by-default, NOT production-default; the production-default robust core is unchanged "
                           "(Option-C pair: 23/26 at the shipped flag state, 24/26 with SETTLE opt-in iff GO)."}


def score_fullbrain(on_dir: str, off_dir: str, out: str, seeds=VERIFY_SEEDS) -> dict:
    rows = {}
    for s in seeds:
        on, on_src = _find_row(on_dir, "on", s)
        off, off_src = _find_row(off_dir, "off", s)
        rows[s] = {"on": on, "off": off, "on_src": on_src, "off_src": off_src}
    rec = score_rows(rows, seeds)
    rec["on_dir"], rec["off_dir"] = on_dir, off_dir
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1, default=str)
    c = rec["counts"]
    print(f"{rec['status']} | {rec['tier']} | valid {c['valid_pairs']}/{c['n']} ON-lb {c['on_load_bearing']} "
          f"OFF-lb {c['off_load_bearing']} credited {c['settle_credited']} -> {out}")
    return rec


def _selftest_contrast() -> bool:
    """The contrast gate must FAIL in each failing direction (a gate that cannot fail is only an integrity smoke)."""
    import contextlib
    import io

    def rep(lb, env, det=True, null=True, les=True):
        return {"affect_marker_settle_env": env, "determinism": {"deterministic": det},
                "per_faculty": [{"faculty": FACULTY, "load_bearing": lb, "null_control_clean": null,
                                 "lesion_reproduced": les}]}

    def rows(on_lb, off_lb, **kw):
        return {s: {"on": rep(on_lb[i], "1"), "off": rep(off_lb[i], None, **kw)}
                for i, s in enumerate(VERIFY_SEEDS)}
    T, F = True, False
    cases = [
        ("clean contrast -> GO", rows([T] * 6, [F] * 6), "GO"),
        ("OFF lb on 1 seed (the known s100) -> GO", rows([T] * 6, [F, F, F, T, F, F]), "GO"),
        ("OFF lb 3/6 -> NO-GO (partial)", rows([T] * 6, [T, T, F, T, F, F]), "NO-GO"),
        ("OFF lb 6/6 -> NO-GO (unattributable)", rows([T] * 6, [T] * 6), "NO-GO"),
        ("ON 5/6 -> NO-GO", rows([T, T, T, T, T, F], [F] * 6), "NO-GO"),
        ("OFF load_bearing None -> UNDEFINED", rows([T] * 6, [F, F, None, F, F, F]), "UNDEFINED"),
        ("OFF non-deterministic -> UNDEFINED", rows([T] * 6, [F] * 6, det=False), "UNDEFINED"),
        ("OFF dirty null -> UNDEFINED", rows([T] * 6, [F] * 6, null=False), "UNDEFINED"),
    ]
    missing = rows([T] * 6, [F] * 6)
    missing[44]["off"] = None
    cases.append(("missing OFF row -> UNDEFINED", missing, "UNDEFINED"))
    leak = rows([T] * 6, [F] * 6)
    leak[42]["off"]["affect_marker_settle_env"] = "1"
    cases.append(("flag leaked into OFF -> UNDEFINED", leak, "UNDEFINED"))
    noflag = rows([T] * 6, [F] * 6)
    noflag[43]["on"]["affect_marker_settle_env"] = None
    cases.append(("flag missing from ON -> UNDEFINED", noflag, "UNDEFINED"))
    ok = True
    for name, rw, want in cases:
        with contextlib.redirect_stdout(io.StringIO()):
            got = score_rows(rw)["status"]
        print(f"  contrast selftest: {name:42s} want={want:9s} got={got:9s} {'ok' if got == want else 'FAIL'}")
        ok = ok and got == want
    return ok


def mechanism_probe(out: str) -> dict:
    """Reproduce, as a committed artifact, the two substrate reads that identified the companion processes (they
    were first seen in exploratory scratch probes before the gate was registered; this makes them citable).
      (a) deliberation: calibration seed 7, mood exactly ON the +2/+3 boundary, unchanged circuit, rest 1000 ms,
          deliberation 60 vs 300 vs 500 ms -> per-pool rates + margin;
      (b) inter-turn rest: seed 42, mood +0.0682 (its 'emo' ladder read), unchanged 60 ms deliberation, three
          consecutive reads on one warm reader at washout 40 / 200 / 1000 / 3000 ms."""
    M = _M()
    C = M.MOOD_CENTERS
    boundary = (C[4] + C[5]) / 2.0
    delib = {}
    for d in (60, 300, 500):
        r = M.AffectMarkerWTA(seed=7, settle=True, deliberation_ms=d, rest_ms=1000)
        lvl, rates, meta = r.select_valence(boundary)
        delib[str(d)] = {"level": lvl, "rates_pos_pools": [round(float(x), 6) for x in rates[3:]],
                         "margin": round(float(meta["margin"]), 6)}
    rest = {}
    for w in (40, 200, 1000, 3000):
        r = M.AffectMarkerWTA(seed=42, settle=False, deliberation_ms=M.WARMUP_STEPS, rest_ms=w)
        rest[str(w)] = [round(float(r.select_valence(0.0682)[2]["margin"]), 6) for _ in range(3)]
    rec = {"probe": "affect_marker_settle_mechanism_probe", "boundary_mood": boundary,
           "a_deliberation_seed7_boundary": delib, "b_rest_seed42_mood0.0682_three_reads": rest}
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec, indent=1))
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
    ok = _selftest_contrast() and ok
    print("SELFTEST", "PASS" if ok else "FAIL")
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--mechanism-probe", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--score-fullbrain", action="store_true")
    ap.add_argument("--on-dir", default=ON_DIR)
    ap.add_argument("--off-dir", default=OFF_DIR)
    ap.add_argument("--seeds", default=" ".join(str(s) for s in VERIFY_SEEDS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.mechanism_probe:
        mechanism_probe(a.out or "research/findings/raw/_affect_marker_settle/mechanism_probe.json")
    if a.calibrate:
        calibrate(a.out or "research/findings/raw/_affect_marker_settle/calibration.json")
    if a.score_fullbrain:
        seeds = tuple(int(x) for x in a.seeds.replace(",", " ").split())
        score_fullbrain(a.on_dir, a.off_dir,
                        a.out or "research/findings/raw/_affect_marker_settle/fullbrain_contrast_verdict.json", seeds)
    if a.verify:
        seeds = tuple(int(x) for x in a.seeds.replace(",", " ").split())
        verify(seeds, a.out or "research/findings/raw/_affect_marker_settle/oplevel_verify.json")


if __name__ == "__main__":
    main()
