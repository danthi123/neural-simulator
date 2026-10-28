"""Writes research/findings/raw/_stable_recal/stop_calibration_record.json (research/stable-recal).

The global-stop distributed-overwrite workspace (`_gnw_distributed_overwrite_workspace_derisk`, the substrate of
webapp/gnw_global_stop.py) re-calibrated by its own rules for the stable integrator. Every figure a cited artifact
holds is READ from the artifact (the old-integrator runaway, the normalization x weight read reliability, the thalamic
support sweep, the knee, the stop effector's clearing window, the sensor's two margin levels, the organ GO-gate counts
per check and the supplementary two-content checks); the calibration itself is read from the organ-gate artifacts'
recorded calibration and, once `research.runners.stable_recal.STOP` exists, checked against it, so the record cannot
disagree with the code. What stays typed here is the JUDGMENT behind each constant (the `rule` strings) and the claims
the prose makes, each of which is checked against the artifacts before the record is written. No brain, no GPU. Run as
a module so the provenance sidecar is written:

  .venv/bin/python -m research.runners.stable_recal_stop_calibration_record \
      --out research/findings/raw/_stable_recal/stop_calibration_record.json

Exit code 1 (and no record) when a cited artifact is missing, a typed claim is not supported by the artifacts, or the
record's values differ from `stable_recal.STOP`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

RAW = "research/findings/raw/_stable_recal"
CAL_SEEDS = (7, 8)
ORGAN_SEEDS = tuple(range(1, 16))
GATE = RAW + "/stop_organ_gate"
GATE_LIT = RAW + "/stop_organ_gate_literal"
HELD = RAW + "/stop_organ_held"
SUSTAIN = RAW + "/stop_sustain_nws"
THAL = RAW + "/stop_thal_support"
WINDOW = RAW + "/stop_window"
MARGINS = RAW + "/stop_margins"
TARGET_BOOST = 0.8      # the middle of the robust clearing band [0.6, 1.0] (judgment, see boost_scale rule)
PRODUCTION = {"w_rec": 34.0, "norm_ws_w": 16.0, "thal_tonic_pa": 700.0, "solo_plateau": 1.0 / 3.0,
              "strong_pa": 5000.0, "margin_ref": 1.0 / 6.0, "boost_scale": 0.85}


def _load(path):
    with open(os.path.join(_REPO, path)) as f:
        return json.load(f)


def _arm(path):
    (name, rec), = _load(path)["arms"].items()
    return name, rec


def _sidecar(path):
    return _load(path + ".prov.json")


CHECK_ORDER = ["confident commit ignites (n>=1)", "a multi-content conflict state exists to clear (co-ignition n>=2)",
               "normalization pool active (norm_ws_w > 0)",
               "conflict margin readable (conflict low < confident high, below margin_ref)",
               "depression boost conflict-gated (0 at a confident commit, fires on conflict)",
               "GLOBAL STOP: conflict n>=2 -> n_ignited=0",
               "SELECTIVITY: confident commit not disrupted (boost=0 -> holds n=1)",
               "divisive-norm LOAD-BEARING (lesion -> burst does NOT clear, n stays >=2)",
               "STD LOAD-BEARING (freeze boost=0 -> conflict HOLDS)",
               "neural-sensor SCRAMBLE breaks the stop (confident margin -> not cleared)",
               "CONFLICT-OFF holds (boost_gain=0 -> not cleared)",
               "continuous stop makes 0 host restore calls (emptying is synaptic depression)",
               "byte-identical substrate (distributed hash == localist base hash)",
               "determinism: cfg.seed seeds the substrate (build-twice hash)",
               "continuous stop clears (fresh build, no restore: n>=2 -> 0)"]


def _organ(files):
    """Per-check pass counts (the 14 Verdict requirements + the continuous-stop outcome evaluate_seed also gates on),
    GO count, failing checks -> seeds, out-of-range cells in the stop trial, and sidecar facts."""
    go, per_check, per_seed, oor, dirty, shas = 0, {k: [] for k in CHECK_ORDER}, {}, [], [], set()
    calibs = []
    for path in files:
        _, rec = _arm(path)
        res = rec["result"]
        seed = int(res["seed"])
        go += bool(res["seed_go"])
        oks = {p["name"]: bool(p["ok"]) for p in res["preconditions"]}
        cont = res["measurements"]["continuous_headline"]
        oks[CHECK_ORDER[-1]] = bool(cont["n_ignited_pre"] >= 2 and cont["n_ignited_post"] == 0)
        for k in CHECK_ORDER:
            if oks.get(k):
                per_check[k].append(seed)
        stop = res["measurements"]["stop"]
        per_seed[seed] = {"go": bool(res["seed_go"]), "verdict": res["verdict"],
                          "failed": [k for k in CHECK_ORDER if not oks.get(k)],
                          "conflict_margin": round(stop["margin"], 4), "boost": round(stop["boost"], 4),
                          "stop_n": [stop["n_ignited_pre"], stop["n_ignited_post"]],
                          "freeze_n": [res["measurements"]["freeze"]["n_ignited_pre"],
                                       res["measurements"]["freeze"]["n_ignited_post"]],
                          "confident_margin": round(res["measurements"]["confident"]["margin"], 4)}
        oor.append(int(rec["stop_trial_membranes"]["n_oor_cells"]))
        sc = _sidecar(path)
        dirty.append(bool(sc["git_dirty"]))
        shas.add(sc["git_sha"])
        calibs.append(rec["calibration"])
    return {"n_seeds": len(files), "n_go": go, "go_seeds": sorted(s for s, v in per_seed.items() if v["go"]),
            "checks_passed_seeds": {k: sorted(v) for k, v in per_check.items()},
            "checks_passed_count": {k: len(v) for k, v in per_check.items()},
            "per_seed": dict(sorted(per_seed.items())),
            "stop_trial_oor_cells_max": max(oor), "any_sidecar_dirty": any(dirty), "git_shas": sorted(shas),
            "calibrations": calibs}


def _read15(n, s, w):
    d = _load("%s/nws%d_s%d.json" % (SUSTAIN, n, s))
    rows = [r for r in d["rows"] if r["w_rec"] == w and r["drive"] == 2500.0]
    assert len(rows) == 1, (n, s, w)
    r = rows[0]
    q = r["read15"][0]
    return {"plateau": q["plateau"], "worst_window": q["min"], "frac_windows_below_half": q["frac_below_half"],
            "neighbour_rate_last_window": max(r["windows"][-1]["rates"][1:])}


def _thal():
    out = {}
    for s in CAL_SEEDS:
        for tag in ("0_25_50", "75_100_200", "400_700"):
            for r in _load("%s/w6p5_nws4_t%s_s%d.json" % (THAL, tag, s))["rows"]:
                if r["drive"] != 0.0:
                    continue
                peak = max(max(w["rates"]) for w in r["windows"])
                out.setdefault("%g" % r["thal_tonic_pa"], {})["s%d" % s] = round(peak, 4)
    return out


def _knee():
    out = {}
    for s in CAL_SEEDS:
        d = _load("%s/stop_ignition_curve_w6p5_nws4_thal50_s%d.json" % (RAW, s))
        (row,) = d["arms"]["on"]
        out["s%d" % s] = {"%g" % c["drive"]: c["read_rates"][0] for c in row["curve"]}
        out["s%d_old_integrator_oor" % s] = {"%g" % c["drive"]: c["n_oor_cells"] for c in d["arms"]["off"][0]["curve"]}
        out["s%d_old_integrator_co_oor" % s] = {"%g" % c["drive"]: c["n_oor_cells"]
                                                for c in d["arms"]["off"][0]["co_drive"]}
    return out


def _old_runaway():
    out = {}
    for s in CAL_SEEDS:
        d = _load("%s/stop_ignition_curve_s%d.json" % (RAW, s))
        row = [r for r in d["arms"]["off"] if r["w_rec"] == 34.0][0]
        ign = [c for c in row["curve"] if c["read_rates"][0] >= 0.1667]
        co = [c for c in row["co_drive"] if c["drive"] == 5000.0][0]
        on = [r for r in d["arms"]["on"]]
        out["s%d" % s] = {
            "solo_knee_pa": min(c["drive"] for c in ign),
            "solo_ignited_rates": sorted({c["read_rates"][0] for c in ign}),
            "solo_ignited_oor_cells": [min(c["n_oor_cells"] for c in ign), max(c["n_oor_cells"] for c in ign)],
            "co_drive_5000_n_ignited": co["n_ignited_read"], "co_drive_5000_oor_cells": co["n_oor_cells"],
            "stable_any_ignition_any_weight": any(c["read_rates"][0] >= 0.1667 for r in on for c in r["curve"]),
            "stable_max_oor": max(c["n_oor_cells"] for r in on for c in r["curve"] + r["co_drive"]),
            "n_ws_cells": row["curve"][0]["n_cells"]}
    return out


def _window():
    out = {}
    for s in CAL_SEEDS:
        d = _load("%s/recal_on_s%d.json" % (WINDOW, s))["arms"]["on"]
        for key, v in d.items():
            out.setdefault(key, {})["s%d" % s] = {b: "%d/%d" % (pb["n_cleared"], pb["n_trials"])
                                                 for b, pb in v["per_boost"].items()}
        o = _load("%s/production_off_s%d.json" % (WINDOW, s))["arms"]["off"]
        out.setdefault("old_integrator_production", {})["s%d" % s] = {
            key: v["min_clearing_boost"] for key, v in o.items()}
    return out


def _robust_band(window):
    """Boosts at which the intact workspace clears on EVERY trial (both seeds, both held sets) and the lesion never."""
    boosts = sorted(window["intact_n3"]["s7"], key=float)
    band = []
    for b in boosts:
        intact = all(window[k]["s%d" % s][b].split("/")[0] == window[k]["s%d" % s][b].split("/")[1]
                     for k in ("intact_n3", "intact_n2") for s in CAL_SEEDS)
        lesion = all(window[k]["s%d" % s][b].startswith("0/") for k in ("norm_lesion_n3", "norm_lesion_n2")
                     for s in CAL_SEEDS)
        if intact and lesion:
            band.append(float(b))
    return band


def _norm_attribution(window, band):
    """Whose is the clearing in the robust band: the fraction of intact trials that clear (normalization present)
    against the fraction of normalization-lesioned trials that clear (control), pooled over both calibration seeds and
    both held sets, at each boost of the band; `attributable_to` = (intact - lesion) / intact."""
    from tools.lab import attributable_to
    out = {}
    for b in ["%g" % x for x in band]:
        def frac(keys):
            num = sum(int(window[k]["s%d" % s][b].split("/")[0]) for k in keys for s in CAL_SEEDS)
            den = sum(int(window[k]["s%d" % s][b].split("/")[1]) for k in keys for s in CAL_SEEDS)
            return num / float(den)
        t, c = frac(("intact_n3", "intact_n2")), frac(("norm_lesion_n3", "norm_lesion_n2"))
        out[b] = {"intact_cleared_frac": round(t, 4), "lesion_cleared_frac": round(c, 4),
                  "attributable_to_normalization": attributable_to(
                      "stop clearing at fixed boost %s: intact vs normalization lesion" % b, t, c)}
    return out


def _margins():
    out = {}
    for s in CAL_SEEDS:
        d = _load("%s/recal_on_s%d.json" % (MARGINS, s))["arms"]["on"]
        out["s%d" % s] = {k: d[k]["margin"] for k in ("confident", "conflict_n3", "conflict_n2")}
    for s in CAL_SEEDS:
        d = _load("%s/production_off_s%d.json" % (MARGINS, s))["arms"]["off"]
        out["old_production_s%d" % s] = {k: d[k]["margin"]["mean"] for k in ("confident", "conflict_n3", "conflict_n2")}
    return out


def _held():
    rows = {}
    for s in ORGAN_SEEDS:
        _, rec = _arm("%s/recal_on_held01_s%d.json" % (HELD, s))
        rows[s] = rec["checks"]
    return {"n_seeds": len(rows), "all_checks_pass_seeds": sorted(s for s, c in rows.items() if all(c.values())),
            "per_check_count": {k: int(sum(c[k] for c in rows.values())) for k in next(iter(rows.values()))}}


def build():
    old = _old_runaway()
    window = _window()
    band = _robust_band(window)
    marg = _margins()
    conf_mean = float(np.mean([marg["s%d" % s]["confident"]["mean"] for s in CAL_SEEDS]))
    conflict_mean = float(np.mean([marg["s%d" % s]["conflict_n3"]["mean"] for s in CAL_SEEDS]))
    midpoint = round((conf_mean + conflict_mean) / 2.0, 3)
    deficit = midpoint - conflict_mean
    scale = round(TARGET_BOOST / deficit)
    lit_deficit = 0.042 - conflict_mean
    lit_scale = round(TARGET_BOOST / lit_deficit)
    thal = _thal()
    knee = _knee()
    sustain = {"nws%d" % n: {"s%d" % s: {"%g" % w: _read15(n, s, w) for w in (4.0, 4.5, 5.0, 6.5, 8.0, 10.0)}
                             for s in CAL_SEEDS} for n in (2, 4, 5, 6, 8, 16)}
    gate = _organ(["%s/recal_on_s%d.json" % (GATE, s) for s in ORGAN_SEEDS])
    lit = _organ(["%s/literal_on_s%d.json" % (GATE_LIT, s) for s in ORGAN_SEEDS])
    prod_off = _organ(["%s/production_off_s%d.json" % (GATE, s) for s in CAL_SEEDS])
    prod_on = _organ(["%s/production_on_s%d.json" % (GATE, s) for s in CAL_SEEDS])
    held = _held()
    calib = gate["calibrations"][0]
    if any(c != calib for c in gate["calibrations"]):
        raise ValueError("the re-calibration organ runs do not share one calibration")
    recal = {"w_rec": calib["w_rec"], "norm_ws_w": calib["norm_ws_w"], "thal_tonic_pa": calib["thal_tonic_pa"],
             "solo_plateau": calib["solo_plateau"], "strong_pa": calib["strong_pa"],
             "margin_ref": calib["margin_ref"], "boost_scale": calib["boost_scale"]}

    def pr(name):
        return {"production": PRODUCTION[name], "recal": recal[name]}

    const = {
        "norm_ws_w": {**pr("norm_ws_w"), "read15_by_norm": sustain,
                      "rule": "the swap organ's rule, applied to this organ: the largest norm_pool->workspace weight whose "
                              "sustained single-pattern state reads reliably in the organ's 15-step late window (fraction of "
                              "windows below half the plateau, steps 100-1000, thal 0, drive 2500, stable). At 16, 8 and 6 "
                              "every weight 4-10 has windows below half; at 5 only weight 5 has none, with a worst window of "
                              "~0.6 of the plateau; at 4 weights 4.5-8 have none and at 6.5 the worst window is 0.80-0.89 of "
                              "the plateau. At 2 the read is reliable but the neighbouring overlapped pattern is recruited "
                              "from weight 6.5 (the organ's confident-commit leak). Read literally, the rule selects 5 "
                              "(weight 5 has no window below half; worst window ~0.6 of the plateau); 4 over 5 is a recorded "
                              "margin judgment (weights 4.5-8 have none and the worst window at 6.5 is 0.80-0.89), not the "
                              "rule's output",
                      "source": ["%s/nws%d_s%d.json" % (SUSTAIN, n, s) for n in (2, 4, 5, 6, 8, 16) for s in CAL_SEEDS]
                      + ["%s/stop_ignition_inhibition_s7.json" % RAW]},
        "w_rec": {**pr("w_rec"),
                  "rule": "at normalization 4 the sustain knee lies between 4.0 (0.37/0.38 of windows below half) and 4.5 "
                          "(none); the neighbour is recruited at 10 and not at 8; production is deliberately supra-critical, "
                          "so 6.5 (~1.44x the knee, inside the reliable and selective band 4.5-8; the swap organ's judgment) "
                          "was chosen. A judgment, not a formula",
                  "source": ["%s/nws4_s%d.json" % (SUSTAIN, s) for s in CAL_SEEDS]},
        "thal_tonic_pa": {**pr("thal_tonic_pa"), "undriven_peak_rate_by_thal": thal,
                          "rule": "the swap organ's rule: the largest thalamic support with no spontaneous ignition over 1000 "
                                  "undriven steps on both calibration seeds (weight 6.5, normalization 4); 0, 25 and 50 pA stay "
                                  "below the criterion (50 pA: seed 8's pattern 2 reads 0.001 in its last window), 75 pA ignites "
                                  "patterns by themselves on both seeds",
                          "source": ["%s/w6p5_nws4_t%s_s%d.json" % (THAL, t, s) for t in ("0_25_50", "75_100_200", "400_700")
                                     for s in CAL_SEEDS]},
        "solo_plateau": {**pr("solo_plateau"),
                         "measured": {"s%d" % s: sustain["nws4"]["s%d" % s]["6.5"]["plateau"] for s in CAL_SEEDS},
                         "rule": "the measured ignited plateau at (6.5, 4) on the stable integrator; the ignition criterion "
                                 "stays IGNITE_FRAC (0.5) x plateau = 0.042, scoped to the stop organ (Rung-2's SOLO_PLATEAU is "
                                 "not touched)",
                         "source": ["%s/nws4_s%d.json" % (SUSTAIN, s) for s in CAL_SEEDS]},
        "strong_pa": {**pr("strong_pa"), "knee_read_rates": {k: v for k, v in knee.items() if "oor" not in k},
                      "rule": "the module's rule STRONG_PA = 2 x IGNITE_PA (the knee): the stable knee by the organ's 45-step "
                              "read at the re-calibration is 400 pA (300 pA reads 0.037 on seed 7 and 350 pA 0.028 on seed 8, "
                              "below the 0.042 criterion; 400 pA ignites on both), so 2 x 400",
                      "source": ["%s/stop_ignition_curve_w6p5_nws4_thal50_s%d.json" % (RAW, s) for s in CAL_SEEDS]},
        "margin_ref": {**pr("margin_ref"), "sensor_levels": marg,
                       "pooled_confident_mean": round(conf_mean, 4), "pooled_conflict_n3_mean": round(conflict_mean, 4),
                       "rule": "the module's stated rule: MARGIN_REF is 'halfway between single-content and co-ignition'. "
                               "Its formula SOLO_PLATEAU x 0.5 is that rule evaluated at a co-ignition margin of 0, which held "
                               "on the old integrator (conflict margin exactly 0 in the lockstep runaway) and does not hold on "
                               "valid membranes (three-content conflict 0.0267 / 0.0351, max 0.0468 > 0.042). So the midpoint "
                               "of the two measured levels (confident and three-content conflict, the gate's conflict, pooled "
                               "over the calibration seeds) = %.3f. Decided BEFORE the organ gate ran; the literal formula "
                               "(0.042) was run through the gate alongside as a comparison arm" % midpoint,
                       "source": ["%s/recal_on_s%d.json" % (MARGINS, s) for s in CAL_SEEDS]
                       + ["%s/production_off_s%d.json" % (MARGINS, s) for s in CAL_SEEDS]},
        "boost_scale": {**pr("boost_scale"), "clearing_window_fixed_boost": window, "robust_band": band,
                        "normalization_attribution_in_band": _norm_attribution(window, band),
                        "target_boost": TARGET_BOOST, "full_conflict_deficit": round(deficit, 4),
                        "rule": "the module's rule: the boost at a full conflict lies inside the window where the intact "
                                "workspace clears and the normalization-lesioned one does not (production: ~0.13 inside "
                                "[0.09, 0.15]). On valid membranes (fixed boost, 8 trials per boost, both calibration seeds, "
                                "three and two held contents) the intact workspace clears on every trial at %s, with a "
                                "non-monotone dip at 0.25-0.5; the lesion never clears up to 1.0 (U >= 1 empties x per spike, "
                                "so a larger boost is the same). Target = %.1f, the middle of the robust band 0.6-1.0 (a "
                                "judgment), at the full-conflict deficit margin_ref - conflict = %.4f: %.1f / %.4f = %.1f, "
                                "rounded to %d" % (", ".join("%g" % b for b in band), TARGET_BOOST, deficit, TARGET_BOOST,
                                                   deficit, TARGET_BOOST / deficit, scale),
                        "source": ["%s/recal_on_s%d.json" % (WINDOW, s) for s in CAL_SEEDS]
                        + ["%s/production_off_s%d.json" % (WINDOW, s) for s in CAL_SEEDS]},
        "unchanged": {"overlap": 15, "ws_norm_w": 5.0, "thal_ws_w": 5.0, "pulse_duration": 110, "boost_gain": 1.0,
                      "std_tau_d_ms": 250.0, "ou_noise_pa": 30.0, "read_free_steps": 45, "post_free_steps": 100,
                      "conf_hold_steps": 90,
                      "rule": "geometry, the depression's time constant, the pulse, the noise and the read windows are not "
                              "constants of the runaway; nothing measured here asks for them to move"},
    }
    for v in const.values():
        for p in v.get("source", []):
            if not os.path.exists(os.path.join(_REPO, p)):
                raise FileNotFoundError("cited source missing: " + p)
    if scale != recal["boost_scale"] or midpoint != recal["margin_ref"]:
        raise ValueError("the organ runs' margin_ref/boost_scale (%s, %s) are not the rule's (%s, %s)" % (
            recal["margin_ref"], recal["boost_scale"], midpoint, scale))
    lcal = lit["calibrations"][0]
    if lcal["margin_ref"] != 0.042 or lcal["boost_scale"] != lit_scale or \
            {k: v for k, v in lcal.items() if k not in ("margin_ref", "boost_scale")} != \
            {k: v for k, v in calib.items() if k not in ("margin_ref", "boost_scale")}:
        raise ValueError("the literal comparison arm differs from the re-calibration in more than margin_ref/scale")

    n_checks = len(CHECK_ORDER)
    stop_ok = set(gate["checks_passed_seeds"][CHECK_ORDER[5]])
    freeze_ok = set(gate["checks_passed_seeds"][CHECK_ORDER[8]])
    claims = {
        "old integrator, production: a single pattern ignites only at rate 1/3 with 103-109 cells out of range": all(
            v["solo_ignited_rates"] == [0.3333] and 103 <= v["solo_ignited_oor_cells"][0]
            and v["solo_ignited_oor_cells"][1] <= 109 for v in old.values()),
        "old integrator, production: three co-driven at 5000 pA ignite with all 270 workspace cells out of range": all(
            v["co_drive_5000_n_ignited"] == 3 and v["co_drive_5000_oor_cells"] == 270 == v["n_ws_cells"]
            for v in old.values()),
        "stable integrator, production thal/normalization: nothing ignites at any weight 5-60, and at most one cell "
        "leaves the range (seed 8, weight 5, all three driven at 8000 pA)": all(
            (not v["stable_any_ignition_any_weight"]) for v in old.values())
        and old["s7"]["stable_max_oor"] == 0 and old["s8"]["stable_max_oor"] == 1,
        "old integrator production window: intact clears from 0.09, lesion from 0.22-0.3": all(
            v["intact_n3"] == 0.09 and v["intact_n2"] == 0.09 and v["norm_lesion_n3"] in (0.22, 0.3)
            and v["norm_lesion_n2"] in (0.22, 0.3) for v in window["old_integrator_production"].values()),
        "robust clearing band is 0.6-1.0 (plus 0.15 not on every case)": band == [0.6, 0.8, 1.0],
        "production on the stable integrator: UNDEFINED on both seeds (nothing ignites)": (
            prod_on["n_go"] == 0 and all(v["verdict"] == "UNDEFINED" for v in prod_on["per_seed"].values())),
        "production on the old integrator: GO on both seeds, 270 cells out of range in the stop trial": (
            prod_off["n_go"] == 2 and prod_off["stop_trial_oor_cells_max"] == 270),
        "re-calibration: GO 5 of 15 (1, 3, 5, 7, 10)": gate["go_seeds"] == [1, 3, 5, 7, 10],
        "re-calibration: no cell out of range in any stop trial": gate["stop_trial_oor_cells_max"] == 0,
        "re-calibration: selectivity, normalization load-bearing, byte-identity and determinism on 15 of 15": all(
            gate["checks_passed_count"][k] == 15 for k in (CHECK_ORDER[6], CHECK_ORDER[7], CHECK_ORDER[12],
                                                         CHECK_ORDER[13])),
        "re-calibration: the global stop on 11 of 15, the freeze holding the conflict on 7 of 15": (
            gate["checks_passed_count"][CHECK_ORDER[5]] == 11 and gate["checks_passed_count"][CHECK_ORDER[8]] == 7),
        "re-calibration: verdicts GO 5 / NO-GO 1 (seed 13) / UNDEFINED 9": (
            sorted(v["verdict"] for v in gate["per_seed"].values()) == ["GO"] * 5 + ["NO-GO"] + ["UNDEFINED"] * 9
            and [s for s, v in gate["per_seed"].items() if v["verdict"] == "NO-GO"] == [13]),
        "re-calibration: the stop clears on 11 of 15; on 5 of those (4, 8, 11, 14, 15) the boost-0 freeze precondition "
        "fails; a standing conflict is removed on 6 of 15 (1, 3, 5, 7, 10, 13)": (
            sorted(stop_ok) == [1, 3, 4, 5, 7, 8, 10, 11, 13, 14, 15]
            and sorted(stop_ok - freeze_ok) == [4, 8, 11, 14, 15] and sorted(stop_ok & freeze_ok) == [1, 3, 5, 7, 10, 13]
            and all(gate["per_seed"][s]["verdict"] == "UNDEFINED" for s in stop_ok - freeze_ok)),
        "norm_ws_w: at 4 weights 4.5-8 have no window below half and the worst window at 6.5 is 0.80-0.89 of the plateau": (
            all(sustain["nws4"]["s%d" % s][w]["frac_windows_below_half"] == 0.0 for s in CAL_SEEDS
                for w in ("4.5", "5", "6.5", "8"))
            and [round(sustain["nws4"]["s%d" % s]["6.5"]["worst_window"] / sustain["nws4"]["s%d" % s]["6.5"]["plateau"], 2)
                 for s in CAL_SEEDS] == [0.89, 0.80]),
        "norm_ws_w: the literal rule selects 5 (weight 5 has no window below half at 5, worst ~0.6 of the plateau; every "
        "weight 4-10 has windows below half at 6, 8 and 16)": (
            all(sustain["nws5"]["s%d" % s]["5"]["frac_windows_below_half"] == 0.0
                and 0.55 <= sustain["nws5"]["s%d" % s]["5"]["worst_window"] / sustain["nws5"]["s%d" % s]["5"]["plateau"]
                <= 0.65 for s in CAL_SEEDS)
            and all(sustain["nws%d" % n]["s%d" % s][w]["frac_windows_below_half"] > 0.0 for n in (6, 8, 16)
                    for s in CAL_SEEDS for w in sustain["nws%d" % n]["s%d" % s])),
        "literal-formula arm: GO 4 of 15": lit["n_go"] == 4,
        "two-content held set: every check on 15 of 15": held["all_checks_pass_seeds"] == list(ORGAN_SEEDS),
        "every organ sidecar is clean and from one commit": (not gate["any_sidecar_dirty"]) and (not lit["any_sidecar_dirty"])
        and len(gate["git_shas"]) == 1,
    }
    bad = [k for k, ok in claims.items() if not ok]
    if bad:
        raise ValueError("the record's prose is not supported by the cited artifacts: %s" % bad)

    try:
        from research.runners import stable_recal as SR
        table = getattr(SR, "STOP", None)
    except Exception:
        table = None
    if table is not None:
        for name, v in recal.items():
            prod, rc = table[name]
            if float(prod) != float(PRODUCTION[name]) or float(rc) != float(v):
                raise ValueError("stable_recal.STOP[%r] = %r differs from the record (%r, %r)" % (
                    name, table[name], PRODUCTION[name], v))
    for g in (gate, lit, prod_off, prod_on):
        g.pop("calibrations", None)
    return {
        "runner": "research/runners/stable_recal_stop_calibration_record.py (reads the cited artifacts; the rules are "
                  "typed judgments, the claims are checked)",
        "backend": "numpy", "device": "cpu", "date": "2026-10-27",
        "what": "global-stop distributed-overwrite workspace (research/runners/_gnw_distributed_overwrite_workspace_derisk.py, "
                "the substrate of webapp/gnw_global_stop.py) re-calibrated by its own rules on the stable integrator "
                "(SIM_STABLE_INTEGRATION=1); every value chosen on the isolated organ on dev seeds 7 and 8, never on a reply",
        "calibration": recal, "production": PRODUCTION,
        "checked_against_stable_recal_STOP": table is not None,
        "seed_roles": {"calibration_seeds": list(CAL_SEEDS), "organ_gate_seeds": list(ORGAN_SEEDS),
                       "reading": "dev seeds 7 and 8 are the calibration seeds (every constant was read off them), so the "
                                  "organ gate on 7 and 8 and any whole-brain comparison on them is IN-SAMPLE; the organ gate "
                                  "on 1-6 and 9-15 is out-of-sample for every constant"},
        "integrator_finding": old,
        "constants": const,
        "organ_go_gate": {
            "n_checks": n_checks, "checks": CHECK_ORDER,
            "recal_stable_dev1_15": gate, "literal_formula_stable_dev1_15": lit,
            "production_old_integrator_s7_s8": prod_off, "production_stable_integrator_s7_s8": prod_on,
            "supplementary_two_content_held_set_dev1_15": held,
            "reading": "at the re-calibration the organ's verdicts over 15 dev seeds are GO 5 (1, 3, 5, 7, 10), NO-GO 1 "
                       "(seed 13, the continuous stop) and UNDEFINED 9, each of the 9 with an unmet validity precondition; "
                       "the GO count does not include the UNDEFINED seeds. The stop clears the conflict (n_ignited 0 after "
                       "the pulse) on 11 of 15, but on 5 of those 11 (seeds 4, 8, 11, 14, 15) the boost-0 freeze "
                       "precondition fails (the three-content conflict is already resolved to one pattern without the "
                       "boost), so the organ gate is UNDEFINED there; a standing conflict is removed on 6 of 15 (1, 3, 5, "
                       "7, 10, 13). Selectivity, normalization load-bearing, byte-identity and determinism hold on 15 of 15. "
                       "The dominant failure is not the stop: the anti-cheats that need the three-content conflict to HOLD "
                       "without the stop (freeze, scramble, conflict-off) pass on 7 of 15, because on valid membranes the "
                       "divisively normalized workspace resolves a three-way co-ignition to one pattern by itself within ~250 "
                       "steps; on "
                       "the old integrator the lockstep runaway made every co-ignited pattern self-sufficient. The remaining "
                       "stop failures are the sensor: a conflict that reads a margin near or above margin_ref gets a boost in "
                       "the 0.25-0.5 dip or none (seeds 2, 9, 12; seed 6 starts with one pattern). With the two contents "
                       "production usually holds, the stop, the normalization lesion, freeze and scramble pass on 15 of 15 "
                       "(supplementary, not the gate; conflict-off, the conflict-gated boost and the continuous stop were not "
                       "run with two). No constant was re-derived after the gate ran",
            "source": [GATE + "/", GATE_LIT + "/", HELD + "/"]},
        "order_of_events": "all constants were fixed from the cited sweeps (curve, inhibition, sustain, thal, knee, window, "
                           "margins) before the organ gate ran; margin_ref's midpoint reading and the literal-formula comparison "
                           "arm were both fixed before. The gate first ran at 16:13 (its sidecars read git_dirty because a test "
                           "helper was being edited meanwhile) and was re-run from clean commits only to re-stamp provenance "
                           "(16:18-16:19, in waves with the outputs committed between them); the results are byte-identical "
                           "across the runs, and no constant was changed after the gate",
        "probe": "research/runners/stable_integration_stop_ignition_probe.py",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rec = build()
    out = os.path.join(_REPO, a.out) if not os.path.isabs(a.out) else a.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print("[stable_recal_stop_calibration_record] wrote %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
