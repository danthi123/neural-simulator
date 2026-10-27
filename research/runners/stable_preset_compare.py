"""STABLE vs STABLE+PRESET-CENTRED -- compare two membrane-census record sets of the same seed (research/stable-mechc).

Reads the per-group census JSONs written by research.runners.membrane_census with SIM_STABLE_INTEGRATION=1 and
SIM_PRESET_CENTRED_HET unset (base, "STABLE") and with both set to "1" (alt, "STABLE+PRESET"), and reports per group:

  (1) membrane validity per organ: out-of-range neurons base / alt, the class of every offender's first reading
      (the census's own `first_reading_classes`), and a mechanism split of the per-neuron rows: phasor (rf_masked),
      conductance (a), high-side, and the LOW-side cells with b < 0 whose own recovery variable holds v down
      (mechanism (c) of research/findings/2026-10-26-stable-integration-dev7.md: no conductance hazard, no refractory,
      b < 0, and the Izhikevich ODE's resting branch for the cell's u at its first excursion, the lower root of
      k (v - vr) (v - vt) = u, lies below the census's v_low);
  (2) the parameters the switch moved: per organ the neurons built, the neurons with b < 0 and the weighted mean of
      a, b, d, C as the bridges drew them (the census's `param_summary`), plus the heterogeneity-draw read-back
      (neurons with a region preset, of which off the distribution centre);
  (3) reply impact: every turn whose per-turn response sha256 differs, the answer-rate lens
      (tools/answer_rate.py classify/rate over the recorded turn replies) and the gate's own scorer on both shards.

Observe-only: reads JSON, runs nothing.

  .venv/bin/python -m research.runners.stable_preset_compare \
      --base research/findings/raw/_stable_mechc/census/stable/s7 \
      --alt research/findings/raw/_stable_mechc/census/stable_preset/s7 \
      --out research/findings/raw/_stable_mechc/compare_s7.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.stable_integration_compare import _load, compare_group  # noqa: E402
from tools import answer_rate as AR  # noqa: E402

GROUPS = ("conv", "infer", "ask", "emo", "inner")
CLASSES = ("phasor", "conductance_a", "high", "u_held_b_neg_c", "low_b_neg_other", "low_b_nonneg", "other")
CONDUCTANCE_HAZARD = 2.0       # the census's own low_conductance threshold, (g_i+g_e)*dt/C


def rest_branch_mV(k, vr, vt, u):
    """Lower root of k (v - vr) (v - vt) = u (the Izhikevich ODE's resting branch for a held u with no input current);
    None when the quadratic has no real root (u above the parabola's minimum: no resting branch)."""
    if not k or u is None:
        return None
    s = float(vr) + float(vt)
    disc = s * s - 4.0 * (float(vr) * float(vt) - float(u) / float(k))
    if disc < 0:
        return None
    return (s - math.sqrt(disc)) / 2.0


def classify_row(row, v_low):
    """Mechanism class of one per-neuron out-of-range row (see the module docstring)."""
    if row.get("rf_masked"):
        return "phasor"
    if row.get("extreme_high_mV") is not None or row.get("n_steps_high"):
        return "high"
    gt = row.get("gtot_dt_over_C")
    if gt is not None and gt >= CONDUCTANCE_HAZARD:
        return "conductance_a"
    b = row.get("b")
    if row.get("extreme_low_mV") is None or b is None:
        return "other"
    if b >= 0:
        return "low_b_nonneg"
    refr = bool(row.get("refractory")) or bool(row.get("refractory_post"))
    branch = rest_branch_mV(row.get("k"), row.get("vr"), row.get("vt"), row.get("u_before"))
    if not refr and branch is not None and branch < v_low:
        return "u_held_b_neg_c"
    return "low_b_neg_other"


def _bridge_organ(b):
    return b.get("organ") or "?"


def _organ_classes(rec):
    """{organ: {class: count}} over the per-neuron rows of every bridge, plus the first-reading classes."""
    v_low = float((rec["census"].get("thresholds") or {}).get("v_low_mV", -110.0))
    out = {}
    for b in rec["census"]["bridges"]:
        o = out.setdefault(_bridge_organ(b), {"classes": collections.Counter(), "first_reading": collections.Counter(),
                                              "rows_truncated": False, "n_oor": 0})
        o["n_oor"] += int(b.get("n_oor_neurons") or 0)
        o["rows_truncated"] = o["rows_truncated"] or bool(b.get("oor_neurons_truncated"))
        for r in b.get("oor_neurons") or []:
            o["classes"][classify_row(r, v_low)] += 1
        for k, c in (b.get("first_reading_classes") or {}).items():
            o["first_reading"][k] += int(c)
    return {k: {"classes": {c: v["classes"].get(c, 0) for c in CLASSES}, "first_reading": dict(v["first_reading"]),
                "rows_truncated": v["rows_truncated"], "n_oor": v["n_oor"]} for k, v in out.items()}


def _organ_params(rec):
    """{organ: parameters as drawn}: neurons, weighted means of a / b / d / C, cells with b < 0, and the heterogeneity-
    draw read-back summed over the organ's bridges."""
    acc = {}
    for b in rec["census"]["bridges"]:
        o = acc.setdefault(_bridge_organ(b), {"n": 0, "n_b_negative": 0, "n_bridges": 0, "n_with_region_preset": 0,
                                              "n_preset_off_centre": 0, "draw_centred": set(), "applied": set(),
                                              "sums": collections.Counter(), "regions": {}})
        o["n_bridges"] += 1
        ph = b.get("preset_centred_het") or {}
        o["applied"].add(str(ph.get("applied")))
        rb = ph.get("readback") or {}
        o["draw_centred"].add(str(rb.get("preset_centred")))
        o["n_with_region_preset"] += int(rb.get("n_with_region_preset") or 0)
        o["n_preset_off_centre"] += int(rb.get("n_preset_off_centre") or 0)
        for region, ps in (b.get("param_summary") or {}).items():
            n = int(ps["n"])
            o["n"] += n
            o["n_b_negative"] += int(ps["n_b_negative"])
            for k in ("a", "b", "d_increment", "C"):
                v = ps.get(k + "_mean")
                if v is not None:
                    o["sums"][k] += v * n
            r = o["regions"].setdefault(region, {"n": 0, "b_sum": 0.0, "n_b_negative": 0, "presets": collections.Counter()})
            r["n"] += n
            r["b_sum"] += (ps.get("b_mean") or 0.0) * n
            r["n_b_negative"] += int(ps["n_b_negative"])
            for k, c in (ps.get("preset_counts") or {}).items():
                r["presets"][k] += int(c)
    out = {}
    for name, o in acc.items():
        n = o["n"]
        out[name] = {"n_bridges": o["n_bridges"], "n_neurons": n, "n_b_negative": o["n_b_negative"],
                     **{k + "_mean": (round(o["sums"][k] / n, 4) if n else None) for k in ("a", "b", "d_increment", "C")},
                     "n_with_region_preset": o["n_with_region_preset"], "n_preset_off_centre": o["n_preset_off_centre"],
                     "draw_centred_seen": sorted(o["draw_centred"]), "switch_applied_seen": sorted(o["applied"]),
                     "regions": {r: {"n": v["n"], "b_mean": (round(v["b_sum"] / v["n"], 4) if v["n"] else None),
                                     "n_b_negative": v["n_b_negative"], "presets": dict(v["presets"])}
                                 for r, v in o["regions"].items()}}
    return out


def _answer_rate(rec):
    reps = (rec.get("workload") or {}).get("turn_replies") or {}
    kinds = {lab: AR.classify(body) for lab, body in reps.items()}
    return kinds, AR.rate(list(reps.values()))


def compare_pair(group, base, alt):
    res = compare_group(group, base, alt)               # keys named off/on there: off = base (STABLE), on = alt
    (_, r_base), (_, r_alt) = base, alt
    cb, ca = _organ_classes(r_base), _organ_classes(r_alt)
    pb, pa = _organ_params(r_base), _organ_params(r_alt)
    organs = []
    for name in list(cb) + [n for n in ca if n not in cb]:
        organs.append({"organ": name, "base": cb.get(name), "alt": ca.get(name),
                       "params_base": {k: v for k, v in (pb.get(name) or {}).items() if k != "regions"},
                       "params_alt": {k: v for k, v in (pa.get(name) or {}).items() if k != "regions"}})
    regions_moved = []
    for name in pa:
        for region, ra in (pa[name]["regions"] or {}).items():
            rb = ((pb.get(name) or {}).get("regions") or {}).get(region)
            if rb is None or ra["b_mean"] is None or rb["b_mean"] is None:
                continue
            if abs(ra["b_mean"] - rb["b_mean"]) > 0.25 or ra["n_b_negative"] != rb["n_b_negative"]:
                regions_moved.append({"organ": name, "region": region, "n": ra["n"], "b_mean_base": rb["b_mean"],
                                      "b_mean_alt": ra["b_mean"], "n_b_negative_base": rb["n_b_negative"],
                                      "n_b_negative_alt": ra["n_b_negative"], "presets_alt": ra["presets"]})
    kb, rate_b = _answer_rate(r_base)
    ka, rate_a = _answer_rate(r_alt)
    transitions = collections.Counter("%s->%s" % (kb[l], ka[l]) for l in kb if l in ka)
    res["mechanism"] = {"organs": organs,
                        "u_held_b_neg_c_base": sum((cb[o]["classes"]["u_held_b_neg_c"]) for o in cb),
                        "u_held_b_neg_c_alt": sum((ca[o]["classes"]["u_held_b_neg_c"]) for o in ca),
                        "n_b_negative_base": sum(p["n_b_negative"] for p in pb.values()),
                        "n_b_negative_alt": sum(p["n_b_negative"] for p in pa.values()),
                        "n_preset_off_centre_alt": sum(p["n_preset_off_centre"] for p in pa.values()),
                        "regions_moved": regions_moved}
    res["answer_rate"] = {"base": rate_b, "alt": rate_a, "transitions": dict(transitions),
                          "turn_kinds_base": kb, "turn_kinds_alt": ka}
    res["env"] = {"base": [r_base.get("stable_integration_env"), r_base.get("preset_centred_het_env")],
                  "alt": [r_alt.get("stable_integration_env"), r_alt.get("preset_centred_het_env")]}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="census dir: SIM_STABLE_INTEGRATION=1 only")
    ap.add_argument("--alt", required=True, help="census dir: SIM_STABLE_INTEGRATION=1 and SIM_PRESET_CENTRED_HET=1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    base, alt = _load(a.base), _load(a.alt)
    groups = [g for g in GROUPS if g in base and g in alt]
    res = {"runner": "research/runners/stable_preset_compare.py", "base_dir": a.base, "alt_dir": a.alt,
           "groups_compared": groups, "groups_missing": sorted((set(base) | set(alt)) - set(groups)),
           "groups": {g: compare_pair(g, base[g], alt[g]) for g in groups}}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    for g in groups:
        r = res["groups"][g]
        m = r["mechanism"]
        print("%-6s oor base %d alt %d | u-held b<0 base %d alt %d | b<0 cells base %d alt %d | preset-off-centre %d | "
              "turns changed %d/%d | answer rate base %s alt %s | scorer all base %s alt %s" % (
                  g, r["membrane"]["n_oor_neurons_off"], r["membrane"]["n_oor_neurons_on"],
                  m["u_held_b_neg_c_base"], m["u_held_b_neg_c_alt"], m["n_b_negative_base"], m["n_b_negative_alt"],
                  m["n_preset_off_centre_alt"], r["replies"]["n_changed"], r["replies"]["n_turns"],
                  r["answer_rate"]["base"]["answer_rate"], r["answer_rate"]["alt"]["answer_rate"],
                  (r["gate_score"]["off"] or {}).get("all"), (r["gate_score"]["on"] or {}).get("all")))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
