"""Instrument acceptance for the long-delay credit benchmark (prereg section 7), applied mechanically to DEV artifacts.

Reads dev JSONs written by research/runners/long_delay_credit_bg.py for the positive-control schedules
(credit_immediate/full, late_immediate/full, sanity_early_ch0/full) and reports, per configuration tag, whether every
section-7 criterion holds. It never reads a prediction arm. The chosen configuration is the FIRST tag (in the order
given on the command line) that passes all criteria; if none passes it says so and exits 2.

  .venv/bin/python -m research.runners.long_delay_credit_bg_instrument --tags base,w40,w10 \
      --dir research/findings/raw/_long_delay_credit_bg/sweep
  (a tag's files are <dir>/<tag>_<schedule>_<anything>.json)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

DEV_SEEDS = tuple(range(7, 16))


def _data_files(d, tag, schedule):
    """The tag's data artifacts. The pattern also matches the provenance sidecars (`<artifact>.json.prov.json`,
    written next to every artifact by research/runners/__init__.py), which hold no rows: they are skipped."""
    return sorted(f for f in glob.glob(os.path.join(d, "%s_%s_*.json" % (tag, schedule)))
                  if not f.endswith(".prov.json"))


def _num(x, missing):
    """A measured 0.0 is a value; only None (not measured) takes the fail-safe default."""
    return missing if x is None else x


def _rows(d, tag, schedule):
    out = {}
    for f in _data_files(d, tag, schedule):
        for r in json.load(open(f))["rows"]:
            if r["arm"] == "full" and r["schedule"] == schedule:
                out[int(r["seed"])] = r
    return out


def criteria(d, tag):
    ci, li, sa = (_rows(d, tag, s) for s in ("credit_immediate", "late_immediate", "sanity_early_ch0"))
    res = {}
    seeds = sorted(set(ci) | set(li))
    probe = [r["probe_lr0"] for r in list(ci.values()) + list(li.values())]
    inst = [r["instrument"] for r in list(ci.values()) + list(li.values())]
    res["c1_commit_rates"] = bool(probe) and all(p["early_commit_rate"] >= 0.9 and p["terminal_commit_rate"] >= 0.9
                                                 for p in probe)
    res["c1_exploration"] = bool(probe) and all(
        p["p_ch0_given_A"] is not None and p["p_ch0_given_B"] is not None
        and 0.15 <= p["p_ch0_given_A"] <= 0.85 and 0.15 <= p["p_ch0_given_B"] <= 0.85 for p in probe)
    res["c1_intervening_commits"] = bool(inst) and all((i["interv_commit_rate_probe"] or 0) >= 0.9
                                                       and (i["decoy_commit_rate_probe"] or 0) >= 0.9 for i in inst)
    res["c2_memory"] = bool(inst) and all((i["mem_hold_acc_probe"] or 0) >= 0.9
                                          and (i["mem_after_decoy_acc_probe"] or 0) >= 0.9
                                          and (i["mem_after_clear_hz"] if i["mem_after_clear_hz"] is not None
                                               else 99) <= 5.0 for i in inst)
    # prereg section 7 item 3: the outcome bars on the critic-trained schedule; the tonic offset on EVERY probe
    # (it is not schedule-limited: a critic tail that leaks into the ITI shows only where the outcome precedes it)
    res["c3_dopamine"] = bool(ci) and bool(inst) and all(
        _num(r["da_outcome_rewarded_max_mean"], 0) >= 0.60 and _num(r["da_outcome_unrewarded_min_mean"], 1) <= 0.47
        for r in ci.values()) and all(abs(_num(i["da_iti_offset_probe"], 1)) <= 0.005 for i in inst)
    res["c4_sanity"] = bool(sa) and all(r["success"] >= 0.9 for r in sa.values())
    pos = [r for r in list(ci.values()) + list(li.values())]
    n_ok = sum(1 for r in ci.values() if r["success"] >= 0.8 and r["response_rate"] >= 0.8)
    n_ok_late = sum(1 for r in li.values() if r["success"] >= 0.8 and r["response_rate"] >= 0.8)
    res["c5_positive_controls"] = n_ok >= 6 and n_ok_late >= 6
    res["c6_no_bound"] = bool(pos) and not any(r["void_floor_ceiling"] for r in pos)
    res["n_seeds"] = {"credit_immediate": len(ci), "late_immediate": len(li), "sanity_early_ch0": len(sa)}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    report, chosen = {}, None
    for tag in a.tags.split(","):
        c = criteria(a.dir, tag)
        ok = all(v for k, v in c.items() if k.startswith("c"))
        report[tag] = {"pass": ok, "criteria": c}
        if ok and chosen is None:
            chosen = tag
    report["chosen"] = chosen
    print(json.dumps(report, indent=1))
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)
    sys.exit(0 if chosen else 2)


if __name__ == "__main__":
    main()
