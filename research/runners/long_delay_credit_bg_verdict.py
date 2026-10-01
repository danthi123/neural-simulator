"""Verdict reader for the long-delay temporal-credit benchmark on the production BG organ.

Implements the frozen bars of research/findings/2026-09-29-long-delay-credit-bg-organ-PREREG.md section 6. It only
READS artifacts written by research/runners/long_delay_credit_bg.py (rows keyed by schedule, arm, seed).

  .venv/bin/python -m research.runners.long_delay_credit_bg_verdict <artifacts...> \
      --runner-sha256 <the SHA frozen by Amendment A> --out <verdict.json>
"""
from __future__ import annotations

import argparse
import json
import sys

from tools.lab import attributable_to

GATE_SEEDS = (42, 43, 44, 100, 101, 102)
CAPABLE, FAILS, CONTRAST, K_OF_6 = 0.75, 0.65, 0.15, 5
RESP_FLOOR, MEM_HOLD_OK, EARLY_SHARE_MIN, DOSE_DELTA, DROP = 0.80, 0.90, 0.5, 0.15, 0.10


def load(paths):
    cells, hashes = {}, set()
    for p in paths:
        if p.endswith(".prov.json"):     # Step 7's shell glob `gate/*.json` also passes the provenance sidecars
            continue
        d = json.load(open(p))
        hashes.add(d.get("runner_sha256"))
        for r in d["rows"]:
            cells[(r["schedule"], r["arm"], int(r["seed"]))] = r
    return cells, hashes


def verdict(cells):
    out = {"gates": {}, "construct_validity": {}, "undefined": [], "mechanisms": {}, "signatures": {}}

    def row(s, a, sd):
        return cells.get((s, a, sd))

    def valid(r):
        """A cell is usable only if its relevant race responded and no plastic pathway hit a floor/ceiling."""
        return (r is not None and (r.get("response_rate") or 0) >= RESP_FLOOR and not r.get("void_floor_ceiling"))

    def succ(s, a, sd):
        r = row(s, a, sd)
        return float(r["success"]) if valid(r) and r.get("success") is not None else None

    # seed definedness per schedule family: the full arm must respond (and hold memory for late schedules)
    defined = {}
    for s in ("credit", "credit_immediate", "late_delayed", "late_immediate"):
        ok = set()
        for sd in GATE_SEEDS:
            r = row(s, "full", sd)
            if not valid(r):
                out["undefined"].append("%s seed %d: full arm missing, silent or at a weight bound" % (s, sd))
                continue
            if s.startswith("late") and (r.get("mem_hold_acc") or 0) < MEM_HOLD_OK:
                out["undefined"].append("%s seed %d: memory hold %.2f < %.2f" % (s, sd, r.get("mem_hold_acc") or 0,
                                                                                 MEM_HOLD_OK))
                continue
            ok.add(sd)
        defined[s] = ok
    defined["dose"] = defined["credit"]

    def gate(name, fam, pred, into="gates"):
        vals = {sd: pred(sd) for sd in sorted(defined[fam])}
        n_true = sum(1 for v in vals.values() if v is True)
        n_undef = sum(1 for v in vals.values() if v is None) + len(GATE_SEEDS) - len(vals)
        state = "PASS" if n_true >= K_OF_6 else ("UNDEFINED" if n_true + n_undef >= K_OF_6 else "FAIL")
        out[into][name] = {"state": state, "n_true": n_true, "n_undefined": n_undef, "per_seed": vals}
        return state

    def ge(s, a, bar):
        return lambda sd: None if succ(s, a, sd) is None else succ(s, a, sd) >= bar

    def le(s, a, bar):
        return lambda sd: None if succ(s, a, sd) is None else succ(s, a, sd) <= bar

    def all_of(*fs):
        def f(sd):
            v = [g(sd) for g in fs]
            return None if any(x is None for x in v) else all(v)
        return f

    def drop(s, a, b):
        return lambda sd: (None if succ(s, a, sd) is None or succ(s, b, sd) is None
                           else succ(s, a, sd) - succ(s, b, sd) >= CONTRAST)

    def early_share(sd):
        r = row("credit", "full", sd)
        if not valid(r) or r.get("early_share") is None:
            return None
        return r["M_early"] > 0 and r["early_share"] >= EARLY_SHARE_MIN

    def dose(sd):
        a = succ("credit", "dose_m8.0_tau300.0", sd)
        b = succ("credit", "dose_m0.5_tau300.0", sd)
        c = succ("credit", "dose_m8.0_tau3000.0", sd)
        if None in (a, b, c):
            return None
        return (a <= b - DOSE_DELTA) and (c >= a + DOSE_DELTA)

    gate("C1_trace_ablation_abolishes_credit", "credit", le("credit", "trace_ablated_integral", FAILS),
         into="construct_validity")
    gate("C2_memory_lesion_spares_credit", "credit", ge("credit", "memory_lesioned", CAPABLE),
         into="construct_validity")
    gate("G1_credit_capability", "credit", ge("credit", "full", CAPABLE))
    gate("G2_credit_lands_on_early_event", "credit", early_share)
    gate("G3_horizon_at_equal_integral", "dose", dose)
    gate("G4_ablated_learner_competent", "credit_immediate", ge("credit_immediate", "trace_ablated_integral",
                                                                CAPABLE))
    gate("G5_memory_positive_control", "late_immediate", ge("late_immediate", "full", CAPABLE))
    gate("G6_memory_necessary_for_terminal_choice", "late_immediate",
         all_of(le("late_immediate", "memory_lesioned", FAILS), drop("late_immediate", "full", "memory_lesioned")))
    gate("G7_terminal_choice_delayed_reward", "late_delayed", ge("late_delayed", "full", CAPABLE))
    gate("G8a_no_leak_credit", "credit", le("credit", "shuffled_yoked", FAILS))
    gate("G8b_no_leak_late", "late_delayed", le("late_delayed", "shuffled_yoked", FAILS))
    gate("G9_dopamine_gates_learning", "credit", le("credit", "reward_to_da_lesioned", FAILS))

    g = {k: v["state"] for k, v in out["gates"].items()}
    if out["construct_validity"]["C1_trace_ablation_abolishes_credit"]["state"] == "PASS" and \
            g["G4_ablated_learner_competent"] != "PASS":
        out["C1_reading"] = "VOID as a horizon effect: the ablated learner did not learn with immediate reward"
    states = list(g.values())
    overall = ("PASS" if all(x == "PASS" for x in states) else
               "FAIL" if any(x == "FAIL" for x in states) else "UNDEFINED")
    out["verdict"] = {"PASS": "GO", "FAIL": "NO-GO", "UNDEFINED": "UNDEFINED"}[overall]

    def mean_succ(s, a):
        v = [succ(s, a, sd) for sd in sorted(defined[s])]
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else None

    for (s, name), arm in {("credit", "slow eligibility tag (equal integral)"): "trace_ablated_integral",
                           ("credit", "slow eligibility tag (equal peak)"): "trace_ablated_peak",
                           ("credit", "memory recurrence"): "memory_lesioned",
                           ("credit", "critic -V (striosome -> SNc)"): "critic_lesioned",
                           ("credit", "reward -> SNc"): "reward_to_da_lesioned",
                           ("late_delayed", "slow eligibility tag (equal integral)"): "trace_ablated_integral",
                           ("late_delayed", "memory recurrence"): "memory_lesioned",
                           ("late_immediate", "memory recurrence"): "memory_lesioned",
                           ("late_immediate", "slow eligibility tag (equal integral)"): "trace_ablated_integral"
                           }.items():
        full, abl = mean_succ(s, "full"), mean_succ(s, arm)
        n_fail = sum(1 for sd in defined[s] if succ(s, arm, sd) is not None and succ(s, arm, sd) <= FAILS)
        if full is None or abl is None:
            cls = "UNDEFINED"
        elif n_fail >= K_OF_6 and full >= CAPABLE:
            cls = "NECESSARY"
        elif full - abl >= DROP:
            cls = "CONTRIBUTING"
        else:
            cls = "NOT NECESSARY"
        # ATTRIBUTION (gates/attribution_required): the NECESSARY/CONTRIBUTING/NOT-NECESSARY split above already
        # asks whether the full-vs-ablated gap clears fixed bars (K_OF_6, DROP) -- attributable_to additionally
        # asks WHAT FRACTION of the full arm's own success rate survives the ablation, on the same two numbers,
        # informational only (does not feed `cls`, any gate above, or the registered verdict).
        attribution = attributable_to("%s | %s (full vs ablated success)" % (s, name), full, abl) \
            if full is not None and abl is not None else None
        out["mechanisms"]["%s | %s" % (s, name)] = {"class": cls, "full_mean": full, "ablated_mean": abl,
                                                   "n_seeds_failed": n_fail, "attribution_fraction": attribution}

    def s1(sd):
        r = row("credit", "misleading_final_cue", sd)
        if not valid(r):
            return None
        return r["success"] >= CAPABLE and abs(r.get("M_decoy") or 0) < abs(r.get("M_early") or 0)
    gate("S1_credit_survives_misleading_decoy", "credit", s1, into="signatures")
    gate("S2_held_cue_resists_decoy", "late_delayed", ge("late_delayed", "misleading_final_cue", CAPABLE),
         into="signatures")

    def s4(sd):
        r = row("credit", "full", sd)
        if not valid(r):
            return None
        t = r["test"]["terminal"]
        return t["response_rate"] >= RESP_FLOOR and (t["acc_among_responders"] or 0) > 0.5
    gate("S4_credit_misattributed_to_held_memory", "credit", s4, into="signatures")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--runner-sha256", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cells, hashes = load(a.paths)
    if hashes != {a.runner_sha256}:
        sys.exit("runner hash mismatch: %s vs preregistered %s" % (sorted(map(str, hashes)), a.runner_sha256))
    v = verdict(cells)
    json.dump(v, open(a.out, "w"), indent=1, default=str)
    print(json.dumps({"verdict": v["verdict"], "gates": {k: x["state"] for k, x in v["gates"].items()}}, indent=1))


if __name__ == "__main__":
    main()
