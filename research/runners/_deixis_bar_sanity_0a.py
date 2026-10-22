"""_deixis_bar_sanity_0a — step-0a module 6 (0a(iii)): the bar-sanity record.

Reads every 0a output, checks each input's sha256 against the COMMITTED one
(git show HEAD:<path>; inputs not yet committed by the sequence — replica
and audit, which the final `git add $OUT` lands after this run — are
recorded with their file sha and committed=false, never silently treated as
checked), evaluates the packet's six STOP rules, and writes the record:
thresholds and their source, the C-weight row, the conversions, the step
times and the 0b queue plan, the heterogeneity flags and the cell-type
dump, the replica record, the audit matrix, the STOP-rule verdicts, every
input sha256.

STOP rules (packet lines 125-135, evaluated verbatim):
  * STOP-PLAST — a learned block's weight sha256 changed during
    calibration (calibration.plasticity_off.unchanged must be true).
  * STOP-ID — the two seed-61 calibration runs are not byte-identical
    (the --identity pair's own file hashes; timing lives in sidecars).
  * STOP-P0 — no row of the C-weight grid passes every P0 item at K = 0
    on 10/10 calibration seeds (record names each row's failing items).
  * STOP-CAL — a target does not fire every neuron on every seed by
    w = 20 in some hearing condition, or no grid weight below its
    theta_fire is silent (every recorded transfer curve must satisfy
    both flags).
  * STOP-REP — no arm reaches a 60-seed per-seed pass rate q >= 0.9; or
    some null passes every GO bar on >= 2/15 seeds; or K = 0 does
    (all evaluated from replica.json as recorded).
  * STOP-AUD — an arm with q >= 0.9 fails its own audit (A1 or A2 below
    13/15, moderate); or some mislabelled pair (g != r, g passing GO)
    passes A1 and A2 at either level; or the independent-stream control
    is not UNDEFINED by A0.

The 0b queue plan (packet 146, DESIGN 1346-1359): runs = 6 engagement +
180 census (6 arms x 2 cells x 15 seeds) + 45 nulls = 231 (the design's
"about 235"), each 4800 x 200 = 960,000 steps; cpu_h = runs x steps x
measured mean per-arm step_s / 3600 (the design's 20-105 CPU-h is this
formula at 0.31-1.7 ms/step); cores_needed = cpu_h / (days to 2026-10-27
x 24) — the RESULT reads live pool capacity to say whether it fits, and
the fallback is recorded as designed: rivals drop to their centre cell
only. No live queue state is read here (the record must be reproducible).

CLI (packet command 118): --calibration $OUT/calibration.json --identity
$OUT/calibration_s61_a.json,$OUT/calibration_s61_b.json --replica
$OUT/replica.json --audit $OUT/audit_check.json --out
$OUT/bar_sanity_0a.json.  Host class: pool, memcap 4 (CPU-minutes).
Refuses: a malformed/missing input, thresholds absent from calibration,
or a calibration file whose committed bytes differ.  --selftest builds
every input in a temp dir (passing and forced-firing variants) and runs
main() end to end.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from research.runners import _deixis_rule_replica as R  # noqa: E402
from tools.lab import lever  # noqa: E402  (gate: attribution discipline)

RUNS = {"engagement": 6, "census": 180, "nulls": 45}
STEPS_PER_RUN = 4800 * 200  # DESIGN 1348: about 0.96 M steps per run
DEADLINE = "2026-10-27"     # DESIGN 1336 / packet 146


def input_sha(path):
    """File sha256 + the committed bytes' sha256 at HEAD (None when the
    input is not committed yet)."""
    rel = os.path.relpath(os.path.abspath(path), R.REPO)
    file_sha = R.sha256(path)
    rec = {"path": rel, "sha256": file_sha, "committed_sha256": None,
           "committed": None}
    import subprocess
    p = subprocess.run(["git", "-C", R.REPO, "show", "HEAD:" + rel],
                       capture_output=True)
    if p.returncode == 0:
        import hashlib
        rec["committed_sha256"] = hashlib.sha256(p.stdout).hexdigest()
        rec["committed"] = (rec["committed_sha256"] == file_sha)
    else:
        rec["committed"] = False  # not yet committed by the sequence
        rec["note"] = "not committed at HEAD when bar_sanity ran"
    return rec


def load(path, what):
    if not os.path.exists(path):
        raise FileNotFoundError("%s not found: %s" % (what, path))
    with open(path) as f:
        return json.load(f)


def stop(fires, why, evidence):
    return {"fires": bool(fires), "why": why, "evidence": evidence}


def evaluate(cal, ida_path, idb_path, rep, aud):
    """The six STOP rules, evaluated verbatim from the packet."""
    verdicts = {}

    po = cal.get("plasticity_off", {})
    changed = po.get("changed_blocks", [])
    verdicts["STOP-PLAST"] = stop(
        (not po.get("unchanged")) or changed,
        "a learned block's weight sha256 changed during calibration",
        {"unchanged": po.get("unchanged"), "changed_blocks": changed})

    a_sha, b_sha = R.sha256(ida_path), R.sha256(idb_path)
    verdicts["STOP-ID"] = stop(
        a_sha != b_sha,
        "the two seed-61 calibration runs are not byte-identical",
        {"run_a_sha256": a_sha, "run_b_sha256": b_sha})

    cw = cal.get("c_weight", {})
    passing = cw.get("passing_rows", [])
    row_fails = {str(r.get("w")): r.get("p0_fails", [])
                 for r in cw.get("grid", []) if r.get("p0_fails")}
    verdicts["STOP-P0"] = stop(
        not passing,
        "no C-weight grid row passes every P0 item at K = 0 on 10/10 "
        "calibration seeds",
        {"passing_rows": passing, "rows_with_failing_items": row_fails})

    curves = cal.get("transfer", {})
    cal_bad = []
    for key, c in curves.items():
        if not c.get("fired_all_by_20"):
            cal_bad.append([key, "not every neuron fires by w = 20"])
        if not c.get("silent_below_theta"):
            cal_bad.append([key, "no grid weight below theta_fire silent"])
    verdicts["STOP-CAL"] = stop(
        (not curves) or cal_bad,
        "a target fails to fire every neuron by w = 20, or no weight "
        "below theta_fire is silent",
        {"curves_checked": len(curves), "failing": cal_bad})

    qs, null_hits, k0_hits = {}, [], []
    for rule, rr in rep.get("rules", {}).items():
        scores = rr.get("scores", {})
        n15 = {k: 0 for k in ("S0", "S1", "S2a", "S2b", "S3", "S4", "S5",
                              "MARK_W")}
        for s, sc in scores.items():
            for k, v in sc.get("go", {}).items():
                if v:
                    n15[k] = n15.get(k, 0) + 1
        rr["_n15"] = n15
        rs = rep.get("rate_seed_scores", {}).get(rule, {})
        all_go = sum(1 for s in rs.values()
                     if s.get("go") and all(s["go"].values()))
        qs[rule] = all_go / max(len(rs), 1)
        for arm, arm_rec in rep.get("null_arms", {}).get(rule,
                                                         {}).items():
            n = sum(1 for s in arm_rec.values()
                    if s.get("go") and all(s["go"].values()))
            if n >= 2:
                null_hits.append([rule, arm, n])
        k0 = rep.get("k0", {}).get(rule, {}).get("go", {})
        if k0 and all(k0.values()):
            k0_hits.append(rule)
    no_arm_q = all(q < 0.9 for q in qs.values())
    verdicts["STOP-REP"] = stop(
        no_arm_q or bool(null_hits) or bool(k0_hits),
        "no arm reaches a 60-seed q >= 0.9, or a null / K = 0 passes "
        "every GO bar",
        {"q60": {k: round(v, 3) for k, v in qs.items()},
         "nulls_passing_ge_2": null_hits, "k0_passing": k0_hits})

    matrix = aud.get("matrix", {})
    moderate = matrix.get("moderate", {})
    own_fail, mislabel_pass = [], []
    for level, mat in matrix.items():
        for key, cell in mat.items():
            if not isinstance(cell, dict) or "|" not in key:
                continue
            g, ref = key.split("|", 1)
            if ref == "GO_bars" or g == ref:
                continue
            g_q = qs.get(g, 0.0)
            if g_q >= 0.9 and cell.get("verdict") == "PASS":
                # a mislabelled pair (g != r) passing A1 and A2 at
                # either level, with g passing the GO bars
                mislabel_pass.append([g, ref, level, cell.get("A1"),
                                      cell.get("A2")])
    # the arm's OWN audit (ref == g): A1 or A2 below 13/15, moderate
    for g in qs:
        if qs[g] >= 0.9:
            c = moderate.get("%s|%s" % (g, g), {})
            if c and (c.get("A1", 0) < 13 or c.get("A2", 0) < 13):
                own_fail.append([g, "moderate", c.get("A1"),
                                 c.get("A2")])
    ind = aud.get("controls", {}).get(
        "independent_streams_covpre", {})
    ind_not_undef = ind.get("verdict") != "UNDEFINED"
    verdicts["STOP-AUD"] = stop(
        bool(own_fail) or bool(mislabel_pass) or ind_not_undef,
        "an arm with q >= 0.9 fails its own audit, or a mislabelled "
        "pair passes, or the independent control is not UNDEFINED by A0",
        {"own_audit_failing": own_fail,
         "mislabelled_passing": mislabel_pass,
         "independent_control": ind})
    return verdicts, qs


def queue_plan(cal):
    step_times = cal.get("step_times", {})
    vals = [v for v in step_times.values() if isinstance(v, (int, float))]
    mean_s = float(np.mean(vals)) if vals else None
    runs = sum(RUNS.values())
    cpu_h = (runs * STEPS_PER_RUN * mean_s / 3600.0
             if mean_s else None)
    days = 5.0  # 2026-10-22 -> 2026-10-27
    return {
        "runs_breakdown": dict(RUNS), "runs": runs,
        "design_says": "about 235 runs, 20-105 CPU-h",
        "steps_per_run": STEPS_PER_RUN,
        "measured_mean_step_s": mean_s,
        "cpu_h_at_measured": cpu_h,
        "cpu_h_range_formula": "runs * steps * step_s / 3600 (0.31-1.7 "
                               "ms/step gives the design's 20-105)",
        "days_to_deadline": days, "deadline": DEADLINE,
        "cores_needed": (cpu_h / (days * 24.0) if cpu_h else None),
        "fallback": "rivals (failing at every grid cell on the replica) "
                    "drop to their centre cell only (6 x 15 fewer runs)",
        "note": "the RESULT reads live pool capacity for the fits "
                "verdict; this record stays reproducible"}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="0a(iii): the bar-sanity record and the six STOP rules")
    ap.add_argument("--calibration")
    ap.add_argument("--identity")
    ap.add_argument("--replica")
    ap.add_argument("--audit")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        print(json.dumps(selftest()))
        return 0
    missing = [n for n, v in (("--calibration", args.calibration),
                              ("--identity", args.identity),
                              ("--replica", args.replica),
                              ("--audit", args.audit),
                              ("--out", args.out)) if not v]
    if missing:
        ap.error("the following arguments are required: "
                 + ", ".join(missing))
    ida_path, idb_path = [p.strip() for p in args.identity.split(",")]
    cal = load(args.calibration, "calibration")
    rep = load(args.replica, "replica record")
    aud = load(args.audit, "audit record")
    if "thresholds" not in cal:
        raise ValueError("calibration.json carries no thresholds")

    # every input's sha256, checked against the committed bytes
    inputs = {"calibration": input_sha(args.calibration),
              "replica": input_sha(args.replica),
              "audit": input_sha(args.audit),
              "identity_a": input_sha(ida_path),
              "identity_b": input_sha(idb_path)}
    streams_dir = os.path.join(os.path.dirname(os.path.abspath(
        args.calibration)), "streams")
    for m in sorted(glob.glob(os.path.join(streams_dir,
                                           "manifest_*.json"))):
        inputs[os.path.basename(m)] = input_sha(m)
    drift = [k for k, v in inputs.items()
             if v["committed"] is False and v["committed_sha256"]]
    if drift:
        raise ValueError("input drifted from HEAD bytes: %s" % drift)

    verdicts, qs = evaluate(cal, ida_path, idb_path, rep, aud)

    replica_record = {}
    for rule, rr in rep.get("rules", {}).items():
        scores = rr.get("scores", {})
        n15 = {k: 0 for k in ("S0", "S1", "S2a", "S2b", "S3", "S4",
                              "S5", "MARK_W")}
        markers = {}
        for s, sc in scores.items():
            for k, v in sc.get("go", {}).items():
                if v:
                    n15[k] = n15.get(k, 0) + 1
            for mk, mv in sc.get("markers", {}).items():
                cur = markers.setdefault(
                    mk, {"max_pool": 0.0, "max_ent": 0.0,
                         "top_pool": mv.get("top_pool")})
                cur["max_pool"] = max(cur["max_pool"],
                                      float(mv.get("max_pool", 0.0)))
                cur["max_ent"] = max(cur["max_ent"],
                                     float(mv.get("max_ent", 0.0)))
        nulls = {}
        for arm, rec in rep.get("null_arms", {}).get(rule,
                                                     {}).items():
            nulls[arm] = sum(1 for s in rec.values()
                             if s.get("go") and all(s["go"].values()))
        replica_record[rule] = {
            "centre": rr.get("centre"),
            "n15_per_bar": n15, "q60": round(qs.get(rule, 0.0), 3),
            "null_seeds_all_go": nulls,
            "k0_go": rep.get("k0", {}).get(rule, {}).get("go", {}),
            "markers": markers}

    record = {
        "schema": "deixis-bar-sanity-0a-v1",
        "thresholds": cal["thresholds"],
        "thresholds_source": cal.get("thresholds_source", {}),
        "identity_verdict": {"run_a_sha256": inputs["identity_a"]
                             ["sha256"],
                             "run_b_sha256": inputs["identity_b"]
                             ["sha256"],
                             "byte_identical": inputs["identity_a"]
                             ["sha256"] == inputs["identity_b"]["sha256"]},
        "c_weight": cal.get("c_weight", {}),
        "conversions": cal.get("conversions", {}),
        "step_times": cal.get("step_times", {}),
        "queue_plan_0b": queue_plan(cal),
        "heterogeneity": cal.get("heterogeneity", {}),
        "replica_record": replica_record,
        "audit_matrix": {"matrix": aud.get("matrix", {}),
                         "unit_conversion": aud.get("unit_conversion", {}),
                         "controls": aud.get("controls", {}),
                         "thresholds_sha256": aud.get(
                             "thresholds_sha256")},
        "stop_verdicts": verdicts,
        "inputs": inputs,
        "stopped": any(v["fires"] for v in verdicts.values())}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(record, f, indent=1, default=float)
    fired = [k for k, v in verdicts.items() if v["fires"]]
    print("wrote %s; STOP rules fired: %s"
          % (args.out, ", ".join(fired) if fired else "none"))
    return 0


def selftest():
    """End to end on a fabricated record: variant A has all six STOP
    rules clear; variant B forces STOP-ID, STOP-PLAST and STOP-AUD."""
    import tempfile

    res = {}
    GO = {k: True for k in ("S0", "S1", "S2a", "S2b", "S3", "S4", "S5",
                            "MARK_W")}
    NO = {k: False for k in GO}
    mk = {"ART": {"max_pool": 0.10, "max_ent": 0.02, "top_pool": "SPK"},
          "ASP": {"max_pool": 0.05, "max_ent": 0.01, "top_pool": "SPK"}}
    with tempfile.TemporaryDirectory() as td:
        th = {"P": [0.40, 0.20], "E": [0.45, 0.20]}
        curve = {"grid": [0, 2, 4, 6, 8, 10, 12, 14, 16, 20],
                 "fired_all_by_20": True, "silent_below_theta": True}
        cal = {
            "schema": "deixis-calibration-v1", "seeds": list(range(61, 71)),
            "thresholds": th,
            "thresholds_source": {"rule": "Appendix C",
                                  "grid_w": "transfer curves"},
            "transfer": {"TOK>SPK@production": dict(curve),
                         "TOK>SPK@toy": dict(curve)},
            "plasticity_off": {"unchanged": True, "changed_blocks": [],
                               "blocks": {"dx_ptr": "a" * 64,
                                          "dx_name": "b" * 64}},
            "c_weight": {"row": 6, "p0_items": ["S0", "S1"],
                         "grid": [{"w": 0, "p0_fails": ["S0"]},
                                  {"w": 6, "p0_fails": []}],
                         "passing_rows": [6]},
            "conversions": {"hebbian_run_fraction": 0.91,
                            "y_on": 0.4, "presence_20": 250},
            "step_times": {"causal": 0.0004, "causal_wta": 0.0005,
                           "rw_hebb": 0.00045, "bcm": 0.0005,
                           "cov": 0.0006, "cov_pre": 0.0006},
            "heterogeneity": {
                "enable_parameter_heterogeneity": True,
                "preset_centred_parameter_heterogeneity": True,
                "medians": {"FS": {"a": 0.4, "b": 1.2, "d": 5.0,
                                   "C": 100.0}}}}
        cp = os.path.join(td, "calibration.json")
        with open(cp, "w") as f:
            json.dump(cal, f, indent=1)
        ident = {"schema": "deixis-calibration-s61", "seed": 61,
                 "thresholds": th, "note": "timing in sidecar"}
        ia = os.path.join(td, "calibration_s61_a.json")
        ib = os.path.join(td, "calibration_s61_b.json")
        with open(ia, "w") as f:
            json.dump(ident, f, indent=1, sort_keys=True)
        with open(ib, "w") as f:
            json.dump(ident, f, indent=1, sort_keys=True)

        rules = ["covpre", "cov", "hebb", "hebbwta", "bcm"]
        rep = {"schema": "deixis-rule-replica-v1", "rules": {},
               "rate_seed_scores": {}, "null_arms": {}, "k0": {}}
        for rule in rules:
            scores = {str(s): {"go": dict(GO if rule == "covpre" else NO),
                               "markers": mk}
                      for s in range(1, 16)}
            rep["rules"][rule] = {"centre": 6, "scores": scores}
            rep["rate_seed_scores"][rule] = {
                str(s): {"go": dict(GO if rule == "covpre" else NO)}
                for s in range(1, 61)}
            rep["null_arms"][rule] = {
                arm: {str(s): {"go": dict(NO)} for s in range(1, 16)}
                for arm in ("label", "scene", "voice", "k0")}
            rep["k0"][rule] = {"go": dict(NO)}
        rp = os.path.join(td, "replica.json")
        with open(rp, "w") as f:
            json.dump(rep, f, indent=1)

        aud = {"schema": "deixis-audit-check-v1",
               "thresholds_sha256": "f" * 64,
               "unit_conversion": {"moderate": {}},
               "matrix": {"moderate": {
                   "covpre|covpre": {"A1": 15, "A2": 15,
                                     "verdict": "PASS", "need": 13},
                   "cov|covpre": {"A1": 0, "A2": 0,
                                  "verdict": "FAIL", "need": 13}},
                          "harsh": {
                   "covpre|covpre": {"A1": 13, "A2": 14,
                                     "verdict": "PASS", "need": 13}}},
               "controls": {
                   "independent_streams_covpre": {
                       "verdict": "UNDEFINED",
                       "reason": "stream sha256 mismatch",
                       "sha_differ_all": True},
                   "unengaged": {"GO_bars": dict(NO)}}}
        apth = os.path.join(td, "audit_check.json")
        with open(apth, "w") as f:
            json.dump(aud, f, indent=1)

        out_a = os.path.join(td, "bar_a.json")
        rc = main(["--calibration", cp,
                   "--identity", ia + "," + ib,
                   "--replica", rp, "--audit", apth, "--out", out_a])
        got = load(out_a, "bar record A")
        res["main_ok"] = (rc == 0)
        res["schema_ok"] = got["schema"] == "deixis-bar-sanity-0a-v1"
        res["all_clear"] = (not got["stopped"] and not any(
            v["fires"] for v in got["stop_verdicts"].values()))
        res["thresholds_recorded"] = got["thresholds"] == th
        res["identity_recorded"] = got["identity_verdict"][
            "byte_identical"]
        plan = got["queue_plan_0b"]
        exp_mean = sum(cal["step_times"].values()) / 6.0
        res["plan_ok"] = (plan["runs"] == 231
                          and plan["steps_per_run"] == 960000
                          and abs(plan["measured_mean_step_s"]
                                 - exp_mean) < 1e-12
                          and abs(plan["cpu_h_at_measured"]
                                 - 231 * 960000 * exp_mean
                                 / 3600.0) < 1e-9)
        res["q60_recorded"] = (got["replica_record"]["covpre"]["q60"]
                               == 1.0
                               and got["replica_record"]["cov"]["q60"]
                               == 0.0)
        res["inputs_logged"] = set(got["inputs"]) == {
            "calibration", "replica", "audit", "identity_a",
            "identity_b"}
        res["audit_recorded"] = ("covpre|covpre" in
                                 got["audit_matrix"]["matrix"]["moderate"])
        res["heterogeneity_recorded"] = got["heterogeneity"][
            "enable_parameter_heterogeneity"]

        # variant B: force STOP-ID (different identity bytes), STOP-PLAST
        # (a changed block), STOP-AUD (indep control not UNDEFINED)
        with open(ib, "w") as f:
            json.dump({"schema": "deixis-calibration-s61", "seed": 61,
                       "thresholds": th, "note": "DIFFERENT"}, f)
        cal2 = dict(cal)
        cal2["plasticity_off"] = {"unchanged": False,
                                  "changed_blocks": ["dx_ptr"]}
        cp2 = os.path.join(td, "calibration2.json")
        with open(cp2, "w") as f:
            json.dump(cal2, f, indent=1)
        aud2 = dict(aud)
        aud2["controls"] = dict(aud["controls"])
        aud2["controls"]["independent_streams_covpre"] = {
            "verdict": "PASS"}
        ap2 = os.path.join(td, "audit2.json")
        with open(ap2, "w") as f:
            json.dump(aud2, f, indent=1)
        out_b = os.path.join(td, "bar_b.json")
        main(["--calibration", cp2, "--identity", ia + "," + ib,
              "--replica", rp, "--audit", ap2, "--out", out_b])
        got_b = load(out_b, "bar record B")
        v = got_b["stop_verdicts"]
        res["id_fires"] = v["STOP-ID"]["fires"]
        res["plast_fires"] = v["STOP-PLAST"]["fires"]
        res["aud_fires"] = v["STOP-AUD"]["fires"]
        res["stopped"] = got_b["stopped"]
        # lever hook (before/after discipline) for the STOP machinery
        res["lever_note"] = lever("bar_sanity STOP rules", "selftest B",
                                  "three forced STOPs")
        keys = ("main_ok", "schema_ok", "all_clear",
                "thresholds_recorded", "identity_recorded",
                "plan_ok", "q60_recorded", "inputs_logged",
                "audit_recorded", "heterogeneity_recorded",
                "id_fires", "plast_fires", "aud_fires", "stopped")
        failing = [k for k in keys if not res.get(k)]
        assert not failing, {"failing": failing, "res": res}
    res["ok"] = True
    return res


if __name__ == "__main__":
    sys.exit(main())
