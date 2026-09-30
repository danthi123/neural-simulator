#!/usr/bin/env python3
"""iw1_score.py -- the registered scorer for integration wave 1 (the three flip packets). Read-only; builds no brain.

Governing text: research/findings/2026-09-29-iw1-wave-battery-PREREGISTRATION.md and the three packet preregs
(research/findings/2026-09-29-iw1-{d6-hebbian-write,wm-focus-bind,xedge-surprise-worldmodel}-flip-PREREGISTRATION.md).
Every cell-validity rule is tools/b2d_score.py's (A1.2 rules 1-3 and 7: sidecars pinned to the registered revision,
git_archive source, manifest verified at start and exit, numpy, no organ error on the probe turn), reused by import.
The one relaxation: a CAPABILITY shard (step R) sets its packet's own flags, so BRAIN_* keys that are exactly the
packet's flags (and its lesion flag in the lesion arms) are allowed there; any other BRAIN_* key still fails the cell.

  R  (reply-level capability, pre-flip, at pin P):
     python tools/iw1_score.py R --packet d6-hebbian-write          --tag iw1-R-d6-<P7>  --pin <P>
     python tools/iw1_score.py R --packet xedge-surprise-worldmodel --tag iw1-R-swm-<P7> --pin <P>
     python tools/iw1_score.py R --packet wm-focus-bind             --tag iw1-R-wmf-<P7> --pin <P>
  B  (combined no-regression, production default at F, against the reference production-default battery
     b2b0924-base @ a308f1e09 -- the 2026-09-23 flipdefaults design: compare, then attribute any drop):
     python tools/iw1_score.py B --base-tag b2b0924-base --base-pin a308f1e09babcc9ed096c3c8046d00040391368c
         --flip-tag iw1-flip-<F7> --flip-pin <F> --flipped d6-hebbian-write,wm-focus-bind,xedge-surprise-worldmodel
  A  (attribute every B1 drop: the dropped faculties' cells at F with the flips forced off, vs the flip battery):
     python tools/iw1_score.py A --b-result <B.json> --tag iw1-alloff-<F7> --pin <F>     (then per packet, bisect)
  X  (one faculty's cells in any such run, for reading): python tools/iw1_score.py X --tag <tag> --pin <F> --faculty <X>

Exit code: 0 PASS, 1 FAIL, 3 UNDEFINED (a cell or precondition that is not DEFINED; never counted as a pass).
Shard layout: research/findings/raw/_load_bearing/_shards/<tag>/s<seed>/<faculty dir>/{lb.json, arm files, sidecars}.
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b2d_score as B  # noqa: E402

SEEDS = [42, 43, 44, 100, 101, 102]
BASE = "research/findings/raw/_load_bearing/_shards"
PACKETS = {
    "d6-hebbian-write": {"row": "d6-hebbian-write", "shard_dir": "d6-hebbian-write",
                         "flags": ["BRAIN_D6_HEBBIAN_STORE", "BRAIN_D6_ENGRAM_VOCAB", "BRAIN_D6_ENGRAM_READTIME"],
                         "lesion_flags": ["BRAIN_D6_HEBBIAN_FREEZE"], "probe_message": "what does the wolf hunt"},
    "xedge-surprise-worldmodel": {"row": "xedge-surprise-worldmodel", "shard_dir": "xedge-surprise-worldmodel",
                                  "flags": ["BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL"], "lesion_flags": [],
                                  "probe_message": "what do you expect?"},
    "wm-focus-bind": {"row": "wm-binding-referent-focus", "shard_dir": "wm-binding-advanced",
                      "flags": ["BRAIN_MULTIREF_FOCUS_BIND", "BRAIN_MULTIREF_LESION_SCOPE"],
                      "lesion_flags": ["BRAIN_MULTIREF_LESION"]},
}
PASS, FAIL, UNDEF = 0, 1, 3


def _allowing(allowed):
    """A sidecar_fails that tolerates BRAIN_* keys inside `allowed` (and still fails on any other BRAIN_* key)."""
    orig = B.sidecar_fails

    def f(pj, pin, is_lb):
        env = dict(pj.get("env") or {})
        pj2 = dict(pj, env={k: v for k, v in env.items() if k not in allowed})
        return orig(pj2, pin, is_lb)
    return f


def _norm(text):
    return re.sub(r"[^a-z0-9 ]+", "", str(text).lower()).strip()


def degenerate_intact(canon, probe_message=None):
    """Why the intact reply on the probe turn cannot carry a capability (adversarial review 2026-09-29): an empty or
    absent answer, an abstention, or an echo of the probe message. A lesion that 'changes' such a reply is not the
    faculty's contribution, so a would-be `regressed` cell with one of these is UNDEFINED (never counted toward a
    PASS). An intact abstention the lesion leaves unchanged stays `pass` (a real negative); this only blocks passes."""
    turn = canon.get("turn")
    out = []
    for k in ("intact_a", "intact_b"):
        t = ((canon.get("arms") or {}).get(k) or {}).get(turn) or {}
        ans = t.get("answer")
        if not isinstance(ans, str) or not ans.strip():
            out.append("%s: empty or absent answer" % k)
        elif t.get("abstained") is True:
            out.append("%s: abstained" % k)
        elif probe_message and _norm(ans) == _norm(probe_message):
            out.append("%s: echoes the probe message" % k)
    return out


def _cell_std(d, fac, pin, allowed, probe_message=None, guard_degenerate=False):
    """b2d_score.judge_dir with the capability-shard relaxation; returns (state, s2_verdict, detail). With
    `guard_degenerate` (step R and F1: pass criteria), a `regressed` cell whose intact reply is degenerate is
    UNDEFINED:degenerate-intact."""
    saved = B.sidecar_fails
    B.sidecar_fails = _allowing(set(allowed))
    try:
        canon = B.judge_dir(d, fac, pin)
    finally:
        B.sidecar_fails = saved
    state = B.final_state(canon, None, None, None)
    if state != "DEFINED":
        return state, None, {"fails": canon["fails"]}
    if canon["kind"] not in B.COVERABLE:
        return "UNDEFINED:kind-%s" % canon["kind"], None, {}
    s2, det = B.score_s2(canon, None)
    det = dict(det, s0=[canon["verdict"], canon["load_bearing"]])
    if guard_degenerate and s2 == "regressed":
        why = degenerate_intact(canon, probe_message)
        if why:
            return "UNDEFINED:degenerate-intact", None, dict(det, fails=why)
    return state, s2, det


def _cell_wmf(d, pin, allowed):
    """The focus-bind probe's own record (kind neural-lesion-opt-in; its arm files are not b2d-shaped)."""
    lb = os.path.join(d, "lb.json")
    if not os.path.exists(lb):
        return "MISSING", None, {"fails": ["no lb.json"]}
    fails = []
    for sc in sorted(glob.glob(os.path.join(d, "*.prov.json"))):
        is_lb = os.path.basename(sc) == "lb.json.prov.json"
        fails += ["%s: %s" % (os.path.basename(sc), x) for x in _allowing(set(allowed))(B._j(sc), pin, is_lb)]
    if not os.path.exists(lb + ".prov.json"):
        fails.append("lb.json has no sidecar")
    rows = [p for p in B._j(lb).get("per_faculty", []) if p.get("faculty") == "wm-binding-referent-focus"]
    if len(rows) != 1:
        return "MISSING", None, {"fails": fails + ["lb.json holds %d focus rows" % len(rows)]}
    p = rows[0]
    if fails:
        return "UNDEFINED:provenance", None, {"fails": fails}
    on, off = p.get("wmf_gate_on") or {}, p.get("wmf_gate_off") or {}
    det = {"verdict": p.get("verdict"), "on": on.get("verdict"), "off": off.get("verdict"),
           "failing_direction_ok": p.get("failing_direction_ok")}
    if p.get("load_bearing") is True and p.get("verdict") == "regressed" and on.get("verdict") == "regressed" \
            and off.get("load_bearing") is not True and p.get("failing_direction_ok") is True:
        return "DEFINED", "regressed", det
    if p.get("load_bearing") is False and on.get("verdict") == "pass":
        return "DEFINED", "pass", det
    return "UNDEFINED:%s" % p.get("verdict"), None, det


def score_R(packet, tag, pin, base=BASE, seeds=SEEDS):
    spec = PACKETS[packet]
    allowed = spec["flags"] + spec["lesion_flags"]
    per = {}
    for s in seeds:
        d = os.path.join(base, tag, "s%d" % s, spec["shard_dir"])
        if packet == "wm-focus-bind":
            st, v, det = _cell_wmf(d, pin, allowed)
        else:
            st, v, det = _cell_std(d, spec["row"], pin, allowed, probe_message=spec.get("probe_message"),
                                   guard_degenerate=True)
        per[str(s)] = {"state": st, "reply_verdict": v, "detail": det}
    n_def = sum(1 for c in per.values() if c["state"] == "DEFINED" and c["reply_verdict"] in ("regressed", "pass"))
    n_lb = sum(1 for c in per.values() if c["state"] == "DEFINED" and c["reply_verdict"] == "regressed")
    if n_def < len(seeds):
        verdict, code = "UNDEFINED", UNDEF
    elif n_lb == len(seeds):
        verdict, code = "PASS", PASS
    else:
        verdict, code = "FAIL", FAIL
    return {"step": "R", "packet": packet, "tag": tag, "pin": pin, "per_seed": per, "n_defined": n_def,
            "n_reply_load_bearing": n_lb, "verdict": verdict}, code


ALL_FLAGS = sorted({f for p in PACKETS.values() for f in p["flags"] + p["lesion_flags"]})


def score_faculty(tag, pin, faculty, base=BASE, seeds=SEEDS, allowed=None):
    """One registry faculty's cells in an attribution run (its job env forces packet flags to 0, so those BRAIN_*
    keys are allowed in the sidecars). Returns per-seed S0/S2 verdicts and the load-bearing counts at both levels;
    a count is UNDEFINED (None) when any seed's cell is not DEFINED."""
    allowed = ALL_FLAGS if allowed is None else allowed
    per, n0, n2, ok = {}, 0, 0, True
    for s in seeds:
        d = os.path.join(base, tag, "s%d" % s, faculty)
        st, v2, det = _cell_std(d, faculty, pin, allowed)
        s0 = (det.get("s0") or [None, None])
        per[str(s)] = {"state": st, "s0": s0, "s2": v2, "detail": det if st != "DEFINED" else {}}
        if st != "DEFINED" or v2 not in ("regressed", "pass") or s0[0] not in B.EXERCISED:
            ok = False
            continue
        n0 += 1 if s0[1] is True else 0
        n2 += 1 if v2 == "regressed" else 0
    return {"step": "X", "tag": tag, "pin": pin, "faculty": faculty, "per_seed": per,
            "s0_n_load_bearing": n0 if ok else None, "s2_n_load_bearing": n2 if ok else None,
            "verdict": "DEFINED" if ok else "UNDEFINED"}, (PASS if ok else UNDEF)


def _counts(res, level):
    return {f: v["n_load_bearing"] for f, v in (res.get(level) or {}).get("per_faculty", {}).items()}


def _lo(v):
    return v if isinstance(v, int) else v[0]


def score_B(base_tag, base_pin, flip_tag, flip_pin, flipped, base=BASE, seeds=SEEDS):
    b = B.score(base, base_tag, base_pin, seeds)
    f = B.score(base, flip_tag, flip_pin, seeds)
    out = {"step": "B", "base": {"tag": base_tag, "pin": base_pin, "verdict": b["verdict"],
                                  "not_defined": b["not_defined"], "preconditions": b["preconditions"]},
           "flip": {"tag": flip_tag, "pin": flip_pin, "verdict": f["verdict"], "not_defined": f["not_defined"],
                    "preconditions": f["preconditions"]}, "flipped": flipped}
    undefined, regressions = [], []
    for name, r in (("base", b), ("flip", f)):
        if [p for p in r["preconditions"] if p["ok"] is not True]:
            undefined.append("%s: scorer precondition failed" % name)
    if f["verdict"] != "COMPLETE":
        undefined.append("B2: the flip battery is %s (not-DEFINED cells: %s)" % (f["verdict"], sorted(f["not_defined"])))
    for level in ("s0", "s2"):
        bc, fc = _counts(b, level), _counts(f, level)
        for fac, n in sorted(bc.items()):
            if fac not in fc:
                out.setdefault("reference_only_faculties", []).append(fac)   # left the registry since; reported
                continue
            nb, nf = n, fc[fac]
            if not isinstance(nb, int):
                # the reference battery's own cell was not DEFINED (e.g. B2b's comprehension-monitor organ crash):
                # nothing to compare against -- reported, never scored
                out.setdefault("reference_undefined_faculties", []).append("%s %s: %s" % (level, fac, nb))
                continue
            if not isinstance(nf, int):
                undefined.append("%s %s: flip battery has undefined cells (%s)" % (level, fac, nf))
                continue
            if nf < nb:
                regressions.append({"level": level, "faculty": fac, "reference": nb, "flip": nf,
                                    "msg": "%s %s: %d -> %d load-bearing seeds" % (level, fac, nb, nf)})
        out["%s_base" % level], out["%s_flip" % level] = bc, fc
    new_rows = {"d6-hebbian-write": "d6-hebbian-write", "xedge-surprise-worldmodel": "xedge-surprise-worldmodel"}
    rows_F = {}
    for p in flipped:
        if p in new_rows:
            n = _counts(f, "s2").get(new_rows[p])
            if isinstance(n, int):
                # the same guarded per-seed read as step R (a degenerate intact reply never counts toward F1)
                fr, _fcode = score_R(p, flip_tag, flip_pin, base=base, seeds=seeds)
                out.setdefault("f1_cells", {})[p] = fr["per_seed"]
                if fr["n_defined"] < len(seeds):
                    n = "UNDEFINED (%d/%d seeds DEFINED after the degenerate-intact guard)" % (fr["n_defined"],
                                                                                            len(seeds))
                else:
                    n = fr["n_reply_load_bearing"]
            rows_F[p] = n
            if n is None:
                undefined.append("F %s: its row is absent from the flip battery (REQUIRED_ENV not lifted?)" % p)
            elif not isinstance(n, int):
                undefined.append("F %s: undefined cells %s" % (p, n))
            elif n != len(seeds):
                regressions.append({"level": "F1", "faculty": new_rows[p], "reference": len(seeds), "flip": n,
                                    "msg": "F1 %s: reply-level load-bearing on %d/%d seeds at the production default"
                                           % (p, n, len(seeds))})
    out["new_rows_s2_at_flip"] = rows_F
    out["regressions"], out["undefined"] = regressions, undefined
    if regressions:
        out["verdict"], code = "FAIL", FAIL
    elif undefined:
        out["verdict"], code = "UNDEFINED", UNDEF
    else:
        out["verdict"], code = "PASS", PASS
    return out, code


def score_A(b_result, tag, pin, base=BASE, seeds=SEEDS):
    """Attribute every B1 regression in a saved `B --out` JSON: the faculty's cells in `tag` (an all-off or a
    one-packet-off run at the same F) against its count in the flip battery. excused = the off run is at or below the
    flip count at both levels (the flips did not cause the drop); flip-caused = the off run is higher at either level.
    F1 regressions are never excused (a flipped row that is not reply-load-bearing 6/6 at the default fails)."""
    rows, verdicts = {}, []
    for r in b_result.get("regressions", []):
        if r["level"] == "F1":
            verdicts.append("FAIL")
            rows[r["faculty"]] = {"verdict": "F1-not-excusable", "regression": r}
            continue
        fac = r["faculty"]
        if fac in rows:
            continue
        x, _code = score_faculty(tag, pin, fac, base=base, seeds=seeds)
        f0 = (b_result.get("s0_flip") or {}).get(fac)
        f2 = (b_result.get("s2_flip") or {}).get(fac)
        if x["verdict"] != "DEFINED" or not isinstance(f0, int) or not isinstance(f2, int):
            v = "UNDEFINED"
        elif x["s0_n_load_bearing"] <= f0 and x["s2_n_load_bearing"] <= f2:
            v = "excused"
        else:
            v = "flip-caused"
        rows[fac] = {"verdict": v, "off_run": {"s0": x["s0_n_load_bearing"], "s2": x["s2_n_load_bearing"]},
                     "flip": {"s0": f0, "s2": f2}}
        verdicts.append(v)
    b_undefined = list(b_result.get("undefined") or [])
    if any(v in ("flip-caused", "FAIL") for v in verdicts):
        verdict, code = "FAIL", FAIL
    elif b_undefined or any(v == "UNDEFINED" for v in verdicts):
        # adversarial review 2026-09-29: excusing every drop says nothing about B's own not-DEFINED criteria (B2
        # completeness, an undefined flip cell): A can never turn a B that was also UNDEFINED into a pass
        verdict, code = "UNDEFINED", UNDEF
    else:
        verdict, code = "PASS", PASS
    return {"step": "A", "tag": tag, "pin": pin, "faculties": rows, "b_undefined": b_undefined,
            "verdict": verdict}, code


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="step", required=True)
    r = sub.add_parser("R")
    r.add_argument("--packet", required=True, choices=sorted(PACKETS))
    r.add_argument("--tag", required=True)
    r.add_argument("--pin", required=True)
    b = sub.add_parser("B")
    b.add_argument("--base-tag", required=True)
    b.add_argument("--base-pin", required=True)
    b.add_argument("--flip-tag", required=True)
    b.add_argument("--flip-pin", required=True)
    b.add_argument("--flipped", required=True, help="comma-separated packet names flipped at --flip-pin")
    at = sub.add_parser("A")
    at.add_argument("--b-result", required=True, help="the JSON written by `B --out`")
    at.add_argument("--tag", required=True, help="iw1-alloff-<F7> or iw1-bisect-<packet>-<F7>")
    at.add_argument("--pin", required=True)
    x = sub.add_parser("X")
    x.add_argument("--tag", required=True)
    x.add_argument("--pin", required=True)
    x.add_argument("--faculty", required=True)
    for p in (r, b, x, at):
        p.add_argument("--base", default=BASE)
        p.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.step == "R":
        res, code = score_R(a.packet, a.tag, a.pin, base=a.base)
    elif a.step == "A":
        res, code = score_A(json.load(open(a.b_result)), a.tag, a.pin, base=a.base)
    elif a.step == "X":
        res, code = score_faculty(a.tag, a.pin, a.faculty, base=a.base)
    else:
        flipped = [x for x in a.flipped.split(",") if x]
        res, code = score_B(a.base_tag, a.base_pin, a.flip_tag, a.flip_pin, flipped, base=a.base)
    # provenance + device for the committed verdict file (gates artifact_provenance / device_and_cost): this scorer
    # builds no brain; every cell it counts had to carry SIM_BACKEND=numpy in its sidecars (b2d_score rule 3)
    res.update(runner="tools/iw1_score.py", argv=list(sys.argv if argv is None else argv),
               sim_backend="numpy (the scored shards'; this scorer builds no brain)")
    print(json.dumps({k: res[k] for k in res if k not in ("s0_base", "s0_flip", "s2_base", "s2_flip")},
                     indent=1, default=str))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1, default=str)
    return code


if __name__ == "__main__":
    sys.exit(main())
