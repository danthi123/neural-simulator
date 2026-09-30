#!/usr/bin/env python3
"""iw2_score.py -- the registered scorer for integration wave 2 (three flip packets). Read-only; builds no brain.

Governing text: research/findings/2026-09-30-iw2-wave-battery-PREREGISTRATION.md and the three packet preregs
(research/findings/2026-09-30-iw2-{false-belief-chat,transitive-chat,neural-selfid}-flip-PREREGISTRATION.md).
Same structure as tools/iw1_score.py (branch research/integration-wave1), with its adversarial-review fixes built in
from the start and one addition (the INTACT-CONTENT check, below). Every cell-validity rule is tools/b2d_score.py's
(A1.2 rules 1-3 and 7: sidecars pinned to the registered revision, git_archive source, manifest verified at start and
exit, numpy, no organ error on the probe turn), reused by import. The one relaxation: a CAPABILITY shard (step R)
sets its packet's own flag, so BRAIN_* keys that are exactly the packet's flag (and its lesion flag in the lesion
arms) are allowed there; any other BRAIN_* key still fails the cell.

A step-R / F1 cell counts as reply-level load-bearing ONLY when all of these hold (anything else never passes):
  * the cell is DEFINED (b2d rules) and its S2 verdict (reply fields answer/abstained/recalled_svo, intact_a ==
    intact_b and lesion == lesion.rep0) is `regressed`;
  * DEGENERATE-INTACT guard (wave-1 review): the intact reply is not empty/absent, not an abstention and not an echo
    of the probe message -- else UNDEFINED:degenerate-intact (never a pass);
  * INTACT-CONTENT check (new in wave 2): the intact reply states the capability's content (each packet's
    `intact_must_contain`, lower-cased substring match) and none of `intact_must_not_contain`. A regressed cell whose
    intact reply is non-degenerate but WRONG (e.g. the false-belief answer names the true location) is DEFINED and
    reads `intact-wrong`: not load-bearing, so R reads FAIL (a real negative on that seed, not an undefined cell).
  * STORE-READ guard (adversarial review 2026-09-30, false-belief-chat only): the intact arm's `false_belief_tom`
    trace must show the belief store carrying the answered slot (a positive late-window rate, strictly above every
    other slot); an all-zero or tied store answers slot 0 -- the stale location -- by argmax default, so it reads
    UNDEFINED:degenerate-intact, never a pass.

  R  (reply-level capability, pre-flip, at pin P):
     python tools/iw2_score.py R --packet false-belief-chat --tag iw2-R-fb-<P7> --pin <P>
     python tools/iw2_score.py R --packet transitive-chat   --tag iw2-R-tr-<P7> --pin <P>
     python tools/iw2_score.py R --packet neural-selfid     --tag iw2-R-si-<P7> --pin <P>
  B  (combined no-regression, production default at F, against the reference production-default battery):
     python tools/iw2_score.py B --base-tag b2b0924-base --base-pin a308f1e09babcc9ed096c3c8046d00040391368c
         --flip-tag iw2-flip-<F7> --flip-pin <F> --flipped false-belief-chat,transitive-chat,neural-selfid
  A  (attribute every B1 drop): python tools/iw2_score.py A --b-result <B.json> --tag iw2-alloff-<F7> --pin <F>
  X  (one faculty's cells in any such run): python tools/iw2_score.py X --tag <tag> --pin <F> --faculty <X>

Exit code: 0 PASS, 1 FAIL, 3 UNDEFINED (a cell or precondition that is not DEFINED; never counted as a pass).
Shard layout: research/findings/raw/_load_bearing/_shards/<tag>/s<seed>/<faculty dir>/{lb.json, arm files, sidecars}.
"""
import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b2d_score as B  # noqa: E402

SEEDS = [42, 43, 44, 100, 101, 102]
BASE = "research/findings/raw/_load_bearing/_shards"
FB_MSG = ("Sally puts the marble in the basket. Sally leaves the room. Anne moves the marble to the box. "
          "Where will Sally look for the marble?")
PACKETS = {
    "false-belief-chat": {"row": "tom-false-belief", "shard_dir": "tom-false-belief",
                          "flags": ["BRAIN_FALSE_BELIEF_CHAT"], "lesion_flags": ["BRAIN_FALSE_BELIEF_LESION"],
                          "probe_message": FB_MSG,
                          # the stale (pre-move) location, never the true one
                          "intact_must_contain": ["basket"], "intact_must_not_contain": ["box"],
                          # the intact answer must be a READ of the belief store, not argmax's slot-0 default
                          "store_trace": "false_belief_tom",
                          "control": {"row": "tom-true-belief-ctl", "shard_dir": "tom-true-belief-ctl",
                                      # a witnessed move: the TRUE location, never the stale one
                                      "intact_must_contain": ["box"], "intact_must_not_contain": ["basket"],
                                      "intact_abstains": False}},
    "transitive-chat": {"row": "reasoning-transitive-chat", "shard_dir": "reasoning-transitive-chat",
                        "flags": ["BRAIN_TRANSITIVE_CHAT"], "lesion_flags": ["BRAIN_TRANSITIVE_LESION"],
                        "probe_message": "does cat precede river?",
                        # the derived yes names the queried target and is framed as the brain's own inference
                        # (frame_derived_answer: "I believe the cat precedes the river, but I reasoned that myself ...")
                        "intact_must_contain": ["river", "reasoned"], "intact_must_not_contain": [],
                        "control": {"row": "transitive-reversed-ctl", "shard_dir": "transitive-reversed-ctl",
                                    # the chain taught in reverse: not derivable, the route must abstain
                                    "intact_must_contain": [], "intact_must_not_contain": [],
                                    "intact_abstains": True}},
    "neural-selfid": {"row": "neural-selfid", "shard_dir": "neural-selfid",
                      "flags": ["BRAIN_NEURAL_SELFID"], "lesion_flags": ["BRAIN_SUBSTRATE_PARSE_LESION"],
                      "probe_message": "what do you use?",
                      # the built-in self fact (brain, use, spikes)
                      "intact_must_contain": ["spikes"], "intact_must_not_contain": [],
                      "control": {"row": "selfid-nofact-ctl", "shard_dir": "selfid-nofact-ctl",
                                  # no stored self fact: the moat abstains
                                  "intact_must_contain": [], "intact_must_not_contain": [],
                                  "intact_abstains": True}},
}
CONTROL_ENV = {"LB_IW2_CONTROLS": "1"}   # the control rows' opt-in probe flag (research/runners/lbf_rows/iw2_controls.py)
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


def _intact_turns(canon):
    turn = canon.get("turn")
    return [(k, ((canon.get("arms") or {}).get(k) or {}).get(turn) or {}) for k in ("intact_a", "intact_b")]


def degenerate_intact(canon, probe_message=None):
    """Why the intact reply on the probe turn cannot carry a capability: an empty or absent answer, an abstention, or
    an echo of the probe message. A lesion that 'changes' such a reply is not the faculty's contribution."""
    out = []
    for k, t in _intact_turns(canon):
        ans = t.get("answer")
        if not isinstance(ans, str) or not ans.strip():
            out.append("%s: empty or absent answer" % k)
        elif t.get("abstained") is True:
            out.append("%s: abstained" % k)
        elif probe_message and _norm(ans) == _norm(probe_message):
            out.append("%s: echoes the probe message" % k)
    return out


def silent_store(canon, trace_key):
    """Why the intact answer is not a read of the false-belief store (adversarial review 2026-09-30). The chat wire
    gives locations slots in order of mention, so the Sally-Anne probe's stale location is ALWAYS slot 0, and the
    organ's read `max(rates, key=rates.get)` returns slot 0 when every rate is 0: a store that never ignited, or did
    not hold across the move, 'answers' the correct false-belief text with no store content behind it. The answered
    slot must carry a positive late-window rate strictly above every other slot; a missing or unreadable trace never
    passes either."""
    out = []
    for k, t in _intact_turns(canon):
        tr = t.get(trace_key)
        if not isinstance(tr, dict):
            out.append("%s: no %s trace" % (k, trace_key))
            continue
        rates, idx = tr.get("belief_rates"), tr.get("belief_loc_idx")
        try:
            r = {int(a): float(b) for a, b in (rates or {}).items()}
            mine = r[int(idx)]
        except (TypeError, ValueError, KeyError):
            out.append("%s: %s carries no readable belief_rates at belief_loc_idx %r" % (k, trace_key, idx))
            continue
        others = [v for a, v in r.items() if a != int(idx)]
        if not mine > 0.0 or any(v >= mine for v in others):
            out.append("%s: belief store silent or tied (slot %s rate %r, others %r): the answer is argmax's "
                       "first-slot default, not a store read" % (k, idx, mine, others))
    return out


def wrong_intact(canon, must, must_not):
    """Why a non-degenerate intact reply does not state the capability's content (packet-specific tokens)."""
    out = []
    for k, t in _intact_turns(canon):
        ans = str(t.get("answer") or "").lower()
        miss = [w for w in must if w not in ans]
        bad = [w for w in must_not if w in ans]
        if miss:
            out.append("%s: lacks %s" % (k, miss))
        if bad:
            out.append("%s: contains %s" % (k, bad))
    return out


def _cell_std(d, fac, pin, allowed, spec=None, guard=False):
    """b2d_score.judge_dir with the capability-shard relaxation; returns (state, s2_verdict, detail). With `guard`
    (step R and F1, pass criteria): a `regressed` cell with a degenerate intact reply is UNDEFINED:degenerate-intact;
    one with a non-degenerate but wrong intact reply is DEFINED `intact-wrong` (not load-bearing)."""
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
    det = dict(det, s0=[canon["verdict"], canon["load_bearing"]],
               intact_answer=(_intact_turns(canon)[0][1].get("answer")),
               intact_abstained=(_intact_turns(canon)[0][1].get("abstained")),
               lesion_answer=(((canon.get("arms") or {}).get("lesion") or {}).get(canon.get("turn")) or {}).get("answer"))
    if guard and s2 == "regressed" and spec is not None:
        why = degenerate_intact(canon, spec.get("probe_message"))
        if why:
            return "UNDEFINED:degenerate-intact", None, dict(det, fails=why)
        wrong = wrong_intact(canon, spec.get("intact_must_contain", []), spec.get("intact_must_not_contain", []))
        if wrong:
            return state, "intact-wrong", dict(det, fails=wrong)
        # right text, but is it a store read? (a wrong text is a FAIL above whatever the store did)
        why = silent_store(canon, spec["store_trace"]) if spec.get("store_trace") else []
        if why:
            return "UNDEFINED:degenerate-intact", None, dict(det, fails=why)
    return state, s2, det


def control_cell(d, fac, pin, allowed, ctl):
    """A permuted-control cell (step R): DEFINED, the intact reply meets the control's expectation (abstains, or names
    the control's content), and the packet's lesion leaves the reply unchanged (S2 `pass`). Returns (state, verdict,
    detail); verdict is `control-ok`, `control-violated` (a DEFINED cell that breaks the prediction) or None."""
    st, s2, det = _cell_std(d, fac, pin, allowed)
    if st != "DEFINED" or s2 not in ("regressed", "pass"):
        return (st if st != "DEFINED" else "UNDEFINED:%s" % s2), None, det
    why = []
    ia = det.get("intact_answer")
    if ctl.get("intact_abstains"):
        ab = det.get("intact_abstained")
        if ab is not True:
            why.append("intact did not abstain (%r)" % ia)
    else:
        if det.get("intact_abstained") is True or not isinstance(ia, str) or not ia.strip():
            why.append("intact is empty or abstained (%r)" % ia)
    low = str(ia or "").lower()
    why += ["intact lacks %r" % w for w in ctl.get("intact_must_contain", []) if w not in low]
    why += ["intact contains %r" % w for w in ctl.get("intact_must_not_contain", []) if w in low]
    if s2 != "pass":
        why.append("the lesion changed the control reply (%r -> %r)" % (ia, det.get("lesion_answer")))
    return "DEFINED", ("control-violated" if why else "control-ok"), dict(det, control_fails=why)


def score_R(packet, tag, pin, base=BASE, seeds=SEEDS):
    """Step R for one packet: the capability row must be reply-level load-bearing (with both intact guards) on 6/6
    seeds AND the packet's permuted control must hold on 6/6 seeds. Any not-DEFINED cell -> UNDEFINED (never a pass);
    a DEFINED miss in either -> FAIL."""
    spec = PACKETS[packet]
    allowed = spec["flags"] + spec["lesion_flags"]
    per, ctl_per = {}, {}
    for s in seeds:
        d = os.path.join(base, tag, "s%d" % s, spec["shard_dir"])
        st, v, det = _cell_std(d, spec["row"], pin, allowed, spec=spec, guard=True)
        per[str(s)] = {"state": st, "reply_verdict": v, "detail": det}
        ctl = spec.get("control")
        if ctl:
            dc = os.path.join(base, tag, "s%d" % s, ctl["shard_dir"])
            cst, cv, cdet = control_cell(dc, ctl["row"], pin, allowed, ctl)
            ctl_per[str(s)] = {"state": cst, "verdict": cv, "detail": cdet}
    n_def = sum(1 for c in per.values()
                if c["state"] == "DEFINED" and c["reply_verdict"] in ("regressed", "pass", "intact-wrong"))
    n_lb = sum(1 for c in per.values() if c["state"] == "DEFINED" and c["reply_verdict"] == "regressed")
    c_def = sum(1 for c in ctl_per.values() if c["state"] == "DEFINED" and c["verdict"])
    c_ok = sum(1 for c in ctl_per.values() if c["state"] == "DEFINED" and c["verdict"] == "control-ok")
    need_ctl = bool(spec.get("control"))
    if n_def < len(seeds) or (need_ctl and c_def < len(seeds)):
        verdict, code = "UNDEFINED", UNDEF
    elif n_lb == len(seeds) and (not need_ctl or c_ok == len(seeds)):
        verdict, code = "PASS", PASS
    else:
        verdict, code = "FAIL", FAIL
    return {"step": "R", "packet": packet, "tag": tag, "pin": pin, "per_seed": per, "n_defined": n_def,
            "n_reply_load_bearing": n_lb, "control": {"row": (spec.get("control") or {}).get("row"),
                                                      "per_seed": ctl_per, "n_defined": c_def, "n_ok": c_ok},
            "verdict": verdict}, code


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
                # the reference battery's own cell was not DEFINED (B2b: 210/216, the wave-1 ruling): nothing to
                # compare against -- reported by name, never scored either way
                out.setdefault("reference_undefined_faculties", []).append("%s %s: %s" % (level, fac, nb))
                continue
            if not isinstance(nf, int):
                undefined.append("%s %s: flip battery has undefined cells (%s)" % (level, fac, nf))
                continue
            if nf < nb:
                regressions.append({"level": level, "faculty": fac, "reference": nb, "flip": nf,
                                    "msg": "%s %s: %d -> %d load-bearing seeds" % (level, fac, nb, nf)})
        out["%s_base" % level], out["%s_flip" % level] = bc, fc
    rows_F = {}
    for p in flipped:
        row = PACKETS[p]["row"]
        n = _counts(f, "s2").get(row)
        if isinstance(n, int):
            # the same guarded per-seed read as step R (degenerate or wrong intact never counts toward F1)
            fr, _fcode = score_R(p, flip_tag, flip_pin, base=base, seeds=seeds)
            out.setdefault("f1_cells", {})[p] = fr["per_seed"]
            if fr["n_defined"] < len(seeds):
                n = "UNDEFINED (%d/%d seeds DEFINED after the intact guards)" % (fr["n_defined"], len(seeds))
            else:
                n = fr["n_reply_load_bearing"]
        rows_F[p] = n
        if n is None:
            undefined.append("F %s: its row is absent from the flip battery (REQUIRED_ENV not lifted?)" % p)
        elif not isinstance(n, int):
            undefined.append("F %s: undefined cells %s" % (p, n))
        elif n != len(seeds):
            regressions.append({"level": "F1", "faculty": row, "reference": len(seeds), "flip": n,
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
    F1 regressions are never excused. A can never turn a B that also carried undefined criteria into a pass."""
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
    at.add_argument("--tag", required=True, help="iw2-alloff-<F7> or iw2-bisect-<packet>-<F7>")
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
        bad = [x for x in flipped if x not in PACKETS]
        if bad:
            ap.error("unknown packet(s) in --flipped: %s" % bad)
        res, code = score_B(a.base_tag, a.base_pin, a.flip_tag, a.flip_pin, flipped, base=a.base)
    # provenance + device for the committed verdict file (gates artifact_provenance / device_and_cost): this scorer
    # builds no brain; every cell it counts had to carry SIM_BACKEND=numpy in its sidecars (b2d_score rule 3)
    res.update(runner="tools/iw2_score.py", argv=list(sys.argv if argv is None else argv),
               sim_backend="numpy (the scored shards'; this scorer builds no brain)")
    print(json.dumps({k: res[k] for k in res if k not in ("s0_base", "s0_flip", "s2_base", "s2_flip")},
                     indent=1, default=str))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1, default=str)
    return code


if __name__ == "__main__":
    sys.exit(main())
