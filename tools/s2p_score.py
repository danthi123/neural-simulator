#!/usr/bin/env python3
"""s2p_score.py -- the registered scorer for the session-2 host-decision packets (read-only; builds no brain).

Packets (each has its own prereg and HANDOFF; this file holds the registered copy of every bar):
  fb-spiking-witness       research/findings/2026-10-01-fb-spiking-witness-PREREGISTRATION.md
  transitive-spiking-reach research/findings/2026-10-01-transitive-spiking-reach-PREREGISTRATION.md
  word-role-parse          research/findings/2026-10-01-word-role-parse-PREREGISTRATION.md
  word-role-commit         research/findings/2026-10-06-word-role-commit-PREREGISTRATION.md

Steps:
  select  dev-seed operating point: the FIRST index of the organ's GRID (grid order) whose battery `all_pass` is
          true on EVERY dev seed 1-8. Reads <sweep-dir>/op<I>_s<S>.json. No such index: verdict NONE (exit 1).
          A missing (op, seed) file before the chosen index makes the selection UNDEFINED (exit 3): never skip.
     python tools/s2p_score.py select --packet fb-spiking-witness --sweep-dir <dir> --out <dir>/selection.json
  O       organ gate on the six gate seeds at the selected index: <dir>/s<S>.json, each DEFINED (its .prov.json sidecar
          passes tools/b2d_score.py's validity rules at the pin: git_archive source, manifest verified, numpy; and
          op_index == the selection, seed matches, prereg == the packet's prereg) and `all_pass`. PASS = 6/6 all_pass;
          FAIL = every cell DEFINED and at least one fails; else UNDEFINED.
     python tools/s2p_score.py O --packet fb-spiking-witness --dir <dir> --selection <sel.json> --pin <sha> --out <f>
  R       reply-level gate through the real /api/brain-chat handler (tools/lb_shard.py shards), with every rule of
          tools/iw2_score.py step R reused by import (DEFINED cell, S2 `regressed`, degenerate-intact guard,
          intact-content check, store-read guard where named, lesion-content check `lesion-kept`, permuted control
          6/6). Added here: every sidecar carries the packet's boolean flags = "1" AND its operating-point variable =
          the selected index (a cell run at another operating point is UNDEFINED).
     python tools/s2p_score.py R --packet fb-spiking-witness --tag s2p-R-fb-<P7> --pin <P> --selection <sel.json>
         --out <f>

Exit code: 0 PASS (select: an index was chosen), 1 FAIL (select: NONE), 3 UNDEFINED.
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
import iw2_score as I  # noqa: E402

DEV_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8]
GATE_SEEDS = [42, 43, 44, 100, 101, 102]
PASS, FAIL, UNDEF = 0, 1, 3

FB_MSG = I.FB_MSG
PACKETS = {
    "fb-spiking-witness": {
        "prereg": "research/findings/2026-10-01-fb-spiking-witness-PREREGISTRATION.md",
        "organ": "tom_spiking_witness", "grid_len": 8,
        "row": "fb-spiking-witness", "shard_dir": "fb-spiking-witness",
        "flags": ["BRAIN_FALSE_BELIEF_CHAT", "BRAIN_FALSE_BELIEF_SPIKING_WITNESS"],
        "op_env": "BRAIN_FALSE_BELIEF_WITNESS_OP", "lesion_flags": ["BRAIN_FALSE_BELIEF_WITNESS_LESION"],
        "probe_message": FB_MSG, "intact_must_contain": ["basket"], "intact_must_not_contain": ["box"],
        "store_trace": "false_belief_tom",
        "control": {"row": "fb-spiking-true-belief-ctl", "shard_dir": "fb-spiking-true-belief-ctl",
                    "intact_must_contain": ["box"], "intact_must_not_contain": ["basket"], "intact_abstains": False}},
    "transitive-spiking-reach": {
        "prereg": "research/findings/2026-10-01-transitive-spiking-reach-PREREGISTRATION.md",
        "organ": "transitive_spiking_reach", "grid_len": 7,
        "row": "transitive-spiking-reach", "shard_dir": "transitive-spiking-reach",
        "flags": ["BRAIN_TRANSITIVE_CHAT", "BRAIN_TRANSITIVE_SPIKING_REACH"],
        "op_env": "BRAIN_TRANSITIVE_REACH_OP", "lesion_flags": ["BRAIN_TRANSITIVE_REACH_LESION"],
        "probe_message": "does cat precede river?", "intact_must_contain": ["river", "reasoned"],
        "intact_must_not_contain": [],
        "control": {"row": "transitive-spiking-reversed-ctl", "shard_dir": "transitive-spiking-reversed-ctl",
                    "intact_must_contain": [], "intact_must_not_contain": [], "intact_abstains": True},
        # ENGAGEMENT (RULING iw2-merged (b), session 7): the route catches every exception and abstains, so a cell
        # counts only if the reply's own `transitive_trace` shows the route ran, the lesion was on in the lesion arms
        # (off in the intact arms) and no error was swallowed.
        "engagement": {"trace_key": "transitive_trace", "route": "spiking_reach"}},
    "word-role-parse": {
        "prereg": "research/findings/2026-10-01-word-role-parse-PREREGISTRATION.md",
        "organ": "word_role_parser", "grid_len": 8,
        "row": "word-role-selfid", "shard_dir": "word-role-selfid",
        "flags": ["BRAIN_NEURAL_SELFID", "BRAIN_WORD_ROLE_PARSE"],
        "op_env": "BRAIN_WORD_ROLE_OP", "lesion_flags": ["BRAIN_WORD_ROLE_LESION"],
        "probe_message": "what do you usually use?", "intact_must_contain": ["spikes"], "intact_must_not_contain": [],
        "control": {"row": "word-role-nofact-ctl", "shard_dir": "word-role-nofact-ctl",
                    "intact_must_contain": [], "intact_must_not_contain": [], "intact_abstains": True},
        # ENGAGEMENT (RULING word-role-engagement, Claude 2026-10-06): `_neural_question_parse` catches every organ
        # exception and declines, so a cell counts only if the reply's own `word_role_trace` shows the route ran,
        # the lesion was on in the lesion arms (off in the intact arms) and no error was swallowed. R data pinned
        # before the trace existed (s2p-R-wr-9ea7411) re-scores UNDEFINED:engagement-no-record; its confirmed PASS
        # stands with the inferred-engagement caveat in research/findings/2026-10-06-word-role-parse-gate.md.
        "engagement": {"trace_key": "word_role_trace", "route": "word_role"}},
    "word-role-commit": {
        # Packet word-role-commit (docs/drafts/spiking-commit/PLAN.md, application A1): the word-role parse decided by
        # the organ's spiking commit stage (COMMIT_GRID, 6 rows) instead of the host ROLE_MIN / margin / argmax read.
        # The lesion cuts only role -> commit gate, so engagement reads `commit_lesioned`, not the word -> role flag.
        "prereg": "research/findings/2026-10-06-word-role-commit-PREREGISTRATION.md",
        "organ": "word_role_commit", "grid_len": 6,
        "row": "word-role-commit-selfid", "shard_dir": "word-role-commit-selfid",
        "flags": ["BRAIN_NEURAL_SELFID", "BRAIN_WORD_ROLE_PARSE", "BRAIN_WORD_ROLE_COMMIT"],
        "op_env": "BRAIN_WORD_ROLE_COMMIT_OP", "lesion_flags": ["BRAIN_WORD_ROLE_COMMIT_LESION"],
        "probe_message": "what do you usually use?", "intact_must_contain": ["spikes"], "intact_must_not_contain": [],
        "control": {"row": "word-role-commit-nofact-ctl", "shard_dir": "word-role-commit-nofact-ctl",
                    "intact_must_contain": [], "intact_must_not_contain": [], "intact_abstains": True},
        "engagement": {"trace_key": "word_role_trace", "route": "word_role_commit", "lesion_key": "commit_lesioned"}},
}


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def score_select(packet, sweep_dir, seeds=DEV_SEEDS):
    """The first GRID index (grid order) whose battery passes on every dev seed. Indices are read from the files
    present; the grid length is the packet's registered one when known, else the largest index found + 1."""
    spec = PACKETS[packet]
    found = sorted({int(m.group(1)) for p in glob.glob(os.path.join(sweep_dir, "op*_s*.json"))
                    for m in [re.search(r"op(\d+)_s\d+\.json$", p)] if m})
    n = spec["grid_len"] if spec["grid_len"] else ((max(found) + 1) if found else 0)
    table, chosen, missing = {}, None, []
    for op in range(n):
        row = {}
        for s in seeds:
            d = _load(os.path.join(sweep_dir, "op%d_s%d.json" % (op, s)))
            if d is None or d.get("seed") != s or d.get("op_index") != op:
                row[str(s)] = None
                missing.append("op%d_s%d" % (op, s))
            else:
                row[str(s)] = bool(d.get("all_pass"))
        table[str(op)] = row
        if chosen is None and all(v is True for v in row.values()):
            chosen = op
            break
        if any(v is None for v in row.values()):
            break   # an undefined row before any chosen index: never skip past it
    if chosen is not None:
        verdict, code = "SELECTED", PASS
    elif missing or n == 0:
        verdict, code = "UNDEFINED", UNDEF
    else:
        verdict, code = "NONE", FAIL
    return {"step": "select", "packet": packet, "sweep_dir": sweep_dir, "dev_seeds": seeds, "grid_len": n,
            "table": table, "missing": missing, "op": chosen, "verdict": verdict}, code


def _selected_op(selection_path, packet):
    sel = _load(selection_path) or {}
    if sel.get("packet") != packet or sel.get("verdict") != "SELECTED" or not isinstance(sel.get("op"), int):
        return None
    return sel["op"]


def score_O(packet, d, selection, pin, seeds=GATE_SEEDS):
    spec = PACKETS[packet]
    op = _selected_op(selection, packet)
    per = {}
    for s in seeds:
        f = os.path.join(d, "s%d.json" % s)
        res = _load(f)
        fails = []
        if op is None:
            fails.append("no SELECTED operating point in %s" % selection)
        if res is None:
            fails.append("missing %s" % f)
        else:
            if res.get("seed") != s:
                fails.append("seed=%r" % res.get("seed"))
            if res.get("op_index") != op:
                fails.append("op_index=%r (selected %r)" % (res.get("op_index"), op))
            if res.get("prereg") != spec["prereg"]:
                fails.append("prereg=%r" % res.get("prereg"))
            pj = _load(f + ".prov.json")
            if pj is None:
                fails.append("no provenance sidecar")
            else:
                fails += B.sidecar_fails(pj, pin, False)
        per[str(s)] = {"state": "DEFINED" if not fails else "UNDEFINED", "fails": fails,
                       "all_pass": (bool(res.get("all_pass")) if res else None),
                       "failed_cases": sorted(k for k, c in ((res or {}).get("cases") or {}).items()
                                              if not c.get("pass"))}
    n_def = sum(1 for c in per.values() if c["state"] == "DEFINED")
    n_ok = sum(1 for c in per.values() if c["state"] == "DEFINED" and c["all_pass"])
    if n_def < len(seeds):
        verdict = "UNDEFINED"
    elif n_ok == len(seeds):
        verdict = "PASS"
    else:
        verdict = "FAIL"
    res = {"step": "O", "packet": packet, "dir": d, "pin": pin, "op": op, "per_seed": per, "n_defined": n_def,
           "n_all_pass": n_ok, "verdict": verdict}
    code = I._earn(res, "s2p step O %s at %s" % (packet, pin),
                   [("organ cells DEFINED on %d/%d seeds" % (len(seeds), len(seeds)), n_def, len(seeds))])
    return res, code


_IW2_ALLOWING = I._allowing   # captured before score_R swaps the module attribute


def _allowing_op(allowed, require_on, op_env, op):
    """iw2_score._allowing plus: the operating-point variable must be present and equal the selected index."""
    inner = _IW2_ALLOWING(set(allowed) | {op_env}, tuple(require_on))

    def f(pj, pin, is_lb):
        out = inner(pj, pin, is_lb)
        v = (pj.get("env") or {}).get(op_env)
        if str(v) != str(op):
            out.append("%s=%r (selected %r)" % (op_env, v, op))
        return out
    return f


def score_R(packet, tag, pin, selection, base=I.BASE, seeds=GATE_SEEDS, at_default=False):
    """`at_default` (task T21's F1 read at the flip revision): the packet's flags and operating point are the
    production default there, so the sidecars must carry NO packet flag and NO operating-point variable (only the
    lesion flag in the lesion arms); every other rule is unchanged."""
    spec = dict(PACKETS[packet])
    op = _selected_op(selection, packet)
    if at_default:
        return _score_R_cells(packet, spec, tag, pin, op, base, seeds, spec["lesion_flags"], [], None)
    if op is None:
        res = {"step": "R", "packet": packet, "tag": tag, "pin": pin, "verdict": "UNDEFINED",
               "undefined_reasons": ["no SELECTED operating point in %s" % selection]}
        return res, UNDEF
    allowed = spec["flags"] + spec["lesion_flags"] + [spec["op_env"]]
    return _score_R_cells(packet, spec, tag, pin, op, base, seeds, allowed, spec["flags"], op)


def engagement_state(d, eng):
    """None when every arm file in shard dir `d` records engagement per `eng` ({"trace_key", "route"[, "lesion_key"]}):
    the last turn body carrying `trace_key` reads that route with no error, and its lesion record (`lesion_key`,
    default `lesioned`) is True in lesion arms and False in intact arms. Otherwise a short reason (the cell then reads
    UNDEFINED:engagement-<reason>)."""
    arms = sorted(p for p in glob.glob(os.path.join(d, "*.json*"))
                  if not p.endswith(".prov.json") and os.path.basename(p) != "lb.json")
    if not arms:
        return "no-arm-files"
    for path in arms:
        name = os.path.basename(path)
        is_lesion = name.startswith("lesion_")
        if not (is_lesion or name.startswith("intact_")):
            continue
        doc = _load(path) or {}
        bodies = [b for b in (doc.values() if isinstance(doc, dict) else []) if isinstance(b, dict) and eng["trace_key"] in b]
        if not bodies:
            return "no-record:%s" % name
        t = bodies[-1][eng["trace_key"]] or {}
        if t.get("error"):
            return "error:%s" % name
        if t.get("route") != eng["route"]:
            return "route-%s:%s" % (t.get("route"), name)
        if bool(t.get(eng.get("lesion_key", "lesioned"))) != is_lesion:
            return "lesion-flag:%s" % name
    return None


def _score_R_cells(packet, spec, tag, pin, op, base, seeds, allowed, require_on, op_required):
    saved = I._allowing
    # every cell read below goes through _cell_std/control_cell, which install I._allowing(...): swap in the
    # operating-point-aware version for the duration of this step only (at default: the plain iw2 rule)
    if op_required is not None:
        I._allowing = lambda a, r=(): _allowing_op(a, r, spec["op_env"], op_required)
    try:
        per, ctl_per = {}, {}
        for s in seeds:
            dd = os.path.join(base, tag, "s%d" % s, spec["shard_dir"])
            st, v, det = I._cell_std(dd, spec["row"], pin, allowed, spec=spec, guard=True, require_on=require_on)
            if st == "DEFINED" and spec.get("engagement"):
                why = engagement_state(dd, spec["engagement"])
                if why:
                    st, v = "UNDEFINED:engagement-%s" % why, None
            per[str(s)] = {"state": st, "reply_verdict": v, "detail": det}
            ctl = spec["control"]
            dc = os.path.join(base, tag, "s%d" % s, ctl["shard_dir"])
            cst, cv, cdet = I.control_cell(dc, ctl["row"], pin, allowed, ctl, require_on=require_on)
            ctl_per[str(s)] = {"state": cst, "verdict": cv, "detail": cdet}
    finally:
        I._allowing = saved
    n_def = sum(1 for c in per.values()
                if c["state"] == "DEFINED" and c["reply_verdict"] in ("regressed", "pass", "intact-wrong", "lesion-kept"))
    n_lb = sum(1 for c in per.values() if c["state"] == "DEFINED" and c["reply_verdict"] == "regressed")
    c_def = sum(1 for c in ctl_per.values() if c["state"] == "DEFINED" and c["verdict"])
    c_ok = sum(1 for c in ctl_per.values() if c["state"] == "DEFINED" and c["verdict"] == "control-ok")
    if n_def < len(seeds) or c_def < len(seeds):
        verdict = "UNDEFINED"
    elif n_lb == len(seeds) and c_ok == len(seeds):
        verdict = "PASS"
    else:
        verdict = "FAIL"
    res = {"step": "R", "packet": packet, "tag": tag, "pin": pin, "op": op, "at_default": op_required is None,
           "per_seed": per, "n_defined": n_def,
           "n_reply_load_bearing": n_lb, "control": {"row": spec["control"]["row"], "per_seed": ctl_per,
                                                     "n_defined": c_def, "n_ok": c_ok}, "verdict": verdict}
    checks = [("capability cells DEFINED on %d/%d seeds" % (len(seeds), len(seeds)), n_def, len(seeds)),
              ("permuted-control cells DEFINED on %d/%d seeds" % (len(seeds), len(seeds)), c_def, len(seeds))]
    code = I._earn(res, "s2p step R %s at %s" % (packet, pin), checks)
    return res, code


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="step", required=True)
    a1 = sub.add_parser("select")
    a1.add_argument("--packet", required=True, choices=sorted(PACKETS))
    a1.add_argument("--sweep-dir", required=True)
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("O")
    a2.add_argument("--packet", required=True, choices=sorted(PACKETS))
    a2.add_argument("--dir", required=True)
    a2.add_argument("--selection", required=True)
    a2.add_argument("--pin", required=True)
    a2.add_argument("--out", required=True)
    a3 = sub.add_parser("R")
    a3.add_argument("--packet", required=True, choices=sorted(PACKETS))
    a3.add_argument("--tag", required=True)
    a3.add_argument("--pin", required=True)
    a3.add_argument("--selection", required=True)
    a3.add_argument("--out", required=True)
    a3.add_argument("--at-default", action="store_true", help="T21 F1: the packet's path is the default here")
    a4 = sub.add_parser("op", help="print the selected operating point (exit 3 if none)")
    a4.add_argument("--packet", required=True, choices=sorted(PACKETS))
    a4.add_argument("--selection", required=True)
    a = ap.parse_args(argv)
    if a.step == "op":
        op = _selected_op(a.selection, a.packet)
        if op is None:
            return UNDEF
        print(op)
        return PASS
    if a.step == "select":
        res, code = score_select(a.packet, a.sweep_dir)
    elif a.step == "O":
        res, code = score_O(a.packet, a.dir, a.selection, a.pin)
    else:
        res, code = score_R(a.packet, a.tag, a.pin, a.selection, at_default=a.at_default)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("s2p %s %s: %s%s" % (a.step, a.packet, res["verdict"],
                               (" op=%s" % res.get("op")) if a.step != "R" or res.get("op") is not None else ""))
    return code


if __name__ == "__main__":
    sys.exit(main())
