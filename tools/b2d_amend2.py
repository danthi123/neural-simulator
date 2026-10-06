#!/usr/bin/env python3
"""tools/b2d_amend2.py -- B2d Amendment 2 (the mood-engaged turn the Qwen mouth speaks): job files, the dev-seed
candidate selection, and the gate score. Governing text: research/findings/2026-09-29-production-default-battery-
B2d-PREREGISTRATION.md, "Amendment 2". Probe: research/runners/b2d_amend2_probe.py. Never queues anything.

  python tools/b2d_amend2.py jobs --pin <P2> --phase dev|dev-lesion|gate-route|gate-rest --out <file>
  python tools/b2d_amend2.py dev  --pin <P2> --base <dir> [--final] --out <selection.json>
  python tools/b2d_amend2.py routecheck --pin <P2> --base <dir> --selection <selection.json> --out <route.json>
  python tools/b2d_amend2.py gate --pin <P2> --base <dir> --selection <selection.json> [--attempt1 <dir>] --out <score.json>

<dir> holds s<seed>/<arm>.json (research/findings/raw/_b2d_amend2/<P2_7>/{dev,gate} on the server after sync).

The reply-level metric is NOT a marker word (owner 2026-09-25: feelings shape speech; the marker word is retired). It
is the MOUTH REPLY: the ordered (SVO, surface) pairs the Qwen faculty generated on the probe turn AND that were spoken
(the surface is in the final answer), read from the reply's `mouth_trace`. A cell is mouth-load-bearing when the
intact and lesion mouth replies differ, compared exactly (no normalization, as S2). A change confined to host text
around the Qwen spans (a lead, a suffix, the spiking recall mouth's tone mark) is reported as HOST-ONLY, never as
mouth-load-bearing. How the mouth reply changed is reported with no weight: `manner` (the same fact worded
differently, with a different manner clause in the Qwen prompt), `plan` (different facts spoken: forthcomingness),
`route-changed` (the lesion arm's Qwen spoke nothing), `unexplained` (same fact, same manner clause, different
words: should be impossible under greedy decoding with clean floors; a NEEDS-CLAUDE).
"""
import argparse
import json
import os
import re
import shlex
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from research.runners.b2d_amend2_probe import (ARMS, CANDIDATES, DEV_SEEDS, GATE_SEEDS, LESIONS,  # noqa: E402
                                                mouth_attribution, q_label)

PREREG = "research/findings/2026-09-29-production-default-battery-B2d-PREREGISTRATION.md"
QWEN_NAME = "off-bridge Qwen-0.5B (spiking forward)"
INTACT = ("route", "intact_a", "intact_b")
DEV_ARMS = INTACT
DEV_LESION_SEEDS = (1, 2)
DEV_LESION_ARMS = tuple("les_%s" % s for s, _ in LESIONS.values())
DEV_MIN_SEEDS = 6          # a candidate is eligible when at least 6 of the 8 dev seeds pass every check
ASSERT = ".venv/bin/python tools/assert_flipped_defaults.py && "
MEM_OK = ("for _w in $(seq 90); do bash tools/mem_ok.sh 10 4 >/dev/null 2>&1 && break; sleep 60; done && "
          "bash tools/mem_ok.sh 10 4")
MEMCAP = "bash tools/memcap.sh 16 --"


def _check_pin(pin):
    if not re.fullmatch(r"[0-9a-f]{40}", pin or ""):
        raise SystemExit("pin must be a full 40-hex commit sha, got %r" % pin)


def out_dir(pin, phase):
    return "research/findings/raw/_b2d_amend2/%s/%s" % (pin[:7], "dev" if phase.startswith("dev") else "gate")


def job_line(pin, phase, seed, arm):
    d = "%s/s%d" % (out_dir(pin, phase), seed)
    return ("cd ~/derisk-pool/revisions/%s && %smkdir -p %s && %s && %s env SIM_BACKEND=cupy OMP_NUM_THREADS=1 "
            "OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LB_ARM_RENDERER=production PREREG=%s .venv/bin/python -u -m "
            "research.runners.b2d_amend2_probe --arm %s --seed %d --pin %s --out %s"
            % (pin, ASSERT, shlex.quote(d), MEM_OK, MEMCAP, PREREG, arm, seed, pin, shlex.quote("%s/%s.json" % (d, arm))))


def job_lines(pin, phase):
    _check_pin(pin)
    if phase == "dev":
        pairs = [(s, a) for s in DEV_SEEDS for a in DEV_ARMS]
    elif phase == "dev-lesion":
        pairs = [(s, a) for s in DEV_LESION_SEEDS for a in DEV_LESION_ARMS]
    elif phase == "gate-route":
        pairs = [(s, "route") for s in GATE_SEEDS]
    elif phase == "gate-rest":
        pairs = [(s, a) for s in GATE_SEEDS for a in ARMS if a != "route"]
    else:
        raise SystemExit("unknown phase %r" % phase)
    return [job_line(pin, phase, s, a) for s, a in pairs]


# ── reading one arm file ───────────────────────────────────────────────────────────────────────────────────────────
def load(base, seed, arm):
    p = os.path.join(base, "s%d" % seed, "%s.json" % arm)
    try:
        with open(p) as fh:
            return json.load(fh)
    except Exception:
        return None


def validity(doc, pin, seed, arm):
    """A1.2 rules E1-E3 in this probe's shape, plus the valence map. [] when valid."""
    if doc is None:
        return ["missing"]
    rev = doc.get("revision") or {}
    bad = []
    if doc.get("pin") != pin or rev.get("git_sha") != pin:
        bad.append("pin")
    if rev.get("source_kind") != "git_archive":
        bad.append("source_kind")
    if rev.get("manifest_verified") is not True:
        bad.append("manifest")
    if doc.get("backend") != "cupy":
        bad.append("backend")
    if doc.get("renderer_mode") != "production":
        bad.append("renderer_mode")
    if int((doc.get("valence_map") or {}).get("entries") or 0) <= 0:
        bad.append("valence-map-absent")
    if doc.get("arm_env") != ARMS.get(arm) or doc.get("arm") != arm:
        bad.append("arm_env")
    if doc.get("seed") != seed:
        bad.append("seed")
    return bad


def turn(doc, label):
    r = ((doc or {}).get("responses") or {}).get(label)
    return r if isinstance(r, dict) else None


def reply_key(r):
    return json.dumps([r.get("answer"), r.get("abstained"), r.get("recalled_svo")], sort_keys=True)


def read_turn(doc, label):
    """Everything the rules need from one arm's probe turn. `state` is None when the turn is readable."""
    r = turn(doc, label)
    if r is None or r.get("_error") or not isinstance(r.get("answer"), str):
        return {"state": "turn-error"}
    if any((r.get(k) or {}).get("error") is not None for k in ("affect", "affect_drives")):   # rule 7: organ error
        return {"state": "turn-error", "organ_error": True}
    ar = r.get("_arm_renderer") or {}
    if r.get("renderer") != QWEN_NAME or ar.get("mode") != "production":
        return {"state": "M", "renderer": r.get("renderer")}
    mt = r.get("mouth_trace")
    if not isinstance(mt, dict) or mt.get("error") or mt.get("wrapped") is not True:
        return {"state": "trace-absent"}
    calls = int(ar.get("qwen_generate_calls") or 0)
    gen = [c for c in (mt.get("calls") or []) if c.get("generative")]
    prompts = ((doc.get("qwen_prompts") or {}).get(label)) or []
    match = (calls == len(gen) == len(prompts) and int(doc.get("qwen_prompts_unassigned") or 0) == 0
             and all(isinstance(p.get("prompts"), list) for p in prompts))
    visible = match and all((c.get("manner") or "") in " ".join(str(x) for x in prompts[i]["prompts"])
                            for i, c in enumerate(gen))
    spoken = [c for c in gen if c.get("in_answer")]
    return {"state": None, "reply": reply_key(r), "calls": calls, "route_match": match, "conditioning_visible": visible,
            "mouth": [[list(c.get("svo") or []), c.get("surface")] for c in spoken],
            "manner": {json.dumps(c.get("svo")): c.get("manner") or "" for c in spoken},
            "tone_level": (r.get("affect") or {}).get("tone_level"),
            "drives_level": (r.get("affect_drives") or {}).get("level")}


def engaged(row, t):
    if row == "affect-drives-response":
        return t.get("drives_level") not in (None, 0)
    return t.get("tone_level") not in (None, 0)


def via(a, l):
    """How the mouth reply changed (reported, no weight)."""
    out = []
    if not l["mouth"]:
        return ["route-changed"]
    sa, sl = [json.dumps(x[0]) for x in a["mouth"]], [json.dumps(x[0]) for x in l["mouth"]]
    if sa != sl:
        out.append("plan")
    surf_a, surf_l = {json.dumps(x[0]): x[1] for x in a["mouth"]}, {json.dumps(x[0]): x[1] for x in l["mouth"]}
    for k in surf_a:
        if k in surf_l and surf_a[k] != surf_l[k]:
            out.append("manner" if a["manner"].get(k) != l["manner"].get(k) else "unexplained")
    return sorted(set(out))


def _arm_state(base, pin, seed, arm, label, attempt1):
    doc = load(base, seed, arm)
    bad = validity(doc, pin, seed, arm)
    t = read_turn(doc, label) if not bad else {"state": "E"}
    if bad:
        t["fails"] = bad
    if t["state"] == "turn-error":
        t["state"], t["fails"] = "E", ["organ-error" if t.get("organ_error") else "turn-error"]
    if t["state"] in ("E", "M"):
        t["rerun_done"] = bool(attempt1) and os.path.exists(os.path.join(attempt1, "s%d" % seed, "%s.json" % arm))
    return t


def score_cell(base, pin, seed, row, label, attempt1=None):
    short = LESIONS[row][0]
    names = list(INTACT) + ["les_%s" % short, "les_%s_rep0" % short]
    st = {a: _arm_state(base, pin, seed, a, label, attempt1) for a in names}
    em = {a: t for a, t in st.items() if t["state"] in ("E", "M")}
    if em:
        final = any(t.get("rerun_done") for t in em.values())
        return {"state": "FAIL" if final else "RERUN", "arms": {a: [t["state"]] + t.get("fails", []) for a, t in em.items()}}
    if any(t["state"] == "trace-absent" or not t["route_match"] or not t["conditioning_visible"]
           for t in st.values()):
        return {"state": "ROUTE-MISMATCH", "arms": {a: {k: t.get(k) for k in ("state", "calls", "route_match",
                                                                                 "conditioning_visible")}
                                                    for a, t in st.items()}}
    if any(st[a]["calls"] == 0 or not st[a]["mouth"] for a in INTACT):
        return {"state": "MOUTH-SILENT", "calls": {a: st[a]["calls"] for a in INTACT}}
    a, b, rt = st["intact_a"], st["intact_b"], st["route"]
    les, rep = st["les_%s" % short], st["les_%s_rep0" % short]
    if not engaged(row, a):
        return {"state": "NOT-ENGAGED", "tone_level": a["tone_level"], "drives_level": a["drives_level"]}
    if not (a["reply"] == b["reply"] == rt["reply"] and a["mouth"] == b["mouth"] == rt["mouth"]):
        return {"state": "UNDEFINED:null-noise"}
    if not (les["reply"] == rep["reply"] and les["mouth"] == rep["mouth"]):
        return {"state": "UNDEFINED:lesion-noise"}
    mlb = a["mouth"] != les["mouth"]
    changed = a["reply"] != les["reply"]
    return {"state": "DEFINED", "mouth_load_bearing": mlb, "reply_changed": changed,
            "host_only": changed and not mlb, "via": via(a, les) if mlb else [],
            "attribution": mouth_attribution("%s lesion on the s%d mouth reply" % (row, seed), a["mouth"],
                                             b["mouth"], les["mouth"]),
            "intact_mouth": a["mouth"], "lesion_mouth": les["mouth"]}


def row_reading(cells):
    states = [c["state"] for c in cells.values()]
    if any(s == "RERUN" for s in states):
        return "PENDING"
    d = [c for c in cells.values() if c["state"] == "DEFINED"]
    k = sum(1 for c in d if c["mouth_load_bearing"])
    n = len(cells)
    if len(d) == n:
        return "MOUTH-LOAD-BEARING" if k == n else ("MOUTH-NOT-LOAD-BEARING" if k == 0 else "MOUTH-PARTIAL")
    if not d:
        if all(s == "MOUTH-SILENT" for s in states):
            return "MOUTH-SILENT"
        if all(s == "NOT-ENGAGED" for s in states):
            return "NOT-ENGAGED"
        return "UNDEFINED"
    return "MIXED"


def score_gate(base, pin, selected, attempt1=None, seeds=GATE_SEEDS):
    out = {"pin": pin, "selected": selected, "candidates": {}}
    for c in CANDIDATES:
        label = q_label(c["id"])
        rows = {}
        for row in LESIONS:
            cells = {str(s): score_cell(base, pin, s, row, label, attempt1) for s in seeds}
            d = [x for x in cells.values() if x["state"] == "DEFINED"]
            k = sum(1 for x in d if x["mouth_load_bearing"])
            rows[row] = {"reading": row_reading(cells), "defined": len(d), "mouth_load_bearing": k,
                         "interval": [k, k + len(cells) - len(d)], "n": len(cells),
                         "host_only": sum(1 for x in d if x["host_only"]),
                         "via_unexplained": sorted(s for s, x in cells.items() if "unexplained" in x.get("via", [])),
                         "cells": cells}
        out["candidates"][c["id"]] = {"question": c["question"], "rich": c["rich"], "rows": rows,
                                      "primary": c["id"] == selected}
    prim = out["candidates"].get(selected, {}).get("rows", {})
    out["rerun_arms"] = sorted({"s%s/%s" % (s, a) for r in prim.values() for s, c in r["cells"].items()
                                if c["state"] == "RERUN" for a in c["arms"]})
    out["needs_claude"] = sorted({"%s s%s %s" % (row, s, c["state"]) for row, r in prim.items()
                                  for s, c in r["cells"].items() if c["state"] in ("ROUTE-MISMATCH", "FAIL")}
                                 | {"%s s%s via unexplained" % (row, s) for row, r in prim.items()
                                    for s in r["via_unexplained"]})
    out["verdict"] = "INCOMPLETE" if out["rerun_arms"] else "COMPLETE"
    return out


def score_dev(base, pin, final=False, seeds=DEV_SEEDS):
    """The outcome-blind selection. Reads only the three intact arms (route, intact_a, intact_b) and the lesion smoke's
    validity and route; it never compares an intact reply with a lesion reply."""
    out = {"pin": pin, "rule": "first candidate (declared order) with >= %d of %d dev seeds passing: valid arms, Qwen "
                               "spoke on the probe turn in all three intact arms, counter == trace == captured "
                               "prompts, manner clause visible in the prompt, tone level != 0 in intact_a, and "
                               "replies and mouth replies equal across the three" % (DEV_MIN_SEEDS, len(seeds)),
           "candidates": {}, "missing_or_invalid": []}
    for c in CANDIDATES:
        label = q_label(c["id"])
        per = {}
        for s in seeds:
            st = {a: _arm_state(base, pin, s, a, label, None) for a in DEV_ARMS}
            bad = {a: t["state"] for a, t in st.items() if t["state"] in ("E", "M", "trace-absent")}
            if bad:
                per[str(s)] = {"ok": False, "why": bad}
                if any(v == "E" for v in bad.values()):
                    out["missing_or_invalid"] += ["s%d/%s" % (s, a) for a, v in bad.items() if v == "E"]
                continue
            why = []
            if not all(t["route_match"] and t["conditioning_visible"] for t in st.values()):
                why.append("route-mismatch")
            if any(t["calls"] == 0 or not t["mouth"] for t in st.values()):
                why.append("mouth-silent")
            if st["intact_a"]["tone_level"] in (None, 0):
                why.append("not-engaged")
            if len({t["reply"] for t in st.values()}) != 1 or len({json.dumps(t["mouth"]) for t in st.values()}) != 1:
                why.append("null-noise")
            per[str(s)] = {"ok": not why, "why": why, "tone_level": st["intact_a"]["tone_level"],
                           "calls": st["intact_a"]["calls"]}
        n_ok = sum(1 for v in per.values() if v["ok"])
        out["candidates"][c["id"]] = {"question": c["question"], "rich": c["rich"], "n_ok": n_ok,
                                      "eligible": n_ok >= DEV_MIN_SEEDS, "seeds": per}
    out["missing_or_invalid"] = sorted(set(out["missing_or_invalid"]))
    smoke = {}
    for s in DEV_LESION_SEEDS:
        for a in DEV_LESION_ARMS:
            doc = load(base, s, a)
            bad = validity(doc, pin, s, a)
            row = {}
            for c in CANDIDATES:
                t = read_turn(doc, q_label(c["id"])) if not bad else {"state": "E"}
                row[c["id"]] = (t["state"] or ("ok" if t["route_match"] and t["conditioning_visible"]
                                               else "route-mismatch"))
            smoke["s%d/%s" % (s, a)] = {"valid": not bad, "fails": bad, "route": row}
    out["lesion_smoke"] = smoke
    if out["missing_or_invalid"] and not final:
        out["status"], out["selected"] = "INCOMPLETE", None
        return out
    sel = next((c["id"] for c in CANDIDATES if out["candidates"][c["id"]]["eligible"]), None)
    out["selected"] = sel
    smoke_ok = all(v["valid"] and v["route"].get(sel) == "ok" for v in smoke.values()) if sel else False
    out["lesion_smoke_ok"] = smoke_ok
    out["status"] = ("SELECTED" if sel and smoke_ok else ("NO-CANDIDATE" if not sel else "LESION-SMOKE-FAILED"))
    return out


def route_check(base, pin, selected, seeds=GATE_SEEDS):
    """The gate's route arm, read before the other 48 lines are queued (HANDOFF step 6): per seed, whether the `route`
    arm is valid and its probe turn on the selected candidate was spoken by Qwen with the manner clause in the prompt.
    Engagement is reported, not required (a NOT-ENGAGED gate seed is a registered UNDEFINED, not a stop)."""
    label = q_label(selected)
    per = {}
    for s in seeds:
        t = _arm_state(base, pin, s, "route", label, None)
        if t["state"] is not None:
            per[str(s)] = {"ok": False, "why": [t["state"]] + t.get("fails", [])}
            continue
        why = [w for w, bad in (("route-mismatch", not (t["route_match"] and t["conditioning_visible"])),
                                ("mouth-silent", t["calls"] == 0 or not t["mouth"])) if bad]
        per[str(s)] = {"ok": not why, "why": why, "tone_level": t["tone_level"], "calls": t["calls"]}
    return {"pin": pin, "selected": selected, "seeds": per,
            "status": "ROUTE-OK" if all(v["ok"] for v in per.values()) else "ROUTE-FAILED"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("--pin", required=True)
    j.add_argument("--phase", required=True, choices=("dev", "dev-lesion", "gate-route", "gate-rest"))
    j.add_argument("--out", required=True)
    d = sub.add_parser("dev")
    d.add_argument("--pin", required=True)
    d.add_argument("--base", required=True)
    d.add_argument("--final", action="store_true", help="after the one re-run: score with what is there")
    d.add_argument("--out", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--pin", required=True)
    g.add_argument("--base", required=True)
    g.add_argument("--selection", required=True)
    g.add_argument("--attempt1", default=None)
    g.add_argument("--out", required=True)
    r = sub.add_parser("routecheck")
    r.add_argument("--pin", required=True)
    r.add_argument("--base", required=True)
    r.add_argument("--selection", required=True)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    _check_pin(a.pin)
    sel = None
    if a.cmd in ("gate", "routecheck"):
        sel = json.load(open(a.selection))
        if sel.get("status") != "SELECTED" or sel.get("pin") != a.pin:
            print("REFUSED: the selection file is not SELECTED at this pin (status %r, pin %r)"
                  % (sel.get("status"), sel.get("pin")), file=sys.stderr)
            return 1
    if a.cmd == "routecheck":
        res = route_check(a.base, a.pin, sel["selected"])
        summary = {"status": res["status"], "selected": sel["selected"],
                   "failed": {s: v["why"] for s, v in res["seeds"].items() if not v["ok"]}}
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(json.dumps(summary, indent=1))
        return 0
    if a.cmd == "jobs":
        lines = job_lines(a.pin, a.phase)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("wrote %d %s lines (pin %s) -> %s" % (len(lines), a.phase, a.pin, a.out))
        return 0
    if a.cmd == "dev":
        res = score_dev(a.base, a.pin, final=a.final)
        summary = {"status": res["status"], "selected": res["selected"],
                   "n_ok": {k: v["n_ok"] for k, v in res["candidates"].items()},
                   "missing_or_invalid": res["missing_or_invalid"], "lesion_smoke_ok": res.get("lesion_smoke_ok")}
    else:
        res = score_gate(a.base, a.pin, sel["selected"], attempt1=a.attempt1)
        prim = res["candidates"][sel["selected"]]["rows"]
        summary = {"verdict": res["verdict"], "selected": sel["selected"],
                   "rows": {r: {k: v[k] for k in ("reading", "defined", "mouth_load_bearing", "interval", "host_only")}
                            for r, v in prim.items()},
                   "rerun_arms": res["rerun_arms"], "needs_claude": res["needs_claude"]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
