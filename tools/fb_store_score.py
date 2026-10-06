#!/usr/bin/env python3
"""Scorer for research/findings/2026-10-05-fb-store-lesion-PREREGISTRATION.md (cloud Claude session 7, 2026-10-05).

Reads the six `tom-false-belief-store` shards (research/findings/raw/_load_bearing/_shards/<tag>/s<seed>/
tom-false-belief-store/) and applies the prereg's section-4 table. Per seed a cell is DEFINED when its lb.json row
reads null_control_clean and lesion_reproduced, the two intact arms answer the false-belief location ("basket", not
abstained), and the lesion ENGAGED: the lesion arm's `false_belief_tom.store_lesioned` is True, or (the predicted
effect of an emptied store) the organ answered nothing on the lesion arm (no `false_belief_tom` block) while it
answered on both intact arms. Anything else is UNDEFINED for that seed (re-run once per the prereg).

    python tools/fb_store_score.py --tag fbstore-<PIN7> --pin <PIN> --out research/findings/raw/_fb_store/verdict.json
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from verdict import UNDEFINED as V_UNDEFINED, Verdict  # noqa: E402

SEEDS = [42, 43, 44, 100, 101, 102]
BASE = "research/findings/raw/_load_bearing/_shards"
ROW = "tom-false-belief-store"
TURN = "tom_fb"


def _arm(d, pattern):
    hits = sorted(p for p in glob.glob(os.path.join(d, pattern)) if p.endswith(".json"))
    if not hits:
        return None
    with open(hits[0]) as f:
        return (json.load(f) or {}).get(TURN)


def judge_cell(d):
    """-> (state, load_bearing, detail)."""
    try:
        with open(os.path.join(d, "lb.json")) as f:
            lb = json.load(f)
    except (OSError, ValueError) as e:
        return "UNDEFINED:no-lb", None, {"error": str(e)}
    rows = [r for r in lb.get("per_faculty", []) if r.get("faculty") == ROW]
    if not rows:
        return "UNDEFINED:row-absent", None, {}
    r = rows[0]
    if r.get("verdict") not in ("regressed", "pass") or not r.get("null_control_clean") or not r.get("lesion_reproduced"):
        return "UNDEFINED:lb-%s" % r.get("verdict"), None, {"lb": {k: r.get(k) for k in (
            "verdict", "null_control_clean", "lesion_reproduced")}}
    ia, ib, le = _arm(d, "intact_a_*"), _arm(d, "intact_b_*"), _arm(d, "lesion_*")
    if ia is None or ib is None or le is None:
        return "UNDEFINED:arm-missing", None, {}
    det = {"intact_answer": ia.get("answer"), "lesion_answer": le.get("answer"),
           "lesion_abstained": le.get("abstained")}
    for arm in (ia, ib):
        if arm.get("abstained") or "basket" not in str(arm.get("answer") or "").lower():
            return "UNDEFINED:intact-wrong", None, det
        if not (arm.get("false_belief_tom") or {}).get("acted"):
            return "UNDEFINED:intact-not-organ", None, det
        if (arm.get("false_belief_tom") or {}).get("store_lesioned"):
            return "UNDEFINED:intact-lesioned", None, det
    fbt = le.get("false_belief_tom")
    engaged = bool((fbt or {}).get("store_lesioned")) or fbt is None
    det["engagement"] = "store_lesioned" if (fbt or {}).get("store_lesioned") else ("organ-silent" if fbt is None
                                                                                   else "none")
    if not engaged:
        return "UNDEFINED:not-engaged", None, det
    return "DEFINED", r.get("verdict") == "regressed", det


def score(tag, pin, base=BASE, seeds=SEEDS):
    per = {}
    for s in seeds:
        st, lb, det = judge_cell(os.path.join(base, tag, "s%d" % s, ROW))
        per[str(s)] = {"state": st, "load_bearing": lb, "detail": det}
    n_def = sum(1 for c in per.values() if c["state"] == "DEFINED")
    n_lb = sum(1 for c in per.values() if c["state"] == "DEFINED" and c["load_bearing"])
    if n_def < len(seeds):
        verdict = "UNDEFINED"
    elif n_lb == len(seeds):
        verdict = "STORE-LOAD-BEARING"
    elif n_lb == 0:
        verdict = "STORE-NOT-LOAD-BEARING"
    else:
        verdict = "MIXED"
    res = {"step": "fb-store", "tag": tag, "pin": pin, "row": ROW, "per_seed": per, "n_defined": n_def,
           "n_reply_load_bearing": n_lb, "verdict": verdict}
    v = Verdict("fb-store-lesion %s at %s" % (tag, pin))
    v.require("cells DEFINED on %d/%d seeds" % (len(seeds), len(seeds)), n_def, expect=len(seeds))
    d = v.decide(go=verdict != "UNDEFINED", verbose=False)
    if verdict != "UNDEFINED" and d["status"] == V_UNDEFINED:
        res["verdict_unearned"], res["verdict"] = verdict, "UNDEFINED"
    res["preconditions"], res["undefined_reasons"] = d["preconditions"], d["undefined_reasons"]
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pin", required=True)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = score(a.tag, a.pin, base=a.base)
    res["runner"] = "tools/fb_store_score.py"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("verdict", "n_defined", "n_reply_load_bearing")}))
    return 0 if res["verdict"] != "UNDEFINED" else 3


if __name__ == "__main__":
    sys.exit(main())
