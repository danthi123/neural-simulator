"""Reader for the C3 L+P default flip: does turning L+P on change anything OUTSIDE the ask path?

Bars (RULING c3-flip-1029b, verbatim in the flip prereg): on the six gate seeds, Stage-2 groups conv, infer, emo and inner,
arm intact_a, flags ON vs unset at one revision F: `abstained` and `recalled_svo` are identical ON vs OFF on every turn;
`answer` is identical on every turn that did not abstain in both arms; no turn where OFF answered and ON did not
(`tools.answer_rate.classify`, FLIP-HURTS-ANSWERS); every other differing leaf is reported (counts and first path per
group), never read as a pass.

Why `answer` is exempt on a turn that abstained in both arms: there the curiosity follow-up sits inside `answer` ("My
curiosity is piqued ..."), and changing whether and what it asks is what L+P does. That behaviour is the C3 gate's
(research/findings/2026-10-29-c3-topic-stop-gate-RESULT.md), not this reader's.

Layout: <root>/s<seed>/<group>__intact_a.json (+ .prov.json), the Stage-2 runner's `--shard --out` files. The verdict is
UNDEFINED unless every cell exists on both sides, the ON sidecars carry both flags = "1", the OFF sidecars carry neither,
and every sidecar names one revision (tools.verdict).

    python -m research.runners.c3_flip_compare --on <ON root> --off <OFF root> [--seeds 42,...] [--out verdict.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners.c3_topic_stop_score import walk_diffs, _show  # noqa: E402
from tools.answer_rate import classify, rate  # noqa: E402
from tools.verdict import Verdict  # noqa: E402

GROUPS = ("conv", "infer", "emo", "inner")
ARM = "intact_a"
GATE_SEEDS = (42, 43, 44, 100, 101, 102)
FLAGS = ("BRAIN_CURIOSITY_TOPIC_STOP", "BRAIN_CURIOSITY_PHASIC_DA")
# host / timing / provenance fields, never compared (the same list as the C3 identity canary)
EXCLUDED_TOP = ("timings_s", "import_s", "wall_s", "elapsed_seconds", "process")
HURTS = "FLIP-HURTS-ANSWERS"


def _load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def cell_paths(root, seed, group):
    shard = os.path.join(root, "s%d" % int(seed), "%s__%s.json" % (group, ARM))
    return shard, shard + ".prov.json"


def _abst(body):
    return isinstance(body, dict) and body.get("abstained") is True


def compare_cell(on, off):
    """One (seed, group) pair: the bar clauses per turn plus every other differing leaf."""
    R_on, R_off = on.get("responses") or {}, off.get("responses") or {}
    labels = list(R_off) + [l for l in R_on if l not in R_off]
    res = {"labels": labels, "abstained_diff": [], "svo_diff": [], "answer_diff": [], "hurts": [],
           "both_abstained": [], "missing_label": [], "n_other_leaf_diffs": 0, "first_other": None,
           "on_topic_stop_traces": 0, "off_topic_stop_traces": 0}
    for l in labels:
        a, b = R_on.get(l), R_off.get(l)
        if not isinstance(a, dict) or not isinstance(b, dict):
            res["missing_label"].append(l)
            continue
        if a.get("abstained") != b.get("abstained"):
            res["abstained_diff"].append(l)
        if a.get("recalled_svo") != b.get("recalled_svo"):
            res["svo_diff"].append(l)
        both = _abst(a) and _abst(b)
        if both:
            res["both_abstained"].append(l)
        elif a.get("answer") != b.get("answer"):
            res["answer_diff"].append(l)
        if classify(b) == "answered" and classify(a) != "answered":
            res["hurts"].append(l)
        for body, key in ((a, "on_topic_stop_traces"), (b, "off_topic_stop_traces")):
            cur = body.get("curiosity")
            if isinstance(cur, dict) and "curiosity_topic_stop" in cur:
                res[key] += 1
    skip = {"responses.%s.answer" % l for l in res["both_abstained"]}
    ref = {k: v for k, v in off.items() if k not in EXCLUDED_TOP}
    got = {k: v for k, v in on.items() if k not in EXCLUDED_TOP}
    for p, x, y in walk_diffs(ref, got):
        if p in skip:
            continue
        res["n_other_leaf_diffs"] += 1
        if res["first_other"] is None:
            res["first_other"] = {"path": p, "off": _show(x), "on": _show(y)}
    res["answered_on"] = rate([R_on[l] for l in labels if l in R_on])["answered"]
    res["answered_off"] = rate([R_off[l] for l in labels if l in R_off])["answered"]
    res["pass"] = not (res["abstained_diff"] or res["svo_diff"] or res["answer_diff"] or res["hurts"]
                       or res["missing_label"])
    return res


def score(on_root, off_root, seeds=GATE_SEEDS, groups=GROUPS):
    cells, missing, shas, env_on_bad, env_off_bad = {}, [], set(), [], []
    for s in seeds:
        for g in groups:
            key = "%d/%s" % (int(s), g)
            (on_p, on_pp), (off_p, off_pp) = cell_paths(on_root, s, g), cell_paths(off_root, s, g)
            on, off, pon, poff = _load(on_p), _load(off_p), _load(on_pp), _load(off_pp)
            if on is None or off is None or pon is None or poff is None:
                missing.append(key)
                continue
            shas.update({pon.get("git_sha"), poff.get("git_sha")})
            eon, eoff = pon.get("env") or {}, poff.get("env") or {}
            if any(eon.get(f) != "1" for f in FLAGS):
                env_on_bad.append(key)
            if any(f in eoff for f in FLAGS):
                env_off_bad.append(key)
            cells[key] = compare_cell(on, off)
    v = Verdict("C3 L+P flip: nothing outside the ask path changes (RULING c3-flip-1029b)")
    n_exp = len(seeds) * len(groups)
    v.require("every cell present on both sides (shard + sidecar)", not missing, note="missing: %s" % missing)
    v.require("ON sidecars carry both flags = 1", bool(cells) and not env_on_bad, note="bad: %s" % env_on_bad)
    v.require("OFF sidecars carry neither flag", bool(cells) and not env_off_bad, note="bad: %s" % env_off_bad)
    v.require("one revision across every sidecar", len(shas) == 1 and None not in shas, note="shas: %s" % sorted(map(str, shas)))
    v.require("cells compared == expected", len(cells) == n_exp, note="%d of %d" % (len(cells), n_exp))
    all_pass = bool(cells) and all(c["pass"] for c in cells.values())
    dec = v.decide(go=all_pass, verbose=False)
    per_group = {}
    for g in groups:
        cs = [c for k, c in cells.items() if k.split("/", 1)[1] == g]
        per_group[g] = {"n_cells": len(cs), "n_pass": sum(c["pass"] for c in cs),
                        "n_other_leaf_diffs": sum(c["n_other_leaf_diffs"] for c in cs),
                        "first_other": next((c["first_other"] for c in cs if c["first_other"]), None),
                        "on_topic_stop_traces": sum(c["on_topic_stop_traces"] for c in cs),
                        "off_topic_stop_traces": sum(c["off_topic_stop_traces"] for c in cs)}
    return {"mode": "c3-flip-compare", "on_root": on_root, "off_root": off_root, "seeds": list(seeds),
            "groups": list(groups), "arm": ARM, "revision": sorted(map(str, shas)), "backend": "numpy", "device": "cpu",
            "device_note": "reader only: compares shard JSON on the CPU and runs no simulation",
            "cells": cells, "per_group": per_group,
            "flags": [HURTS] if any(c["hurts"] for c in cells.values()) else [],
            "status": dec["status"], "go": dec["go"], "preconditions": dec["preconditions"],
            "undefined_reasons": dec["undefined_reasons"]}


def render(res):
    out = ["C3 flip compare: %s  (ON %s vs OFF %s)" % (res["status"], res["on_root"], res["off_root"])]
    for r in res["undefined_reasons"]:
        out.append("  UNDEFINED: %s" % r)
    for k, c in res["cells"].items():
        bad = ["%s %s" % (n, c[n]) for n in ("abstained_diff", "svo_diff", "answer_diff", "hurts", "missing_label") if c[n]]
        out.append("  cell %-10s %s  answered on/off %d/%d  both-abstained %d  other-leaf-diffs %d%s"
                   % (k, "PASS" if c["pass"] else "FAIL", c["answered_on"], c["answered_off"], len(c["both_abstained"]),
                      c["n_other_leaf_diffs"], ("  " + "; ".join(bad)) if bad else ""))
    for g, p in res["per_group"].items():
        out.append("  group %-5s %d/%d PASS  other-leaf-diffs %d  first %s  topic-stop traces on/off %d/%d"
                   % (g, p["n_pass"], p["n_cells"], p["n_other_leaf_diffs"], json.dumps(p["first_other"])[:160],
                      p["on_topic_stop_traces"], p["off_topic_stop_traces"]))
    if res["flags"]:
        out.append("  " + " ".join(res["flags"]))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--on", required=True)
    ap.add_argument("--off", required=True)
    ap.add_argument("--seeds", default=",".join(map(str, GATE_SEEDS)))
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = score(a.on, a.off, tuple(int(x) for x in a.seeds.split(",")), tuple(a.groups.split(",")))
    print(render(res))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
