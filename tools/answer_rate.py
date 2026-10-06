#!/usr/bin/env python3
"""answer_rate.py -- does a default-on candidate make the brain ANSWER LESS than the flag-off default? (REPORTED check)

Why (Claude's 10-06 review of Hermes): wm-focus-bind was put forward for the iw1 default flip although, on its own
probe, flag-on abstained ("I don't know about that.") on 2 of 4 sessions where flag-off answered all 4, on 6/6 seeds.
Every registered bar passed; none of them asks whether the flipped brain still ANSWERS. Flipping a faculty that makes
replies worse fails the integration goal (RULING iw1-wmf-hold). This is the mechanical version of that review.

What it reads: for each seed, the PROBE reply of every INTACT arm file in a flag-on cell and a flag-off cell. The
probe turn of an arm file is its last turn body that carries an `answer` (every packet's cell ends on its probe).
A reply is an ABSTENTION when `abstained is True`, or when its text matches the abstention phrases the production
code emits (webapp/server.py: "I don't know about that."; webapp/d6_hebbian_chat.py: "I haven't learned about X
yet"); an empty answer is a non-answer. A reply whose flag says False but whose text matches is counted too and listed
as `text_only` (the flag is the primary signal; the text is the fallback the task names, so a record that predates the
flag stays detectable). Answer rate = answered / total.

Verdict field: `flags` carries `FLIP-HURTS-ANSWERS` when, on ANY measured seed, the flag-on answer rate is below the
flag-off one. That is REPORTED beside the registered verdict and changes no PASS/FAIL; the flip then needs a Claude
ruling (a NEEDS line) before it lands. A seed missing either side is `UNMEASURED` (listed, never read as "no harm").

Two cell layouts:
  prefixed  one shard dir holding `on_intact_*` and `off_intact_*` arm files (the wm-focus-bind probe's layout):
     python tools/answer_rate.py prefixed 'research/findings/raw/_load_bearing/_shards/<tag>/s*/<faculty dir>'
  paired    the same faculty dir under two shard tags, `--on-tag` (flip/flag-on) and `--off-tag` (flag-off, e.g. the
            iw*-alloff-<F7> attribution run), per seed:
     python tools/answer_rate.py paired --on-tag iw2-flip-<F7> --off-tag iw2-alloff-<F7> --faculty tom-false-belief
Library: `report(per_seed)`, `prefixed_per_seed(dirs)`, `paired_per_seed(base, on_tag, off_tag, faculty, seeds)`.
Exit 0 always (reported, not gated) unless --strict, which exits 4 on FLIP-HURTS-ANSWERS and 3 when nothing measured.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "research/findings/raw/_load_bearing/_shards"
SEEDS = [42, 43, 44, 100, 101, 102]
HURTS = "FLIP-HURTS-ANSWERS"
# the abstention phrases the production code emits (webapp/server.py 6846 "I don't know about that.";
# webapp/d6_hebbian_chat.py "I haven't learned about <x> yet")
ABSTAIN_RES = [re.compile(r"\bi\s+(?:do\s+not|don['’]?t)\s+know\s+about\s+that\b", re.I),
               re.compile(r"\bi\s+haven[’']?t\s+learned\s+about\b.*\byet\b", re.I)]


def classify(body):
    """'answered' | 'abstained' | 'abstained-text-only' | 'empty' for one reply body (a dict with `answer`)."""
    if not isinstance(body, dict):
        return "empty"
    ans = body.get("answer")
    if body.get("abstained") is True:
        return "abstained"
    if not isinstance(ans, str) or not ans.strip():
        return "empty"
    if any(p.search(ans) for p in ABSTAIN_RES):
        return "abstained-text-only"
    return "answered"


def rate(bodies):
    """{n, answered, abstained, empty, text_only, answer_rate} (answer_rate None when n == 0)."""
    kinds = [classify(b) for b in bodies]
    n = len(kinds)
    answered = kinds.count("answered")
    return {"n": n, "answered": answered,
            "abstained": kinds.count("abstained") + kinds.count("abstained-text-only"),
            "empty": kinds.count("empty"), "text_only": kinds.count("abstained-text-only"),
            "answer_rate": (answered / n) if n else None}


def probe_body(path):
    """The last turn body carrying an `answer` in one arm file (None if unreadable or none)."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None
    found = [v for v in doc.values() if isinstance(v, dict) and "answer" in v]
    return found[-1] if found else None


def _arm_files(d):
    return sorted(p for p in glob.glob(os.path.join(d, "*.json"))
                  if not p.endswith(".prov.json") and os.path.basename(p) != "lb.json")


def probes_in(d, arm_re):
    """Probe bodies of the arm files in `d` whose basename matches `arm_re`."""
    out = []
    for p in _arm_files(d):
        if re.search(arm_re, os.path.basename(p)):
            b = probe_body(p)
            if b is not None:
                out.append(b)
    return out


def report(per_seed, label=""):
    """`per_seed` = {seed: {"on": [bodies], "off": [bodies]}} -> the reported check (JSON-able)."""
    rows, hurts, unmeasured = {}, [], []
    for seed in sorted(per_seed, key=lambda s: (0, int(s), "") if str(s).isdigit() else (1, 0, str(s))):
        on, off = rate(per_seed[seed].get("on") or []), rate(per_seed[seed].get("off") or [])
        row = {"on": on, "off": off}
        if on["n"] == 0 or off["n"] == 0:
            row["status"] = "UNMEASURED"
            unmeasured.append(str(seed))
        elif on["answer_rate"] < off["answer_rate"]:
            row["status"] = HURTS
            hurts.append(str(seed))
        else:
            row["status"] = "ok"
        rows[str(seed)] = row
    measured = len(rows) - len(unmeasured)
    flags = [HURTS] if hurts else []
    status = HURTS if hurts else ("UNMEASURED" if (not measured or unmeasured) else "OK")
    return {"check": "answer_rate", "label": label, "per_seed": rows, "n_seeds": len(rows), "n_measured": measured,
            "unmeasured_seeds": unmeasured, "hurt_seeds": hurts, "flags": flags, "status": status,
            "rule": "FLIP-HURTS-ANSWERS when flag-on answer rate < flag-off on any measured seed; reported only, "
                    "changes no registered PASS/FAIL; the flip then needs a Claude ruling (NEEDS line)",
            "abstention_detection": "abstained flag; fallback text patterns %s" % [p.pattern for p in ABSTAIN_RES]}


def prefixed_per_seed(dirs):
    """One shard dir per seed (`.../s<seed>/<faculty>`) holding on_intact_* / off_intact_* arm files."""
    out = {}
    for d in dirs:
        m = re.search(r"/s(\d+)(?:/|$)", d + "/")
        seed = m.group(1) if m else d
        out[seed] = {"on": probes_in(d, r"^on_intact"), "off": probes_in(d, r"^off_intact")}
    return out


def paired_per_seed(base, on_tag, off_tag, faculty, seeds=SEEDS):
    out = {}
    for s in seeds:
        on_d = os.path.join(base, on_tag, "s%d" % s, faculty)
        off_d = os.path.join(base, off_tag, "s%d" % s, faculty)
        out[str(s)] = {"on": probes_in(on_d, r"^intact"), "off": probes_in(off_d, r"^intact")}
    return out


def render(rep):
    lines = ["answer-rate check %s: %s  (measured %d/%d seeds)" % (rep.get("label") or "", rep["status"],
                                                                  rep["n_measured"], rep["n_seeds"])]
    for s, r in rep["per_seed"].items():
        fmt = lambda x: "-" if x["answer_rate"] is None else "%d/%d (abstained %d, empty %d)" % (
            x["answered"], x["n"], x["abstained"], x["empty"])
        lines.append("  seed %-4s on %-34s off %-34s %s" % (s, fmt(r["on"]), fmt(r["off"]), r["status"]))
    if rep["flags"]:
        lines.append("  %s on seed(s) %s: the flip needs a Claude ruling (NEEDS line) before it lands"
                     % (HURTS, ",".join(rep["hurt_seeds"])))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    p1 = sub.add_parser("prefixed")
    p1.add_argument("dirs", nargs="+", help="shard dirs or globs (one per seed)")
    p2 = sub.add_parser("paired")
    p2.add_argument("--on-tag", required=True)
    p2.add_argument("--off-tag", required=True)
    p2.add_argument("--faculty", required=True, help="the faculty's shard dir name")
    p2.add_argument("--base", default=BASE)
    p2.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    for p in (p1, p2):
        p.add_argument("--json", action="store_true")
        p.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    if a.mode == "prefixed":
        dirs = sorted({d for g in a.dirs for d in (glob.glob(g) or [g]) if os.path.isdir(d)})
        if not dirs:
            print("answer_rate: no shard dir matched", file=sys.stderr)
            return 2
        rep = report(prefixed_per_seed(dirs), label=a.dirs[0])
    else:
        rep = report(paired_per_seed(a.base, a.on_tag, a.off_tag, a.faculty, [int(x) for x in a.seeds.split(",")]),
                     label="%s: %s vs %s" % (a.faculty, a.on_tag, a.off_tag))
    print(json.dumps(rep, indent=1) if a.json else render(rep))
    if a.strict:
        return 4 if rep["flags"] else (3 if rep["status"] == "UNMEASURED" else 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
