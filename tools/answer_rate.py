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
            exactly one of --off-tag / --off-arm: `--off-arm <lesion arm stem>` takes the off side from the LESION arm
            beside the intact arms in the ON-tag dir (the alloff cells abstain by design via REQUIRED_ENV, so the off
            side must be a cell the brain ran normally -- RULING answer-rate-off); its label carries `off = lesion arm`:
     python tools/answer_rate.py paired --on-tag iw1-R-d6-049d4bb --off-arm lesion_d6_hebbian_write --faculty d6-hebbian-write
Library: `report(per_seed)`, `prefixed_per_seed(dirs)`,
         `paired_per_seed(base, on_tag, off_tag, faculty, seeds, off_arm=None)`.
A record written by the CLI (`--json`) is a SMOKE RECORD that tools/smoke_first.py can validate (RULING smoke-first-1018):
beside the report it carries `mode`, `faculty`, `on_tag`, `off_tag` / `off_arm`, `seeds`, `evidence` (one {file, sha256}
per probe file read), and the stamps the commit gates read from any result JSON (`backend`, `device`, `analysis_only`,
`script`, `argv`: this tool reads existing arm files and runs no brain). `paired` refuses `--on-tag` == `--off-tag`
(a flip compared with itself can never show a harm). Redirect it to `research/findings/smoke/<name>.json`, outside raw/.
Exit 0 always (reported, not gated) unless --strict, which exits 4 on FLIP-HURTS-ANSWERS and 3 when nothing measured.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
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


def probes_in(d, arm_re, files=None):
    """Probe bodies of the arm files in `d` whose basename matches `arm_re`. `files` (a list), when given, receives the
    path of every arm file that yielded a body (the evidence a smoke record cites)."""
    out = []
    for p in _arm_files(d):
        if re.search(arm_re, os.path.basename(p)):
            b = probe_body(p)
            if b is not None:
                out.append(b)
                if files is not None:
                    files.append(p)
    return out


def report(per_seed, label="", mode=None):
    """`per_seed` = {seed: {"on": [bodies], "off": [bodies]}} -> the reported check (JSON-able). `mode` ("paired" |
    "prefixed"; the CLI passes it, added 2026-10-18) is recorded as the report's `mode` key ONLY when given, so
    tools/smoke_first.py can tell a paired smoke record from a prefixed one while every other caller (tools/iw2_score.py)
    gets the report byte-for-byte as before; no existing key changes."""
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
    rep = {"check": "answer_rate", "label": label, "per_seed": rows, "n_seeds": len(rows), "n_measured": measured,
           "unmeasured_seeds": unmeasured, "hurt_seeds": hurts, "flags": flags, "status": status,
           "rule": "FLIP-HURTS-ANSWERS when flag-on answer rate < flag-off on any measured seed; reported only, "
                   "changes no registered PASS/FAIL; the flip then needs a Claude ruling (NEEDS line)",
           "abstention_detection": "abstained flag; fallback text patterns %s" % [p.pattern for p in ABSTAIN_RES]}
    if mode is not None:
        rep["mode"] = mode
    return rep


def prefixed_per_seed(dirs, evidence=None):
    """One shard dir per seed (`.../s<seed>/<faculty>`) holding on_intact_* / off_intact_* arm files. `evidence` (a dict),
    when given, is filled with {seed: {"on": [paths], "off": [paths]}}."""
    out = {}
    for d in dirs:
        m = re.search(r"/s(\d+)(?:/|$)", d + "/")
        seed = m.group(1) if m else d
        files = {"on": [], "off": []}
        out[seed] = {"on": probes_in(d, r"^on_intact", files["on"]), "off": probes_in(d, r"^off_intact", files["off"])}
        if evidence is not None:
            evidence[seed] = files
    return out


def paired_per_seed(base, on_tag, off_tag, faculty, seeds=SEEDS, off_arm=None, evidence=None):
    """Per seed: on = `^intact` in the on-tag dir; off = `^intact` in the off-tag dir, or, with `off_arm`, the lesion
    arm of that name in the ON-tag dir (off = lesion arm; the alloff cells abstain by design via REQUIRED_ENV).
    `evidence` (a dict), when given, is filled with {seed: {"on": [paths], "off": [paths]}}."""
    out = {}
    for s in seeds:
        on_d = os.path.join(base, on_tag, "s%d" % s, faculty)
        files = {"on": [], "off": []}
        out[str(s)] = {"on": probes_in(on_d, r"^intact", files["on"]),
                       "off": probes_in(on_d, "^" + re.escape(off_arm), files["off"]) if off_arm
                              else probes_in(os.path.join(base, off_tag, "s%d" % s, faculty), r"^intact", files["off"])}
        if evidence is not None:
            evidence[str(s)] = files
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp_record(rep, argv, evidence, extra=None):
    """The fields only the CLI adds to a report (tools/iw2_score.py, a library caller, gets the bare report): the
    evidence (one {file, sha256} per probe file), the stamps the commit gates read from any result JSON, and `extra`."""
    ev = {}
    for seed, sides in (evidence or {}).items():
        ev[str(seed)] = {side: [{"file": os.path.relpath(p, ROOT) if os.path.abspath(p).startswith(ROOT + os.sep) else p,
                                 "sha256": _sha256(p)} for p in paths] for side, paths in sides.items()}
    rep.update(extra or {})
    rep.update({"evidence": ev, "script": "tools/answer_rate.py", "argv": list(argv),
                "backend": "numpy", "device": "cpu", "analysis_only": True})
    return rep


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
    p2.add_argument("--off-tag", help="flag-off shard tag (alloff attribution run); exactly one of --off-tag/--off-arm")
    p2.add_argument("--off-arm", dest="off_arm",
                    help="off side = this LESION arm inside the ON-tag dir (label: off = lesion arm)")
    p2.add_argument("--faculty", required=True, help="the faculty's shard dir name")
    p2.add_argument("--base", default=BASE)
    p2.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    for p in (p1, p2):
        p.add_argument("--json", action="store_true")
        p.add_argument("--strict", action="store_true")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    a = ap.parse_args(raw_argv)
    evidence = {}
    if a.mode == "prefixed":
        dirs = sorted({d for g in a.dirs for d in (glob.glob(g) or [g]) if os.path.isdir(d)})
        if not dirs:
            print("answer_rate: no shard dir matched", file=sys.stderr)
            return 2
        rep = stamp_record(report(prefixed_per_seed(dirs, evidence), label=a.dirs[0], mode="prefixed"), raw_argv,
                           evidence)
    else:
        if bool(a.off_tag) == bool(a.off_arm):                       # both or neither: exactly one off side
            print("answer_rate: paired needs EXACTLY one of --off-tag / --off-arm", file=sys.stderr)
            return 2
        if a.off_tag and a.off_tag == a.on_tag:
            print("answer_rate: --on-tag and --off-tag are both %r: a flip compared with itself can never show a harm"
                  % a.on_tag, file=sys.stderr)
            return 2
        label = ("%s: %s vs off-arm %s (off = lesion arm)" % (a.faculty, a.on_tag, a.off_arm) if a.off_arm
                 else "%s: %s vs %s" % (a.faculty, a.on_tag, a.off_tag))
        seeds = [int(x) for x in a.seeds.split(",")]
        rep = report(paired_per_seed(a.base, a.on_tag, a.off_tag, a.faculty, seeds, off_arm=a.off_arm,
                                     evidence=evidence), label=label, mode="paired")
        stamp_record(rep, raw_argv, evidence, {"faculty": a.faculty, "on_tag": a.on_tag, "off_tag": a.off_tag or None,
                                               "off_arm": a.off_arm or None, "seeds": seeds})
    print(json.dumps(rep, indent=1) if a.json else render(rep))
    if a.strict:
        return 4 if rep["flags"] else (3 if rep["status"] == "UNMEASURED" else 0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
