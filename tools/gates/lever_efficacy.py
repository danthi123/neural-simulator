"""Failure class 1 — MANIPULATION-NEVER-ENGAGED (10 incidents, the most expensive class).

EVIDENCE. A 17-agent audit of 539 sites found a lesion whose target gate was NEVER DECLARED anywhere: the
"intact" arm read 0.735 and the "lesion" arm 0.765 — the same run twice, under two names. In the gap#4 arc the
crux `kp` (learned-feedback) arm HAD NEVER RUN — it was gated on a feedback value no arm supplies — and its
recorded numbers matched `fixed_fa` BYTE-IDENTICALLY. `tools/lab.py:lever()` exists to catch exactly this and is
imported by 2 of 1330 runners, so the manipulation is verified nowhere on the path that produces a verdict.

WHAT IT DETECTS — the arithmetic signature the class always leaves: two *distinct* arms reporting *identical*
values, since two different rules cannot agree to fifteen significant figures. In artifact JSON: sibling dicts
under one parent that share a metric schema and agree on EVERY numeric field. In findings markdown: two
differently-labelled rows of one table whose numeric cells are identical.

CALIBRATION (narrow on purpose; a gate that cries wolf gets ignored). Measured base rate over the whole corpus:
63 / 7636 artifact JSONs and 15 / 1841 findings, ~0.8% each. Ties that are NOT evidence are excluded, and those
exclusions are why the rate is that low: exact 0.0/±1.0 (and 0.5, 100.0 in prose) are floors, ceilings and chance
baselines; whole numbers are counts, which tie constantly; a simple rational p/q with q<=1000 (1/3, 11/18, 5/54)
is a quantized small-n accuracy two honest arms CAN both land on; under 10 significant digits (JSON) or under 3
decimals (prose) is rounding, which manufactures ties — only a raw computed float carries enough entropy for an
exact match to be proof; a pair must agree on >=2 *distinct* surviving values, so one shared number is not enough;
and provenance keys (seed, config, args, hash, elapsed, ...) are identical across arms by design.

WHAT IT CANNOT CATCH. (1) An inert lever whose arm still differs by noise — identity is sufficient evidence of a
dead arm, never necessary, so a 99%-inert lever looks fine here; only `lab.lever()` (assert the manipulated
quantity moved) catches that, and this gate is no substitute for it. (2) A lever never declared in ANY arm: if
the second arm was never written, there is nothing to compare. (3) Arms in separate files or separate runs.
(4) Rounded prose below 3 decimals. (5) A control block legitimately computed once and copied into both arms —
that reads as a violation, is reported, and a human resolves it.

NON-BLOCKING, on purpose: the artifact that *documents* a dead-arm failure (a retraction, this class's own
evidence) is numerically indistinguishable from one that *commits* it, and blocking the former would be perverse.
"""
from __future__ import annotations

import json, os, re, tempfile
from fractions import Fraction

NAME = "lever-efficacy"
CLASS_ID = "1"
BLOCKING = False

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MAX_BYTES = 2_000_000
_MAX_REPORT = 40
_DENY = ("seed", "config", "args", "hash", "timestamp", "elapsed", "duration", "filename", "_path")
_NUM = re.compile(r"-?\d+\.\d+|-?\d+")


def _sig(x):
    r = repr(float(x)).split("e")[0]
    return len(r.replace("-", "").replace(".", "").lstrip("0").rstrip("0"))


def _informative(v):
    """A value whose exact repetition across two arms cannot be coincidence."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    x = float(v)
    if x != x or abs(x) == float("inf") or x in (0.0, 1.0, -1.0) or x == int(x) or _sig(x) < 10:
        return False
    return abs(float(Fraction(x).limit_denominator(1000)) - x) > 1e-12 * max(1.0, abs(x))


def _metrics(d):
    return {k: v for k, v in d.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
            and not any(t in k.lower() for t in _DENY)}


def _scan_json(node, where, out, depth=0):
    if depth > 12:
        return
    if isinstance(node, dict):
        groups = {}
        for k, v in node.items():
            if isinstance(v, dict) and _metrics(v):
                groups.setdefault(frozenset(_metrics(v)), []).append((k, v))
        for schema, arms in groups.items():
            for i in range(len(arms)):
                for j in range(i + 1, len(arms)):
                    (ka, va), (kb, vb) = arms[i], arms[j]
                    ma, mb = _metrics(va), _metrics(vb)
                    if any(ma[m] != mb[m] for m in schema):
                        continue
                    hard = sorted(m for m in schema if _informative(ma[m]))
                    if len(hard) >= 2:
                        out.append("%s: arms '%s' and '%s' are numerically IDENTICAL on all %d metrics "
                                   "(e.g. %s=%r) — verify the manipulation engaged (lab.lever)"
                                   % (where, ka, kb, len(schema), hard[0], ma[hard[0]]))
        for k, v in node.items():
            _scan_json(v, "%s/%s" % (where, k), out, depth + 1)
    elif isinstance(node, list):
        for i, v in enumerate(node[:200]):
            _scan_json(v, "%s[%d]" % (where, i), out, depth + 1)


def _prose_ok(tok):
    if "." not in tok or len(tok.split(".")[1]) < 3:
        return False
    return float(tok) not in (0.0, 1.0, -1.0, 0.5, 100.0)


def _scan_md(text, label, out):
    table = []
    for line in text.splitlines() + [""]:
        s = line.strip()
        if s.startswith("|") and len(s.strip("|").split("|")) >= 2:
            table.append([c.strip() for c in s.strip("|").split("|")])
            continue
        if len(table) >= 4:
            rows, groups = table[2:], {}
            for r in rows:
                if set("".join(r)) <= set("-: |"):
                    continue
                toks = tuple(t for c in r[1:] for t in _NUM.findall(c))
                if not r[0] or len(toks) < 2 or len({t for t in toks if _prose_ok(t)}) < 2:
                    continue
                groups.setdefault(toks, []).append(r[0])
            for toks, labels in groups.items():
                uniq = sorted(set(labels))
                if len(uniq) >= 2:
                    out.append("%s: table rows %s carry IDENTICAL numbers %s — two distinct arms cannot agree "
                               "exactly; verify each arm actually ran" % (label, uniq[:3], list(toks[:4])))
        table = []


def check(paths):
    """paths = staged files. Empty => scan the natural corpus (findings JSON artifacts + findings markdown)."""
    if paths:
        js = [p for p in paths if p.endswith(".json")]
        md = [p for p in paths if p.endswith(".md")]
    else:
        js, md = [], []
        for root, _dirs, files in os.walk(os.path.join(_REPO, "research", "findings")):
            for f in files:
                (js if f.endswith(".json") else md if f.endswith(".md") else []).append(os.path.join(root, f))
    out = []
    for p in js + md:
        full = p if os.path.isabs(p) else os.path.join(_REPO, p)
        try:
            if os.path.getsize(full) > _MAX_BYTES:
                continue
            text = open(full, encoding="utf-8", errors="replace").read()
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            continue
        rel = os.path.relpath(full, _REPO)
        if p.endswith(".json"):
            try:
                doc = json.loads(text)
            except json.JSONDecodeError:
                continue                      # a .json that is not JSON is some other gate's problem
            _scan_json(doc, rel, out)
        else:
            _scan_md(text, rel, out)
    return out[:_MAX_REPORT]


def selftest():
    """Failing direction FIRST: the gate must CATCH a dead arm, then must NOT fire on honest data."""
    # the real gap#4 artifact: `plain_fa` was byte-identical to `test_fixed` because the arm never ran.
    bad_arm = {"align_deep": 0.3118920968955565, "align_mid": 0.8750803696315148}
    live_arm = {"align_deep": 0.3305892010068128, "align_mid": 0.6868002132110415}
    dup_md = ("| arm | acc | delta |\n|---|---|---|\n"
              "| intact | 0.411 | -0.047 |\n| lesion | 0.411 | -0.047 |\n")
    ok_md = ("| arm | acc | delta |\n|---|---|---|\n"
             "| intact | 0.411 | -0.047 |\n| lesion | 0.402 | -0.051 |\n")
    problems = []
    with tempfile.TemporaryDirectory() as d:
        def w(name, obj):
            p = os.path.join(d, name)
            open(p, "w").write(obj if isinstance(obj, str) else json.dumps(obj))
            return p

        must_catch = [
            ("byte-identical JSON arms", w("dead.json", {"r": {"fixed_fa": bad_arm, "kp": dict(bad_arm)}})),
            ("identical markdown arm rows", w("dead.md", dup_md)),
        ]
        for what, p in must_catch:
            if not check([p]):
                problems.append("MISSED %s — the gate cannot fail; it is worthless" % what)

        must_pass = [
            ("distinct arms", w("live.json", {"r": {"fixed_fa": bad_arm, "kp": live_arm}})),
            ("distinct markdown rows", w("live.md", ok_md)),
            ("shared 0.0/1.0 floor+ceiling", w("triv.json", {"r": {"a": {"x": 0.0, "y": 1.0},
                                                                   "b": {"x": 0.0, "y": 1.0}}})),
            ("quantized small-n accuracy tie", w("quant.json", {"r": {"a": {"x": 1 / 3, "y": 11 / 18},
                                                                      "b": {"x": 1 / 3, "y": 11 / 18}}})),
            ("shared config/seed block", w("cfg.json", {"r": {"a": {"seed": 42.5, "lr_config": 0.0033333333333},
                                                              "b": {"seed": 42.5, "lr_config": 0.0033333333333}}})),
            ("ONE shared number (a control block, not proof)",
             w("one.json", {"r": {"a": {"ctl": 0.3118920968955565}, "b": {"ctl": 0.3118920968955565}}})),
        ]
        for what, p in must_pass:
            if check([p]):
                problems.append("FALSE POSITIVE on %s — the gate would cry wolf" % what)
    return problems
