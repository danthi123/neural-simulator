"""CLASS VN — a Verdict that can only print GO or UNDEFINED, never NO-GO. BLOCKING, for NEWLY ADDED runners.

EVIDENCE (2026-09-24, A10 reward-value afferent, runner at f3fa99c4a). Every outcome criterion ((A) OFF identity,
(B) contra > confirm, (C) lesion differential < 1e-6) was registered with `Verdict.require(...)` AND fed into
`decide(go=...)`. A `require` that is false makes the verdict UNDEFINED, so a false criterion could never reach
`decide` as a NO-GO: the runner printed UNDEFINED for a (C) that was measured (0.10767) and false. Its first
finding reported "(C) UNDEFINED" for a criterion that had failed. The review of 58c400ff6 caught it; the class was
named and left out of research/FAILURE_LOG.md (review of 7d5c2743d).

WHAT IT CHECKS (AST, per function of an ADDED research/runners/*.py): each `<v>.decide(go=<expr>)` /
`<v>.decide(<expr>)` call. The expression's variable names are collected (a bare name, like `go`, is expanded to
the names in its last assignment above the call, e.g. `go = bool(a and b)`; names in call position such as
`bool`/`all` are ignored). It is flagged when
  (1) the expression is a constant True, or
  (2) every one of those names is also passed as the MEASURED value of a `.require(...)` in the same function
      (second positional argument or `measured=`), so any false criterion is already a failed precondition.
An in-file waiver `# verdict-can-read-nogo: ok <reason>` exempts the file (for an instrument check whose GO really is
only its preconditions).

SCOPE: ADDED files only, like SN and FM: finished arcs are not re-litigated, and a legacy flood would get the hook
bypassed. CANNOT CATCH: criteria required through a DIFFERENT variable name or through an expression the gate does
not resolve (attributes, subscripts, helper calls); a GO computed in one function and decided in another. Those are
judgement; `verify-go` skeptics and the review own them.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess

NAME = "verdict-can-read-nogo"
CLASS_ID = "VN"
BLOCKING = True

_WAIVER = re.compile(r"#\s*verdict-can-read-nogo:\s*ok\s+\S")


def _load_names(node):
    """Names read by `node`, excluding names in call position (bool, all, any, float...)."""
    called = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            called.add(id(n.func))
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            and id(n) not in called}


def _assignments(func):
    """name -> [(lineno, value)] for simple `name = value` assignments in `func`."""
    out = {}
    for n in ast.walk(func):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            out.setdefault(n.targets[0].id, []).append((n.lineno, n.value))
    return out


def _reaching(assigns, name, lineno):
    """The value of the last `name = ...` above `lineno` (the A10 runner re-assigned `go` AFTER decide)."""
    prior = [(ln, v) for ln, v in assigns.get(name, []) if ln < lineno]
    return max(prior, key=lambda t: t[0])[1] if prior else None


def _go_expr(call):
    for kw in call.keywords:
        if kw.arg == "go":
            return kw.value
    return call.args[0] if call.args else None


def _measured_expr(call):
    for kw in call.keywords:
        if kw.arg == "measured":
            return kw.value
    return call.args[1] if len(call.args) > 1 else None


def violations_in_text(text: str) -> list[str]:
    if _WAIVER.search(text):
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out = []
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))] + [tree]
    for func in funcs:
        body_calls = [n for n in ast.walk(func) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        decides = [c for c in body_calls if c.func.attr == "decide"]
        if not decides:
            continue
        required = set()
        for c in body_calls:
            if c.func.attr == "require":
                m = _measured_expr(c)
                if isinstance(m, ast.Name):
                    required.add(m.id)
        assigns = _assignments(func)
        for c in decides:
            expr = _go_expr(c)
            if expr is None:
                continue
            if isinstance(expr, ast.Name):
                expr = _reaching(assigns, expr.id, c.lineno) or expr
            if isinstance(expr, ast.Constant) and expr.value is True:
                out.append("line %d: decide(go=True) -- a constant GO can never read NO-GO" % c.lineno)
                continue
            names = _load_names(expr)
            if names and names <= required:
                out.append("line %d: every name in decide()'s GO (%s) is also a Verdict.require precondition, so a "
                           "false criterion reads UNDEFINED, never NO-GO" % (c.lineno, ", ".join(sorted(names))))
    return list(dict.fromkeys(out))   # the module scope also walks into functions: report each call once


def _added_runner_paths():
    try:
        out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=A"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    return [p for p in out.split() if p.startswith("research/runners/") and p.endswith(".py")]


def check(paths=None):
    added = set(_added_runner_paths())
    targets = [p for p in (paths or sorted(added)) if p in added]
    problems = []
    for p in targets:
        if not os.path.isfile(p):
            continue
        try:
            text = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for v in violations_in_text(text):
            problems.append("%s: %s\n        FIX: keep only real preconditions in require(); pass the criteria to "
                            "decide(go=...) so a false one reads NO-GO (or waive with "
                            "`# verdict-can-read-nogo: ok <reason>`)." % (p, v))
    return problems


def selftest():
    """FAILING DIRECTION FIRST: the A10 f3fa99c4a pattern must fire."""
    bad = []
    defect = ("def main(v):\n"
              "    go_a = x == 1\n    go_c = abs(d) < 1e-6\n"
              "    v.require('(A) OFF identity', go_a, expect=True)\n"
              "    v.require('(C) lesion collapses', go_c, expect=True)\n"
              "    go = bool(go_a and go_c)\n"
              "    return v.decide(go=go)\n")
    if not violations_in_text(defect):
        bad.append("does not catch criteria that are all required AND decided (the A10 f3fa99c4a pattern)")
    if not violations_in_text(defect.replace("    return v.decide(go=go)\n",
                                             "    d = v.decide(go=go)\n    go = bool(d['go'])\n    return d\n")):
        bad.append("misses the pattern when `go` is re-assigned after decide() (the exact f3fa99c4a shape)")
    fixed = ("def main(v):\n"
             "    v.require('arm built', built, expect=True)\n"
             "    go_a = x == 1\n    go_c = abs(d) < 1e-6\n"
             "    go = bool(go_a and go_c)\n"
             "    return v.decide(go=go)\n")
    if violations_in_text(fixed):
        bad.append("false positive on a runner whose criteria go only to decide()")
    if not violations_in_text("def f(v):\n    v.require('x', a, expect=True)\n    return v.decide(go=True)\n"):
        bad.append("does not catch a constant decide(go=True)")
    if violations_in_text(defect + "# verdict-can-read-nogo: ok instrument check whose GO is its preconditions\n"):
        bad.append("ignores an explicit waiver")
    return bad
