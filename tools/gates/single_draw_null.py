"""CLASS SN — a SINGLE random draw used as a pass/fail NULL control. BLOCKING, for NEWLY ADDED runners.

EVIDENCE (2026-09-23). `_curiosity_metacog_lowconfidence_coupling_derisk.py` pre-registered a "shuffle control"
G4 as ONE permutation per seed (`default_rng(90000+seed).permutation(7)`) and required |rho_shuffle| < 0.5.
The response was strictly monotone in its input, so the shuffled rho was fully determined by which permutation
the seed happened to draw — not by the mechanism. For n=7, P(|rho| >= 0.5) under a random permutation is 0.267,
so ~27% of seeds failed "the control" by construction. The 6-seed run read NOT-GO 4/6 on exactly that basis
(seeds 43/44), and the seed-42 smoke read GO only because its draw was benign. A null control must be a
DISTRIBUTION (>= 1000 permutations, observed statistic reported as a percentile / p-value), not one draw.

WHAT IT CHECKS. Runner files under research/runners/ that are ADDED in the staged set (git diff --cached
--diff-filter=A). A file is flagged when it (1) draws a permutation / shuffle (`.permutation(`, `.shuffle(`),
(2) names a pass/fail check after it (a dict key or variable containing "shuffle" or "perm" next to a comparison),
and (3) carries NO marker of a null distribution: `n_perm`, `N_PERM`, `null_dist`, `percentile`, `n_shuffles`,
`N_SHUF`, `perm_null`. An explicit in-file waiver `# single-draw-null: ok <reason>` exempts it.

SCOPE, and why only ADDED files: the same heuristic matches ~219 existing runners (measured 2026-09-23). Most of
those are finished arcs; blocking every future edit to them would get the gate bypassed with --no-verify, which
disables every other gate too. Scoping to new runners stops the class from recurring without a legacy flood.

CANNOT CATCH: a single draw in an EXISTING runner; a single draw whose check is named without "shuffle"/"perm";
a file that mentions `percentile` for an unrelated reason while still gating on one draw; a null distribution that
exists but is too small (e.g. 20 permutations). Those are judgement — `verify-go` skeptics own them.
"""
from __future__ import annotations

import os
import re
import subprocess

NAME = "single-draw-null"
CLASS_ID = "SN"
BLOCKING = True

_DRAW = re.compile(r"\.(permutation|shuffle)\(")
_CHECK = re.compile(r"(?i)[\"']?[A-Za-z0-9_]*(shuffle|perm)[A-Za-z0-9_]*[\"']?\s*(:|=)[^\n]*(<|>)")
_DIST = re.compile(r"n_perm|N_PERM|null_dist|percentile|n_shuffles|N_SHUF|perm_null")
_WAIVER = re.compile(r"#\s*single-draw-null:\s*ok\s+\S")


def violations_in_text(text: str) -> list[str]:
    if _WAIVER.search(text):
        return []
    if not _DRAW.search(text):
        return []
    if not _CHECK.search(text):
        return []
    if _DIST.search(text):
        return []
    return ["draws ONE permutation/shuffle and gates pass/fail on it, with no null DISTRIBUTION "
            "(>=1000 perms, report percentile). A single draw is a random number, not a control."]


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
            problems.append("%s: %s\n        FIX: build a permutation null distribution, or waive with "
                            "`# single-draw-null: ok <reason>` if the draw is not a control." % (p, v))
    return problems


def selftest():
    """FAILING DIRECTION FIRST: the exact 2026-09-23 pattern must fire."""
    bad = []
    defect = ("rng = np.random.default_rng(90000 + seed)\nperm = rng.permutation(7)\n"
              "checks = {\"G4_shuffle_collapses\": abs(rho_shuf) < 0.5}\n")
    if not violations_in_text(defect):
        bad.append("does not catch a single seed-drawn permutation used as a pass/fail control")
    fixed = defect + "N_PERM = 10000\nnull = [rho(rng.permutation(n)) for _ in range(N_PERM)]\npercentile = 0\n"
    if violations_in_text(fixed):
        bad.append("false positive on a runner that builds a permutation null distribution")
    if violations_in_text(defect + "# single-draw-null: ok the draw is a stimulus order, not a control\n"):
        bad.append("ignores an explicit waiver")
    if violations_in_text("x = rng.permutation(5)\nprint(x)\n"):
        bad.append("false positive on a permutation that gates nothing")
    return bad
