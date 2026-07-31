"""FAILURE CLASS 6 -- wrong-quantity comparison: two correct numbers of DIFFERENT quantities, compared as if
they were the same one. 7 incidents; the row `docs/FAILURE_GATE_MATRIX.md` calls "the one with NO mechanical
guard", because nothing in the repo verified that two compared numbers measure the same thing.

THE EVIDENCE THIS GATE IS BUILT FROM
  * `circ` vs `circ_dW` (2026-07-30) -- the headline was the circular resultant of the weight CHANGE, scored
    against a randset null; an arm measured the same statistic on the FINAL weights, which are dominated by the
    random init. `lr=0` "beating" the learning arm was arithmetically guaranteed. It retracted a validated
    6-seed GO for one cycle before a manual trace restored it. The finding's own words: *"No check in the repo
    verifies that two compared numbers are the same quantity."*
  * `0.588 = 67% of 0.8719` -- a NULL-SUBTRACTED difference (0.6705 - 0.0822) over a RAW, un-subtracted ceiling.
  * the older `sum_finalQ` vs `mean_distance_overall` conflation recorded in CLAUDE.md (~3x apart).

WHAT THIS GATE CHECKS -- deliberately ONE narrow, decidable shape, not the class:
    a comparison operator (`vs` / `versus` / `% of` / `compared to` / `against` / `beats` / `exceeds`)
    separating TWO number-bearing metric names, where one name is the other PLUS ONE qualifier token drawn from
    the delta/raw/final/null family. `circ` against `circ_dW` is that shape by construction: the extra token
    IS the statement that the two are different quantities.

CALIBRATION, MEASURED BEFORE SHIPPING (a gate that cries wolf gets ignored and is worse than none). Run over
all 1841 `research/findings/*.md`, all `docs/**/*.md` and the root boards: **1 hit, and it is the incident
itself** -- the `GAP_CLOSURE_MISSION.md:275` line narrating the retraction, which carries an acknowledgement, so
the gate clears it. Unacknowledged false positives on the existing corpus: **zero** (verified by disabling the
acknowledgement filter: 1 hit, that line). Accepting bare English words as metric names instead of requiring
backticks or snake_case was measured at **22 corpus hits, all noise**; that one restriction is the calibration.

WHAT THIS GATE CANNOT CATCH -- so class 6 is NOT closed:
  * THE FLAGSHIP INCIDENT'S OWN LAYOUT. `circ` sat in a table in section 2 and `circ(dW)` in prose in section 3;
    nothing put them on one line, so no line-scoped rule sees it. This gate catches the *board one-liner*, not
    the finding that produced it.
  * Same NAME, different provenance -- "67% of oracle" is two identically-named `circ` values, one
    null-subtracted and one not. Indistinguishable from text; deciding it needs the artifact.
  * Different-stem conflations (`sum_finalQ` vs `mean_distance_overall`): no shared stem to key on.
  * Anything outside markdown -- code, JSON artifacts, commit messages.
An acknowledgement in the line (or the token `quantity-checked`) suppresses the finding, so the escape is prose
and is abusable. This blocks re-publishing the shape unnoticed; it does not prove like-for-like.
"""
from __future__ import annotations

import glob
import os
import re
import tempfile

NAME = "quantity-mismatch"
CLASS_ID = "6"
BLOCKING = True

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The qualifier family from the recorded incidents: a delta / normalisation / stage marker. A name that carries
# one of these and a name that does not are, by construction, not the same quantity.
QUALIFIERS = frozenset({
    "dw", "dv", "dr", "delta", "deltas", "change", "changed", "diff", "difference",
    "final", "raw", "null", "init", "initial", "subtracted", "cumulative", "cum", "normalised", "normalized",
})

CMP_RE = re.compile(
    r"\bvs\.?\b|\bversus\b|%\s*of\b|\bcompared\s+(?:to|with)\b|\bagainst\b|\bbeats?\b|\bexceeds?\b", re.I)
# A metric name is CODE-LIKE: backticked, or snake_case. A bare English word is never a metric here -- that
# restriction is what took the corpus hit count from 22 (all noise) to 1.
IDENT_RE = re.compile(r"`([^`\n]{2,45})`|\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)\b")
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
ACK_RE = re.compile(
    r"⛔|\bRETRACT\w*|\bWITHDRAWN\b|\bCORRECTION\b|\bWRONG(?:LY)?\b|like-for-like|different\s+quantit\w*"
    r"|not\s+the\s+same\s+quantity|\bmismatch\w*|quantity-checked", re.I)
WINDOW = 110          # chars either side of the operator; also clipped at markdown cell boundaries


def _normalise(name):
    """`circ(dW)` and `circ_dW` are the same written two ways. Strip emphasis, fold the call form."""
    return re.sub(r"\(([A-Za-z0-9_]+)\)$", r"_\1", name.strip().strip("*"))


def _tokens(name):
    return [t for t in re.split(r"[^A-Za-z0-9]+", name.lower()) if t]


def qualifier_between(short, long_):
    """The qualifier token q if tokens(long_) is tokens(short) with exactly one extra q from QUALIFIERS."""
    ts, tl = _tokens(short), _tokens(long_)
    if not ts or len(tl) != len(ts) + 1:
        return None
    for i in range(len(tl)):
        if tl[:i] + tl[i + 1:] == ts and tl[i] in QUALIFIERS:
            return tl[i]
    return None


def _identifiers(segment):
    out = []
    for m in IDENT_RE.finditer(segment):
        v = _normalise(m.group(1) or m.group(2))
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{1,45}", v) and v not in out:
            out.append(v)
    return out


def scan_line(line):
    """[(name_a, name_b, qualifier, snippet)] for every wrong-quantity comparison on this line."""
    if ACK_RE.search(line):
        return []
    found, seen = [], set()
    for m in CMP_RE.finditer(line):
        lo = line.rfind("|", max(0, m.start() - WINDOW), m.start())
        left = line[(lo + 1 if lo >= 0 else max(0, m.start() - WINDOW)):m.start()]
        hi = line.find("|", m.end(), m.end() + WINDOW)
        right = line[m.end():(hi if hi >= 0 else min(len(line), m.end() + WINDOW))]
        if len(NUM_RE.findall(left + right)) < 2:
            continue          # two figures in play: it must be a comparison OF MEASUREMENTS, not of concepts
        for a in _identifiers(left):
            for b in _identifiers(right):
                q = qualifier_between(a, b) or qualifier_between(b, a)
                if q and (a, b) not in seen:
                    seen.add((a, b))
                    found.append((a, b, q, (left + m.group(0) + right).strip()[:150]))
    return found


def _corpus():
    return (sorted(glob.glob(os.path.join(ROOT, "research", "findings", "*.md")))
            + sorted(glob.glob(os.path.join(ROOT, "docs", "**", "*.md"), recursive=True))
            + sorted(glob.glob(os.path.join(ROOT, "*.md"))))


def check(paths):
    # An EMPTY list means "staged mode, nothing of my kind staged" -> nothing to check. Only paths=None means
    # "standalone run, scan the corpus". Without this, the pre-commit driver's --diff-filter=A scoping is undone
    # by this gate's own corpus fallback -- which fired 192 doc-type hits on 2026-04/05 legacy findings the
    # moment the Tier-1 classification gave them frontmatter.
    if paths is not None and len(paths) == 0:
        return []
    targets = [p for p in (paths or []) if p.endswith(".md")]
    if not paths:
        targets = _corpus()
    problems = []
    for p in targets:
        full = p if os.path.isabs(p) else os.path.join(ROOT, p)
        if not os.path.isfile(full):
            continue
        with open(full, errors="ignore") as fh:
            for ln, line in enumerate(fh, 1):
                for a, b, q, snip in scan_line(line):
                    problems.append(
                        "%s:%d compares `%s` against `%s` -- they differ by the '%s' qualifier, so they are "
                        "different quantities (class 6). Recompute both the same way, or say why they are "
                        "comparable on that line (or tag it `quantity-checked`). >>> %s"
                        % (os.path.relpath(full, ROOT), ln, a, b, q, snip))
    return problems


def selftest():
    """FAILING DIRECTION FIRST: build the recorded incident and require the gate to CATCH it."""
    bad = ("# f\n\nSix seeds: `circ_dW` 0.6705 vs `circ` 0.0359 -- the lr0 arm wins.\n"
           "Also `circ_dW` 0.588 = 67% of `circ` 0.8719.\n"
           "And `circ(dW)` mean **0.6705** vs `circ` **0.0359** measured on the FINAL weights.\n")
    # Must NOT fire. Each line is a CALIBRATION CONTROL for one narrowing decision. The `self_cum` and
    # `d_model` lines are REAL corpus lines that the looser prototypes flagged, kept verbatim so that any
    # widening of the identifier rule fails here instead of on the owner's next commit.
    good = ("# f\n\n`lr0_circ_dW` 0.004 vs `btsp_circ_dW` 0.193 -- same statistic, two arms.\n"
            "⛔ RETRACTED: I compared `circ` 0.0359 against `circ_dW` 0.6705, two different quantities.\n"
            "`circ_dW` vs `circ` (no figures quoted here).\n"
            "`circ_dW` 0.6705 overall vs `btsp_circ_dW` 0.1930 in that arm -- same statistic, one subset.\n"
            "`sum_finalQ` 2.57 vs `mean_distance_overall` 0.86 -- different stems, uncatchable.\n"
            "   self_cum 213 vs robust 1157 (5.4x weaker), self-rank 12; not\n"
            "| model | d_model | n_layers | params | ratio vs 83M | data 20.0 |\n"
            # a board megaline: two related names 400 chars apart are NOT a comparison; only the window says so
            + "CYCLE 421 the store reached `circ_dW` 0.6705 on six seeds and the queue was restocked overnight, "
              "then the lane check reported the schedule vs the budget at 3.0 hours, and much later in the same "
              "entry, after the replay pass and the consolidation smoke and two unrelated de-risks were banked, "
              "a separate section notes `circ` 0.0359 for the oracle row.\n")
    problems = []
    with tempfile.TemporaryDirectory() as d:
        pb, pg = os.path.join(d, "bad.md"), os.path.join(d, "good.md")
        for path, text in ((pb, bad), (pg, good)):
            with open(path, "w") as fh:
                fh.write(text)
        hits = check([pb])
        if len(hits) < 3:
            problems.append("DID NOT CATCH the circ/circ_dW incident: %d hit(s), expected 3 "
                            "(`vs`, `%% of`, and the `circ(dW)` call form)" % len(hits))
        elif not all("circ" in h for h in hits):
            problems.append("caught something, but not the circ pair: %r" % hits)
        fps = check([pg])
        if fps:
            problems.append("FALSE POSITIVE on calibrated-clean lines: %r" % fps[:2])
    if qualifier_between("circ", "circ_dW") != "dw":
        problems.append("qualifier_between failed on the canonical pair")
    if qualifier_between("lr0_circ_dW", "btsp_circ_dW") is not None:
        problems.append("qualifier_between fires on a same-quantity arm pair")
    if scan_line("`circ_dW` 0.67 vs `circ` 0.03") == []:
        problems.append("scan_line is unfailable on the minimal positive case")
    return problems
