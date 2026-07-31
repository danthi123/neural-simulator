"""Failure class 8 -- STALE POINTER / UNMAINTAINED REGISTRY (5 recorded incidents).

THE EVIDENCE. `docs/RETRACTED.md` holds ONE data row while 21-25 findings carry a retraction marker in their
own FILENAME, so `tools/check_docs.py` W1 -- defined purely against that registry -- can only ever fire on one
path. It reports "OK" over a corpus it does not cover. The roadmap's gap#4 ledger row described a framing that
was superseded the SAME DAY it was written. The common shape: a summary doc keeps pointing at a result that has
since died, and nothing derived from the result's own state ever contradicts the pointer.

WHAT THIS GATE DOES. Derived, not maintained: for every citation of a finding inside a governed summary doc
(CLAUDE.md, GAP_CLOSURE_MISSION.md, ROADMAP.md, docs/plans/*.md), it reads the CITED FILE'S OWN declared
frontmatter `status:`. If that status is `retracted` or `superseded` and the citing bullet/row carries no
retraction marker, that is a problem. It also runs the reverse direction: when a STAGED finding declares
retracted/superseded, every governed doc is rescanned for unmarked citations of it -- the "you retracted it and
left the board pointing at it" case, which is how this class actually happens.

DECLARED STATUS ONLY. `tools/finding_status.py` measured its keyword heuristic at roughly 1-in-6 precision
(a doc that PERFORMS a retraction reads identically to one that IS retracted), so keywords are not used here at
all. Consequence: coverage equals declaration. At the time of writing, 4 of 413 resolvable citations point at a
finding that declares anything, so this gate is nearly blind -- and it SAYS SO (one INFO line, below) rather
than printing a clean tick over an uncovered corpus, which is precisely the class-8 failure mode.

WHAT IT CANNOT CATCH:
  * a finding that IS dead but never declared it (undeclared => invisible here, reported only as coverage INFO);
  * truncated / prose citations that carry no `<name>.md` token, and citations by paraphrase or by claim;
  * a pointer that is stale in SUBSTANCE while the cited doc is still `live` (the gap#4 ledger row -- staleness
    of meaning is a semantic problem no status field encodes);
  * stale pointers into non-finding targets (plans, runners, artifacts, commits).
Non-blocking by design: it reports, it does not stop a commit, because its input (frontmatter) is opt-in.
"""
from __future__ import annotations

import glob
import io
import os
import re
import tempfile

NAME = "stale-pointer"
CLASS_ID = "8"
BLOCKING = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STALE_STATUSES = ("retracted", "superseded")
MARKERS = ("⛔", "retract", "void", "supersed", "withdrawn")
FIXED_DOCS = ("CLAUDE.md", "GAP_CLOSURE_MISSION.md", "ROADMAP.md")
CITE_RE = re.compile(r"(?:[A-Za-z0-9._\-]+/)*([12]\d{3}-\d{2}-\d{2}-[A-Za-z0-9._\-]+?)\.md")
DECL_RE = re.compile(r"^status:\s*([a-z-]+)\s*$", re.M)
# The coverage INFO line fires only when the gate is blind AT SCALE (>=20 citations, <5% declared). Below that
# it stays silent: a warning that prints on every commit is the cry-wolf failure, not a safeguard.
INFO_MIN_CITES, INFO_MAX_DECL_FRAC = 20, 0.05


def _governed(root):
    out = [d for d in FIXED_DOCS if os.path.isfile(os.path.join(root, d))]
    out += sorted(os.path.relpath(p, root) for p in glob.glob(os.path.join(root, "docs", "plans", "*.md")))
    return out


def _declared_status(root, rel):
    """The cited file's OWN declared status, or None. Frontmatter must be the first thing in the file."""
    path = os.path.join(root, rel)
    if not os.path.isfile(path):
        return None
    with io.open(path, encoding="utf-8", errors="ignore") as fh:
        head = "".join(next(fh, "") for _ in range(15))
    if not head.startswith("---"):
        return None
    m = DECL_RE.search(head.split("\n---", 1)[0])
    return m.group(1).strip() if m else None


def _is_block_start(line):
    t = line.lstrip()
    return (not t) or t.startswith(("- ", "* ", "+ ", "|", "#", ">")) or bool(re.match(r"^\d+[.)]\s", t))


def _block(lines, i):
    """The bullet / table row / paragraph containing line i -- a marker anywhere in it counts as marking it."""
    s = i
    while s > 0 and not _is_block_start(lines[s]):
        s -= 1
    e = i
    while e + 1 < len(lines) and lines[e + 1].strip() and not _is_block_start(lines[e + 1]):
        e += 1
    return "\n".join(lines[s:e + 1]).lower()


def _scan(root, docs, only=None):
    """-> (problems, n_citations, n_declared). `only`: restrict to these finding rel-paths."""
    problems, n_cite, n_decl, cache = [], 0, 0, {}
    for rel in docs:
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            continue
        lines = io.open(path, encoding="utf-8", errors="ignore").read().split("\n")
        infence = False
        for i, text in enumerate(lines):
            if re.match(r"^[ \t]*```", text):
                infence = not infence
                continue
            if infence:
                continue
            for m in CITE_RE.finditer(text):
                target = os.path.join("research", "findings", m.group(1) + ".md")
                if not os.path.isfile(os.path.join(root, target)):
                    continue                      # plans / renamed / truncated: not certain, so not reported
                if target not in cache:
                    cache[target] = _declared_status(root, target)
                st = cache[target]
                if only is not None and target not in only:
                    continue
                n_cite += 1
                if st is None:
                    continue
                n_decl += 1
                if st in STALE_STATUSES and not any(k in _block(lines, i) for k in MARKERS):
                    problems.append("%s:%d cites %s (declared %s) with no retraction marker on the line"
                                    % (rel, i + 1, os.path.basename(target), st))
    return problems, n_cite, n_decl


def check(paths, root=ROOT):
    paths = [os.path.relpath(os.path.abspath(p), root) if os.path.isabs(p) else p for p in (paths or [])]
    docs = _governed(root)
    staged_docs = [p for p in paths if p in docs]
    staged_stale = {p for p in paths
                    if p.startswith("research/findings/") and _declared_status(root, p) in STALE_STATUSES}

    problems, n_cite, n_decl = [], 0, 0
    if not paths:                                  # no staging context: scan the natural corpus
        problems, n_cite, n_decl = _scan(root, docs)
    else:
        if staged_docs:
            problems, n_cite, n_decl = _scan(root, staged_docs)
        if staged_stale:                           # a newly-dead finding invalidates pointers ANYWHERE
            extra, _, _ = _scan(root, docs, only=staged_stale)
            problems += [p for p in extra if p not in problems]
    if n_cite >= INFO_MIN_CITES and n_decl < INFO_MAX_DECL_FRAC * n_cite:
        problems.append("INFO (not a violation): only %d of %d cited findings declare a frontmatter status, so "
                        "this gate can check %d%% of the citations. Declare `status:` to widen it."
                        % (n_decl, n_cite, round(100.0 * n_decl / n_cite)))
    return problems


def _fixture(d):
    os.makedirs(os.path.join(d, "research", "findings"))
    os.makedirs(os.path.join(d, "docs", "plans"))

    def w(rel, text):
        with io.open(os.path.join(d, rel), "w", encoding="utf-8") as fh:
            fh.write(text)
    w("research/findings/2026-01-02-fx-dead.md", "---\nstatus: retracted\n---\n\nbody\n")
    w("research/findings/2026-01-03-fx-old.md", "---\nstatus: superseded\n---\n\nbody\n")
    w("research/findings/2026-01-04-fx-live.md", "---\nstatus: live\n---\n\nbody\n")
    # No frontmatter, but the body screams retraction -- the keyword heuristic's ~1-in-6 trap. Must stay silent.
    w("research/findings/2026-01-05-fx-nodecl.md", "# ⛔⛔ RETRACTED VOID WITHDRAWN\n")
    w("ROADMAP.md", "\n".join([
        "- the result stands, see research/findings/2026-01-02-fx-dead.md",          # 1  MUST catch
        "- ⛔ withdrawn: research/findings/2026-01-02-fx-dead.md",                    # 2  marked -> silent
        "- current: research/findings/2026-01-04-fx-live.md",                        # 3  live   -> silent
        "- see research/findings/2026-01-05-fx-nodecl.md",                           # 4  undeclared -> silent
        "```",
        "- fenced research/findings/2026-01-02-fx-dead.md",                          # 6  code   -> silent
        "```", ""]))
    w("GAP_CLOSURE_MISSION.md", "- board still points at 2026-01-02-fx-dead.md\n")   # 1  MUST catch
    w("docs/plans/p.md", "| lane | research/findings/2026-01-03-fx-old.md | ok |\n")  # 1 MUST catch
    return d


def selftest():
    """Prove the FAILING direction first, then that the silent cases stay silent."""
    bad = []
    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        probs = check([], root=d)
        must = {"ROADMAP.md:1": "retracted citation with no marker",
                "GAP_CLOSURE_MISSION.md:1": "retracted citation on the board",
                "docs/plans/p.md:1": "superseded citation in a plan table row"}
        for anchor, what in must.items():
            if not any(p.startswith(anchor + " ") for p in probs):
                bad.append("MISSED %s (%s); got %r" % (anchor, what, probs))
        for anchor, why in (("ROADMAP.md:2", "marker on the line"), ("ROADMAP.md:3", "status live"),
                            ("ROADMAP.md:4", "status undeclared -- keywords must not be used"),
                            ("ROADMAP.md:6", "inside a fenced block")):
            if any(p.startswith(anchor + " ") for p in probs):
                bad.append("FALSE POSITIVE at %s (%s); got %r" % (anchor, why, probs))
        # staged a newly-retracted finding -> its pointers must be found even though no doc was staged
        rev = check(["research/findings/2026-01-02-fx-dead.md"], root=d)
        if not any(p.startswith("ROADMAP.md:2 ") for p in rev) \
                or not any(p.startswith("GAP_CLOSURE_MISSION.md:1 ") for p in rev):
            bad.append("reverse direction MISSED: staging a retracted finding found %r" % rev)
        if any(p.startswith("docs/plans/p.md") for p in rev):
            bad.append("reverse direction leaked an unrelated finding's citation: %r" % rev)
        # staging one doc must not drag in the others
        one = check(["ROADMAP.md"], root=d)
        if any(p.startswith(("GAP_CLOSURE_MISSION.md", "docs/plans/")) for p in one):
            bad.append("staged-path scoping broken: %r" % one)
        if any(p.startswith("INFO") for p in probs):
            bad.append("INFO line fired on a %d-citation fixture (threshold is %d)" % (5, INFO_MIN_CITES))
    return bad
