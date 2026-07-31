"""Failure class 8 -- STALE POINTER / UNMAINTAINED REGISTRY (5 recorded incidents).

THE EVIDENCE. `docs/RETRACTED.md` holds ONE data row while 21-25 findings carry a retraction marker in their own
FILENAME, so `tools/check_docs.py` W1 -- defined purely against that registry -- can only ever fire on one path.
It reports "OK" over a corpus it does not cover. The roadmap's gap#4 ledger row described a framing that was
superseded the SAME DAY it was written. The shape is always the same: a summary doc keeps pointing at a result
that has since died, and nothing derived from the result's OWN state ever contradicts the pointer.

WHAT THIS GATE DOES -- derived, not maintained. For every citation of a finding inside a governed summary doc
(CLAUDE.md, GAP_CLOSURE_MISSION.md, ROADMAP.md, docs/plans/*.md) it reads the CITED FILE'S own declared
frontmatter `status:`. If that status is `retracted` or `superseded` and the citing bullet / table row carries
no retraction marker, that is a problem. It also runs the reverse direction: when a STAGED finding declares
retracted/superseded, every governed doc is rescanned for unmarked citations of it -- the "you retracted it and
left the board pointing at it" case, which is how this class actually happens.

DECLARED STATUS ONLY. `tools/finding_status.py` measured its keyword heuristic at about 1-in-6 precision (a doc
that PERFORMS a retraction reads identically to one that IS retracted), so keywords are not used here at all.
Consequence: coverage equals declaration -- and when the gate is blind AT SCALE it SAYS SO (one INFO line)
instead of printing a clean tick over an uncovered corpus, which is precisely the class-8 failure mode it
exists to prevent. MEASURED 2026-07-31 (`_scan(ROOT, _governed(ROOT))` over 290 governed docs): 434 of 447
resolvable citations now declare a status (97.1%), so the INFO line correctly stays silent -- the "4 of ~407 /
nearly blind" figure this docstring carried at build time has been overtaken by the backfill.

IT NAMES THE SUCCESSOR (2026-07-31). Detecting the stale pointer was only half the repair. `status: superseded`
says a document is dead and says NOTHING about what replaced it, so the fixer this gate summons then traced
descendants BY HAND -- 4 of 5 markers in one fanout pass, unaudited, which is a hand-maintained registry
rebuilt per citation and therefore the same class-8 failure one level down. So the message now QUOTES the cited
finding's own `superseded_by:` frontmatter, verbatim, and RESOLVES it: a successor that does not exist is marked
UNRESOLVED, and a successor that is ITSELF retracted/superseded is flagged rather than recommended. Where the
field is ABSENT the message says so in words -- silence there is what made the reconstruction manual, and an
omitted clause is indistinguishable from a gate that forgot to look.

    ROADMAP.md:12 cites <dead>.md (declared superseded) with no retraction marker on the line
        — cite instead: 2026-07-25-the-rerun.md
        — successor declared but NOT usable, so this still needs a hand fix: X.md [⚠ itself declared superseded]
        — NO successor declared: <dead>.md carries no `superseded_by:`, so what replaces it has to be
          reconstructed by hand here and at every other citation

HONEST SCOPE OF THE SUCCESSOR CLAUSE: it fires on ZERO real citations today, because the corpus currently has
zero unmarked citations of a declared-dead finding (0 violations, measured, unchanged by this edit) AND zero of
its 23 declared retracted/superseded findings carry a `superseded_by:` at all. It is exercised on real corpus
files only through a scratch harness, and in the selftest fixtures. What is verified is the MECHANISM; what is
NOT established is any effect on a live pointer, because there is no live pointer to have an effect on.

WHAT IT CANNOT CATCH: a finding that IS dead but never declared it (invisible; only counted in the INFO line);
truncated or prose citations carrying no `<name>.md` token, and citation by paraphrase or by claim; a pointer
stale in SUBSTANCE while the cited doc is still `live` (the gap#4 ledger row -- staleness of meaning is not
encoded in any status field); stale pointers into non-finding targets (plans, runners, artifacts, commits); and,
for the successor field specifically, whether the named successor is the RIGHT one -- resolvability is checked,
truthfulness is not, and a missing `superseded_by:` is reported but never itself blocks (it would flag every
already-dead finding in the corpus, and a gate that flags hundreds of legacy files gets disabled).
Non-blocking by design: its input (frontmatter) is opt-in, so absence of a problem is not evidence of health.
"""
from __future__ import annotations

import glob
import io
import os
import re
import sys
import tempfile

NAME = "stale-pointer"
CLASS_ID = "8"
BLOCKING = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ONE PARSER, deliberately shared with the CLI rather than duplicated here. This gate used to carry its own copy
# of the frontmatter reader; adding a SECOND field to a second copy is how a producer and its consumer end up
# disagreeing about a format (logged 2026-07-31: "the provenance door writes `.prov.json` while its gate looked
# for `.cmd.json`" -- an integration seam invisible from inside either half). If this import breaks, the registry
# reports the gate as BROKEN and blocks, which is the correct outcome: a status parser that silently diverges is
# worse than one that is loudly absent.
if ROOT not in sys.path:                          # so the gate does not depend on the caller's cwd
    sys.path.insert(0, ROOT)
from tools.finding_status import (                # noqa: E402
    declared_status as _fs_declared_status,
    successor_parts,
)

STALE_STATUSES = ("retracted", "superseded")
MARKERS = ("⛔", "retract", "void", "supersed", "withdrawn")
FIXED_DOCS = ("CLAUDE.md", "GAP_CLOSURE_MISSION.md", "ROADMAP.md")
CITE_RE = re.compile(r"(?:[A-Za-z0-9._\-]+/)*([12]\d{3}-\d{2}-\d{2}-[A-Za-z0-9._\-]+?)\.md")
BLOCK_START = re.compile(r"^[ \t]*(?:[-*+]\s|[|#>]|\d+[.)]\s|$)")   # bullet / table row / heading / blank
# The coverage INFO line fires only when the gate is blind AT SCALE (>=20 citations, <5% declared). Below that
# it stays silent: a warning that prints on every commit is the cry-wolf failure, not a safeguard.
INFO_MIN_CITES, INFO_MAX_DECL_FRAC = 20, 0.05


def _governed(root):
    docs = [d for d in FIXED_DOCS if os.path.isfile(os.path.join(root, d))]
    return docs + sorted(os.path.relpath(p, root) for p in glob.glob(os.path.join(root, "docs/plans/*.md")))


def _declared_status(root, rel):
    """The cited file's OWN declared status, or None. Frontmatter must open the file."""
    if not os.path.isfile(os.path.join(root, rel)):
        return None
    return _fs_declared_status(os.path.join(root, rel))


def _successor_note(root, rel):
    """The clause telling the fixer WHAT TO CITE INSTEAD -- or, explicitly, that nobody said.

    Appended to every violation, never emitted alone: it does not create violations, it finishes the ones the
    gate already found. That is the calibration argument, and it is measured, not assumed -- 0 of the 23
    findings that declare retracted/superseded carry a `superseded_by:` today, so promoting "no successor" to
    its own violation row would flag all 23 legacy files on the first commit that touched them, and a gate that
    flags legacy files by the dozen gets switched off. Violation count on the real corpus is 0 before this
    change and 0 after it (measured); only the text of a violation changed.
    """
    parts = successor_parts(os.path.join(root, rel), root=root)
    if not parts:
        return ("— NO successor declared: %s carries no `superseded_by:`, so what replaces it has to be "
                "reconstructed by hand here and at every other citation" % os.path.basename(rel))
    rendered = "; ".join(r for r, _ in parts)
    if any(ok for _, ok in parts):
        return "— cite instead: %s" % rendered
    # "cite instead: <a dangling or equally-dead pointer>" would be an instruction to make the problem again.
    return "— successor declared but NOT usable, so this still needs a hand fix: %s" % rendered


def _block(lines, i):
    """The bullet / row / paragraph holding line i -- a marker anywhere IN IT counts, but not a neighbour's."""
    s = i
    while s > 0 and not BLOCK_START.match(lines[s]):
        s -= 1
    e = i
    while e + 1 < len(lines) and lines[e + 1].strip() and not BLOCK_START.match(lines[e + 1]):
        e += 1
    return "\n".join(lines[s:e + 1]).lower()


def _scan(root, docs, only=None):
    """-> (problems, n_citations, n_declared). `only`: restrict to these finding rel-paths."""
    problems, n_cite, n_decl, cache, notes = [], 0, 0, {}, {}
    for rel in docs:
        if not os.path.isfile(os.path.join(root, rel)):
            continue
        lines = io.open(os.path.join(root, rel), encoding="utf-8", errors="ignore").read().split("\n")
        infence = False
        for i, text in enumerate(lines):
            if re.match(r"^[ \t]*```", text):
                infence = not infence
            if infence:
                continue
            for m in CITE_RE.finditer(text):
                target = "research/findings/%s.md" % m.group(1)
                if not os.path.isfile(os.path.join(root, target)):
                    continue                       # plan / renamed / truncated: not certain, so not reported
                if target not in cache:
                    cache[target] = _declared_status(root, target)
                if only is not None and target not in only:
                    continue
                n_cite += 1
                if cache[target] is None:
                    continue
                n_decl += 1
                if cache[target] in STALE_STATUSES and not any(k in _block(lines, i) for k in MARKERS):
                    if target not in notes:
                        notes[target] = _successor_note(root, target)
                    problems.append("%s:%d cites %s (declared %s) with no retraction marker on the line %s"
                                    % (rel, i + 1, os.path.basename(target), cache[target], notes[target]))
    return problems, n_cite, n_decl


def check(paths, root=ROOT):
    paths = [os.path.relpath(os.path.abspath(p), root) if os.path.isabs(p) else p for p in (paths or [])]
    docs = _governed(root)
    staged_stale = {p for p in paths
                    if p.startswith("research/findings/") and _declared_status(root, p) in STALE_STATUSES}
    problems, n_cite, n_decl = [], 0, 0
    if not paths:                                  # no staging context: scan the natural corpus
        problems, n_cite, n_decl = _scan(root, docs)
    else:
        staged_docs = [p for p in paths if p in docs]
        if staged_docs:
            problems, n_cite, n_decl = _scan(root, staged_docs)
        if staged_stale:                           # a newly-dead finding invalidates pointers ANYWHERE
            problems += [p for p in _scan(root, docs, only=staged_stale)[0] if p not in problems]
    if n_cite >= INFO_MIN_CITES and n_decl < INFO_MAX_DECL_FRAC * n_cite:
        problems.append("INFO (not a violation): only %d of %d cited findings declare a frontmatter status, so "
                        "this gate can check %d%% of the citations. Declare `status:` to widen it."
                        % (n_decl, n_cite, round(100.0 * n_decl / n_cite)))
    return problems


def _fixture(d):
    os.makedirs(os.path.join(d, "research/findings"))
    os.makedirs(os.path.join(d, "docs/plans"))
    for rel, text in (
            # NO `superseded_by:` -- the absence case, which must be SPOKEN, not skipped.
            ("research/findings/2026-01-02-fx-dead.md", "---\nstatus: retracted\n---\n\nbody\n"),
            ("research/findings/2026-01-03-fx-old.md", "---\nstatus: superseded\n"
                                                       "superseded_by: 2026-01-09-fx-heir.md\n---\n\nbody\n"),
            ("research/findings/2026-01-04-fx-live.md", "---\nstatus: live\n---\n\nbody\n"),
            # No frontmatter, body screaming retraction: the keyword heuristic's ~1-in-6 trap. Must stay silent.
            ("research/findings/2026-01-05-fx-nodecl.md", "# ⛔⛔ RETRACTED VOID WITHDRAWN\n"),
            # `status:` in the BODY is prose about someone else's status, not a declaration.
            ("research/findings/2026-01-06-fx-body.md", "# on the registry\n\nstatus: retracted\n"),
            # successor pointer that dangles -- a stale pointer INSIDE the repair instruction
            ("research/findings/2026-01-07-fx-dangle.md", "---\nstatus: superseded\n"
                                                          "superseded_by: 2026-99-99-gone.md\n---\n"),
            # successor that is ITSELF dead: naming it without a warning just moves the reader to another corpse
            ("research/findings/2026-01-08-fx-chain.md", "---\nstatus: superseded\n"
                                                         "superseded_by: 2026-01-03-fx-old.md\n---\n"),
            ("research/findings/2026-01-09-fx-heir.md", "---\nstatus: live\n---\n\nbody\n"),
            ("research/findings/2026-01-10-fx-heir2.md", "---\nstatus: live\n---\n\nbody\n"),
            # `superseded_by:` in the BODY is prose about the convention, not a declaration (mirrors #5)
            ("research/findings/2026-01-11-fx-succbody.md", "---\nstatus: superseded\n---\n\n"
                                                            "superseded_by: 2026-01-09-fx-heir.md\n"),
            ("research/findings/2026-01-12-fx-list.md", "---\nstatus: superseded\nsuperseded_by:\n"
                                                        "  - 2026-01-09-fx-heir.md\n"
                                                        "  - 2026-01-10-fx-heir2.md\n---\n"),
            ("ROADMAP.md", "- the result stands, see research/findings/2026-01-02-fx-dead.md\n"   # 1 MUST catch
                           "- ⛔ withdrawn: research/findings/2026-01-02-fx-dead.md\n"             # 2 marked
                           "- current: research/findings/2026-01-04-fx-live.md\n"                 # 3 live
                           "- see research/findings/2026-01-05-fx-nodecl.md\n"                    # 4 undeclared
                           "- see research/findings/2026-01-06-fx-body.md\n"                      # 5 body-only
                           "```\n- fenced research/findings/2026-01-02-fx-dead.md\n```\n"         # 7 fenced
                           "- see research/findings/2026-01-07-fx-dangle.md\n"                    # 9 dangling
                           "- see research/findings/2026-01-08-fx-chain.md\n"                     # 10 dead heir
                           "- see research/findings/2026-01-11-fx-succbody.md\n"                  # 11 body prose
                           "- see research/findings/2026-01-12-fx-list.md\n"),                    # 12 block list
            ("GAP_CLOSURE_MISSION.md", "- board still points at 2026-01-02-fx-dead.md\n"),        # 1 MUST catch
            ("docs/plans/p.md", "| lane | research/findings/2026-01-03-fx-old.md | ok |\n")):     # 1 MUST catch
        io.open(os.path.join(d, rel), "w", encoding="utf-8").write(text)
    return d


def _info_fixture(d, k):
    """k citations of findings that declare NOTHING -- the coverage reporter's only input."""
    os.makedirs(os.path.join(d, "research/findings"))
    names = ["2026-02-%02d-fx-undeclared.md" % (j + 1) for j in range(k)]
    for n in names:
        io.open(os.path.join(d, "research/findings", n), "w", encoding="utf-8").write("# no frontmatter\n")
    io.open(os.path.join(d, "ROADMAP.md"), "w", encoding="utf-8").write(
        "".join("- see research/findings/%s\n" % n for n in names))
    return d


def selftest():
    """Prove the FAILING direction first, then that every silent case stays silent."""
    bad = []
    with tempfile.TemporaryDirectory() as d:
        _fixture(d)
        probs = check([], root=d)
        hit = lambda rows, anchor: any(p.startswith(anchor + " ") for p in rows)      # noqa: E731
        msg = lambda rows, anchor: next((p for p in rows if p.startswith(anchor + " ")), "")   # noqa: E731
        for anchor, what in (("ROADMAP.md:1", "retracted citation, no marker"),
                             ("GAP_CLOSURE_MISSION.md:1", "retracted citation on the board"),
                             ("docs/plans/p.md:1", "superseded citation in a plan table row"),
                             ("ROADMAP.md:9", "superseded citation, successor dangles"),
                             ("ROADMAP.md:10", "superseded citation, successor itself dead"),
                             ("ROADMAP.md:11", "superseded citation, `superseded_by:` only in the body"),
                             ("ROADMAP.md:12", "superseded citation, block-list successors")):
            if not hit(probs, anchor):
                bad.append("MISSED %s (%s); got %r" % (anchor, what, probs))

        # ---- THE SUCCESSOR CLAUSE, failing direction: each of these is a way to leave the fixer guessing ----
        # A gate that detects the stale pointer but says nothing about the replacement sends someone to trace
        # descendants by hand -- which is the incident (4 of 5 markers, unaudited) this clause exists to end.
        for anchor, needle, what in (
                ("docs/plans/p.md:1", "2026-01-09-fx-heir.md", "did NOT quote the declared successor"),
                ("ROADMAP.md:1", "NO successor declared", "stayed SILENT that no successor is declared"),
                ("ROADMAP.md:9", "UNRESOLVED", "recommended a successor whose file does not exist, unmarked"),
                ("ROADMAP.md:10", "itself declared superseded", "recommended a successor that is itself dead"),
                ("ROADMAP.md:11", "NO successor declared", "read `superseded_by:` out of the BODY as prose"),
                ("ROADMAP.md:12", "2026-01-09-fx-heir.md", "dropped the first entry of a block-list successor"),
                ("ROADMAP.md:12", "2026-01-10-fx-heir2.md", "dropped the second entry of a block-list successor")):
            if needle not in msg(probs, anchor):
                bad.append("%s at %s: %r" % (what, anchor, msg(probs, anchor)))
        if "2026-01-09-fx-heir" in msg(probs, "ROADMAP.md:11"):
            bad.append("quoted a BODY `superseded_by:` as if declared: %r" % msg(probs, "ROADMAP.md:11"))
        if "NO successor" in msg(probs, "docs/plans/p.md:1"):
            bad.append("claimed NO successor while one IS declared: %r" % msg(probs, "docs/plans/p.md:1"))
        # "cite instead: <dangling>" / "cite instead: <also dead>" would instruct the fixer to recreate the
        # defect. Where NOTHING usable is declared the clause must not read as an instruction.
        for anchor, why in (("ROADMAP.md:9", "successor dangles"), ("ROADMAP.md:10", "successor itself dead")):
            if "cite instead" in msg(probs, anchor):
                bad.append("told the fixer to CITE an unusable successor (%s): %r" % (why, msg(probs, anchor)))
            if "NOT usable" not in msg(probs, anchor):
                bad.append("did not say the successor is unusable (%s): %r" % (why, msg(probs, anchor)))
        if "NOT usable" in msg(probs, "ROADMAP.md:12"):
            bad.append("called two live successors unusable: %r" % msg(probs, "ROADMAP.md:12"))

        for anchor, why in (("ROADMAP.md:2", "marker is on the line"), ("ROADMAP.md:3", "status live"),
                            ("ROADMAP.md:4", "undeclared -- keywords must NOT be used"),
                            ("ROADMAP.md:5", "`status:` in the body is prose, not a declaration"),
                            ("ROADMAP.md:7", "inside a fenced block")):
            if hit(probs, anchor):
                bad.append("FALSE POSITIVE at %s (%s); got %r" % (anchor, why, probs))
        if any(p.startswith("INFO") for p in probs):
            bad.append("INFO fired on a 5-citation fixture (threshold %d): %r" % (INFO_MIN_CITES, probs))
        # reverse direction: staging a newly-retracted finding must find its pointers, and ONLY its pointers
        rev = check(["research/findings/2026-01-02-fx-dead.md"], root=d)
        if not (hit(rev, "ROADMAP.md:1") and hit(rev, "GAP_CLOSURE_MISSION.md:1")):
            bad.append("reverse direction MISSED: staging a retracted finding gave %r" % rev)
        if any(p.startswith("docs/plans/") for p in rev):
            bad.append("reverse direction leaked an unrelated finding's citation: %r" % rev)
        one = check(["ROADMAP.md"], root=d)
        if any(p.startswith(("GAP_CLOSURE_MISSION.md", "docs/plans/")) for p in one):
            bad.append("staged-path scoping broken, unstaged docs reported: %r" % one)
    # the coverage reporter, BOTH directions: silent below the threshold, and it must actually fire above it
    with tempfile.TemporaryDirectory() as d2:
        quiet = check([], root=_info_fixture(d2, INFO_MIN_CITES - 1))
        if quiet:
            bad.append("INFO fired below the %d-citation threshold (cry-wolf): %r" % (INFO_MIN_CITES, quiet))
    with tempfile.TemporaryDirectory() as d3:
        loud = check([], root=_info_fixture(d3, INFO_MIN_CITES))
        if len(loud) != 1 or not loud[0].startswith("INFO"):
            bad.append("coverage reporter DEAD: %d undeclared citations gave %r" % (INFO_MIN_CITES, loud))
    return bad
