"""CLASS D — DOCUMENT TYPE AND PLACEMENT: information recorded where it does not belong.

THE EVIDENCE (measured 2026-07-31, owner-raised). The plan/finding boundary has not blurred, it has collapsed:

    docs/plans/*.md ............ 287 files, of which 229 (80%) REPORT A RESULT (GO / NEGATIVE / measured / 6-seed)
    research/findings/*.md ..... 1841 files, of which 161 have a plan/design/spec-shaped TITLE
    findings mentioning "consolidation" .......... 367
    findings mentioning "btsp" ................... 96

A plan is FORWARD-looking (what we intend, and why). A finding is BACKWARD-looking (what we measured, with the
artifact). When a plan asserts a result, that result is invisible to every check built for findings -- it carries
no status, no mechanism, no artifact citation, and no claim check. 80% of plans are in that position, which means
a large fraction of this project's asserted results sit outside the entire gate system.

WHAT THIS GATE ENFORCES (narrow, and only on documents that declare frontmatter, so the 2100+ legacy files do
not fire -- legacy is audited on next touch, per the Tier-1/Tier-2 audit plan):

  1. `type:` must be declared and must MATCH the directory. finding -> research/findings/, plan -> docs/plans/.
  2. A `plan` must NOT assert a measured result. If it has one, the result belongs in a finding that the plan
     cites -- otherwise the assertion escapes claim_check, status, mechanism-conflict and single-seed entirely.
  3. A `finding` must cite at least one artifact path. A finding with no artifact is a plan, an opinion, or a
     summary -- all of which have their own homes.

WHAT IT CANNOT CATCH: whether a document is REDUNDANT with 20 others on the same mechanism. Duplication is a
judgement about contribution, not a property of one file. That is handled from the other end, by the mechanism
registry (research/biology/<id>.md) naming ONE current_finding and forcing every other live claim on that
mechanism to resolve -- see tools/biology_check.check_mechanism_status.
"""
from __future__ import annotations

import glob
import os
import re
import tempfile

NAME = "doc-type"
CLASS_ID = "D"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TYPE_DIR = {"finding": os.path.join("research", "findings"),
            "plan": os.path.join("docs", "plans"),
            "biology": os.path.join("research", "biology")}

# A RESULT ASSERTION, not a mention. Anchored to verdict-shaped statements so a plan that *discusses* a prior GO
# is not flagged -- only one that ASSERTS an outcome of its own.
RESULT_RE = re.compile(
    r"^[#>*\s|-]*\**\s*(?:RESULT|VERDICT|OUTCOME)\b"                 # a result/verdict heading or line
    r"|\b(?:6-seed|6 seeds|n=6)\b[^.\n]{0,40}\b(?:GO|PASS|NEGATIVE|REFUTED)\b"
    r"|\bis a (?:GO|NO-GO)\b|\bverdict:\s*\S", re.I | re.M)
ARTIFACT_RE = re.compile(r"[\w.\-*?\[\]]+(?:/[\w.\-*?\[\]]+)+\.(?:json|jsonl)")
FM_RE = re.compile(r"^type:\s*([a-z-]+)\s*$", re.M)


def _frontmatter(text):
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    return text[3:end] if end > 0 else None


def _check_one(path, text=None):
    problems = []
    rel = os.path.relpath(path, _ROOT) if os.path.isabs(path) else path
    # Skills + rules are NOT research docs: they use a `name:`/`description:` (or `paths:`) frontmatter schema,
    # never a research `type:`. Exempt them so the research-doc-type gate does not force `type:` on a SKILL.md.
    relslash = rel.replace("\\", "/")
    if ".claude/skills/" in relslash or "/.claude/rules/" in relslash or relslash.startswith(".claude/rules/") or ".hermes/skills/" in relslash:
        return problems
    try:
        text = text if text is not None else open(path, errors="ignore").read()
    except OSError:
        return problems
    fm = _frontmatter(text)
    if fm is None:
        return problems                                   # legacy: audited on next touch, not flagged now
    m = FM_RE.search(fm)
    if not m:
        problems.append("%s: frontmatter present but no `type:` — declare finding | plan | biology" % rel)
        return problems
    t = m.group(1)
    want = TYPE_DIR.get(t)
    if want and want not in rel.replace("\\", "/"):
        problems.append("%s: declared `type: %s` but lives outside %s/ — a document's type must match its home"
                        % (rel, t, want))
    body = text[text.find("\n---", 3) + 4:] if _frontmatter(text) else text
    if t == "plan":
        hit = RESULT_RE.search(body)
        if hit:
            problems.append(
                "%s: a PLAN asserts a measured result (%r). Move the result into a finding and cite it — an "
                "assertion here escapes claim_check, status, mechanism-conflict and single-seed entirely. "
                "(80%% of plans currently do this.)" % (rel, hit.group(0).strip()[:48]))
    if t == "finding" and not ARTIFACT_RE.search(body):
        problems.append("%s: a FINDING cites no artifact path. A finding reports a MEASUREMENT; with no artifact "
                        "it is a plan, an opinion or a summary, each of which has its own home." % rel)
    return problems


def check(paths):
    # An EMPTY list means "staged mode, nothing of my kind staged" -> nothing to check. Only paths=None means
    # "standalone run, scan the corpus". Without this, the pre-commit driver's --diff-filter=A scoping is undone
    # by this gate's own corpus fallback -- which fired 192 doc-type hits on 2026-04/05 legacy findings the
    # moment the Tier-1 classification gave them frontmatter.
    if paths is not None and len(paths) == 0:
        return []
    problems = []
    targets = [p for p in (paths or []) if p.endswith(".md")]
    if not targets:
        # paths PROVIDED but none of mine -> nothing to check. Only paths=None (standalone) scans the corpus.
        if paths is not None:
            return []
        # no staged set at all: scan frontmatter-bearing docs
        targets = [p for p in glob.glob(os.path.join(_ROOT, "research/findings/*.md"))
                   + glob.glob(os.path.join(_ROOT, "docs/plans/*.md"))
                   + glob.glob(os.path.join(_ROOT, "research/biology/*.md"))]
    for p in targets:
        full = p if os.path.isabs(p) else os.path.join(_ROOT, p)
        if os.path.exists(full):
            problems += _check_one(full)
    return problems


def selftest():
    """FAILING DIRECTION FIRST: build documents the gate MUST catch, and fail if it does not."""
    bad = []
    with tempfile.TemporaryDirectory() as d:
        # 1. a plan asserting a result
        p1 = os.path.join(d, "plan.md")
        open(p1, "w").write("---\ntype: plan\n---\n\n## RESULT\n\nIt is a GO.\n")
        if not _check_one(p1):
            bad.append("did NOT catch a plan asserting a result")
        # 2. a finding with no artifact
        p2 = os.path.join(d, "finding.md")
        open(p2, "w").write("---\ntype: finding\n---\n\nWe observed 0.4567 improvement.\n")
        if not any("cites no artifact" in x for x in _check_one(p2)):
            bad.append("did NOT catch a finding with no artifact citation")
        # 3. frontmatter with no type
        p3 = os.path.join(d, "notype.md")
        open(p3, "w").write("---\nstatus: live\n---\n\nbody\n")
        if not any("no `type:`" in x for x in _check_one(p3)):
            bad.append("did NOT catch missing type declaration")
        # 4. NEGATIVE CONTROL — a legacy doc with no frontmatter must NOT be flagged (else 2100+ files fire and
        #    the gate gets disabled, which is worse than no gate).
        p4 = os.path.join(d, "legacy.md")
        open(p4, "w").write("# Legacy finding\n\nIt is a GO at 6 seeds.\n")
        if _check_one(p4):
            bad.append("FALSE POSITIVE: flagged a legacy no-frontmatter document")
        # 5. NEGATIVE CONTROL — a SKILL.md (name/description frontmatter, no research `type:`) must NOT be flagged.
        sk = os.path.join(d, ".claude", "skills", "x", "SKILL.md")
        os.makedirs(os.path.dirname(sk), exist_ok=True)
        open(sk, "w").write("---\nname: x\ndescription: a skill\n---\n\nbody\n")
        if _check_one(sk):
            bad.append("FALSE POSITIVE: flagged a .claude/skills SKILL.md as a research doc")
    return bad
