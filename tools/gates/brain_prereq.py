"""brain_prereq (class BP, BLOCK): a new-core experiment or eval that never asked the record or the biology.

WHY (owner 2026-11-02, after the pivot to the new core in brain/). In its first day the new core ran a dozen
experiments without the record check and read weak results on a 0.1B cortex as mechanism verdicts. The owner had
to correct it twice. A remembered rule did not hold, so it is a check at commit time.

WHAT IT ENFORCES
1. A newly ADDED .py under brain/experiments/ or brain/eval/ carries in its first 40 lines
     # Capability: <name>     a capability heading in brain/PREREQUISITES.md (case-insensitive substring)
     # Record: <path>         an existing file: the before_you_build output, finding or plan this builds on
   and, when that capability's section says NOT-READY,
     # Expected-fail: <why>   so a failure is recorded as expected, never as a verdict.
   Not checked: the files that existed on branch brain-v2 when this gate landed (GRANDFATHERED), and, during a
   merge, files already present in MERGE_HEAD (they were checked on their own branch).
2. When brain/PREREQUISITES.md is staged, every capability section (one with a `Readiness:` line) cites at least
   one source: a catalog entry (`CAT X.nn`, `ROAD`, a sim-catalog/references path), a research/biology binding
   (`BIO/<id>.md` or `research/biology/<id>.md`, which must exist), a Kandel chapter, or a DOI.

WHAT IT CANNOT CATCH: a record that was cited and not read, or a capability label that does not match what the
script tests. Those stay judgement.
"""
import os
import re
import subprocess

NAME = "brain-prereq"
CLASS_ID = "BP"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCOPE = ("brain/experiments/", "brain/eval/")
_PREREQ = "brain/PREREQUISITES.md"
GRANDFATHERED = frozenset({            # on brain-v2 at d1563c039, written before this gate existed
    "brain/eval/reinstate_grid.py", "brain/eval/weekly.py",
    "brain/experiments/key_variants.py", "brain/experiments/reinstate_multi.py",
    "brain/experiments/reinstate_probe.py", "brain/experiments/reinstate_select.py",
    "brain/experiments/scale_ladder.py", "brain/experiments/smoke_chat.py",
})
_SOURCE_RE = re.compile(r"\bCAT\s+[A-Z]{1,2}\.\d+|\bROAD\b|sim-catalog/references/\S+|\bBIO/[\w.-]+\.md"
                        r"|research/biology/[\w.-]+\.md|\bKandel\b[^\n]{0,40}?\bCh\w*\.?\s*\d+|\b10\.\d{4,9}/\S+")
_BIO_RE = re.compile(r"(?:\bBIO/|research/biology/)([\w.-]+\.md)")


def _in_scope(p):
    p = p.replace("\\", "/")
    return (p.endswith(".py") and p.startswith(_SCOPE) and not p.endswith("__init__.py")
            and p not in GRANDFATHERED)


def _git(*args):
    try:
        r = subprocess.run(["git", "-C", _ROOT] + list(args), capture_output=True, text=True, timeout=10)
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def _sections(prereq_text):
    """{lowercased heading: section body} for every markdown heading of level 2 or 3."""
    out, cur, buf = {}, None, []
    for line in prereq_text.splitlines():
        m = re.match(r"^#{2,3}\s+(.*)", line)
        if m:
            if cur is not None:
                out[cur] = "\n".join(buf)
            cur, buf = m.group(1).strip().lower(), []
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        out[cur] = "\n".join(buf)
    return out


def check_text(rel, text, prereq_text, root=_ROOT):
    """Problems with one new-core test script's header."""
    head = "\n".join(text.splitlines()[:40])
    cap = re.search(r"^#\s*Capability:\s*(.+)$", head, re.M)
    rec = re.search(r"^#\s*Record:\s*(\S+)", head, re.M)
    exp = re.search(r"^#\s*Expected-fail:\s*\S", head, re.M)
    if prereq_text is None:
        return ["%s: brain/PREREQUISITES.md is missing; no new-core test may land without it" % rel]
    probs = []
    if not cap:
        probs.append("%s: no `# Capability: <name>` header naming a section of brain/PREREQUISITES.md" % rel)
    else:
        want = cap.group(1).strip().lower()
        secs = _sections(prereq_text)
        hits = [h for h in secs if want in h]
        if not hits:
            probs.append("%s: capability %r matches no heading in brain/PREREQUISITES.md" % (rel, want))
        elif "not-ready" in secs[hits[0]].lower() and not exp:
            probs.append("%s: capability %r is NOT-READY: add `# Expected-fail: <missing prerequisites>` so a "
                         "failure is not read as a verdict" % (rel, want))
    if not rec:
        probs.append("%s: no `# Record: <path>` header (run tools/before_you_build.sh first; cite its output or "
                     "the finding/plan this builds on)" % rel)
    elif not os.path.exists(os.path.join(root, rec.group(1))):
        probs.append("%s: `# Record:` path %s does not exist" % (rel, rec.group(1)))
    return probs


def check_prereq_doc(prereq_text, root=_ROOT):
    """Every capability section must cite a real source; cited biology bindings must exist."""
    probs = []
    for head, body in _sections(prereq_text).items():
        if not re.search(r"^\s*Readiness:", body, re.M):
            continue
        if not _SOURCE_RE.search(body):
            probs.append("%s: section %r cites no source (CAT/ROAD/sim-catalog path, BIO/<id>.md, Kandel chapter "
                         "or DOI)" % (_PREREQ, head))
        for bio in _BIO_RE.findall(body):
            if not os.path.exists(os.path.join(root, "research", "biology", bio)):
                probs.append("%s: section %r cites research/biology/%s, which does not exist" % (_PREREQ, head, bio))
    return probs


def _carried_by_merge(rel):
    mh = (_git("rev-parse", "-q", "--verify", "MERGE_HEAD") or "").strip()
    return bool(mh) and _git("cat-file", "-e", "%s:%s" % (mh, rel)) is not None


def check(paths):
    if not paths:
        return []
    paths = [p.replace("\\", "/") for p in paths]
    cand = [p for p in paths if _in_scope(p) and not _carried_by_merge(p)]
    staged = set((_git("diff", "--cached", "--name-only", "--diff-filter=ACM") or "").split())
    doc_staged = _PREREQ in paths or _PREREQ in staged
    if not cand and not doc_staged:
        return []
    pp = os.path.join(_ROOT, _PREREQ)
    prereq = open(pp, errors="ignore").read() if os.path.exists(pp) else None
    problems = []
    for rel in cand:
        try:
            text = open(os.path.join(_ROOT, rel), errors="ignore").read()
        except OSError:
            continue
        problems += check_text(rel, text, prereq)
    if doc_staged and prereq is not None:
        problems += check_prereq_doc(prereq)
    return problems


def selftest():
    """FAILING DIRECTION FIRST, then the cases that must pass."""
    bad = []
    prereq = ("## Episodic recall\nBiology:\n- CAT D.13 pattern completion\nReadiness: PARTIAL\n"
              "## Theory of mind\nBiology:\n- Kandel Ch 59\nReadiness: NOT-READY (no TPJ)\n")
    rec = "# Record: tools/gates/brain_prereq.py\n"
    if not check_text("brain/eval/x.py", "import os\n", prereq):
        bad.append("did NOT catch a script with no headers")
    if not check_text("brain/eval/x.py", "# Capability: mind reading\n" + rec, prereq):
        bad.append("did NOT catch an unknown capability")
    if not check_text("brain/eval/x.py", "# Capability: theory of mind\n" + rec, prereq):
        bad.append("did NOT catch a NOT-READY capability without Expected-fail")
    if not check_text("brain/eval/x.py", "# Capability: episodic recall\n# Record: no/such/file.md\n", prereq):
        bad.append("did NOT catch a Record path that does not exist")
    if not check_text("brain/eval/x.py", "# Capability: episodic recall\n" + rec, None):
        bad.append("did NOT catch a missing PREREQUISITES.md")
    if check_text("brain/eval/x.py", "# Capability: episodic recall\n" + rec, prereq):
        bad.append("FALSE POSITIVE on a fully-declared PARTIAL script")
    if check_text("brain/eval/x.py", "# Capability: theory of mind\n" + rec + "# Expected-fail: no TPJ\n", prereq):
        bad.append("FALSE POSITIVE on a NOT-READY script that declares Expected-fail")
    if not check_prereq_doc("## Sleep\nBiology: trust me\nReadiness: NOT-READY\n"):
        bad.append("did NOT catch a capability section with no source")
    if not check_prereq_doc("## Sleep\n- BIO/no-such-binding-xyz.md\nReadiness: NOT-READY\n"):
        bad.append("did NOT catch a cited biology binding that does not exist")
    if check_prereq_doc(prereq):
        bad.append("FALSE POSITIVE on sections citing CAT and Kandel")
    if check_prereq_doc("## Notes\nfree text, no readiness line\n"):
        bad.append("FALSE POSITIVE on a non-capability section")
    if _in_scope("brain/experiments/scale_ladder.py"):
        bad.append("grandfathered brain-v2 file is in scope (would block the brain-v2 merge)")
    if not _in_scope("brain/experiments/new_probe.py"):
        bad.append("a new experiment is out of scope")
    return bad


if __name__ == "__main__":
    print("selftest:", selftest())
