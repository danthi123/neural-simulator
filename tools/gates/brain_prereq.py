"""brain_prereq (class BP, BLOCK) — a new-core experiment or eval that never asked the record or the biology.

WHY (owner 2026-11-02, after the pivot to the new core in brain/). The new core was started as a clean slate,
outside the project's tooling, and in its first day ran a dozen experiments (reinstatement gain grids, key
variants, a scale ladder) without `tools/before_you_build.sh`, and read weak results on a 0.1B cortex as if they
were mechanism verdicts. The owner had to correct it twice ("you're testing on a very small brain"; "what real
biology requires for a capability to function"), and the lesson was already on file: finding
2026-07-31-why-we-hit-walls-the-missing-companion-process, quoted in CLAUDE.md. The owner's words: "I don't like
that I have to reactively fix your approach to things." A remembered rule did not hold, so it is a gate.

WHAT IT ENFORCES, on newly-added .py files under brain/experiments/ and brain/eval/: the header (first 40 lines)
must carry
  # Capability: <name>     a capability heading in brain/PREREQUISITES.md (case-insensitive substring match)
  # Record: <path>         an existing file: the before_you_build output or the finding/plan this test builds on
and, when that capability's section in PREREQUISITES.md says NOT-READY,
  # Expected-fail: <why>   so a failure is recorded as expected (missing prerequisites), never as a verdict.

WHAT IT CANNOT CATCH: a record that was cited and not read, or a capability label that does not match what the
script actually tests. Those stay judgement.
"""
import os
import re

NAME = "brain-prereq"
CLASS_ID = "BP"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCOPE = ("brain/experiments/", "brain/eval/")


def _in_scope(p):
    p = p.replace("\\", "/")
    return p.endswith(".py") and p.startswith(_SCOPE) and not p.endswith("__init__.py")


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
    head = "\n".join(text.splitlines()[:40])
    cap = re.search(r"^#\s*Capability:\s*(.+)$", head, re.M)
    rec = re.search(r"^#\s*Record:\s*(\S+)", head, re.M)
    exp = re.search(r"^#\s*Expected-fail:\s*\S", head, re.M)
    probs = []
    if prereq_text is None:
        return ["%s — brain/PREREQUISITES.md is missing; no new-core test may land without the prerequisites map"
                % rel]
    if not cap:
        probs.append("%s — no `# Capability: <name>` header naming a section of brain/PREREQUISITES.md" % rel)
    else:
        want = cap.group(1).strip().lower()
        secs = _sections(prereq_text)
        hits = [h for h in secs if want in h]
        if not hits:
            probs.append("%s — capability %r matches no heading in brain/PREREQUISITES.md" % (rel, want))
        elif "not-ready" in secs[hits[0]].lower() and not exp:
            probs.append("%s — capability %r is NOT-READY in brain/PREREQUISITES.md: add `# Expected-fail: <the "
                         "missing prerequisites>` so a failure is not read as a verdict" % (rel, want))
    if not rec:
        probs.append("%s — no `# Record: <path>` header (run tools/before_you_build.sh first; cite its output "
                     "or the finding/plan this builds on)" % rel)
    elif not os.path.exists(os.path.join(root, rec.group(1))):
        probs.append("%s — `# Record:` path %s does not exist" % (rel, rec.group(1)))
    return probs


def check(paths):
    if paths is None:
        import glob
        paths = [os.path.relpath(p, _ROOT) for s in _SCOPE for p in glob.glob(os.path.join(_ROOT, s, "**/*.py"),
                                                                                  recursive=True)]
    cand = [p for p in paths if _in_scope(p)]
    if not cand:
        return []
    pp = os.path.join(_ROOT, "brain", "PREREQUISITES.md")
    prereq = open(pp, errors="ignore").read() if os.path.exists(pp) else None
    problems = []
    for rel in cand:
        try:
            text = open(os.path.join(_ROOT, rel), errors="ignore").read()
        except Exception:
            continue
        problems += check_text(rel, text, prereq)
    return problems


def selftest():
    """FAILING DIRECTION FIRST: missing headers, an unknown capability and a NOT-READY test without an
    expected-fail line MUST be caught; a fully-declared script must pass."""
    bad = []
    prereq = "## Episodic recall\nReadiness: PARTIAL\n## Theory of mind\nReadiness: NOT-READY (no TPJ)\n"
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
        bad.append("FALSE POSITIVE on a fully-declared READY/PARTIAL script")
    if check_text("brain/eval/x.py", "# Capability: theory of mind\n" + rec + "# Expected-fail: no TPJ\n", prereq):
        bad.append("FALSE POSITIVE on a NOT-READY script that declares Expected-fail")
    return bad


if __name__ == "__main__":
    print("selftest:", selftest())
