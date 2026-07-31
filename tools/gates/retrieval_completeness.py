"""CLASS R — THE RECORD'S OWN RETRIEVAL LAYER CANNOT SEE PART OF THE RECORD.

THE DEFECT, measured 2026-07-31. Every consumer of the findings corpus globbed it FLAT:

    glob.glob('research/findings/*.md')      -> 1845 files
    glob.glob('research/findings/**/*.md')   -> 1887 files

**42 findings were invisible to the record's own index** — 24 `_*_scoping.md`, 6 production-reviewer
verdicts, 2 iteration plans — because they sat one directory down in `research/findings/raw/`. They were
absent from the RAG index (`tools/rag/build_llamaindex_full.py`), from `research/runners/local_corpus.py`,
from `tools/queue_add.sh`'s prior-run check, and from `tools/before_you_build.sh` — the mandated FIRST MOVE
before any lever against any defect.

The sharpest instance: `before_you_build.sh` step 2 is literally "IS THERE A RESEARCH GATE / SCOPE DOC
ALREADY?" and searched `*scope*.md`. **24 of the 42 files it could not see are named `_*_scoping.md`.** The
check built to find prior scoping work was structurally blind to most of the scoping work.

WHY THIS IS ITS OWN FAILURE CLASS. A document that the corpus query cannot return is a document that gets
re-derived, and re-derivation is this project's most expensive recurring failure (~94 GPU-hours on a NO-GO
banked a week earlier). Every other check here asks whether a claim is TRUE; this one asks whether the record
can be READ AT ALL. A false claim gets caught downstream. An unreadable one is silently absent, and absence
looks exactly like "we never tried this."

WHAT THIS GATE ENFORCES. Any file that globs `research/findings` for `.md` must do so RECURSIVELY:
  * Python — the pattern must contain `**` AND the call must pass `recursive=True` (without it Python treats
    `**` as a single `*` and the fix silently does nothing — the trap that makes this worth a gate rather
    than a code review).
  * Shell — use `find`, or `grep -r --include='*.md'`; a literal `findings/*.md` glob is flagged.

WHAT IT CANNOT CATCH: a consumer that reads the corpus through some other route entirely (a hardcoded file
list, a database). It checks the globs that exist, not the ones nobody wrote.

DELIBERATELY NOT ENFORCED: that findings must stay flat. Sharding closed months into `findings/archive/YYYY-MM/`
is a live proposal, and it is SAFE precisely when this gate holds — which is why the gate is the precondition
for that move rather than an obstacle to it.
"""
from __future__ import annotations

import os
import re
import tempfile

NAME = "retrieval-completeness"
CLASS_ID = "R"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Files that read the findings corpus. Kept explicit rather than repo-wide: a gate that scans everything
# flags every finding that merely QUOTES a glob in prose, which is how a gate earns its way to being ignored.
WATCHED = [
    "tools/rag/build_llamaindex_full.py",
    "tools/rag/rag_search.py",
    "tools/before_you_build.sh",
    "tools/queue_add.sh",
    "research/runners/local_corpus.py",
]

# A flat glob over the findings directory: `research/findings/*.md`, or the os.path.join spelling.
FLAT_PY_RE = re.compile(r"""["']research[/\\]findings[/\\]\*\.md["']"""
                        r"""|join\([^)]*["']findings["']\s*,\s*["']\*\.md["']""")
FLAT_SH_RE = re.compile(r"""(?<!--include=')research/findings/\*[\w.*-]*\.md""")
# The Python trap: `**` in the pattern does nothing unless the call passes recursive=True.
# Matches BOTH spellings — the literal path `findings/**/*.md` and the os.path.join form `..., "**", "*.md"`.
# (The first version of this regex required `.md` to follow `**` immediately and so missed `**/*.md`, the
# exact form actually used. Its own selftest caught that, which is why the selftest must fail first.)
GLOBSTAR_RE = re.compile(r"""findings[/\\]\*\*|["']\*\*["']""")
RECURSIVE_KW_RE = re.compile(r"glob\.glob\([^)]*recursive\s*=\s*True|glob\.iglob\([^)]*recursive\s*=\s*True")


def _strip_comments(text, shell):
    if shell:
        return re.sub(r"^\s*#.*$", "", text, flags=re.M)
    text = re.sub(r'"""(?:.|\n)*?"""', "", text)
    return re.sub(r"^\s*#.*$", "", text, flags=re.M)


def _check_one(path, rel=None):
    rel = (rel or os.path.relpath(path, _ROOT)).replace("\\", "/")
    try:
        raw = open(path, errors="ignore").read()
    except OSError:
        return []
    shell = rel.endswith(".sh")
    code = _strip_comments(raw, shell)
    problems = []
    flat = FLAT_SH_RE.search(code) if shell else FLAT_PY_RE.search(code)
    if flat:
        problems.append(
            "%s: globs the findings corpus FLAT (%r). 42 findings live one directory down in "
            "research/findings/raw/ and are invisible to a flat glob — including 24 `_*_scoping.md`. Use "
            "`**/*.md` with recursive=True (python) or `find` / `grep -r --include='*.md'` (shell)."
            % (rel, flat.group(0)[:60]))
    if not shell and GLOBSTAR_RE.search(code) and "glob.glob" in code and not RECURSIVE_KW_RE.search(code):
        problems.append(
            "%s: uses a `**` findings pattern but never passes recursive=True. Python treats `**` as a "
            "single `*` without it, so the pattern LOOKS fixed and matches exactly what the flat glob did — "
            "a silent no-op is worse than the original bug." % rel)
    return problems


def check(paths):
    # Unlike the content gates this one is CHEAP and CLOSED (5 named files), so it runs on every invocation
    # including staged mode: the whole point is that a regression here is invisible in its own output.
    problems = []
    for rel in WATCHED:
        full = os.path.join(_ROOT, rel)
        if os.path.exists(full):
            problems += _check_one(full, rel)
    return problems


def selftest():
    """FAILING DIRECTION FIRST: build the two real regressions and fail if the gate misses either."""
    bad = []
    with tempfile.TemporaryDirectory() as d:
        # 1. THE ORIGINAL DEFECT — a flat python glob.
        p1 = os.path.join(d, "flat.py")
        open(p1, "w").write("import glob\npaths = glob.glob('research/findings/*.md')\n")
        if not _check_one(p1, "tools/flat.py"):
            bad.append("did NOT catch a FLAT python glob over research/findings")
        # 2. THE SILENT NO-OP — `**` written without recursive=True. This is the one a code review passes.
        p2 = os.path.join(d, "noop.py")
        open(p2, "w").write("import glob\npaths = glob.glob('research/findings/**/*.md')\n")
        if not any("recursive=True" in x for x in _check_one(p2, "tools/noop.py")):
            bad.append("did NOT catch `**` used WITHOUT recursive=True (the silent no-op)")
        # 3. THE FLAT SHELL GLOB.
        p3 = os.path.join(d, "flat.sh")
        open(p3, "w").write("ls -t research/findings/*.md | head -3\n")
        if not _check_one(p3, "tools/flat.sh"):
            bad.append("did NOT catch a FLAT shell glob over research/findings")
        # 4. NEGATIVE CONTROL — the CORRECT python form must pass, else the gate is unsatisfiable.
        p4 = os.path.join(d, "ok.py")
        open(p4, "w").write("import glob\npaths = glob.glob('research/findings/**/*.md', recursive=True)\n")
        if _check_one(p4, "tools/ok.py"):
            bad.append("FALSE POSITIVE: flagged the correct recursive python form")
        # 5. NEGATIVE CONTROL — the correct shell forms must pass.
        p5 = os.path.join(d, "ok.sh")
        open(p5, "w").write("find research/findings -name '*.md'\n"
                            "grep -rl --include='*.md' -- \"$R\" research/findings/\n")
        if _check_one(p5, "tools/ok.sh"):
            bad.append("FALSE POSITIVE: flagged the correct shell forms (find / grep -r --include)")
        # 6. NEGATIVE CONTROL — prose in a comment must not fire, or every file documenting the fix trips it.
        p6 = os.path.join(d, "prose.py")
        open(p6, "w").write("# we used to call glob.glob('research/findings/*.md') here\nx = 1\n")
        if _check_one(p6, "tools/prose.py"):
            bad.append("FALSE POSITIVE: flagged a flat glob quoted only in a comment")
    return bad
