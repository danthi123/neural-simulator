"""CLASS BAR — a pre-registered bar nobody checked could be passed (or failed).

EVIDENCE (2026-10-06). Two bars cost about two days each in one week. T1's engagement bar (a) asked for an LHb rate
ratio above 2 over a window in which no brain can exceed 1.25. Stage-1 C4's clause "scrambled-map r < 0.20 on all 6
seeds" is failed by a brain that follows its learned values exactly about 56% of the time. Both were a bar a PERFECT
brain fails, found only after the run. The mirror defect is a bar a NULL brain passes.

THE GATE. A commit that ADDS or MODIFIES a markdown file whose basename contains `prereg` or the word `amendment`
(`AMENDMENT-A0.1.md`, `x-AMENDMENT-1.md`; not `amendments`; case-insensitive; this repo's pre-registrations are named
`*-PREREG.md` / `*-PREREGISTRATION.md`) BLOCKS unless the file carries a line
    bar-sanity: <path to a committed JSON record from tools/bar_sanity.py>        every such line must be OK, or
    bar-sanity: N/A <reason, >= 20 characters>                                    (no bar: a process note, a recount)
The record is read from the INDEX (it must be committed with, or before, the file) and must:
  * parse, and carry the fields tools/bar_sanity.py writes;
  * read OK when its verdict is RE-DERIVED from its own pass_rate / kinds / thresholds (a hand-edited verdict fails);
  * hold thresholds no looser than 0.95 / 0.05 unless it gives an override reason (>= 20 characters);
  * have n >= 200 draws per generator;
  * not be STALE: when its spec or any of its scorer files is in the index with a different sha256 than the one
    recorded, the scorer moved after the measurement and the record is void. (A file absent from the index is not
    judged: a record may be made in a checkout the spec's scorer lives in and merged later.)
A line `bar-sanity-before: <path>` or any other prose naming a record is ignored, so an amendment can cite the
UNREACHABLE record of the bar it replaces without it counting.

SCOPE. Only files git reports ADDED / MODIFIED / RENAMED in the commit being made are checked; every existing file is
grandfathered. The registry passes only added files, so (like CLASS PRA) this gate reads `git diff --cached` itself.
A clean `git merge` is not checked (pre-merge-commit runs before MERGE_HEAD exists; each side's commits were judged
when made); a conflicted merge checks only a file that differs from EVERY parent.

WHAT IT CANNOT CATCH. A spec whose perfect brain was written to pass and whose null brain was written to fail: the
record proves the bar is neither impossible nor free against the brains the author chose, not that the choice was fair.
The spec is reviewed like the bar. It also cannot see a bar added in a file whose name has neither word; and an
`N/A` reason is checked for length, not truth.
"""
from __future__ import annotations

import json
import os
import re

NAME = "bar_sanity_line"
CLASS_ID = "BAR"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TARGET_RE = re.compile(r"^(?:[^/]*prereg[^/]*|(?:[^/]*[-_.])?amendment(?![a-z])[^/]*)\.md$", re.I)
_SPECS = (":(icase)*prereg*.md", ":(icase)*amendment*.md")
_LINE_RE = re.compile(r"^[ \t>*+-]*(?:\*\*)?bar-sanity:(?:\*\*)?[ \t]*(.*?)[ \t]*$", re.I)
PERFECT_MIN, NULL_MAX, MIN_REASON, MIN_N = 0.95, 0.05, 20, 200
_REQUIRED = ("spec", "spec_sha256", "scorer_files", "n", "kinds", "pass_rate", "thresholds", "verdict")


def _verdict(rec):
    """Re-derive the verdict from the record's own numbers (same rule as tools/bar_sanity.verdict_from)."""
    kinds, rate, th = rec["kinds"], rec["pass_rate"], rec["thresholds"]
    perf = [g for g, k in kinds.items() if k == "perfect"]
    nul = [g for g, k in kinds.items() if k == "null"]
    if not perf or not nul or any(g not in rate for g in perf + nul):
        return "INVALID"
    if any(rate[g] < th["perfect_min"] for g in perf):
        return "UNREACHABLE"
    if any(rate[g] > th["null_max"] for g in nul):
        return "TOOTHLESS"
    return "OK"


def judge_record(path, rec, index_sha):
    """Pure. Problems with record `rec` (already parsed) cited as `path`. index_sha(relpath) -> sha256 of that file in
    the index, or None when it is not there."""
    out = []
    miss = [k for k in _REQUIRED if k not in rec]
    if miss:
        return ["%s: record lacks field(s) %s -- write it with tools/bar_sanity.py" % (path, ", ".join(miss))]
    try:
        derived = _verdict(rec)
        th = rec["thresholds"]
        loosened = float(th["perfect_min"]) < PERFECT_MIN or float(th["null_max"]) > NULL_MAX
        n = int(rec["n"])
    except (KeyError, TypeError, ValueError) as e:
        return ["%s: record is malformed (%s)" % (path, e)]
    if derived != "OK":
        out.append("%s: re-derived verdict is %s, not OK (the record says %r). A bar a perfect brain cannot pass, or a "
                   "null brain passes, is not registered: change the bar or the instrument first."
                   % (path, derived, rec.get("verdict")))
    elif rec.get("verdict") != "OK":
        out.append("%s: record says %r but its numbers read OK -- regenerate it with tools/bar_sanity.py"
                   % (path, rec.get("verdict")))
    if loosened and len(str(th.get("override_reason") or "").strip()) < MIN_REASON:
        out.append("%s: thresholds looser than perfect >= %.2f / null <= %.2f with no override_reason (>= %d chars)"
                   % (path, PERFECT_MIN, NULL_MAX, MIN_REASON))
    if n < MIN_N:
        out.append("%s: only n=%d draws per generator (need >= %d)" % (path, n, MIN_N))
    stale = []
    if isinstance(rec.get("spec"), str):
        have = index_sha(rec["spec"])
        if have is not None and have != rec["spec_sha256"]:
            stale.append(rec["spec"])
    for f, sha in (rec.get("scorer_files") or {}).items():
        have = index_sha(f)
        if have is not None and have != sha:
            stale.append(f)
    if stale:
        out.append("%s: STALE -- %s changed after the record was made (sha256 differs). Re-run tools/bar_sanity.py."
                   % (path, ", ".join(sorted(set(stale)))))
    return out


def judge_file(path, text, read_blob):
    """Pure. Problems for one prereg/amendment file. read_blob(relpath) -> bytes of that path in the index, or None."""
    lines = [m.group(1) for m in (_LINE_RE.match(ln) for ln in text.splitlines()) if m]
    if not lines:
        return ["%s: no `bar-sanity:` line. Run `python tools/bar_sanity.py <spec> --out <record.json>` BEFORE "
                "registering the bar, commit the record, and add `bar-sanity: <record path>` (or `bar-sanity: N/A "
                "<reason >= %d chars>` when the file registers no bar)." % (path, MIN_REASON)]
    out = []
    for val in lines:
        if re.match(r"^n/?a\b", val, re.I):
            reason = re.sub(r"^n/?a\b[\s:;,.\-—]*", "", val, flags=re.I).strip()
            if len(reason) < MIN_REASON:
                out.append("%s: `bar-sanity: N/A` needs a reason of at least %d characters (got %d)"
                           % (path, MIN_REASON, len(reason)))
            continue
        rec_path = val.strip("`<>\"' ")
        if not rec_path:
            out.append("%s: empty `bar-sanity:` line" % path)
            continue
        raw = read_blob(rec_path)
        if raw is None:
            out.append("%s: bar-sanity record %s is not in the index (commit the record with, or before, this file)"
                       % (path, rec_path))
            continue
        try:
            rec = json.loads(raw.decode("utf-8", "replace"))
        except ValueError as e:
            out.append("%s: bar-sanity record %s is not JSON (%s)" % (path, rec_path, e))
            continue
        import hashlib

        def index_sha(rel, _rb=read_blob):
            b = _rb(rel)
            return None if b is None else hashlib.sha256(b).hexdigest()

        out += judge_record(rec_path, rec, index_sha)
    return out


# --- wiring: read the commit git is about to write -------------------------------------------------------------
def _pra():
    from tools.gates import prereg_amendment_order as pra
    return pra


def check(paths, root=None, invocation=None):
    """`paths` (the registry's --diff-filter=A list) is NOT used: a MODIFIED file is in scope too."""
    pra = _pra()
    root = os.path.abspath(root or _ROOT)
    try:
        env = pra._git_env(root)
        kind = invocation if invocation is not None else pra._detect_invocation(root)
        parents = pra._parents(root, env, kind)
        if not parents:
            return []                    # unborn branch / amended root commit; None: a clean auto-merge, not checked
        ch = [pra._changes(root, env, p, None, _SPECS, True) for p in parents]
        cand = [p for p in sorted(set(ch[0]).intersection(*ch[1:]))
                if _TARGET_RE.match(os.path.basename(p)) and all(c[p][0] in ("A", "M", "R", "C") for c in ch)]
        if not cand:
            return []

        def read_blob(rel):
            r = pra._git(["cat-file", "blob", ":%s" % rel], root, env, ok_codes=(0, 128))
            return r.stdout if r.returncode == 0 else None

        probs = []
        for p in cand:
            blob = read_blob(p)
            probs += judge_file(p, (blob or b"").decode("utf-8", "replace"), read_blob)
        return probs
    except pra._GitReadError as e:
        return ["CLASS BAR could not read the commit being made (%s) -- failing CLOSED. Retry; if git itself is "
                "broken, fix that first." % e]


# --- selftest ---------------------------------------------------------------------------------------------------
def _rec(perfect=0.99, null=0.01, verdict="OK", **kw):
    r = {"spec": "docs/drafts/x/bar_sanity_spec.py", "spec_sha256": "a" * 64,
         "scorer_files": {"research/runners/x.py": "b" * 64}, "n": 1000,
         "kinds": {"perfect": "perfect", "null": "null"}, "pass_rate": {"perfect": perfect, "null": null},
         "thresholds": {"perfect_min": 0.95, "null_max": 0.05, "override_reason": None}, "verdict": verdict}
    r.update(kw)
    return r


def _pure(bad):
    none = lambda rel: None                                              # noqa: E731
    ok = judge_record("r.json", _rec(), none)
    if ok:
        bad.append("FALSE POSITIVE: a good record was refused: %s" % ok)
    if not judge_record("r.json", _rec(perfect=0.44, verdict="UNREACHABLE"), none):
        bad.append("did NOT refuse an UNREACHABLE record")
    if not judge_record("r.json", _rec(null=0.4, verdict="TOOTHLESS"), none):
        bad.append("did NOT refuse a TOOTHLESS record")
    if not judge_record("r.json", _rec(perfect=0.44, verdict="OK"), none):
        bad.append("did NOT refuse a record whose stated verdict OK disagrees with its own numbers (hand-edited)")
    if not judge_record("r.json", _rec(n=50), none):
        bad.append("did NOT refuse a record with n=50")
    loose = _rec(perfect=0.8)
    loose["thresholds"] = {"perfect_min": 0.5, "null_max": 0.05, "override_reason": None}
    if not judge_record("r.json", loose, none):
        bad.append("did NOT refuse a loosened threshold with no reason")
    loose["thresholds"]["override_reason"] = "the bar is a sanity floor, not a claim; see the packet"
    if judge_record("r.json", loose, none):
        bad.append("FALSE POSITIVE: a loosened threshold WITH a reason was refused")
    if not judge_record("r.json", _rec(), lambda rel: "c" * 64 if rel.startswith("research") else None):
        bad.append("did NOT refuse a record whose scorer file changed after the measurement (stale)")
    if judge_record("r.json", _rec(), lambda rel: "a" * 64 if rel.startswith("docs") else None):
        bad.append("FALSE POSITIVE: a record whose spec sha matches was refused")
    if not judge_record("r.json", {"verdict": "OK"}, none):
        bad.append("did NOT refuse a record missing its fields")
    good = json.dumps(_rec()).encode()
    blob = {"docs/rec.json": good}.get
    cases = [
        ("no line", "# prereg\n\nbar: x > 2\n", True),
        ("good record", "bar-sanity: docs/rec.json\n", False),
        ("good record, bullet and backticks", "- **bar-sanity:** `docs/rec.json`\n", False),
        ("missing record", "bar-sanity: docs/missing.json\n", True),
        ("N/A with a reason", "bar-sanity: N/A this file registers no bar; it records a recount only\n", False),
        ("N/A without a reason", "bar-sanity: N/A\n", True),
        ("N/A with a short reason", "bar-sanity: N/A no bar\n", True),
        ("-before line is not a record", "bar-sanity-before: docs/rec.json\n", True),
        ("one good, one bad", "bar-sanity: docs/rec.json\nbar-sanity: docs/missing.json\n", True),
    ]
    for label, text, should_fail in cases:
        got = bool(judge_file("p.md", text, blob))
        if got != should_fail:
            bad.append("judge_file case %r: expected %s, got %s" % (label, "a problem" if should_fail else "none",
                                                                    "a problem" if got else "none"))
    for name, want in (("2026-10-06-x-PREREG.md", True), ("2026-10-06-x-PREREGISTRATION.md", True),
                       ("AMENDMENT-A0.1.md", True), ("amendment_2.md", True), ("RESULT.md", False),
                       ("x-PREREG.txt", False), ("notes-on-amendments.md", False),
                       ("2026-09-24-x-PREREG-AMENDMENT-1.md", True), ("amendments.md", False)):
        if bool(_TARGET_RE.match(name)) != want:
            bad.append("file-name rule: %s should %smatch" % (name, "" if want else "not "))


def _repo(bad):
    """The wiring: a scratch repo, real `git diff --cached`. A new/modified prereg without a line blocks; an old
    one is grandfathered; a clean file with a committed OK record passes."""
    import subprocess
    import tempfile
    pra = _pra()
    with tempfile.TemporaryDirectory() as td:
        env0 = pra._stripped_env()

        def g(*a):
            r = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false"] + list(a),
                               cwd=td, env=env0, capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                raise RuntimeError("selftest git %s: %s" % (a[:2], r.stderr.strip()[:120]))

        def w(rel, text):
            full = os.path.join(td, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(text)

        g("init", "-q", "-b", "main")
        g("config", "user.email", "gate-selftest@example.invalid")
        g("config", "user.name", "gate selftest")
        w("docs/old-PREREG.md", "# old, before the gate\n")
        w("docs/notes.md", "nothing\n")
        g("add", "-A")
        g("commit", "-q", "-m", "base")
        if check([], root=td, invocation="commit"):
            bad.append("FALSE POSITIVE: a clean tree (grandfathered prereg) was flagged")
        w("docs/old-PREREG.md", "# old, before the gate\n\n## AMENDMENT 1\nbar: r < 0.2\n")
        g("add", "-A")
        if not check([], root=td, invocation="commit"):
            bad.append("did NOT block a MODIFIED prereg with no bar-sanity line (a gate that sees only added "
                       "files misses every amendment appended to an existing prereg)")
        w("docs/new-AMENDMENT-1.md", "# amendment\n\nbar: r < 0.2\n")
        w("docs/old-PREREG.md", "# old, before the gate\n")
        g("add", "-A")
        if not check([], root=td, invocation="commit"):
            bad.append("did NOT block an ADDED amendment with no bar-sanity line")
        w("docs/new-AMENDMENT-1.md", "# amendment\n\nbar-sanity: docs/rec.json\n")
        g("add", "-A")
        if not check([], root=td, invocation="commit"):
            bad.append("did NOT block an amendment whose cited record is not committed")
        w("docs/rec.json", json.dumps(_rec(perfect=0.44, verdict="UNREACHABLE")))
        g("add", "-A")
        if not check([], root=td, invocation="commit"):
            bad.append("did NOT block an amendment citing an UNREACHABLE record")
        w("docs/rec.json", json.dumps(_rec()))
        g("add", "-A")
        if check([], root=td, invocation="commit"):
            bad.append("FALSE POSITIVE: an amendment citing a committed OK record was blocked: %s"
                       % check([], root=td, invocation="commit"))
        if check([], root=td, invocation="merge"):
            bad.append("a clean auto-merge must not be checked")


def selftest():
    """FAILING DIRECTION FIRST in each block, then the no-false-positive cases."""
    bad = []
    _pure(bad)
    try:
        _repo(bad)
    except (RuntimeError, OSError, subprocess_error()) as e:
        bad.append("selftest scratch repo could not be built: %s" % e)
    return bad


def subprocess_error():
    import subprocess
    return subprocess.SubprocessError


if __name__ == "__main__":
    print("class BAR bar_sanity_line — run via the registry (tools/gates), no standalone report.")
