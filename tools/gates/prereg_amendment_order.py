"""CLASS PRA — an AMENDMENT to an EXISTING pre-registration, committed TOGETHER WITH run data.

THE GAP (opus review, 2026-09-25, of branch research/slotbinder-gate-latency). `gates/prereg_before_run` (CLASS
PR) only looks at pre-registrations git reports as ADDED. Every real amendment in this repo lands by MODIFYING an
existing `*PREREG*.md`, so none was ever checked for run order. An amendment is a pre-registration in miniature
(new thresholds, arms or gates written down before the next run), so the same rule applies to it: when the
amendment and the data land in ONE commit, git history cannot show the amendment came first.

THE RULE. A commit BLOCKS when BOTH hold:
  (a) it MODIFIES (or renames) a `research/findings/*prereg*.md` / `docs/plans/*prereg*.md` file (same filename
      rule as `prereg_before_run`) and that file gains AMENDMENT ACTIVITY relative to the commit's parent(s):
        N1  a new amendment entry: its ID (the `6` of `## AMENDMENT 6`, `**Amendment 6 (...)**`, `**A6, ...**`,
            `ADDENDUM A6`) is in no parent version. Entries are recognised in every form the corpus uses:
            `#` headings starting `Amendment`/`Addendum`; bold paragraphs or bullets starting `**AMENDMENT`,
            `**Amendment`, `**ADDENDUM`; and, INSIDE an amendment-log section (`## AMENDMENT LOG`, `### Amendment
            log: ...`), bold `**A<n>` entries followed by a date/delimiter and any top-level bold bullet (the
            checklist's AMENDMENT LOG form: `- **2026-09-24, filed after ...**`). ID-less entries are keyed by
            their text.
        N2  a new entry line reusing an existing ID (an ADDENDUM to amendment N, a rewritten amendment heading) —
            UNLESS it is a declared record subsection: its label says `record` and a non-record entry with the same
            ID exists (`### AMENDMENT 6, smoke record (appended after the declared smoke ran ...)`). A record
            subsection's own body never counts either.
        N3  the registered BODY of an amendment that already exists in the parent changed (compared with
            whitespace collapsed, so re-wrapping is not a change). This is the slot-binder case (835fc252e): the
            DRAFT item of AMENDMENT 2 was rewritten into its completed registration in the same commit as the
            N=8/32/128 data it was chosen from.
        N4  new non-placeholder prose inside an amendment-log section that no entry owns.
  (b) the same commit also writes (adds, modifies, renames or copies — anything but a deletion) a data artifact
      under `research/findings/raw/**`. NOT data: `*.prov.json` sidecars, `*.progress_*` sidecars, and everything
      under `research/findings/raw/_provenance/` — `runs.jsonl` there is appended automatically by EVERY runner
      invocation in the checkout (it changes in ~1 of 4 commits), so it says nothing about which runs landed.

ESCAPE. A line added by this commit to that prereg reading `amendment-same-commit: <reason, >=15 chars>` (e.g.
"the amendment only records the smoke that produced these files; no rule changes"). Scoped to the commit's own
added lines, so an old amendment's escape cannot cover a new one.

WHAT IS READ — the staged index the hook is COMMITTING, never a stand-in for it.
  * The registry passes only `--diff-filter=A` paths, which cannot see an overwritten (status M) artifact; so this
    gate ignores `paths` and reads `git diff --cached` itself, on every commit, even when `paths` is empty.
  * `GIT_INDEX_FILE` is honoured: `git commit -a` and `git commit -- <paths>` commit a TEMPORARY index that git
    names in that variable (`.git/index.lock`, `.git/next-index-<pid>.lock`, or the same under
    `.git/worktrees/<name>/`), and the default index does not hold what is being committed. One rule serves the
    hook and the selftest alike (`_git_env`): drop the GIT_* location variables, then restore `GIT_INDEX_FILE`
    whenever it lies inside the repo's own git dir. So the selftest's temporary-index case runs the hook's code path.
  * MERGES are checked, not skipped, and do not false-block: a path counts only if it differs from EVERY parent
    (HEAD and each MERGE_HEAD), amendment activity only if it is new relative to every parent. Merging a branch
    whose amendment and data landed in separate, correctly ordered commits therefore passes; an evil merge that
    writes a new amendment and new data itself is still caught.
  * FAILS CLOSED. If the index, MERGE_HEAD or a blob cannot be read, or git times out, the gate returns a problem
    rather than a silent pass: a gate that says nothing when it could not look is indistinguishable from a pass.
  * COST. ~3 git calls per commit, plus 2 blob reads per modified prereg (memoised within a call). No history walk.

WHAT THIS GATE CANNOT CATCH (stated, not hidden).
  * A blanket rule, like `prereg_before_run`: it does not know which artifact an amendment governs, so an
    amendment that only records a finished run blocks until its author adds the escape or splits the commit.
  * A prereg EDITED after its run with no amendment marker at all (the rewritten body of the ORIGINAL
    registration, outside any amendment/log section) — N3 covers amendment bodies only. That broader rule would fire
    on every results-appended-to-prereg commit this repo makes; it stays a reviewer's call.
  * A run executed BEFORE the amendment commit but committed AFTER it (git order looks right, the run predates the
    text). The first version of this gate tried to catch it from `.prov.json` git SHAs of artifacts CITED in an
    amendment, but amendments cite what had been SEEN (docs/BUILD_LANE_CHECKLIST.md), so the premise was inverted:
    0 correct and 3 false blocks over 400 commits (review 2026-09-25). It was removed, not weakened. A correct
    version needs a declaration of what an amendment GOVERNS, which no prereg writes today.
  * Amendment markers outside the forms above (an amendment written as plain prose), and preregs whose filename
    lacks "prereg" or that live outside the two directories.
  * A `GIT_INDEX_FILE` pointing OUTSIDE the repo's git dir (a hand-set custom index) is not honoured; the default
    index is read instead. git itself never does this for `commit`, `commit -a` or `commit -- <paths>`.

REPLAY (2026-09-25, the last 2500 commits of main, 534 of them merges; the gate's own `_problems` on each commit's
real trees, merges judged against every parent). Truth = a new amendment/log entry or an edit to a committed
amendment's text, landing with run data. 36 commits modified a prereg while writing under raw/; 19 BLOCK, all true
(TP 19, FP 0), and none of the other 17 is a miss (FN 0): 10 wrote only `_provenance/runs.jsonl`, 1 is a record
subsection (414e1ba4f), 6 append results to the ORIGINAL registration with no amendment (the declared blind spot
above). Most of the 19 are records of runs the amendment was written after, i.e. the escape line's intended use.
The review's 5 named misses: 835fc252e, 0ee39e293, 356c9f040, f2b9e69b7 now block; 72b744c4c wrote only the
provenance log. The first version (hook semantics) blocked 13 of these commits: 11 true, 2 false (414e1ba4f,
5b5ea1b74), and missed 8. Judging a merge against HEAD alone would have false-blocked 19 of the 534 merges.

MUTATION-VERIFY. tests/test_gate_prereg_amendment_order.py::test_selftest_kills_mutants applies 12 mutants and
requires selftest() -- the registry's only trust signal -- to FAIL on each: check() returns early on empty `paths`;
the prereg status filter stops matching `M`; the raw filter counts only ADDED files; the detector is unwired; bold
entries are not detected; GIT_INDEX_FILE is dropped; merges are judged against HEAD alone; the record exemption,
the N3 body check, the provenance exclusion or the escape is removed; a git error returns no problem.
"""
from __future__ import annotations

import difflib
import os
import re
import subprocess

NAME = "prereg_amendment_order"
CLASS_ID = "PRA"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PREREG_RE = re.compile(r"^(research/findings|docs/plans)/[^/]*prereg[^/]*\.md$", re.I)
_RAW_RE = re.compile(r"^research/findings/raw/.+")
_RAW_NOT_DATA_RE = re.compile(r"(?:\.prov\.json$|^research/findings/raw/_provenance/|\.progress_[^/]*$)")
_ESCAPE_RE = re.compile(r"^\s*[-*]?\s*amendment-same-commit:\s*(.{15,})$", re.I)
_GIT_TIMEOUT = 20

# --- amendment-entry grammar --------------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_BOLD_RE = re.compile(r"^(\s*)(?:[-*+]\s+|\d+[.)]\s+)?\*\*\s*(.*)$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_WORD_RE = re.compile(r"^(amendment(s?)|addend(um|a))\b[\s\-]*(.*)$", re.I)
_LOGWORD_RE = re.compile(r"^(?:log|history)\b[\s\-]*(.*)$", re.I)
_CORR_RE = re.compile(r"^correction\b\s*(\d+)?", re.I)
_ID_RE = re.compile(r"^(?:no\.?\s*)?([A-Za-z]?\d+[a-z]?|[A-Z])(?=$|[\s,:;.()—–\-*\]])")
_DELIM_RE = re.compile(r"^(?:$|[(:,;.—–\-*])")
_AFORM_RE = re.compile(r"^A(\d+)(?=\s*(?:$|[,:;(—–]|-{1,2}\s|\.?\*\*))")
_RECORD_RE = re.compile(r"\brecord\b", re.I)
_SECNUM_RE = re.compile(r"^(?:\u00a7\s*)?\d+(?:\.\d+)*[.)]?\s+")          # `## 6. AMENDMENT LOG` -> `AMENDMENT LOG`
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_PLACEHOLDER_RE = re.compile(r"^\s*(?:[-*+]\s+)?[(_*\s]*(?:none|no amendments?|n/?a)\b", re.I)
_GIT_ENV_STRIP = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                  "GIT_COMMON_DIR", "GIT_CEILING_DIRECTORIES", "GIT_PREFIX")


def _collapse(s):
    return " ".join(s.split())


def _norm_id(tok):
    t = tok.upper()
    return t[1:] if re.match(r"^A\d", t) else t


def _label_kind(label, log_ctx):
    """('container'|'entry'|None, key). `label` is a heading's text or the text after a leading `**`."""
    s = label.strip().lstrip("*_ ").strip()
    m = _WORD_RE.match(s)
    if m:
        plural = bool(m.group(2)) or (m.group(3) or "").lower() == "a"
        rest = m.group(4)
        lm = _LOGWORD_RE.match(rest)
        if lm:
            cm = _CORR_RE.match(lm.group(1))
            if cm:
                return "entry", "LOGCORR" + (cm.group(1) or "")
            return "container", None
        if plural:
            return "container", None
        im = _ID_RE.match(rest)
        if im:
            return "entry", _norm_id(im.group(1))
        if _DELIM_RE.match(rest):
            return "entry", "TEXT:" + _collapse(s).lower()
        return None, None
    if log_ctx:
        am = _AFORM_RE.match(s)
        if am:
            return "entry", am.group(1)
    return None, None


def _direct_log(stack):
    """True when the innermost amendment-ish heading enclosing this line is an amendment-LOG section (not an entry)."""
    for _, k, _ in reversed(stack):
        if k == "container":
            return True
        if k == "entry":
            return False
    return False


class _Doc:
    """A prereg parsed into amendment entries. owner[i] = index of the entry that owns line i (the innermost one),
    or None; in_log[i] = line i lies inside an amendment-log section; head[i] = line i is a heading."""

    def __init__(self, text):
        self.lines = text.splitlines()
        n = len(self.lines)
        self.owner, self.in_log, self.head = [None] * n, [False] * n, [False] * n
        self.entries = []                                   # [{"key", "line", "i"}]
        stack, cur_bold, fence, prev_blank = [], None, False, True
        for i, ln in enumerate(self.lines):
            if _FENCE_RE.match(ln):
                fence = not fence
            hm = None if fence else _HEADING_RE.match(ln)
            if hm and i > 0 and self.head[i - 1] and stack and stack[-1][0] == len(hm.group(1)):
                self.head[i] = True           # a heading WRAPPED onto the next line (W2 splits): same section
            elif hm:
                lvl = len(hm.group(1))
                stack = [s for s in stack if s[0] < lvl]
                cur_bold = None
                kind, key = _label_kind(_SECNUM_RE.sub("", hm.group(2)), _direct_log(stack))
                e = None
                if kind == "entry":
                    e = len(self.entries)
                    self.entries.append({"key": key, "line": ln.strip(), "i": i})
                stack.append((lvl, kind, e))
                self.head[i] = True
            elif not fence:
                bm = _BOLD_RE.match(ln)
                if bm and (prev_blank or _LIST_RE.match(ln)):
                    direct = _direct_log(stack)
                    kind, key = _label_kind(bm.group(2), direct)
                    # the checklist's AMENDMENT LOG form: a top-level bold entry directly in a log section, e.g.
                    # `- **2026-09-24, filed after ...**` -- the first one, or any later one that carries a date
                    if (kind is None and direct and len(bm.group(1)) < 2
                            and (cur_bold is None or _DATE_RE.search(bm.group(2)[:60]))):
                        kind, key = "entry", "TEXT:" + _collapse(ln).lower()
                    if kind == "entry":
                        cur_bold = len(self.entries)
                        self.entries.append({"key": key, "line": ln.strip(), "i": i})
            own = cur_bold
            if own is None:
                for _, _, e in reversed(stack):
                    if e is not None:
                        own = e
                        break
            self.owner[i] = own
            self.in_log[i] = any(k == "container" for _, k, _ in stack)
            prev_blank = not ln.strip()

    def body(self, e):
        return _collapse(" ".join(ln for j, ln in enumerate(self.lines) if self.owner[j] == e and j != self.entries[e]["i"]))

    def log_prose(self):
        return {_collapse(ln) for j, ln in enumerate(self.lines)
                if self.in_log[j] and self.owner[j] is None and not self.head[j] and ln.strip()
                and not _PLACEHOLDER_RE.match(ln)}


def _added_indices(old_lines, new_lines):
    sm = difflib.SequenceMatcher(None, [l.rstrip() for l in old_lines], [l.rstrip() for l in new_lines],
                                 autojunk=False)
    out = set()
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            out.update(range(j1, j2))
    return out


def _amendment_activity(new_text, parent_texts):
    """Pure. [(code, description), ...] of amendment activity in `new_text` that is new relative to EVERY parent
    text (one parent for a normal commit, 2+ for a merge). Empty means none."""
    new = _Doc(new_text)
    parents = [_Doc(t) for t in parent_texts]
    added = None
    for p in parents:
        a = _added_indices(p.lines, new.lines)
        added = a if added is None else (added & a)
    added = added or set()
    parent_keys = {e["key"] for p in parents for e in p.entries}
    plain_keys = parent_keys | {e["key"] for e in new.entries if not _RECORD_RE.search(e["line"])}

    def is_record(e):
        ent = new.entries[e]
        return bool(_RECORD_RE.search(ent["line"])) and ent["key"] in plain_keys and not ent["key"].startswith("TEXT:")

    out = []
    for idx, ent in enumerate(new.entries):
        if is_record(idx) or ent["i"] not in added:
            continue
        if ent["key"] not in parent_keys:
            out.append(("N1", "new amendment entry %r" % ent["line"][:80]))
        else:
            out.append(("N2", "new entry line for existing amendment %s: %r" % (ent["key"], ent["line"][:80])))
    for idx, ent in enumerate(new.entries):
        if is_record(idx) or ent["i"] in added:
            continue
        bodies = []
        for p in parents:
            for pidx, pent in enumerate(p.entries):
                if pent["line"] == ent["line"]:
                    bodies.append(p.body(pidx))
        if bodies and all(new.body(idx) != b for b in bodies):
            out.append(("N3", "the registered body of existing amendment %r changed" % ent["line"][:80]))
    prose = new.log_prose()
    for p in parents:
        prose -= p.log_prose()
    if prose:
        out.append(("N4", "new amendment-log prose %r" % sorted(prose)[0][:80]))
    return out


def _escaped(new_text, parent_texts):
    new_lines = new_text.splitlines()
    added = None
    for t in parent_texts:
        a = _added_indices(t.splitlines(), new_lines)
        added = a if added is None else (added & a)
    return any(_ESCAPE_RE.match(new_lines[i]) for i in (added or ()))


def _is_data_artifact(path):
    return bool(_RAW_RE.match(path)) and not _RAW_NOT_DATA_RE.search(path)


def _problems(prereg_changes, raw_written):
    """Pure. prereg_changes: [(path, new_text, [parent_text, ...]), ...] for every MODIFIED/renamed prereg.
    raw_written: every path the commit writes (not deletes), new relative to every parent."""
    data = [p for p in raw_written if _is_data_artifact(p)]
    if not data:
        return []
    out = []
    for path, new_text, parent_texts in prereg_changes:
        act = _amendment_activity(new_text, parent_texts)
        if not act or _escaped(new_text, parent_texts):
            continue
        out.append(
            "CLASS PRA %s gains amendment activity [%s: %s] in the SAME commit as %d raw data artifact(s) (e.g. %s). "
            "Git history cannot show the amendment was written before the data. Fix: commit the amendment FIRST, "
            "then run, then commit the artifacts -- or, if the amendment only records runs it does not govern, add "
            "a line `amendment-same-commit: <reason>` to the file in this commit."
            % (path, act[0][0], act[0][1], len(data), data[0]))
    return out


# --- wiring: the REAL staged index for the real commit; `root=` for a scratch repo ------------------------------
class _GitReadError(Exception):
    pass


def _stripped_env():
    env = dict(os.environ)
    for k in _GIT_ENV_STRIP:
        env.pop(k, None)
    return env


def _git_env(root):
    """The environment every git call in check() uses -- ONE rule for the real hook and for a scratch repo, so the
    selftest exercises the path the hook runs. GIT_* location variables are dropped (git finds the repo from
    cwd=root; a hook's GIT_DIR must not leak into a scratch repo), then GIT_INDEX_FILE is put back whenever it lies
    inside root's own git dir. Measured 2026-09-25 in a scratch repo: the hook gets `.git/index` (relative) for a
    plain commit, `.git/index.lock` for `commit -a`, `.git/next-index-<pid>.lock` for `commit -- <paths>`, and the
    same names under `.git/worktrees/<name>/` in a linked worktree -- the last three hold what is being COMMITTED."""
    env = _stripped_env()
    idx = os.environ.get("GIT_INDEX_FILE")
    if idx:
        gd = _git(["rev-parse", "--absolute-git-dir"], root, env).stdout.decode("utf-8", "replace").strip()
        idx_abs = os.path.realpath(os.path.join(root, idx))
        if gd and idx_abs.startswith(os.path.realpath(gd) + os.sep):
            env["GIT_INDEX_FILE"] = idx_abs
    return env


def _git(args, root, env, ok_codes=(0,)):
    try:
        r = subprocess.run(["git"] + args, cwd=root, env=env, capture_output=True, timeout=_GIT_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise _GitReadError("`git %s` timed out after %ss" % (" ".join(args[:3]), _GIT_TIMEOUT))
    except OSError as e:
        raise _GitReadError("`git %s` could not run: %s" % (" ".join(args[:3]), e))
    if r.returncode not in ok_codes:
        raise _GitReadError("`git %s` exited %d: %s" % (" ".join(args[:3]), r.returncode,
                                                       r.stderr.decode("utf-8", "replace").strip()[:160]))
    return r


def _parents(root, env):
    """[HEAD] (or [] on an unborn branch) + every MERGE_HEAD when a merge is being committed."""
    r = _git(["rev-parse", "-q", "--verify", "HEAD^{commit}"], root, env, ok_codes=(0, 1))
    head = r.stdout.decode().strip()
    if not head:
        return []
    mh = _git(["rev-parse", "--path-format=absolute", "--git-path", "MERGE_HEAD"], root, env).stdout.decode().strip()
    merge_heads = []
    if mh and os.path.exists(mh):
        try:
            with open(mh, encoding="utf-8") as fh:
                merge_heads = [ln.split()[0] for ln in fh if ln.strip()]
        except OSError as e:
            raise _GitReadError("MERGE_HEAD exists but cannot be read: %s" % e)
    return [head] + merge_heads


def _staged_changes(root, env, parent):
    """{new_path: (status_letter, old_path)} of the staged index relative to `parent`."""
    out = _git(["diff", "--cached", "-z", "--name-status", "-M", parent, "--"], root, env).stdout
    toks = out.decode("utf-8", "surrogateescape").split("\0")
    res, i = {}, 0
    while i < len(toks) and toks[i]:
        st = toks[i][:1]
        if st in "RC":
            res[toks[i + 2]] = (st, toks[i + 1])
            i += 3
        else:
            res[toks[i + 1]] = (st, toks[i + 1])
            i += 2
    return res


def check(paths, root=None):
    """`paths` (the registry's --diff-filter=A list) is deliberately NOT used: see the docstring."""
    root = os.path.abspath(root or _ROOT)
    blobs = {}

    def blob(spec):
        if spec not in blobs:
            blobs[spec] = _git(["cat-file", "blob", spec], root, env).stdout.decode("utf-8", "replace")
        return blobs[spec]

    try:
        env = _git_env(root)
        parents = _parents(root, env)
        if not parents:
            return []                                              # first commit: nothing can be MODIFIED
        changes = [_staged_changes(root, env, p) for p in parents]
        common = set(changes[0])
        for c in changes[1:]:
            common &= set(c)                                       # a merge: new relative to EVERY parent
        raw_written = sorted(p for p in common if _RAW_RE.match(p) and all(c[p][0] != "D" for c in changes))
        prereg_changes = []
        for p in sorted(common):
            if not _PREREG_RE.match(p) or changes[0][p][0] not in ("M", "R"):
                continue
            if any(c[p][0] == "D" for c in changes):
                continue
            parent_texts = [blob("%s:%s" % (sha, c[p][1])) if c[p][0] != "A" else ""
                            for sha, c in zip(parents, changes)]
            prereg_changes.append((p, blob(":" + p), parent_texts))
        if not prereg_changes:
            return []
        return _problems(prereg_changes, raw_written)
    except _GitReadError as e:
        return ["CLASS PRA could not read the commit being made (%s) -- failing CLOSED: this gate cannot say the "
                "commit is clean without reading it. Retry; if git itself is broken, fix that first." % e]


# --- selftest -------------------------------------------------------------------------------------------------
_ST_LOG = "# prereg\n\nthresholds: G1 >= 0.5\n\n## Amendment log\n\n(none at filing)\n"
_ST_BOLD = "\n**AMENDMENT 1: 2026-01-02, after the seed-7 smoke, before round 2.** G1 is now >= 0.6.\n"


def _selftest_pure(bad):
    base = _ST_LOG
    raw = ["research/findings/raw/x/s42.json"]
    p = "research/findings/2026-01-01-x-PREREG.md"
    heading = base + "\n## AMENDMENT 1 (2026-01-02, before round 2)\n\nG1 >= 0.6\n"
    bold = base + _ST_BOLD
    aform = base + "\n**A1, 2026-01-02 ~12:45 EDT, instrument crash fix. The gate is unchanged.**\n"
    logbullet = base + "\n- **2026-01-02, filed after the seed-7 smoke completed.** G1 unchanged.\n"
    record = heading + "\n### AMENDMENT 1, smoke record (appended after the declared smoke ran)\n\nG1 read 0.7.\n"
    rewrite = heading.replace("G1 >= 0.6", "G1 >= 0.65")
    escaped = bold + "- amendment-same-commit: these files are the seed-7 smoke this amendment records\n"
    criteria = base.replace("thresholds: G1 >= 0.5", "thresholds:\n\n- **A1 (route):** G1 >= 0.5")
    criteria2 = criteria.replace("G1 >= 0.5", "G1 >= 0.5\n- **A2 (two referents):** G2")
    fenced = base.replace("G1 >= 0.5\n", "G1 >= 0.5\n\n```\n## AMENDMENT 9 (inside a code block)\n```\n")
    prose = base.replace("G1 >= 0.5\n", "G1 >= 0.5\n\nsome clarifying prose outside any amendment.\n")

    def fires(new, old, r=raw):
        return bool(_problems([(p, new, [old])], r))

    # FAILING DIRECTION: every real amendment form, beside a data artifact, MUST block
    for label, new in (("heading", heading), ("bold", bold), ("A<n> log entry", aform),
                       ("dated log bullet", logbullet)):
        if not fires(new, base):
            bad.append("did NOT catch a new %s amendment beside a raw data artifact" % label)
    nolog = "# prereg\n\nthresholds: G1 >= 0.5\n"
    if not fires(nolog + "\n**Amendment 1 (2026-01-02; a stricter gate).** G1 >= 0.6\n", nolog):
        bad.append("did NOT catch a bold `**Amendment N (...)**` paragraph in a prereg with no amendment-log section")
    if not fires(rewrite, heading):
        bad.append("did NOT catch the registered body of an existing amendment rewritten beside data (N3)")
    if not fires(heading, base, ["research/findings/raw/_provenance/runs.jsonl", "research/findings/raw/x/s7.json"]):
        bad.append("a data artifact listed after the provenance log was not seen")
    # PASSING / no false positive
    if fires(record, heading):
        bad.append("FALSE POSITIVE: a declared record subsection of an already-committed amendment")
    if fires(escaped, base):
        bad.append("FALSE POSITIVE: the amendment-same-commit escape was not honoured")
    if fires(heading, base, ["research/findings/raw/_provenance/runs.jsonl", "research/findings/raw/x/a.json.prov.json",
                             "research/findings/raw/x/a.json.progress_slotbinder.json"]):
        bad.append("FALSE POSITIVE: provenance log / sidecars are not run data")
    if fires(heading, base, []):
        bad.append("FALSE POSITIVE: no raw artifact at all")
    if fires(prose, base):
        bad.append("FALSE POSITIVE: prose outside any amendment section")
    if fires(criteria2, criteria):
        bad.append("FALSE POSITIVE: an `**A2 (...)**` criterion bullet outside an amendment log")
    if fires(fenced, base):
        bad.append("FALSE POSITIVE: an amendment heading inside a fenced code block")
    if fires(heading.replace("G1 >= 0.6\n", "G1 >=\n0.6\n"), heading):
        bad.append("FALSE POSITIVE: re-wrapping an amendment's body is not a change")


def _selftest_repo(bad):
    """check()-level, in a scratch repo: the wiring the registry actually runs."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        env0 = _stripped_env()

        def g(*args, env=None):
            r = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false"] + list(args),
                               cwd=td, env=env or env0, capture_output=True, text=True, timeout=_GIT_TIMEOUT)
            if r.returncode != 0:
                raise RuntimeError("selftest git %s: %s" % (args[:2], r.stderr.strip()[:120]))
            return r.stdout

        def write(rel, text, mode="w"):
            full = os.path.join(td, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, mode, encoding="utf-8") as fh:
                fh.write(text)

        pre, raw = "research/findings/2026-01-01-st-PREREG.md", "research/findings/raw/st/s7.json"
        g("init", "-q", "-b", "main")
        g("config", "user.email", "gate-selftest@example.invalid")
        g("config", "user.name", "gate selftest")
        write(pre, _ST_LOG)
        write(raw, '{"v": 1}')
        g("add", "-A")
        g("commit", "-q", "-m", "base")

        # (1) amendment + an OVERWRITTEN raw artifact, NOTHING added -> the registry's `paths` is empty
        write(pre, _ST_BOLD, "a")
        write(raw, '{"v": 2}')
        g("add", "-u")
        if not check([], root=td):
            bad.append("check() did NOT catch an amendment committed with an overwritten (status M) raw artifact "
                       "and nothing added -- part 1 must read the staged index itself, whatever `paths` holds")
        g("reset", "-q")

        # (2) `git commit -a` / `git commit -- <paths>` stage into a TEMPORARY index named by GIT_INDEX_FILE
        alt = os.path.join(td, ".git", "selftest-next-index")
        env_alt = dict(env0, GIT_INDEX_FILE=alt)
        g("read-tree", "HEAD", env=env_alt)
        g("add", pre, raw, env=env_alt)
        saved = os.environ.get("GIT_INDEX_FILE")
        try:
            os.environ["GIT_INDEX_FILE"] = alt
            if not check([], root=td):
                bad.append("check() ignored GIT_INDEX_FILE: an amendment + raw artifact staged in the temporary "
                           "index a `git commit -a` / `commit -- paths` hook sees was not caught")
        finally:
            if saved is None:
                os.environ.pop("GIT_INDEX_FILE", None)
            else:
                os.environ["GIT_INDEX_FILE"] = saved
        if check([], root=td):
            bad.append("FALSE POSITIVE: the default index is clean (the change is only in the temporary index)")
        g("checkout", "--", ".")

        # (3) a MERGE bringing in an amendment and its data from correctly ordered commits must NOT block
        g("checkout", "-q", "-b", "lane")
        write(pre, _ST_BOLD, "a")
        g("commit", "-q", "-am", "amendment first")
        write("research/findings/raw/st/s42.json", '{"v": 3}')
        g("add", "-A")
        g("commit", "-q", "-m", "then the data")
        g("checkout", "-q", "main")
        write("unrelated.txt", "x\n")
        g("add", "-A")
        g("commit", "-q", "-m", "main moves on")
        g("merge", "-q", "--no-ff", "--no-commit", "lane")
        merge_problems = check(["research/findings/raw/st/s42.json"], root=td)
        if merge_problems:
            bad.append("FALSE POSITIVE on a merge of correctly ordered history: %s" % merge_problems[0][:100])
        # ...but an evil merge that writes a NEW amendment and new data itself is still caught
        write(pre, "\n## AMENDMENT 2 (written in the merge)\n\nG1 >= 0.7\n", "a")
        write("research/findings/raw/st/s43.json", '{"v": 4}')
        g("add", "-A")
        if not check([], root=td):
            bad.append("did NOT catch a merge that itself writes a new amendment and new data")
        g("merge", "--abort")

        # (4) FAIL CLOSED: a root whose index cannot be read is a problem, never a silent pass
        broken = os.path.join(td, "broken")
        write("broken/.git", "gitdir: %s\n" % os.path.join(td, "no-such-gitdir"))
        if not check([], root=broken):
            bad.append("check() FAILED OPEN: an unreadable repository/index returned no problem")


def selftest():
    """FAILING DIRECTION FIRST in each block, then the no-false-positive cases."""
    bad = []
    _selftest_pure(bad)
    try:
        _selftest_repo(bad)
    except (RuntimeError, OSError, subprocess.SubprocessError) as e:
        bad.append("selftest scratch repo could not be built: %s" % e)
    return bad


if __name__ == "__main__":
    print("class PRA prereg-amendment-order — run via the registry (tools/gates), no standalone report.")
