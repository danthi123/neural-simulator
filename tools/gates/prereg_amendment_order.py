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
            `#` headings starting `Amendment`/`Addendum`/`Erratum`; bold paragraphs or bullets starting
            `**AMENDMENT`, `**Amendment`, `**ADDENDUM`, `**Erratum`; the same word after a one- or two-word
            qualifier (`### Seed-integrity amendment - <date>`, `**Pre-formal amendment after ...**`, `**v3
            amendment (...)**`; a qualifier such as `this`/`why`/`per`/`by` makes it a reference, not an entry, and
            so does an ID after a qualified word: `## Rerun under AMENDMENT-1 (v2): NO-GO`, `### Results under
            amendment 1` and `**Verdict after amendment 1:**` point AT amendment 1);
            and, INSIDE an amendment-log section (`## AMENDMENT LOG`, `### Amendment log: ...`), bold `**A<n>`
            entries followed by a date/delimiter and any top-level bold bullet (the checklist's AMENDMENT LOG form:
            `- **2026-09-24, filed after ...**`). A bold entry must open a paragraph or a list item; one glued to
            the line above counts only when it names an ID and a date (`**AMENDMENT 3 (2026-09-25, ...)**`),
            because a wrapped REFERENCE (`**AMENDMENT 3** governs ...`) has the same shape. ID-less entries are
            keyed by their text; errata by `ERR` + their ID.
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

WHAT IS READ — the commit git is about to write, never a stand-in for it.
  * The registry passes only `--diff-filter=A` paths, which cannot see an overwritten (status M) artifact; so this
    gate ignores `paths` and reads `git diff --cached` itself, on every commit, even when `paths` is empty.
  * `GIT_INDEX_FILE` is honoured: `git commit -a` and `git commit -- <paths>` commit a TEMPORARY index that git
    names in that variable (`.git/index.lock`, `.git/next-index-<pid>.lock`, or the same under
    `.git/worktrees/<name>/`), and the default index does not hold what is being committed. One rule serves the
    hook and the selftest alike (`_git_env`): drop the GIT_* location variables, then restore `GIT_INDEX_FILE`
    whenever it lies inside the repo's own git dir. So the selftest's temporary-index case runs the hook's code path.
  * THE PARENTS are the ones the new commit will have, which is not always HEAD. The gate asks which git command
    is running the hook: the nearest ancestor `git` process whose working directory is this checkout (git chdirs
    to the work-tree top before any hook, from a subdirectory or with `-C`; measured 2026-09-25), read from
    /proc. Then:
      - MERGE_HEAD present (a conflicted merge, or `merge --no-commit`, finished by `git commit` OR by `git merge
        --continue`): judged against HEAD and every MERGE_HEAD. A path counts only if it differs from EVERY parent,
        amendment activity only if it is new relative to every parent. A merge that itself writes a new amendment
        and new data is caught. `git merge --continue` runs the hook from a process whose command is `merge`, so
        MERGE_HEAD is read BEFORE the command is consulted; the other order fails OPEN on exactly that commit
        (review r4, 2026-09-25: an evil merge committed rc=0), and the selftest pins it.
      - `git merge` / `git pull` with no MERGE_HEAD: a CLEAN AUTO-MERGE. git runs pre-merge-commit BEFORE it writes
        MERGE_HEAD (builtin/merge.c prepare_to_commit), so HEAD is the only parent visible, and judging against it
        alone false-blocked 19 of 534 real main merges (and this branch's own sync merge, 2026-09-25). Nothing is
        checked: git refuses to start a merge over staged changes, so the tree is the strategy's output and no
        person wrote any of it, and each side's commits were judged when they were made. A merge a person edits
        is finished by `git commit` or `git merge --continue` with MERGE_HEAD present, above.
      - `git commit --amend` (or an unambiguous abbreviation, `--am` / `--ame` / `--amen`): the new commit REPLACES
        HEAD, so it is judged against HEAD's parents. Diffing against HEAD let an amendment committed alone and
        its data added by `--amend` land as ONE commit and pass (review 2026-09-25).
      - anything else: HEAD.
  * FAILS CLOSED. If the index, MERGE_HEAD or a blob cannot be read, or git times out, the gate returns a problem
    rather than a silent pass: a gate that says nothing when it could not look is indistinguishable from a pass.
  * COST. 4 git calls on a commit that modifies no prereg (one more under `--amend` and per extra merge parent),
    plus a few /proc reads: 20 ms on this repo. Rename detection runs only over the two prereg pathspecs and the
    raw/ diff runs without it: over the whole tree `-M` took 5.8 s per 1000-commit-divergent parent (review
    2026-09-25; 1.72 s against 0.07 s here, tree to tree), and past the timeout the gate fails closed. When a
    prereg IS modified: one raw/-limited diff and 2 blob reads per modified prereg. No history walk.

WHAT THIS GATE CANNOT CATCH (stated, not hidden).
  * A blanket rule, like `prereg_before_run`: it does not know which artifact an amendment governs, so an
    amendment that only records a finished run blocks until its author adds the escape or splits the commit. Every
    one of the replay's 19 blocks is exactly that (see REPLAY): the escape is the expected path for a record.
  * A prereg EDITED after its run with no amendment marker at all (the rewritten body of the ORIGINAL
    registration, outside any amendment/log section) — N3 covers amendment bodies only. That broader rule would fire
    on every results-appended-to-prereg commit this repo makes; it stays a reviewer's call.
  * A run executed BEFORE the amendment commit but committed AFTER it (git order looks right, the run predates the
    text). The first version of this gate tried to catch it from `.prov.json` git SHAs of artifacts CITED in an
    amendment, but amendments cite what had been SEEN (docs/BUILD_LANE_CHECKLIST.md), so the premise was inverted:
    0 correct and 3 false blocks over 400 commits (review 2026-09-25). It was removed, not weakened. A correct
    version needs a declaration of what an amendment GOVERNS, which no prereg writes today.
  * Amendment markers outside the forms above: an amendment written as plain prose, and labels named only by a
    generic word -- `**Instrument revision, before any mechanism run.**`, `**Wording correction (...)**`, `##
    Declared deviations` -- because `correction`/`revision` also open 20+ ordinary sections of whole-file
    registrations in the corpus (`## Locked correction`, `## Correction design`). Preregs whose filename lacks
    "prereg" or that live outside the two directories.
  * No /proc (not Linux), or no ancestor git process working in this checkout (the gate run by hand): the gate
    cannot tell which command is committing and judges against HEAD. A clean auto-merge then false-blocks (the
    fail-closed direction; finish it with `git merge --no-commit` + `git commit`), and `--amend` goes unseen.
  * `git merge --squash` + `git commit`: one commit holding a whole lane is judged against HEAD, so a lane whose
    amendment and data were ordered BLOCKS -- correctly by the rule's own terms (the squash erases that order from
    history), but it is friction to know about. `git rebase` and a clean `cherry-pick` run no pre-commit hook.
  * A prereg created by RENAMING a file whose old name is not a prereg: rename detection is limited to the prereg
    pathspecs, so it reads as ADDED, which is `prereg_before_run`'s scope.
  * A `GIT_INDEX_FILE` pointing OUTSIDE the repo's git dir (a hand-set custom index) is not honoured; the default
    index is read instead. git itself never does this for `commit`, `commit -a` or `commit -- <paths>`.

REPLAY (2026-09-25, re-run after the second fix round at origin/main d98913868: the last 2500 commits, 536 of them
merges; the gate's own `_evaluate` on each commit's real trees, merges judged against every parent as a `git commit`
with MERGE_HEAD would be -- a clean auto-merge is not checked at all, so this is the stricter reading). Truth = a new
amendment/log entry or an edit to a committed amendment's text, landing with run data. 36 commits modified a prereg
while writing under raw/; 19 BLOCK, all true by that definition (TP 19, FP 0), and none of the other 17 is a miss
(FN 0): 10 wrote only `_provenance/runs.jsonl`, 1 is a record subsection (414e1ba4f), 6 append results to the
ORIGINAL registration with no amendment (the declared blind spot above). No merge blocks; judged against HEAD alone,
19 merges would have. The grammar added in this round (qualified, erratum and glued forms) changes no verdict in the
window, and now recognises 4 of the review's 5 latent forms (`Instrument revision` is declared above instead).
WHAT THE 19 ARE (each commit's amendment text and data list read by hand -- a judgement, not a measurement):
0 are GOVERNED-data blocks (runs made under a rule that the same commit's amendment text sets: the ordering failure
the gate exists for). All 19 are RECORDED-data blocks: the data were produced before or beside the amendment and it
cites them as what was seen -- smokes, calibration grids, design sweeps, sizing runs, probe results, a queue-removal
record, provenance and precision corrections (2def39c76 and 6994e80f2 say they were committed BEFORE any run they
govern; 835fc252e completes a registration its own DRAFT said would be chosen from the sizing runs it lands with).
So over this window the gate's whole cost is escape friction, 19 `amendment-same-commit:` lines or split commits in
2500 commits, and its value is preventive. The one real post-launch change in the window, a9eda3d0a's edit to a
governed scorer, was CODE, which no prereg-text gate can see; 4cf8c0237 later disclosed it by amendment.

MUTATION-VERIFY. tests/test_gate_prereg_amendment_order.py::test_selftest_kills_mutants applies each listed mutant
and requires selftest() -- the registry's only trust signal -- to FAIL on it: among them check() returning early on
empty `paths`, the M status filter, the raw filter, the detector unwired, bold entries, GIT_INDEX_FILE, merge
parents, the clean-auto-merge rule, the amend rule and its flag parser, prereg rename detection, the record
exemption (removed, or applied to any label saying `record`), N3, N4, the provenance exclusion, the escape
(removed, or matched on any line), and fail-closed for git errors and an unreadable MERGE_HEAD.
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
_PREREG_SPECS = (":(glob,icase)research/findings/*prereg*.md", ":(glob,icase)docs/plans/*prereg*.md")
_RAW_RE = re.compile(r"^research/findings/raw/.+")
_RAW_SPECS = ("research/findings/raw/",)
_RAW_NOT_DATA_RE = re.compile(r"(?:\.prov\.json$|^research/findings/raw/_provenance/|\.progress_[^/]*$)")
_ESCAPE_RE = re.compile(r"^\s*[-*]?\s*amendment-same-commit:\s*(.{15,})$", re.I)
_GIT_TIMEOUT = 20

# --- amendment-entry grammar --------------------------------------------------------------------------------
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_BOLD_RE = re.compile(r"^(\s*)(?:[-*+]\s+|\d+[.)]\s+)?\*\*\s*(.*)$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_WORD_RE = re.compile(r"^(amendments?|addend(?:um|a)|errat(?:um|a))\b[\s\-]*(.*)$", re.I)
_QUAL_RE = re.compile(r"^((?:[A-Za-z0-9][\w-]*\s+){1,2})(amendment|addendum|erratum)\b[\s\-]*(.*)$", re.I)
_QUAL_STOP = frozenset(
    "a an the this that these those its our my their each every any no which what why how per for in of by to from "
    "on see and or same prior previous next later earlier above below".split())
_PLURAL = frozenset(("amendments", "addenda", "errata"))
_TEMPORAL_RE = re.compile(r"^(?:after|before|filed|written|dated)\b", re.I)
_LOGWORD_RE = re.compile(r"^(?:log|history)\b[\s\-]*(.*)$", re.I)
_CORR_RE = re.compile(r"^correction\b\s*(\d+)?", re.I)
_ID_RE = re.compile(r"^(?:no\.?\s*)?([A-Za-z]?\d+[a-z]?|[A-Z])(?=$|[\s,:;.()—–\-*\]])")
_DELIM_RE = re.compile(r"^(?:$|[(:,;.—–\-*])")
_AFORM_RE = re.compile(r"^A(\d+)(?=\s*(?:$|[,:;(—–]|-{1,2}\s|\.?\*\*))")
_RECORD_RE = re.compile(r"\brecord\b", re.I)
_SECNUM_RE = re.compile(r"^(?:§\s*)?\d+(?:\.\d+)*[.)]?\s+")          # `## 6. AMENDMENT LOG` -> `AMENDMENT LOG`
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
        word, rest = m.group(1).lower(), m.group(2)
    else:
        q = _QUAL_RE.match(s)
        if not q or _QUAL_STOP.intersection(q.group(1).lower().split()):
            q = None
        word, rest = (q.group(2).lower(), q.group(3)) if q else (None, "")
        # a QUALIFIED label that names an ID points AT that amendment (`## Rerun under AMENDMENT-1 (v2): NO-GO`,
        # `### Results under amendment 1`, `**Verdict after amendment 1:**`): a results heading, not an entry. Read as
        # an entry it false-blocked results appended with their data as N2 (review r4, 2026-09-25). A qualified
        # ENTRY in the corpus never carries an ID (`Seed-integrity amendment - <date>`, `v3 amendment (...)`).
        if q and not _DATE_RE.match(rest) and _ID_RE.match(rest):
            word, rest = None, ""
    if word:
        lm = _LOGWORD_RE.match(rest)
        if lm:
            cm = _CORR_RE.match(lm.group(1))
            if cm:
                return "entry", "LOGCORR" + (cm.group(1) or "")
            return "container", None
        if word in _PLURAL:
            return "container", None
        im = None if _DATE_RE.match(rest) else _ID_RE.match(rest)      # `amendment - 2026-08-03`: a date, no ID
        if im:
            return "entry", ("ERR" if word.startswith("errat") else "") + _norm_id(im.group(1))
        if _DELIM_RE.match(rest) or _TEMPORAL_RE.match(rest) or _DATE_RE.match(rest):
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
                if bm:
                    glued = not (prev_blank or _LIST_RE.match(ln))
                    direct = _direct_log(stack)
                    kind, key = _label_kind(bm.group(2), direct)
                    # glued to the line above: an entry only if it names an ID AND a date -- a wrapped reference
                    # (`**AMENDMENT 3** governs ...`) has the same shape and carries neither
                    if glued and not (kind == "entry" and not key.startswith("TEXT:")
                                      and _DATE_RE.search(bm.group(2)[:60])):
                        kind = None
                    # the checklist's AMENDMENT LOG form: a top-level bold entry directly in a log section, e.g.
                    # `- **2026-09-24, filed after ...**` -- the first one, or any later one that carries a date
                    elif (kind is None and direct and len(bm.group(1)) < 2
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


# --- which git command is committing? (the parents depend on it) ------------------------------------------------
_GIT_GLOBAL_WITH_VALUE = frozenset(("-c", "-C", "--git-dir", "--work-tree", "--namespace", "--super-prefix",
                                    "--config-env", "--attr-source"))
# `git commit`'s long options (git 2.55, `git commit --git-completion-helper-all`), each with whether it takes a
# REQUIRED value that may be the next word. parse-options accepts any unambiguous prefix of a name, so `--mess
# --amend` is a message and `--am` is --amend; a name this table lacks is taken as valueless (the next word is then
# read as an option, which can only make an amend visible, never hide one).
_COMMIT_LONG = {
    "ahead-behind": False, "all": False, "allow-empty": False, "allow-empty-message": False, "amend": False,
    "author": True, "branch": False, "cleanup": True, "date": True, "dry-run": False, "edit": False, "file": True,
    "fixup": True, "gpg-sign": False, "include": False, "interactive": False, "inter-hunk-context": True,
    "long": False, "message": True, "null": False, "only": False, "patch": False, "pathspec-file-nul": False,
    "pathspec-from-file": True, "porcelain": False, "post-rewrite": False, "quiet": False, "reedit-message": True,
    "reset-author": False, "reuse-message": True, "short": False, "signoff": False, "squash": True, "status": False,
    "template": True, "trailer": True, "unified": True, "untracked-files": False, "verbose": False, "verify": False,
}
_COMMIT_LONG.update({"no-" + k: False for k in list(_COMMIT_LONG)})
_SHORT_WITH_VALUE = "mFcCtU"


def _resolve_commit_long(name):
    """The `git commit` long option `name` names: itself, or the one option it is an unambiguous prefix of."""
    if name in _COMMIT_LONG:
        return name
    hits = [o for o in _COMMIT_LONG if o.startswith(name)]
    return hits[0] if len(hits) == 1 else None


def _invocation_kind(argv):
    """Pure. 'merge' | 'amend' | 'commit' for the argv of the git process running the hook; None if unreadable.
    git parse-options accepts any unambiguous prefix of a long option: `--am`, `--ame`, `--amen` all mean --amend
    for `git commit`, and a later `--no-amend` cancels it. An option's separate VALUE is skipped (`-m --amend` and
    `--mess --amend` are messages, `--auth --amend` an author), and nothing after `--` is an option."""
    if not argv:
        return None
    base = os.path.basename(argv[0])
    i, sub = 1, None
    if base.startswith("git-"):
        sub = base[4:]
    else:
        while i < len(argv):
            a = argv[i]
            i += 1
            if a in _GIT_GLOBAL_WITH_VALUE:
                i += 1
            elif not a.startswith("-"):
                sub = a
                break
    if sub is None:
        return None
    if sub in ("merge", "pull"):
        return "merge"
    amend = False
    while i < len(argv):
        a = argv[i]
        i += 1
        if a == "--":
            break
        if a.startswith("--"):
            name, eq, _ = a[2:].partition("=")
            opt = _resolve_commit_long(name)
            if opt == "amend":
                amend = True
            elif opt == "no-amend":
                amend = False
            elif opt and _COMMIT_LONG[opt] and not eq:
                i += 1                            # `--message --amend`, `--mess --amend`: the value is the next word
        elif a.startswith("-") and len(a) > 1:
            for j, ch in enumerate(a[1:]):
                if ch in _SHORT_WITH_VALUE:
                    if j == len(a) - 2:
                        i += 1                    # `-qam msg`: the value is the next word
                    break
    return "amend" if amend else "commit"


def _invoking_git_argv(root, start=None):
    """argv of the nearest ancestor `git` process (from `start`, default this process's parent) -- but only if it is
    working in `root`. git chdirs to the work-tree top before running any hook, so a hook's git process always
    matches; a check() run on a scratch repo from INSIDE a real hook (the selftest) does not, and must not borrow the
    outer command. None when there is none, or /proc cannot be read."""
    want = os.path.realpath(root)
    pid = start if start is not None else os.getppid()
    try:
        for _ in range(64):
            if pid <= 1:
                return None
            with open("/proc/%d/comm" % pid, encoding="utf-8", errors="replace") as fh:
                comm = fh.read().strip()
            if comm == "git" or comm.startswith("git-"):
                if os.path.realpath("/proc/%d/cwd" % pid) != want:
                    return None
                with open("/proc/%d/cmdline" % pid, "rb") as fh:
                    return [a.decode("utf-8", "surrogateescape") for a in fh.read().split(b"\0")[:-1]]
            with open("/proc/%d/stat" % pid, encoding="utf-8", errors="replace") as fh:
                pid = int(fh.read().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def _detect_invocation(root):
    return _invocation_kind(_invoking_git_argv(root))


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


def _parents(root, env, kind):
    """The parents the commit being made will have. [] on an unborn branch; None for a clean auto-merge, which is
    not checked (see the docstring: pre-merge-commit runs before git writes MERGE_HEAD)."""
    r = _git(["rev-parse", "-q", "--verify", "HEAD^{commit}"], root, env, ok_codes=(0, 1))
    head = r.stdout.decode().strip()
    if not head:
        return []
    # MERGE_HEAD BEFORE `kind`: `git merge --continue` runs this hook as kind 'merge' with MERGE_HEAD present, and
    # a person may have written anything into that tree while resolving
    mh = _git(["rev-parse", "--path-format=absolute", "--git-path", "MERGE_HEAD"], root, env).stdout.decode().strip()
    if mh and os.path.lexists(mh):
        try:
            with open(mh, encoding="utf-8") as fh:
                merge_heads = [ln.split()[0] for ln in fh if ln.strip()]
        except OSError as e:
            raise _GitReadError("MERGE_HEAD exists but cannot be read: %s" % e)
        return [head] + merge_heads
    if kind == "merge":
        return None
    if kind == "amend":
        return _git(["rev-parse", "HEAD^@"], root, env).stdout.decode().split()
    return [head]


def _changes(root, env, parent, target, specs, renames):
    """{new_path: (status_letter, old_path)} from `parent` to the staged index (target None) or to commit `target`,
    limited to `specs`. Rename detection only where it is asked for (the prereg specs)."""
    args = ["diff", "-z", "--name-status", "-M" if renames else "--no-renames"]
    args += (["--cached", parent] if target is None else [parent, target]) + ["--"] + list(specs)
    toks = _git(args, root, env).stdout.decode("utf-8", "surrogateescape").split("\0")
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


def _evaluate(root, env, parents, target=None):
    """The decision for a commit with these `parents` whose tree is the staged index (target None) or the commit
    `target` (the replay). Raises _GitReadError. One code path for the hook, the selftest and the replay."""
    blobs = {}

    def blob(spec):
        if spec not in blobs:
            blobs[spec] = _git(["cat-file", "blob", spec], root, env).stdout.decode("utf-8", "replace")
        return blobs[spec]

    pre = [_changes(root, env, p, target, _PREREG_SPECS, True) for p in parents]
    modified = [p for p in sorted(set(pre[0]).intersection(*pre[1:]))       # a merge: new relative to EVERY parent
                if _PREREG_RE.match(p) and pre[0][p][0] in ("M", "R") and all(c[p][0] != "D" for c in pre)]
    if not modified:
        return []
    raw = [_changes(root, env, p, target, _RAW_SPECS, False) for p in parents]
    raw_written = sorted(p for p in set(raw[0]).intersection(*raw[1:])
                         if _RAW_RE.match(p) and all(c[p][0] != "D" for c in raw))
    new_ref = ":%s" if target is None else target + ":%s"
    prereg_changes = [(p, blob(new_ref % p), [blob("%s:%s" % (sha, c[p][1])) if c[p][0] != "A" else ""
                                              for sha, c in zip(parents, pre)])
                      for p in modified]
    return _problems(prereg_changes, raw_written)


def check(paths, root=None, invocation=None):
    """`paths` (the registry's --diff-filter=A list) is deliberately NOT used: see the docstring. `invocation`
    ('commit' | 'amend' | 'merge') overrides the /proc detection -- the selftest passes it so a check() on a scratch
    repo does not depend on which real git command happens to be running the hook around it."""
    root = os.path.abspath(root or _ROOT)
    try:
        env = _git_env(root)
        kind = invocation if invocation is not None else _detect_invocation(root)
        parents = _parents(root, env, kind)
        if not parents:
            return []                    # unborn branch / amended root commit: nothing is MODIFIED; None: auto-merge
        return _evaluate(root, env, parents)
    except _GitReadError as e:
        return ["CLASS PRA could not read the commit being made (%s) -- failing CLOSED: this gate cannot say the "
                "commit is clean without reading it. Retry; if git itself is broken, fix that first." % e]


# --- selftest -------------------------------------------------------------------------------------------------
_ST_LOG = "# prereg\n\nthresholds: G1 >= 0.5\n\n## Amendment log\n\n(none at filing)\n"
_ST_NOLOG = "# prereg\n\nthresholds: G1 >= 0.5\n"
_ST_BOLD = "\n**AMENDMENT 1: 2026-01-02, after the seed-7 smoke, before round 2.** G1 is now >= 0.6.\n"
_ST_ESCAPE = "- amendment-same-commit: these files are the seed-7 smoke this amendment records\n"


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
    escaped = bold + _ST_ESCAPE
    criteria = base.replace("thresholds: G1 >= 0.5", "thresholds:\n\n- **A1 (route):** G1 >= 0.5")
    criteria2 = criteria.replace("G1 >= 0.5", "G1 >= 0.5\n- **A2 (two referents):** G2")
    fenced = base.replace("G1 >= 0.5\n", "G1 >= 0.5\n\n```\n## AMENDMENT 9 (inside a code block)\n```\n")
    prose = base.replace("G1 >= 0.5\n", "G1 >= 0.5\n\nsome clarifying prose outside any amendment.\n")
    nolog = _ST_NOLOG

    def fires(new, old, r=raw):
        return bool(_problems([(p, new, [old])], r))

    # FAILING DIRECTION: every real amendment form, beside a data artifact, MUST block
    for label, new in (("heading", heading), ("bold", bold), ("A<n> log entry", aform),
                       ("dated log bullet", logbullet)):
        if not fires(new, base):
            bad.append("did NOT catch a new %s amendment beside a raw data artifact" % label)
    for label, new in (
            ("bold `**Amendment N (...)**` paragraph with no log section",
             "\n**Amendment 1 (2026-01-02; a stricter gate).** G1 >= 0.6\n"),
            ("qualified heading `### Seed-integrity amendment - <date>`",
             "\n### Seed-integrity amendment - 2026-01-02\n\nseed 7 replaced by seed 8.\n"),
            ("qualified bold `**Pre-formal amendment after ...**`",
             "\n**Pre-formal amendment after independent audit.** G1 >= 0.6\n"),
            ("versioned bold `**v3 amendment (...)**`", "\n**v3 amendment (fix round 2): G1 is EXTENDED.** G1b\n"),
            ("`## Erratum (<date>, ...)` heading", "\n## Erratum (2026-01-02, before any round-2 result)\n\nG1 0.6\n"),
            ("bold amendment glued to the line above, with an ID and a date",
             "prose line\n**AMENDMENT 1 (2026-01-02, before round 2).** G1 >= 0.6\n")):
        if not fires(nolog + new, nolog):
            bad.append("did NOT catch a new %s beside data" % label)
    if not fires(rewrite, heading):
        bad.append("did NOT catch the registered body of an existing amendment rewritten beside data (N3)")
    if not fires(base + "\n- 2026-01-02: G1 raised to 0.6 after the smoke\n", base):
        bad.append("did NOT catch a plain (unbolded) bullet added to the amendment log (N4)")
    if not fires(base + "\n## AMENDMENT 7 (record of a new gate)\n\nG1 >= 0.7\n", base):
        bad.append("the record exemption covered a NEW amendment whose label merely says `record` (no AMENDMENT 7 "
                   "exists to be recorded)")
    esc_parent = base + "- amendment-same-commit: an older escape, written for an earlier amendment\n"
    if not fires(esc_parent + _ST_BOLD, esc_parent):
        bad.append("an escape line already in the parent covered a NEW amendment (the escape must be one of the "
                   "commit's own added lines)")
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
    for label, new in (("`### Why amendment 2's statistic is withdrawn`", "\n### Why amendment 2's statistic is withdrawn\n"),
                       ("`**What this amendment does NOT change:**`", "\n**What this amendment does NOT change:** x\n"),
                       ("`### Compute for this amendment`", "\n### Compute for this amendment\n\n2 GPU-h\n")):
        if fires(nolog + new, nolog):
            bad.append("FALSE POSITIVE: a reference to an amendment, %s, read as a new entry" % label)
    glued_ref = heading.replace("thresholds: G1 >= 0.5\n", "thresholds: G1 >= 0.5, see\n**AMENDMENT 1** below.\n")
    if fires(glued_ref, heading):
        bad.append("FALSE POSITIVE: a wrapped `**AMENDMENT 1** ...` reference glued to prose read as an entry")
    # a QUALIFIED label that names an ID points AT the amendment: results appended with their data (review r4)
    for label, add in (("`## Rerun under AMENDMENT-1 (v2, seed 7): NO-GO`",
                        "\n## Rerun under AMENDMENT-1 (v2, seed 7): NO-GO\n\nG1 read 0.4.\n"),
                       ("`### Results under amendment 1`", "\n## Results\n\n### Results under amendment 1\n\nG1 0.4\n"),
                       ("`**Verdict after amendment 1:**`", "\n## Results\n\n**Verdict after amendment 1:** NO-GO.\n")):
        if fires(heading + add, heading):
            bad.append("FALSE POSITIVE: a results label naming an existing amendment, %s, read as an entry for it"
                       % label)
    per = heading.replace("thresholds: G1 >= 0.5\n", "thresholds: G1 >= 0.5\n\n**Per amendment 1, G1 is read at seed 7.**\n")
    if fires(per, heading):
        bad.append("FALSE POSITIVE: `**Per amendment 1, ...**` outside the amendment read as an entry")
    # ...and a stop-word qualifier (`this`, `the`) makes a reference even with NO ID -- pins _QUAL_STOP
    for label, new in (("`**This amendment: ...**`", "\n**This amendment: G1 is read at seed 7.**\n"),
                       ("`**The amendment (2026-01-02)**`", "\n**The amendment (2026-01-02)** is read at seed 7.\n")):
        if fires(nolog + new, nolog):
            bad.append("FALSE POSITIVE: a reference to an amendment, %s, read as a new entry" % label)
    # which git command is committing: the flag parser the parents depend on
    for argv, want in ((["git", "commit", "--amend", "--no-edit"], "amend"),
                       (["/usr/lib/git-core/git", "commit", "-q", "-a", "--amen"], "amend"),
                       (["git", "-c", "commit.gpgsign=false", "-C", "sub", "commit", "--am"], "amend"),
                       (["git", "commit", "-qam", "--amend"], "commit"),
                       (["git", "commit", "-m", "--amend"], "commit"),
                       (["git", "commit", "--message", "--amend"], "commit"),
                       (["git", "commit", "--amend", "--no-amend"], "commit"),
                       (["git", "commit", "-a", "--", "--amend"], "commit"),
                       (["git", "commit", "--author", "--amend <x@y>"], "commit"),
                       (["git", "commit", "--mess", "--amend"], "commit"),
                       (["git", "commit", "--auth", "--amend"], "commit"),
                       (["git", "commit", "-U", "--amend"], "commit"),
                       (["git", "commit", "--message=x", "--amend"], "amend"),
                       (["git", "commit", "--amend", "--no-am"], "commit"),
                       (["git", "commit", "--a", "--amend"], "amend"),
                       (["git", "-c", "merge.ff=false", "merge", "--no-edit", "lane"], "merge"),
                       (["git", "pull", "--no-rebase", ".", "lane"], "merge"),
                       ([], None)):
        if _invocation_kind(argv) != want:
            bad.append("_invocation_kind(%r) = %r, expected %r" % (argv, _invocation_kind(argv), want))


def _selftest_walker(bad, td):
    """_invoking_git_argv against a REAL process tree: a process named `git` (a symlink to /bin/sh), working in td,
    whose child writes its pid. Walking up from that child must find the `git` argv; asked about another root, it
    must find nothing -- the rule that keeps a real hook's git command out of a check() on a scratch repo."""
    if not (os.path.isdir("/proc/self") and os.path.exists("/bin/sh")):
        return
    import time
    fake = os.path.join(td, ".git", "git")
    pidfile = os.path.join(td, ".git", "selftest-child.pid")
    os.symlink("/bin/sh", fake)
    proc = subprocess.Popen([fake, "-c", 'sh -c "echo \\$\\$ > \'%s\'; exec sleep 30"; :' % pidfile,
                             "commit", "--amend"], cwd=td, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    child = None
    try:
        child, deadline = None, time.time() + 5
        while child is None and time.time() < deadline:
            try:
                with open(pidfile, encoding="utf-8") as fh:
                    child = int(fh.read().strip() or 0) or None
            except (OSError, ValueError):
                time.sleep(0.01)
        if child is None:
            bad.append("walker selftest: the fake git process never started its child")
            return
        argv = _invoking_git_argv(td, start=child)
        if _invocation_kind(argv) != "amend":
            bad.append("_invoking_git_argv did not find the ancestor `git commit --amend` process working in the "
                       "repo (got %r)" % (argv,))
        if _invoking_git_argv(os.path.join(td, ".git"), start=child) is not None:
            bad.append("_invoking_git_argv borrowed a git process working in ANOTHER directory")
    finally:
        if child:
            try:
                os.kill(child, 9)
            except OSError:
                pass
        proc.kill()
        proc.wait()
        for pid_path in (fake, pidfile):
            try:
                os.unlink(pid_path)
            except OSError:
                pass


def _selftest_repo(bad):
    """check()-level, in a scratch repo: the wiring the registry actually runs. Every check() names its
    `invocation`, so the answer does not depend on which real git command is running the hook around the selftest."""
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
        if not check([], root=td, invocation="commit"):
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
            if not check([], root=td, invocation="commit"):
                bad.append("check() ignored GIT_INDEX_FILE: an amendment + raw artifact staged in the temporary "
                           "index a `git commit -a` / `commit -- paths` hook sees was not caught")
        finally:
            if saved is None:
                os.environ.pop("GIT_INDEX_FILE", None)
            else:
                os.environ["GIT_INDEX_FILE"] = saved
        if check([], root=td, invocation="commit"):
            bad.append("FALSE POSITIVE: the default index is clean (the change is only in the temporary index)")
        g("checkout", "--", ".")

        # (3) a prereg RENAMED (to another prereg name) in the same commit as its new amendment and data
        pre2 = "research/findings/2026-01-01-st-renamed-PREREG.md"
        g("mv", pre, pre2)
        write(pre2, "\n## AMENDMENT 5 (r)\n", "a")                 # small: the file must stay >50% similar
        write("research/findings/raw/st/s9.json", "{}")
        g("add", "-A")
        if not check([], root=td, invocation="commit"):
            bad.append("did NOT catch an amendment in a RENAMED prereg beside data (prereg rename detection lost)")
        g("reset", "-q", "--hard")

        # (4) a MERGE bringing in an amendment and its data from correctly ordered commits must NOT block
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
        merge_problems = check(["research/findings/raw/st/s42.json"], root=td, invocation="commit")
        if merge_problems:
            bad.append("FALSE POSITIVE on a merge of correctly ordered history: %s" % merge_problems[0][:100])
        # (4b) the same tree at pre-merge-commit time of a CLEAN auto-merge: git has not written MERGE_HEAD yet
        mh = os.path.join(td, ".git", "MERGE_HEAD")
        os.rename(mh, mh + ".st")
        if not check([], root=td, invocation="commit"):
            bad.append("selftest premise broken: the merged tree judged against HEAD alone should block")
        if check([], root=td, invocation="merge"):
            bad.append("FALSE POSITIVE: a clean auto-merge (git merge running pre-merge-commit, no MERGE_HEAD yet) "
                       "was judged against HEAD alone")
        os.rename(mh + ".st", mh)
        # ...but an evil merge that writes a NEW amendment and new data itself is still caught
        write(pre, "\n## AMENDMENT 2 (written in the merge)\n\nG1 >= 0.7\n", "a")
        write("research/findings/raw/st/s43.json", '{"v": 4}')
        g("add", "-A")
        if not check([], root=td, invocation="commit"):
            bad.append("did NOT catch a merge that itself writes a new amendment and new data")
        # ...including when it is finished by `git merge --continue`: the hook's git command is then `merge`, with
        # MERGE_HEAD present. Consulting the command before MERGE_HEAD reads this as a clean auto-merge (fail OPEN)
        if not check([], root=td, invocation="merge"):
            bad.append("check() FAILED OPEN on `git merge --continue`: a merge that writes a new amendment and new "
                       "data, run as `merge` with MERGE_HEAD present, was read as an unchecked clean auto-merge")
        g("merge", "--abort")
        # (4c) an unreadable MERGE_HEAD fails CLOSED -- over a CLEAN index, where judging against HEAD alone would
        # find nothing, so a fail-open cannot hide behind a HEAD-only block
        os.mkdir(mh)
        if not check([], root=td, invocation="commit"):
            bad.append("check() FAILED OPEN: MERGE_HEAD exists but cannot be read, and no problem was returned")
        os.rmdir(mh)
        # (4d) `commit --amend` of a MERGE commit is judged against ALL of its parents (HEAD^@, not HEAD^)
        g("merge", "-q", "--no-ff", "--no-edit", "lane")
        if not _evaluate(td, _git_env(td), [g("rev-parse", "HEAD^1").strip()]):
            bad.append("selftest premise broken: the merge's tree judged against its first parent alone should block")
        if check([], root=td, invocation="amend"):
            bad.append("FALSE POSITIVE: `--amend` of a merge of correctly ordered history was judged against its "
                       "first parent alone")

        # (5) `git commit --amend` REPLACES HEAD: judged against HEAD's parents, not HEAD
        write(pre, "\n## AMENDMENT 3 (committed alone)\n\nG1 >= 0.8\n", "a")
        g("commit", "-q", "-am", "amendment alone")
        write("research/findings/raw/st/s44.json", "{}")
        g("add", "-A")
        if check([], root=td, invocation="commit"):
            bad.append("selftest premise broken: data after a committed amendment should pass as a NEW commit")
        if not check([], root=td, invocation="amend"):
            bad.append("did NOT catch `commit --amend` folding data into the commit that holds a new amendment")
        g("commit", "-q", "-m", "the data, as its own commit")
        write("research/findings/raw/st/s45.json", "{}")
        g("add", "-A")
        if check([], root=td, invocation="amend"):
            bad.append("FALSE POSITIVE: `--amend` of a data-only commit whose amendment was committed before it")

        _selftest_walker(bad, td)

        # (6) FAIL CLOSED: a root whose index cannot be read is a problem, never a silent pass
        broken = os.path.join(td, "broken")
        write("broken/.git", "gitdir: %s\n" % os.path.join(td, "no-such-gitdir"))
        if not check([], root=broken, invocation="commit"):
            bad.append("check() FAILED OPEN: an unreadable repository/index returned no problem")


def selftest():
    """FAILING DIRECTION FIRST in each block, then the no-false-positive cases."""
    bad = []
    _selftest_pure(bad)
    try:
        _selftest_repo(bad)
    except (RuntimeError, OSError, subprocess.SubprocessError, _GitReadError) as e:
        bad.append("selftest scratch repo could not be built: %s" % e)
    return bad


if __name__ == "__main__":
    print("class PRA prereg-amendment-order — run via the registry (tools/gates), no standalone report.")
