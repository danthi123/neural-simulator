"""CLASS FM — A FINDING CLAIMS A MECHANISM THAT ISN'T ON MAIN.

THE FAILURE IT CLOSES (2026-09-23). `research/findings/2026-09-23-episodic-s100-cupy-repeats-7of7-loadbearing-
robust-core-23.md` landed on `main` claiming robust-core-23 via `BRAIN_EPISODIC_STORE_VERIFY=1` (and
`LB_EPISODIC_DRIVE_PROBE=1`) -- but the branch implementing the store-reliability fix behind that flag
(`research/lbf-fix-episodic-store`) had not yet been merged. `main` had ZERO references to
`BRAIN_EPISODIC_STORE_VERIFY` anywhere in tracked code, so `main` could not reproduce the finding's own
headline: the flag it cites to explain WHY s100 is now clean does not exist yet on the branch the finding was
committed to. (Since fixed by merge `d69e7ddb`, which is why the corpus audit below reads clean today.)

Every existing gate misses this specific shape:
  * `claim_check` traces a finding's NUMBERS to a cited ARTIFACT, never to whether an env flag the artifact
    depended on has any code behind it on the branch being committed.
  * `claim_verdict_consistency` (CVV) checks a claim against its cited artifact's OWN verdict, not against
    whether the MECHANISM the artifact ran under is reachable from `main`.
  * `production_integration` (PI) checks whether a flag is WIRED to the production `/api/brain-chat` default;
    it does not check whether the flag's reader exists in the tree AT ALL.
  * `biology_check` binds a mechanism to a citable SOURCE, not to a citable LINE OF CODE.

WHAT THIS GATE ENFORCES. For a finding (`research/findings/*.md`, excluding `raw/`) that is IN SCOPE -- its
frontmatter declares `status: live` and a positive verdict (a frontmatter `verdict:` field containing a bare
`GO`-shaped token and no NO-GO-family negation, or, when the frontmatter carries no usable single-line verdict,
the same title/`**Status/Verdict**`-line positive-marker scan `gates/single_seed` uses) -- every env flag of the
form `BRAIN_[A-Z0-9_]+` / `LB_[A-Z0-9_]+` the finding cites, either as an assignment (`FLAG=1`) or inside
backticks, must have >= 1 reference in tracked code under `sim/`, `webapp/`, `research/runners/`, `tools/` (a
`git ls-files`-scoped scan of those four directories, falling back to a plain directory walk only when `git` is
unavailable -- e.g. inside this module's own selftest fixtures). A flag with zero references means the finding's
headline mechanism is not reachable from the tree being committed: main cannot reproduce it. The message names
the flag and, via `git log --all -S<flag>` + `git branch -a --contains`, a CANDIDATE unmerged branch to merge
instead of re-deriving the fix.

ESCAPE. A per-line `<!--flag-not-on-main: <reason>-->` comment on the SAME line as the citation clears that one
citation -- for a finding that intentionally documents unmerged research (a design note, a pre-registration, a
plan-shaped finding describing what a branch WILL do). This is a per-line escape, not a per-file one, so a
finding citing five flags cannot clear all five by explaining one.

SCOPE LIMITS, stated rather than hidden:
  * A frontmatter `verdict:` written as a YAML folded/literal block scalar (`verdict: >` / `verdict: |`, 39 of
    the corpus's 290 declared-verdict findings) carries its real text on SUBSEQUENT indented lines this parser
    does not read; those findings fall through to the same title/Status-line scan `single_seed` uses, which can
    miss a positive verdict stated only deep in a multi-line block. Under-inclusive, not over-inclusive: a miss
    here means a finding escapes the gate, never that a clean finding gets flagged.
  * "Reference in tracked code" is presence, not correctness: `os.environ.get("BRAIN_X")` behind a branch that
    never executes still counts, exactly the residual `production_integration` (PI) exists to close on the
    WIRED axis. This gate answers one narrower question: does the reader exist in the tree at all.
  * Wired through the shared registry loop (`tools/githooks/pre-commit` GATE 5, `--diff-filter=A`), this gate
    -- like every sibling CONTENT gate here (`single_seed`, `doc_type`, `claim_verdict_consistency`) -- sees a
    newly ADDED finding pre-commit, not a finding EDITED after the fact to introduce a new flag citation or
    flip its verdict positive. `check()` itself makes no added/modified distinction (it checks whatever `.md`
    paths it is given), so a modified finding IS caught when linted directly (`python -m
    tools.gates.finding_mechanism_on_main <path>`) or via any tool that stages the fuller diff -- the gap is in
    what the hook currently hands the registry, the same documented limit every other content gate accepts
    (retro-firing on a decade of modified legacy findings is the cry-wolf failure they all avoid).
  * `git log --all -S<flag>` / `git branch -a --contains` is best-effort (bounded by timeout, silenced on any
    git failure) -- a missing branch suggestion is not evidence the flag was never merged, only that this
    particular search did not locate it.
"""
from __future__ import annotations

import os
import re
import subprocess

NAME = "finding-mechanism-on-main"
CLASS_ID = "FM"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CODE_DIRS = ("sim", "webapp", "research/runners", "tools")
_CODE_EXTS = (".py", ".sh", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".html")
_MAX_BYTES = 2_000_000
_GIT_TIMEOUT = 30

# --- flag citation extraction -------------------------------------------------------------------------------
# The trailing-char class is deliberately narrower than the leading one (`[A-Z0-9_]*[A-Z0-9]`, not `[A-Z0-9_]+`):
# a match must END in a letter/digit. Without that, "...either `BRAIN_GNW_STOP_TRIGGER_*` flag..." (a real
# corpus hit, 2026-09-23 first audit) captured the garbage token `BRAIN_GNW_STOP_TRIGGER_` -- `\b` sits happily
# between the trailing `_` (word char) and `*` (non-word), so `[A-Z0-9_]+` swallowed the underscore right up to
# the glob star. `_*[A-Z0-9]` backtracks past it, giving the real prefix `BRAIN_GNW_STOP_TRIGGER` instead.
_ASSIGN_RE = re.compile(r"\b((?:BRAIN|LB)_[A-Z0-9_]*[A-Z0-9])\s*=\s*1\b")
_BACKTICK_RE = re.compile(r"`([^`]*)`")
_FLAG_TOKEN_RE = re.compile(r"\b((?:BRAIN|LB)_[A-Z0-9_]*[A-Z0-9])\b")
_ESCAPE_RE = re.compile(r"<!--\s*flag-not-on-main\s*:\s*.+?-->", re.I)

# --- in-scope detection: status: live + a positive verdict ---------------------------------------------------
_BLOCK_SCALAR = (">", "|", ">-", "|-", ">+", "|+")
_NEGATIVE_VERDICT = re.compile(r"NO[-\s_]?GO|REFUTED|RETRACT\w*|\bVOID\b|WITHDRAWN|CONFOUNDED|\bNEGATIVE\b|"
                               r"UNDEFINED|HONEST[ _-]?NEGATIVE")
_GO_WORD = re.compile(r"\bGO\b")
_POS_HEADLINE = re.compile(r"\b(SOLVED|WORKS|CONFIRMED|VALIDATED|BREAKTHROUGH|SURPASSED|CLOSED|SUCCESS)\b")
_NEG_HEADLINE = re.compile(r"⛔|\b(RETRACT\w*|VOID|NO-GO|NEGATIVE|REFUTED|CONFOUNDED|WITHDRAWN|FALSE)\b")
_GO_FALSE_FRIENDS = re.compile(r"NO[-/ ]GO|GO[-/ ]NO[-/ ]GO|GO[- ]gates?", re.I)
_GO_TOKEN = re.compile(r"(?<![A-Za-z-])GO(?![-A-Za-z])")


def _frontmatter(text):
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    fm = {}
    for line in text[3:end].splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            k, _, v = line.partition(":")
            fm[k.strip().lower()] = v.strip().strip('"').strip("'")
    return fm


def _headline_zone(text):
    """Title + Status/Verdict-labelled lines from the head of the BODY (frontmatter excluded) -- the CLAIM
    zone, mirroring gates/single_seed's own scan so the two gates agree on what "asserts positive" means."""
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end > 0:
            body = text[end + 4:]
    out = []
    for line in body.splitlines()[:80]:
        s = line.strip()
        if s.startswith("#") or re.search(r"\*\*(Status|Verdict|Result|Headline)\b", s, re.I):
            out.append(s)
    return "\n".join(out)


def _verdict_field_positive(verdict):
    """True/False from a single-line frontmatter `verdict:` value, or None when it carries no usable signal
    (empty, or a YAML block-scalar indicator whose real text sits on later indented lines this parser skips)."""
    v = verdict.strip()
    if not v or v in _BLOCK_SCALAR:
        return None
    stripped = _NEGATIVE_VERDICT.sub(" ", v)
    if _NEGATIVE_VERDICT.search(v):
        return bool(_GO_WORD.search(stripped))    # "GO (... ) but the run REFUTED X" is not our real case; rare
    return bool(_GO_WORD.search(stripped))


def _in_scope(text):
    fm = _frontmatter(text)
    if fm.get("status", "").lower() != "live":
        return False
    field_verdict = _verdict_field_positive(fm.get("verdict", ""))
    if field_verdict is not None:
        return field_verdict
    zone = _headline_zone(text)
    if _NEG_HEADLINE.search(zone):
        return False
    if _POS_HEADLINE.search(zone):
        return True
    return bool(_GO_TOKEN.search(_GO_FALSE_FRIENDS.sub("", zone)))


def _line_flags(line):
    flags = set(m.group(1) for m in _ASSIGN_RE.finditer(line))
    for bt_m in _BACKTICK_RE.finditer(line):
        bt = bt_m.group(1)
        for m in _FLAG_TOKEN_RE.finditer(bt):
            # A citation immediately followed by `*` names a FAMILY of flags ("BRAIN_FOO_*"), not one literal
            # flag with a code reference to check -- a real corpus hit (`BRAIN_GNW_STOP_TRIGGER_*`) asserting
            # "no code path touches ANY flag in this family" is a different, weaker claim than this gate targets.
            if bt[m.end():m.end() + 1] == "*":
                continue
            flags.add(m.group(1))
    return flags


def _cited_flags(text):
    """flag -> first UNESCAPED line number citing it. A citation on a line carrying the per-line escape is
    skipped for THAT line; the same flag cited again, unescaped, elsewhere still fires."""
    out = {}
    for i, line in enumerate(text.splitlines(), start=1):
        if _ESCAPE_RE.search(line):
            continue
        for f in _line_flags(line):
            out.setdefault(f, i)
    return out


# --- the code corpus ------------------------------------------------------------------------------------------
# git subprocess calls MUST NOT inherit GIT_DIR/GIT_WORK_TREE/etc: the pre-commit hook (and this gate's own
# selftest, run FROM INSIDE that hook by the registry's selftest-first check) sets those to locate the REAL
# repo, and git prefers them over `cwd` for repo discovery. Caught live: `check(paths, root=d)` against a
# selftest tempdir `d` returned `git ls-files` for the REAL repo (non-empty, so the os.walk fallback below
# never ran), then every join of a real path onto `root=d` failed to open -> an EMPTY corpus -> every "flag
# IS in code" calibration case in the selftest below read as a false positive the moment the hook ran it.
_GIT_ENV_STRIP = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                  "GIT_COMMON_DIR", "GIT_CEILING_DIRECTORIES", "GIT_PREFIX")


def _git_env():
    env = dict(os.environ)
    for k in _GIT_ENV_STRIP:
        env.pop(k, None)
    return env


def _tracked_files(root, dirs):
    try:
        r = subprocess.run(["git", "ls-files"] + list(dirs), cwd=root, env=_git_env(),
                           capture_output=True, text=True, timeout=_GIT_TIMEOUT)
        if r.returncode == 0:
            names = [ln.strip().replace("\\", "/") for ln in r.stdout.split("\n") if ln.strip()]
            # SANITY CHECK, not just a clean exit status: `git ls-files` can succeed while resolving against a
            # DIFFERENT repo than `root` (the env leak above). If not even the first returned path exists
            # under `root`, git was not actually scoped to this tree -- fall through to the walk instead of
            # trusting a result from the wrong repository.
            if names and any(os.path.exists(os.path.join(root, n)) for n in names[:20]):
                return names
    except (OSError, subprocess.SubprocessError):
        pass
    # FALLBACK: no git, an empty/failed result, or the sanity check above rejected a wrong-repo result -- walk
    # the filesystem directly. This is what makes the gate's own selftest fixtures (plain tempdirs, never
    # `git init`-ed) exercisable at all, including when the selftest itself runs from inside a git hook.
    out = []
    for d in dirs:
        base = os.path.join(root, d)
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [dn for dn in dirnames if dn not in (".git", "__pycache__", "node_modules", ".venv")]
            for fn in filenames:
                out.append(os.path.relpath(os.path.join(dirpath, fn), root).replace(os.sep, "/"))
    return out


def _code_corpus(root):
    out = []
    for rel in _tracked_files(root, _CODE_DIRS):
        if not rel.endswith(_CODE_EXTS):
            continue
        full = os.path.join(root, rel)
        try:
            if os.path.getsize(full) > _MAX_BYTES:
                continue
            with open(full, encoding="utf-8", errors="replace") as fh:
                out.append((rel, fh.read()))
        except OSError:
            continue
    return out


def _flag_referenced(corpus, flag):
    pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(flag) + r"(?![A-Za-z0-9_])")
    for rel, content in corpus:
        if flag in content and pat.search(content):
            return True, rel
    return False, None


def _suggest_branch(root, flag):
    """A branch name whose history touches `flag`, or None. Best-effort: any git failure -> None, never raised."""
    try:
        r = subprocess.run(["git", "log", "--all", "--pretty=%H", "-S%s" % flag, "--",
                            "sim", "webapp", "research/runners", "tools"],
                           cwd=root, env=_git_env(), capture_output=True, text=True, timeout=_GIT_TIMEOUT)
        if r.returncode != 0:
            return None
        for commit in [c for c in r.stdout.split("\n") if c.strip()][:10]:
            b = subprocess.run(["git", "branch", "-a", "--contains", commit], cwd=root, env=_git_env(),
                               capture_output=True, text=True, timeout=_GIT_TIMEOUT)
            if b.returncode != 0:
                continue
            for ln in b.stdout.split("\n"):
                # `git branch -a --contains` marks the current branch with `*` and a branch checked out in a
                # LINKED WORKTREE with `+` (this repo uses worktrees per agent session -- without stripping
                # `+` too, a hit here rendered as the literal branch name `"+ agent-nav-curriculum"`).
                name = ln.strip().lstrip("*+").strip()
                if name and name not in ("main", "HEAD") and "HEAD ->" not in name:
                    return name
    except (OSError, subprocess.SubprocessError):
        return None
    return None


# --- the check itself -------------------------------------------------------------------------------------
def _candidates(paths, root):
    if paths is None:
        import glob
        return sorted(os.path.relpath(p, root) for p in
                      glob.glob(os.path.join(root, "research", "findings", "*.md")))
    return [p for p in paths if p.replace(os.sep, "/").endswith(".md")
            and "research/findings/" in p.replace(os.sep, "/")
            and "/raw/" not in p.replace(os.sep, "/")]


def check(paths, root=_ROOT):
    if paths is not None and len(paths) == 0:
        return []
    problems = []
    corpus = None
    for rel in _candidates(paths, root):
        full = rel if os.path.isabs(rel) else os.path.join(root, rel)
        try:
            with open(full, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        if not _in_scope(text):
            continue
        flags = _cited_flags(text)
        if not flags:
            continue
        if corpus is None:
            corpus = _code_corpus(root)
        for flag, lineno in sorted(flags.items()):
            ok, hit = _flag_referenced(corpus, flag)
            if ok:
                continue
            branch = _suggest_branch(root, flag)
            msg = ("%s:%d cites env flag `%s` from a `status: live` GO finding, but `%s` has ZERO references in "
                   "tracked code (sim/, webapp/, research/runners/, tools/) -- main cannot reproduce this "
                   "finding's own headline." % (rel, lineno, flag, flag))
            if branch:
                msg += (" Candidate unmerged branch carrying it: `%s` (found via `git log --all -S%s`) -- "
                        "merge it, don't re-derive the fix." % (branch, flag))
            else:
                msg += (" No branch located via `git log --all -S%s` -- check by hand (a squashed/rebased "
                        "commit, or a flag that was renamed)." % flag)
            msg += (" Escape (only for a finding that intentionally documents unmerged research): "
                    "`<!--flag-not-on-main: <reason>-->` on the SAME line as the citation.")
            problems.append(msg)
    return problems


# --- selftest ----------------------------------------------------------------------------------------------
def _write(d, rel, text):
    p = os.path.join(d, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(text)
    return p


_GO_FIXTURE = ("---\nstatus: live\nlane: test\nverdict: GO\n---\n\n"
              "# a GO finding\n\nRun with `BRAIN_MISSING_FLAG=1` and `LB_ALSO_MISSING`.\n")


def selftest():
    """FAILING DIRECTION FIRST (a GO finding citing a flag absent from tracked code MUST be caught), then the
    calibration cases that must stay silent."""
    import tempfile
    bad = []
    with tempfile.TemporaryDirectory() as d:
        f_missing = _write(d, "research/findings/a_missing.md", _GO_FIXTURE)
        probs = check([f_missing], root=d)
        if not any("BRAIN_MISSING_FLAG" in p for p in probs):
            bad.append("MISSED: a GO finding citing a flag with ZERO code references was not flagged: %r" % probs)
        if not any("LB_ALSO_MISSING" in p for p in probs):
            bad.append("MISSED: a second unreferenced flag on the same finding was not (also) flagged: %r" % probs)
        if len(probs) != 2:
            bad.append("expected exactly 2 problems (one per missing flag), got %d: %r" % (len(probs), probs))

        # --- flag PRESENT in tracked code: assignment form ---
        _write(d, "sim/bridge.py", 'if os.environ.get("BRAIN_PRESENT_ASSIGN") == "1":\n    pass\n')
        f_present = _write(d, "research/findings/b_present.md",
                           "---\nstatus: live\nverdict: GO\n---\n\n# ok\n\nSet `BRAIN_PRESENT_ASSIGN=1`.\n")
        if check([f_present], root=d):
            bad.append("FALSE POSITIVE: a flag WITH a tracked-code reference (assignment form) was flagged: %r"
                      % check([f_present], root=d))

        # --- flag PRESENT in tracked code: backtick form, referenced via .get(...) with no literal '=1' ---
        _write(d, "tools/foo_runner.sh", 'export BRAIN_PRESENT_BACKTICK=1\n')
        f_present2 = _write(d, "research/findings/c_present2.md",
                            "---\nstatus: live\nverdict: GO\n---\n\n# ok\n\nSee `BRAIN_PRESENT_BACKTICK`.\n")
        if check([f_present2], root=d):
            bad.append("FALSE POSITIVE: a backtick-cited flag present in code was flagged: %r"
                      % check([f_present2], root=d))

        # --- REGRESSION PIN: inherited GIT_DIR/GIT_WORK_TREE must not leak into `git ls-files -C root` ---
        # Caught live (2026-09-23): the pre-commit hook that RUNS this very selftest sets GIT_DIR/GIT_WORK_TREE
        # to the real repo, and git prefers those over `cwd` for repo discovery -- so `git ls-files` against
        # this tempdir silently returned the REAL repo's file list instead of failing, every real path failed
        # to open under `root=d`, the corpus came back empty, and the "flag IS in code" case just above read
        # as a false positive the moment this selftest ran from inside the hook rather than standalone.
        _prev_env = {k: os.environ.get(k) for k in _GIT_ENV_STRIP}
        try:
            os.environ["GIT_DIR"] = os.path.join(_ROOT, ".git")
            os.environ["GIT_WORK_TREE"] = _ROOT
            leaked = check([f_present2], root=d)
            if leaked:
                bad.append("REGRESSED: a real GIT_DIR/GIT_WORK_TREE in the environment leaked into the "
                          "tempdir's git scan (env not stripped): %r" % leaked)
        finally:
            for k, v in _prev_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        # --- the escape ---
        f_escaped = _write(d, "research/findings/d_escaped.md",
                           "---\nstatus: live\nverdict: GO\n---\n\n# ok\n\n"
                           "Uses `BRAIN_UNMERGED_ON_PURPOSE=1` <!--flag-not-on-main: pre-registration, branch "
                           "research/xyz not yet merged-->\n")
        if check([f_escaped], root=d):
            bad.append("FALSE POSITIVE: an escaped citation was still flagged: %r" % check([f_escaped], root=d))

        # --- out of scope: status not live ---
        f_super = _write(d, "research/findings/e_superseded.md",
                         "---\nstatus: superseded\nverdict: GO\n---\n\n# was a GO\n\n`BRAIN_MISSING_FLAG2=1`\n")
        if check([f_super], root=d):
            bad.append("FALSE POSITIVE: a superseded finding was checked: %r" % check([f_super], root=d))

        # --- out of scope: verdict NO-GO ---
        f_nogo = _write(d, "research/findings/f_nogo.md",
                        "---\nstatus: live\nverdict: NO-GO\n---\n\n# a negative\n\n`BRAIN_MISSING_FLAG3=1`\n")
        if check([f_nogo], root=d):
            bad.append("FALSE POSITIVE: a NO-GO verdict finding was checked: %r" % check([f_nogo], root=d))

        # --- out of scope: no frontmatter status at all (legacy) ---
        f_legacy = _write(d, "research/findings/g_legacy.md", "# a legacy finding\n\n`BRAIN_MISSING_FLAG4=1`\n")
        if check([f_legacy], root=d):
            bad.append("FALSE POSITIVE: a finding with no frontmatter status was checked: %r" % check([f_legacy], root=d))

        # --- headline-fallback scope: block-scalar verdict, positive TITLE marker ---
        f_block = _write(d, "research/findings/h_block.md",
                         "---\nstatus: live\nverdict: >\n  a longer explanation on later lines\n---\n\n"
                         "# the mechanism WORKS end to end\n\nUses `BRAIN_MISSING_FLAG5=1`.\n")
        probs_block = check([f_block], root=d)
        if not any("BRAIN_MISSING_FLAG5" in p for p in probs_block):
            bad.append("MISSED: a block-scalar-verdict finding with a positive TITLE marker was not checked: %r"
                      % probs_block)

        # --- non-finding / raw / empty-list scoping ---
        if check([]) != []:
            bad.append("paths=[] (nothing staged of our kind) must return [] immediately: %r" % check([]))
        f_raw = _write(d, "research/findings/raw/i_raw.json", "{}")
        if check([f_raw], root=d):
            bad.append("FALSE POSITIVE: a raw/ artifact path was treated as a finding: %r" % check([f_raw], root=d))

        # --- standalone/audit mode (paths=None) scans the corpus and finds the same missing-flag finding ---
        audit = check(None, root=d)
        if not any("BRAIN_MISSING_FLAG" in p for p in audit):
            bad.append("standalone audit mode (paths=None) did not scan the corpus: %r" % audit)
    return bad


if __name__ == "__main__":
    hits = check(None)
    print("CLASS FM corpus audit: %d GO/live finding(s) cite an env flag absent from tracked code" % len(hits))
    for h in hits:
        print("  ⛔", h)
