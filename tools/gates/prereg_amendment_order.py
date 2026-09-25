"""CLASS PRA — an AMENDMENT appended to an EXISTING prereg is never checked for run-order, only a brand-new
pre-registration is.

THE GAP (opus review, 2026-09-25, of branch research/slotbinder-gate-latency). `gates/prereg_before_run` (CLASS
PR) only looks at files git reports as ADDED (`--diff-filter=A`): a pre-registration committed together with the
run artifacts it governs. Its own docstring already names the blind spot: "A prereg committed first, then EDITED
after the run (an amendment without an amendment log) -- that is a content judgement; the reviewer's job." That
blind spot is exactly where an amendment lives — every real AMENDMENT in this repo is added by MODIFYING an
existing `*PREREG*.md`/`*PREREGISTRATION*.md` file, which `--diff-filter=A` never sees, so it was never checked
at all. An amendment is a pre-registration in miniature (new thresholds/gates written down before the next run);
the same run-order requirement applies to it, and nothing enforced it.

THE GATE, two parts.

(1) MECHANICAL, always on. If a commit MODIFIES a `research/findings/*prereg*.md` / `docs/plans/*prereg*.md`
file (case-insensitive, same filename pattern `gates/prereg_before_run` uses) and the lines it ADDS to that file
include a new amendment heading — this repo's real heading styles all match `^#{1,6}\\s*amendment\\b`
case-insensitively: `## AMENDMENT LOG`, `## Amendment log`, `## AMENDMENT N (...)`, `## Amendment N, <date>: ...`,
`## Amendment N — ...`, `### AMENDMENT N, smoke record`, `### Amendment N ADDENDUM` — AND the same commit also
stages any `research/findings/raw/**` artifact (excluding `.prov.json` sidecars), BLOCK. The amendment and the
data it governs cannot be ordered by git history when they land in one commit, exactly the reasoning
`prereg_before_run` already applies to a brand-new prereg. Escape: a line added in the SAME commit reading
`amendment-same-commit: <reason, >=15 chars>` (mirrors `prereg_before_run`'s `prereg-same-commit:` escape, but
scoped to this commit's OWN added lines rather than the whole file, so an old amendment's escape cannot silently
cover a new one). Skipped entirely during a merge (`MERGE_HEAD` present): a merge combines two branches' history
and "who wrote the amendment" is not attributable to either side alone — the same reasoning
`prereg_before_run._merged_in_unchanged` applies to ADDED preregs, applied here as a blanket skip since a
modified file's merged content has no single "our side" blob to compare.

(2) PROVENANCE, best-effort, on newly-ADDED raw artifacts only. Even when part (1) passes (the amendment and the
artifact land in DIFFERENT commits, so git's commit order looks right), the artifact's own run could still
predate the amendment it is presented as governed by — `prereg_before_run`'s second declared blind spot. When a
newly-added `research/findings/raw/**` artifact carries a `.prov.json` sidecar with a `git_sha`, and exactly ONE
amendment section (across every `*prereg*.md` in the repo) cites this artifact's path (a `research/findings/raw/
...` token found in that section's body, longest match wins; a tie across sections with different headings is
ambiguous and skipped rather than guessed at), this checks whether that amendment's introducing commit
(identified via `git log -S"<exact heading line>"`, only trusted when it resolves to exactly one commit) is an
ANCESTOR of the artifact's recorded `git_sha` (`git merge-base --is-ancestor`). If it is provably NOT an
ancestor, the artifact's own recorded run predates the amendment text it is cited under — BLOCK. Any ambiguity
(citation matches zero or >1 amendment sections, the introducing commit does not resolve to exactly one SHA, the
sidecar is missing/unparseable, `git_sha` does not resolve to a real commit, or the ancestor check itself
errors) is a SKIP, never a block: this half is a heuristic textual-citation matcher layered on real git history,
and a gate that guesses wrong when unsure is worse than one that says nothing.

WHAT THIS GATE CANNOT CATCH (stated, not hidden).
  * Part (1) needs the exact heading text as MARKDOWN — a prose sentence merely mentioning "amendment" without a
    `#`-prefixed heading line is invisible (deliberately: the corpus's real amendments are always headed).
  * Part (1) is a blanket block, not scoped to which artifact the new amendment actually governs — same coarse
    granularity as `prereg_before_run` itself; a modified prereg's unrelated amendment landing beside an
    unrelated artifact commit still blocks. The fix is the same one `prereg_before_run` offers: split the commit.
  * Part (2)'s citation match is textual (a path string inside the amendment's prose), not semantic — an
    amendment governing a run by describing it in prose without ever naming the artifact path is invisible to
    it, and this is deliberately the CHEAP, reliable half: no attempt is made to parse which run "this amendment
    governs" from unstructured English.
  * Part (2) only ever fires on paths the registry hands this gate as newly ADDED (the same `--diff-filter=A`
    scope every sibling content gate here shares) — an artifact re-staged after being modified, or one that
    never gets re-staged at all once committed, is not re-checked.
  * Neither part follows a prereg living outside the two directories `prereg_before_run` already covers, or
    named without "prereg" in its filename.
  * A merge is skipped ENTIRELY for part (1) (see above) — a genuine same-commit violation smuggled in via a
    merge commit is not caught by this gate (though the ORIGINAL commit on the source branch would have been).

MUTATION-VERIFY. Commenting out the `problems.append(...)` in `_order_problems` (part 1) or forcing
`_provenance_problem` to always `return []` (part 2) makes `selftest()` fail — each half's failing-direction
case has no other way to pass.
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess

NAME = "prereg_amendment_order"
CLASS_ID = "PRA"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PREREG_RE = re.compile(r"^(research/findings|docs/plans)/[^/]*prereg[^/]*\.md$", re.I)
_RAW_RE = re.compile(r"^research/findings/raw/.+")
_AMEND_RE = re.compile(r"^#{1,6}\s*amendment\b", re.I)
_ESCAPE_RE = re.compile(r"^\s*[-*]?\s*amendment-same-commit:\s*(.{15,})$", re.I)
_RAW_TOKEN_RE = re.compile(r"research/findings/raw/[A-Za-z0-9_\-./]+")
_GIT_TIMEOUT = 20

# git subprocess calls must not inherit GIT_DIR/GIT_WORK_TREE/etc: a pre-commit hook sets these to locate the
# REAL repo, and git prefers them over `cwd` — which breaks any selftest that points `root` at a scratch tempdir
# repo of its own. Same fix, same reason, as tools/gates/finding_mechanism_on_main.py's `_git_env`.
_GIT_ENV_STRIP = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                  "GIT_COMMON_DIR", "GIT_CEILING_DIRECTORIES", "GIT_PREFIX")


def _git_env():
    env = dict(os.environ)
    for k in _GIT_ENV_STRIP:
        env.pop(k, None)
    return env


def _git(args, root):
    try:
        return subprocess.run(["git"] + args, cwd=root, env=_git_env(), capture_output=True, text=True,
                              timeout=_GIT_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None


# --- part (1): pure logic, testable with no real git repo ----------------------------------------------------
def _added_lines_from_diff(diff_text):
    out = []
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            out.append(line[1:])
    return out


def _order_problems(modified_preregs, raw_staged, added_lines_by_path):
    """Pure. modified_preregs: [path,...] (git status M, prereg-filename-matching). raw_staged: [path,...]
    staged raw artifacts (any status, not .prov.json). added_lines_by_path: {prereg_path: [added line, ...]}."""
    if not raw_staged:
        return []
    out = []
    for p in modified_preregs:
        lines = added_lines_by_path.get(p, [])
        new_amend = [ln for ln in lines if _AMEND_RE.match(ln.strip())]
        if not new_amend:
            continue
        if any(_ESCAPE_RE.match(ln) for ln in lines):
            continue
        out.append(
            "CLASS PRA %s is MODIFIED, adding a new amendment heading (%r), in the SAME commit as %d raw run "
            "artifact(s) (e.g. %s). The amendment and the data it governs cannot be ordered by git history this "
            "way. Fix: commit the amendment FIRST, then run, then commit the artifacts -- or add a line "
            "`amendment-same-commit: <reason>` among the lines this commit adds to the file."
            % (p, new_amend[0].strip()[:70], len(raw_staged), raw_staged[0])
        )
    return out


# --- part (2): textual citation match + real git ancestry -----------------------------------------------------
def _extract_amendment_sections(text):
    """[(heading_line, section_body), ...]: each amendment heading through the next heading line (any level) or
    EOF, so the section's prose (where an artifact path gets cited) is captured without over-reading into the
    NEXT amendment."""
    lines = text.splitlines()
    heads = [i for i, ln in enumerate(lines) if _AMEND_RE.match(ln.strip())]
    out = []
    for i in heads:
        j = i + 1
        while j < len(lines) and not lines[j].lstrip().startswith("#"):
            j += 1
        out.append((lines[i].strip(), "\n".join(lines[i:j])))
    return out


def _best_citation(sections_by_file, artifact_path):
    """(file, heading) of the single LONGEST research/findings/raw/... token cited in some amendment section
    that is a prefix of `artifact_path` (or equal to it) -- or None if there is no match, or a tied-length
    match across more than one distinct (file, heading) pair (ambiguous -> skip, never guess)."""
    matches = []
    for f, sections in sections_by_file.items():
        for heading, body in sections:
            for tok in _RAW_TOKEN_RE.findall(body):
                tok = tok.rstrip("`.,;:)>]")
                if artifact_path == tok or artifact_path.startswith(tok.rstrip("/") + "/"):
                    matches.append((len(tok), f, heading))
    if not matches:
        return None
    top_len = max(m[0] for m in matches)
    top = {(f, h) for (l, f, h) in matches if l == top_len}
    if len(top) != 1:
        return None
    return next(iter(top))


def _prereg_files(root):
    out = []
    for d in ("research/findings", "docs/plans"):
        base = os.path.join(root, d)
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            rel = d + "/" + name
            if _PREREG_RE.match(rel):
                out.append(rel)
    return sorted(out)


def _sections_by_file(root):
    out = {}
    for f in _prereg_files(root):
        try:
            with open(os.path.join(root, f), encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        sections = _extract_amendment_sections(text)
        if sections:
            out[f] = sections
    return out


def _amendment_commit(root, prereg_path, heading_line):
    """The single commit that introduced `heading_line` into `prereg_path`'s history, or None if it does not
    resolve to EXACTLY one commit (never merged/never found/still ambiguous -> None, so callers skip)."""
    r = _git(["log", "-S" + heading_line, "--pretty=%H", "--", prereg_path], root)
    if r is None or r.returncode != 0:
        return None
    shas = [s for s in r.stdout.splitlines() if s.strip()]
    return shas[0] if len(shas) == 1 else None


def _resolve_commit(root, sha):
    if not sha:
        return None
    r = _git(["rev-parse", "--verify", "-q", str(sha) + "^{commit}"], root)
    if r is None or r.returncode != 0:
        return None
    out = r.stdout.strip()
    return out or None


def _is_ancestor(root, ancestor, descendant):
    """True/False, or None (unknown -> caller must skip, never block on a maybe)."""
    r = _git(["merge-base", "--is-ancestor", ancestor, descendant], root)
    if r is None:
        return None
    if r.returncode == 0:
        return True
    if r.returncode == 1:
        return False
    return None


def _provenance_problem(root, sections_by_file, artifact_path):
    hit = _best_citation(sections_by_file, artifact_path)
    if hit is None:
        return []
    prereg_file, heading = hit
    amend_commit = _amendment_commit(root, prereg_file, heading)
    if not amend_commit:
        return []
    prov_path = os.path.join(root, artifact_path + ".prov.json")
    if not os.path.exists(prov_path):
        return []
    try:
        with open(prov_path, encoding="utf-8", errors="replace") as fh:
            prov = json.load(fh)
    except (OSError, ValueError):
        return []
    sha = prov.get("git_sha") if isinstance(prov, dict) else None
    artifact_commit = _resolve_commit(root, sha) if sha else None
    if not artifact_commit:
        return []
    anc = _is_ancestor(root, amend_commit, artifact_commit)
    if anc is False:
        return ["CLASS PRA %s's provenance (git_sha=%s, resolved %s) is NOT a descendant of %s, the commit that "
                "introduced the amendment heading it is cited under in %s (%r) -- this artifact's own recorded "
                "run predates the amendment it is presented as governed by."
                % (artifact_path, sha, artifact_commit[:9], amend_commit[:9], prereg_file, heading[:70])]
    return []


# --- wiring: real git for the real commit; a `root=` override for the selftest's scratch repo -----------------
def _staged_status(root):
    r = _git(["diff", "--cached", "--name-status"], root)
    if r is None or r.returncode != 0:
        return []
    out = []
    for ln in r.stdout.splitlines():
        parts = ln.split("\t")
        if len(parts) >= 2:
            out.append((parts[0][:1], parts[-1]))
    return out


def _added_lines_for(root, path):
    r = _git(["diff", "--cached", "-U0", "--", path], root)
    if r is None or r.returncode != 0:
        return []
    return _added_lines_from_diff(r.stdout)


def _is_merging(root):
    r = _git(["rev-parse", "-q", "--verify", "MERGE_HEAD"], root)
    return bool(r is not None and r.returncode == 0 and r.stdout.strip())


def check(paths, root=None):
    root = root or _ROOT
    if not paths:
        return []                                                  # commit-scoped: nothing to say about the whole repo
    problems = []

    if not _is_merging(root):
        status = _staged_status(root)
        modified_preregs = [p for st, p in status if st == "M" and _PREREG_RE.match(p)]
        raw_staged = [p for st, p in status if _RAW_RE.match(p) and not p.endswith(".prov.json")]
        added_lines_by_path = {p: _added_lines_for(root, p) for p in modified_preregs}
        problems += _order_problems(modified_preregs, raw_staged, added_lines_by_path)

    added_raw = [p for p in paths if _RAW_RE.match(p) and not p.endswith(".prov.json")
                and os.path.exists(os.path.join(root, p))]
    if added_raw:
        sections_by_file = _sections_by_file(root)
        if sections_by_file:
            for ap in added_raw:
                problems += _provenance_problem(root, sections_by_file, ap)
    return problems


def selftest():
    """FAILING DIRECTION FIRST for both halves, THEN the passing/no-false-positive cases."""
    bad = []

    # --- part (1): pure, no git needed ---
    amend_lines = ["## AMENDMENT 7 (2026-09-25, before any run) -- new thresholds", "some other added prose line"]
    escape_lines = amend_lines + ["- amendment-same-commit: artifacts here are unrelated integrity smokes only"]
    raws = ["research/findings/raw/x/s42.json"]
    p = "research/findings/2026-01-01-x-PREREGISTRATION.md"

    if not _order_problems([p], raws, {p: amend_lines}):
        bad.append("part(1) did NOT catch a modified prereg adding an amendment heading beside a raw artifact")
    if _order_problems([p], raws, {p: escape_lines}):
        bad.append("part(1) FALSE POSITIVE: the amendment-same-commit escape was not honoured")
    if _order_problems([p], raws, {p: []}):
        bad.append("part(1) FALSE POSITIVE on a modified prereg with NO added lines at all")
    if _order_problems([p], raws, {p: ["just some added prose, no heading at all"]}):
        bad.append("part(1) FALSE POSITIVE: modified prereg with NO new amendment heading was still flagged")
    if _order_problems([p], [], {p: amend_lines}):
        bad.append("part(1) FALSE POSITIVE: no raw artifact staged at all")
    if _order_problems([], raws, {}):
        bad.append("part(1) FALSE POSITIVE: no modified prereg at all")

    if not _AMEND_RE.match("## Amendment 2 — A2 production wiring (2026-09-25)"):
        bad.append("part(1) heading regex did not match a real 'Amendment N —' corpus style")
    if not _AMEND_RE.match("### AMENDMENT 6, smoke record (appended after the declared smoke ran)"):
        bad.append("part(1) heading regex did not match a real nested '### AMENDMENT N,' corpus style")
    if _AMEND_RE.match("this line just mentions an amendment in prose, not a heading"):
        bad.append("part(1) heading regex FALSE POSITIVE on non-heading prose")

    # --- part (2): needs a real (scratch) git repo for log -S / merge-base --is-ancestor ---
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        def g(*args):
            r = subprocess.run(["git"] + list(args), cwd=td, env=_git_env(), capture_output=True, text=True,
                               timeout=_GIT_TIMEOUT)
            return r

        g("init", "-q")
        g("config", "user.email", "gate-selftest@example.invalid")
        g("config", "user.name", "gate selftest")

        heading = "## AMENDMENT 1 (selftest, unique-marker-3f9a) -- registered before any run"
        prereg_rel = "research/findings/2026-01-01-selftest-PREREGISTRATION.md"
        prereg_full = os.path.join(td, prereg_rel)
        os.makedirs(os.path.dirname(prereg_full), exist_ok=True)
        with open(prereg_full, "w", encoding="utf-8") as fh:
            fh.write("# prereg\n\nbase text.\n")
        g("add", "-A")
        g("commit", "-q", "-m", "base prereg")

        with open(prereg_full, "a", encoding="utf-8") as fh:
            fh.write("\n%s\n\ngoverns research/findings/raw/selftest/s7.json.\n" % heading)
        g("add", "-A")
        g("commit", "-q", "-m", "add amendment 1")

        # a commit AFTER the amendment -- an artifact recorded at this SHA is a legitimate post-amendment run
        with open(os.path.join(td, "unrelated.txt"), "w", encoding="utf-8") as fh:
            fh.write("later\n")
        g("add", "-A")
        g("commit", "-q", "-m", "later, unrelated commit")
        after_commit = g("rev-parse", "HEAD").stdout.strip()

        before_commit = g("rev-parse", "HEAD~2").stdout.strip()   # the base commit, BEFORE the amendment existed

        sections = _sections_by_file(td)
        if prereg_rel not in sections:
            bad.append("part(2) did not find the amendment section in the scratch prereg at all")

        art_rel = "research/findings/raw/selftest/s7.json"
        art_full = os.path.join(td, art_rel)
        os.makedirs(os.path.dirname(art_full), exist_ok=True)
        with open(art_full, "w", encoding="utf-8") as fh:
            fh.write('{"score": 1}')

        # FAILING DIRECTION: the artifact's recorded git_sha predates the amendment -> MUST be caught
        with open(art_full + ".prov.json", "w", encoding="utf-8") as fh:
            json.dump({"git_sha": before_commit}, fh)
        if not _provenance_problem(td, sections, art_rel):
            bad.append("part(2) did NOT catch an artifact whose recorded git_sha predates the amendment it cites")

        # PASSING: the artifact's recorded git_sha comes AFTER the amendment -> no problem
        with open(art_full + ".prov.json", "w", encoding="utf-8") as fh:
            json.dump({"git_sha": after_commit}, fh)
        if _provenance_problem(td, sections, art_rel):
            bad.append("part(2) FALSE POSITIVE: artifact's git_sha is a real descendant of the amendment commit")

        # PASSING: no citation at all for an unrelated artifact path
        unrelated_rel = "research/findings/raw/selftest/unrelated_never_cited.json"
        if _provenance_problem(td, sections, unrelated_rel):
            bad.append("part(2) FALSE POSITIVE: no amendment cites this artifact path at all")

        # PASSING: sha does not resolve to any real commit -> skip, not a block
        with open(art_full + ".prov.json", "w", encoding="utf-8") as fh:
            json.dump({"git_sha": "0000000"}, fh)
        if _provenance_problem(td, sections, art_rel):
            bad.append("part(2) FALSE POSITIVE: an unresolvable git_sha must be a SKIP, not a block")

    return bad


if __name__ == "__main__":
    print("class PRA prereg-amendment-order — run via the registry (tools/gates), no standalone report.")
