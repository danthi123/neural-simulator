"""CLASS HD -- a plan that adds HOST code deciding content, filed without saying so, and rebuilt build after build.

EVIDENCE. The project rule is "brain-based only" (CLAUDE.md non-negotiable 1): host code is legitimate only for the
world, the body and the teacher; a host computation that decides content is at most a declared teaching scaffold for
its neural replacement. 2026-10-16: the owner caught a Stage-2 C1 plan built on hand-written English grammar
templates; Claude retracted it and saved a memory rule. 2026-10-21: the owner caught the same pattern again. The moat
allowlist claim checker (a host filter on the language mouth that reads English with hand-written rules, word lists
and phrase tables) went through EIGHT builds in two days
(research/findings/2026-10-20-moat-allowlist-claim-checker-PREREGISTRATION.md, Amendments 1-7 and VERDICTS), each a
small step from the last; the research gate only asked "how do we fix the filter", never "should this be host code at
all". A memory note did not stop it; this check fails loudly.

THE GATE. A commit that ADDS, MODIFIES or RENAMES a pre-registration (the file set CLASS SMK reads: `*prereg*.md`
directly under research/findings/ or docs/plans/) or a packet (`docs/drafts/**/PACKET.md`,
`research/coordination/*-PACKET.md`) BLOCKS unless, after the commit, the file carries EXACTLY ONE header line
    host-decision: none <reason, >= 20 chars, saying why nothing it builds decides content on the host>
or the scaffold block, the four fields on the lines right after it (same block: no blank line, heading or `---`
between), each present once and non-empty:
    host-decision: scaffold
    decides: <what the host code decides>
    allowed-by: <a CLAUDE.md non-negotiable number, an owner directive date YYYY-MM-DD, or a RULING key of section 6>
    replaced-by: <the neural mechanism, organ or board task that retires it; none / TBD / n/a / empty are refused>
    cap: <integer 1-5> builds
THE CAP (preregs only; a packet has no amendments). A scaffold prereg whose amendment count after the commit is >= its
cap blocks (the prereg is build 1, each amendment one more build) unless the file carries `cap-ruling: <KEY>`, a
Claude-signed RULING of section 6 of the runbook AS MAIN HOLDS IT that cites this prereg's file name (the validator
CLASS SMK uses for `smoke-hurts-ruling`, tools/smoke_first.py). Every cited cap-ruling must validate.
Amendments are counted with CLASS PRA's entry grammar (tools/gates/prereg_amendment_order.py `_Doc`): distinct
amendment IDs, so `## AMENDMENT 6` counts once however many sub-headings reuse its ID; `## AMENDMENT LOG` containers
and errata are not builds; the bold and log-entry forms PRA recognises count too, so `### Amendment 8` or
`**AMENDMENT 8 (...)**` cannot step around the cap.

SCOPE. "Touch it, declare it": grandfathering covers UNTOUCHED files only. A file first added on or after CUTOFF
(2026-10-21; git date and file-name date, the later wins, as CLASS SMK decides "new") and an older file a commit edits
or renames are held to the same header; the date only words the message. No silent passes: a header that cannot be
parsed blocks. Like CLASS SMK, the registry's added-files list is not used: the changed set is read from `git diff
--cached` per parent (renames on); a clean `git merge` is not checked (each side's commits were judged when made); the
runbook is read from main, never from the index, so a commit cannot carry its own ruling. Commit time only: preregs
for measurement lines are already covered at queue time by CLASS SMK.

WHAT IT CANNOT CATCH. A declaration that lies (`none` on a plan whose code does decide content; a `decides:` that
understates it): the gate makes the claim explicit and reviewable, it cannot read the code. An amendment written as
plain prose, or one that reuses an earlier amendment's ID. A ruling signed `(Claude` by another hand (main is the trust
anchor). A plan outside the two scopes (a SPEC.md, a PLAN.md, `PACKET-<x>.md` under docs/drafts). One valid
cap-ruling unlocks every later amendment of that prereg.

    python tools/gates/host_decision_declared.py check <file> ...   judge working-tree files as if a commit touched them
"""
from __future__ import annotations

import os
import re
import sys

NAME = "host_decision_declared"
CLASS_ID = "HD"
BLOCKING = True

CUTOFF = "2026-10-21"
MIN_REASON = 20
CAP_MIN, CAP_MAX = 1, 5

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PACKET_SPECS = (":(icase)docs/drafts/*PACKET.md", ":(icase)research/coordination/*-PACKET.md")
_PACKET_RE = re.compile(r"^(?:docs/drafts/(?:[^/]+/)*PACKET\.md|research/coordination/[^/]*-PACKET\.md)$", re.I)
_FIELDS = ("decides", "allowed-by", "replaced-by", "cap")
_KEYS = ("host-decision", "cap-ruling") + _FIELDS
_BOUNDARY_RE = re.compile(r"^\s*(?:$|#{1,6}\s|-{3,}\s*$|\*{3,}\s*$|```|~~~)")
_PLACEHOLDERS = frozenset(("none", "tbd", "tba", "tbc", "n/a", "na", "todo", "unknown", "nothing", "-", "--", "?",
                           "...", "—", "…"))
_NN_RE = re.compile(r"non[- ]?negotiable\s*(?:#|no\.?\s*)?\d{1,2}\b|\bNN\s*-?\s*\d{1,2}\b", re.I)
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.\-]*")

TEMPLATE = ("`host-decision: none <why nothing it builds decides content on the host, at least %d chars>`, or the scaffold "
            "block `host-decision: scaffold` + `decides: <what the host code decides>` + `allowed-by: <CLAUDE.md "
            "non-negotiable N | owner directive YYYY-MM-DD | RULING KEY>` + `replaced-by: <the neural mechanism, organ "
            "or board task that retires it>` + `cap: <%d-%d> builds`, one field per line right after it"
            % (MIN_REASON, CAP_MIN, CAP_MAX))
_FIELD_HINT = {
    "decides": "`decides: <what the host code decides>`",
    "allowed-by": "`allowed-by: <CLAUDE.md non-negotiable N | owner directive YYYY-MM-DD | RULING KEY>`",
    "replaced-by": "`replaced-by: <the neural mechanism, organ or board task that retires it>`",
    "cap": "`cap: <%d-%d> builds`" % (CAP_MIN, CAP_MAX),
}


def _sf():
    from tools import smoke_first
    return smoke_first


def _pra():
    from tools.gates import prereg_amendment_order as pra
    return pra


def _smk():
    from tools.gates import flip_smoke_first as smk
    return smk


_LINE_RE = None


def _line_re():
    global _LINE_RE
    if _LINE_RE is None:
        _LINE_RE = _sf().key_line_re(_KEYS)
    return _LINE_RE


# --- reading one file -------------------------------------------------------------------------------------------
def kind_of(rel):
    """'prereg' | 'packet' | None for a repo-relative path."""
    if _sf().PREREG_RE.match(rel):
        return "prereg"
    if _PACKET_RE.match(rel):
        return "packet"
    return None


def _declarations(text):
    """(lines, [(index, key, value)]) for every HD key line outside fenced code."""
    lines, out, fence = text.splitlines(), [], False
    fence_re, line_re = _pra()._FENCE_RE, _line_re()
    for i, ln in enumerate(lines):
        if fence_re.match(ln):
            fence = not fence
            continue
        if fence:
            continue
        m = line_re.match(ln)
        if m:
            out.append((i, m.group("key").lower(), m.group("val")))
    return lines, out


def _block(lines, i):
    """(lo, hi): the run of lines around line i up to a blank line, a heading, a rule (`---`) or a fence."""
    lo = hi = i
    while lo > 0 and not _BOUNDARY_RE.match(lines[lo - 1]):
        lo -= 1
    while hi + 1 < len(lines) and not _BOUNDARY_RE.match(lines[hi + 1]):
        hi += 1
    return lo, hi


def _clean(val):
    return val.replace("`", "").replace("**", "").strip().strip("\"'").strip()


def _is_template(v):
    return bool(re.fullmatch(r"<.*>", v.strip(), re.S))


def _first_word(v):
    parts = re.split(r"[\s,;:()\[\]]+", v.strip())
    return parts[0].lower().strip(".'\"*") if parts else ""


def amendments(text):
    """Distinct amendment IDs in `text` (CLASS PRA's entry grammar), errata and log corrections excluded."""
    keys = []
    for e in _pra()._Doc(text).entries:
        k = e["key"]
        if k.startswith(("ERR", "LOGCORR")) or k in keys:
            continue
        keys.append(k)
    return keys


def _allowed_by_ok(v, runbook_fn):
    if _NN_RE.search(v) or _DATE_RE.search(v):
        return True
    entries = _sf().ruling_entries(runbook_fn() or "")
    return any(t in entries for t in _TOKEN_RE.findall(v))


def judge(rel, text, kind, runbook_fn, state_fn=None):
    """Pure. Problems for ONE prereg / packet the commit touches. runbook_fn() -> the runbook as MAIN holds it (called
    only when a RULING must be checked); state_fn() -> a phrase saying why the file is judged (for the messages)."""
    sf = _sf()
    lines, decl = _declarations(text)
    head = [d for d in decl if d[1] == "host-decision"]
    why = state_fn() if (state_fn and len(head) != 1) else ""
    where = "%s (%s%s)" % (rel, kind, ", " + why if why else "")
    if not head:
        return ["%s has no `host-decision:` header. Every plan says up front whether it adds host code that decides "
                "content (CLAUDE.md non-negotiable 1, brain-based only; CLASS HD). Add exactly one: %s." % (where, TEMPLATE)]
    if len(head) > 1:
        return ["%s has %d `host-decision:` lines (lines %s); keep exactly one: %s."
                % (where, len(head), ", ".join(str(d[0] + 1) for d in head), TEMPLATE)]
    i, _, raw = head[0]
    v = _clean(raw)
    lo, hi = _block(lines, i)
    fields = {}
    for j, key, val in decl:
        if key in _FIELDS and lo <= j <= hi:
            fields.setdefault(key, []).append((j, _clean(val)))
    m = re.match(r"^(none|scaffold)\b[\s:;,.\-—–]*(.*)$", v, re.I)
    if not m:
        return ["%s: `host-decision: %s` cannot be parsed (an unparsable header blocks; it is never skipped). Write %s."
                % (rel, raw.strip(), TEMPLATE)]
    if m.group(1).lower() == "none":
        reason = m.group(2).strip()
        probs = []
        if _is_template(reason) or len(reason) < MIN_REASON:
            probs.append("%s: `host-decision: none` needs a reason of at least %d characters saying why nothing it "
                         "builds decides content on the host (got %d: %r). Or, if host code does decide content, "
                         "declare the scaffold block: %s." % (rel, MIN_REASON, 0 if _is_template(reason) else len(reason),
                                                              reason, TEMPLATE))
        if fields:
            probs.append("%s: declares `host-decision: none` but its header block also carries %s: a plan is `none` OR "
                         "a scaffold; if host code decides content, write `host-decision: scaffold`."
                         % (rel, ", ".join("`%s:`" % k for k in _FIELDS if k in fields)))
        return probs
    # --- scaffold
    probs = []
    missing = [k for k in _FIELDS if k not in fields]
    if missing:
        outside = sorted({k for j, k, _ in decl if k in missing and not lo <= j <= hi})
        probs.append("%s: `host-decision: scaffold` (line %d) but its header block lacks %s. Add %s on the lines right "
                     "after it (no blank line between)%s. The full block: %s."
                     % (rel, i + 1, ", ".join("`%s:`" % k for k in missing), " + ".join(_FIELD_HINT[k] for k in missing),
                        ("; %s found outside the block does not count" % ", ".join("`%s:`" % k for k in outside))
                        if outside else "", TEMPLATE))
    vals = {}
    for k in _FIELDS:
        got = fields.get(k, [])
        if len(got) > 1:
            probs.append("%s: `%s:` appears %d times in the header block (lines %s); keep exactly one."
                         % (rel, k, len(got), ", ".join(str(j + 1) for j, _ in got)))
        elif got:
            val = got[0][1]
            if not val or _is_template(val):
                probs.append("%s: `%s:` is empty or still the template placeholder: write %s." % (rel, k, _FIELD_HINT[k]))
            else:
                vals[k] = val
    if "decides" in vals and _first_word(vals["decides"]) in _PLACEHOLDERS:
        probs.append("%s: `decides: %s` names nothing. Say what the host code decides; if it decides nothing, the "
                     "header is `host-decision: none <reason>`." % (rel, vals["decides"]))
    if "replaced-by" in vals and _first_word(vals["replaced-by"]) in _PLACEHOLDERS:
        probs.append("%s: `replaced-by: %s` is refused (none / TBD / n/a / empty name nothing). Name the neural mechanism, "
                     "organ or board task that retires this host code: %s." % (rel, vals["replaced-by"],
                                                                               _FIELD_HINT["replaced-by"]))
    if "allowed-by" in vals and not _allowed_by_ok(vals["allowed-by"], runbook_fn):
        probs.append("%s: `allowed-by: %s` names no directive. Cite a CLAUDE.md non-negotiable number (`CLAUDE.md "
                     "non-negotiable 1`), an owner directive date (YYYY-MM-DD) or the KEY of a Claude-signed RULING in "
                     "section 6 of %s as main holds it." % (rel, vals["allowed-by"], sf.RUNBOOK_REL))
    cap = None
    if "cap" in vals:
        cm = re.match(r"^(\d+)(?![\d.,])\s*(?:builds?\b)?", vals["cap"], re.I)
        if not cm or not CAP_MIN <= int(cm.group(1)) <= CAP_MAX:
            probs.append("%s: `cap: %s` must be an integer %d-%d followed by `builds` (the prereg is build 1, each "
                         "amendment one more): %s." % (rel, vals["cap"], CAP_MIN, CAP_MAX, _FIELD_HINT["cap"]))
        else:
            cap = int(cm.group(1))
    if kind == "prereg" and cap is not None:
        ams = amendments(text)
        if len(ams) >= cap:
            keys = [sf._first_token(val) for _, k, val in decl if k == "cap-ruling"]
            cited = [x for x in keys if x]
            head_msg = ("%s declares `cap: %d builds` and carries %d amendment(s) after this commit (%s): build %d "
                        "exceeds the cap." % (rel, cap, len(ams), ", ".join(a[:24] for a in ams), len(ams) + 1))
            if not cited:
                probs.append("%s Stop and ask whether this should be host code at all (CLAUDE.md non-negotiable 1). "
                             "Another build needs a Claude-signed RULING in section 6 of %s, on main, that cites %s, "
                             "cited here as `cap-ruling: <KEY>`." % (head_msg, sf.RUNBOOK_REL, os.path.basename(rel)))
            else:
                entries = sf.ruling_entries(runbook_fn() or "")
                for key in cited:
                    ok, w = sf._ruling_ok(entries, key, [os.path.basename(rel), rel])
                    if not ok:
                        probs.append("%s `cap-ruling: %s` %s." % (head_msg, key, w))
    return probs


# --- the commit being made --------------------------------------------------------------------------------------
def _state(root, rev, status, hist_rel, rel, kind):
    """The phrase saying why a touched file is judged: new since CUTOFF, or an older file this commit edits."""
    sf = _sf()
    new = status == "A"
    if not new:
        nd = sf.name_date(rel) or ""
        if nd >= CUTOFF:
            new = True
        else:
            hist = sf.file_history(root, rev, hist_rel)
            new = max(hist[-1][1] if hist else "", nd) >= CUTOFF
    if new:
        return "new: first added on or after %s" % CUTOFF
    return ("an already-committed %s this commit %s: touch it, declare it; grandfathering covers untouched files only"
            % (kind, "renames" if status == "R" else "modifies"))


def check(paths, root=None, invocation=None):
    """`paths` (the registry's --diff-filter=A list) is not used: the changed set is read per parent, like CLASS SMK."""
    pra, sf, smk = _pra(), _sf(), _smk()
    root = os.path.abspath(root or _ROOT)
    try:
        env = pra._git_env(root)
        kind = invocation if invocation is not None else pra._detect_invocation(root)
        parents = pra._parents(root, env, kind)
        if not parents:
            return []                    # unborn branch / amended root commit; None: a clean auto-merge, not checked
        specs = tuple(smk._SPECS) + _PACKET_SPECS
        ch = [pra._changes(root, env, p, None, specs, True) for p in parents]
        cand = [p for p in sorted(set(ch[0]).intersection(*ch[1:]))
                if kind_of(p) and all(c[p][0] != "D" for c in ch)]
        if not cand:
            return []
        runbook = {}

        def runbook_fn():
            if "t" not in runbook:
                runbook["t"] = sf.read_runbook(root)          # main's, never the index's or the working tree's
            return runbook["t"]

        probs = []
        for p in cand:
            r = pra._git(["cat-file", "blob", ":%s" % p], root, env, ok_codes=(0, 128))
            text = r.stdout.decode("utf-8", "replace") if r.returncode == 0 else ""
            st, old = ch[0][p]
            k = kind_of(p)
            probs += judge(p, text, k, runbook_fn,
                           state_fn=lambda p=p, st=st, old=old, k=k: _state(root, parents[0], st, old, p, k))
        return probs
    except pra._GitReadError as e:
        return ["CLASS HD could not read the commit being made (%s) -- failing CLOSED. Retry; if git itself is broken, "
                "fix that first." % e]


# --- selftest ---------------------------------------------------------------------------------------------------
_RUNBOOK = ("# runbook\n\n## 5. Idle\n\n- idle-key: not a ruling (Claude)\n\n## 6. RULINGS (x)\n\n"
            "- cap-ok-1021: one more build of p-PREREGISTRATION.md is allowed (Claude 2026-10-21)\n"
            "- unsigned-1021: about p-PREREGISTRATION.md (Hermes)\n"
            "- other-1021: signed but about another-PREREGISTRATION.md (Claude)\n")
_SCAF = ("host-decision: scaffold\ndecides: which rendered sentences the mouth may keep\n"
         "allowed-by: CLAUDE.md non-negotiable 1 (a teaching scaffold)\nreplaced-by: board task #271, the deixis organ\n"
         "cap: %s builds\n")
_NONE = "host-decision: none it registers bars for a spiking organ; no host code chooses content\n"


def _am(n):
    return "".join("\n## AMENDMENT %d (2026-10-21, Claude)\nmore\n" % (k + 1) for k in range(n))


def _pure(bad):
    rb = (lambda: _RUNBOOK)
    p = "research/findings/p-PREREGISTRATION.md"
    cases = [
        ("no header", "# t\n", "prereg", True),
        ("none with a reason", "---\n" + _NONE + "---\n# t\n", "prereg", False),
        ("bold bullet none", "- **host-decision:** none -- it registers bars for a spiking organ only\n", "prereg", False),
        ("none, short reason", "host-decision: none spiking only\n", "prereg", True),
        ("none, template reason", "host-decision: none <why nothing it builds decides content on the host>\n", "prereg",
         True),
        ("none plus scaffold fields", _NONE + "decides: x\n", "prereg", True),
        ("two host-decision lines", _NONE + "\n" + _NONE, "prereg", True),
        ("only in a fence", "```\n" + _NONE + "```\n", "prereg", True),
        ("unparsable value", "host-decision: maybe later, after the build lands\n", "prereg", True),
        ("scaffold, complete", _SCAF % 3, "prereg", False),
        ("scaffold, missing decides", (_SCAF % 3).replace("decides: which rendered sentences the mouth may keep\n", ""),
         "prereg", True),
        ("scaffold, field after a blank line", (_SCAF % 3).replace("cap: 3 builds\n", "\ncap: 3 builds\n"), "prereg",
         True),
        ("scaffold, empty allowed-by", (_SCAF % 3).replace("CLAUDE.md non-negotiable 1 (a teaching scaffold)", ""),
         "prereg", True),
        ("scaffold, allowed-by names nothing", (_SCAF % 3).replace("CLAUDE.md non-negotiable 1", "my judgement"),
         "prereg", True),
        ("scaffold, allowed-by a ruling key", (_SCAF % 3).replace("CLAUDE.md non-negotiable 1", "RULING other-1021"),
         "prereg", False),
        ("scaffold, replaced-by TBD", (_SCAF % 3).replace("board task #271, the deixis organ", "TBD"), "prereg", True),
        ("scaffold, replaced-by n/a", (_SCAF % 3).replace("board task #271, the deixis organ", "n/a"), "prereg", True),
        ("scaffold, decides none", (_SCAF % 3).replace("which rendered", "none of the rendered"), "prereg", True),
        ("scaffold, duplicated decides", (_SCAF % 3).replace("cap:", "decides: more\ncap:"), "prereg", True),
        ("cap 0", _SCAF % 0, "prereg", True),
        ("cap 6", _SCAF % 6, "prereg", True),
        ("cap words", _SCAF % "three", "prereg", True),
        ("cap 3, two amendments", _SCAF % 3 + _am(2), "prereg", False),
        ("cap 3, three amendments", _SCAF % 3 + _am(3), "prereg", True),
        ("cap 3, three amendments, ruling", _SCAF % 3 + _am(3) + "cap-ruling: cap-ok-1021\n", "prereg", False),
        ("cap 3, three amendments, bold forms", _SCAF % 3 + "\n**AMENDMENT 1 (x)** a\n\n### Amendment 2 (x)\nb\n"
         "\n## AMENDMENT 3 (x)\nc\n", "prereg", True),
        ("cap 3, amendment log container only", _SCAF % 3 + "\n## AMENDMENT LOG\n(none)\n" + _am(2), "prereg", False),
        ("cap ruling unsigned", _SCAF % 3 + _am(3) + "cap-ruling: unsigned-1021\n", "prereg", True),
        ("cap ruling about another file", _SCAF % 3 + _am(3) + "cap-ruling: other-1021\n", "prereg", True),
        ("cap ruling from section 5", _SCAF % 3 + _am(3) + "cap-ruling: idle-key\n", "prereg", True),
        ("cap ruling invented", _SCAF % 3 + _am(3) + "cap-ruling: made-up\n", "prereg", True),
        ("packet, amendments past the cap", _SCAF % 1 + _am(3), "packet", False),
        ("packet, no header", "# packet\n", "packet", True),
    ]
    for label, text, kind, should_fail in cases:
        try:
            got = bool(judge(p, text, kind, rb))
        except Exception as e:                                   # noqa: BLE001 -- a raise is a selftest failure
            bad.append("judge case %r RAISED %s: %s" % (label, type(e).__name__, e))
            continue
        if got != should_fail:
            bad.append("judge case %r: expected %s, got %s" % (label, "a problem" if should_fail else "none",
                                                               "a problem" if got else "none"))
    scope = [("research/findings/2026-10-22-x-PREREGISTRATION.md", "prereg"), ("docs/plans/x-prereg.md", "prereg"),
             ("docs/drafts/x/PACKET.md", "packet"), ("docs/drafts/a/b/PACKET.md", "packet"),
             ("research/coordination/x-PACKET.md", "packet"), ("docs/drafts/x/PLAN.md", None),
             ("research/findings/2026-10-22-x-RESULT.md", None), ("research/coordination/x-PACKET-notes.md", None),
             ("research/findings/raw/x/x-PREREG.md", None)]
    for rel, want in scope:
        if kind_of(rel) != want:
            bad.append("kind_of(%r): expected %s, got %s" % (rel, want, kind_of(rel)))


def _repo(bad):
    """The wiring, in a scratch repo with real `git diff --cached`: an old prereg edited without the header blocks, an
    untouched one passes; a new prereg, a new packet under each path, and a scaffold prereg past its cap block; a
    cap-ruling on main passes and one written into the same commit does not; a non-plan file and a clean merge pass."""
    import subprocess
    import tempfile
    pra, sf = _pra(), _sf()
    with tempfile.TemporaryDirectory() as td:
        env0 = pra._stripped_env()

        def g(*a, date=None):
            env = dict(env0)
            if date:
                env["GIT_COMMITTER_DATE"] = env["GIT_AUTHOR_DATE"] = date + "T12:00:00+00:00"
            r = subprocess.run(["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false"] + list(a),
                               cwd=td, env=env, capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                raise RuntimeError("selftest git %s: %s" % (a[:2], r.stderr.strip()[:120]))

        def w(rel, text):
            full = os.path.join(td, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as fh:
                fh.write(text)

        def rm(rel):
            os.remove(os.path.join(td, rel))

        def blocks(label, want=True):
            g("add", "-A")
            try:
                got = check([], root=td, invocation="commit")
            except Exception as e:                               # noqa: BLE001 -- a raise is a selftest failure
                got, want = ["RAISED %s: %s" % (type(e).__name__, e)], not want
            if bool(got) != want:
                bad.append("%s: expected %s, got %s" % (label, "a block" if want else "a pass", got or "a pass"))
            g("reset", "-q")

        if not check([], root=os.path.join(td, "missing"), invocation="commit"):     # git cannot run there
            bad.append("did NOT fail CLOSED when git could not read the commit")
        g("init", "-q", "-b", "main")
        g("config", "user.email", "gate-selftest@example.invalid")
        g("config", "user.name", "gate selftest")
        old = "research/findings/2026-10-01-old-PREREGISTRATION.md"
        w(old, "# an old prereg, before the gate\n")
        w(sf.RUNBOOK_REL, _RUNBOOK)
        g("add", "-A")
        g("commit", "-q", "-m", "base", date="2026-10-01")
        blocks("a clean tree (an untouched old prereg without a header)", want=False)
        w(old, "# an old prereg, before the gate\n\n## AMENDMENT 1\nmore\n")
        blocks("an OLD prereg edited without the header")
        w(old, "# an old prereg, before the gate\n" + _NONE + "\n## AMENDMENT 1\nmore\n")
        blocks("an OLD prereg edited WITH `host-decision: none`", want=False)
        g("checkout", "--", old)
        new = "research/findings/2026-10-22-new-PREREGISTRATION.md"
        w(new, "# new\n")
        blocks("a NEW prereg with no header")
        rm(new)
        for pk in ("docs/drafts/x/PACKET.md", "research/coordination/x-PACKET.md"):
            w(pk, "# a packet\n")
            blocks("a NEW packet at %s with no header" % pk)
            rm(pk)
        w("docs/drafts/x/PLAN.md", "# a plan that is no packet\n")
        blocks("a non-plan markdown file", want=False)
        rm("docs/drafts/x/PLAN.md")
        scaf = "research/findings/2026-10-22-p-PREREGISTRATION.md"
        w(scaf, _SCAF % 2 + _am(2))
        blocks("a scaffold prereg whose amendments reach its cap")
        w(scaf, _SCAF % 2 + _am(2) + "cap-ruling: self-made\n")
        w(sf.RUNBOOK_REL, _RUNBOOK + "- self-made: allows 2026-10-22-p-PREREGISTRATION.md (Claude 2026-10-22)\n")
        blocks("a cap-ruling written into the runbook IN THE SAME COMMIT (main must hold it)")
        g("add", sf.RUNBOOK_REL)
        g("commit", "-q", "-m", "Claude's ruling lands on main", date="2026-10-22")
        blocks("a scaffold prereg past its cap with a cap-ruling on main", want=False)
        w(scaf, "# new\n")
        g("add", "-A")
        if check([], root=td, invocation="merge"):
            bad.append("a clean auto-merge must not be checked")
        g("reset", "-q")


def selftest():
    """FAILING DIRECTION FIRST in each block, then the no-false-positive cases."""
    bad = []
    _pure(bad)
    try:
        _repo(bad)
    except (RuntimeError, OSError, _subprocess_error()) as e:
        bad.append("selftest scratch repo could not be built: %s" % e)
    return bad


def _subprocess_error():
    import subprocess
    return subprocess.SubprocessError


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2 or argv[0] != "check":
        print(__doc__.strip().splitlines()[-1].strip())
        return 0 if not argv else 2
    sf, bad = _sf(), 0
    for f in argv[1:]:
        rel = os.path.relpath(os.path.abspath(f), _ROOT)
        k = kind_of(rel)
        if not k:
            print("skip: %s is neither a prereg nor a packet path" % rel)
            continue
        with open(os.path.join(_ROOT, rel), encoding="utf-8", errors="replace") as fh:
            probs = judge(rel, fh.read(), k, lambda: sf.read_runbook(_ROOT))
        bad += bool(probs)
        print("\n".join("REFUSED: " + x for x in probs) if probs else "ok: %s" % rel)
    return 2 if bad else 0


if __name__ == "__main__":
    sys.path.insert(0, _ROOT)
    sys.exit(main())
