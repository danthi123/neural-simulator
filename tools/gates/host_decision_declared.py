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

SCOPE. Three kinds of plan file, judged when a commit ADDS, MODIFIES, RENAMES, DELETES or moves one out of scope:
  prereg  the file set CLASS SMK reads (`*prereg*.md` directly under research/findings/ or docs/plans/);
  packet  `docs/drafts/**/PACKET*.md` and `research/coordination/*PACKET*.md` (so `PACKET-m2-fixes.md` and
          `x-PACKET-v2.md` count);
  plan    a lane's plan under docs/drafts/**: a file whose name has the token SPEC, PLAN, DESIGN, HANDOFF or
          AMENDMENT (the moat lane's own plan was docs/drafts/moat-allowlist/SPEC.md, edited in 16 commits across
          builds 1-7; RULING c1-learned-1016 put the C1 method plan at docs/drafts/learned-constructions/PLAN.md).
Out of scope, stated: docs/plans/ design and plan documents other than preregs (171 historical files; they order no
build by themselves), WORKLOG/RESULTS files, and a prereg named outside SMK's rule (`*-PRE-REGISTRATION.md`, a
subdirectory of research/findings/): every prereg gate shares that blind spot.

THE HEADER. "Touch it, declare it": a file first added on or after CUTOFF (2026-10-21; git date and file-name date,
the later wins, as CLASS SMK decides "new") and an older file a commit edits or renames carry EXACTLY ONE header line,
up front (in the frontmatter or before the first `##` heading, within the first 60 lines; a line inside a code fence,
an HTML comment or a 4-space indented code block does not count):
    host-decision: none <reason, >= 20 chars, saying why nothing it builds decides content on the host>
or the scaffold block, the four fields on the lines right after it (same block: no blank line, heading or `---`
between), each present once and non-empty:
    host-decision: scaffold
    decides: <what the host code decides>
    allowed-by: <CLAUDE.md non-negotiable 1-12 | an owner directive date YYYY-MM-DD, not after today | the KEY of a
                 Claude-signed RULING of section 6, as the whole field or after the word RULING; a value opening with
                 not / no / never is refused>
    replaced-by: <the neural mechanism, organ or board task that retires it; none / TBD / n/a / later / pending /
                  unknown and the same words in brackets, underscores or dots are refused>
    cap: <integer 1-5> builds
No silent passes: a header that cannot be parsed blocks. `none` is refused outright on KNOWN_SCAFFOLDS, the files
review has already found to plan host code that decides content (the moat allowlist prereg and SPEC).

THE CAP. A scaffold has an identity: its normalized `decides:` text and its normalized `replaced-by:` text. Every
in-scope file at the commit's tree that declares a scaffold sharing either with another is in ONE group (transitively),
so a successor prereg or an amendment FILE for the same host decision cannot start a fresh count. The group's builds
are the sum, over its preregs, of 1 (the prereg) + its amendments, or the highest `build N` any of their headings
names when that is larger; its allowed builds are the smallest `cap:` in the group plus the grants of its cap rulings. A touched file of a group whose builds exceed that BLOCKS (a packet or plan
of the group too: it would order another build). A grant is the line `cap-raise: <prereg file name> +N builds` in the
body of a section-6 RULING of tools/smoke_first.py's runbook AS MAIN HOLDS IT, signed by Claude (a parenthetical whose
first or last `;`/`,` segment begins `Claude` or `cloud Claude`; `(Hermes, relaying what Claude said)` is not),
cited in a group file as `cap-ruling: <KEY>`. Each distinct cited ruling adds its N once; a ruling that HOLDs or
rules NO-GO carries no grant and adds nothing; once grants are needed, every cap-ruling the judged file cites must
carry one. A prereg's amendments are the LARGER of: the distinct entries CLASS PRA's grammar recognises (errata and
log corrections included) plus every amendment/addendum/erratum ID that opens a heading, a `<summary>`, a bold-led
line, a paragraph or a dated list item (`#2`, `[2]`, `II`, `two` read as IDs; a plain line glued to the one above is
a W2-wrapped reference and does not count); and the literal count of headings opening with amendment / addendum /
erratum / revision (an ID reused, or a `## AMENDMENT 3, record` subsection, counts again: title a record
`### Record for amendment 3`; a heading wrapped onto the next line counts once). A scaffold prereg with a code fence
still open at the end of the file blocks (the text after it cannot be read).

HISTORY. For a touched file that existed in a parent (each parent of a merge), against that parent's version: a
scaffold stays a scaffold (`none` blocks); `cap:` only goes down (more builds come from grants); and for a scaffold
prereg the amendment count never drops and no amendment entry CLASS PRA recognises disappears (folding, relabelling
or moving amendments under an `AMENDMENT LOG` blocks; a correction is an erratum, which counts as a build). A plan
whose parent version declared a scaffold, or declared nothing, is never deleted or moved out of scope (`git mv` to
an out-of-scope name included); one that declared `none` may be. There is no escape for these moves: none is needed
by an honest plan, and an escape that a later commit could reuse would reopen them.

WIRING. Like CLASS SMK, the registry's added-files list is not used: the changed set is read from `git diff --cached`
per parent (renames on), the group from `git grep --cached`, the runbook from main (never from the index, so a commit
cannot carry its own ruling). A clean `git merge` (pre-merge-commit runs the registry before MERGE_HEAD exists) is
judged for THE CAP ONLY over the files the merge changes, so two sides that each stay under a cap cannot merge into
a file over it. Commit time only: preregs for measurement lines are already covered at queue time by CLASS SMK.

WHAT IT CANNOT CATCH, AND WHAT REVIEW MUST CHECK. The gate makes the claim explicit and reviewable; it cannot read
the code. (1) A dishonest `none`: Claude's prereg/packet merge review and verify-go read the header against the code
the plan's commits touch and ask "does any host code here choose what the brain says, hears or keeps?"; a `none` on
such a plan is a finding, and the file joins KNOWN_SCAFFOLDS. (2) A `decides:` that understates the host decision.
(3) A successor plan that rewords BOTH `decides:` and `replaced-by:` is a new scaffold to the gate: review compares
a new scaffold header with the lane's earlier plans. (4) A build that amends no prereg (code only, or a packet) is
not counted; the cap counts prereg amendments as the proxy. (5) A ruling written into main's section 6 by another
hand and signed `(Claude ...)`: main is the trust anchor, and agent_merge_policy lets Hermes write
research/coordination/ on main (raised with the owner). (6) An amendment written as plain prose that opens with no
amendment word, or a body rewritten under an unchanged heading. (7) Whether a `#NNN` board task in `replaced-by:`
exists: the board is not in the repository.

    python tools/gates/host_decision_declared.py check <file> ...   judge working-tree files as if a commit touched them
"""
from __future__ import annotations

import datetime
import os
import re
import sys

NAME = "host_decision_declared"
CLASS_ID = "HD"
BLOCKING = True

CUTOFF = "2026-10-21"
MIN_REASON = 20
CAP_MIN, CAP_MAX = 1, 5
HEAD_MAX_LINE = 60               # the header is up front: within this many lines
NN_COUNT = 12                    # CLAUDE.md's non-negotiables; tests pin this against CLAUDE.md
KNOWN_SCAFFOLDS = frozenset((    # review found these plan host code that decides content: `none` is refused on them
    "research/findings/2026-10-20-moat-allowlist-claim-checker-PREREGISTRATION.md",
    "docs/drafts/moat-allowlist/SPEC.md",
))

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PACKET_SPECS = (":(icase)docs/drafts/*PACKET*.md", ":(icase)research/coordination/*PACKET*.md")
_PLAN_SPECS = (":(icase)docs/drafts/*.md",)
_PACKET_RE = re.compile(r"^(?:docs/drafts/(?:[^/]+/)*[^/]*PACKET[^/]*\.md|research/coordination/[^/]*PACKET[^/]*\.md)$",
                        re.I)
_PLAN_RE = re.compile(r"^docs/drafts/(?:[^/]+/)*(?:[^/]*[-_.])?(?:SPEC|PLAN|DESIGN|HANDOFF|AMENDMENT)(?:[-_.][^/]*)?\.md$",
                      re.I)
_FIELDS = ("decides", "allowed-by", "replaced-by", "cap")
_KEYS = ("host-decision", "cap-ruling") + _FIELDS
_BOUNDARY_RE = re.compile(r"^\s*(?:$|#{1,6}\s|-{3,}\s*$|\*{3,}\s*$|```|~~~)")
_SECTION_RE = re.compile(r"^#{2,6}\s")
_INDENT_CODE_RE = re.compile(r"^(?: {4,}|\t)")
_VALUE_RE = re.compile(r"^(none|scaffold)\b[\s:;,.\-—–]*(.*)$", re.I | re.S)
_CAP_RE = re.compile(r"^(\d+)\s*builds?\b\s*(?:\([^)]*\))?\s*[.;]?\s*$", re.I)
_NOTHING_RE = re.compile(r"^(?:none|nothing|no|nil|null|nobody|tb[dac]|t b d|to be (?:decided|determined|defined|done|"
                         r"confirmed|named|written|chosen)|later|pending|unknown|undecided|unclear|unspecified|todo|"
                         r"to do|not yet|n a|na)\b")
_NN_RE = re.compile(r"non[- ]?negotiable\s*(?:#|no\.?\s*)?(\d{1,3})\b|\bNN\s*-?\s*(\d{1,3})\b", re.I)
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_NEGATED_RE = re.compile(r"^(?:not|no|never|without)\b", re.I)
_RULING_WORD_RE = re.compile(r"\bRULING\s+`?([A-Za-z0-9][A-Za-z0-9_.\-]*)", re.I)
_GRANT_RE = re.compile(r"cap-raise:\s*`?(?P<file>[^\s`]+?)`?\s*\+\s*(?P<n>\d+)\s*builds?\b", re.I)
_SIGNER_RE = re.compile(r"^(?:cloud\s+)?Claude\b", re.I)
_HEAD_AM_RE = re.compile(r"^[\W_]*(?:amendment|addendum|erratum|revision)\b(?P<rest>.*)$", re.I)
_LINE_AM_RE = re.compile(r"^\s*(?:(?:[-*+>]|\d+[.)])\s*)*(?:<[^>]*>\s*)*[\W_]{0,8}?(?P<word>amendment|addendum|erratum)"
                         r"\b(?P<rest>.*)$", re.I)
_CONTAINER_RE = re.compile(r"^[\W_]*(?:log|history)\b", re.I)
_LIST_LEAD_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_BOLD_LEAD_RE = re.compile(r"^\s*(?:(?:[-*+>]|\d+[.)])\s*)*(?:\*\*|__)")
_ID_RE = re.compile(r"^[\s\-–—:]*(?:no\.?\s*)?[#\[(]?\s*(?P<id>[A-Za-z]?\d+[A-Za-z]?(?![\w])|[IVXLC]{2,}(?![\w])|one|two|"
                    r"three|four|five|six|seven|eight|nine|ten|eleven|twelve|[A-Za-z](?![\w]))", re.I)
_WORDS = {w: str(i) for i, w in enumerate("one two three four five six seven eight nine ten eleven twelve".split(), 1)}
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
_BUILD_HEAD_RE = re.compile(r"\bbuilds?\s*#?\s*\d{1,3}(?:\s*(?:[-–]|to|and|,)\s*\d{1,3})*\b", re.I)

TEMPLATE = ("`host-decision: none <why nothing it builds decides content on the host, at least %d chars>`, or the scaffold "
            "block `host-decision: scaffold` + `decides: <what the host code decides>` + `allowed-by: <CLAUDE.md "
            "non-negotiable N | owner directive YYYY-MM-DD | RULING KEY>` + `replaced-by: <the neural mechanism, organ "
            "or board task that retires it>` + `cap: <%d-%d> builds`, one field per line right after it, before the "
            "first `##` heading" % (MIN_REASON, CAP_MIN, CAP_MAX))
_FIELD_HINT = {
    "decides": "`decides: <what the host code decides>`",
    "allowed-by": "`allowed-by: <CLAUDE.md non-negotiable N | owner directive YYYY-MM-DD | RULING KEY>`",
    "replaced-by": "`replaced-by: <the neural mechanism, organ or board task that retires it>`",
    "cap": "`cap: <%d-%d> builds`" % (CAP_MIN, CAP_MAX),
}
_GRANT_HINT = ("a Claude-signed RULING in section 6 of the runbook, on main, whose body carries `cap-raise: <prereg file "
               "name> +N builds`, cited here as `cap-ruling: <KEY>`")


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


def _today():
    return datetime.date.today().isoformat()


# --- reading one file -------------------------------------------------------------------------------------------
def kind_of(rel):
    """'prereg' | 'packet' | 'plan' | None for a repo-relative path."""
    if _sf().PREREG_RE.match(rel):
        return "prereg"
    if _PACKET_RE.match(rel):
        return "packet"
    if _PLAN_RE.match(rel):
        return "plan"
    return None


def _specs():
    return tuple(_smk()._SPECS) + _PACKET_SPECS + _PLAN_SPECS


def _scan(text):
    """(lines, decl, first_section, open_fence). decl: [(index, key, value)] for every HD key line that renders as text
    (not in a code fence, an HTML comment or a 4-space indented code block). first_section: the index of the first
    `##`-or-deeper heading outside a fence (len(lines) if none). open_fence: the line of a fence never closed, or None."""
    lines, out = text.splitlines(), []
    fence_re, line_re = _pra()._FENCE_RE, _line_re()
    fence, fence_at, comment, first = False, None, False, None
    for i, ln in enumerate(lines):
        if comment:
            comment = "-->" not in ln
            continue
        if fence_re.match(ln):
            fence = not fence
            fence_at = i if fence else None
            continue
        if fence:
            continue
        if first is None and _SECTION_RE.match(ln):
            first = i
        at = ln.find("<!--")
        if at >= 0 and "-->" not in ln[at:]:
            comment = True
            if not ln[:at].strip():
                continue
        if _INDENT_CODE_RE.match(ln):
            continue
        m = line_re.match(ln)
        if m:
            out.append((i, m.group("key").lower(), m.group("val")))
    return lines, out, (len(lines) if first is None else first), (fence_at if fence else None)


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


def _norm(v):
    return " ".join(re.sub(r"[\W_]+", " ", v.lower()).split())


def _names_nothing(v):
    """True when a field value names nothing: empty after punctuation, or a placeholder word (`(TBD)`, `[n/a]`, `_TBD_`,
    `T.B.D.`, `N / A`, `to be decided`, `later`, ...)."""
    s = _norm(v)
    return not s or bool(_NOTHING_RE.match(s))


def _parse(text):
    """The header of one file, parsed once: state is missing | multiple | unparsable | none | scaffold."""
    lines, decl, first, open_fence = _scan(text)
    head = [d for d in decl if d[1] == "host-decision"]
    h = {"lines": lines, "decl": decl, "first": first, "open_fence": open_fence, "head": head, "state": "missing",
         "fields": {}, "vals": {}, "cap": None, "reason": "", "line": None, "raw": "", "block": None, "rulings": []}
    for _, k, v in decl:
        t = _sf()._first_token(v) if k == "cap-ruling" else ""
        if t and t not in h["rulings"]:
            h["rulings"].append(t)
    if len(head) > 1:
        h["state"] = "multiple"
    if len(head) != 1:
        return h
    i, _, raw = head[0]
    h["line"], h["raw"] = i, raw
    lo, hi = h["block"] = _block(lines, i)
    for j, key, val in decl:
        if key in _FIELDS and lo <= j <= hi:
            h["fields"].setdefault(key, []).append((j, _clean(val)))
    m = _VALUE_RE.match(_clean(raw))
    if not m:
        h["state"] = "unparsable"
        return h
    h["state"], h["reason"] = m.group(1).lower(), m.group(2).strip()
    for k, got in h["fields"].items():
        if len(got) == 1 and got[0][1] and not _is_template(got[0][1]):
            h["vals"][k] = got[0][1]
    cm = _CAP_RE.match(h["vals"].get("cap", ""))
    if cm and CAP_MIN <= int(cm.group(1)) <= CAP_MAX:
        h["cap"] = int(cm.group(1))
    return h


# --- counting builds --------------------------------------------------------------------------------------------
def _id_of(word, rest):
    if _DATE_RE.match(rest.strip(" \t-–—:(")):
        return None                                  # `AMENDMENT - 2026-08-03`: a date, no ID
    m = _ID_RE.match(rest)
    if not m:
        return None
    tok = m.group("id")
    low = tok.lower()
    if low in _WORDS:
        tok = _WORDS[low]
    elif len(tok) > 1 and re.fullmatch(r"[IVXLC]+", tok, re.I):
        vals = [_ROMAN[c] for c in tok.upper()]
        tok = str(sum(-v if k + 1 < len(vals) and v < vals[k + 1] else v for k, v in enumerate(vals)))
    else:
        tok = _pra()._norm_id(tok)
        tok = re.sub(r"^0+(?=\d)", "", tok)
    return ("ERR" if word.lower().startswith("errat") else "") + tok.upper()


def _amend_scan(text):
    """(entries, ids, literal, top). entries: CLASS PRA's distinct entry keys (errata and log corrections included).
    ids: entries plus every amendment ID that opens a line after its decoration, where the line is a heading, a
    `<summary>`, bold-led, opens a paragraph, or is a dated list item (a plain line glued to the one above is a wrapped
    reference: W2 splits long prose lines, so any word can open one). literal: headings opening with an amendment word
    (containers excluded; a reused ID counts again; a heading wrapped onto the next line counts only with an ID). top:
    the highest `build N` a heading names."""
    pra = _pra()
    entries = []
    for e in pra._Doc(text).entries:
        if e["key"] not in entries:
            entries.append(e["key"])
    ids, literal, top, fence, opens, prev_level = list(entries), 0, 0, False, True, 0
    for ln in text.splitlines():
        if pra._FENCE_RE.match(ln):
            fence, opens, prev_level = not fence, True, 0
            continue
        if fence:
            continue
        hm = pra._HEADING_RE.match(ln)
        if hm:
            m = _HEAD_AM_RE.match(pra._SECNUM_RE.sub("", hm.group(2)))
            wrapped = prev_level == len(hm.group(1))
            if m and not _CONTAINER_RE.match(m.group("rest")) and not (wrapped and not _id_of("", m.group("rest"))):
                literal += 1
            for b in _BUILD_HEAD_RE.findall(hm.group(2)):
                top = max([top] + [int(n) for n in re.findall(r"\d+", b)])
        m = _LINE_AM_RE.match(ln)
        if m and not _CONTAINER_RE.match(m.group("rest")):
            listed = bool(_LIST_LEAD_RE.match(ln))
            if (hm or "<summary" in ln[:60].lower() or _BOLD_LEAD_RE.match(ln) or (opens and not listed)
                    or (listed and _DATE_RE.search(ln[:100]))):
                k = _id_of(m.group("word"), m.group("rest"))
                if k and k not in ids:
                    ids.append(k)
        opens = not ln.strip() or bool(hm) or ln.strip() == "---"
        prev_level = len(hm.group(1)) if hm else 0
    return entries, ids, literal, top


def _stable_count(text):
    """The amendment count the HISTORY check compares: PRA's entries, literal headings and `build N` headings only. A
    prose line that opens with an amendment word moves when a paragraph is re-wrapped, so it is left out here."""
    entries, _, literal, top = _amend_scan(text)
    return max(len(entries), literal, top - 1)


def amendments(text):
    """The distinct amendment IDs of `text` (see _amend_scan): errata (`ERR1`) and log corrections count."""
    return _amend_scan(text)[1]


def build_count(text):
    """The number of amendments of a prereg: the largest of its distinct IDs, its literal amendment headings and the
    highest `build N` a heading names minus 1. The prereg itself is build 1, so it has build_count(text) + 1 builds."""
    _, ids, literal, top = _amend_scan(text)
    return max(len(ids), literal, top - 1)


# --- rulings ----------------------------------------------------------------------------------------------------
def _claude_signed(body):
    """A parenthetical whose first or last `;`/`,` segment begins `Claude` or `cloud Claude` (473 of the 475 signed
    section-6 rulings on main at 3351f6b2f; `(Hermes, relaying what Claude said)` is not)."""
    for p in re.findall(r"\(([^()]*)\)", body):
        segs = [s.strip() for s in re.split(r"[;,]", p) if s.strip()]
        if segs and (_SIGNER_RE.match(segs[0]) or _SIGNER_RE.match(segs[-1])):
            return True
    return False


def _entries(runbook_fn):
    return {k: b for k, b in _sf().ruling_entries(runbook_fn() or "").items() if _claude_signed(b)}


def _grant(entries, key, names):
    """(n, why_not) for a cited cap-ruling KEY: the builds its grants give to the files `names` (basenames or paths)."""
    ok, why = _sf()._ruling_ok(entries, key, list(names))
    if not ok:
        return 0, why
    n = 0
    for g in _GRANT_RE.finditer(entries[key]):
        f, k = g.group("file").strip("`'\""), int(g.group("n"))
        if (f in names or os.path.basename(f) in names) and CAP_MIN <= k <= CAP_MAX:
            n += k
    if not n:
        return 0, ("is a RULING about this plan but grants no builds: a cap ruling carries `cap-raise: <prereg file "
                   "name> +N builds` (N %d-%d) in its body; a ruling that holds or rules NO-GO grants none"
                   % (CAP_MIN, CAP_MAX))
    return n, ""


def _allowed_by_ok(v, runbook_fn, today):
    v = v.strip()
    if _NEGATED_RE.match(v):
        return False
    for a, b in _NN_RE.findall(v):
        if 1 <= int(a or b) <= NN_COUNT:
            return True
    for d in _DATE_RE.findall(v):
        try:
            datetime.date.fromisoformat(d)
        except ValueError:
            continue
        if "2026-01-01" <= d <= today:
            return True
    keys = set(_RULING_WORD_RE.findall(v))
    whole = re.sub(r"\s*\(.*\)\s*$", "", v).strip("` ")
    if whole:
        keys.add(whole)
    if not keys:
        return False
    entries = _entries(runbook_fn)
    return any(k in entries for k in keys)


# --- judging ----------------------------------------------------------------------------------------------------
def _header_problems(rel, kind, h, where, runbook_fn, today):
    probs = []
    if h["state"] == "missing":
        return ["%s has no `host-decision:` header (a line in a code fence, an HTML comment or a 4-space indented block "
                "does not count). Every plan says up front whether it adds host code that decides content (CLAUDE.md "
                "non-negotiable 1, brain-based only; CLASS HD). Add exactly one: %s." % (where, TEMPLATE)]
    if h["state"] == "multiple":
        return ["%s has %d `host-decision:` lines (lines %s); keep exactly one: %s."
                % (where, len(h["head"]), ", ".join(str(d[0] + 1) for d in h["head"]), TEMPLATE)]
    if h["state"] == "unparsable":
        return ["%s: `host-decision: %s` cannot be parsed (an unparsable header blocks; it is never skipped). Write %s."
                % (rel, h["raw"].strip(), TEMPLATE)]
    if h["line"] >= min(h["first"], HEAD_MAX_LINE):
        probs.append("%s: the `host-decision:` header is on line %d%s. It belongs up front: in the frontmatter or right "
                     "under the title, before the first `##` heading and within the first %d lines."
                     % (rel, h["line"] + 1, (", after the first `##` heading (line %d)" % (h["first"] + 1))
                        if h["line"] >= h["first"] else "", HEAD_MAX_LINE))
    fields, decl, (lo, hi) = h["fields"], h["decl"], h["block"]
    if h["state"] == "none":
        reason = h["reason"]
        if _is_template(reason) or len(reason) < MIN_REASON:
            probs.append("%s: `host-decision: none` needs a reason of at least %d characters saying why nothing it "
                         "builds decides content on the host (got %d: %r). Or, if host code does decide content, "
                         "declare the scaffold block: %s." % (rel, MIN_REASON, 0 if _is_template(reason) else len(reason),
                                                              reason, TEMPLATE))
        if fields:
            probs.append("%s: declares `host-decision: none` but its header block also carries %s: a plan is `none` OR "
                         "a scaffold; if host code decides content, write `host-decision: scaffold`."
                         % (rel, ", ".join("`%s:`" % k for k in _FIELDS if k in fields)))
        if rel in KNOWN_SCAFFOLDS:
            probs.append("%s is a known host scaffold (review found that it plans host code that decides content; "
                         "KNOWN_SCAFFOLDS in tools/gates/host_decision_declared.py): `none` is refused. Declare the "
                         "scaffold block: %s." % (rel, TEMPLATE))
        return probs
    missing = [k for k in _FIELDS if k not in fields]
    if missing:
        outside = sorted({k for j, k, _ in decl if k in missing and not lo <= j <= hi})
        probs.append("%s: `host-decision: scaffold` (line %d) but its header block lacks %s. Add %s on the lines right "
                     "after it (no blank line, heading or `---` between)%s. The full block: %s."
                     % (rel, h["line"] + 1, ", ".join("`%s:`" % k for k in missing),
                        " + ".join(_FIELD_HINT[k] for k in missing),
                        ("; %s found outside the block does not count" % ", ".join("`%s:`" % k for k in outside))
                        if outside else "", TEMPLATE))
    for k in _FIELDS:
        got = fields.get(k, [])
        if len(got) > 1:
            probs.append("%s: `%s:` appears %d times in the header block (lines %s); keep exactly one."
                         % (rel, k, len(got), ", ".join(str(j + 1) for j, _ in got)))
        elif got and k not in h["vals"]:
            probs.append("%s: `%s:` is empty or still the template placeholder: write %s." % (rel, k, _FIELD_HINT[k]))
    vals = h["vals"]
    if "decides" in vals and _names_nothing(vals["decides"]):
        probs.append("%s: `decides: %s` names nothing. Say what the host code decides; if it decides nothing, the "
                     "header is `host-decision: none <reason>`." % (rel, vals["decides"]))
    if "replaced-by" in vals and _names_nothing(vals["replaced-by"]):
        probs.append("%s: `replaced-by: %s` is refused (none / TBD / n/a / later / pending / empty, bracketed or not, "
                     "name nothing). Name the neural mechanism, organ or board task that retires this host code: %s."
                     % (rel, vals["replaced-by"], _FIELD_HINT["replaced-by"]))
    if "allowed-by" in vals and not _allowed_by_ok(vals["allowed-by"], runbook_fn, today):
        probs.append("%s: `allowed-by: %s` names no directive. Cite a CLAUDE.md non-negotiable 1-%d (`CLAUDE.md "
                     "non-negotiable 1`), an owner directive date (YYYY-MM-DD, not after today) or the KEY of a "
                     "Claude-signed RULING in section 6 of %s as main holds it (the whole field, or after the word "
                     "RULING); a value that opens with not / no / never names no permission."
                     % (rel, vals["allowed-by"], NN_COUNT, _sf().RUNBOOK_REL))
    if "cap" in vals and h["cap"] is None:
        probs.append("%s: `cap: %s` must be an integer %d-%d followed by `builds` (an optional parenthetical after it; "
                     "the prereg is build 1, each amendment one more): %s."
                     % (rel, vals["cap"], CAP_MIN, CAP_MAX, _FIELD_HINT["cap"]))
    if kind == "prereg" and h["open_fence"] is not None:
        probs.append("%s: a code fence opened on line %d is never closed, so the text after it cannot be read and its "
                     "amendments cannot be counted (a scaffold prereg fails closed). Close the fence."
                     % (rel, h["open_fence"] + 1))
    return probs


def _history_problems(rel, kind, text, h, parent_texts):
    probs = []
    for pt in parent_texts:
        ph = _parse(pt)
        if ph["state"] == "scaffold" and h["state"] == "none":
            probs.append("%s: its parent version declares `host-decision: scaffold` and this commit makes it `none`. A "
                         "scaffold plan stays a scaffold (its builds are what the cap counts): keep the scaffold block; "
                         "if the host code was retired, say so in an amendment." % rel)
        if ph["cap"] is not None and h["cap"] is not None and h["cap"] > ph["cap"]:
            probs.append("%s: `cap:` goes up from %d to %d builds in this commit. A cap only goes down; more builds come "
                         "from %s." % (rel, ph["cap"], h["cap"], _GRANT_HINT))
        if kind == "prereg" and h["state"] == "scaffold":
            before, after = _stable_count(pt), _stable_count(text)
            if after < before:
                probs.append("%s: %d amendment(s) in the parent version, %d after this commit. A scaffold prereg's "
                             "amendments are never removed, folded together or moved out of an amendment heading."
                             % (rel, before, after))
            gone = [k for k in _amend_scan(pt)[0] if k not in _amend_scan(text)[0]]
            if gone:
                probs.append("%s: amendment entr%s %s of the parent version %s gone after this commit (removed or "
                             "relabelled). A scaffold prereg's amendment headings are a record: append, never relabel; a "
                             "correction is an erratum (it counts as a build)."
                             % (rel, "y" if len(gone) == 1 else "ies", ", ".join(k[:40] for k in gone),
                                "is" if len(gone) == 1 else "are"))
    return probs


def _cap_problems(rel, kind, text, h, runbook_fn, peers_fn):
    """The group cap: the builds of every scaffold prereg sharing this one's identity against the smallest cap + grants."""
    infos = {}
    for prel, (pkind, ptext) in ((peers_fn() if peers_fn else {}) or {}).items():
        if prel != rel:
            ph = _parse(ptext)
            if ph["state"] == "scaffold":
                infos[prel] = (pkind, ptext, ph)
    infos[rel] = (kind, text, h)
    ident = {r: {(k, _norm(i[2]["vals"][k])) for k in ("decides", "replaced-by")
                 if i[2]["vals"].get(k) and not _names_nothing(i[2]["vals"][k])} for r, i in infos.items()}
    group, todo = {rel}, [rel]
    while todo:
        r = todo.pop()
        for o, ks in ident.items():
            if o not in group and ks & ident[r]:
                group.add(o)
                todo.append(o)
    members = sorted(group)
    caps = [infos[r][2]["cap"] for r in members if infos[r][2]["cap"] is not None]
    if not caps:
        return []
    cap = min(caps)
    counts, top = {}, 0                    # amendments per prereg; the highest `build N` any heading names
    for r in members:
        if infos[r][0] == "prereg":
            _, ids, literal, t = _amend_scan(infos[r][1])
            counts[r], top = max(len(ids), literal), max(top, t)
    builds = max(sum(1 + n for n in counts.values()), top)
    if builds <= cap:
        return []
    names = set()
    for r in members:
        names.update((r, os.path.basename(r)))
    entries = _entries(runbook_fn)
    grants = {}
    for r in members:
        for key in infos[r][2]["rulings"]:
            if key not in grants:
                grants[key] = _grant(entries, key, names)
    granted = sum(n for n, _ in grants.values())
    bad = [(key, grants[key][1]) for key in h["rulings"] if grants[key][1]]
    named = ("; a heading names build %d" % top) if top > sum(1 + n for n in counts.values()) else ""
    if len(members) == 1:
        head = ("%s declares `cap: %d builds` and carries %d amendment(s) after this commit (%s%s): build %d exceeds the "
                "cap" % (rel, cap, counts.get(rel, 0), ", ".join(x[:24] for x in amendments(text)) or "none by ID", named,
                         builds))
    else:
        head = ("%s declares a scaffold shared (same `decides:` or `replaced-by:`) with %s; together they hold %d builds "
                "(%s%s) under the smallest cap, %d: build %d exceeds the cap"
                % (rel, ", ".join(r for r in members if r != rel), builds,
                   "; ".join("%s: 1 + %d amendment(s)" % (os.path.basename(r), n) for r, n in sorted(counts.items()))
                   or "no prereg", named, cap, builds))
    probs = ["%s, and `cap-ruling: %s` %s." % (head, key, why) for key, why in bad]   # every cited one must grant
    if builds > cap + granted and not bad:
        probs.append("%s%s. Stop and ask whether this should be host code at all (CLAUDE.md non-negotiable 1). Another "
                     "build needs %s (citing %s)." % (head, " plus %d granted" % granted if granted else "", _GRANT_HINT,
                                                      os.path.basename(rel)))
    return probs


def judge(rel, text, kind, runbook_fn, state_fn=None, parents=(), peers_fn=None, today=None, cap_only=False):
    """Pure. Problems for ONE plan the commit touches. runbook_fn() -> the runbook as MAIN holds it; state_fn() -> why the
    file is judged (for the messages); parents: the file's text in each parent that has it; peers_fn() -> {path: (kind,
    text)} of the in-scope files at the commit's tree; cap_only: a clean auto-merge, where only the group cap is judged."""
    h = _parse(text)
    if cap_only:
        if h["state"] == "scaffold" and h["cap"] is not None:
            return _cap_problems(rel, kind, text, h, runbook_fn, peers_fn)
        return []
    why = state_fn() if (state_fn and h["state"] in ("missing", "multiple")) else ""
    where = "%s (%s%s)" % (rel, kind, ", " + why if why else "")
    probs = _header_problems(rel, kind, h, where, runbook_fn, today or _today())
    if h["state"] in ("none", "scaffold"):
        probs += _history_problems(rel, kind, text, h, parents)
    if h["state"] == "scaffold" and h["cap"] is not None:
        probs += _cap_problems(rel, kind, text, h, runbook_fn, peers_fn)
    return probs


def judge_removal(rel, parent_texts, kind, dest=None):
    """Problems for a plan this commit deletes, or moves out of scope (to `dest`)."""
    what = ("deletes %s" % rel) if dest is None else ("moves %s out of the plan scope (to %s)" % (rel, dest))
    hs = [_parse(t) for t in parent_texts]
    if rel in KNOWN_SCAFFOLDS or any(h["state"] == "scaffold" for h in hs):
        return ["This commit %s, a %s that declares `host-decision: scaffold` (or a known host scaffold). A scaffold "
                "plan stays where the gate reads it: its builds are what its cap counts, and deleting or moving it "
                "would reset them. Keep it under a prereg, packet or plan path." % (what, kind)]
    if not hs or not all(h["state"] == "none" and len(h["reason"]) >= MIN_REASON for h in hs):
        return ["This commit %s, a %s that never declared its host decision. Touch it, declare it: commit `host-decision: "
                "none <reason>` on it first (a separate commit), then delete or move it; if host code decides content, "
                "it is a scaffold and stays." % (what, kind)]
    return []


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
    pra, sf = _pra(), _sf()
    root = os.path.abspath(root or _ROOT)
    try:
        env = pra._git_env(root)
        kind = invocation if invocation is not None else pra._detect_invocation(root)
        parents = pra._parents(root, env, kind)
        if parents == []:
            return []                    # an unborn branch / amended root commit
        specs = _specs()
        cache = {}

        def blob(spec):                  # a blob the diff or the grep named must be readable, or the gate fails closed
            return pra._git(["cat-file", "blob", spec], root, env).stdout.decode("utf-8", "replace")

        def runbook_fn():
            if "rb" not in cache:
                cache["rb"] = sf.read_runbook(root)          # main's, never the index's or the working tree's
            return cache["rb"]

        def peers_fn():
            if "peers" not in cache:
                r = pra._git(["grep", "--cached", "-z", "-l", "-I", "-i", "-F", "-e", "host-decision", "--"]
                             + list(specs), root, env, ok_codes=(0, 1))
                out = {}
                for rel in r.stdout.decode("utf-8", "surrogateescape").split("\0"):
                    k = kind_of(rel) if rel else None
                    if k:
                        out[rel] = (k, blob(":%s" % rel))
                cache["peers"] = out
            return cache["peers"]

        if parents is None:              # a clean auto-merge: the cap only, over the files the merge changes
            head = pra._git(["rev-parse", "HEAD"], root, env).stdout.decode().strip()
            ch = pra._changes(root, env, head, None, specs, True)
            probs = []
            for p in sorted(ch):
                if kind_of(p) and ch[p][0] != "D":
                    probs += judge(p, blob(":%s" % p), kind_of(p), runbook_fn, peers_fn=peers_fn, cap_only=True)
            return probs
        ch = [pra._changes(root, env, p, None, specs, True) for p in parents]
        probs = []
        for p in sorted(set(ch[0]).intersection(*ch[1:])):
            st, old = ch[0][p]
            k = kind_of(p)
            if all(c[p][0] == "D" for c in ch):
                if k:
                    probs += judge_removal(p, [blob("%s:%s" % (sha, p)) for sha in parents], k)
                continue
            if any(c[p][0] == "D" for c in ch):
                continue
            if not k:
                if st == "R" and kind_of(old):           # `git mv` of a plan to an out-of-scope name
                    probs += judge_removal(old, [blob("%s:%s" % (parents[0], old))], kind_of(old), dest=p)
                continue
            prev = [blob("%s:%s" % (sha, c[p][1])) for sha, c in zip(parents, ch) if c[p][0] != "A"]
            probs += judge(p, blob(":%s" % p), k, runbook_fn,
                           state_fn=lambda p=p, st=st, old=old, k=k: _state(root, parents[0], st, old, p, k),
                           parents=prev, peers_fn=peers_fn)
        return probs
    except pra._GitReadError as e:
        return ["CLASS HD could not read the commit being made (%s) -- failing CLOSED. Retry; if git itself is broken, "
                "fix that first." % e]


# --- selftest ---------------------------------------------------------------------------------------------------
_RUNBOOK = ("# runbook\n\n## 5. Idle\n\n- idle-key: cap-raise: p-PREREGISTRATION.md +1 builds (Claude)\n\n"
            "## 6. RULINGS (x)\n\n"
            "- cap-ok-1021: one more build of p-PREREGISTRATION.md (Claude 2026-10-21)\n"
            "  cap-raise: p-PREREGISTRATION.md +1 builds\n"
            "- hold-1021: HOLD p-PREREGISTRATION.md, no further build (Claude 2026-10-21)\n"
            "- relay-1021: cap-raise: p-PREREGISTRATION.md +3 builds (Hermes, relaying what Claude said)\n"
            "- unsigned-1021: cap-raise: p-PREREGISTRATION.md +1 builds (Hermes)\n"
            "- other-1021: cap-raise: another-PREREGISTRATION.md +1 builds (Claude)\n")
_SCAF = ("host-decision: scaffold\ndecides: which rendered sentences the mouth may keep\n"
         "allowed-by: CLAUDE.md non-negotiable 1 (a teaching scaffold)\nreplaced-by: board task #271, the deixis organ\n"
         "cap: %s builds\n")
_NONE = "host-decision: none it registers bars for a spiking organ; no host code chooses content\n"


def _am(n, start=1):
    return "".join("\n## AMENDMENT %d (2026-10-21, Claude)\nmore\n" % k for k in range(start, start + n))


def _pure(bad):
    rb = (lambda: _RUNBOOK)
    p = "research/findings/p-PREREGISTRATION.md"
    peer = "research/findings/q-PREREGISTRATION.md"
    cases = [
        ("no header", "# t\n", "prereg", {}, True),
        ("none with a reason", "---\n" + _NONE + "---\n# t\n", "prereg", {}, False),
        ("bold bullet none", "- **host-decision:** none -- it registers bars for a spiking organ only\n", "prereg", {},
         False),
        ("none, short reason", "host-decision: none spiking only\n", "prereg", {}, True),
        ("none, template reason", "host-decision: none <why nothing it builds decides content on the host>\n", "prereg",
         {}, True),
        ("none plus scaffold fields", _NONE + "decides: x\n", "prereg", {}, True),
        ("two host-decision lines", _NONE + "\n" + _NONE, "prereg", {}, True),
        ("only in a fence", "```\n" + _NONE + "```\n", "prereg", {}, True),
        ("only in an HTML comment", "<!--\n" + _NONE + "-->\n", "prereg", {}, True),
        ("only in an indented code block", "# t\n\n    " + _NONE, "prereg", {}, True),
        ("after the first section", "# t\n\n## 1. bars\n\n" + _NONE, "prereg", {}, True),
        ("unparsable value", "host-decision: maybe later, after the build lands\n", "prereg", {}, True),
        ("none on a known scaffold", _NONE, "prereg", {"rel": sorted(KNOWN_SCAFFOLDS)[1]}, True),
        ("scaffold, complete", _SCAF % 3, "prereg", {}, False),
        ("scaffold, missing decides", (_SCAF % 3).replace("decides: which rendered sentences the mouth may keep\n", ""),
         "prereg", {}, True),
        ("scaffold, field after a blank line", (_SCAF % 3).replace("cap: 3 builds\n", "\ncap: 3 builds\n"), "prereg", {},
         True),
        ("scaffold, empty allowed-by", (_SCAF % 3).replace("CLAUDE.md non-negotiable 1 (a teaching scaffold)", ""),
         "prereg", {}, True),
        ("scaffold, allowed-by names nothing", (_SCAF % 3).replace("CLAUDE.md non-negotiable 1", "my judgement"),
         "prereg", {}, True),
        ("scaffold, allowed-by non-negotiable 99", (_SCAF % 3).replace("non-negotiable 1", "non-negotiable 99"),
         "prereg", {}, True),
        ("scaffold, allowed-by a future date", (_SCAF % 3).replace("CLAUDE.md non-negotiable 1", "owner 2099-01-01"),
         "prereg", {}, True),
        ("scaffold, allowed-by negated", (_SCAF % 3).replace("CLAUDE.md non-negotiable 1", "NOT non-negotiable 1"),
         "prereg", {}, True),
        ("scaffold, allowed-by a ruling key", (_SCAF % 3).replace("CLAUDE.md non-negotiable 1", "RULING other-1021"),
         "prereg", {}, False),
        ("scaffold, allowed-by a stray key", (_SCAF % 3).replace("CLAUDE.md non-negotiable 1", "my other-1021 note"),
         "prereg", {}, True),
        ("scaffold, replaced-by TBD", (_SCAF % 3).replace("board task #271, the deixis organ", "TBD"), "prereg", {}, True),
        ("scaffold, replaced-by (TBD)", (_SCAF % 3).replace("board task #271, the deixis organ", "(TBD)"), "prereg", {},
         True),
        ("scaffold, replaced-by T.B.D.", (_SCAF % 3).replace("board task #271, the deixis organ", "T.B.D."), "prereg", {},
         True),
        ("scaffold, replaced-by n/a", (_SCAF % 3).replace("board task #271, the deixis organ", "n/a"), "prereg", {}, True),
        ("scaffold, decides none", (_SCAF % 3).replace("which rendered", "none of the rendered"), "prereg", {}, True),
        ("scaffold, decides [n/a]", (_SCAF % 3).replace("which rendered sentences the mouth may keep", "[n/a]"),
         "prereg", {}, True),
        ("scaffold, duplicated decides", (_SCAF % 3).replace("cap:", "decides: more\ncap:"), "prereg", {}, True),
        ("cap 0", _SCAF % 0, "prereg", {}, True),
        ("cap 6", _SCAF % 6, "prereg", {}, True),
        ("cap words", _SCAF % "three", "prereg", {}, True),
        ("cap without builds", (_SCAF % 3).replace("3 builds", "3 weeks"), "prereg", {}, True),
        ("cap 3, two amendments", _SCAF % 3 + _am(2), "prereg", {}, False),
        ("cap 3, three amendments", _SCAF % 3 + _am(3), "prereg", {}, True),
        ("cap 3, three amendments, ruling", _SCAF % 3 + _am(3) + "cap-ruling: cap-ok-1021\n", "prereg", {}, False),
        ("cap 3, four amendments, one +1 ruling", _SCAF % 3 + _am(4) + "cap-ruling: cap-ok-1021\n", "prereg", {}, True),
        ("cap 3, three amendments, bold forms", _SCAF % 3 + "\n**AMENDMENT 1 (x)** a\n\n### Amendment 2 (x)\nb\n"
         "\n## AMENDMENT 3 (x)\nc\n", "prereg", {}, True),
        ("cap 2, a reused amendment ID", _SCAF % 2 + _am(1) + "\n## AMENDMENT 1 (continued)\nx\n", "prereg", {}, True),
        ("cap 2, `## AMENDMENT #2`", _SCAF % 2 + _am(1) + "\n## AMENDMENT #2\nx\n", "prereg", {}, True),
        ("cap 2, an erratum", _SCAF % 2 + _am(1) + "\n## ERRATUM 1 (2026-10-21)\nx\n", "prereg", {}, True),
        ("cap 2, `## Build 3`", _SCAF % 2 + "\n## Build 3 (filed before measurement)\nx\n", "prereg", {}, True),
        ("cap 2, a plain-line amendment", _SCAF % 2 + _am(1) + "\nAmendment 2 (2026-10-21): new rules\n", "prereg", {},
         True),
        ("cap 3, an unclosed fence", _SCAF % 3 + "\n```text\nx\n", "prereg", {}, True),
        ("cap 3, amendment log container only", _SCAF % 3 + "\n## AMENDMENT LOG\n(none)\n" + _am(2), "prereg", {}, False),
        ("cap ruling unsigned", _SCAF % 3 + _am(3) + "cap-ruling: unsigned-1021\n", "prereg", {}, True),
        ("a grant plus a cited HOLD", _SCAF % 3 + _am(3) + "cap-ruling: cap-ok-1021\ncap-ruling: hold-1021\n", "prereg",
         {}, True),
        ("cap ruling relayed by Hermes", _SCAF % 3 + _am(3) + "cap-ruling: relay-1021\n", "prereg", {}, True),
        ("cap ruling that HOLDs (no grant)", _SCAF % 3 + _am(3) + "cap-ruling: hold-1021\n", "prereg", {}, True),
        ("cap ruling about another file", _SCAF % 3 + _am(3) + "cap-ruling: other-1021\n", "prereg", {}, True),
        ("cap ruling from section 5", _SCAF % 3 + _am(3) + "cap-ruling: idle-key\n", "prereg", {}, True),
        ("cap ruling invented", _SCAF % 3 + _am(3) + "cap-ruling: made-up\n", "prereg", {}, True),
        ("packet, amendments past the cap", _SCAF % 1 + _am(3), "packet", {}, False),
        ("packet, no header", "# packet\n", "packet", {}, True),
        ("history: scaffold -> none", _NONE, "prereg", {"parents": [_SCAF % 3]}, True),
        ("history: cap 3 -> 5", _SCAF % 5 + _am(3), "prereg", {"parents": [_SCAF % 3 + _am(2)]}, True),
        ("history: amendments folded", _SCAF % 3 + "\n## AMENDMENT 1 (builds 2-3, folded)\nx\n", "prereg",
         {"parents": [_SCAF % 3 + _am(2)]}, True),
        ("history: an amendment relabelled", _SCAF % 3 + _am(1) + _am(1, 3), "prereg",
         {"parents": [_SCAF % 3 + _am(2)]}, True),
        ("history: a reused-ID heading removed", _SCAF % 3 + _am(1), "prereg",
         {"parents": [_SCAF % 3 + _am(1) + "\n## AMENDMENT 1 (continued)\nx\n"]}, True),
        ("history: one more amendment appended", _SCAF % 3 + _am(2), "prereg", {"parents": [_SCAF % 3 + _am(1)]}, False),
        ("group: a successor prereg resets nothing", _SCAF % 5, "prereg",
         {"peers": {peer: ("prereg", _SCAF % 3 + _am(2))}}, True),
        ("group: an unrelated scaffold is its own group", (_SCAF % 5).replace("which rendered", "which heard").replace(
            "#271, the deixis organ", "#300, a hearing organ"), "prereg", {"peers": {peer: ("prereg", _SCAF % 3 + _am(2))}},
         False),
    ]
    for label, text, kind, kw, should_fail in cases:
        peers = kw.get("peers")
        try:
            got = bool(judge(kw.get("rel", p), text, kind, rb, parents=kw.get("parents", ()),
                             peers_fn=(lambda peers=peers: peers) if peers else None, today="2026-10-21"))
        except Exception as e:                                   # noqa: BLE001 -- a raise is a selftest failure
            bad.append("judge case %r RAISED %s: %s" % (label, type(e).__name__, e))
            continue
        if got != should_fail:
            bad.append("judge case %r: expected %s, got %s" % (label, "a problem" if should_fail else "none",
                                                               "a problem" if got else "none"))
    scope = [("research/findings/2026-10-22-x-PREREGISTRATION.md", "prereg"), ("docs/plans/x-prereg.md", "prereg"),
             ("docs/drafts/x/PACKET.md", "packet"), ("docs/drafts/a/b/PACKET.md", "packet"),
             ("docs/drafts/x/PACKET-m2-fixes.md", "packet"), ("research/coordination/x-PACKET.md", "packet"),
             ("research/coordination/x-PACKET-v2.md", "packet"), ("docs/drafts/x/SPEC.md", "plan"),
             ("docs/drafts/x/PLAN.md", "plan"), ("docs/drafts/x/HANDOFF.md", "plan"), ("docs/drafts/x/WORKLOG.md", None),
             ("research/findings/2026-10-22-x-RESULT.md", None), ("research/findings/raw/x/x-PREREG.md", None),
             ("docs/plans/2026-10-22-x-PLAN.md", None)]
    for rel, want in scope:
        if kind_of(rel) != want:
            bad.append("kind_of(%r): expected %s, got %s" % (rel, want, kind_of(rel)))
    if judge_removal(p, [_SCAF % 3], "prereg") == [] or judge_removal(p, ["# t\n"], "prereg") == []:
        bad.append("judge_removal let a scaffold or an undeclared plan be deleted")
    if judge_removal(p, [_NONE], "prereg"):
        bad.append("judge_removal refused to delete a plan that declared `none`")


def _repo(bad):
    """The wiring, in a scratch repo with real `git diff --cached`: an old prereg edited without the header blocks, an
    untouched one passes; a new prereg, a new packet under each path, and a scaffold prereg past its cap block; a
    cap-ruling on main passes and one written into the same commit does not; history moves block; a successor prereg
    joins its group; deleting or moving a plan out of scope blocks; a clean merge is judged for the cap only."""
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

        def blocks(label, want=True, kind="commit", add=True):
            if add:
                g("add", "-A")
            try:
                got = check([], root=td, invocation=kind)
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
        g("rm", "-q", old)
        blocks("an undeclared prereg DELETED", add=False)
        g("checkout", "HEAD", "--", old)
        os.makedirs(os.path.join(td, "docs/drafts/x"), exist_ok=True)
        g("mv", old, "docs/drafts/x/BARS.md")
        blocks("an undeclared prereg MOVED out of scope", add=False)
        g("reset", "-q", "--hard")
        rm("docs/drafts/x/BARS.md")
        new = "research/findings/2026-10-22-new-PREREGISTRATION.md"
        w(new, "# new\n")
        blocks("a NEW prereg with no header")
        rm(new)
        for pk in ("docs/drafts/x/PACKET.md", "research/coordination/x-PACKET.md", "docs/drafts/x/SPEC.md"):
            w(pk, "# a packet\n")
            blocks("a NEW plan at %s with no header" % pk)
            rm(pk)
        w("docs/drafts/x/WORKLOG.md", "# a log that is no plan\n")
        blocks("a non-plan markdown file", want=False)
        rm("docs/drafts/x/WORKLOG.md")
        scaf = "research/findings/2026-10-22-p-PREREGISTRATION.md"
        w(scaf, _SCAF % 2 + _am(1))
        g("add", "-A")
        g("commit", "-q", "-m", "a scaffold prereg at its cap", date="2026-10-22")
        w(scaf, _NONE + _am(2))
        blocks("a scaffold prereg turned `none` in the commit that adds its next amendment")
        w(scaf, _SCAF % 2 + _am(2))
        blocks("a scaffold prereg whose amendments reach its cap")
        g("checkout", "--", scaf)
        w("research/findings/2026-10-23-p-build3-PREREGISTRATION.md", _SCAF % 5)
        blocks("a successor prereg for the same scaffold (a fresh cap)")
        rm("research/findings/2026-10-23-p-build3-PREREGISTRATION.md")
        w(scaf, _SCAF % 2 + _am(2) + "cap-ruling: self-made\n")
        w(sf.RUNBOOK_REL, _RUNBOOK + "- self-made: allows it (Claude 2026-10-22)\n"
          "  cap-raise: 2026-10-22-p-PREREGISTRATION.md +1 builds\n")
        blocks("a cap-ruling written into the runbook IN THE SAME COMMIT (main must hold it)")
        g("add", sf.RUNBOOK_REL)
        g("commit", "-q", "-m", "Claude's ruling lands on main", date="2026-10-22")
        blocks("a scaffold prereg past its cap with a cap-ruling on main", want=False)
        g("checkout", "--", scaf)
        g("rm", "-q", scaf)
        blocks("a scaffold prereg DELETED", add=False)
        g("reset", "-q", "--hard")
        w(scaf, _SCAF % 2 + _am(1) + "\n## AMENDMENT 1b (2026-10-22)\nside\n")    # a merge result, no MERGE_HEAD
        blocks("a clean auto-merge that takes a scaffold past its cap", kind="merge")
        w(scaf, "# a merge that leaves no scaffold\n")
        blocks("a clean auto-merge of a file with no scaffold", want=False, kind="merge")


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


def _worktree_peers(root):
    out = {}
    for top in ("research/findings", "docs/plans", "docs/drafts", "research/coordination"):
        for dp, _, fs in os.walk(os.path.join(root, top)):
            for f in fs:
                rel = os.path.relpath(os.path.join(dp, f), root)
                k = kind_of(rel)
                if not k:
                    continue
                with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as fh:
                    t = fh.read()
                if "host-decision" in t.lower():
                    out[rel] = (k, t)
    return out


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2 or argv[0] != "check":
        print(__doc__.strip().splitlines()[-1].strip())
        return 0 if not argv else 2
    sf, bad, peers = _sf(), 0, {}
    for f in argv[1:]:
        rel = os.path.relpath(os.path.abspath(f), _ROOT)
        k = kind_of(rel)
        if not k:
            print("skip: %s is neither a prereg, a packet nor a plan path" % rel)
            continue
        if "p" not in peers:
            peers["p"] = _worktree_peers(_ROOT)
        with open(os.path.join(_ROOT, rel), encoding="utf-8", errors="replace") as fh:
            probs = judge(rel, fh.read(), k, lambda: sf.read_runbook(_ROOT), peers_fn=lambda: peers["p"])
        bad += bool(probs)
        print("\n".join("REFUSED: " + x for x in probs) if probs else "ok: %s" % rel)
    return 2 if bad else 0


if __name__ == "__main__":
    sys.path.insert(0, _ROOT)
    sys.exit(main())
