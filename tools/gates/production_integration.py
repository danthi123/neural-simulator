"""CLASS PI — A GO THAT IS NOT WIRED INTO THE PRODUCTION DEFAULT, CLAIMED AS IF IT WERE.

THE DEFECT, named by the owner 2026-08-11. Every existing discipline scores a MECHANISM (is it spiking? is the
de-risk a GO?). NONE scored whether a GO is reachable + ON-BY-DEFAULT in the production `/api/brain-chat` turn with its
host scaffold retired. So ~40 spiking faculties were banked "GO/closed/integrated" while the live chat ran a HOST
pipeline: `QuestionRouter.match_fact` (keyword-overlap) chooses WHAT to say, a numpy VSA composer recalls it, and an
off-bridge Qwen transformer speaks it — the three acts that define the brain (CHOOSE, GENERATE, LEARN) all host-or-absent
(code-traced: research/findings/2026-08-11-PRODUCTION-chat-pipeline-is-largely-HOST-not-one-brain-spiking-code-traced-honest-inventory.md).
The drift was invisible because the words "wired / one-brain / integrated / on-by-default / production" in a finding are
CLAIMS THAT TOUCH NO CHECK.

WHAT THIS GATE ENFORCES. A declared-truth ledger (docs/PRODUCTION_INTEGRATION_LEDGER.yaml) whose every row's levels
carry machine-checkable anchors into live source, plus a claim<->ledger link. Three sub-checks:

  A  LEDGER <-> SOURCE (anti-lying-ledger). For each row's default_anchor the gate reads the RHS literal at `assign` and
     DERIVES the expected value FROM the on_by_default LEVEL (NO => the OFF literal must be present at the declared count,
     the ON literal absent; YES => the reverse). For scaffold_symbol, the host symbol must be PRESENT iff
     scaffold_retired=NO. So flipping the source (enable_neural_render False->True) WITHOUT moving the row blocks; and
     hand-editing the row to on_by_default:YES while the source literal stays False ALSO blocks. The expected literal is
     derived from the level, never a separable hand-set `expect` (the adversary's theatre-hole fix).

  B  CLAIM <-> LEDGER (anti-overclaim). A staged doc containing a production-integration CLAIM TOKEN ("wired into
     production", "on by default", "integrated into /api/brain-chat", "one brain in production", "scaffold retired", ...)
     must declare `integration_faculty: <key>` in frontmatter, and that row's levels must SUPPORT the claim
     (wired+on_by_default for wired/on-by-default/integrated; + scaffold_retired for one-brain/scaffold-retired). Else the
     row's own `host_scaffold_in_default` is printed as the refutation. This mirrors closure_names_mechanism: the author
     must NAME the key; the key's status is authoritative.

  C  RATCHET (standing measurement). Recompute headline.total_faculties == len(rows) and headline.scaffold_retired ==
     count(scaffold_retired==YES); a mismatch blocks (mechanical drift, like summary_doc_freshness).

  D  SCAFFOLD-RETIREMENT FORCING FUNCTION (added 2026-09-02, owner steer: make scaffold-retirement a first-class,
     GATE-BLOCKED state, not a thing you can leave on forever). Every row carries a retire_status ∈
       RETIRED                     the host cheat is gone from the default path (scaffold_retired must be YES)
       RETIRABLE_NOW <YYYY-MM-DD>   the neural replacement already authors the verdict; only dead/overridden/demoted
                                    host code remains — retire it. The DATE is when it became retirable.
       BLOCKED:<frontier-row>       host cognition is still in the default path AND its neural replacement is not GO;
                                    <frontier-row> names the ledger row carrying the unmet wall.
       LEGITIMATE                   world/body/clock/initial-education (curriculum/corpus) — EXEMPT, never owes retirement.
       ADDITIVE                     no host-COGNITION scaffold in the default path (a substrate-consolidation gap or a
                                    permanent additive config bound is not a cheat) — a valid terminal state.
     Four sub-rules block a commit:
       (a) an on_by_default:YES row with NO retire_status BLOCKS (you must classify a live faculty).
       (b) a BLOCKED:<key> must name a REAL frontier row that is genuinely NOT yet in the production default —
           de_risked!=YES OR on_by_default!=YES. A dangling key, or one naming a row already fully in production
           (de_risked=YES AND on_by_default=YES), BLOCKS. (The literal owner spec was "de_risked!=YES"; the ledger
           currently has only neural-render+perception-motor at de_risked!=YES, both world/body-ish, so enforcing that
           ALONE would collapse every BLOCKED onto neural-render and erase the genuine reward/topic-swap dependencies.
           "de_risked!=YES OR on_by_default!=YES" is the strict-superset guard that still forbids the case the rule
           targets — naming a solved, deployed capability as your blocker — while letting each BLOCKED point at its
           actual unmet frontier.)
       (c) a RETIRABLE_NOW dated more than K_RETIRABLE_DAYS (14) ago BLOCKS — the forcing function: a cheat that CAN be
           retired cannot be left ON indefinitely; ship the retirement or re-classify with evidence.
       (d) LEGITIMATE / ADDITIVE / RETIRED are valid terminal states (the curriculum/world/body/clock never owe a
           retirement; a completed retirement is done).
     Check D runs whenever the ledger or an anchored source file is staged (like A + C).

HONEST BOUNDARY. A static gate proves reachable + default-on (config-as-source), NOT correctness, and cannot see a
runtime brain.json manifest override (developed_brain_io) — those need a nightly BEHAVIORAL probe that builds the default
ChatBrain and runs a lesion battery. LEVEL-3 ("spiking, on by default") credit is only real under a LESION test (disable
the spiking path -> the default answer must CHANGE); this gate enforces CONSISTENCY-WITH-THE-GRADE, the probe enforces
the grade's truth.
"""
from __future__ import annotations

import os
import re

NAME = "production-integration"
CLASS_ID = "PI"
BLOCKING = True

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEDGER_REL = "docs/PRODUCTION_INTEGRATION_LEDGER.yaml"

# Files whose edit must re-check the anchors (a source flip without a ledger move must block).
ANCHORED_FILES = (
    "webapp/server.py",
    "research/runners/brain_chat_tui.py",
    "research/runners/rf_phasor_composer.py",
    "research/runners/brain_conversational_agent.py",
    "research/runners/rich_answer_composer.py",
    LEDGER_REL,
)

# Narrow, explicit PRODUCTION-INTEGRATION claim tokens. Check B is a HEURISTIC overclaim-catcher (the hard mechanical
# enforcement is Check A's source anchors + Check C's ratchet); it is deliberately CONSERVATIVE (few false positives)
# because a gate that false-alarms on goal/drift prose gets bypassed. Generic goal vocabulary ("on by default in
# production") is NOT a token — it is how the goal is stated; only unambiguous faculty-CLAIM phrasings fire.
CLAIM_TOKENS = [
    (re.compile(r"\bwired into (?:the )?(?:production|/api/brain-chat|the live loop)\b", re.I), "wired"),
    (re.compile(r"\bintegrated into (?:the )?(?:production|/api/brain-chat|live loop|default turn)\b", re.I), "integrated"),
    (re.compile(r"\bnow (?:on[- ]by[- ]default|the production default|running by default)\b", re.I), "on_by_default"),
    (re.compile(r"\b(?:its |the )host scaffold (?:is|has been) retired\b", re.I), "scaffold_retired"),
]
FACULTY_RE = re.compile(r"^integration_faculty:\s*(\S+)\s*$", re.M)
# A DESCRIPTIVE / NEGATED / drift-discussion use is not an affirmative claim (mirrors closure_names_mechanism's guard):
# "NONE checked whether it was wired into production", "NOT yet integrated", "the goal is to become on by default",
# "stayed host". Searched UNANCHORED in the ~48 chars before the claim phrase (a negation a few words back still governs).
NEG_CONTEXT_RE = re.compile(
    r"\b(?:not|no|none|never|whether|without|isn't|wasn't|aren't|weren't|neither|nor|"
    r"stay(?:ed|s)?|still|host|de-?risk|default-off|would|to\s+become|become(?:s)?|"
    r"must|should|needs?|goal|target|yet)\b",
    re.I)


# ---------------------------------------------------------------- minimal YAML read (rows + anchors + headline)
def _load_ledger(text):
    """Parse the ledger with PyYAML if present, else a scoped fallback for our fixed schema. Returns dict or None."""
    try:
        import yaml  # noqa
        return yaml.safe_load(text)
    except Exception:
        pass
    # Fallback: parse just what the gate needs (headline.total_faculties/scaffold_retired + row levels + anchors).
    data = {"headline": {}, "rows": []}
    # headline scalars
    for k in ("total_faculties", "scaffold_retired", "default_on_spiking_faculties"):
        m = re.search(r"^\s{2}%s:\s*(\d+)" % k, text, re.M)
        if m:
            data["headline"][k] = int(m.group(1))
    # rows: split on "  - key:"
    for block in re.split(r"\n\s{2}-\s+key:\s*", text)[1:]:
        row = {}
        row["key"] = block.splitlines()[0].strip()
        for lvl in ("de_risked", "wired", "on_by_default", "scaffold_retired"):
            m = re.search(r"^\s{4}%s:\s*(\S+)" % lvl, block, re.M)
            if m:
                row[lvl] = m.group(1).strip()
        for f in ("host_scaffold_in_default",):
            m = re.search(r'^\s{4}%s:\s*"?(.+?)"?\s*$' % f, block, re.M)
            if m:
                row[f] = m.group(1)
        # retire_status: quote- and trailing-comment-robust (PyYAML strips comments; the fallback must too).
        m = re.search(r'^\s{4}retire_status:\s*"([^"]*)"|^\s{4}retire_status:\s*([^"#\n]+?)\s*(?:#.*)?$', block, re.M)
        if m:
            row["retire_status"] = (m.group(1) if m.group(1) is not None else m.group(2)).strip()
        # default_anchor entries
        anchors = []
        for a in re.split(r"\n\s{6}-\s+file:\s*", block)[1:]:
            anc = {"file": a.splitlines()[0].strip()}
            for fld in ("assign", "off_value", "on_value", "count"):
                m = re.search(r"^\s{8}%s:\s*(.+?)\s*$" % fld, a, re.M)
                if m:
                    anc[fld] = m.group(1).strip().strip('"')
            if "count" in anc:
                anc["count"] = int(anc["count"])
            if "assign" in anc:
                anchors.append(anc)
        if anchors:
            row["default_anchor"] = anchors
        sc = re.search(r"^\s{4}scaffold_symbol:\s*\n\s{6}file:\s*(\S+)\s*\n\s{6}symbol:\s*\"?(.+?)\"?\s*$", block, re.M)
        if sc:
            row["scaffold_symbol"] = {"file": sc.group(1).strip(), "symbol": sc.group(2).strip()}
        data["rows"].append(row)
    return data


def _read(rel):
    full = rel if os.path.isabs(rel) else os.path.join(_ROOT, rel)
    try:
        return open(full, errors="ignore").read()
    except OSError:
        return None


def _count_assign(text, assign, value):
    """Count `assign = value` occurrences, whitespace- and quote-robust. value: True/False or a (maybe-quoted) string."""
    v = str(value).strip().strip('"').strip("'")
    if v in ("True", "False", "None"):
        pat = r"\b%s\s*(?::[^=\n]+)?=\s*%s\b" % (re.escape(assign), v)
    else:
        pat = r"\b%s\s*(?::[^=\n]+)?=\s*['\"]%s['\"]" % (re.escape(assign), re.escape(v))
    return len(re.findall(pat, text))


# ---------------------------------------------------------------- level normalizer (THE SILENT-DEATH FIX, 2026-08-26)
def _level(v):
    """Normalize a ledger LEVEL to the canonical 'YES'/'NO'/'PARTIAL' token.

    PyYAML's safe_load applies the YAML-1.1 boolean rule and coerces the bare `on_by_default: YES` / `NO`
    (and de_risked/wired/scaffold_retired) to Python True/False. The old checks then compared
    str(True).upper()=='TRUE' against the literal 'YES'/'NO' — matching NEITHER — so sub-check A was vacuous
    for every YES/NO row, sub-check B mis-read every faculty as wired=False/on_by_default=False, and the
    ratchet counted 0 scaffold_retired (research/FAILURE_LOG.md 2026-08-26 'PI GATE SUB-CHECKS A+B SILENTLY
    DEAD'). Normalizing here makes the checks compare like-for-like whether PyYAML coerced or the fallback
    parser left a raw string. True->'YES', False->'NO', anything else -> str(v).upper() (so 'PARTIAL',
    'yes', a quoted "YES" all land correctly)."""
    if v is True:
        return "YES"
    if v is False:
        return "NO"
    return str(v).strip().upper()


# ---------------------------------------------------------------- the three sub-checks (operate on a parsed ledger)
def _check_anchors(data):
    problems = []
    for row in data.get("rows", []):
        key = row.get("key", "?")
        lvl = _level(row.get("on_by_default", ""))
        for anc in row.get("default_anchor", []) or []:
            src = _read(anc["file"])
            if src is None:
                problems.append("[A] row %r: default_anchor file %s not found" % (key, anc["file"]))
                continue
            off_ct = _count_assign(src, anc["assign"], anc.get("off_value", "False"))
            on_ct = _count_assign(src, anc["assign"], anc.get("on_value", "True"))
            want = anc.get("count", 1)
            if lvl == "NO":
                if off_ct != want or on_ct != 0:
                    problems.append(
                        "[A] row %r says on_by_default:NO but %s in %s has off=%d (expect %d) / on=%d (expect 0) — "
                        "the source and the ledger disagree: either the faculty was turned ON without recording it, or "
                        "the ledger is stale." % (key, anc["assign"], anc["file"], off_ct, want, on_ct))
            elif lvl == "YES":
                if on_ct < 1 or off_ct != 0:
                    problems.append(
                        "[A] row %r says on_by_default:YES but %s in %s has on=%d / off=%d (expect off=0) — the ledger "
                        "claims default-on while the source still assigns the OFF value." % (key, anc["assign"], anc["file"], on_ct, off_ct))
            # PARTIAL / other: no literal derivable, skip (Check B still governs any claim).
        sc = row.get("scaffold_symbol")
        if sc:
            src = _read(sc["file"])
            if src is None:
                problems.append("[A] row %r: scaffold_symbol file %s not found" % (key, sc["file"]))
            else:
                present = sc["symbol"] in src
                retired = _level(row.get("scaffold_retired", "")) == "YES"
                if retired and present:
                    problems.append("[A] row %r says scaffold_retired:YES but the host symbol %r is STILL PRESENT in %s"
                                    % (key, sc["symbol"], sc["file"]))
                if not retired and not present:
                    problems.append("[A] row %r says scaffold_retired:%s but the host symbol %r is GONE from %s — the "
                                    "scaffold was retired without recording it." % (key, row.get("scaffold_retired"), sc["symbol"], sc["file"]))
    return problems


def _check_ratchet(data):
    problems = []
    h = data.get("headline", {}) or {}
    rows = data.get("rows", [])
    if "total_faculties" in h and h["total_faculties"] != len(rows):
        problems.append("[C] headline.total_faculties=%s but the ledger has %d rows" % (h["total_faculties"], len(rows)))
    got_retired = sum(1 for r in rows if _level(r.get("scaffold_retired", "")) == "YES")
    if "scaffold_retired" in h and h["scaffold_retired"] != got_retired:
        problems.append("[C] headline.scaffold_retired=%s but %d rows have scaffold_retired:YES" % (h["scaffold_retired"], got_retired))
    return problems


def _check_claim(text, rel, data):
    """A doc using a production-integration claim token must name a supporting ledger key."""
    hit = None
    for rx, kind in CLAIM_TOKENS:
        for m in rx.finditer(text):
            # skip a DESCRIPTIVE / NEGATED use (drift-discussion, a goal, a de-risk) — only an AFFIRMATIVE claim fires.
            pre = text[max(0, m.start() - 48):m.start()]
            if NEG_CONTEXT_RE.search(pre):
                continue
            hit = (m.group(0), kind)
            break
        if hit:
            break
    if not hit:
        return []
    phrase, kind = hit
    fm = FACULTY_RE.search(text[:text.find("\n---", 3) + 4] if text.startswith("---") else "")
    rows = {r.get("key"): r for r in data.get("rows", [])}
    if not fm:
        return ["[B] %s: uses a production-integration claim %r but declares no `integration_faculty: <key>`. Name the "
                "ledger row so the claim can be adjudicated against the production default." % (rel, phrase[:50])]
    key = fm.group(1)
    row = rows.get(key)
    if row is None:
        return ["[B] %s: integration_faculty:%r is not a row in %s" % (rel, key, LEDGER_REL)]
    wired = _level(row.get("wired", "")) == "YES"
    on_def = _level(row.get("on_by_default", "")) == "YES"
    retired = _level(row.get("scaffold_retired", "")) == "YES"
    need_retired = kind in ("onebrain", "scaffold_retired")
    ok = wired and on_def and (retired if need_retired else True)
    if not ok:
        return ["[B] %s: claims %r for faculty %r, but its ledger row is wired=%s on_by_default=%s scaffold_retired=%s. "
                "Host still in the default path: %s" % (rel, phrase[:40], key, row.get("wired"), row.get("on_by_default"),
                                                        row.get("scaffold_retired"), row.get("host_scaffold_in_default", "?"))]
    return []


# ---------------------------------------------------------------- Check D: scaffold-retirement forcing function
K_RETIRABLE_DAYS = 14  # a RETIRABLE_NOW cheat older than this BLOCKS — the forcing function.
_RETIRE_TERMINAL = {"RETIRED", "LEGITIMATE", "ADDITIVE"}
_RETIRE_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _check_retire_status(data, today=None):
    """Every row must carry an honest retire_status; RETIRABLE_NOW ages out; BLOCKED must name a live frontier."""
    import datetime
    if today is None:
        today = datetime.date.today()
    problems = []
    rows = data.get("rows", [])
    keys = {r.get("key") for r in rows}
    for row in rows:
        key = row.get("key", "?")
        on = _level(row.get("on_by_default", ""))
        rs = row.get("retire_status")
        rs = "" if rs is None else str(rs).strip()
        # (a) absent on an on_by_default:YES row BLOCKS.
        if not rs:
            if on == "YES":
                problems.append("[D] row %r is on_by_default:YES but has no retire_status — classify it "
                                "(RETIRED / RETIRABLE_NOW <date> / BLOCKED:<frontier-row> / LEGITIMATE / ADDITIVE)." % key)
            continue
        head = re.split(r"[:\s]", rs, 1)[0]
        if head == "RETIRED":
            # (d) consistency: a RETIRED row's scaffold_retired must actually be YES.
            if _level(row.get("scaffold_retired", "")) != "YES":
                problems.append("[D] row %r retire_status:RETIRED but scaffold_retired!=YES — the two must agree." % key)
        elif head in ("LEGITIMATE", "ADDITIVE"):
            pass  # (d) valid terminal states — never owe a retirement.
        elif head == "RETIRABLE_NOW":
            m = _RETIRE_DATE_RE.search(rs)
            if not m:
                problems.append("[D] row %r retire_status:RETIRABLE_NOW must carry a YYYY-MM-DD date "
                                "(e.g. 'RETIRABLE_NOW 2026-09-02')." % key)
            else:
                try:
                    d = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                except ValueError:
                    problems.append("[D] row %r retire_status:RETIRABLE_NOW carries an invalid date %r." % (key, m.group(0)))
                    continue
                age = (today - d).days
                if age > K_RETIRABLE_DAYS:
                    problems.append("[D] row %r retire_status:RETIRABLE_NOW dated %s is %d days old (> %d) — a cheat "
                                    "scaffold that CAN be retired cannot be left ON: ship the retirement (flip to "
                                    "RETIRED) or re-classify with evidence." % (key, m.group(0), age, K_RETIRABLE_DAYS))
        elif head == "BLOCKED":
            fr = rs.partition(":")[2].strip()
            if not fr:
                problems.append("[D] row %r retire_status:BLOCKED must name a frontier row (BLOCKED:<key>)." % key)
            elif fr not in keys:
                problems.append("[D] row %r retire_status:BLOCKED:%s — %r is not a row in the ledger (dangling frontier)."
                                % (key, fr, fr))
            else:
                frow = next(r for r in rows if r.get("key") == fr)
                fde = _level(frow.get("de_risked", ""))
                fon = _level(frow.get("on_by_default", ""))
                if fde == "YES" and fon == "YES":
                    problems.append("[D] row %r is BLOCKED on %r, but %r is ALREADY in the production default "
                                    "(de_risked=YES AND on_by_default=YES) — not a genuine unmet frontier. Name a row "
                                    "not yet in production (de_risked!=YES or on_by_default!=YES), or re-classify." % (key, fr, fr))
        else:
            problems.append("[D] row %r has an unknown retire_status %r (expected RETIRED / RETIRABLE_NOW <date> / "
                            "BLOCKED:<row> / LEGITIMATE / ADDITIVE)." % (key, rs))
    return problems


# ---------------------------------------------------------------- entry points
def check(paths):
    if paths is None or len(paths) == 0:
        return []  # legacy audited on touch
    norm = [p.replace("\\", "/") for p in paths]
    problems = []
    ledger_text = _read(LEDGER_REL)
    if ledger_text is None:
        # only complain if something in scope needs it
        if any(p == LEDGER_REL or p in ANCHORED_FILES for p in norm):
            return ["[PI] %s is missing — the production-integration ledger must exist." % LEDGER_REL]
        return []
    data = _load_ledger(ledger_text)
    if not data or not data.get("rows"):
        return ["[PI] %s failed to parse / has no rows." % LEDGER_REL]

    # A + C + D run when the ledger or any anchored source file is staged.
    if any(p == LEDGER_REL or p in ANCHORED_FILES for p in norm):
        problems += _check_anchors(data)
        problems += _check_ratchet(data)
        problems += _check_retire_status(data)
    # B runs on staged governed docs.
    for p in norm:
        if not p.endswith(".md"):
            continue
        if not (p.startswith("research/findings/") or p.endswith("GAP_CLOSURE_MISSION.md")
                or p.endswith("ROADMAP.md") or "docs/plans/" in p):
            continue
        t = _read(p)
        if t:
            problems += _check_claim(t, p, data)
    return problems


def selftest():
    """FAILING DIRECTION FIRST for each sub-check, then negative controls."""
    import tempfile
    bad = []

    # ---- Check A: level-derived literal ----
    src = "x=1\nenable_neural_render = True\nenable_neural_render=True\n"  # source turned ON (2x True)
    data_off = {"headline": {"total_faculties": 1, "scaffold_retired": 0},
                "rows": [{"key": "neural-render", "on_by_default": "NO", "scaffold_retired": "NO",
                          "default_anchor": [{"file": "__mem__", "assign": "enable_neural_render",
                                              "off_value": "False", "on_value": "True", "count": 2}]}]}
    # monkeypatch _read to serve the in-memory source
    import tools.gates.production_integration as self_mod
    orig_read = self_mod._read
    self_mod._read = lambda rel: src if rel == "__mem__" else orig_read(rel)
    try:
        if not _check_anchors(data_off):
            bad.append("[A] did NOT catch on_by_default:NO while source assigns True (source flipped on, ledger stale)")
        # theatre hole: row flipped to YES while source still False
        src2 = "enable_neural_render = False\nenable_neural_render=False\n"
        self_mod._read = lambda rel: src2 if rel == "__mem__" else orig_read(rel)
        data_yes = {"rows": [{"key": "nr", "on_by_default": "YES", "scaffold_retired": "NO",
                              "default_anchor": [{"file": "__mem__", "assign": "enable_neural_render",
                                                  "off_value": "False", "on_value": "True", "count": 2}]}]}
        if not _check_anchors(data_yes):
            bad.append("[A] THEATRE HOLE: did NOT catch on_by_default:YES while source still assigns False")
        # negative control: consistent (NO + source False)
        data_ok = {"rows": [{"key": "nr", "on_by_default": "NO", "scaffold_retired": "NO",
                             "default_anchor": [{"file": "__mem__", "assign": "enable_neural_render",
                                                 "off_value": "False", "on_value": "True", "count": 2}]}]}
        if _check_anchors(data_ok):
            bad.append("[A] FALSE POSITIVE: flagged a consistent NO-level row with matching OFF source")
        # scaffold symbol: retired=NO but symbol GONE
        self_mod._read = lambda rel: "no such thing here" if rel == "__mem__" else orig_read(rel)
        data_sc = {"rows": [{"key": "cs", "scaffold_retired": "NO",
                             "scaffold_symbol": {"file": "__mem__", "symbol": "def match_fact"}}]}
        if not _check_anchors(data_sc):
            bad.append("[A] did NOT catch scaffold_retired:NO while the host symbol is GONE")
        self_mod._read = lambda rel: "def match_fact(self):" if rel == "__mem__" else orig_read(rel)
        data_sc2 = {"rows": [{"key": "cs", "scaffold_retired": "YES",
                              "scaffold_symbol": {"file": "__mem__", "symbol": "def match_fact"}}]}
        if not _check_anchors(data_sc2):
            bad.append("[A] did NOT catch scaffold_retired:YES while the host symbol is STILL PRESENT")
    finally:
        self_mod._read = orig_read

    # ---- Check C: ratchet ----
    if not _check_ratchet({"headline": {"total_faculties": 5}, "rows": [{"key": "a"}]}):
        bad.append("[C] did NOT catch total_faculties mismatch")
    if not _check_ratchet({"headline": {"scaffold_retired": 3},
                           "rows": [{"key": "a", "scaffold_retired": "NO"}]}):
        bad.append("[C] did NOT catch scaffold_retired count mismatch")
    if _check_ratchet({"headline": {"total_faculties": 1, "scaffold_retired": 0},
                       "rows": [{"key": "a", "scaffold_retired": "NO"}]}):
        bad.append("[C] FALSE POSITIVE: flagged a consistent header")

    # ---- Check B: claim<->ledger ----
    data = {"rows": [{"key": "content-selection", "wired": "NO", "on_by_default": "NO", "scaffold_retired": "NO",
                      "host_scaffold_in_default": "QuestionRouter.match_fact"},
                     {"key": "semantic-recall", "wired": "YES", "on_by_default": "YES", "scaffold_retired": "YES"}]}
    with tempfile.TemporaryDirectory() as d:
        def claim(body, fac=None):
            fm = "---\ntype: finding\n%s---\n\n%s\n" % (("integration_faculty: %s\n" % fac) if fac else "", body)
            return fm
        # 1. claim token, no integration_faculty declared
        if not _check_claim(claim("This faculty is now wired into production and load-bearing."), "research/findings/a.md", data):
            bad.append("[B] did NOT catch a production claim with no integration_faculty")
        # 2. claim naming a row that does NOT support it
        if not _check_claim(claim("Now integrated into /api/brain-chat.", "content-selection"), "research/findings/b.md", data):
            bad.append("[B] did NOT catch an 'integrated' claim on a wired=NO row")
        # 3. negative control: claim naming a row that DOES support it
        if _check_claim(claim("Now integrated into /api/brain-chat.", "semantic-recall"), "research/findings/c.md", data):
            bad.append("[B] FALSE POSITIVE: flagged a supported integration claim")
        # 4. negative control: no claim token at all
        if _check_claim(claim("Held-out accuracy rises to 0.61 under the expander."), "research/findings/d.md", data):
            bad.append("[B] FALSE POSITIVE: flagged an ordinary finding with no production claim")
        # 5. negative control: a NEGATED / descriptive drift-discussion use must NOT fire
        if _check_claim(claim("NONE checked whether it was wired into the production default; the drift is the goal."),
                        "research/findings/e.md", data):
            bad.append("[B] FALSE POSITIVE: flagged a NEGATED/descriptive 'wired into production' (drift discussion)")
        if _check_claim(claim("Today it is NOT integrated into /api/brain-chat; the goal is to become on by default."),
                        "research/findings/f.md", data):
            bad.append("[B] FALSE POSITIVE: flagged a 'NOT integrated / goal to become' descriptive use")

    # ---- PyYAML BOOLEAN COERCION (the 2026-08-26 silent-death guard) ----
    # PyYAML's safe_load coerces bare `on_by_default: YES/NO` (+ wired/scaffold_retired) to Python True/False.
    # The pre-fix checks compared str(True).upper()=='TRUE' against 'YES'/'NO' — matching NEITHER — so A was
    # vacuous, B mis-read every row as wired=False/on=False, and C counted 0 retired (all SILENTLY: the string-only
    # cases above still passed, which is exactly why the bug shipped). These cases feed BOOLEAN levels as safe_load
    # produces them and FAIL if _level is removed/neutered — the registry's proof that the revived checks are live.
    _orig_read = self_mod._read
    self_mod._read = lambda rel: "flag=False\nflag=False\n" if rel == "__mem__" else _orig_read(rel)
    try:
        # A: on_by_default coerced to True(bool) while the source assigns the OFF value -> MUST flag.
        d_boolA = {"rows": [{"key": "b", "on_by_default": True, "scaffold_retired": False,
                             "default_anchor": [{"file": "__mem__", "assign": "flag",
                                                 "off_value": "False", "on_value": "True", "count": 2}]}]}
        if not _check_anchors(d_boolA):
            bad.append("[A/bool] SILENT-DEATH: on_by_default=True(bool) + OFF source NOT flagged (the _level fix is dead)")
        # A: on_by_default True(bool) consistent with an ON source -> must NOT flag.
        self_mod._read = lambda rel: "flag=True\nflag=True\n" if rel == "__mem__" else _orig_read(rel)
        d_boolAok = {"rows": [{"key": "b", "on_by_default": True, "scaffold_retired": False,
                               "default_anchor": [{"file": "__mem__", "assign": "flag",
                                                   "off_value": "False", "on_value": "True", "count": 2}]}]}
        if _check_anchors(d_boolAok):
            bad.append("[A/bool] FALSE POSITIVE: on_by_default=True(bool) consistent with the ON source was flagged")
    finally:
        self_mod._read = _orig_read
    # C: scaffold_retired coerced to True(bool) MUST be counted (else a false ratchet mismatch).
    if _check_ratchet({"headline": {"scaffold_retired": 1},
                       "rows": [{"key": "a", "scaffold_retired": True}, {"key": "b", "scaffold_retired": False}]}):
        bad.append("[C/bool] SILENT-DEATH: scaffold_retired=True(bool) not counted (the _level fix is dead)")
    # B: a SUPPORTED claim whose row levels are BOOLEANS (wired=True, on_by_default=True) must NOT flag.
    d_boolB = {"rows": [{"key": "semantic-recall", "wired": True, "on_by_default": True, "scaffold_retired": True}]}
    supported = "---\ntype: finding\nintegration_faculty: semantic-recall\n---\n\nNow integrated into /api/brain-chat.\n"
    if _check_claim(supported, "research/findings/g.md", d_boolB):
        bad.append("[B/bool] FALSE POSITIVE: a supported claim with BOOLEAN levels (wired=True) was flagged (the _level fix is dead)")

    # ---- Check D: retire-status forcing function (FAILING DIRECTION FIRST, then negative controls) ----
    import datetime as _dt
    NOW = _dt.date(2026, 9, 2)
    # fixture frontier targets (carry their own valid retire_status so they add no problems of their own):
    fr_off = {"key": "neural-render", "de_risked": "PARTIAL", "on_by_default": "NO",
              "retire_status": "LEGITIMATE"}                                              # a genuine frontier
    fr_live = {"key": "semantic-recall", "de_risked": "YES", "on_by_default": "YES",
               "retire_status": "LEGITIMATE"}                                             # already in production

    def _d(rows):
        return _check_retire_status({"rows": rows + [fr_off, fr_live]}, today=NOW)

    # (a) absent retire_status on an on_by_default:YES row MUST flag.
    if not _d([{"key": "x", "on_by_default": "YES"}]):
        bad.append("[D] did NOT catch a missing retire_status on an on_by_default:YES row")
    # (a) absent on an on_by_default:NO row must NOT flag.
    if _d([{"key": "x", "on_by_default": "NO"}]):
        bad.append("[D] FALSE POSITIVE: flagged a missing retire_status on an on_by_default:NO row")
    # (c) RETIRABLE_NOW past the age MUST flag (dated 2026-08-01 vs today 2026-09-02 -> 32 days > 14).
    if not _d([{"key": "x", "on_by_default": "YES", "retire_status": "RETIRABLE_NOW 2026-08-01"}]):
        bad.append("[D] did NOT catch a RETIRABLE_NOW aged past K_RETIRABLE_DAYS")
    # (c) RETIRABLE_NOW within the age must NOT flag (dated today).
    if _d([{"key": "x", "on_by_default": "YES", "retire_status": "RETIRABLE_NOW 2026-09-02"}]):
        bad.append("[D] FALSE POSITIVE: flagged a fresh (0-day) RETIRABLE_NOW")
    # RETIRABLE_NOW with no date MUST flag.
    if not _d([{"key": "x", "on_by_default": "YES", "retire_status": "RETIRABLE_NOW"}]):
        bad.append("[D] did NOT catch a RETIRABLE_NOW with no date")
    # (b) BLOCKED naming a dangling key MUST flag.
    if not _d([{"key": "x", "on_by_default": "YES", "retire_status": "BLOCKED:no-such-row"}]):
        bad.append("[D] did NOT catch a BLOCKED naming a non-existent frontier row")
    # (b) BLOCKED naming a row already fully in production (de=YES AND on=YES) MUST flag.
    if not _d([{"key": "x", "on_by_default": "YES", "retire_status": "BLOCKED:semantic-recall"}]):
        bad.append("[D] did NOT catch a BLOCKED naming an already-in-production frontier (de=YES AND on=YES)")
    # (b) BLOCKED with no frontier named MUST flag.
    if not _d([{"key": "x", "on_by_default": "YES", "retire_status": "BLOCKED:"}]):
        bad.append("[D] did NOT catch a BLOCKED with no frontier row named")
    # (b) BLOCKED naming a genuine frontier (on=NO) must NOT flag.
    if _d([{"key": "x", "on_by_default": "YES", "retire_status": "BLOCKED:neural-render"}]):
        bad.append("[D] FALSE POSITIVE: flagged a BLOCKED naming a genuine unmet frontier row")
    # (d) RETIRED with scaffold_retired!=YES MUST flag; with scaffold_retired=YES must NOT.
    if not _d([{"key": "x", "on_by_default": "YES", "scaffold_retired": "NO", "retire_status": "RETIRED"}]):
        bad.append("[D] did NOT catch a RETIRED row whose scaffold_retired!=YES")
    if _d([{"key": "x", "on_by_default": "YES", "scaffold_retired": "YES", "retire_status": "RETIRED"}]):
        bad.append("[D] FALSE POSITIVE: flagged a consistent RETIRED row (scaffold_retired=YES)")
    # (d) LEGITIMATE / ADDITIVE are valid terminal states — must NOT flag.
    if _d([{"key": "x", "on_by_default": "YES", "retire_status": "LEGITIMATE"}]):
        bad.append("[D] FALSE POSITIVE: flagged a valid LEGITIMATE terminal state")
    if _d([{"key": "x", "on_by_default": "YES", "retire_status": "ADDITIVE"}]):
        bad.append("[D] FALSE POSITIVE: flagged a valid ADDITIVE terminal state")
    # an unknown status MUST flag.
    if not _d([{"key": "x", "on_by_default": "YES", "retire_status": "MAYBE_LATER"}]):
        bad.append("[D] did NOT catch an unknown retire_status token")
    # BOOLEAN-coerced on_by_default (PyYAML) must still be read as YES by (a)/(b).
    if not _check_retire_status({"rows": [{"key": "x", "on_by_default": True}, fr_off, fr_live]}, today=NOW):
        bad.append("[D/bool] did NOT catch a missing retire_status on on_by_default=True(bool)")

    # ---- scoping ----
    if check(None) or check([]):
        bad.append("SCOPE LEAK: standalone/empty mode must not scan")
    return bad
