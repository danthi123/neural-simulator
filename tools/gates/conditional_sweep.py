"""FAILURE CLASS 10 -- single-axis-sweep-reported-as-absolute (3 incidents).

EVIDENCE. A sweep varies one axis while every other axis sits at whatever value it happened to have, and the
winner is then written down as *the* optimum. The gap#5 read-density sweep reported an optimum of 0.15/0.25
that, once a co-varying axis (`w_max`) was moved to a correct bound, scored BELOW density 1.0 -- the answer was
INVERTED, not merely narrow. The session note: "the same error I made with the density optimum itself... I keep
reporting them as absolute ones". The cost is that the conditional answer gets tuned on top of, so the next
finding inherits the wrong operating point.

THE GATE. In a frontmatter-bearing finding, an optimum-claim that names a parameter AND its winning value must
carry, within +/-8 lines, either (a) at least one OTHER parameter pinned to a value ("at k=v", "k held at v", a
config block, a command line), or (b) a cited artifact that EXISTS and records >=2 config keys. That is the
minimum needed for a reader to know what the answer is conditional ON.

WHAT THIS GATE CANNOT CATCH.
  * Whether the held-fixed values were themselves CORRECT. The gap#5 optimum was stated with `dwell=180` right
    beside it and was still inverted, because `w_max=150` was inside the plasticity bound trap (class 2). This
    gate enforces DISCLOSURE of the conditioning, never its validity.
  * Optimum-claims whose parameter is an ordinary English word with no marker (no backticks, no `_`, not `--`,
    not in the small vocabulary below) -- e.g. "the best setting was 3". Deliberate: broadening the parameter
    pattern to bare nouns makes every "at best, 3 seeds" a false alarm, and a gate that cries wolf gets ignored.
  * Claims split across lines, or a sweep reported with no winning VALUE at all ("density was the operative
    axis") -- there is nothing to bind the conditioning to.
  * Findings with no `status:` frontmatter, and anything under research/findings/raw/.
Advisory (BLOCKING=False): the detector reads prose, so a miss is likely and a block would be the wrong trade.
"""
from __future__ import annotations

import glob
import json
import os
import re

NAME = "conditional-sweep"
CLASS_ID = "10"
BLOCKING = False

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CORPUS = os.path.join(_REPO, "research", "findings", "*.md")
_WINDOW = 8          # lines either side of the claim in which the conditioning must appear
_NEAR = 60           # chars between the optimum word and the parameter it is about

# Parameter-ish words that appear bare in this project's prose. Kept SHORT on purpose: every addition here is a
# new chance to fire on ordinary English.
_VOCAB = ("density", "dwell", "sparsity", "threshold", "epochs", "seeds", "lr", "gain", "sigma", "temperature")

_OPT = re.compile(r"\b(optimum|optimal|optima|best|peaks?\s+at|sweet\s+spot)\b", re.I)
# k=v / k: v / k at v / k of v / --k v   (value must be numeric -- a pinned axis, not a mention)
_ASSIGN = re.compile(
    r"(?P<mark>`|--)?(?P<k>[A-Za-z_][A-Za-z0-9_.-]{0,40})`?\s*(?P<sep>=|:|held\s+at|at|of)\s+?"
    r"\**`?(?P<v>-?\d+(?:\.\d+)?(?:e-?\d+)?)",
    re.I,
)
# Metadata that carries a number but is not an experimental axis. Counting these as "held fixed" would make the
# gate unfailable on every frontmatter-bearing finding -- which is failure class 3, the one this package exists
# to stop.
_NOT_AXIS = frozenset(("date", "status", "lane", "mechanism", "claim_check", "seed", "seeds", "figure", "fig",
                       "table", "section", "line", "commit", "page", "ref", "eq", "footnote", "cycle", "day"))
_BARE_NUM = re.compile(r"\**`?(-?\d+(?:\.\d+)?)")
_ARTIFACT = re.compile(r"(?:research/findings/)?raw/[\w./*-]+\.json")
_MARKED_PARAM = re.compile(r"`(?P<b>[A-Za-z_][A-Za-z0-9_.-]{0,40})`|--(?P<d>[a-z][a-z0-9_-]{1,40})|"
                           r"\b(?P<u>[A-Za-z]+_[A-Za-z0-9_]+)\b|\b(?P<v>" + "|".join(_VOCAB) + r")\b", re.I)


def _norm(k):
    return k.strip("`*- ").lower()


def _held_axes(text):
    """Every EXPERIMENTAL axis pinned to a numeric value in `text`.

    An axis counts only if it is pinned with `=`/`:` or its name is marked as a parameter (backticks, `--`, an
    underscore, or the small vocabulary). Without that, ordinary prose -- "a ratio of 0.98", "at best 3" -- would
    supply the conditioning evidence for free and no finding could ever fail this gate.
    """
    out = set()
    for m in _ASSIGN.finditer(text):
        k = _norm(m.group("k"))
        if k in _NOT_AXIS or not k:
            continue
        marked = bool(m.group("mark")) or "_" in k or k in _VOCAB
        if m.group("sep") in ("=", ":") or marked:
            out.add(k)
    return out


def _artifact_records_config(rel):
    """True if a cited artifact exists and records >=2 config-looking keys."""
    if os.path.isabs(rel):
        path = rel
    else:
        path = rel if rel.startswith("research/") else os.path.join("research", "findings", rel)
    for cand in glob.glob(os.path.join(_REPO, path))[:4]:
        try:
            if os.path.getsize(cand) > 20_000_000:
                continue
            with open(cand, "r", encoding="utf-8", errors="replace") as fh:
                obj = json.load(fh)
        except (OSError, ValueError):
            continue                      # unreadable artifact is not evidence -- but is not THIS class's fault
        if not isinstance(obj, dict):
            continue
        pools = [obj] + [v for k, v in obj.items()
                         if k.lower() in ("config", "cfg", "params", "args", "settings") and isinstance(v, dict)]
        for pool in pools:
            if sum(1 for v in pool.values() if isinstance(v, (int, float, bool, str))) >= 2:
                return True
    return False


def _claims(lines):
    """(line_index, parameter, value) for each optimum-claim that names a parameter and its winning value."""
    out = []
    for i, line in enumerate(lines):
        for om in _OPT.finditer(line):
            hit = None
            for am in _ASSIGN.finditer(line):                       # form A: "optimal w_max=150"
                if min(abs(am.start() - om.end()), abs(om.start() - am.end())) <= _NEAR:
                    hit = (_norm(am.group("k")), am.group("v"))
                    break
            if hit is None:                                        # form B: "`density` ... optimum 0.25"
                tail = _BARE_NUM.match(line[om.end():].lstrip()[:12])
                before = line[max(0, om.start() - _NEAR):om.start()]
                marks = [g for m in _MARKED_PARAM.finditer(before) for g in m.groups() if g]
                if tail and marks:
                    hit = (_norm(marks[-1]), tail.group(1))
            if hit and hit[0] not in ("", "seed", "n", "p"):
                out.append((i, hit[0], hit[1]))
                break                                              # one claim per line is enough
    return out


def _scan(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as e:
        return ["%s: unreadable: %s" % (path, e)]
    head = text.split("---", 2)
    if not text.startswith("---") or len(head) < 3 or "status:" not in head[1]:
        return []                                                  # scope: frontmatter-bearing findings only
    lines = text.splitlines()
    probs = []
    for i, param, value in _claims(lines):
        window = "\n".join(lines[max(0, i - _WINDOW): i + _WINDOW + 1])
        others = _params_in(window) - {param}
        if others:
            continue
        if any(_artifact_records_config(a) for a in _ARTIFACT.findall(window)):
            continue
        probs.append(
            "%s:%d: optimum claim '%s = %s' names no OTHER axis held fixed within +/-%d lines "
            "(class 10: a single-axis optimum is CONDITIONAL -- record which axes were held and at what values, "
            "or cite an artifact that does)" % (os.path.relpath(path, _REPO), i + 1, param, value, _WINDOW))
    return probs


def check(paths):
    targets = [p for p in paths if p.endswith(".md") and "/raw/" not in p.replace(os.sep, "/")]
    targets = [p if os.path.isabs(p) else os.path.join(_REPO, p) for p in targets]
    if not paths:
        targets = sorted(glob.glob(_CORPUS))
    probs = []
    for p in targets:
        if os.path.isfile(p):
            probs += _scan(p)
    return probs


def _fixture(tmp, body):
    p = os.path.join(tmp, "2026-07-31-fixture.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(body)
    return p


def selftest():
    import tempfile
    bad = ("---\nstatus: live\nlane: gap#5\ndate: 2026-07-31\n---\n\n# result\n\n"
           "The read-`density` sweep gives an optimum 0.15 for place specificity.\n\nIt is the headline.\n")
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        # FAILING DIRECTION FIRST: the recorded incident verbatim -- an optimum with nothing held.
        if not check([_fixture(tmp, bad)]):
            out.append("MISS: an optimum-claim naming a parameter and value, with NO other axis pinned "
                       "and no artifact, was not flagged -- the gate cannot fail on its own incident")
        # ... and it must still fail when the claim is written as an assignment.
        if not check([_fixture(tmp, bad.replace("an optimum 0.15", "the optimal density=0.15"))]):
            out.append("MISS: 'optimal density=0.15' with no other axis pinned was not flagged")
        # Frontmatter is the scope: the same text without it is out of scope, not a violation.
        if check([_fixture(tmp, bad.split("---")[2])]):
            out.append("SCOPE: flagged a file with no status: frontmatter")
        # CALIBRATION -- these must NOT fire, or the gate cries wolf and gets ignored.
        for label, good in (
            ("held-fixed clause", bad.replace("It is the headline.", "Held fixed: `dwell`=180, `w_max`=150.")),
            ("config block", bad.replace("It is the headline.", "```\ndwell = 180\nw_max = 150\n```")),
            ("bare English 'best'", bad.replace("gives an optimum 0.15 for place specificity",
                                                "is the best evidence we have, at best a weak one")),
            ("no winning value", bad.replace("an optimum 0.15", "an optimum somewhere in the middle")),
        ):
            probs = check([_fixture(tmp, good)])
            if probs:
                out.append("FALSE POSITIVE (%s): %s" % (label, probs[0][:120]))
        # The artifact branch must actually be reachable.
        art = os.path.join(tmp, "raw")
        os.makedirs(art, exist_ok=True)
        with open(os.path.join(art, "cfg.json"), "w", encoding="utf-8") as fh:
            json.dump({"config": {"dwell": 180, "w_max": 150}}, fh)
        if not _artifact_records_config(os.path.relpath(os.path.join(art, "cfg.json"), _REPO)):
            out.append("BROKEN: artifact-evidence branch rejects a JSON that does record 2 config keys")
        if _artifact_records_config("raw/does-not-exist-%d.json" % os.getpid()):
            out.append("BROKEN: artifact-evidence branch accepts a nonexistent artifact")
    return out
