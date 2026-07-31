"""FAILURE CLASS 10 -- single-axis-sweep-reported-as-absolute (3 incidents).

EVIDENCE. A sweep varies one axis while every other axis sits at whatever value it happened to have, and the
winner is written down as *the* optimum. The gap#5 read-density sweep reported an optimum of 0.15 that, once a
co-varying axis (`w_max`, inside the plasticity bound trap) was moved to a correct bound, scored BELOW density
1.0 -- the answer was INVERTED, not merely narrow. The session note: "the same error I made with the density
optimum itself... I keep reporting them as absolute ones". The cost is that the next finding tunes on top of a
conditional answer whose conditions nobody wrote down.

THE GATE. In a frontmatter-bearing finding, an optimum-claim that names a parameter AND its winning value must
carry, within +/-8 lines, either (a) at least one OTHER axis pinned to a value ("at k=v", "k held at v", "dwell
30", a config block, a command line), or (b) a cited artifact that EXISTS and records >=2 config keys.

CALIBRATION (all 1841 findings, scope filter lifted): 49 optimum-claims detected, 3 unconditioned -- `N_BIAS=6`
and `prop_k=16` sweet spots stated with no other axis, plus one where the claim binds to the wrong token. ~1
flag per 600 files: it will not train anyone to ignore it. Every false-positive shape found during that pass is
a case in selftest(), so re-widening the detector fails loudly.

WHAT THIS GATE CANNOT CATCH.
  * Whether the held values were CORRECT. The gap#5 optimum was stated with `dwell=180` beside it and was still
    inverted, because `w_max=150` sat inside the bound trap (class 2). This enforces DISCLOSURE of the
    conditioning, never its validity -- on the incident itself it fires only if nothing at all was recorded.
  * "the optimum at density=0.15" -- an `at k=v` clause reads as conditioning even when the only axis named is
    the claimed one. Closing that re-flags every legitimate "best X at k=v"; the trade was made deliberately.
  * Unmarked English parameters ("the best setting was 3"), claims split over two lines, claims with no winning
    value, categorical claims ("best config = WITHOUT disjoint-DG"), and "a peak at -1" (a result location).
  * Findings with no `status:` frontmatter, and anything under research/findings/raw/.
Advisory (BLOCKING=False): the detector reads prose, precision ~2/3 on the calibration sample -- worth reading,
not worth blocking a commit over.
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

# Bare words this project uses as parameters. SHORT on purpose: each entry is a new chance to fire on English.
_VOCAB = ("density", "dwell", "sparsity", "threshold", "epochs", "lr", "gain", "sigma", "temperature")
# Metadata + measurements: they carry a number but are not swept axes. Excluding them matters in BOTH directions
# -- as a claim target they cry wolf ("the new best (4.41, p=0.018)"), and as evidence a nearby `acc=0.458` would
# silently satisfy the conditioning requirement, which is a check that cannot fail: failure class 3.
_NOT_AXIS = frozenset("""date status lane mechanism claim_check seed seeds figure fig table section line commit
    page ref eq footnote cycle day p n k result results score acc accuracy mean means median sum total ratio
    count runs configs permutation perm finalq distance dist loss auc f1 pct percent delta sem std sd ci chance
    baseline narrow excess improvement margin value""".split())
_NOT_AXIS_SUB = ("acc", "score", "err", "loss", "corr", "pval", "p_value", "_mean", "mean_", "_sum", "sum_",
                 "circ", "_pct", "_ratio", "_dw", "d_")

_NUM = r"-?\d+(?:\.\d+)?(?:e-?\d+)?"
_MARK = r"`[A-Za-z_][A-Za-z0-9_.-]{0,40}`|--[a-z][a-z0-9_-]{1,40}|\b[A-Za-z]+_[A-Za-z0-9_]+\b|\b(?:%s)\b" \
        % "|".join(_VOCAB)
_OPT = re.compile(r"\b(optimum|optimal|optima|best|peaks?\s+at|sweet\s+spot)\b", re.I)
_ASSIGN = re.compile(r"(?P<mark>`|--)?(?P<k>[A-Za-z_][A-Za-z0-9_.-]{0,40})`?\s*"
                     r"(?P<sep>=|:|\bheld\s+at\b|\bat\b|\bof\b)\s*\**`?(?P<v>%s)" % _NUM, re.I)
_SPACED = re.compile(r"(?P<k>%s)\s+\**`?(?P<v>%s)\b" % (_MARK, _NUM), re.I)     # "dwell 30", "sat_frac 0.000"
_CLI = re.compile(r"--(?P<k>[a-z][a-z0-9_-]{1,40})[ =]\**`?(?P<v>%s)" % _NUM, re.I)
_WINNER = re.compile(r"^[\s:=,(]*(?:(?P<p>%s)\s*(?:=|:|at|of)?\s*)?\**`?(?P<v>%s)\b" % (_MARK, _NUM), re.I)
_MARKED_PARAM = re.compile(_MARK, re.I)
_AT = re.compile(r"\bat\s+$", re.I)              # "best X at k=v": k is the CONDITION, not the claim
_TRAILING = re.compile(r"^\s*(?:/|%|pp\b|x\b|×|-?\s*seeds?\b)", re.I)   # 20/20, 6-seed, 18% -- counts, not settings
_ARTIFACT = re.compile(r"(?:research/findings/)?raw/[\w./*-]+\.json")


def _norm(k):
    return k.strip("`*-_ ").lower()


def _is_axis(k):
    return bool(k) and k not in _NOT_AXIS and not any(s in k for s in _NOT_AXIS_SUB)


def _pins(text):
    """(start, end, key, claimable, value) for every axis pinned to a number in `text`.

    `claimable` = written as a parameter (backticks, `--`, an underscore, the vocabulary) and not introduced by
    "at". Claims require it; conditioning evidence accepts every pin.
    """
    out = []
    for rx in (_ASSIGN, _SPACED, _CLI):
        for m in rx.finditer(text):
            k = _norm(m.group("k"))
            if not _is_axis(k) or _TRAILING.match(text[m.end():m.end() + 8]):
                continue
            gd = m.groupdict()
            marked = rx is not _ASSIGN or bool(gd.get("mark")) or "_" in m.group("k") or k in _VOCAB
            if not marked and gd.get("sep") not in ("=", ":"):
                continue
            out.append((m.start(), m.end(), k, marked and not _AT.search(text[max(0, m.start() - 6):m.start()]),
                        m.group("v")))
    return out


def _held_axes(text):
    return {k for _s, _e, k, _c, _v in _pins(text)}


def _artifact_records_config(rel, base=None):
    """True if a cited artifact exists and records >=2 config-looking keys.

    Resolved against the citing document's own directory first (that is what `raw/x.json` means in a finding),
    then the repo root.
    """
    cands = [rel] if os.path.isabs(rel) else [os.path.join(base or "", rel), os.path.join(_REPO, rel),
                                              os.path.join(_REPO, "research/findings", rel)]
    for cand in [c for pat in cands for c in glob.glob(pat)][:4]:
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
        if any(sum(1 for v in pool.values() if isinstance(v, (int, float, bool, str))) >= 2 for pool in pools):
            return True
    return False


def _claims(lines):
    """(line_index, axis, winning_value) for each optimum-claim that names an axis AND its winning value."""
    out = []
    for i, line in enumerate(lines):
        for om in _OPT.finditer(line):
            hit = None
            for s, e, k, claimable, v in _pins(line):              # form A: "the optimal density=0.15"
                if claimable and min(abs(s - om.end()), abs(om.start() - e)) <= _NEAR:
                    hit = (k, v)
                    break
            if hit is None:                                        # form B: "`density` ... optimum 0.15"
                tail = line[om.end():om.end() + 48]
                w = _WINNER.match(tail)
                if w and not _TRAILING.match(tail[w.end():]):      # "the best 6-seed result" is not an optimum
                    name = w.group("p")
                    if name is None and not om.group(0).lower().startswith("peak"):
                        pre = _MARKED_PARAM.findall(line[max(0, om.start() - _NEAR):om.start()])
                        name = pre[-1] if pre else None            # the axis was named BEFORE the optimum word
                    # "with a peak at -1" names no axis: that is a result location, not a parameter optimum.
                    if name and _is_axis(_norm(name)):
                        hit = (_norm(name), w.group("v"))
            if hit:
                out.append((i, hit[0], hit[1]))
                break                                              # one claim per line is enough
    return out


def _scan(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError as e:
        return ["%s: unreadable: %s" % (path, e)]
    if not text.startswith("---") or "status:" not in text.split("---", 2)[1]:
        return []                                                  # scope: frontmatter-bearing findings only
    lines = text.splitlines()
    end = next((j for j in range(1, len(lines)) if lines[j].strip() == "---"), 0)
    lines = [""] * (end + 1) + lines[end + 1:]     # blank the frontmatter: `date: 2026-07-31` is not a held axis
    probs = []
    for i, axis, value in _claims(lines):
        window = "\n".join(lines[max(0, i - _WINDOW): i + _WINDOW + 1])
        if _held_axes(window) - {axis}:
            continue
        if any(_artifact_records_config(a, os.path.dirname(path)) for a in _ARTIFACT.findall(window)):
            continue
        probs.append("%s:%d: optimum claim '%s = %s' records no OTHER axis held fixed within +/-%d lines "
                     "(class 10: a single-axis optimum is CONDITIONAL -- state which axes were held and at what "
                     "values, or cite an artifact that does)"
                     % (os.path.relpath(path, _REPO), i + 1, axis, value, _WINDOW))
    return probs


def check(paths):
    targets = [p if os.path.isabs(p) else os.path.join(_REPO, p)
               for p in paths if p.endswith(".md") and "/raw/" not in p.replace(os.sep, "/")]
    if not paths:
        targets = sorted(glob.glob(_CORPUS))
    probs = []
    for p in targets:
        if os.path.isfile(p):
            probs += _scan(p)
    return probs


def _fx(tmp, body):
    p = os.path.join(tmp, "2026-07-31-fixture.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(body)
    return p


def selftest():
    import tempfile
    fm = "---\nstatus: live\nlane: gap#5\ndate: 2026-07-31\n---\n\n# result\n\n"
    bad = fm + "The read-`density` sweep gives an optimum 0.15 for place specificity.\n\nIt is the headline.\n"
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        # FAILING DIRECTION FIRST. Each of these MUST be caught; `date:`/`lane:` in the frontmatter must not
        # count as the held axis (that bug made the first draft of this gate unfailable).
        for label, txt in (
            ("the recorded incident", bad),
            ("assignment form", bad.replace("an optimum 0.15", "the optimal density=0.15")),
            ("peaks-at form", bad.replace("gives an optimum 0.15", "peaks at `density` 0.15")),
            ("a metric is not a held axis", bad.replace("It is the headline.", "Held-out acc=0.458, p=0.03.")),
        ):
            if not check([_fx(tmp, txt)]):
                out.append("MISS (%s): an optimum-claim with NO other axis pinned and no artifact was not "
                           "flagged -- the gate cannot fail on its own incident" % label)
        if check([_fx(tmp, bad.split("---")[2])]):
            out.append("SCOPE: flagged a file with no status: frontmatter")
        # CALIBRATION. Every one of these shapes produced a false positive during the 1841-file calibration pass;
        # if any fires again the gate is crying wolf and will be ignored.
        for label, txt in (
            ("held-fixed clause", bad.replace("It is the headline.", "Held fixed: `dwell`=180, `w_max`=150.")),
            ("config block", bad.replace("It is the headline.", "```\ndwell = 180\nw_max = 150\n```")),
            ("spaced pins", bad.replace("It is the headline.", "At dwell 30, sat_frac 0.000, ONE lap.")),
            ("cited artifact", bad.replace("It is the headline.", "Raw: `raw/cfg.json`.")),
            ("bare English", fm + "This is the best evidence we have, at best a weak one.\n"),
            ("no winning value", bad.replace("an optimum 0.15", "an optimum somewhere in the middle")),
            ("metric optimum", fm + "The new best is 6-seed 4.41 (p=0.018, n=6, finalQ=4.44).\n"),
            ("count not a setting", fm + "`n_bins_potentiated = 20/20` on every seed, with a peak at 1.\n"),
            ("at-clause conditions it", fm + "The best weight-decay at `n_pool`=1000 was 0.01 on 6 seeds.\n"),
        ):
            if label == "cited artifact":                          # NEVER written inside the repo
                os.makedirs(os.path.join(tmp, "raw"), exist_ok=True)
                with open(os.path.join(tmp, "raw", "cfg.json"), "w", encoding="utf-8") as fh:
                    json.dump({"dwell": 180, "w_max": 150}, fh)
            probs = check([_fx(tmp, txt)])
            if probs:
                out.append("FALSE POSITIVE (%s): %s" % (label, probs[0].split(" (class 10")[0]))
        # The artifact branch must be reachable in both directions.
        with open(os.path.join(tmp, "cfg.json"), "w", encoding="utf-8") as fh:
            json.dump({"config": {"dwell": 180, "w_max": 150}}, fh)
        if not _artifact_records_config(os.path.join(tmp, "cfg.json")):
            out.append("BROKEN: artifact-evidence rejects a JSON that does record 2 config keys")
        if _artifact_records_config("raw/does-not-exist-%d.json" % os.getpid()):
            out.append("BROKEN: artifact-evidence accepts a nonexistent artifact")
    return out
