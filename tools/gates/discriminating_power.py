"""FAILURE CLASS 4 -- comparison with no discriminating power (7 incidents).

EVIDENCE. A place-specificity CONTROL agreed with its own treatment to <1e-6 in 29 of 36 arm-runs (often
1e-9) while the runner printed confident NEGATIVES. A transport ceiling of 0.130 equalled the reservoir
null of 0.130 against a chance level of 0.111 -- the CEILING tied the NULL, so no arm could have been
discriminated. And a metric reading EXACTLY 1.000 on every seed is a ceiling, not a result. Each time the
instrument had zero resolution while the verdict was written as if it had some. The honest verdict for a
tie is UNINTERPRETABLE, not negative.

WHAT IT CHECKS, in staged artifact JSONs only (given no paths it asks git for the staged set):
  (a1) STEM TIE -- a metric and its own control/null sibling in one JSON object, paired by NAME STEM
       ("r_real" vs "r_permuted", "score" vs "score_control"), agreeing to <1e-6.
  (a2) CROSS TIE -- a control/null and an unrelated-named metric in one object, equal to within 1e-12
       (the ceiling-ties-the-null case, where the names share no stem). Held to exact equality, and the
       treatment side must not be a parameter name (threshold/target/max/...): a knob deliberately SET to
       the measured null is legitimate.
  (b) CEILING/FLOOR -- a per-seed series (3..16 values, or one key across 3..16 per-seed dicts) that is
      EXACTLY 1.0 or EXACTLY 0.0 on every seed, in an artifact that also records a verdict -- i.e. the
      flat metric is being used as evidence.

CALIBRATION (every exclusion below is a false positive this gate actually produced against
research/findings/raw; together they take the hit rate from 7.5% to 2.2% of 7637 artifacts):
  * an INTEGER tie is config (n_seeds, steps) -- EXCEPT both arms at exactly 1.0, the
    treatment-and-its-lesion-both-at-ceiling case, which IS reported;
  * a tie at ~0 (<1e-4) is the ideal frozen control; a tie on a coarse rational k/n, n<=64 (two arms
    scoring 10/27) is DISCRETENESS, not degeneracy;
  * bools are not metrics (a list of `true` pass-flags is not a ceiling); NaN/inf is not a measurement;
  * ceiling/floor only on bounded [0,1] performance names -- a `*_ratio` of 1.0 means EQUAL, not maxed --
    and at the FLOOR error-ish and control-named keys are skipped: 0 false-accepts and a lesion pinned at
    0 are the GOAL, not a dead instrument;
  * FROZEN CONTROLS (stated, per the class spec): a deliberately frozen arm SHOULD tie its treatment, so
    an artifact whose text carries a freeze marker (lr/learning_rate/eta/plasticity*/gain set to 0, or an
    lr0/frozen/nolearn/plasticity_off key or arm name) is exempt from (a) WHOLE-FILE. File-scoped on
    purpose: it prefers missing a tie to crying wolf, because a gate that cries wolf gets ignored.

CANNOT CATCH: a control that merely differs by less than noise (a power question needing the seed
distribution, not a tie); a tie split across two files, or carried only in a .log or in prose; a ceiling on
a metric never written per-seed; a cross-named tie looser than 1e-12; a coarse-rational tie; a conceptually
wrong control that is numerically distinct; anything in a frozen-marked artifact. ADVISORY
(BLOCKING=False): a tie is strong evidence of zero resolution, but only a human can judge whether the
enclosing claim actually rests on that comparison.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
from fractions import Fraction

NAME = "discriminating-power"
CLASS_ID = "4"
BLOCKING = False

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MAX_BYTES = 4_000_000
_CTRL = ("control", "null", "lesioned", "lesion", "permuted", "permute", "shuffled", "shuffle",
         "scrambled", "scramble", "sham", "baseline", "untrained", "chance", "ablated", "ablation")
_TREAT = ("treatment", "treated", "treat", "intact", "trained", "learned", "actual", "real",
          "main", "full", "true", "test")
_FROZEN = re.compile(r'"[a-z_0-9]*(lr|learning_rate|eta|plasticity[a-z_]*|gain)"\s*:\s*0(\.0*)?\s*[,}\]]'
                     r'|"[a-z_0-9]*(lr0|lr_0|frozen|freeze|nolearn|no_learn|plasticity_off)[a-z_0-9]*"', re.I)
_VERDICT = re.compile(r'"(go|verdict|checks|conclusion|result|status)"\s*:', re.I)
_PERF = (r"acc|accuracy|correct|success|recall|precision|f1|auc|score|jaccard|corr|sim|similarity|"
         r"top1|top5|r2|selectivity")
_CEIL_RE = re.compile(r"(?:^|_)(%s|frac|fraction|prop|proportion|prob)(?:_|$)" % _PERF, re.I)
_FLOOR_RE = re.compile(r"(?:^|_)(%s)(?:_|$)" % _PERF, re.I)     # 0.0 is the GOAL for error-ish metrics
_FLOOR_SKIP = re.compile("|".join(("floor", "false", "err", "fail", "miss", "loss", "viol") + _CTRL), re.I)
_STRUCT = re.compile(r"^(n|num|count|seed|step|size|dim|len|idx|index|iter|epoch|trial|id)(_|$)"
                     r"|_(seed|seeds|steps|size|dim|len|count|n)$", re.I)
_PARAM = re.compile(r"thresh|target|cutoff|criter|bound|limit|max|min|weight|^w_|_w$|lr|alpha|beta|gain|"
                    r"scale|tau|dt|delay|drive|level|param|config|cfg|version", re.I)
_MSG = "%s: no discriminating power -- verdict UNINTERPRETABLE, not negative: "


def _stem(key, tokens):
    s = key.lower()
    for t in sorted(tokens, key=len, reverse=True):
        s = s.replace(t, "")
    return re.sub(r"[^a-z0-9]", "", s)


def _num(v):   # bools are not metrics (a list of pass-flags is not a ceiling); NaN/inf is not a reading
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _coarse(v):
    """v is a simple rational k/n, n<=64: two arms scoring 10/27 tie by DISCRETENESS, not degeneracy."""
    return abs(float(Fraction(v).limit_denominator(64)) - v) < 1e-9


def _walk(node, path, dicts, lists, depth=0):
    if depth > 40:
        return                                          # a pathological nesting must not crash the hook
    if isinstance(node, dict):
        dicts.append((path or "<root>", node))
        for k, v in node.items():
            _walk(v, "%s.%s" % (path, k) if path else str(k), dicts, lists, depth + 1)
    elif isinstance(node, list):
        lists.append((path or "<root>", node))
        for i, v in enumerate(node):
            if isinstance(v, (dict, list)):
                _walk(v, "%s[%d]" % (path, i), dicts, lists, depth + 1)


def _ties(rel, dicts, frozen, has_verdict):
    if frozen:
        return []                                       # stated exclusion: a frozen arm SHOULD tie
    out = []
    for where, d in dicts:
        nums = {k: float(v) for k, v in d.items() if _num(v) and not _STRUCT.match(k)}
        ctrls = [k for k in nums if any(t in k.lower() for t in _CTRL)]
        for kc in ctrls:
            c, sc = nums[kc], _stem(kc, _CTRL)
            for kt, t in nums.items():
                if kt in ctrls:
                    continue
                same_stem = _stem(kt, _TREAT) == sc
                if not (same_stem and t == 1.0 and c == 1.0):    # both arms pinned at the 1.0 ceiling
                    if float(t).is_integer() and float(c).is_integer():
                        continue                        # an integer tie is config (n_seeds, steps)
                    if max(abs(t), abs(c)) < 1e-4 or _coarse(t):
                        continue                        # ~0 is the ideal control; k/n is a coincidence
                tol = 1e-6 if same_stem else 1e-12
                if abs(t - c) >= tol or (not same_stem and (_PARAM.search(kt) or not has_verdict)):
                    continue                            # a knob SET to the null is legitimate
                out.append((_MSG + "'%s'=%.10g ties its %s '%s'=%.10g (|diff|=%.1e < %.0e) at %s; no "
                            "lr=0/frozen marker in this artifact") %
                           (rel, kt, t, "control" if same_stem else "unrelated-named control",
                            kc, c, abs(t - c), tol, where))
    return out


def _flat(rel, lists, has_verdict):
    if not has_verdict:
        return []                                       # raw data with no verdict is not "evidence"
    out, series = [], []
    for where, li in lists:
        if not 3 <= len(li) <= 16:
            continue                                    # a long array is data, not a per-seed series
        if all(_num(v) for v in li):
            series.append((where, where.rsplit(".", 1)[-1], [float(v) for v in li]))
        elif all(isinstance(v, dict) for v in li):
            for k in sorted(set(li[0]).intersection(*[set(d) for d in li[1:]])):
                if all(_num(d[k]) for d in li):
                    series.append(("%s[].%s" % (where, k), k, [float(d[k]) for d in li]))
    for where, key, vals in series:
        if _STRUCT.match(key):
            continue
        for lvl, word, named in ((1.0, "ceiling", _CEIL_RE.search(key)),
                                 (0.0, "floor", _FLOOR_RE.search(key) and not _FLOOR_SKIP.search(key))):
            if named and all(v == lvl for v in vals):
                out.append((_MSG + "'%s' is EXACTLY %.1f on all %d seeds/runs (%s) at %s, in an artifact "
                            "that reports a verdict") % (rel, key, lvl, len(vals), word, where))
    return out


def check(paths):
    # An EMPTY list means "staged mode, nothing of my kind staged" -> nothing to check. Only paths=None means
    # "standalone run, scan the corpus". Without this, the pre-commit driver's --diff-filter=A scoping is undone
    # by this gate's own corpus fallback -- which fired 192 doc-type hits on 2026-04/05 legacy findings the
    # moment the Tier-1 classification gave them frontmatter.
    if paths is not None and len(paths) == 0:
        return []
    if not paths:
        try:
            r = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
                               cwd=_ROOT, capture_output=True, text=True, timeout=30)
            paths = [os.path.join(_ROOT, p) for p in r.stdout.split("\n") if p.strip()]
        except (OSError, subprocess.SubprocessError):
            return []                                   # no git => no staged set; nothing to judge
    problems = []
    for p in paths:
        if not p.endswith(".json") or not os.path.isfile(p) or os.path.getsize(p) > _MAX_BYTES:
            continue
        try:
            text = open(p, "r", encoding="utf-8", errors="replace").read()
            data = json.loads(text)
        except OSError as e:
            problems.append("%s: unreadable (%s) -- gate could not run" % (p, e))
            continue
        except json.JSONDecodeError:
            continue                                    # malformed JSON is another gate's class
        dicts, lists = [], []
        _walk(data, "", dicts, lists)
        rel = os.path.relpath(p, _ROOT) if p.startswith(_ROOT) else p
        verdict = bool(_VERDICT.search(text))
        problems += _ties(rel, dicts, bool(_FROZEN.search(text)), verdict) + _flat(rel, lists, verdict)
    return problems


def selftest():
    """Written failing-direction first: every case below is one the gate MUST catch or MUST NOT fire on.
    Each MUST-NOT case is a false positive this gate actually produced against research/findings/raw."""
    import tempfile

    def fix(tmp, name, obj):
        with open(os.path.join(tmp, name), "w", encoding="utf-8") as f:
            json.dump(obj, f)
        return os.path.join(tmp, name)

    catch = [
        ("a1 stem tie at 1e-10", {"verdict": "NEGATIVE: place code is not specific",
                                  "means": {"r_real": 0.4712345678, "r_permuted": 0.4712345679}}),
        ("a1 treatment and its lesion both pinned at 1.0",
         {"verdict": "GO", "mean_deliberates_intact": 1.0, "mean_deliberates_lesion": 1.0}),
        ("a2 ceiling equals its null", {"go": True, "transport": 0.130, "reservoir_null": 0.130}),
        ("b metric EXACTLY 1.0 on 6 seeds",
         {"go": True, "per_seed": [{"seed": s, "accuracy": 1.0} for s in range(6)]}),
        ("b metric EXACTLY 0.0 on 6 seeds", {"verdict": "NO-GO", "held_acc": [0.0] * 6}),
    ]
    quiet = [
        ("a separated treatment/control", {"go": True, "means": {"r_real": 0.81, "r_permuted": -0.06},
                                           "per_seed": [{"acc": 0.7 + s / 100} for s in range(6)]}),
        ("a declared lr=0 frozen arm", {"go": True, "lr": 0.0, "circ": 0.35877, "circ_control": 0.35877}),
        ("int config + boolean checks", {"go": True, "n_steps": 1800, "n_steps_control": 1800,
                                         "checks": [True] * 6}),
        ("boolean pass-flags", {"go": True, "per_seed": [{"success": True} for _ in range(6)]}),
        ("a verdict-less raw artifact", {"held_acc": [1.0] * 6}),
        ("a threshold SET to the null", {"go": True, "threshold": 0.13, "reservoir_null": 0.13}),
        ("a tie at ~0 (the ideal control)", {"go": True, "drift": 0.0, "drift_control": 0.0}),
        ("a discrete 10/27 tie", {"go": True, "plain_acc": 10 / 27, "permuted_inherit": 10 / 27}),
        ("a control pinned at the floor",
         {"go": True, "per_seed": [{"lesion_recall": 0.0} for _ in range(6)]}),
        ("a ratio of 1.0 (equal, not maxed)", {"go": True, "actor_ratio": [1.0] * 6}),
        ("zero false-accepts (the goal)", {"go": True, "moat_false_accept": [0.0] * 6}),
        ("NaN is not a measurement", {"go": True, "score": float("nan"), "score_control": float("nan")}),
    ]
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (label, obj) in enumerate(catch):
            if not check([fix(tmp, "c%d.json" % i, obj)]):
                bad.append("MISS -- the gate did NOT catch: %s" % label)
        for i, (label, obj) in enumerate(quiet):
            hits = check([fix(tmp, "q%d.json" % i, obj)])
            if hits:
                bad.append("FALSE POSITIVE on %s: %s" % (label, hits[0][:110]))
        if check([os.path.join(tmp, "absent.txt")]):
            bad.append("FALSE POSITIVE: a non-JSON path fired")
    return bad
