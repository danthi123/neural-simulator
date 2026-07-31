"""FAILURE CLASS 4 -- comparison with no discriminating power (7 incidents).

EVIDENCE. A place-specificity CONTROL agreed with its own treatment to <1e-6 in 29 of 36 arm-runs (often
1e-9) while the runner printed confident NEGATIVES. A transport ceiling of 0.130 equalled the reservoir
null of 0.130 against a chance level of 0.111 -- the CEILING tied the NULL, so no arm could ever have been
discriminated. And a metric reading EXACTLY 1.000 on every seed is a ceiling, not a result. In all of them
the instrument had zero resolution and the verdict was reported as if it had some: the correct verdict is
UNINTERPRETABLE, not negative.

WHAT THIS GATE DOES (staged artifact JSONs only; with no paths it asks git for the staged set):
  (a) TIE -- a numeric metric and its own control/null sibling in the same JSON object agreeing to <1e-6.
      Pairing is by NAME STEM ("r_real" vs "r_permuted", "score" vs "score_control"), so unrelated numbers
      are never compared. Both values must be non-integral and >=1e-4 in magnitude: an integer tie is
      almost always config (n_seeds, steps) and a tie at ~0 is the ideal frozen control, not a defect.
  (b) CEILING/FLOOR -- a per-seed metric list (3..16 values, or the same key across 3..16 per-seed dicts)
      that is EXACTLY 1.0 or EXACTLY 0.0 on every seed, in an artifact that also records a verdict
      (go / verdict / checks / conclusion) -- i.e. the flat metric is being used as evidence.

FROZEN-CONTROL EXCLUSION (stated, per the class spec). A deliberately frozen arm SHOULD tie its treatment
on the frozen quantity -- that is the ideal control, not a defect. If the artifact text records a freeze
marker anywhere (lr/learning_rate/eta/plasticity* set to 0, an lr0/frozen/nolearn/plasticity_off key or
arm name), the WHOLE FILE is exempt from (a). This is deliberately file-scoped and therefore conservative:
it prefers missing a real tie to crying wolf, because a gate that cries wolf gets ignored.

WHAT THIS GATE CANNOT CATCH: a control that is merely too weak (differs, but by less than noise -- that is
a power question needing the seed distribution, not a tie); a tie between two files (arms written to
separate artifacts); a tie carried only in a .log or in prose; a ceiling on a metric the artifact never
names per-seed; a control that is conceptually wrong but numerically distinct; anything inside a
frozen-marked artifact. It is ADVISORY (BLOCKING=False): "tie" is strong evidence of no resolution, but
only a human can decide whether the enclosing claim actually rests on that comparison.
"""
from __future__ import annotations

import json
import os
import re
import subprocess

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
                     r'|"[a-z_0-9]*(lr0|lr_0|frozen|freeze|nolearn|no_learn|plasticity_off)[a-z_0-9]*"',
                     re.IGNORECASE)
_VERDICT = re.compile(r'"(go|GO|verdict|checks|conclusion|result|status)"\s*:', re.IGNORECASE)
_METRIC = re.compile(r"acc|score|corr|sim|auc|f1|precis|recall|hit|success|rate|frac|ratio|prob|prop|"
                     r"top\d|sep|margin|discrim|selectiv|r2|perf|reward|match|win|jaccard|dist|err", re.I)
_STRUCTURAL = re.compile(r"^(n|num|count|seed|step|size|dim|len|idx|index|iter|epoch|trial|id)(_|$)"
                         r"|_(seed|seeds|steps|size|dim|len|count|n)$", re.I)


def _stem(key, tokens):
    s = key.lower()
    for t in sorted(tokens, key=len, reverse=True):
        s = s.replace(t, "")
    return re.sub(r"[^a-z0-9]", "", s)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _walk(node, path, dicts, lists):
    if isinstance(node, dict):
        dicts.append((path, node))
        for k, v in node.items():
            _walk(v, "%s.%s" % (path, k) if path else str(k), dicts, lists)
    elif isinstance(node, list):
        lists.append((path, node))
        for i, v in enumerate(node):
            if isinstance(v, (dict, list)):
                _walk(v, "%s[%d]" % (path, i), dicts, lists)


def _ties(path, dicts, frozen):
    if frozen:
        return []                                    # stated exclusion: a frozen arm SHOULD tie
    out = []
    for where, d in dicts:
        nums = {k: float(v) for k, v in d.items() if _is_num(v) and not _STRUCTURAL.match(k)}
        ctrls = [k for k in nums if any(t in k.lower() for t in _CTRL)]
        for kc in ctrls:
            sc = _stem(kc, _CTRL)
            for kt, t in nums.items():
                if kt in ctrls or _stem(kt, _TREAT) != sc:
                    continue
                c = nums[kc]
                if float(t).is_integer() and float(c).is_integer():
                    continue                          # integer tie => config, not a measurement
                if max(abs(t), abs(c)) < 1e-4 or abs(t - c) >= 1e-6:
                    continue
                out.append("%s: no discriminating power -- verdict UNINTERPRETABLE, not negative: "
                           "'%s'=%.10g ties its control '%s'=%.10g (|diff|=%.1e < 1e-6) at %s; no "
                           "lr=0/frozen marker in this artifact" %
                           (path, kt, t, kc, c, abs(t - c), where or "<root>"))
    return out


def _flat_series(path, dicts, lists, has_verdict):
    if not has_verdict:
        return []                                    # raw data with no verdict is not "used as evidence"
    out, series = [], []
    for where, li in lists:
        if 3 <= len(li) <= 16 and all(_is_num(v) for v in li):
            series.append((where, where.rsplit(".", 1)[-1], [float(v) for v in li]))
        elif 3 <= len(li) <= 16 and all(isinstance(v, dict) for v in li):
            keys = set(li[0])
            for k in sorted(keys.intersection(*[set(d) for d in li[1:]])):
                if all(_is_num(d[k]) for d in li):
                    series.append(("%s[].%s" % (where, k), k, [float(d[k]) for d in li]))
    for where, key, vals in series:
        if _STRUCTURAL.match(key) or not _METRIC.search(key):
            continue
        for lvl, word in ((1.0, "ceiling"), (0.0, "floor")):
            if all(v == lvl for v in vals):
                out.append("%s: no discriminating power -- verdict UNINTERPRETABLE, not negative: "
                           "'%s' is EXACTLY %.1f on all %d seeds/runs (%s) at %s, in an artifact that "
                           "reports a verdict" % (path, key, lvl, len(vals), word, where))
    return out


def _staged():
    try:
        r = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
                           cwd=_ROOT, capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as e:   # no git / not a repo: nothing staged to judge
        return [], "git unavailable: %s" % e
    return [os.path.join(_ROOT, p) for p in r.stdout.split("\n") if p.strip()], None


def check(paths):
    if not paths:
        paths, _err = _staged()
    problems = []
    for p in paths:
        if not p.endswith(".json") or not os.path.isfile(p):
            continue
        try:
            if os.path.getsize(p) > _MAX_BYTES:
                continue
            text = open(p, "r", encoding="utf-8", errors="replace").read()
            data = json.loads(text)
        except (OSError, UnicodeError) as e:
            problems.append("%s: unreadable (%s) -- gate could not run" % (p, e))
            continue
        except json.JSONDecodeError:
            continue                                  # malformed JSON is another gate's class
        dicts, lists = [], []
        _walk(data, "", dicts, lists)
        rel = os.path.relpath(p, _ROOT) if p.startswith(_ROOT) else p
        problems += _ties(rel, dicts, bool(_FROZEN.search(text)))
        problems += _flat_series(rel, dicts, lists, bool(_VERDICT.search(text)))
    return problems


def _fixture(tmp, name, obj):
    p = os.path.join(tmp, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    return p


def selftest():
    import tempfile
    bad = []
    with tempfile.TemporaryDirectory() as tmp:
        # --- MUST CATCH (the failing direction) ---
        tie = _fixture(tmp, "tie.json", {"verdict": "NEGATIVE: place code is not specific",
                                         "means": {"r_real": 0.4712345678, "r_permuted": 0.4712345679}})
        if not check([tie]):
            bad.append("MISS (a): a treatment tying its permuted control to 1e-10 was NOT caught")
        ceil2 = _fixture(tmp, "ceil2.json", {"go": True, "transport": 0.130, "reservoir_null": 0.130})
        if not check([ceil2]):
            bad.append("MISS (a): a ceiling equal to its null (0.130 vs 0.130) was NOT caught")
        flat = _fixture(tmp, "flat.json", {"go": True,
                                           "per_seed": [{"seed": s, "accuracy": 1.0} for s in range(6)]})
        if not check([flat]):
            bad.append("MISS (b): a metric at EXACTLY 1.0 on all 6 seeds was NOT caught")
        zero = _fixture(tmp, "zero.json", {"verdict": "NO-GO", "hit_rate": [0.0] * 6})
        if not check([zero]):
            bad.append("MISS (b): a metric at EXACTLY 0.0 on all 6 seeds was NOT caught")
        # --- MUST NOT FIRE (calibration: a gate that cries wolf gets ignored) ---
        ok = _fixture(tmp, "ok.json", {"go": True, "means": {"r_real": 0.81, "r_permuted": -0.06},
                                       "per_seed": [{"seed": s, "accuracy": 0.7 + s / 100} for s in range(6)]})
        if check([ok]):
            bad.append("FALSE POSITIVE: a genuinely separated treatment/control fired: %s" % check([ok]))
        frozen = _fixture(tmp, "frozen.json", {"go": True, "lr": 0.0,
                                               "circ": 0.3587736905, "circ_control": 0.3587736905})
        if check([frozen]):
            bad.append("FALSE POSITIVE: a declared lr=0 frozen control fired: %s" % check([frozen]))
        cfg = _fixture(tmp, "cfg.json", {"go": True, "n_steps": 1800, "n_steps_control": 1800,
                                         "checks": [True, True, True, True, True, True]})
        if check([cfg]):
            bad.append("FALSE POSITIVE: integer config + boolean checks fired: %s" % check([cfg]))
        raw = _fixture(tmp, "raw.json", {"rates": [1.0] * 6})   # no verdict => not used as evidence
        if check([raw]):
            bad.append("FALSE POSITIVE: a verdict-less raw artifact fired: %s" % check([raw]))
        if check([os.path.join(tmp, "nope.txt")]):
            bad.append("FALSE POSITIVE: a non-JSON path fired")
    return bad
