"""CLASS OP — OPERATING POINT: the artifact records a target AND the value that missed it, and nobody looked.

THE FAILURE CLASS (cause 2 of `research/findings/2026-07-31-why-we-hit-walls-the-missing-companion-process.md`).
A paper gives you the MECHANISM; the OPERATING POINT it runs at is implicit in the animal and never written down,
so we pick it by tuning and never check it. The measured instance: lane D's normalization arms ran V1 at a mean
firing rate of **0.0043** against a homeostatic target of **0.002** — 2.15x hot — and BOTH numbers are sitting in
the SAME artifact, `research/findings/raw/laneD_norm/AGG_norm_arms.json` (`homeo_target` at the root,
`v1_firing_rate` inside every arm). Nothing compared them. The arm that actually HIT the target (`ms0.75`,
0.0020) was read as a middling sweep point rather than as the only run at its own declared operating point.
The sibling instance in the same findings doc — gap#4 running at E~0.04 where phi'=E(1-E) vanishes ~1600x over
depth — is the same shape and is NOT catchable here (see WHAT THIS GATE CANNOT CATCH: no target was recorded).

THE GATE. For each `*.json` under `research/findings/raw/`: if the artifact records BOTH
  (a) a numeric SET-POINT — a key that unambiguously names one (`homeo_target`, `homeostasis_target_rate`,
      `target_firing_rate`, `firing_rate_setpoint`, ...); and
  (b) a numeric ACHIEVED value from the matching family (`*firing_rate*`, `*active_rate*`, `*activity_level*`,
      `frac_active`, `sparsity`, ... with the usual `_mean`/`_final` stat suffixes),
then any achieved value outside [0.5x, 1.5x] of the set-point is a problem. **It never INFERS a target.** An
artifact with a firing rate and no recorded target passes, always — that is the deliberate scope limit, not an
oversight, and it is why the gap#4 instance above is out of reach.

WHY THESE EXACT RULES. Four narrowings; each was a false positive OBSERVED on this corpus while building the
gate. The ablation is measured, not asserted — every figure below is `flagged` from `corpus_scan()` with that
one rule removed (reproduce by monkeypatching the named symbol):
                                                          artifacts flagged   problem lines
      SHIPPED, all four rules                                     1                 1
      minus the bare-`target_rate` exclusion                      1                 1
      minus the achieved allow-list (`_ACH_RE`)                   8                 8
      minus the mechanism-off guard (`_mechanism_off`)            4                10
      minus the same-units guard (`_regime`)                      1                 1
      ALL FOUR DROPPED                                           58               175
  · **A generic `*_rate` is NOT an achieved firing rate** (+7 artifacts when dropped). `homeo_target=40.0` vs
    `scored_rate=2168.0` reads as a 53x miss and is two unrelated quantities — `_cortex_conversation_*`,
    `_multibridge_*`, `_production_cortex_4bridge`, `_phase1_composer_*`. The achieved side is an explicit STEM
    allow-list, never a suffix match.
  · **A disabled mechanism is not an operating point** (+3 artifacts when dropped). `_volley_n800_ablate`,
    `_volley_n800_norhythm` and `_volley_ping_n800_STEP1_GO` each record `homeostasis_target_rate=0.035` next
    to `place_homeostasis: false`. A set-point whose enable-flags are ALL false is skipped.
  · **Bare `target_rate` is NOT a set-point** and **cross-unit pairs are refused**: BOTH cost zero on today's
    corpus and are kept anyway, honestly labelled as forward guards. 96 files carry `target_rate` meaning "the
    firing rate OF the target word" (`catastrophic_forgetting_probe_*`, `silent_interval_*`, `direct_binding_*`
    pair it with `top_rate`); it costs nothing only because `top_rate` is not in the achieved allow-list, and
    the day someone writes `firing_rate` beside it, it would fire on a measurement. Same for units: nothing
    today pairs a 0.002 fraction with a 40.0 Hz figure, but an early draft of this gate reported exactly that
    as a 53x miss. Dropping all four at once gives 58 artifacts / 175 lines — the guards interlock.
  · **Scoping by subtree.** A set-point nested at `/per_run[2]/` is compared only against values under
    `/per_run[2]/`; one at the root (or inside `args`/`config`/`params`) applies to the whole artifact.

CALIBRATION over the full corpus (7151 artifacts; `python -m tools.gates.operating_point`, ~1s): **1 artifact
flags** — `laneD_norm/AGG_norm_arms.json`, the case that motivated the gate. Verified lossless: parsing all
7151 files instead of byte-filtering to 25 gives an IDENTICAL problem list, and raising the traversal caps to
depth=40 / list=100000 / leaves=5e6 changes the verdict by zero artifacts. A gate that flags hundreds of legacy
files gets switched off, which is worse than no gate; this one is scoped so its full-corpus audit IS its
evidence. `check([])` returns nothing (nothing of this kind staged); `check(None)` runs the audit.

ESCAPE HATCH. A top-level non-empty `"operating_point_ack": "<why>"` exempts the artifact. That is not a
loophole — the failure being prevented is that NOBODY NOTICED, so writing down "ran 2.15x hot, homeostasis was
too slow to converge in 400 epochs" IS the fix. What must not happen is the number passing in silence.

WHAT THIS GATE CANNOT CATCH.
  · **A target the artifact never recorded** — by construction, and this is the majority of the failure class.
    The gap#4 phi'=E(1-E) case, the BTSP-protocol case, and every run whose set-point lived only in argv or in
    the paper are all invisible here. This gate makes the RECORDED contradiction impossible to miss; it does
    nothing about the unrecorded one. The complementary fix is to make runners WRITE their set-points.
  · **Whether the recorded target is the RIGHT one.** A run declaring `homeo_target: 0.05` and hitting 0.05
    passes cleanly even if the biology implies 0.002. Consistency, not correctness.
  · **Non-rate operating points** — weight scale, drive in pA, plateau duration, learning-rate regime. The
    families here are firing rate / activity / sparsity only. Widening the allow-list is how this gate grows.
  · **Cross-unit misses.** A genuine 1000x error recorded as 0.002 vs 4.3 reads as a unit mismatch and is
    refused. Deliberate, and it is a real hole: a run that is catastrophically off target looks exactly like a
    run whose two numbers are in different units, and this gate resolves that ambiguity toward silence.
  · **Sub-tolerance drift.** 1.4x off target is silent. TOL is 0.5 because homeostatic set-points in biology
    are not tight and because sweep arms legitimately vary; the measured case is 2.15x.
  · Artifacts outside `research/findings/raw/`, non-JSON artifacts, and anything never staged.
"""
from __future__ import annotations

import json
import math
import os
import re

NAME = "operating-point"
CLASS_ID = "OP"
BLOCKING = True

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RAW_MARK = "research/findings/raw/"
_SIDECARS = (".cmd.json", ".provenance.json", ".prov.json")

# Achieved must land within [1-TOL, 1+TOL] x target. See the docstring for why 0.5 and not something tighter.
TOL = 0.5

# --- (a) SET-POINT keys. Narrow on purpose: every spelling here either carries a homeostasis word or spells the
# quantity out. Bare `target_rate` is excluded -- 96 corpus files use it to mean "the target item's rate".
#
# An ENUMERATED VOCABULARY, not a regex, because the audit's byte pre-filter and the matcher must never be able
# to disagree about what a set-point key is. That exact split -- two halves of one system, written together,
# diverging on a spelling -- is what made the class-P gate report every automatically-provenanced artifact as
# unprovenanced. Here both halves are BUILT from `_SETPOINT_KEYS`, so divergence is not expressible.
_HOMEO_WORDS = ("homeo", "homeostatic", "homeostasis")
_SP_WORDS = ("target", "setpoint", "set_point")
_QTY_WORDS = ("rate", "firing_rate", "activity", "sparsity")
_QTY_EXPLICIT = ("firing_rate", "activity_level", "sparsity")     # unambiguous without a homeostasis word


def _build_setpoint_keys():
    keys = set()
    for h in _HOMEO_WORDS:
        for sp in _SP_WORDS:
            for joiner in ("_", ""):                               # homeo_target / homeotarget
                keys.add(h + joiner + sp)
                keys.update(h + joiner + sp + "_" + q for q in _QTY_WORDS)
            keys.update(h + "_" + q + "_" + sp for q in _QTY_WORDS)
    for q in _QTY_EXPLICIT:
        keys.update(pre + "_" + q for pre in ("target", "desired"))
        keys.update(q + "_" + sp for sp in _SP_WORDS)
    return frozenset(keys)


_SETPOINT_KEYS = _build_setpoint_keys()

# --- (b) ACHIEVED keys. An explicit STEM allow-list; a generic `*_rate` suffix is what produced the
# `scored_rate` / `top_rate` false alarms.
_ACH_STEM = (r"(firing_rate|fire_rate|spike_rate|mean_rate|rate_mean|active_rate|activity_rate"
             r"|activity_level|activity_mean|mean_activity|frac_active|active_frac|sparsity)")
_ACH_RE = re.compile(r"(^|_)" + _ACH_STEM + r"(_(mean|final|post|overall|avg|average|achieved|hz))*$")
_ACH_EXCLUDE = re.compile(r"(_target(_|$)|^target_|_setpoint(_|$)|^setpoint|_set_point(_|$)"
                          r"|learning|_lr(_|$)|^lr_|gain)")

# A set-point sitting in a config block governs the whole run, so it may be compared against values anywhere.
_CFGISH_RE = re.compile(r"^/(args|argv|config|cfg|params|parameters|settings|hyper\w*|opts|options)$")

# Mechanism words: an enable-flag sharing one of these with the set-point key can switch the set-point off.
_MECH_WORDS = _HOMEO_WORDS

# Traversal caps. MEASURED: raising them to depth=40 / list=100000 / leaves=5e6 changes the full-corpus verdict
# by exactly zero artifacts, so on this tree they hide nothing. They exist so one pathological file cannot
# stall a commit -- an unbounded check that stalls gets bypassed with --no-verify, disabling every other gate.
_MAX_DEPTH = 12
_MAX_LIST = 300
_MAX_LEAVES = 40000

# The audit's pre-filter, built from the SAME vocabulary the matcher uses (see `_SETPOINT_KEYS`). A JSON key is
# always written `"key"`, so this is exact, not heuristic: a file the filter skips cannot contain a set-point.
_NEEDLES = tuple(sorted(('"%s"' % k).encode("ascii") for k in _SETPOINT_KEYS))
_NEEDLE_RE = re.compile(b"|".join(re.escape(n) for n in _NEEDLES))   # one pass, not 141 substring scans


def _prefilter(blob: bytes) -> bool:
    return _NEEDLE_RE.search(blob) is not None


def _is_setpoint(key: str) -> bool:
    return key.lower() in _SETPOINT_KEYS


def _is_artifact(path: str) -> bool:
    p = path.replace("\\", "/")
    return p.endswith(".json") and _RAW_MARK in p and not any(p.endswith(s) for s in _SIDECARS)


def _leaves(obj, prefix="", depth=0, out=None):
    """{json-path: (key, value)} for every finite numeric leaf. Bools are excluded (they are flags, not values)."""
    if out is None:
        out = {}
    if depth > _MAX_DEPTH or len(out) > _MAX_LEAVES:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            kp = "%s/%s" % (prefix, k)
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                if math.isfinite(v):
                    out[kp] = (str(k), float(v))
            elif isinstance(v, (dict, list)):
                _leaves(v, kp, depth + 1, out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:_MAX_LIST]):
            if isinstance(v, (dict, list)):
                _leaves(v, "%s[%d]" % (prefix, i), depth + 1, out)
    return out


def _container(obj, path):
    """The dict that directly holds `path`'s key, for the enable-flag lookup. None if it cannot be resolved."""
    cur, i, n = obj, 0, len(path)
    node = None
    while i < n:
        if path[i] == "/":
            j = i + 1
            while j < n and path[j] not in "/[":
                j += 1
            key = path[i + 1:j]
            if not isinstance(cur, dict) or key not in cur:
                return None
            node, cur, i = cur, cur[key], j
        elif path[i] == "[":
            j = path.find("]", i)
            if j < 0:
                return None
            try:
                idx = int(path[i + 1:j])
            except ValueError:
                return None
            if not isinstance(cur, list) or idx >= len(cur):
                return None
            cur, i = cur[idx], j + 1
        else:
            return None
    return node


def _parent_path(path: str) -> str:
    cut = max(path.rfind("/"), path.rfind("["))
    return path[:cut] if cut > 0 else ""


def _regime(v: float):
    """Coarse unit class. Refusing cross-regime pairs is what keeps unit confusion out (see docstring)."""
    if v <= 0:
        return None
    return "frac" if v <= 1.0 else "hz"


def _mechanism_off(root, tkey: str, tpath: str) -> bool:
    """True when every enable-flag naming this mechanism (sibling, or top level) is False.

    `_volley_*n800_*.json` record `homeostasis_target_rate=0.035` beside `place_homeostasis: false`. A set-point
    for a switched-off mechanism is not an operating point and flagging it is noise.
    """
    words = [w for w in _MECH_WORDS if w in tkey.lower()]
    if not words:
        return False
    flags = []
    for scope in (_container(root, tpath), root):
        if not isinstance(scope, dict):
            continue
        for k, v in scope.items():
            if isinstance(v, bool) and any(w in str(k).lower() for w in words):
                flags.append(v)
    return bool(flags) and not any(flags)


def _analyse(data):
    """[(deviation, target_path, target_value, achieved_path, achieved_key, achieved_value, n_miss, n_cmp)]."""
    if isinstance(data, dict) and isinstance(data.get("operating_point_ack"), str) \
            and data["operating_point_ack"].strip():
        return []
    leaves = _leaves(data)
    targets = [(kp, k, v) for kp, (k, v) in leaves.items() if _is_setpoint(k)]
    if not targets:
        return []                                            # never INFER a target -- the scope limit, stated
    achieved = [(kp, k, v) for kp, (k, v) in leaves.items()
                if _ACH_RE.search(k.lower()) and not _ACH_EXCLUDE.search(k.lower())
                and not _is_setpoint(k)]
    if not achieved:
        return []
    findings = []
    for tpath, tkey, tval in sorted(targets):
        treg = _regime(tval)
        if treg is None or _mechanism_off(data, tkey, tpath):
            continue
        par = _parent_path(tpath)
        if par:
            cands = [a for a in achieved if a[0].startswith(par + "/") or a[0].startswith(par + "[")]
            if not cands and _CFGISH_RE.match(par):
                cands = achieved                             # a config block governs the whole run
        else:
            cands = achieved                                 # root-level set-point: the lane D shape
        cands = [a for a in cands if _regime(a[2]) == treg]   # same units, or no comparison at all
        if not cands:
            continue
        miss = [(abs(v - tval) / tval, p, k, v) for p, k, v in cands if abs(v - tval) / tval > TOL]
        if miss:
            miss.sort(key=lambda t: (-t[0], t[1]))
            dev, apath, akey, aval = miss[0]
            findings.append((dev, tpath, tval, apath, akey, aval, len(miss), len(cands)))
    findings.sort(key=lambda t: (-t[0], t[1]))
    return findings


def _problems_for(path: str) -> list:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []                                            # unparseable is class P's business, not ours
    out = []
    for dev, tpath, tval, apath, akey, aval, n_miss, n_cmp in _analyse(data):
        out.append(
            "CLASS OP operating point contradicts the artifact's own target: %s — %s=%g but %s=%g (%.2fx, "
            "%.0f%% off; %d of %d compared values miss by >%.0f%%). Fix: hit the recorded set-point, or record "
            "why it was not hit (top-level \"operating_point_ack\": \"<why>\"). Both numbers were already in "
            "this file."
            % (path, tpath.lstrip("/"), tval, apath.lstrip("/"), aval,
               (aval / tval) if tval else float("nan"), 100.0 * dev, n_miss, n_cmp, 100.0 * TOL))
    return out


def check(paths) -> list:
    # Contract: None => full audit; [] => nothing of our kind staged; non-empty => check exactly those.
    if paths is None:
        return corpus_scan()["problems"]
    if not paths:
        return []
    problems = []
    for path in paths:
        if _is_artifact(path) and os.path.exists(path):
            problems += _problems_for(path)
    return problems


def corpus_scan() -> dict:
    """The full audit. Cheap because `_prefilter` rejects on bytes before anything is parsed."""
    hits, scanned, parsed, artifacts = [], 0, 0, 0
    for root, _d, names in os.walk(os.path.join(_REPO_ROOT, "research", "findings", "raw")):
        for n in sorted(names):
            p = os.path.join(root, n)
            if not _is_artifact(p):
                continue
            scanned += 1
            try:
                with open(p, "rb") as fh:
                    blob = fh.read()
            except OSError:
                continue
            if not _prefilter(blob):
                continue
            parsed += 1
            rel = os.path.relpath(p, _REPO_ROOT).replace("\\", "/")
            got = [s.replace(p, rel, 1) for s in _problems_for(p)]
            artifacts += 1 if got else 0
            hits += got
    # `flagged` counts ARTIFACTS; `problems` can be longer (one line per set-point instance in a file).
    return {"scanned": scanned, "parsed": parsed, "flagged": artifacts, "problems": hits}


def selftest() -> list:
    import tempfile
    bad = []
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, "research", "findings", "raw")
        os.makedirs(raw)

        def w(name, obj):
            p = os.path.join(raw, name)
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(obj if isinstance(obj, str) else json.dumps(obj))
            return p

        # ---------------------------------------------------------------- THE FAILING DIRECTION FIRST.
        # Every case here is one the gate MUST catch; if any produces no problem the gate is not a gate.
        must_catch = {
            # the real lane D shape, reduced: homeo_target at the root, v1_firing_rate inside the arms
            "lane D (2.15x hot)": w("laneD.json", {
                "description": "lane D normalization arms", "homeo_target": 0.002,
                "arms": [{"arm": "base", "v1_firing_rate": 0.0043},
                         {"arm": "ms0.75", "v1_firing_rate": 0.002}]}),
            # the same failure with the sign flipped -- a run far COLDER than its target is equally unnoticed
            "undershoot (0.3x cold)": w("cold.json", {"homeo_target": 0.002, "v1_firing_rate_mean": 0.0006}),
            # a set-point nested in a per-run record, compared only inside its own subtree
            "nested per-run set-point": w("perrun.json", {
                "per_run": [{"homeostasis_target_rate": 0.035, "source_active_rate_mean": 0.0083}]}),
            # Hz on both sides: the regime guard must not become a blanket excuse to skip
            "hz regime": w("hz.json", {"target_firing_rate": 5.0, "v1_firing_rate": 22.0}),
            # an EMPTY ack is not an ack (the class-P empty-exemption bug, re-run here on purpose)
            "empty ack": w("emptyack.json", {"operating_point_ack": "  ", "homeo_target": 0.002,
                                             "v1_firing_rate": 0.0043}),
            # enable-flags present but one is True => the mechanism is on, the set-point is live
            "one enable flag true": w("mixedflag.json", {"per_run": [
                {"homeostasis_target_rate": 0.035, "place_homeostasis": False, "homeostasis_on": True,
                 "source_active_rate_mean": 0.0083}]}),
        }
        for label, p in must_catch.items():
            if not check([p]):
                bad.append("GATE CANNOT FAIL: %s (%s) produced no problem" % (label, os.path.basename(p)))

        # a caught case must NAME both numbers, or the message cannot be acted on
        msg = (check([must_catch["lane D (2.15x hot)"]]) or [""])[0]
        for token in ("homeo_target", "0.002", "0.0043", "2.15"):
            if token not in msg:
                bad.append("message omits %r: %s" % (token, msg[:140]))

        # EVERY set-point spelling must be caught by the matcher AND survive the audit's byte pre-filter. The
        # class-P gate shipped with these two halves disagreeing about one suffix and silently checked nothing.
        for i, key in enumerate(sorted(_SETPOINT_KEYS)):
            body = {key: 0.002, "v1_firing_rate": 0.0043}
            p = w("vocab_%d.json" % i, body)
            if not check([p]):
                bad.append("VOCABULARY GAP: set-point key %r is not caught by check()" % key)
            if not _prefilter(json.dumps(body).encode("ascii")):
                bad.append("PRE-FILTER GAP: the audit would skip a file containing %r without parsing it" % key)
        if _prefilter(b'{"target_rate": 0.07, "top_rate": 0.955}'):
            bad.append("pre-filter accepts bare target_rate; it must not be a set-point spelling")

        # ---------------------------------------------------------------- ONLY THEN: it must not cry wolf.
        # Each of these is a false positive MEASURED on the real corpus before the rule that kills it.
        outside = os.path.join(td, "elsewhere.json")
        with open(outside, "w", encoding="utf-8") as fh:
            json.dump({"homeo_target": 0.002, "v1_firing_rate": 0.0043}, fh)
        must_pass = {
            "within tolerance": w("ok.json", {"homeo_target": 0.002, "v1_firing_rate": 0.0022}),
            # THE scope rule: an achieved value with no recorded target is never flagged, ever
            "no target recorded": w("notgt.json", {"v1_firing_rate": 0.0043, "orient_decode": 0.42}),
            "no achieved recorded": w("noach.json", {"homeo_target": 0.002, "orient_decode": 0.42}),
            # catastrophic_forgetting_probe_*: `target_rate` is the TARGET WORD's rate, not a set-point
            "target_rate vs top_rate": w("cfp.json", {"post_per_word": [{"target_rate": 0.07, "top_rate": 0.955}]}),
            # _cortex_*/_multibridge_*: homeo_target=40Hz vs scored_rate=2168, two unrelated quantities
            "generic *_rate not achieved": w("scored.json", {"args": {"homeo_target": 40.0},
                                                             "detail": [{"scored_rate": 2168.0}]}),
            # cross-unit: a fraction must never be compared against a Hz figure
            "cross regime": w("units.json", {"homeo_target": 40.0, "v1_firing_rate": 0.0043}),
            # _volley_*n800_*: the set-point is recorded but the mechanism is off
            "mechanism disabled": w("volley.json", {"per_run": [
                {"homeostasis_target_rate": 0.035, "place_homeostasis": False,
                 "source_active_rate_mean": 0.0083}]}),
            "explicit ack": w("ack.json", {"operating_point_ack": "ran 2.15x hot, homeostasis too slow",
                                           "homeo_target": 0.002, "v1_firing_rate": 0.0043}),
            # a nested set-point must not reach across sibling records
            "other subtree": w("subtree.json", {
                "per_run": [{"homeostasis_target_rate": 0.035, "source_active_rate_mean": 0.034},
                            {"v1_firing_rate": 0.9}]}),
            "zero/negative target": w("zero.json", {"homeo_target": 0.0, "v1_firing_rate": 0.0043}),
            "not an artifact (.md)": w("note.md", "homeo_target 0.002 v1_firing_rate 0.0043"),
            "json outside raw/": outside,
            "unparseable": w("broken.json", '{"homeo_target": 0.002, '),
            "list root": w("listroot.json", [{"homeo_target": 0.002}]),
        }
        for label, p in must_pass.items():
            probs = check([p])
            if probs:
                bad.append("FALSE POSITIVE: %s flagged — %s" % (label, probs[0][:110]))

        mixed = check([must_catch["lane D (2.15x hot)"], must_pass["within tolerance"],
                       must_pass["not an artifact (.md)"], must_pass["no target recorded"]])
        if len(mixed) != 1:
            bad.append("batch check returned %d problems, expected exactly 1" % len(mixed))
        if check([]):
            bad.append("check([]) returned problems; nothing staged of our kind must be silent")
        if not isinstance(check(None), list):
            bad.append("check(None) must return a list (the full audit)")
    return bad


if __name__ == "__main__":
    res = corpus_scan()
    print("class OP operating point — scanned %d artifacts, parsed %d carrying a recorded set-point, "
          "flagged %d artifact(s) / %d problem(s)"
          % (res["scanned"], res["parsed"], res["flagged"], len(res["problems"])))
    for p in res["problems"]:
        print("  " + p)
