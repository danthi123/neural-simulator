"""Scorer for the BRAIN_EMOTION_CATEGORY_BOUNDED default-ON flip: one verdict over bars B1-B8 and the decision rule.

Prereg: research/findings/2026-10-27-emo-bounded-flip-PREREGISTRATION.md (bars in section 2 are the ruling's own words,
`emo-bg-merge-1026` item (2); this module adds NO bar and changes none: it is the reader section 6 owes). Reads data only;
it runs no brain, so it may read gate-seed shards (the gate-seed guard is for the runs that PRODUCE them).

    python -m research.runners.emo_flip_score \\
        --on-dir  research/findings/raw/_emo_bounded_flip/gate_on \\
        --off-dir research/findings/raw/_emo_bounded_flip/gate_off \\
        --out     research/findings/raw/_emo_bounded_flip/verdict.json      [--pin <full sha>] [--only B4,B6] [--strict]

READING OF EACH BAR (the clause text is in the prereg, not here):
  B1 B2 B3  `stage2_exit_gate.aggregate` over the ON directory, emo group only: n_correct, n_lesion_effect and n_reproduced
            for `lesion_affect`, n_permuted_tracks. PASS needs 6 of 6 DEFINED seeds; a defined seed that fails is FAIL; an
            undefined seed (a shard missing, an errored turn, a failed null) is UNDEFINED, never a pass.
  B4        every `feel_*` response of every ON shard (all five arms) carries emotion_category.bounded_flag True,
            bounded_env "1", bounded_readback True and oor_count_session 0. A shard, a turn or a key that is not there is
            UNDEFINED (it is not evidence the guard held); a value that breaks the bar is FAIL and names shard and turn.
  B5        the same aggregate over the OFF directory: the per-seed `intact` named categories equal the 10-26 gate's
            (research/findings/raw/_stage2_exit/gate/aggregate.json) on every seed, and seed 101 joy reads None. A mismatch
            makes the OFF half UNDEFINED and the ON half is not read until it is explained (prereg 2, B5).
  B6        ON vs OFF, same seed and arm, WHOLE RECORD: every non-feel turn identical in every field at every depth after
            the five engagement keys are dropped (at any depth) from both sides; every feel turn's text before its category
            clause identical. The first differing path is printed. (`emotion_category_dev --flip-check` is NOT the B6
            feel reader for this flip: its FEEL-NOT-APPEND-ONLY assumes the OFF arm has no clause, which holds for the
            organ flag, not for this one: here BOTH arms append a clause, so it fires on a perfect brain. Pinned by
            tests/test_emo_flip_score.py.)
  B7        `emotion_category_dev.flip_check` FLIP-HURTS-ANSWERS over every turn of the Stage-2 shards, and
            `tools/answer_rate.py` paired over the emotion-category lb_shard row, six gate seeds. Either flag is FAIL.
  B8        the `emotion-category` lb_shard row at each seed, flag ON: reply-level load-bearing (docs/TERMS.md: the lesion
            changes `answer`, the null is clean, the lesion reproduces) in every DEFINED cell, 6 of 6 seeds (Claude confirmed
            this reading 2026-10-27). A cell whose intact arms do not show the flag engaged (bounded_flag and
            bounded_readback True) is UNDEFINED. The flag-OFF row is reported beside it.
  DEC       PASS on all eight -> flip the default ON before the 11-21 re-pin; anything else -> STOP with the data.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import os
import sys

os.environ.setdefault("SIM_BACKEND", "numpy")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners import stage2_exit_gate as G  # noqa: E402
from tools.lab import attributable_to  # noqa: E402  (whose is the lesion's difference: gates/attribution_required)

PREREG = "research/findings/2026-10-27-emo-bounded-flip-PREREGISTRATION.md"
GATE_SEEDS = tuple(G.GATE_SEEDS)
ARMS = ("intact_a", "intact_b", "permuted", "lesion_affect", "lesion_affect.rep0")     # prereg 3: five arms
ENGAGEMENT_KEYS = ("bounded_flag", "bounded_env", "bounded_readback", "oor_count", "oor_count_session")
STAGE2_ON_DIR = "research/findings/raw/_emo_bounded_flip/gate_on"
STAGE2_OFF_DIR = "research/findings/raw/_emo_bounded_flip/gate_off"
LB_BASE = "research/findings/raw/_load_bearing/_shards"
LB_ON_TAG = "emo-bounded-flip-gate-on"
LB_OFF_TAG = "emo-bounded-flip-gate-off"
LB_FACULTY = "emotion-category"
REFERENCE = "research/findings/raw/_stage2_exit/gate/aggregate.json"
REFERENCE_JOY_NONE_SEED = 101          # the 10-26 gate's emo result: seed 101 intact joy named None
GROUP = "emo"
BARS = ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8")
PASS, FAIL, UNDEFINED = "PASS", "FAIL", "UNDEFINED"
HURTS = "FLIP-HURTS-ANSWERS"


# ══════════════════════════════════════════════ small pure helpers ═══════════════════════════════════════════════════
def _status(failures, undefined):
    """FAIL beats UNDEFINED beats PASS: a defined cell that breaks the bar decides the bar whatever else is missing."""
    return FAIL if failures else (UNDEFINED if undefined else PASS)


def _bar(bar, failures, undefined, detail, **extra):
    d = {"bar": bar, "status": _status(failures, undefined), "detail": detail, "failures": list(failures),
         "undefined": list(undefined)}
    d.update(extra)
    return d


def _short(x, n=90):
    s = repr(x)
    return s if len(s) <= n else s[:n - 3] + "..."


def drop_keys(obj, keys=ENGAGEMENT_KEYS):
    """A copy of `obj` with every dict key in `keys` removed at ANY depth (lists are walked)."""
    if isinstance(obj, dict):
        return {k: drop_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [drop_keys(v, keys) for v in obj]
    return obj


def first_diff(a, b, path=""):
    """The first differing path between two JSON-like values, as 'path: a-value vs b-value', or None when identical.
    Dict keys are walked in sorted order; a key present on one side only is a difference. Exact equality (the build is
    deterministic: intact_a == intact_b whole-record on the 10-26 gate shards); NaN equals NaN."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            p = "%s.%s" % (path, k) if path else str(k)
            if k not in a:
                return "%s: absent vs %s" % (p, _short(b[k]))
            if k not in b:
                return "%s: %s vs absent" % (p, _short(a[k]))
            d = first_diff(a[k], b[k], p)
            if d:
                return d
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return "%s: list length %d vs %d" % (path or "<root>", len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, "%s[%d]" % (path, i))
            if d:
                return d
        return None
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return None
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))
                                       and not isinstance(a, bool) and not isinstance(b, bool)):
        return "%s: %s vs %s" % (path or "<root>", _short(a), _short(b))
    return None if a == b else "%s: %s vs %s" % (path or "<root>", _short(a), _short(b))


_CLAUSES = None


def clauses():
    """The five sentences the feel read-out can append (emotion_category_organ.feel_clause), longest first."""
    global _CLAUSES
    if _CLAUSES is None:
        from research.runners import emotion_category_organ as E
        _CLAUSES = sorted({E.feel_clause(c) for c in list(G.EMOTION_LEXICON) + [None]}, key=len, reverse=True)
    return _CLAUSES


def split_clause(answer):
    """(text before the category clause, the clause) when the answer ENDS with one of the organ's clauses; else
    (the whole answer, None)."""
    a = str(answer if answer is not None else "")
    for c in clauses():
        if a.endswith(c):
            return a[:-len(c)], c
    return a, None


def _errored(resp):
    return (not isinstance(resp, dict)) or ("_error" in resp)


def _feel_labels(seed, arm):
    return [l for l, _m, role in G.turns_for(GROUP, seed, arm) if role == "feel"]


# ══════════════════════════════════════════════ shard access (disk or memory) ═══════════════════════════════════════
def _load(path):
    if not os.path.exists(path):
        return None, "shard missing"
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:  # noqa: BLE001
        return None, "shard unreadable (%s)" % type(e).__name__
    if not isinstance(d, dict) or not isinstance(d.get("responses"), dict):
        return None, "shard has no responses object"
    return d, None


def dir_loader(root):
    """get(seed, arm) -> (shard or None, why-not) over a Stage-2 shard directory."""
    return lambda seed, arm: _load(G.shard_path(root, seed, GROUP, arm))


def mem_loader(data):
    """get(seed, arm) over {seed: {arm: shard}} held in memory (the bar-sanity specs and the tests)."""
    def get(seed, arm):
        sh = (data.get(seed) or {}).get(arm)
        if sh is None:
            return None, "shard missing"
        if not isinstance(sh, dict) or not isinstance(sh.get("responses"), dict):
            return None, "shard has no responses object"
        return sh, None
    return get


# ══════════════════════════════════════════════ B4: the guard held on every ON feel read ═════════════════════════════
def score_b4(get, seeds=GATE_SEEDS, arms=ARMS):
    failures, undefined, n_shards, n_reads = [], [], 0, 0
    for seed in seeds:
        for arm in arms:
            tag = "s%d/emo__%s" % (int(seed), arm)
            shard, why = get(int(seed), arm)
            if shard is None:
                undefined.append("%s: %s" % (tag, why))
                continue
            n_shards += 1
            resp = shard["responses"]
            want = _feel_labels(int(seed), arm)
            extra = sorted(l for l in resp if l.startswith("feel_") and l not in want)
            if not want:
                undefined.append("%s: the registered turn table has no feel turn" % tag)
            for l in want:
                if l not in resp:
                    undefined.append("%s %s: feel turn missing from the shard" % (tag, l))
            for l in [x for x in want if x in resp] + extra:
                r = resp[l]
                n_reads += 1
                where = "%s %s" % (tag, l)
                if _errored(r):
                    undefined.append("%s: errored turn, no emotion read" % where)
                    continue
                ec = r.get("emotion_category")
                if not isinstance(ec, dict):
                    undefined.append("%s: no emotion_category block" % where)
                    continue
                need = ("bounded_flag", "bounded_env", "bounded_readback", "oor_count_session")
                absent = [k for k in need if k not in ec]
                if absent:
                    undefined.append("%s: lacks key(s) %s" % (where, ",".join(absent)))
                    continue
                if ec["bounded_flag"] is not True:
                    failures.append("%s: bounded_flag=%s (want True)" % (where, _short(ec["bounded_flag"])))
                if ec["bounded_env"] != "1":
                    failures.append("%s: bounded_env=%s (want '1')" % (where, _short(ec["bounded_env"])))
                if ec["bounded_readback"] is not True:
                    failures.append("%s: bounded_readback=%s (want True)" % (where, _short(ec["bounded_readback"])))
                o = ec["oor_count_session"]
                if isinstance(o, bool) or not isinstance(o, int):
                    undefined.append("%s: oor_count_session=%s is not an integer count" % (where, _short(o)))
                elif o != 0:
                    failures.append("%s: oor_count_session=%d (want 0)" % (where, o))
    return _bar("B4", failures, undefined,
                "%d ON shard(s), %d feel read(s) checked; %d violation(s), %d undefined" % (
                    n_shards, n_reads, len(failures), len(undefined)),
                n_shards=n_shards, n_feel_reads=n_reads)


# ══════════════════════════════════════════════ B6: ON vs OFF, whole record apart from the five keys ═════════════════
def score_b6(get_on, get_off, seeds=GATE_SEEDS, arms=ARMS):
    failures, undefined, n_pairs, n_nonfeel, n_feel = [], [], 0, 0, 0
    for seed in seeds:
        for arm in arms:
            tag = "s%d/emo__%s" % (int(seed), arm)
            on, why_on = get_on(int(seed), arm)
            off, why_off = get_off(int(seed), arm)
            if on is None or off is None:
                undefined.append("%s: pair missing (ON %s; OFF %s)" % (tag, why_on or "ok", why_off or "ok"))
                continue
            n_pairs += 1
            ro, rf = on["responses"], off["responses"]
            if set(ro) != set(rf):
                undefined.append("%s: turn labels differ (ON only %s; OFF only %s)" % (
                    tag, sorted(set(ro) - set(rf)), sorted(set(rf) - set(ro))))
                continue
            for l in rf:
                a, b = ro[l], rf[l]
                where = "%s %s" % (tag, l)
                if _errored(a) or _errored(b):
                    undefined.append("%s: errored turn (ON %s, OFF %s)" % (where, _errored(a), _errored(b)))
                    continue
                if l.startswith("feel_"):
                    n_feel += 1
                    pa, ca = split_clause(a.get("answer"))
                    pb, cb = split_clause(b.get("answer"))
                    if (ca is None) != (cb is None):
                        failures.append("%s: category clause present in the %s arm only" % (
                            where, "ON" if ca else "OFF"))
                    elif pa != pb:
                        i = next((k for k in range(min(len(pa), len(pb))) if pa[k] != pb[k]), min(len(pa), len(pb)))
                        failures.append("%s: feel text before the clause differs at char %d (ON %s vs OFF %s)" % (
                            where, i, _short(pa[i:i + 40], 50), _short(pb[i:i + 40], 50)))
                else:
                    n_nonfeel += 1
                    d = first_diff(drop_keys(a), drop_keys(b))
                    if d:
                        failures.append("%s: first differing path %s (ON vs OFF)" % (where, d))
    return _bar("B6", failures, undefined,
                "%d ON/OFF pair(s): %d non-feel turn(s) compared whole-record (five engagement keys dropped at any depth), "
                "%d feel turn(s) compared before the clause; %d difference(s), %d undefined" % (
                    n_pairs, n_nonfeel, n_feel, len(failures), len(undefined)),
                n_pairs=n_pairs, n_nonfeel_turns=n_nonfeel, n_feel_turns=n_feel)


# ══════════════════════════════════════════════ B1-B3 and B5: the Stage-2 aggregate ══════════════════════════════════
def stage2_aggregate(root, seeds=GATE_SEEDS):
    """The real aggregate over a directory holding only the emo group (the other groups' missing shards, and the stage
    verdict, are expected and not read). stdout of the scorer goes to stderr."""
    with contextlib.redirect_stdout(sys.stderr):
        return G.aggregate(root, seeds=tuple(int(s) for s in seeds), informational=False)


def _grp(agg):
    return (agg.get("groups") or {}).get(GROUP) or {}


def _undefined_seeds(g, seeds):
    out = []
    for s in seeds:
        row = (g.get("seeds") or {}).get(s) or (g.get("seeds") or {}).get(str(s)) or {}
        if row.get("undefined_reasons") or "correct" not in row:
            out.append("s%d: %s" % (int(s), "; ".join(row.get("undefined_reasons") or ["no row"])[:160]))
    return out


def score_b123(agg, seeds=GATE_SEEDS):
    g = _grp(agg)
    n = len(seeds)
    if not g:
        return [_bar(b, [], ["no emo group in the aggregate"], "no emo group") for b in ("B1", "B2", "B3")]
    nd = g.get("n_defined", 0)
    und = _undefined_seeds(g, seeds)
    rows = {int(s): r for s, r in (g.get("seeds") or {}).items()}
    les = "lesion_affect"
    n_c = g.get("n_correct", 0)
    n_e = (g.get("n_lesion_effect") or {}).get(les, 0)
    n_r = (g.get("n_reproduced") or {}).get(les, 0)
    n_t = g.get("n_permuted_tracks", 0)
    n_t_def = sum(1 for s in rows if rows[s].get("tracks") is not None and not rows[s].get("undefined_reasons"))

    b1_fail = ["%s: not all four categories named correctly (named %s)" % (
        "s%d" % s, _short(rows[s].get("intact", {}).get("named"), 120)) for s in sorted(rows)
        if not rows[s].get("undefined_reasons") and rows[s].get("correct") is False]
    b2_fail = []
    for s in sorted(rows):
        r = rows[s]
        if r.get("undefined_reasons") or "correct" not in r:
            continue
        L = (r.get("lesions") or {}).get(les) or {}
        if L.get("effect") is False:
            b2_fail.append("s%d: lesion_affect does not change the named category on every feel turn (changed %s)" % (
                s, _short(L.get("category_changed"), 120)))
        if L.get("reproduced") is False:
            b2_fail.append("s%d: lesion_affect does not reproduce against its repeat" % s)
    b3_fail = ["s%d: permuted exposure not tracked" % s for s in sorted(rows)
               if not rows[s].get("undefined_reasons") and rows[s].get("tracks") is False]
    b3_und = list(und) + ["s%d: permuted tracking undefined (the intact arm does not name the categories)" % s
                          for s in sorted(rows) if not rows[s].get("undefined_reasons") and "correct" in rows[s]
                          and rows[s].get("tracks") is None]
    b2_und = list(und) + ["s%d: lesion_affect effect/reproduction undefined" % s for s in sorted(rows)
                          if not rows[s].get("undefined_reasons") and "correct" in rows[s]
                          and ((rows[s].get("lesions") or {}).get(les) or {}).get("effect") is None]
    # ATTRIBUTION (tools.lab): per defined seed, treatment = feel turns whose NAMED category the lesion_affect arm changed;
    # control = feel turns whose reply differs between the two intact builds (zero when the null is clean, which the
    # aggregate already requires). Printed to stderr, kept in the B2 record; it decides nothing beyond the aggregate.
    attribution = {}
    with contextlib.redirect_stdout(sys.stderr):
        for s in sorted(rows):
            r = rows[s]
            L = (r.get("lesions") or {}).get(les) or {}
            if r.get("undefined_reasons") or not isinstance(L.get("category_changed"), dict):
                continue
            treat = float(sum(1 for v in L["category_changed"].values() if v))
            ctrl = float(sum(1 for x in (r.get("null_diffs") or []) if str(x[0]).startswith("feel_")))
            attribution["s%d" % s] = attributable_to(
                "emo lesion_affect named-category change, seed %d (control = intact_a vs intact_b feel replies)" % s, treat, ctrl)
    return [
        _bar("B1", b1_fail, und, "n_correct %d of %d defined seeds (need %d of %d)" % (n_c, nd, n, n),
             n_correct=n_c, n_defined=nd),
        _bar("B2", b2_fail, b2_und, "lesion_affect n_lesion_effect %d, n_reproduced %d of %d defined seeds (need %d, %d of %d)"
             % (n_e, n_r, nd, n, n, n), n_lesion_effect=n_e, n_reproduced=n_r, n_defined=nd,
             attributable_fraction=attribution),
        _bar("B3", b3_fail, b3_und, "n_permuted_tracks %d of %d tracking-defined seeds (need %d of %d)" % (n_t, n_t_def, n, n),
             n_permuted_tracks=n_t, n_tracking_defined=n_t_def),
    ]


def score_b5(off_agg, reference, seeds=GATE_SEEDS):
    """The OFF arms reproduce the 10-26 gate's emo result: per-seed intact named categories equal on every seed, and (as the
    bar's own words say) seed 101 joy named None. A mismatch is UNDEFINED (the OFF half), and `mismatch` is True so the
    caller does not read the ON half until it is explained."""
    og, rg = _grp(off_agg), _grp(reference or {})
    undefined, mismatches, compared = [], [], 0
    if not rg:
        return _bar("B5", [], ["reference aggregate has no emo group (%s missing or unreadable)" % REFERENCE],
                    "no reference", mismatch=False)
    for s in seeds:
        ref = (rg.get("seeds") or {}).get(str(s)) or (rg.get("seeds") or {}).get(s)
        row = (og.get("seeds") or {}).get(s) or (og.get("seeds") or {}).get(str(s)) or {}
        if ref is None:
            undefined.append("s%d: the 10-26 reference has no such seed (B5 is read at the gate seeds)" % int(s))
            continue
        if row.get("undefined_reasons") or "intact" not in row:
            undefined.append("s%d: OFF row undefined (%s)" % (int(s), "; ".join(row.get("undefined_reasons") or ["no row"])[:140]))
            continue
        compared += 1
        a, b = row["intact"].get("named"), (ref.get("intact") or {}).get("named")
        if a != b:
            mismatches.append("s%d: OFF named %s != 10-26 reference %s" % (int(s), _short(a, 110), _short(b, 110)))
    if REFERENCE_JOY_NONE_SEED in [int(x) for x in seeds]:
        ref = (rg.get("seeds") or {}).get(str(REFERENCE_JOY_NONE_SEED)) or {}
        if ((ref.get("intact") or {}).get("named") or {}).get("joy", "absent") is not None:
            undefined.append("reference s%d joy is not None: the bar's premise (s101 joy None) is not in the reference file"
                             % REFERENCE_JOY_NONE_SEED)
        row = (og.get("seeds") or {}).get(REFERENCE_JOY_NONE_SEED) or (og.get("seeds") or {}).get(str(REFERENCE_JOY_NONE_SEED)) or {}
        if "intact" in row and not row.get("undefined_reasons") and ((row["intact"].get("named") or {}).get("joy", "absent") is not None):
            mismatches.append("s%d: OFF joy named %s (the 10-26 result is None)" % (
                REFERENCE_JOY_NONE_SEED, _short((row["intact"].get("named") or {}).get("joy"))))
    # a mismatch makes the OFF half UNDEFINED (prereg 2, B5): it is not a FAIL of the flip, and the ON half is held
    und = list(undefined) + mismatches
    return _bar("B5", [], und, "OFF intact named categories compared with the 10-26 gate on %d of %d seed(s); %d mismatch(es)"
                % (compared, len(seeds), len(mismatches)), mismatch=bool(mismatches), compared=compared)


# ══════════════════════════════════════════════ B7: answers ═══════════════════════════════════════════════════════════
def score_b7(on_dir, off_dir, seeds=GATE_SEEDS, arms=ARMS, lb_base=LB_BASE, on_tag=LB_ON_TAG, off_tag=LB_OFF_TAG):
    from research.runners.emotion_category_dev import flip_check
    from tools import answer_rate as AR
    failures, undefined = [], []
    fc = flip_check(on_dir, off_dir, [int(s) for s in seeds])
    expect = {"s%d/emo__%s.json" % (int(s), a) for s in seeds for a in arms}
    seen = set(fc["pairs"])
    for k in sorted(expect - seen):
        undefined.append("flip_check %s: pair missing" % k)
    if HURTS in fc["flags"]:
        bad = sorted(k for k, v in fc["pairs"].items() if v["answered_on"] < v["answered_off"])
        failures.append("flip_check %s on %s" % (HURTS, ",".join(bad)))
    rep = AR.report(AR.paired_per_seed(lb_base, on_tag, off_tag, LB_FACULTY, [int(s) for s in seeds]),
                    label="%s: %s vs %s" % (LB_FACULTY, on_tag, off_tag), mode="paired")
    if rep["flags"]:
        failures.append("answer_rate paired %s on seed(s) %s" % (HURTS, ",".join(rep["hurt_seeds"])))
    for s in rep["unmeasured_seeds"]:
        undefined.append("answer_rate paired seed %s: UNMEASURED (a side has no probe reply)" % s)
    return _bar("B7", failures, undefined,
                "flip_check %d pair(s), flags %s (FEEL-NOT-APPEND-ONLY / FLIP-CHANGES-NONFEEL are not read here: see B6); "
                "answer_rate paired status %s over %d seed(s)" % (fc["n_pairs"], fc["flags"], rep["status"], rep["n_seeds"]),
                flip_check_flags=fc["flags"], answer_rate=rep)


# ══════════════════════════════════════════════ B8: the load-bearing row ═════════════════════════════════════════════
def _lb_cell(base, tag, seed, want_flag, pin=None):
    """('LB'|'NOT-LB'|'UNDEFINED', reason) for one lb_shard cell of the emotion-category row."""
    cell_dir = os.path.join(base, tag, "s%d" % int(seed), LB_FACULTY)
    p = os.path.join(cell_dir, "lb.json")
    if not os.path.exists(p):
        return "UNDEFINED", "lb.json missing"
    try:
        with open(p) as f:
            rep = json.load(f)
    except Exception as e:  # noqa: BLE001
        return "UNDEFINED", "lb.json unreadable (%s)" % type(e).__name__
    row = next((r for r in (rep.get("per_faculty") or []) if r.get("faculty") == LB_FACULTY), None)
    if row is None:
        return "UNDEFINED", "no %s row in lb.json" % LB_FACULTY
    if rep.get("UNRELIABLE"):
        return "UNDEFINED", "the battery marked this shard UNRELIABLE"
    if pin:
        from tools.lb_shard import cell_prov_fails
        fails, _cov = cell_prov_fails(cell_dir, pin, rep.get("per_faculty", []))
        if fails:
            return "UNDEFINED", "provenance does not match the pin: %s" % _short(fails, 160)
    # engagement: the intact arms must show the flag in the state it is meant to be in for this tag
    arms = sorted(a for a in glob.glob(os.path.join(cell_dir, "intact*.json")) if not a.endswith(".prov.json"))
    if not arms:
        return "UNDEFINED", "no intact arm file beside lb.json"
    for a in arms:
        try:
            with open(a) as f:
                doc = json.load(f)
        except Exception as e:  # noqa: BLE001
            return "UNDEFINED", "%s unreadable (%s)" % (os.path.basename(a), type(e).__name__)
        feel = next((v for k, v in sorted(doc.items()) if isinstance(v, dict) and "emotion_category" in v), None) \
            if isinstance(doc, dict) else None
        ec = (feel or {}).get("emotion_category")
        if not isinstance(ec, dict) or "bounded_flag" not in ec or "bounded_readback" not in ec:
            return "UNDEFINED", "%s carries no engagement keys on its feel read" % os.path.basename(a)
        if want_flag and not (ec["bounded_flag"] is True and ec["bounded_readback"] is True):
            return "UNDEFINED", "flag not engaged in %s (bounded_flag=%s, bounded_readback=%s)" % (
                os.path.basename(a), _short(ec["bounded_flag"]), _short(ec["bounded_readback"]))
        if not want_flag and ec["bounded_flag"] is not False:
            return "UNDEFINED", "OFF cell %s shows bounded_flag=%s (the flag leaked in)" % (
                os.path.basename(a), _short(ec["bounded_flag"]))
    lb, verdict = row.get("load_bearing"), row.get("verdict")
    clean, repro = row.get("null_control_clean"), row.get("lesion_reproduced")
    if lb is True and verdict == "regressed" and clean is True and repro is True:
        return "LB", "regressed; null clean; lesion reproduced"
    if lb is False and verdict == "pass" and clean is True and repro is True:
        return "NOT-LB", "the lesion did not change the reply (verdict pass)"
    return "UNDEFINED", "load_bearing=%s verdict=%s null_clean=%s reproduced=%s" % (
        _short(lb), _short(verdict), _short(clean), _short(repro))


def score_b8(base=LB_BASE, on_tag=LB_ON_TAG, off_tag=LB_OFF_TAG, seeds=GATE_SEEDS, pin=None):
    failures, undefined, on_cells, off_cells = [], [], {}, {}
    for s in seeds:
        st, why = _lb_cell(base, on_tag, s, True, pin)
        on_cells["s%d" % int(s)] = {"status": st, "why": why}
        if st == "NOT-LB":
            failures.append("s%d (flag ON): %s" % (int(s), why))
        elif st == "UNDEFINED":
            undefined.append("s%d (flag ON): %s" % (int(s), why))
        st0, why0 = _lb_cell(base, off_tag, s, False, pin)
        off_cells["s%d" % int(s)] = {"status": st0, "why": why0}
    n_lb = sum(1 for v in on_cells.values() if v["status"] == "LB")
    n_off = sum(1 for v in off_cells.values() if v["status"] == "LB")
    return _bar("B8", failures, undefined,
                "emotion-category row, flag ON: load-bearing in %d of %d seeds (need %d of %d); flag OFF beside it: %d of %d" % (
                    n_lb, len(seeds), len(seeds), len(seeds), n_off, len(seeds)),
                on_cells=on_cells, off_cells=off_cells, n_load_bearing_on=n_lb, n_load_bearing_off=n_off)


# ══════════════════════════════════════════════ the decision ═════════════════════════════════════════════════════════
def decide(bars):
    """DEC (prereg 2): PASS on B1-B8 -> flip the default ON before the 11-21 re-pin; anything else -> STOP with the data.
    A B5 mismatch holds the ON half unread. Returns {'status', 'action', 'fail', 'undefined', 'held'}."""
    by = {b["bar"]: b for b in bars}
    missing = [x for x in BARS if x not in by]
    held = bool((by.get("B5") or {}).get("mismatch"))
    fail = [x for x in BARS if x in by and by[x]["status"] == FAIL]
    und = [x for x in BARS if x in by and by[x]["status"] == UNDEFINED] + ["%s(not read)" % x for x in missing]
    if not fail and not und:
        return {"status": PASS, "action": "flip the default ON before the 11-21 re-pin", "fail": [], "undefined": [], "held": False}
    if fail:
        act = "STOP with the data: %s FAIL" % ",".join(fail)
        if und:
            act += "; %s UNDEFINED" % ",".join(und)
        return {"status": FAIL, "action": act, "fail": fail, "undefined": und, "held": held}
    act = ("STOP with the data: %s UNDEFINED; re-run each missing or UNDEFINED cell once at the same revision (logged before "
           "its result is read), then rescore; a still-UNDEFINED cell never counts" % ",".join(und))
    if held:
        act += "; B5 does not reproduce the 10-26 gate: the ON half is not read until that is explained"
    return {"status": UNDEFINED, "action": act, "fail": [], "undefined": und, "held": held}


def score_all(on_dir, off_dir, seeds=GATE_SEEDS, arms=ARMS, lb_base=LB_BASE, on_tag=LB_ON_TAG, off_tag=LB_OFF_TAG,
              reference_path=REFERENCE, pin=None, only=None):
    only = set(only or BARS)
    bars = []
    ref = None
    if os.path.exists(reference_path):
        with open(reference_path) as f:
            ref = json.load(f)
    if only & {"B1", "B2", "B3"}:
        bars += [b for b in score_b123(stage2_aggregate(on_dir, seeds), seeds) if b["bar"] in only]
    if "B4" in only:
        bars.append(score_b4(dir_loader(on_dir), seeds, arms))
    if "B5" in only:
        bars.append(score_b5(stage2_aggregate(off_dir, seeds), ref, seeds) if ref is not None
                    else _bar("B5", [], ["reference aggregate %s missing" % reference_path], "no reference", mismatch=False))
    if "B6" in only:
        bars.append(score_b6(dir_loader(on_dir), dir_loader(off_dir), seeds, arms))
    if "B7" in only:
        bars.append(score_b7(on_dir, off_dir, seeds, arms, lb_base, on_tag, off_tag))
    if "B8" in only:
        bars.append(score_b8(lb_base, on_tag, off_tag, seeds, pin))
    bars.sort(key=lambda b: BARS.index(b["bar"]))
    return bars


def render(bars, dec):
    lines = []
    for b in bars:
        lines.append("%-3s %-9s %s" % (b["bar"], b["status"], b["detail"]))
        for k in ("failures", "undefined"):
            for x in b[k][:12]:
                lines.append("      %s %s" % ("FAIL" if k == "failures" else "UNDEFINED", x))
            if len(b[k]) > 12:
                lines.append("      ... and %d more %s" % (len(b[k]) - 12, k))
    n_def = sum(1 for b in bars if b["status"] != UNDEFINED)
    lines.append("bars DEFINED: %d of %d read" % (n_def, len(bars)))
    lines.append("DEC %s -> %s" % (dec["status"], dec["action"]))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--on-dir", default=STAGE2_ON_DIR)
    ap.add_argument("--off-dir", default=STAGE2_OFF_DIR)
    ap.add_argument("--seeds", default=",".join(str(s) for s in GATE_SEEDS))
    ap.add_argument("--lb-base", default=LB_BASE)
    ap.add_argument("--lb-on-tag", default=LB_ON_TAG)
    ap.add_argument("--lb-off-tag", default=LB_OFF_TAG)
    ap.add_argument("--reference", default=REFERENCE, help="the 10-26 gate aggregate B5 compares the OFF arms with")
    ap.add_argument("--pin", default=None, help="full sha: B8 cells must match it in their provenance sidecars (lb_shard rule)")
    ap.add_argument("--only", default=None, help="comma-separated bars to read (default all of B1..B8)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--strict", action="store_true", help="exit 1 unless DEC is PASS")
    a = ap.parse_args(argv)
    seeds = tuple(int(x) for x in a.seeds.split(","))
    only = [x.strip().upper() for x in a.only.split(",")] if a.only else None
    bars = score_all(a.on_dir, a.off_dir, seeds, ARMS, a.lb_base, a.lb_on_tag, a.lb_off_tag, a.reference, a.pin, only)
    dec = decide(bars) if not only else {"status": "PARTIAL", "action": "only %s read; no decision" % ",".join(sorted(only)),
                                         "fail": [], "undefined": [], "held": False}
    print(render(bars, dec))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump({"runner": "research/runners/emo_flip_score.py", "prereg": PREREG, "seeds": list(seeds),
                       "on_dir": a.on_dir, "off_dir": a.off_dir, "lb_tags": [a.lb_on_tag, a.lb_off_tag], "pin": a.pin,
                       "bars": {b["bar"]: b for b in bars}, "dec": dec}, f, indent=1, default=str)
        print("-> %s" % a.out)
    return 1 if (a.strict and dec["status"] != PASS) else 0


if __name__ == "__main__":
    sys.exit(main())
