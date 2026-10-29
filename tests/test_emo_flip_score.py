"""research/runners/emo_flip_score.py: the reader the BRAIN_EMOTION_CATEGORY_BOUNDED flip prereg owes
(research/findings/2026-10-27-emo-bounded-flip-PREREGISTRATION.md, sections 2 and 6).

Every test runs on SYNTHETIC records in the shapes the Stage-2 runner and tools/lb_shard.py write (no brain is built).
The failing direction comes first in each block: a brain that breaks a bar must read FAIL, a record that cannot show the
bar held must read UNDEFINED (never PASS), and only then does the perfect brain read PASS.
"""
import copy
import json
import math
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

from research.runners import emo_flip_score as S  # noqa: E402
from research.runners import emotion_category_organ as E  # noqa: E402
from research.runners import stage2_exit_gate as G  # noqa: E402

SEEDS = S.GATE_SEEDS
ARMS = S.ARMS
CATS = ("joy", "fear", "anger", "sadness")
AB = "I don't know about that."


# ── synthetic Stage-2 emo shards (production reply surface; emotion_category block with the engagement keys) ──────────
def _feel_answer(d, arousal, cat):
    sign = "positive" if d > 0.03 else ("negative" if d < -0.03 else "neutral")
    return ("My affect monitor reads %s right now (valence differential %+.3f, %s arousal) -- that's a functional read of "
            "my mood state, not a felt experience.%s" % (sign, d, arousal, E.feel_clause(cat)))


def _named(arm, seed, joy_none=False):
    """The category each feel turn names for this arm (feel_base first)."""
    base = base_of(arm)
    if base == "permuted":
        got = [G.SITUATION_SWAP[c] for c in CATS]
    elif base == "lesion_affect":
        got = [None] * 4
    else:
        got = list(CATS)
    if joy_none and base in ("intact_a", "intact_b", "permuted"):
        got = [None if c == "joy" else c for c in got] if base != "permuted" else [None if c == "sadness" else c for c in got]
    return [None] + got


def base_of(arm):
    return arm.split(".rep")[0]


def make_shard(seed, arm, on=True, joy_none=False, oor=0, **kw):
    named = _named(arm, seed, joy_none)
    lesion = base_of(arm) == "lesion_affect"
    resp, k = {}, 0
    for label, _msg, role in G.turns_for("emo", seed, arm):
        if role == "feel":
            cat = named[k]
            # the printed number is the PRODUCTION ladder's, independent of whether the organ latched a channel
            d = 0.0 if (lesion or k == 0) else (0.06 if (k == 1) != (base_of(arm) == "permuted") and k in (1, 4) else -0.06)
            k += 1
            ec = {"category": cat, "latched": [cat] if cat else [], "route": "emotion_category", "read_invariant": True,
                  "bounded_flag": on, "bounded_env": "1" if on else None, "bounded_readback": on,
                  "oor_count": 0, "oor_count_session": oor}
            resp[label] = {"answer": _feel_answer(d, "moderate", cat), "abstained": False, "recalled_svo": None,
                           "source": G.EXPECTED_SOURCE, "inner_state_readout": True,
                           "affect": {"differential": d, "lesioned": lesion, "valence_sign": "+" if d > 0 else "-"},
                           "emotion_category": ec}
        else:
            resp[label] = {"answer": AB, "abstained": True, "recalled_svo": None, "source": G.EXPECTED_SOURCE,
                           "affect": {"differential": 0.01, "lesioned": lesion}, "verified": True}
    sh = {"seed": seed, "group": "emo", "arm": arm, "responses": resp, "sources": [G.EXPECTED_SOURCE],
          "pythonhashseed_ok": True, "corpus_present": True,
          "builtin_facts": [["dog", "chase", "cat"], ["cat", "eat", "fish"]],
          "turns": [{"label": l, "message": m, "role": r} for l, m, r in G.turns_for("emo", seed, arm)],
          "process": {"peak_rss_mb": 6300.0, "ltm_bundle": "wikidata_100k"}, "timings_s": {"x": 1.0 + seed}}
    sh.update(kw)
    return sh


def tree(on=True, joy_none_seeds=(), **kw):
    return {s: {a: make_shard(s, a, on=on, joy_none=(s in joy_none_seeds), **kw) for a in ARMS} for s in SEEDS}


def write_tree(root, data):
    for seed, arms in data.items():
        for arm, sh in arms.items():
            p = G.shard_path(str(root), seed, "emo", arm)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                json.dump(sh, f)
    return str(root)


def drop_shard(data, seed, arm):
    d = copy.deepcopy(data)
    del d[seed][arm]
    return d


# ══════════════════════════════════════════════ pure helpers ═════════════════════════════════════════════════════════
def test_drop_keys_removes_the_five_at_any_depth_and_only_them():
    o = {"a": 1, "bounded_flag": True, "n": {"oor_count": 2, "keep": [1, {"bounded_env": "1", "x": 3}]},
         "oor_count_session": 0, "bounded_readback": False, "bounded_extra": 9}
    assert S.drop_keys(o) == {"a": 1, "n": {"keep": [1, {"x": 3}]}, "bounded_extra": 9}
    assert set(S.ENGAGEMENT_KEYS) == {"bounded_flag", "bounded_env", "bounded_readback", "oor_count", "oor_count_session"}


def test_first_diff_names_the_path_and_catches_one_key_drift():
    # first_diff(a, b) prints 'path: <a value> vs <b value>'; B6 calls it as (ON, OFF)
    a = {"affect": {"differential": 0.0386, "tags": ["x", {"k": 1}]}, "answer": "hi"}
    assert S.first_diff(a, copy.deepcopy(a)) is None
    b = copy.deepcopy(a)
    b["affect"]["differential"] = 0.0387
    assert S.first_diff(a, b).startswith("affect.differential: ")
    b = copy.deepcopy(a)
    b["affect"]["tags"][1]["k"] = 2
    assert S.first_diff(a, b).startswith("affect.tags[1].k: ")
    b = copy.deepcopy(a)
    b["new_key"] = 1
    assert S.first_diff(a, b) == "new_key: absent vs 1"
    b = copy.deepcopy(a)
    del b["answer"]
    assert S.first_diff(a, b) == "answer: 'hi' vs absent"
    assert S.first_diff({"x": [1, 2]}, {"x": [1, 2, 3]}).startswith("x: list length 2 vs 3")
    assert S.first_diff({"x": float("nan")}, {"x": float("nan")}) is None          # NaN == NaN here
    assert S.first_diff({"x": 1}, {"x": 1.0}) is None                              # json int/float of one number
    assert S.first_diff({"x": True}, {"x": 1}) is not None                         # a bool is not a count


def test_split_clause_reads_each_of_the_five_clauses_only_at_the_end():
    for cat in CATS + (None,):
        pre = "My affect monitor reads positive right now (valence differential +0.060, moderate arousal) -- f."
        assert S.split_clause(pre + E.feel_clause(cat)) == (pre, E.feel_clause(cat))
    assert S.split_clause("plain text") == ("plain text", None)
    assert S.split_clause(E.feel_clause("joy") + " And more.")[1] is None           # not at the end: not the clause
    assert S.split_clause(None) == ("", None)


# ══════════════════════════════════════════════ B4 ═══════════════════════════════════════════════════════════════════
def b4(data):
    return S.score_b4(S.mem_loader(data))


def test_b4_a_value_that_breaks_the_bar_is_fail_and_names_shard_and_turn():
    for key, bad in (("bounded_flag", False), ("bounded_readback", False), ("bounded_env", "0"), ("bounded_env", None),
                     ("oor_count_session", 1), ("oor_count_session", 19)):
        d = tree()
        d[101]["lesion_affect.rep0"]["responses"]["feel_joy"]["emotion_category"][key] = bad
        r = b4(d)
        assert r["status"] == S.FAIL, (key, bad)
        assert any("s101/emo__lesion_affect.rep0 feel_joy" in f and key in f for f in r["failures"]), r["failures"]


def test_b4_a_shard_turn_or_key_that_is_not_there_is_undefined_never_a_pass():
    d = tree()
    for sh in d[42].values():                                            # a pre-merge shard: no engagement keys at all
        for l, r in sh["responses"].items():
            if l.startswith("feel_"):
                for k in S.ENGAGEMENT_KEYS:
                    r["emotion_category"].pop(k)
    r = b4(d)
    assert r["status"] == S.UNDEFINED and not r["failures"]
    assert any(x.startswith("s42/emo__intact_a feel_base: lacks key(s)") for x in r["undefined"])
    r = b4(drop_shard(tree(), 100, "permuted"))
    assert r["status"] == S.UNDEFINED and any("s100/emo__permuted: shard missing" in x for x in r["undefined"])
    d = tree()
    del d[44]["intact_b"]["responses"]["feel_anger"]
    assert b4(d)["status"] == S.UNDEFINED
    d = tree()
    d[43]["intact_a"]["responses"]["feel_fear"] = {"_error": "RuntimeError: boom"}
    assert b4(d)["status"] == S.UNDEFINED
    d = tree()
    del d[43]["intact_a"]["responses"]["feel_fear"]["emotion_category"]
    r = b4(d)
    assert r["status"] == S.UNDEFINED and any("no emotion_category block" in x for x in r["undefined"])
    d = tree()
    d[42]["intact_a"]["responses"]["feel_joy"]["emotion_category"]["oor_count_session"] = False   # a bool is not a count
    assert b4(d)["status"] == S.UNDEFINED
    d = tree()
    d[42]["intact_a"]["responses"]["feel_joy"]["emotion_category"]["oor_count_session"] = None
    assert b4(d)["status"] == S.UNDEFINED


def test_b4_all_five_arms_are_required_including_the_repeat():
    d = tree()
    for s in SEEDS:
        del d[s]["lesion_affect.rep0"]
    r = b4(d)
    assert r["status"] == S.UNDEFINED and len(r["undefined"]) == 6


def test_b4_a_failure_decides_even_with_something_missing_and_a_perfect_brain_passes():
    d = tree()
    d[42]["intact_a"]["responses"]["feel_joy"]["emotion_category"]["oor_count_session"] = 2
    d = drop_shard(d, 43, "permuted")
    assert b4(d)["status"] == S.FAIL
    r = b4(tree())
    assert r["status"] == S.PASS and r["n_shards"] == 30 and r["n_feel_reads"] == 150
    # a per-read oor_count is not the bar's field: the session count is
    d = tree()
    d[42]["intact_a"]["responses"]["feel_joy"]["emotion_category"]["oor_count"] = 0
    assert b4(d)["status"] == S.PASS


def test_b4_the_off_arm_does_not_pass_b4():
    assert b4(tree(on=False))["status"] == S.FAIL                         # flag not engaged: bounded_flag False


# ══════════════════════════════════════════════ B6 ═══════════════════════════════════════════════════════════════════
def b6(on, off):
    return S.score_b6(S.mem_loader(on), S.mem_loader(off))


def _pair():
    """A perfect brain across the flip: seed 101 joy is None OFF and joy ON (the fix); engagement keys differ; every
    other field of every non-feel turn is identical."""
    return tree(on=True), tree(on=False, joy_none_seeds=(101,))


def test_b6_a_changed_non_feel_record_is_fail_with_the_first_differing_path():
    # one-key drift the three-field flip-check cannot see (affect.differential is not answer/abstained/recalled_svo)
    on, off = _pair()
    on[102]["intact_a"]["responses"]["s_fear_2"]["affect"]["differential"] = 0.0100001
    r = b6(on, off)
    assert r["status"] == S.FAIL and len(r["failures"]) == 1
    assert "s102/emo__intact_a s_fear_2: first differing path affect.differential" in r["failures"][0]
    for field, val in (("answer", "I do know about that."), ("abstained", False), ("recalled_svo", ["a", "b", "c"])):
        on, off = _pair()
        on[42]["permuted"]["responses"]["s_joy_1"][field] = val
        r = b6(on, off)
        assert r["status"] == S.FAIL and "s_joy_1: first differing path %s" % field in r["failures"][0], field
    on, off = _pair()
    on[42]["intact_b"]["responses"]["s_joy_1"]["new_field"] = 1                       # a key only the ON arm carries
    assert "new_field: 1 vs absent" in b6(on, off)["failures"][0]
    on, off = _pair()
    del on[42]["intact_b"]["responses"]["s_joy_1"]["verified"]
    assert "verified: absent vs True" in b6(on, off)["failures"][0]
    on, off = _pair()                                                                 # a SIXTH engagement-like key is not dropped
    on[42]["intact_b"]["responses"]["s_joy_1"]["bounded_extra"] = True
    assert b6(on, off)["status"] == S.FAIL


def test_b6_feel_text_before_the_clause_must_be_identical():
    on, off = _pair()
    a = on[43]["intact_a"]["responses"]["feel_fear"]
    a["answer"] = a["answer"].replace("positive", "negative").replace("reads neutral", "reads negative")
    a["answer"] = a["answer"].replace("-0.060", "-0.061")
    r = b6(on, off)
    assert r["status"] == S.FAIL and "s43/emo__intact_a feel_fear: feel text before the clause differs" in r["failures"][0]
    on, off = _pair()                                                                 # the clause missing from the ON reply
    a = on[43]["intact_a"]["responses"]["feel_fear"]
    a["answer"] = S.split_clause(a["answer"])[0]
    r = b6(on, off)
    assert r["status"] == S.FAIL and "category clause present in the OFF arm only" in r["failures"][0]
    on, off = _pair()                                                                 # a sentence appended AFTER the clause
    on[43]["intact_a"]["responses"]["feel_fear"]["answer"] += " Also, hello."
    assert b6(on, off)["status"] == S.FAIL


def test_b6_undefined_when_the_comparison_cannot_be_made():
    on, off = _pair()
    assert b6(drop_shard(on, 42, "intact_a"), off)["status"] == S.UNDEFINED
    assert b6(on, drop_shard(off, 42, "lesion_affect.rep0"))["status"] == S.UNDEFINED
    del on[42]["intact_a"]["responses"]["s_joy_1"]
    r = b6(on, off)
    assert r["status"] == S.UNDEFINED and "turn labels differ" in r["undefined"][0]
    on, off = _pair()
    off[42]["intact_a"]["responses"]["s_joy_1"] = {"_error": "x"}
    assert b6(on, off)["status"] == S.UNDEFINED
    on, off = _pair()
    off[42]["intact_a"]["responses"]["s_joy_1"]["answer"] = "changed"                 # FAIL still decides over UNDEFINED
    assert b6(drop_shard(on, 44, "intact_b"), drop_shard(off, 44, "intact_b"))["status"] == S.FAIL


def test_b6_a_perfect_brain_passes_even_when_the_clause_and_the_five_keys_differ():
    on, off = _pair()
    r = b6(on, off)
    assert r["status"] == S.PASS and r["n_pairs"] == 30 and r["n_nonfeel_turns"] == 240 and r["n_feel_turns"] == 150
    # seed 101 really does differ across the flip in its clause and in the five keys
    ja, jb = on[101]["intact_a"]["responses"]["feel_joy"], off[101]["intact_a"]["responses"]["feel_joy"]
    assert ja["answer"] != jb["answer"] and ja["emotion_category"]["bounded_flag"] != jb["emotion_category"]["bounded_flag"]
    # the five keys at depth in a non-feel turn may differ too ("at any depth")
    on, off = _pair()
    on[42]["intact_a"]["responses"]["s_joy_1"]["emotion_category"] = {"category": None, "bounded_flag": True, "oor_count": 0}
    off[42]["intact_a"]["responses"]["s_joy_1"]["emotion_category"] = {"category": None, "bounded_flag": False, "oor_count": 3}
    assert b6(on, off)["status"] == S.PASS
    # ... but not the category beside them
    off[42]["intact_a"]["responses"]["s_joy_1"]["emotion_category"]["category"] = "joy"
    assert b6(on, off)["status"] == S.FAIL


def test_the_prereg_flip_check_feel_flag_fires_on_a_perfect_pair_which_is_why_b6_does_not_read_it(tmp_path):
    """emotion_category_dev.flip_check flags FEEL-NOT-APPEND-ONLY unless ON == OFF + one clause, i.e. unless the OFF arm has
    NO clause. That is the organ-flag comparison (BRAIN_EMOTION_CATEGORY=0). Bounded ON vs UNSET: BOTH arms append a clause,
    so it fires on every feel turn of a perfect brain. B6 reads the ruling's wording (text before the clause) instead."""
    from research.runners.emotion_category_dev import flip_check
    on, off = _pair()
    on_dir, off_dir = write_tree(tmp_path / "on", on), write_tree(tmp_path / "off", off)
    rep = flip_check(on_dir, off_dir, SEEDS)
    assert "FEEL-NOT-APPEND-ONLY" in rep["flags"]
    assert all(len(v["feel_bad"]) == 5 for v in rep["pairs"].values())              # every feel turn of every pair
    assert "FLIP-HURTS-ANSWERS" not in rep["flags"] and "FLIP-CHANGES-NONFEEL" not in rep["flags"]
    assert S.score_b6(S.dir_loader(on_dir), S.dir_loader(off_dir))["status"] == S.PASS


# ══════════════════════════════════════════════ B1-B3 and B5 (the Stage-2 aggregate) ═════════════════════════════════
def agg(data, tmp_path, name):
    return S.stage2_aggregate(write_tree(tmp_path / name, data), SEEDS)


def test_b123_a_defined_seed_that_fails_is_fail_and_a_missing_shard_is_undefined(tmp_path):
    ok = S.score_b123(agg(tree(), tmp_path, "ok"), SEEDS)
    assert [b["status"] for b in ok] == [S.PASS] * 3
    assert ok[1]["attributable_fraction"] == {"s%d" % x: 1.0 for x in SEEDS}      # the lesion owns the whole change; the null none
    # one seed's joy goes unnamed (the 10-26 s101 failure): B1 FAIL, B2 FAIL (a lesion cannot change a None), B3 UNDEFINED
    bad = S.score_b123(agg(tree(joy_none_seeds=(101,)), tmp_path, "joynone"), SEEDS)
    assert [b["status"] for b in bad] == [S.FAIL, S.FAIL, S.UNDEFINED]
    assert "s101" in bad[0]["failures"][0] and bad[0]["n_correct"] == 5
    # the permuted arm made inert (it reads like the intact one): only B3 fails
    d = tree()
    for s in SEEDS:
        d[s]["permuted"] = make_shard(s, "permuted")
        d[s]["permuted"]["responses"] = copy.deepcopy(d[s]["intact_a"]["responses"])
    r = S.score_b123(agg(d, tmp_path, "perm_inert"), SEEDS)
    assert [b["status"] for b in r] == [S.PASS, S.PASS, S.FAIL]
    # the lesion made inert (it names what the intact arm names): B2 fails, and it cannot be called reproduced-only
    d = tree()
    for s in SEEDS:
        for arm in ("lesion_affect", "lesion_affect.rep0"):
            d[s][arm]["responses"] = copy.deepcopy(d[s]["intact_a"]["responses"])
    r = S.score_b123(agg(d, tmp_path, "les_inert"), SEEDS)
    assert [b["status"] for b in r] == [S.PASS, S.FAIL, S.PASS]
    # a missing repeat shard: that seed is undefined for all three, and nothing is called a pass
    r = S.score_b123(agg(drop_shard(tree(), 100, "lesion_affect.rep0"), tmp_path, "norep"), SEEDS)
    assert [b["status"] for b in r] == [S.UNDEFINED] * 3
    # a lesion that does not reproduce
    d = tree()
    d[42]["lesion_affect.rep0"]["responses"]["feel_joy"]["answer"] += " (different)"
    r = S.score_b123(agg(d, tmp_path, "unrep"), SEEDS)
    assert r[1]["status"] == S.FAIL and "does not reproduce" in " ".join(r[1]["failures"])


def _reference(tmp_path):
    """A stand-in for the 10-26 gate aggregate: the OFF brain with seed 101 joy unnamed, as JSON (string seed keys)."""
    return json.loads(json.dumps(agg(tree(on=False, joy_none_seeds=(101,)), tmp_path, "ref"), default=str))


def test_b5_the_off_arms_must_reproduce_the_10_26_result_else_the_off_half_is_undefined(tmp_path):
    ref = _reference(tmp_path)
    same = S.score_b5(agg(tree(on=False, joy_none_seeds=(101,)), tmp_path, "off_same"), ref, SEEDS)
    assert same["status"] == S.PASS and same["compared"] == 6 and not same["mismatch"]
    fixed = S.score_b5(agg(tree(on=False), tmp_path, "off_fixed"), ref, SEEDS)       # OFF now names joy on s101: no reproduction
    assert fixed["status"] == S.UNDEFINED and fixed["mismatch"] and any("s101" in x for x in fixed["undefined"])
    other = S.score_b5(agg(tree(on=False, joy_none_seeds=(42, 101)), tmp_path, "off_other"), ref, SEEDS)
    assert other["status"] == S.UNDEFINED and other["mismatch"]
    missing = S.score_b5(agg(drop_shard(tree(on=False, joy_none_seeds=(101,)), 43, "intact_b"), tmp_path, "off_missing"), ref, SEEDS)
    assert missing["status"] == S.UNDEFINED and not missing["mismatch"]
    assert S.score_b5(agg(tree(on=False), tmp_path, "off_noref"), {}, SEEDS)["status"] == S.UNDEFINED
    # a reference that no longer says s101 joy None is not the bar's premise
    bad_ref = copy.deepcopy(ref)
    bad_ref["groups"]["emo"]["seeds"]["101"]["intact"]["named"]["joy"] = "joy"
    r = S.score_b5(agg(tree(on=False), tmp_path, "off_badref"), bad_ref, SEEDS)
    assert r["status"] == S.UNDEFINED and any("premise" in x for x in r["undefined"])
    # a dev seed has no 10-26 reference: UNDEFINED (missing), not a pass
    r = S.score_b5(S.stage2_aggregate(write_tree(tmp_path / "dev", {7: tree(on=False)[42]}), (7,)), ref, (7,))
    assert r["status"] == S.UNDEFINED and "no such seed" in r["undefined"][0]


# ══════════════════════════════════════════════ B7 ═══════════════════════════════════════════════════════════════════
def write_lb(base, tag, seed, answers=("The channel is joy.", "The channel is joy."), flag=True, verdict="regressed",
             lb=True, clean=True, repro=True, with_lb=True, arms=("intact_a", "intact_b"), lesion_arms=("", ".rep0"),
             lesion_flag=None, commit_lesioned=True, intact_commit_lesioned=False):
    """One row cell: the intact arms, the lesion arm and its repeats (`lesion_arms`: suffixes after
    lesion_emotion_category_s<seed>.json), and lb.json. `lesion_flag` (default `flag`) and `commit_lesioned` set the
    lesion arms' engagement; `intact_commit_lesioned` the intact arms'."""
    cell = os.path.join(str(base), tag, "s%d" % seed, "emotion-category")
    os.makedirs(cell, exist_ok=True)

    def doc(ans, cat, fl, cl):
        return {"emocat_s1": {"answer": AB, "abstained": True},
                "emocat_feel": {"answer": ans, "abstained": False, "emotion_category": {
                    "category": cat, "commit_lesioned": cl, "bounded_flag": fl, "bounded_env": "1" if fl else None,
                    "bounded_readback": fl, "oor_count": 0, "oor_count_session": 0}}}
    for arm, ans in zip(arms, answers):
        with open(os.path.join(cell, "%s_emocat_s1_emocat_s2_emocat_feel_s%d.json" % (arm, seed)), "w") as f:
            json.dump(doc(ans, "joy", flag, intact_commit_lesioned), f)
    lf = flag if lesion_flag is None else lesion_flag
    for sfx in lesion_arms:
        with open(os.path.join(cell, "lesion_emotion_category_s%d.json%s" % (seed, sfx)), "w") as f:
            json.dump(doc("No single emotion channel latched.", None, lf, commit_lesioned), f)
    if with_lb:
        rep = {"seed": seed, "per_faculty": [{"faculty": "emotion-category", "turn": "emocat_feel", "kind": "neural-lesion",
                                              "load_bearing": lb, "verdict": verdict, "null_control_clean": clean,
                                              "lesion_reproduced": repro}]}
        with open(os.path.join(cell, "lb.json"), "w") as f:
            json.dump(rep, f)


def lb_tree(base, on_kw=None, off_kw=None, bad=None):
    """Both tags, six seeds. bad = {(tag-kind, seed): kwargs} overrides one cell."""
    for s in SEEDS:
        kw_on, kw_off = dict(on_kw or {}), dict(off_kw or {})
        kw_on.update((bad or {}).get(("on", s), {}))
        kw_off.update((bad or {}).get(("off", s), {}))
        write_lb(base, S.LB_ON_TAG, s, **dict({"flag": True}, **kw_on))
        write_lb(base, S.LB_OFF_TAG, s, **dict({"flag": False}, **kw_off))
    return str(base)


def test_b7_a_flip_that_hurts_answers_is_fail_from_either_instrument(tmp_path):
    on, off = _pair()
    on_dir, off_dir = write_tree(tmp_path / "on", on), write_tree(tmp_path / "off", off)
    lb = lb_tree(tmp_path / "lb")
    ok = S.score_b7(on_dir, off_dir, SEEDS, ARMS, lb)
    assert ok["status"] == S.PASS and "FLIP-HURTS-ANSWERS" not in ok["flip_check_flags"]
    # (1) flip_check: OFF answered a turn that ON abstains on
    on2 = copy.deepcopy(on)
    off2 = copy.deepcopy(off)
    off2[43]["intact_a"]["responses"]["s_joy_1"] = {"answer": "the cat eats the fish", "abstained": False, "recalled_svo": None,
                                                    "source": G.EXPECTED_SOURCE}
    r = S.score_b7(write_tree(tmp_path / "on2", on2), write_tree(tmp_path / "off2", off2), SEEDS, ARMS, lb)
    assert r["status"] == S.FAIL and "flip_check FLIP-HURTS-ANSWERS on s43/emo__intact_a.json" in r["failures"][0]
    # (2) answer_rate paired: the flag-ON row abstains on seed 100 where flag-OFF answers
    lb2 = lb_tree(tmp_path / "lb2", bad={("on", 100): {"answers": ("I don't know about that.", "I don't know about that.")}})
    r = S.score_b7(on_dir, off_dir, SEEDS, ARMS, lb2)
    assert r["status"] == S.FAIL and any("answer_rate paired FLIP-HURTS-ANSWERS on seed(s) 100" in x for x in r["failures"])


def test_b7_missing_pairs_and_unmeasured_seeds_are_undefined(tmp_path):
    on, off = _pair()
    lb = lb_tree(tmp_path / "lb")
    r = S.score_b7(write_tree(tmp_path / "on", drop_shard(on, 44, "permuted")), write_tree(tmp_path / "off", off), SEEDS, ARMS, lb)
    assert r["status"] == S.UNDEFINED and any("s44/emo__permuted.json: pair missing" in x for x in r["undefined"])
    r = S.score_b7(write_tree(tmp_path / "on3", on), write_tree(tmp_path / "off3", off), SEEDS, ARMS, str(tmp_path / "nolb"))
    assert r["status"] == S.UNDEFINED and sum("UNMEASURED" in x for x in r["undefined"]) == 6


# ══════════════════════════════════════════════ B8 ═══════════════════════════════════════════════════════════════════
def test_b8_load_bearing_on_six_of_six_defined_cells_else_not_a_pass(tmp_path):
    ok = S.score_b8(lb_tree(tmp_path / "ok"))
    assert ok["status"] == S.PASS and ok["n_load_bearing_on"] == 6 and ok["n_load_bearing_off"] == 6
    # a DEFINED cell that is not load-bearing (the lesion did not move the reply): FAIL
    r = S.score_b8(lb_tree(tmp_path / "notlb", bad={("on", 102): dict(lb=False, verdict="pass")}))
    assert r["status"] == S.FAIL and "s102 (flag ON)" in r["failures"][0] and r["n_load_bearing_on"] == 5
    # a noisy / unreliable cell, a missing cell, a dirty null, a lesion that did not reproduce: UNDEFINED
    for label, kw in (("noisy", dict(lb=None, verdict="noisy")), ("dirty", dict(clean=False, lb=None, verdict="noisy")),
                      ("unrep", dict(repro=False, lb=None, verdict="noisy")), ("nolb", dict(with_lb=False))):
        r = S.score_b8(lb_tree(tmp_path / label, bad={("on", 43): kw}))
        assert r["status"] == S.UNDEFINED and not r["failures"], label
    # lb True but the null was not clean must not read as load-bearing
    r = S.score_b8(lb_tree(tmp_path / "lbdirty", bad={("on", 43): dict(clean=False)}))
    assert r["status"] == S.UNDEFINED
    # the flag was not engaged in a flag-ON cell: that is not a measurement of the flag ON
    r = S.score_b8(lb_tree(tmp_path / "noeng", bad={("on", 44): dict(flag=False)}))
    assert r["status"] == S.UNDEFINED and "flag not engaged" in r["undefined"][0]
    # a FAIL decides over an UNDEFINED cell
    r = S.score_b8(lb_tree(tmp_path / "mix", bad={("on", 44): dict(with_lb=False), ("on", 42): dict(lb=False, verdict="pass")}))
    assert r["status"] == S.FAIL
    # the OFF row is reported beside it, and does not change the bar
    r = S.score_b8(lb_tree(tmp_path / "offnot", bad={("off", 42): dict(lb=False, verdict="pass")}))
    assert r["status"] == S.PASS and r["n_load_bearing_off"] == 5 and r["off_cells"]["s42"]["status"] == "NOT-LB"


def test_b8_reads_the_lesion_arms_engagement_too(tmp_path):
    """verify-go 2026-10-28: B8's engagement check read only intact*.json, so a cell whose lesion and .rep0 arms ran
    with the flag OFF and without the lesion (commit_lesioned False, bounded_flag False) still read LB. Each arm must
    now show its tag's flag state and its role's lesion state, else the cell is UNDEFINED (never LB)."""
    assert S.score_b8(lb_tree(tmp_path / "ok"))["status"] == S.PASS
    # the panel's mutation: the lesion arms of an ON cell ran OFF and unlesioned
    r = S.score_b8(lb_tree(tmp_path / "mut", bad={("on", 42): dict(lesion_flag=False, commit_lesioned=False)}))
    assert r["status"] == S.UNDEFINED and r["on_cells"]["s42"]["status"] == "UNDEFINED"
    assert "flag not engaged in lesion_emotion_category_s42.json" in r["on_cells"]["s42"]["why"]
    # only the .rep0 repeat ran with the flag OFF: still UNDEFINED, and named
    assert S.score_b8(lb_tree(tmp_path / "rep"))["status"] == S.PASS
    cell = os.path.join(str(tmp_path / "rep"), S.LB_ON_TAG, "s43", "emotion-category")
    p = os.path.join(cell, "lesion_emotion_category_s43.json.rep0")
    d = json.load(open(p))
    d["emocat_feel"]["emotion_category"]["bounded_flag"] = False
    json.dump(d, open(p, "w"))
    r = S.score_b8(str(tmp_path / "rep"))
    assert r["status"] == S.UNDEFINED and "lesion_emotion_category_s43.json.rep0" in r["on_cells"]["s43"]["why"]
    # the lesion did not engage on the lesion arm (flag state right): UNDEFINED
    r = S.score_b8(lb_tree(tmp_path / "nocommit", bad={("on", 44): dict(commit_lesioned=False)}))
    assert r["status"] == S.UNDEFINED and "commit_lesioned=False" in r["on_cells"]["s44"]["why"]
    # the lesion key absent from the lesion arm's read: UNDEFINED, not assumed
    r = S.score_b8(lb_tree(tmp_path / "absent", bad={("on", 100): dict(commit_lesioned=None)}))
    assert r["status"] == S.UNDEFINED and "commit_lesioned" in r["on_cells"]["s100"]["why"]
    # the lesion leaked into an intact arm: UNDEFINED
    r = S.score_b8(lb_tree(tmp_path / "leak", bad={("on", 101): dict(intact_commit_lesioned=True)}))
    assert r["status"] == S.UNDEFINED and "leaked into the intact arm" in r["on_cells"]["s101"]["why"]
    # an OFF cell whose lesion arm ran with the flag ON: UNDEFINED beside the bar ("the flag leaked in")
    r = S.score_b8(lb_tree(tmp_path / "offles", bad={("off", 102): dict(lesion_flag=True)}))
    assert r["status"] == S.PASS and r["off_cells"]["s102"]["status"] == "UNDEFINED"
    assert "leaked in" in r["off_cells"]["s102"]["why"]
    # no lesion arm at all beside lb.json: UNDEFINED
    r = S.score_b8(lb_tree(tmp_path / "noles", bad={("on", 42): dict(lesion_arms=())}))
    assert r["status"] == S.UNDEFINED and "no lesion arm file" in r["on_cells"]["s42"]["why"]


# ══════════════════════════════════════════════ the decision ═════════════════════════════════════════════════════════
def _b(name, status, **kw):
    return dict({"bar": name, "status": status, "detail": "", "failures": [], "undefined": []}, **kw)


def test_dec_pass_only_when_all_eight_pass_and_anything_else_is_stop_with_the_data():
    allpass = [_b(x, S.PASS) for x in S.BARS]
    d = S.decide(allpass)
    assert d["status"] == S.PASS and "flip the default ON" in d["action"]
    for i in range(8):
        bars = copy.deepcopy(allpass)
        bars[i]["status"] = S.FAIL
        d = S.decide(bars)
        assert d["status"] == S.FAIL and d["fail"] == [S.BARS[i]] and d["action"].startswith("STOP with the data")
        bars[i]["status"] = S.UNDEFINED
        d = S.decide(bars)
        assert d["status"] == S.UNDEFINED and "re-run" in d["action"] and "never counts" in d["action"]
    d = S.decide(allpass[:7])                                         # a bar not read is not a pass
    assert d["status"] == S.UNDEFINED and "B8(not read)" in d["undefined"]
    bars = copy.deepcopy(allpass)
    bars[4] = _b("B5", S.UNDEFINED, mismatch=True)
    d = S.decide(bars)
    assert d["held"] and "ON half is not read" in d["action"]
    bars[0]["status"] = S.FAIL
    assert S.decide(bars)["status"] == S.FAIL                         # FAIL is named first, UNDEFINED beside it
    assert "B1 FAIL" in S.decide(bars)["action"] and "B5 UNDEFINED" in S.decide(bars)["action"]


def test_cli_end_to_end_prints_each_bar_writes_the_record_and_strict_gates_the_exit_code(tmp_path, capsys):
    on, off = _pair()
    on_dir, off_dir = write_tree(tmp_path / "on", on), write_tree(tmp_path / "off", off)
    lb = lb_tree(tmp_path / "lb")
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps(agg(tree(on=False, joy_none_seeds=(101,)), tmp_path, "refsrc"), default=str))
    out = tmp_path / "verdict.json"
    args = ["--on-dir", on_dir, "--off-dir", off_dir, "--lb-base", lb, "--reference", str(ref), "--out", str(out)]
    assert S.main(args) == 0
    txt = capsys.readouterr().out
    for b in S.BARS:
        assert "\n%-3s PASS" % b in "\n" + txt
    assert "DEC PASS -> flip the default ON" in txt and "bars DEFINED: 8 of 8 read" in txt
    rec = json.loads(out.read_text())
    assert rec["dec"]["status"] == S.PASS and set(rec["bars"]) == set(S.BARS)
    assert S.main(args + ["--strict"]) == 0
    # break one thing: B4 fails, the decision is STOP, --strict exits 1, exit 0 without it
    on[43]["permuted"]["responses"]["feel_sadness"]["emotion_category"]["oor_count_session"] = 3
    on_dir2 = write_tree(tmp_path / "on_bad", on)
    args2 = ["--on-dir", on_dir2] + args[2:]
    assert S.main(args2) == 0
    txt = capsys.readouterr().out
    assert "B4  FAIL" in txt and "s43/emo__permuted feel_sadness: oor_count_session=3" in txt and "DEC FAIL -> STOP with the data" in txt
    assert S.main(args2 + ["--strict"]) == 1
    capsys.readouterr()
    # --only reads a subset and decides nothing
    assert S.main(["--on-dir", on_dir, "--off-dir", off_dir, "--only", "B4,B6"]) == 0
    txt = capsys.readouterr().out
    assert "DEC PARTIAL" in txt and "\nB1 " not in "\n" + txt and "\nB6 " in "\n" + txt


# ══════════════════════════════════════════════ pins against the repo ════════════════════════════════════════════════
def test_the_five_engagement_keys_are_the_organs_and_the_prereg_names_them():
    here = os.path.join(REPO, "research", "runners", "emotion_category_organ.py")
    src = open(here, encoding="utf8").read()
    for k in S.ENGAGEMENT_KEYS:
        assert '"%s"' % k in src, k
    prereg = open(os.path.join(REPO, S.PREREG), encoding="utf8").read()
    for k in S.ENGAGEMENT_KEYS:
        assert "`%s`" % k in prereg, k
    assert S.ARMS == ("intact_a", "intact_b", "permuted", "lesion_affect", "lesion_affect.rep0")
    assert set(G.arms_for("emo", with_informational=False)) == set(S.ARMS)


def test_b8_pin_failure_is_undefined(tmp_path, monkeypatch):
    import tools.lb_shard as LB
    base = lb_tree(tmp_path / "lb")
    monkeypatch.setattr(LB, "cell_prov_fails", lambda cell_dir, pin, rows, treatment=None: (["git_sha='x' (want y IN FULL)"], []))
    r = S.score_b8(base, pin="y" * 40)
    assert r["status"] == S.UNDEFINED and "provenance does not match the pin" in r["undefined"][0]
    monkeypatch.setattr(LB, "cell_prov_fails", lambda cell_dir, pin, rows, treatment=None: ([], []))
    assert S.score_b8(base, pin="y" * 40)["status"] == S.PASS


REF = os.path.join(REPO, S.REFERENCE)
GATE_1026 = os.path.join(REPO, "research", "findings", "raw", "_stage2_exit", "gate")


@pytest.mark.skipif(not (os.path.exists(REF) and os.path.isdir(GATE_1026)), reason="the committed 10-26 gate data is not in this checkout")
def test_the_committed_10_26_gate_reads_as_the_prereg_describes_it():
    """Instrument sanity on REAL shards: the 10-26 gate's emo group is the OFF reference (s101 joy None, 5 of 6 correct); read
    as an OFF directory it reproduces itself (B5 PASS), read as an ON directory it fails B1 and B2 on s101, and B4 is UNDEFINED
    on it (a pre-merge shard carries no engagement keys). Its intact_a and intact_b responses are identical whole-record, so
    the whole-record B6 is reachable by a deterministic build (self-pair: PASS)."""
    ref = json.load(open(REF))
    agg_ = S.stage2_aggregate(GATE_1026, SEEDS)
    assert S.score_b5(agg_, ref, SEEDS)["status"] == S.PASS
    b1, b2, b3 = S.score_b123(agg_, SEEDS)
    assert (b1["status"], b1["n_correct"]) == (S.FAIL, 5) and b2["status"] == S.FAIL and b3["status"] == S.UNDEFINED
    assert "s101" in b1["failures"][0]
    assert S.score_b4(S.dir_loader(GATE_1026))["status"] == S.UNDEFINED
    assert S.score_b6(S.dir_loader(GATE_1026), S.dir_loader(GATE_1026))["status"] == S.PASS
    a, b = S.dir_loader(GATE_1026), S.dir_loader(GATE_1026)
    for s in SEEDS:
        ia, ib = a(s, "intact_a")[0]["responses"], b(s, "intact_b")[0]["responses"]
        assert all(S.first_diff(S.drop_keys(ia[l]), S.drop_keys(ib[l])) is None for l in ia), s


def test_a_dev_seed_read_is_possible_and_reads_clean_for_the_gate_preflight(tmp_path, capsys):
    """tools/gate_preflight.py needs the registered scorer to read a dev-seed run of this packet CLEAN before any gate line is
    queued. Dev seed 7 has no 10-26 reference (B5) and no row shards unless run, so the preflight read is --only B1,B2,B3,B4,B6."""
    from tools.gate_preflight import classify
    on = {7: {a: make_shard(7, a, on=True) for a in ARMS}}
    off = {7: {a: make_shard(7, a, on=False) for a in ARMS}}
    args = ["--on-dir", write_tree(tmp_path / "on", on), "--off-dir", write_tree(tmp_path / "off", off), "--seeds", "7",
            "--only", "B1,B2,B3,B4,B6"]
    assert S.main(args) == 0
    out = capsys.readouterr().out
    assert "B1  PASS" in out and "B4  PASS" in out and "B6  PASS" in out and "DEC PARTIAL" in out
    assert classify(out)[0] == "CLEAN"
    # a pre-merge shard (no engagement keys) is an instrument defect, not a coverage gap: the preflight must FLAG it
    for sh in on[7].values():
        for l, r in sh["responses"].items():
            if l.startswith("feel_"):
                r["emotion_category"] = {k: v for k, v in r["emotion_category"].items() if k not in S.ENGAGEMENT_KEYS}
    args[1] = write_tree(tmp_path / "on_old", on)
    assert S.main(args) == 0
    assert classify(capsys.readouterr().out)[0] == "FLAGGED"
