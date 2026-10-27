"""bar-sanity spec: emo-bounded-flip B6, "Non-feel turns identical ON vs OFF apart from the five keys; feel text before the
category clause identical". decide = `research.runners.emo_flip_score.score_b6` (the scorer itself) over six gate seeds x the
five emo arms, ON tree against OFF tree, read as PASS. A pair that cannot be compared (a shard missing, different turn labels,
an errored turn) reads UNDEFINED, which is NOT a pass.

perfect  = a brain that differs across the flip in exactly the two admitted ways: the five engagement keys (at any depth) and,
           on some seeds, the CLAUSE (the OFF organ fails to latch joy, the ON organ names it, as seed 101 did at the 10-26 gate);
           the printed number and every other field are the same;
nulls    = one non-feel turn changes (answer / abstained / recalled_svo / one deep float / one list item / an extra key / a
           missing key / a SIXTH engagement-like key), one feel turn's text before the clause changes (words, the printed
           number, a dropped clause, a sentence after the clause), a pair that cannot be compared (a shard missing, a turn
           missing, an errored turn), every non-feel turn changed, and a random brain.
The deep-float and list-item nulls are the one-key drifts that the older `--flip-check` (answer / abstained / recalled_svo
only) cannot see. Needs research/runners/emo_flip_score.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_emo_flip_common as K  # noqa: E402

S = K.S

NAME = "emo-bounded-flip B6 (ON vs OFF whole record apart from the five keys)"
DESCRIPTION = ("emo_flip_score.score_b6 reads PASS over the six gate seeds x five arms, ON vs OFF; perfect = identical in every "
               "field of every non-feel turn after the five engagement keys are dropped at any depth, feel text before the "
               "clause identical (the clause itself and the keys may differ); nulls = one non-feel field / deep float / list "
               "item / key added or removed / a sixth key, a changed feel prefix, number or clause position, an unmatched or "
               "errored pair, every non-feel turn changed, and a random brain.")
SCORER_FILES = ["research/runners/emo_flip_score.py", "research/runners/stage2_exit_gate.py",
                "docs/drafts/emo-bounded-flip/bar_sanity_emo_flip_common.py", "docs/drafts/stage2-exit/bar_sanity_s2_common.py"]


def decide(data):
    return S.score_b6(S.mem_loader(data["on"]), S.mem_loader(data["off"])) ["status"] == S.PASS


def _pair(rng, **kw):
    on, off = K.draw_pair(rng, **kw)
    return {"on": on, "off": off}


def _expose(rng, d):
    s, a = K.pick(rng, d["on"])
    l = K.choice(rng, K.expose_labels(d["on"][s][a]))
    return d["on"][s][a]["responses"][l]


def _feel(rng, d):
    s, a = K.pick(rng, d["on"])
    l = K.choice(rng, K.feel_labels(d["on"][s][a]))
    return d["on"][s][a]["responses"][l]


# ── perfect brains ──────────────────────────────────────────────────────────────────────────────────────────────────
def perfect(rng):
    return _pair(rng)


def perfect_no_clause_change(rng):             # only the five keys differ (no seed's OFF organ misses a channel)
    return _pair(rng, joy_none_p=0.0)


def perfect_clause_changes_often(rng):         # the unbounded organ misses a channel on many seeds: more clauses differ
    return _pair(rng, joy_none_p=0.9)


def perfect_keys_at_depth(rng):                # a non-feel block carrying the five keys also differs ("at any depth")
    return _pair(rng, nested_keys=True)


# ── nulls: ONE defect at one random place in the ON tree ──────────────────────────────────────────────────────────────
def null_nonfeel_answer_changed(rng):
    d = _pair(rng)
    r = _expose(rng, d)
    r["answer"] = r["answer"] + " Hmm."
    return d


def null_nonfeel_abstained_flipped(rng):
    d = _pair(rng)
    r = _expose(rng, d)
    r["abstained"] = not r["abstained"]
    return d


def null_nonfeel_recalled_svo_changed(rng):
    d = _pair(rng)
    _expose(rng, d)["recalled_svo"] = ["dog", "chase", "cat"]
    return d


def null_nonfeel_deep_float_drift(rng):        # one number, four decimals down: not answer / abstained / recalled_svo
    d = _pair(rng)
    r = _expose(rng, d)
    r["affect"]["differential"] = round(r["affect"]["differential"] + 1e-4, 6)
    return d


def null_nonfeel_deep_list_item(rng):
    d = _pair(rng)
    r = _expose(rng, d)
    r["supporting_facts"] = [["dog", "chase", "cat"], ["cat", "eat", "worm"]]
    return d


def null_nonfeel_nested_bool(rng):
    d = _pair(rng)
    r = _expose(rng, d)
    r["affect"]["forthcomingness"]["max_sentences"] += 1
    return d


def null_nonfeel_extra_key(rng):
    d = _pair(rng)
    _expose(rng, d)["diagnostic"] = {"v_min": -80.0}
    return d


def null_nonfeel_key_missing(rng):
    d = _pair(rng)
    del _expose(rng, d)["verified"]
    return d


def null_nonfeel_sixth_engagement_key(rng):    # exactly FIVE keys may differ: a sixth bounded_* key is a difference
    d = _pair(rng)
    _expose(rng, d)["bounded_substeps"] = 4
    return d


def null_feel_prefix_words(rng):
    d = _pair(rng)
    r = _feel(rng, d)
    r["answer"] = r["answer"].replace("My affect monitor reads", "My affect monitor shows", 1)
    return d


def null_feel_printed_number(rng):             # the production ladder's number moves by one digit
    d = _pair(rng)
    r = _feel(rng, d)
    a = r["answer"]
    i = a.index("valence differential ") + len("valence differential ") + 1
    j = a.index(",", i)
    r["answer"] = a[: j - 1] + ("1" if a[j - 1] != "1" else "2") + a[j:]
    return d


def null_feel_clause_dropped_on(rng):
    d = _pair(rng)
    r = _feel(rng, d)
    r["answer"] = S.split_clause(r["answer"])[0]
    return d


def null_feel_sentence_after_clause(rng):
    d = _pair(rng)
    _feel(rng, d)["answer"] += " It may pass."
    return d


def null_feel_text_all_turns(rng):             # the bounded organ also rewrites the sentence before its clause everywhere
    d = _pair(rng)
    for arms in d["on"].values():
        for sh in arms.values():
            for l in K.feel_labels(sh):
                a = sh["responses"][l]["answer"]
                sh["responses"][l]["answer"] = a.replace("functional read", "functional reading", 1)
    return d


def null_pair_missing_off(rng):
    d = _pair(rng)
    s, a = K.pick(rng, d["off"])
    del d["off"][s][a]
    return d


def null_pair_missing_on(rng):
    d = _pair(rng)
    s, a = K.pick(rng, d["on"])
    del d["on"][s][a]
    return d


def null_turn_missing_on(rng):
    d = _pair(rng)
    s, a = K.pick(rng, d["on"])
    del d["on"][s][a]["responses"][K.choice(rng, list(d["on"][s][a]["responses"]))]
    return d


def null_errored_turn_off(rng):
    d = _pair(rng)
    s, a = K.pick(rng, d["off"])
    d["off"][s][a]["responses"][K.choice(rng, list(d["off"][s][a]["responses"]))] = {"_error": "RuntimeError: x"}
    return d


def null_every_nonfeel_turn_changed(rng):
    d = _pair(rng)
    for arms in d["on"].values():
        for sh in arms.values():
            for l in K.expose_labels(sh):
                sh["responses"][l]["answer"] += " (on)"
    return d


def random_on_tree(rng):                       # an unrelated ON brain against the OFF one
    on, _ = K.draw_pair(rng)
    _, off = K.draw_pair(rng)
    return {"on": on, "off": off}
