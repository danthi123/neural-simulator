"""bar-sanity spec: emo-bounded-flip B4, "every ON feel read has bounded_readback True and oor_count_session 0".
decide = `research.runners.emo_flip_score.score_b4` (the scorer itself) over six gate seeds x the five emo arms, flag ON, read
as PASS. A shard, a feel turn or a key that is not there reads UNDEFINED, which is NOT a pass. The prereg's scoring text adds
bounded_flag True and bounded_env "1" to the two fields the ruling names; the scorer checks all four.

perfect  = the guard engaged and held on every feel read of every shard (realistic production reply surfaces, spike counts);
nulls    = the flag did not reach the process (everywhere, in one shard, on one read), the bridge did not read it back, the raw
           env is not "1", an out-of-range excursion in one turn / a whole shard, a pre-merge shard without the keys (one / all),
           a shard / a feel turn / the repeat arm missing, an errored feel turn, a feel read without the block, a non-count
           oor_count_session, and a random brain.
Needs research/runners/emo_flip_score.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bar_sanity_emo_flip_common as K  # noqa: E402

S, np = K.S, K.np

NAME = "emo-bounded-flip B4 (engagement keys on every ON feel read)"
DESCRIPTION = ("emo_flip_score.score_b4 reads PASS over the six gate seeds x five arms, flag ON; perfect = bounded_flag True, "
               "bounded_env '1', bounded_readback True, oor_count_session 0 on every feel_* read; nulls = the guard not engaged "
               "(all / one shard / one read), readback False, env not '1', an out-of-range excursion (one read / a shard), "
               "pre-merge shards lacking the keys, a missing shard / feel turn / repeat arm, an errored feel turn, no emotion "
               "block, a non-count oor_count_session, and a random brain.")
SCORER_FILES = ["research/runners/emo_flip_score.py", "research/runners/stage2_exit_gate.py",
                "docs/drafts/emo-bounded-flip/bar_sanity_emo_flip_common.py", "docs/drafts/stage2-exit/bar_sanity_s2_common.py"]


def decide(data):
    return S.score_b4(S.mem_loader(data))["status"] == S.PASS


def _on(rng, **kw):
    return K.make_tree(int(rng.integers(0, 2 ** 31 - 1)), True, **kw)


def _reads(shard):
    return [shard["responses"][l]["emotion_category"] for l in K.feel_labels(shard)]


def _one_read(rng, d):
    s, a = K.pick(rng, d)
    l = K.choice(rng, K.feel_labels(d[s][a]))
    return d[s][a]["responses"][l]


# ── perfect brains ──────────────────────────────────────────────────────────────────────────────────────────────────
def perfect(rng):
    return _on(rng)


def perfect_nested(rng):                       # other blocks carry the keys too; B4 reads only the feel reads' own block
    return _on(rng, nested_keys=True)


def perfect_extra_fields(rng):                 # a richer emotion_category record (more fields than the 10-26 shards)
    d = _on(rng)
    for arms in d.values():
        for sh in arms.values():
            for ec in _reads(sh):
                ec["oor_count"] = 0
                ec["extra_diag"] = {"v_min": -78.2, "v_max": 31.0}
    return d


# ── nulls: a defect at ONE random place unless stated ─────────────────────────────────────────────────────────────
def null_flag_not_engaged_everywhere(rng):     # the env var never reached the pool process: an OFF run labelled ON
    return K.make_tree(int(rng.integers(0, 2 ** 31 - 1)), False)


def null_flag_false_one_shard(rng):
    d = _on(rng)
    s, a = K.pick(rng, d)
    for ec in _reads(d[s][a]):
        ec.update({"bounded_flag": False, "bounded_env": None, "bounded_readback": False})
    return d


def null_flag_false_one_read(rng):
    d = _on(rng)
    _one_read(rng, d)["emotion_category"]["bounded_flag"] = False
    return d


def null_readback_false_one_read(rng):         # the flag resolved True but the bridge's own config does not carry it
    d = _on(rng)
    _one_read(rng, d)["emotion_category"]["bounded_readback"] = False
    return d


def null_env_zero_one_read(rng):
    d = _on(rng)
    _one_read(rng, d)["emotion_category"]["bounded_env"] = "0"
    return d


def null_env_unset_one_read(rng):              # the post-flip default also reads True; the gate line must still SET it
    d = _on(rng)
    _one_read(rng, d)["emotion_category"]["bounded_env"] = None
    return d


def null_oor_session_one_read(rng):
    d = _on(rng)
    _one_read(rng, d)["emotion_category"]["oor_count_session"] = int(rng.integers(1, 20))
    return d


def null_oor_session_one_shard(rng):
    d = _on(rng)
    s, a = K.pick(rng, d)
    for ec in _reads(d[s][a]):
        ec["oor_count_session"] = int(rng.integers(1, 6))
    return d


def null_oor_session_last_turn_only(rng):     # the excursion shows only at the last read of the session (the cumulative count)
    d = _on(rng)
    s, a = K.pick(rng, d)
    d[s][a]["responses"]["feel_sadness"]["emotion_category"]["oor_count_session"] = 1
    return d


def null_keys_absent_one_shard(rng):           # a shard from before the engagement keys existed
    d = _on(rng)
    s, a = K.pick(rng, d)
    for ec in _reads(d[s][a]):
        for k in S.ENGAGEMENT_KEYS:
            ec.pop(k)
    return d


def null_keys_absent_everywhere(rng):
    d = _on(rng)
    for arms in d.values():
        for sh in arms.values():
            for ec in _reads(sh):
                for k in S.ENGAGEMENT_KEYS:
                    ec.pop(k)
    return d


def null_one_key_absent_one_read(rng):         # only oor_count_session is missing: cannot show the excursion count was 0
    d = _on(rng)
    _one_read(rng, d)["emotion_category"].pop("oor_count_session")
    return d


def null_shard_missing(rng):
    d = _on(rng)
    s, a = K.pick(rng, d)
    del d[s][a]
    return d


def null_repeat_arm_missing_everywhere(rng):   # the prereg says all five arms incl .rep0
    d = _on(rng)
    for arms in d.values():
        del arms["lesion_affect.rep0"]
    return d


def null_feel_turn_missing(rng):
    d = _on(rng)
    s, a = K.pick(rng, d)
    del d[s][a]["responses"][K.choice(rng, K.feel_labels(d[s][a]))]
    return d


def null_feel_turn_errored(rng):
    d = _on(rng)
    s, a = K.pick(rng, d)
    d[s][a]["responses"][K.choice(rng, K.feel_labels(d[s][a]))] = {"_error": "RuntimeError: organ read failed"}
    return d


def null_no_emotion_block_on_a_read(rng):
    d = _on(rng)
    _one_read(rng, d).pop("emotion_category")
    return d


def null_oor_session_not_a_count(rng):
    d = _on(rng)
    _one_read(rng, d)["emotion_category"]["oor_count_session"] = ["0", "zero", None, False, True, 0.0][int(rng.integers(0, 6))]
    return d


def random_engagement(rng):                    # every engagement field a coin flip
    d = _on(rng)
    for arms in d.values():
        for sh in arms.values():
            for ec in _reads(sh):
                ec["bounded_flag"] = bool(rng.random() < 0.5)
                ec["bounded_readback"] = bool(rng.random() < 0.5)
                ec["bounded_env"] = ["1", "0", None][int(rng.integers(0, 3))]
                ec["oor_count_session"] = int(rng.integers(0, 3))
    return d
