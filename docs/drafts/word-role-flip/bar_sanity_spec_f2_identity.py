"""bar-sanity spec: word-role-flip F2 (the default IS the capability) -- REGISTRATION_TEXT.md §2 F2.

THE BAR. At the flip revision F, `default` (no flags set) equals `on:both` (both word-role flags `=1` explicitly)
on EVERY script turn of iw1 and iw2, byte-identical apart from the runners' existing volatile-key drops: the
registered identity `--compare` (expect=same) returns IDENTICAL for both runners. The record already exists twice
at ce2e028 (research/findings/raw/_iw1/identity/ce2e028/compare_default_onwr.json and the iw2 twin); F2 re-measures
it at F. The prereg records it as measured, pre-gate dev-seed identity.

THE DECISION IS THE REAL ONE. The spec calls the runners' own registered scorers: iw1_default_identity.compare()
and iw2_default_identity.compare() with expect="same". decide() passes iff BOTH return ok (IDENTICAL). The
compare functions own every precondition (same seed, script length, every status 200, every turn present, corpus +
same LTM bundle; iw2 additionally same script sha and the determinism env) -- an UNDEFINED precondition reads as
not passing, as the registered verdict applies it.

BRAINS (a synthetic arm pair per runner: same seed, same script, same corpus/bundle).
  perfect            every turn's sha_canon and reply fields identical; volatile record keys (wall time, pid-like
                     noise) differ, as they do between two real arm runs.
  perfect_noisy      identical canon plus extra volatile record noise and a different recorded git_sha (compare
                     reports git_shas but does not precondition on them: both arms run at the same revision F).
  null_one_turn      exactly one turn's canonical payload differs (answer + sha_canon) in one runner.
  null_seed_mismatch the arms were built at different seeds -> compare returns UNDEFINED (not passing).
  null_corpus_drop   one arm records corpus_present=False -> degraded-brain guard -> UNDEFINED.
  random             per draw, identical or k in 1..n_turns turns differ, each with probability 1/2.
  info_vacuous_pair  identical except one turn flips a NON-canonical volatile key only -- reports that canon is
                     what the identity bar reads (turns whose canon is equal cannot fail it).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners import iw1_default_identity as IW1  # noqa: E402
from research.runners import iw2_default_identity as IW2  # noqa: E402

NAME = "word-role-flip F2 (default == on:both on every script turn of iw1 and iw2)"
DESCRIPTION = ("Two synthetic arm records per runner -> the registered iw1/iw2 compare(expect=same); PASS iff both "
               "return IDENTICAL (every precondition of the real compare holds).")
SCORER_FILES = ["research/runners/iw1_default_identity.py", "research/runners/iw2_default_identity.py",
                "docs/drafts/word-role-flip/bar_sanity_spec_f2_identity.py"]

_N = 12
_SEED = 7
_MSGS = ["what do you usually use?", "the cat eats fish", "Sally puts the marble in the basket.", "hi",
         "the ball precedes the worm", "who are you", "the dog and the cat walked in", "open the pod bay doors",
         "what might the robin eat", "the wolf hunts the deer", "define curiosity", "goodbye"]
_ANSWERS = ["I use spikes.", "the cat eats the fish", "the marble is in the basket", "hello.",
            "the ball precedes the worm", "I am the brain.", "both walked in", "I cannot do that.",
            "the robin eats the worm", "the wolf hunts the deer", "curiosity is an appetite for the unknown.",
            "goodbye."]
_VOLATILE = ("wall_s", "elapsed_s", "import_s", "pid", "hostname_note")


def _turn(i, seed):
    return {"message": _MSGS[i], "status": 200,
            "sha_canon": "canon-%02d-%d" % (i, _SEED), "sha_raw": "raw-%02d" % i,
            "answer": _ANSWERS[i], "abstained": False, "recalled_svo": ["brain", "use", "spikes"]}


def _rec(arm, rng, diff_turns=(), seed=_SEED, corpus=True, volatile=True):
    turns = [_turn(i, seed) for i in range(_N)]
    for i in diff_turns:
        turns[i]["sha_canon"] = "canon-%02d-%d-DIFF" % (i, seed)
        turns[i]["answer"] = "a different reply than the intact arm's."
    rec = {"arm": arm, "seed": seed, "n_turns": _N, "turns": turns,
           "corpus_present": corpus, "ltm_bundle": "shipped-ltm-v3", "sim_backend": "numpy",
           "git_sha": "f" * 40, "script_sha": "script-%02d" % _N, "threads": dict(IW2.DETERMINISM_ENV),
           "session_sha_canon": "session-%s" % arm}
    if volatile:
        rec["wall_s"] = float(rng.uniform(1.0, 99.0))
        rec["import_s"] = float(rng.uniform(0.1, 5.0))
    return rec


def _pair(rng, diff_turns=(), seed=_SEED, corpus=True, other=None):
    a, b = _rec("default", rng, (), seed, corpus), _rec("on:both", rng, diff_turns, seed, corpus)
    if other:
        other(a, b, rng)
    return {"iw1": {"a": a, "b": b}, "iw2": {"a": dict(a), "b": dict(b)}}


def decide(data):
    """The registered identity check: both runners' own compare() must return IDENTICAL."""
    ok = True
    for mod, pair in ((IW1, data["iw1"]), (IW2, data["iw2"])):
        v = mod.compare(pair["a"], pair["b"], "same")
        ok = ok and bool(v["ok"])
    return bool(ok)


def perfect(rng):
    return _pair(rng)


def perfect_noisy(rng):
    def noise(a, b, r):
        b["git_sha"] = "a" * 40                       # reported, not preconditioned: same revision, other record
        b["wall_s"] = float(r.uniform(1.0, 99.0))
        b["extra_record_key"] = {"dropped_by": "canon"}
    return _pair(rng, other=noise)


def null_one_turn(rng):
    k = int(rng.integers(_N))
    return _pair(rng, diff_turns=(k,))


def null_seed_mismatch(rng):
    d = _pair(rng)
    other_seed = int(rng.integers(100, 200))
    for runner in d.values():
        runner["b"]["seed"] = other_seed            # arms built at different seeds -> UNDEFINED, not passing
    return d


def null_corpus_drop(rng):
    d = _pair(rng)
    d["iw2"]["b"]["corpus_present"] = False
    d["iw1"]["b"]["corpus_present"] = False
    return d


def random(rng):
    """A random responder: each arm draws its own reply per turn independently, so the two arms are
    identical only by luck (~0 for 12 turns) -- the identity bar must not pass a random brain."""
    a = _rec("default", rng)
    b = _rec("on:both", rng)
    for t in b["turns"]:
        if rng.random() < 2.0 / 3.0:                   # a random reply, otherwise (rarely) the same one
            t["sha_canon"] = "canon-rand-%d" % int(rng.integers(1 << 30))
            t["answer"] = "random reply %d." % int(rng.integers(1 << 20))
    d = {"iw1": {"a": a, "b": b}, "iw2": {"a": dict(a), "b": dict(b)}}
    return d


def info_vacuous_pair(rng):
    def noise(a, b, r):
        b["turns"][0]["wall_s"] = float(r.uniform(1.0, 9.0))   # a turn-level volatile key: canon equal -> ok
    return _pair(rng, other=noise)
