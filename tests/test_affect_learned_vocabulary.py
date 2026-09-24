"""BRAIN_AFFECT_LEARNED_VOCAB (default-off): data-level checks of the learned affect vocabulary on a tiny synthetic
heard stream (numpy backend, small circuit).

Off: appraise_text is byte-identical to the pinned pre-change module (1d5766620) on neutral and affective sentences.
Learning: a word heard only beside a negative seed learns to read negative; a word heard equally often in every
context reads 0; the learned_edge lesion returns every learned word to 0.0; under the flag, the lesion's appraisal
equals the flag-off appraisal exactly.
"""
import os
import subprocess
import types

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

PINNED = "1d5766620"
SENTENCES = ["What does the cat eat?", "I am sad and lonely today.", "What a wonderful, happy day!",
             "The treaty was signed in 1648.", "I feel sorrow and gloom.", "", "war war war"]


def _clear(monkeypatch):
    for k in ("BRAIN_AFFECT_LEARNED_VOCAB", "BRAIN_AFFECT_LEARNED_VOCAB_LESION", "BRAIN_AFFECT_LEARNED_VOCAB_PATH"):
        monkeypatch.delenv(k, raising=False)


def test_off_byte_identical_to_pinned(monkeypatch):
    _clear(monkeypatch)
    from research.runners import affect_production_organ as AO
    src = subprocess.run(["git", "show", "%s:research/runners/affect_production_organ.py" % PINNED],
                         capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType("_pinned_affect_production_organ")
    mod.__file__ = AO.__file__
    exec(compile(src, "pinned_affect_production_organ.py", "exec"), mod.__dict__)
    for t in SENTENCES:
        assert AO.appraise_text(t) == mod.appraise_text(t)


@pytest.fixture(autouse=True)
def _mechanism_constants(monkeypatch):
    """These tests check the mechanism on a tiny synthetic stream, not the production operating point: no
    strong-affect margin on the read (the production V_MIN is calibrated for the real stream)."""
    from research.runners import affect_learned_vocabulary as A
    monkeypatch.setattr(A, "V_MIN", 0.0)


def _tiny(tmp_path):
    from research.runners import affect_learned_vocabulary as A
    vocab = ["sad", "happy", "gloom", "table", "sunny", "chair"]
    innate = {"sad": -0.725, "happy": 0.875}
    vid = {w: i for i, w in enumerate(vocab)}
    rng = np.random.default_rng(0)
    rows = []
    for k in range(1200):
        r = rng.random()
        if r < 0.3:
            ws = ["sad", "gloom", "table"]
        elif r < 0.6:
            ws = ["happy", "sunny", "table"]
        else:
            ws = ["chair", "table"]
        row = np.full(A.CHUNK, -1, dtype=np.int32)
        row[: len(ws)] = [vid[w] for w in ws]
        rows.append(row)
    lav = A.LearnedAffectVocabulary(3, vocab, innate, n_replicas=1, warmup=200, g=500.0, u_dep=0.0)
    lav.train(np.stack(rows))
    p = tmp_path / "w.npz"
    lav.save(str(p))
    return A, lav, str(p)


def test_no_us_presentation_increment_is_exactly_zero_when_simulated(tmp_path):
    """The execution identity the trainer relies on: with no innate US afferent heard, the CS-alone and CS+US trials
    are the same deterministic run, so the increment is exactly 0.0 (also with non-zero learned synapses)."""
    A, lav, p = _tiny(tmp_path)
    assert np.abs(lav.u).max() > 0
    lav.simulate_all = True
    row = np.full(A.CHUNK, -1, dtype=np.int32)
    for ws in (["gloom", "table"], ["sunny", "chair", "table"], ["table"]):
        row[:] = -1
        row[: len(ws)] = [lav.vid[w] for w in ws]
        inc = lav.present_and_learn(row)
        assert np.array_equal(inc, np.zeros_like(inc))
    row[:3] = [lav.vid["sad"], lav.vid["gloom"], lav.vid["table"]]
    assert lav.present_and_learn(row)[0, 1] > 0          # a heard negative seed does evoke a V- increment


def test_learned_word_reads_signed_and_uniform_word_reads_zero(tmp_path):
    A, lav, p = _tiny(tmp_path)
    rd = A.load_reader(p, 3)
    assert rd.read("gloom") < 0 < rd.read("sunny")
    assert rd.read("table") == 0.0 and rd.read("chair") == 0.0
    assert rd.read("never_heard") == 0.0
    rd.set_lesion("learned_edge")
    assert rd.read("gloom") == 0.0 and rd.read("sunny") == 0.0


def test_flag_on_hears_learned_word_and_lesion_equals_off(tmp_path, monkeypatch):
    A, lav, p = _tiny(tmp_path)
    _clear(monkeypatch)
    from research.runners import affect_production_organ as AO
    off = [AO.appraise_text(t) for t in SENTENCES]
    monkeypatch.setenv("BRAIN_AFFECT_LEARNED_VOCAB", "1")
    monkeypatch.setenv("BRAIN_AFFECT_LEARNED_VOCAB_PATH", p)
    monkeypatch.setenv("BRAIN_CHAT_SEED", "3")
    A._READER.clear()
    on = AO.appraise_text("I feel gloom.")
    assert on["valence"] < 0 and on["learned_words"] == ["gloom"]
    monkeypatch.setenv("BRAIN_AFFECT_LEARNED_VOCAB_LESION", "1")
    les = [AO.appraise_text(t) for t in SENTENCES]
    core = lambda d: {k: d[k] for k in ("valence", "arousal", "n_hits", "words")}  # noqa: E731
    assert [core(d) for d in les] == [core(d) for d in off]
    A._READER.clear()
