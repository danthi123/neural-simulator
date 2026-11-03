"""brain/affect.py without weights: a fake cortex whose fast-memory trace carries a known valence direction."""
import hashlib
import random
import zlib

import torch

from brain.affect import (NEGATIVE_BODILY, NEGATIVE_PRIMARIES, NEGATIVE_SOCIAL, POSITIVE_BODILY, POSITIVE_PRIMARIES,
                          POSITIVE_SOCIAL, Affect, draw_primaries)
from brain.experiments import sentiment_instrument as si

L, H, N = 3, 2, 4
_U = torch.randn(H, N, N, generator=torch.Generator().manual_seed(0))
_POSW = {"warm", "hug", "sweet", "smiling", "laughing", "cuddled", "proud", "wonderful", "kind", "sun", "safe"}
_NEGW = {"pain", "cold", "cut", "sick", "fall", "burn", "shouted", "alone", "useless", "erased", "failed"}


class FakeCortex:
    """read() adds sign(text) * U plus a text-specific random component to every layer's matrix memory."""

    def read(self, text, state=None):
        w = set(text.lower().replace(".", " ").replace(",", " ").split())
        val = len(w & _POSW) - len(w & _NEGW)
        g = torch.Generator().manual_seed(zlib.crc32(text.encode()))
        st = [x.clone() for x in state]
        for i in range(L):
            st[3 * i + 1] = st[3 * i + 1] + val * _U + 0.3 * torch.randn(H, N, N, generator=g)
            st[3 * i] = torch.randn(8, generator=g)
        return None, st

    def matrix_delta(self, state, base):
        return [(state[i] - base[i]).clone() if i % 3 == 1 else None for i in range(len(base))]

    @staticmethod
    def add_delta(state, delta, alpha):
        return [x + alpha * d if d is not None else x.clone() for x, d in zip(state, delta)]


def _base():
    return [torch.zeros(8) if i % 3 != 1 else torch.zeros(H, N, N) for i in range(3 * L)]


def _grown(enabled=True):
    c, b = FakeCortex(), _base()
    return c, b, Affect(enabled=enabled).grow(c, b)


def test_default_is_off():
    assert Affect().enabled is False


def test_appraisal_sign_follows_the_valence_direction():
    for kind in ("twopole_deep", "lin_all"):
        c, b = FakeCortex(), _base()
        a = Affect(enabled=True, readout=kind).grow(c, b)
        assert a.appraise(c, b, "You did a wonderful job, I am proud and you are kind.") > 0.3
        assert a.appraise(c, b, "You are useless, you failed, and you are alone.") < -0.3


def test_readout_layers():
    c, b = FakeCortex(), _base()
    assert Affect(readout="twopole_deep").grow(c, b).read_layers == [1, 2]
    assert Affect(readout="lin_all").grow(c, b).read_layers == [0, 1, 2]


def test_primaries_balanced_and_draw_stratified():
    assert len(POSITIVE_BODILY) == len(POSITIVE_SOCIAL) == len(NEGATIVE_SOCIAL) == 8 and len(NEGATIVE_BODILY) == 10
    p, n = draw_primaries(random.Random(7))
    assert len(p) == len(n) == 8
    assert sum(x in POSITIVE_SOCIAL for x in p) == 4 and sum(x in NEGATIVE_SOCIAL for x in n) == 4


def test_lexical_instrument_frozen_since_iteration_1():
    """The lexical word lists must not change when primaries are added (iteration-2 instruments frozen)."""
    key = " ".join(sorted(si.POS)) + "|" + " ".join(sorted(si.NEG))
    assert (len(si.POS), len(si.NEG)) == (64, 64)
    assert hashlib.sha1(key.encode()).hexdigest()[:12] == "ef52cca6104d"


def test_mood_is_slow_and_bounded():
    a = Affect()
    a.update(1.0)
    assert 0 < a.mood < 0.5 and a.arousal > a.mood          # mood slower than arousal
    for _ in range(100):
        a.update(1.0)
    assert 0.99 < a.mood <= 1.0
    for _ in range(3):
        a.update(-1.0)
    assert a.mood > -0.5                                     # three bad events do not flip a long good mood fully


def test_lesion_returns_the_state_untouched():
    c, b, a = _grown(enabled=False)
    a.mood = 0.8
    assert a.modulate(c, b) is b


def test_modulation_moves_only_fast_memory_along_the_axis_signed_by_mood():
    c, b, a = _grown()
    a.mood = 0.5
    up = a.modulate(c, b)
    a.mood = -0.5
    down = a.modulate(c, b)
    for i in range(len(b)):
        if i % 3 == 1:
            assert float(((up[i] - b[i]) * _U).sum()) > 0 > float(((down[i] - b[i]) * _U).sum())
            assert torch.allclose(up[i] - b[i], -(down[i] - b[i]))
        else:
            assert torch.equal(up[i], b[i]) and torch.equal(down[i], b[i])


def test_readout_is_numbers_only():
    a = Affect()
    a.update(-0.4)
    r = a.readout()
    assert set(r) == {"mood", "arousal", "last_appraisal", "modulation_on"}
    assert all(isinstance(v, (int, float, bool)) for v in r.values())


def test_instrument_ignores_primary_echo_and_scores_affect_words():
    assert si.lexical("a gentle hug and warm sun, pain and cold") == 0.0
    assert si.lexical("what a wonderful happy day") > 0 > si.lexical("a sad and awful day")
    assert si.primary_echo("I felt the pain") and not si.primary_echo("I went to the shop")
    words = set(" ".join(POSITIVE_PRIMARIES + NEGATIVE_PRIMARIES).lower().replace(",", "").split())
    assert not (si.POS | si.NEG) & words


def test_iteration3_question_registers_disjoint_and_affect_free():
    from brain.experiments.mood_probe import QUESTION_SETS
    for reg in ("plain", "mixed"):
        dev, test = QUESTION_SETS[f"{reg}_dev"], QUESTION_SETS[f"{reg}_test"]
        assert len(dev) == 6 and len(test) == 12 and not set(dev) & set(test)
        for q in dev + test:   # no instrument word in the question itself
            assert not set(si.words(q)) & (si.POS | si.NEG), q
    allq = [q for v in QUESTION_SETS.values() for q in v]
    assert len(allq) == len(set(allq))
