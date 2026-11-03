"""brain/affect.py without weights: a fake cortex whose fast-memory trace carries a known valence direction."""
import zlib

import torch

from brain.affect import NEGATIVE_PRIMARIES, POSITIVE_PRIMARIES, Affect
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
    c, b, a = _grown()
    assert a.appraise(c, b, "You did a wonderful job, I am proud and you are kind.") > 0.3
    assert a.appraise(c, b, "You are useless, you failed, and you are alone.") < -0.3


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
