# Capability: Lifelong learning
# Record: brain/research/2026-11-21-sleep-language-keep-prereg.md
# Expected-fail: none (mechanics only: bank sampling, RNG, subsets, persistence, off-switch identity; effects are measured on lives)
"""Brain(keep_bank="prior") (K-P) on the stub cortex with a sampling pipe and a small differentiable fake adapter forward,
through the REAL sleep loop (brain/sleep.py), so longer keep texts and the keep reference are exercised."""
import importlib.util
import json
import os
import subprocess
import sys

import pytest
import torch

import brain.brain as bb
from brain import clock
from brain.brain import Brain
from brain.eval.one_night import one_night
from brain.plastic_cortex import PlasticCortex
from brain.tests.test_brain_key_switch import KW, ROOT, StubCortex, _Pipe
from brain.tests.test_use_scale import T0, _fake_plastic

BASE = "2aa9e736a"   # the head before K-P: the off switch must equal it
V = _Pipe.V


class SampPipe(_Pipe):
    def sample_logits(self, logits, temperature=1.0, top_p=0.3):
        return int(torch.multinomial(torch.softmax(logits.float() / max(temperature, 1e-6), -1), 1))


class SampStub(StubCortex):
    def __init__(self):
        super().__init__()
        self.pipe = SampPipe()


class DiffPlastic(PlasticCortex):
    """Real adapter parameters; logits depend on them (so the real loop gets gradients) when `enabled`."""
    def forward(self, ids, base):
        g = torch.Generator().manual_seed(1)
        P, Q = torch.randn(V, 4, generator=g), torch.randn(6, V, generator=g)
        t = torch.arange(len(ids)).float()[:, None] * 0.37 + torch.arange(V).float()[None, :] * 0.07
        out = torch.sin(t) * 2.0
        if self.enabled:
            out = out + P[ids] @ (self.A_k["0"] @ self.B_k["0"]) @ Q
        return out, None


def _plastic_of(self):
    if self.cortex.plastic is None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.sleep_seed)
            p = _fake_plastic(self.raw_cortex)
            p.__class__ = DiffPlastic
        self.cortex.plastic = p
    return self.cortex.plastic


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(Brain, "_plastic", _plastic_of)
    monkeypatch.setattr(bb, "KEEP_BANK_N", 24)
    monkeypatch.setattr(bb, "KEEP_BANK_NIGHT", 5)
    ticks = iter(range(10 ** 6))
    clock.set(lambda: T0 + next(ticks))
    yield
    clock.set(None)


def _brain(home, cls=Brain, cortex=None, **kw):
    b = cls(home, cortex=cortex or SampStub(), **KW, **kw)
    for t in ("My cat is called Pixel.", "I drive a red Volvo.", "My sister's name is Marguerite."):
        b.hear(t)
    b.cortex.consolidated_on = False
    return b


def _adapters(home):
    return torch.load(os.path.join(home, "cortex_adapters.pt"), weights_only=False)["adapters"]


def test_off_is_byte_identical_to_base_head_and_writes_no_bank(tmp_path, small):
    try:
        src = subprocess.run(["git", "-C", ROOT, "show", f"{BASE}:brain/brain.py"], capture_output=True, text=True,
                             check=True).stdout
    except Exception:
        pytest.skip("base commit not available")
    path = str(tmp_path / "_ref_brain_for_test.py")
    with open(path, "w") as fh:
        fh.write(src)
    try:
        spec = importlib.util.spec_from_file_location("brain._ref_brain_for_test", path)
        ref = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ref
        spec.loader.exec_module(ref)
    finally:
        os.remove(path)
    ref.Brain._plastic = _plastic_of
    try:
        homes = {}
        for name, cls, kw in (("ref", ref.Brain, {}), ("new", Brain, {}), ("none", Brain, {"keep_bank": None})):
            clock.set(lambda: T0)
            torch.manual_seed(3)
            b = _brain(str(tmp_path / name), cls=cls, **kw)
            for _ in range(3):
                b.sleep()
            b.save()
            homes[name] = str(tmp_path / name)
        a0 = _adapters(homes["ref"])
        for n in ("new", "none"):
            an = _adapters(homes[n])
            assert a0.keys() == an.keys() and all(torch.equal(a0[k], an[k]) for k in a0)
            assert open(os.path.join(homes["ref"], "meta.json")).read() == open(os.path.join(homes[n], "meta.json")).read()
            assert not os.path.exists(os.path.join(homes[n], "keep_bank.json"))
            assert "keep_bank" not in json.load(open(os.path.join(homes[n], "meta.json")))
        assert any(float(v.abs().sum()) > 0 for k, v in a0.items() if "B" in k)   # the nights changed the adapters: not vacuous
    finally:
        sys.modules.pop("brain._ref_brain_for_test", None)


def test_bank_deterministic_stored_and_reloaded(tmp_path, small):
    a, b = _brain(str(tmp_path / "a")), _brain(str(tmp_path / "b"))
    ta, tb = a._keep_bank(), b._keep_bank()
    assert len(ta) == 24 and ta == tb
    assert all(t.startswith("User:") and t.endswith("\n\n") for t in ta) and len(set(ta)) > 1
    f = tmp_path / "a" / "keep_bank.json"
    assert json.load(open(f))["texts"] == ta
    # reload: a new Brain on the home reads the file, never resamples
    blob = json.load(open(f))
    blob["texts"] = ["User: hand-edited.\n\n"] * 3
    json.dump(blob, open(f, "w"))
    c = Brain(str(tmp_path / "a"), cortex=SampStub(), **KW, keep_bank="prior")
    assert c._keep_bank() == blob["texts"]
    # another seed -> another bank
    d = _brain(str(tmp_path / "d"), sleep_seed=7)
    assert d._keep_bank() != ta


def test_bank_sampling_leaves_global_rng_and_replay_seed_untouched(tmp_path, small, monkeypatch):
    seen = []
    real = bb.replay_into_cortex
    monkeypatch.setattr(bb, "replay_into_cortex", lambda p, base, eps, generic, **k: seen.append((list(eps), k["seed"], list(generic)))
                        or real(p, base, eps, generic, **k))
    a, b = _brain(str(tmp_path / "a")), _brain(str(tmp_path / "b"), keep_bank="prior")
    torch.manual_seed(5)
    s0 = torch.get_rng_state()
    b._keep_bank()
    assert torch.equal(s0, torch.get_rng_state())
    a.sleep(), b.sleep()
    assert [t for t, _ in seen[0][0]] == [t for t, _ in seen[1][0]] and seen[0][1] == seen[1][1]   # same replay episodes and seed
    assert seen[0][2] == bb.ORDINARY and seen[1][2] != bb.ORDINARY and len(seen[1][2]) == 5


def test_subset_fresh_each_night_and_engagement_record(tmp_path, small):
    b = _brain(str(tmp_path / "h"), keep_bank="prior")
    recs = []
    for _ in range(3):
        b.sleep()
        recs.append(dict(b.last_keep_record))
        b.save()
    assert [r["bank_size"] for r in recs] == [24] * 3 and [r["subset_n"] for r in recs] == [5] * 3
    assert len({r["subset_hash"] for r in recs}) == 3
    assert all(r["mean_keep_kl"] is not None and r["mean_keep_kl"] >= 0 for r in recs)
    assert recs[1]["mean_keep_kl"] > 0 and recs[2]["mean_keep_kl"] > 0     # the keep term is active from night 2
    bank = set(b._keep_bank())
    for n in range(3):
        sub = __import__("random").Random(b.sleep_seed + n + bb.KEEP_BANK_SEED_OFFSET).sample(b._keep_bank(), 5)
        assert set(sub) <= bank
    assert json.load(open(tmp_path / "h" / "meta.json"))["keep_bank"] == "prior"


def test_long_keep_texts_through_the_real_loop():
    """sleep() with 64+ token keep texts of different lengths: every text has its reference, no shape bug."""
    from brain.sleep import sleep
    stub = SampStub()
    p = _fake_plastic(stub)
    p.__class__ = DiffPlastic
    base = [torch.zeros(1)]
    gen = ["User:" + " word" * n + "\n\n" for n in (5, 40, 70, 90)]
    eps = [("User: My cat is called Pixel.\n\n", 1.0)]
    losses = sleep(p, base, eps, gen, steps=60, lr=1e-2, kl_weight=3.0, seed=1)
    assert len(losses) == 60 and all(map(lambda r: r[0] == r[0] and r[1] == r[1], losses))
    assert max(k for _, k in losses) > 0


def test_meta_persistence_resume_refusal_and_validation(tmp_path, small):
    home = str(tmp_path / "h")
    b = _brain(home, keep_bank="prior")
    b.sleep(), b.save()
    b.release()
    b2 = Brain(home, cortex=SampStub(), **KW)                       # None adopts what the home was saved with
    assert b2._keep_bank_mode == "prior" and "keep_bank" not in b2.meta
    b2.release()
    with pytest.raises(ValueError):
        Brain(str(tmp_path / "x"), cortex=SampStub(), **KW, keep_bank="nope")
    # switching it ON in a home that already slept without it is refused
    plain = _brain(str(tmp_path / "p"))
    plain.sleep(), plain.save()
    plain.release()
    with pytest.raises(ValueError, match="keep_bank"):
        Brain(str(tmp_path / "p"), cortex=SampStub(), **KW, keep_bank="prior")
    # a fresh (never slept) home may be switched on
    fresh = _brain(str(tmp_path / "f"))
    fresh.save()
    fresh.release()
    assert Brain(str(tmp_path / "f"), cortex=SampStub(), **KW, keep_bank="prior")._keep_bank_mode == "prior"


def test_one_night_helper(tmp_path, small):
    src = str(tmp_path / "snap")
    b = _brain(src)
    b.sleep(), b.save()
    b.release()
    before = open(os.path.join(src, "meta.json")).read()
    kw = dict(KW)
    out_b = one_night(src, str(tmp_path / "B"), None, cortex=SampStub(), sleep_seed=0, **kw)
    out_k = one_night(src, str(tmp_path / "K"), "prior", cortex=SampStub(), sleep_seed=0, **kw)
    assert open(os.path.join(src, "meta.json")).read() == before and not os.path.exists(os.path.join(src, "keep_bank.json"))
    assert out_b["night"] == out_k["night"] == 2 and out_b["keep_record"] is None
    assert out_k["keep_record"]["bank_size"] == 24
    assert os.path.exists(tmp_path / "K" / "keep_bank.json") and not os.path.exists(tmp_path / "B" / "keep_bank.json")
    assert json.load(open(tmp_path / "K" / "meta.json"))["keep_bank"] == "prior"
    with pytest.raises(FileExistsError):
        one_night(src, str(tmp_path / "K"), "prior", cortex=SampStub(), **kw)


class NpPipe(_Pipe):
    """Samples like the real rwkv PIPELINE on CPU: with the global numpy RNG."""
    def sample_logits(self, logits, temperature=1.0, top_p=0.3):
        import numpy as np
        p = torch.softmax(logits.float() / max(temperature, 1e-6), -1).numpy().astype("float64")
        return int(np.random.choice(len(p), p=p / p.sum()))


class NpStub(StubCortex):
    def __init__(self):
        super().__init__()
        self.pipe = NpPipe()


def test_bank_numpy_sampler_deterministic_and_numpy_rng_untouched(tmp_path, small):
    """The rwkv pipeline draws from np.random on CPU: the bank must seed it and leave the global numpy RNG as it was."""
    import numpy as np
    a = _brain(str(tmp_path / "a"), cortex=NpStub())
    b = _brain(str(tmp_path / "b"), cortex=NpStub())
    np.random.seed(11)
    s0 = np.random.get_state()
    ta = a._keep_bank()
    s1 = np.random.get_state()
    assert s0[0] == s1[0] and np.array_equal(s0[1], s1[1]) and s0[2:] == s1[2:]
    np.random.seed(999)                      # a different global numpy state must not change the bank
    assert b._keep_bank() == ta and len(set(ta)) > 1


class ScriptPipe(_Pipe):
    """Emits a fixed script (crossing a blank line and an Assistant turn), recording every sampler call's settings."""
    SCRIPT = [ord(c) for c in (" Hi there.\n\nAssistant: Hello!\n\nUser: more text here and more and more and more words go on" + " " * 64)[:64]]

    def __init__(self, end_at=None):
        self.calls, self.i, self.end_at = [], 0, end_at

    def sample_logits(self, logits, temperature=1.0, top_p=0.3):
        self.calls.append((temperature, top_p))
        if self.end_at is not None and self.i >= self.end_at:
            self.i = 0
            return 0                                   # END_OF_TEXT
        tok = self.SCRIPT[self.i % len(self.SCRIPT)]
        self.i += 1
        return tok


class ScriptStub(StubCortex):
    def __init__(self, end_at=None):
        super().__init__()
        self.pipe = ScriptPipe(end_at)


def _script_sampler(monkeypatch, cx):
    """The bank draws with bb._sample_full (full softmax); script it through the pipe's recorder, top_p fixed at 1.0."""
    monkeypatch.setattr(bb, "_sample_full", lambda lg, temperature: cx.pipe.sample_logits(lg, temperature, 1.0))


def test_bank_is_plain_continuation_top_p_1_no_penalty_no_turn_stop(tmp_path, small, monkeypatch):
    monkeypatch.setattr(bb, "KEEP_BANK_N", 3)
    cx = ScriptStub()
    _script_sampler(monkeypatch, cx)
    spoke, piped = [], []
    b = _brain(str(tmp_path / "h"), cortex=cx)
    monkeypatch.setattr(type(cx), "speak", lambda *a, **k: spoke.append(1) or ("", None))
    real_pipe_sampler = cx.pipe.sample_logits
    monkeypatch.setattr(cx.pipe, "sample_logits", lambda *a, **k: (piped.append(a[2:] or k), real_pipe_sampler(*a, **k))[1])
    cx.pipe.calls.clear()
    cx.pipe.i = 0
    texts = b._keep_bank()
    assert not spoke                                    # speak (stops at turns, top_p 0.3, penalties) is not used
    assert all(c == (1.0,) for c in piped)              # only via _sample_full (scripted here), never rwkv's top_p path
    assert bb.KEEP_BANK_TOP_P == 1.0
    assert len(cx.pipe.calls) == 3 * bb.KEEP_BANK_TOKENS      # exactly KEEP_BANK_TOKENS samples per text
    assert all(c == (bb.KEEP_BANK_TEMP, 1.0) for c in cx.pipe.calls)
    for t in texts:                                     # crossed "\n\nAssistant:" and the next User turn
        assert "\n\nAssistant:" in t and "\n\nUser:" in t and t.startswith("User:") and t.endswith("\n\n")
        assert len(t) > len("User:") + bb.KEEP_BANK_TOKENS - 2
    blob = json.load(open(tmp_path / "h" / "keep_bank.json"))
    assert blob["top_p"] == 1.0 and blob["presence_penalty"] == 0.0 and blob["frequency_penalty"] == 0.0
    assert blob["stop_rule"] == "64 tokens or end of text" and blob["mean_tokens"] == bb.KEEP_BANK_TOKENS


def test_bank_stops_at_end_of_text_and_records_mean_length(tmp_path, small, monkeypatch):
    monkeypatch.setattr(bb, "KEEP_BANK_N", 4)
    cx = ScriptStub(end_at=10)
    _script_sampler(monkeypatch, cx)
    b = _brain(str(tmp_path / "h"), cortex=cx)
    b._keep_bank()
    blob = json.load(open(tmp_path / "h" / "keep_bank.json"))
    assert blob["mean_tokens"] == 10


def test_rwkv_top_p_1_is_greedy_so_the_bank_uses_a_full_softmax_sampler(tmp_path, small, monkeypatch):
    """Regression (review of A1): rwkv PIPELINE.sample_logits(top_p=1.0) on CPU is greedy over a 65536 vocabulary
    (the float32 cumsum never reaches 1.0). The bank must not depend on it: with the REAL rwkv sampler on the pipe and
    vocabulary-sized logits, the bank is still diverse, and _sample_full matches the softmax."""
    import numpy as np
    from rwkv.utils import PIPELINE
    g = torch.Generator().manual_seed(0)
    lg = torch.randn(65536, generator=g) * 3
    np.random.seed(0)
    draws = [PIPELINE.sample_logits(None, lg, temperature=1.0, top_p=1.0) for _ in range(50)]
    assert set(draws) == {int(torch.argmax(lg))}        # the trap: the package sampler collapses to greedy
    np.random.seed(0)
    ours = [bb._sample_full(lg, 1.0) for _ in range(50)]
    assert len(set(ours)) > 40                           # full softmax at T=1 over a flat-ish 65536 vocabulary
    small_lg = torch.tensor([0.0, 1.0, 2.0])
    np.random.seed(1)
    freq = np.bincount([bb._sample_full(small_lg, 1.0) for _ in range(20000)], minlength=3) / 20000
    assert np.allclose(freq, torch.softmax(small_lg, -1).numpy(), atol=0.02)

    class RealPipe(_Pipe):
        sample_logits = PIPELINE.sample_logits           # the package's own sampler (would be greedy)

    class BigVocabStub(StubCortex):
        def __init__(self):
            super().__init__()
            self.pipe = RealPipe()

        def forward(self, tokens, state):
            lg_, st = super().forward(tokens, state)
            # a tail of tiny but nonzero mass, as in a real 65536 vocabulary: the float32 cumsum then ends below 1.0
            return torch.cat([lg_, torch.full((65536 - len(lg_),), float(lg_.max()) - 18.0)]), st

    b = _brain(str(tmp_path / "h"), cortex=BigVocabStub())
    texts = b._keep_bank()
    assert len(set(texts)) > len(texts) // 2             # greedy would give ONE text repeated 24 times
