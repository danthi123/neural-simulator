# Capability: Lifelong learning
# Record: brain/research/2026-11-23-kp-identity-prereg.md (section 1, code item 6)
# Expected-fail: none (mechanics only: the turn rule, the masked keep loss, the off/"prior" identity to the head; effects are read on the gate and lives)
"""K-P-U (Brain(keep_bank="prior_u")) and K-P-0 (Brain(keep_bank="prior_0")): the mask on hand-made texts, a full-cover
pair equals the str path, the "prior_u" bank equals the "prior" bank, and keep_bank=None / "prior" stay byte-identical to
the head before the switch (stub cortex, 3 nights; and the real 0.1B cortex for the "prior" bank texts and keep loss)."""
import importlib.util
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

import brain.brain as bb
from brain import clock
from brain.brain import Brain
from brain.keep_mask import char_frame_mask, frame_turn_positions, turn_labels
from brain.tests.test_brain_key_switch import KW, ROOT
from brain.tests.test_keep_bank import DiffPlastic, SampStub, _adapters, _brain, _plastic_of, small  # noqa: F401
from brain.tests.test_use_scale import T0, _fake_plastic

HEAD = "68a05818b"   # the head before K-P-U / K-P-0: None and "prior" must equal it


# ---------------------------------------------------------------- the turn rule on hand-made texts
def _chars(text):
    """A char-level tokenizer: ids are code points, decode is chr (so token i = character i)."""
    return [ord(c) for c in text], (lambda ids: "".join(chr(i) for i in ids))


def _kept_text(text):
    ids, dec = _chars(text)
    pos = set(frame_turn_positions(ids, dec, "User:"))
    # position i predicts char i+1: the kept characters are i+1 for i in pos (and the last char via the last position)
    return "".join(text[i + 1] for i in sorted(pos) if i + 1 < len(text))


def test_user_assistant_text():
    t = "User: hi there\n\nAssistant: hello, I am a bot\n\nUser: ok\n\n"
    kept = _kept_text(t)
    assert "hi there" in kept and "ok" in kept
    assert "hello" not in kept and "Assistant" not in kept and "bot" not in kept
    assert [lab for _, lab in turn_labels(t)] == ["User", "Assistant", "User"]


def test_user_bot_text_and_multiword_label():
    t = "User: what is up\nBot: nothing much\nMy Friend Al: hey\nUser: fine\n"
    kept = _kept_text(t)
    assert "what is up" in kept and "fine" in kept
    assert "nothing much" not in kept and "hey" not in kept
    assert [lab for _, lab in turn_labels(t)] == ["User", "Bot", "My Friend Al", "User"]
    # four words before ":" is not a label: the line continues the open turn
    t2 = "User: a\nthe one big dog: x\n"
    assert [lab for _, lab in turn_labels(t2)] == ["User"]
    assert "x" in _kept_text(t2)


def test_single_newline_turn_and_continuation():
    t = "User: line one\nline two runs on\nAI: reply\nmore reply\n"
    kept = _kept_text(t)
    assert "line one" in kept and "line two runs on" in kept
    assert "reply" not in kept and "more" not in kept


def test_unlabelled_prose_inside_user_turn_is_kept():
    t = "User: The village sits at the foot of a hill. In spring the fields turn green.\n\n"
    ids, dec = _chars(t)
    pos = frame_turn_positions(ids, dec, "User:")
    assert pos == list(range(len(ids)))             # the whole text is the frame turn
    assert all(char_frame_mask(t, "User:"))
    assert frame_turn_positions([], dec, "User:") == []


def test_multichar_tokens_label_on_token_last_char():
    # tokens: "User", ":", " hi", "\n\n", "Assistant", ":", " yo", "\n\n"
    toks = ["User", ":", " hi", "\n\n", "Assistant", ":", " yo", "\n\n"]
    dec = lambda ids: "".join(toks[i] for i in ids)
    pos = frame_turn_positions(list(range(len(toks))), dec, "User:")
    # tokens in the User turn: 0..3 ("\n\n" ends before the label); positions predicting them: 0..2
    assert pos == [0, 1, 2]


# ---------------------------------------------------------------- the masked keep loss (real sleep loop)
def _loop(gen, steps=30):
    from brain.sleep import sleep
    stub = SampStub()
    torch.manual_seed(5)
    p = _fake_plastic(stub)
    p.__class__ = DiffPlastic
    eps = [("User: My cat is called Pixel.\n\n", 1.0), ("User: I drive a red Volvo.\n\n", 0.5)]
    losses = sleep(p, [torch.zeros(1)], eps, gen, steps=steps, lr=1e-2, kl_weight=3.0, seed=1)
    return losses, {k: v.detach().clone() for k, v in p.state_dict().items()}


def test_full_cover_pair_equals_str_path():
    stub = SampStub()
    gen = ["User:" + " word" * n + "\n\nAssistant: ok\n\n" for n in (3, 9, 20)]
    pairs = [(g, list(range(len(stub.pipe.encode(g))))) for g in gen]
    ls, ss = _loop(gen)
    lp, sp = _loop(pairs)
    assert ls == lp
    assert all(torch.equal(ss[k], sp[k]) for k in ss)
    assert max(k for _, k in ls) > 0


def test_partial_pair_changes_the_keep_loss_and_empty_pair_is_zero():
    stub = SampStub()
    g = "User:" + " word" * 12 + "\n\n"
    n = len(stub.pipe.encode(g))
    lf, _ = _loop([g], steps=10)
    lh, _ = _loop([(g, list(range(n // 2)))], steps=10)
    le, _ = _loop([(g, [])], steps=10)
    assert [k for _, k in lf] != [k for _, k in lh]
    assert all(k == 0.0 for _, k in le)
    assert [r for r, _ in lf][0] == [r for r, _ in lh][0] == [r for r, _ in le][0]   # replay RNG and first replay unchanged


# ---------------------------------------------------------------- Brain: banks, records, meta
def test_prior_u_bank_equals_prior_bank_and_record(tmp_path, small):
    a, u = _brain(str(tmp_path / "a"), keep_bank="prior"), _brain(str(tmp_path / "u"), keep_bank="prior_u")
    assert a._keep_bank() == u._keep_bank()
    fa, fu = json.load(open(tmp_path / "a" / "keep_bank.json")), json.load(open(tmp_path / "u" / "keep_bank.json"))
    assert fa == fu
    seen = []
    orig = bb.replay_into_cortex
    bb.replay_into_cortex = lambda p, base, eps, generic, **k: (seen.append(list(generic)), orig(p, base, eps, generic, **k))[1]
    try:
        a.sleep(), u.sleep()
    finally:
        bb.replay_into_cortex = orig
    ga, gu = seen
    assert all(isinstance(x, str) for x in ga) and [t for t, _ in gu] == ga   # the same nightly subset
    assert all(isinstance(pos, list) for _, pos in gu)
    assert "kept_pos_share" not in a.last_keep_record and "mode" not in a.last_keep_record
    r = u.last_keep_record
    assert r["mode"] == "prior_u" and 0.0 < r["kept_pos_share"] <= 1.0
    assert r["subset_hash"] == a.last_keep_record["subset_hash"]
    u.save()
    assert json.load(open(tmp_path / "u" / "meta.json"))["keep_bank"] == "prior_u"


def test_prior_0_bank_frame_and_refusals(tmp_path, small):
    z = _brain(str(tmp_path / "z"), keep_bank="prior_0")
    tz = z._keep_bank()
    f = json.load(open(tmp_path / "z" / "keep_bank.json"))
    assert f["frame"] == "" and f["seed"] == z.sleep_seed + bb.KEEP_BANK_SEED_OFFSET and len(tz) == 24
    assert all(t.endswith("\n\n") for t in tz)
    a = _brain(str(tmp_path / "a"), keep_bank="prior")
    assert tz != a._keep_bank()
    z.sleep(), z.save()
    assert z.last_keep_record["mode"] == "prior_0" and "kept_pos_share" not in z.last_keep_record
    z.release()
    # switching mode in a home that slept is refused; the same mode reopens
    with pytest.raises(ValueError, match="keep_bank"):
        Brain(str(tmp_path / "z"), cortex=SampStub(), **KW, keep_bank="prior")
    with pytest.raises(ValueError, match="keep_bank"):
        Brain(str(tmp_path / "z"), cortex=SampStub(), **KW, keep_bank="prior_u")
    Brain(str(tmp_path / "z"), cortex=SampStub(), **KW, keep_bank="prior_0").release()
    # a "User:" bank file under prior_0 (or a "" bank under prior_u) is refused, never silently reused
    a.save(), a.release()
    m = json.load(open(tmp_path / "a" / "meta.json"))
    m["keep_bank"] = "prior_0"
    m["sleeps"] = 0
    json.dump(m, open(tmp_path / "a" / "meta.json", "w"))
    b = Brain(str(tmp_path / "a"), cortex=SampStub(), **KW, keep_bank="prior_0")
    with pytest.raises(ValueError, match="frame"):
        b._keep_bank()
    with pytest.raises(ValueError):
        Brain(str(tmp_path / "x"), cortex=SampStub(), **KW, keep_bank="prior_x")


def test_life_cli_choices():
    from brain.eval import life
    import argparse
    seen = {}

    def fake_run(cortex, seed, days, split, homes, out, full, nosleep, resume, kw, *a, **k):
        seen.update(kw)
        raise SystemExit(0)
    orig = life.run
    life.run = fake_run
    try:
        for kb in ("prior", "prior_u", "prior_0"):
            with pytest.raises(SystemExit):
                life.main(["--seed", "1", "--homes", "h", "--out", "o", "--keep-bank", kb], cortex=object())
            assert seen["keep_bank"] == kb
        with pytest.raises(SystemExit) as e:
            life.main(["--seed", "1", "--homes", "h", "--out", "o", "--keep-bank", "nope"], cortex=object())
        assert e.value.code == 2
    finally:
        life.run = orig
    assert argparse


# ---------------------------------------------------------------- identity to the head
def _load_head(tmp_path, rel, name):
    try:
        src = subprocess.run(["git", "-C", ROOT, "show", f"{HEAD}:{rel}"], capture_output=True, text=True,
                             check=True).stdout
    except Exception:
        pytest.skip("head commit not available")
    path = str(tmp_path / f"_{name}.py")
    with open(path, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    os.remove(path)
    return mod


@pytest.fixture
def head(tmp_path):
    rs = _load_head(tmp_path, "brain/sleep.py", "brain._head_sleep_for_test")
    rb = _load_head(tmp_path, "brain/brain.py", "brain._head_brain_for_test")
    rb.replay_into_cortex = rs.sleep     # the head brain runs the head sleep loop
    yield rb
    sys.modules.pop("brain._head_sleep_for_test", None)
    sys.modules.pop("brain._head_brain_for_test", None)


def test_none_and_prior_three_nights_byte_identical_to_head(tmp_path, small, head, monkeypatch):
    head.Brain._plastic = _plastic_of
    monkeypatch.setattr(head, "KEEP_BANK_N", 24)
    monkeypatch.setattr(head, "KEEP_BANK_NIGHT", 5)
    for kb in (None, "prior"):
        homes, recs = {}, {}
        for name, cls in (("ref", head.Brain), ("new", Brain)):
            clock.set(lambda: T0)
            torch.manual_seed(3)
            h = str(tmp_path / f"{name}_{kb}")
            b = _brain(h, cls=cls, keep_bank=kb)
            recs[name] = []
            for _ in range(3):
                recs[name].append((b.sleep(), b.last_keep_record))
            b.save()
            homes[name] = h
        assert recs["ref"] == recs["new"]
        a0, a1 = _adapters(homes["ref"]), _adapters(homes["new"])
        assert a0.keys() == a1.keys() and all(torch.equal(a0[k], a1[k]) for k in a0)
        for f in ("meta.json", "keep_bank.json"):
            p0, p1 = os.path.join(homes["ref"], f), os.path.join(homes["new"], f)
            assert os.path.exists(p0) == os.path.exists(p1) == (kb is not None or f == "meta.json")
            if os.path.exists(p0):
                assert open(p0).read() == open(p1).read()
        assert any(float(v.abs().sum()) > 0 for k, v in a0.items() if "B" in k)


@pytest.fixture(scope="module")
def cortex01():
    from brain.cortex import Cortex, resolve_model
    try:
        return Cortex(resolve_model("0.1b"))
    except Exception as e:   # no local checkpoint
        pytest.skip(f"0.1B cortex unavailable: {e}")


def test_prior_bank_texts_and_keep_loss_on_01b_equal_head(tmp_path, head, monkeypatch, cortex01):
    """The real 0.1B cortex (CPU): "prior" bank texts, per-step replay/keep losses, adapters and bank file equal the head."""
    for mod in (head, bb):
        monkeypatch.setattr(mod, "KEEP_BANK_N", 4)
        monkeypatch.setattr(mod, "KEEP_BANK_TOKENS", 10)
        monkeypatch.setattr(mod, "KEEP_BANK_NIGHT", 3)
    kw = dict(sleep_on_restart=False, sleep_steps=5, sleep_lr=1e-3, sleep_kl_weight=1.0, adapter_rank=2, sleep_seed=11)
    out = {}
    for name, cls in (("ref", head.Brain), ("new", Brain)):
        ticks = iter(range(10 ** 6))
        clock.set(lambda: T0 + next(ticks))
        random.seed(1), np.random.seed(1), torch.manual_seed(1)
        h = str(tmp_path / name)
        b = cls(h, cortex=cortex01, keep_bank="prior", **kw)
        b.hear("My cat is called Pixel.")
        losses = [b.sleep() for _ in range(2)]
        b.save()
        out[name] = (b._keep_bank(), losses, [b.last_keep_record], _adapters(h),
                     open(os.path.join(h, "keep_bank.json")).read())
        b.release()
    clock.set(None)
    (t0, l0, r0, a0, f0), (t1, l1, r1, a1, f1) = out["ref"], out["new"]
    assert t0 == t1 and len(t0) == 4 and all(t.startswith("User:") for t in t0)
    assert l0 == l1 and max(k for night in l0 for _, k in night) > 0
    assert r0 == r1 and f0 == f1
    assert a0.keys() == a1.keys() and all(torch.equal(a0[k], a1[k]) for k in a0)
