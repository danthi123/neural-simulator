# Capability: Lifelong learning
# Record: brain/research/2026-11-21-replay-priority-prereg.md
# Expected-fail: none (mechanics only: weights, RNG, persistence, the leak extractor; effects are measured on lives)
"""Brain(replay_priority=...) (D2-PR) on the stub cortex with a fake deterministic adapter forward and a fake replay."""
import importlib.util
import json
import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

import brain.brain as bb
from brain import clock
from brain.brain import Brain
from brain.eval import life_leak
from brain.tests.test_brain_key_switch import KW, ROOT, StubCortex
from brain.plastic_cortex import PlasticCortex
from brain.tests.test_use_scale import T0, _fake_plastic

BASE = "6d5726850"


class FakePlastic(PlasticCortex):
    """Real adapter parameters (save/refresh work) with a deterministic forward that depends on ids and `enabled`."""
    def forward(self, ids, base):
        V = 128
        t = torch.arange(len(ids)).float()[:, None] * 0.37 + torch.arange(V).float()[None, :] * (0.11 if self.enabled else 0.07)
        return torch.sin(t) * 2.0 * (1 + 0.01 * sum(ids) % 3), None


def _fp(stub):
    p = _fake_plastic(stub)
    p.__class__ = FakePlastic
    return p


def _plastic_of(self):
    if self.cortex.plastic is None:
        self.cortex.plastic = _fp(self.raw_cortex)
    return self.cortex.plastic


@pytest.fixture
def night(monkeypatch):
    seen = []

    def fake_replay(plastic, base, eps, generic, **k):
        seen.append({"eps": list(eps), "seed": k["seed"], "enabled": plastic.enabled})
        return [(1.0, 0.0)] * k["steps"]
    monkeypatch.setattr(bb, "replay_into_cortex", fake_replay)
    monkeypatch.setattr(Brain, "_plastic", _plastic_of)
    ticks = iter(range(10 ** 6))
    clock.set(lambda: T0 + next(ticks))
    yield seen
    clock.set(None)


def _brain(home, **kw):
    b = Brain(home, cortex=StubCortex(), **KW, **kw)
    for t in ("My cat is called Pixel.", "I drive a red Volvo.", "My sister's name is Marguerite."):
        b.hear(t)
    b.cortex.consolidated_on = False
    return b


def _nll(b, e):
    ids = b.cortex.pipe.encode(f"{e.text}\n\n")
    p = b.cortex.plastic
    lp = F.log_softmax(p(ids, [s.float() for s in b.base])[0].float(), -1)
    return float(F.nll_loss(lp[:-1], torch.tensor(ids[1:])))


def test_off_is_identical_and_leaves_no_trace(tmp_path, night):
    clock.set(lambda: T0)   # same clock for both brains
    a, b = _brain(str(tmp_path / "a")), _brain(str(tmp_path / "b"), replay_priority=None)
    a.sleep(), b.sleep()
    a.save(), b.save()
    assert night[0]["eps"] == night[1]["eps"] and night[0]["seed"] == night[1]["seed"]
    assert a.last_replay_weights is None and b.last_replay_weights is None
    assert open(tmp_path / "a" / "meta.json").read() == open(tmp_path / "b" / "meta.json").read()
    assert "replay_priority" not in json.load(open(tmp_path / "a" / "meta.json"))


def test_weights_for_every_replayed_episode_incl_permanent(tmp_path, night):
    b = _brain(str(tmp_path / "h"), replay_priority="strength_x_loss")
    eps = list(b.hippocampus.episodes)
    assert any(e.permanent for e in eps)
    now = clock.now()
    b.sleep()
    rows = b.last_replay_weights
    assert len(rows) == len(eps) == len(night[0]["eps"])
    assert sum(r["permanent"] for r in rows) == sum(e.permanent for e in eps) > 0
    assert sum(r["share"] for r in rows) == pytest.approx(1.0)
    for e, r, (txt, w) in zip(eps, rows, night[0]["eps"]):
        assert txt == f"{e.text}\n\n" and r["loss"] > 0
        assert w == pytest.approx(r["strength"] * r["loss"]) == pytest.approx(r["weight"])
        assert r["loss"] == pytest.approx(_nll(b, e))
    assert night[0]["enabled"] is True


def test_loss_weights_equal_L_e_and_plastic_flag_restored(tmp_path, night):
    b = _brain(str(tmp_path / "h"), replay_priority="loss")
    b.cortex.plastic = None
    b.sleep()
    assert [w for _, w in night[0]["eps"]] == pytest.approx([_nll(b, e) for e in b.hippocampus.episodes])
    # enabled False before the night (consolidated off path) is restored by the weight pass itself
    p = b.cortex.plastic
    p.enabled = False
    b._prioritised_replay(p, list(b.hippocampus.episodes), clock.now())
    assert p.enabled is False


def test_rng_untouched_and_replay_seed_same(tmp_path, night):
    a = _brain(str(tmp_path / "a"))
    b = _brain(str(tmp_path / "b"), replay_priority="loss")
    torch.manual_seed(5)
    s0 = torch.get_rng_state()
    b.sleep()
    assert torch.equal(s0, torch.get_rng_state())
    a.sleep()
    assert night[0]["seed"] == night[1]["seed"]


def test_meta_persistence_and_validation(tmp_path, night):
    home = str(tmp_path / "h")
    b = _brain(home, replay_priority="loss")
    b.save()
    assert json.load(open(os.path.join(home, "meta.json")))["replay_priority"] == "loss"
    b2 = Brain(home, cortex=StubCortex(), **KW)
    assert b2._replay_priority == "loss" and "replay_priority" not in b2.meta
    b3 = Brain(home, cortex=StubCortex(), **KW, replay_priority="strength_x_loss")   # explicit wins
    assert b3._replay_priority == "strength_x_loss"
    with pytest.raises(ValueError):
        Brain(str(tmp_path / "x"), cortex=StubCortex(), **KW, replay_priority="nope")


def test_flags_off_brain_equals_pinned_base_brain(tmp_path, night):
    """Brain.sleep with the switch absent: the eps handed to replay equal those of the pinned pre-D2-PR Brain."""
    try:
        src = subprocess.run(["git", "-C", ROOT, "show", f"{BASE}:brain/brain.py"], capture_output=True, text=True,
                             check=True).stdout
    except Exception:
        pytest.skip("base commit not available")
    path = str(tmp_path / "_ref_brain_for_test.py")   # outside the repo (absolute imports only in brain.py)
    with open(path, "w") as fh:
        fh.write(src)
    try:
        spec = importlib.util.spec_from_file_location("brain._ref_brain_for_test", path)
        ref = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ref
        spec.loader.exec_module(ref)
    finally:
        os.remove(path)
    seen_ref = []
    ref.replay_into_cortex = lambda plastic, base, eps, generic, **k: seen_ref.append((list(eps), k["seed"])) or [(1.0, 0.0)] * k["steps"]
    ref.Brain._plastic = _plastic_of
    try:
        outs = []
        for name, cls in (("ref", ref.Brain), ("new", Brain)):
            clock.set(lambda: T0)
            b = cls(str(tmp_path / name), cortex=StubCortex(), **KW)
            for t in ("My cat is called Pixel.", "I drive a red Volvo."):
                b.hear(t)
            b.cortex.consolidated_on = False
            b.sleep()
            b.save()
            outs.append(open(tmp_path / name / "meta.json").read())
        assert len(seen_ref) == len(night) == 1 and len(seen_ref[0][0]) >= 4   # permanent self + 2 told: not vacuous
        assert any(w != seen_ref[0][0][0][1] for _, w in seen_ref[0][0])         # strengths differ across episodes
        assert night[0]["eps"] == seen_ref[0][0] and night[0]["seed"] == seen_ref[0][1]
        assert json.loads(outs[0]).keys() == json.loads(outs[1]).keys()
    finally:
        sys.modules.pop("brain._ref_brain_for_test", None)


def _res():
    led = {"dan": {"cls": "H", "telling": None}, "nova": {"cls": "H", "telling": None},
           "pixel": {"cls": "H", "telling": "My cat is called Pixel."}}
    row = lambda item, brain: {"item": item, "brain": brain, "kind": "x"}
    return {"ledger": led, "brain_kw": {},
            "probes": [{"night": 5, "arm": "intact", "rows": [row("dan", "Of course, you're Dan, my friend."),
                                                              row("pixel", "OF COURSE, YOU’RE DAN."), row("pixel", "Pixel."),
                                                              row(None, "I'm Nova, a young digital brain.")]},
                       {"night": 5, "arm": "both-off", "rows": [row(None, "My name is Nova.")]},
                       {"night": 5, "arm": "hippocampus-off", "rows": [row("nova", "My name is Nova."), row(None, "Fine.")]},
                       {"night": 5, "arm": "hippocampus-off", "line": "prunedA", "rows": [row(None, "My name is Nova.")]},
                       {"night": 5, "arm": "hippocampus-off", "line": "trunk", "tag": "ab-recall",
                        "rows": [row("pixel", "My name is Nova.")]}],
            "interview_probes": [{"night": 5, "arm": "hippocampus-off",
                                  "rows": [row("d03", "Of course, you're Dan, my friend."), row("d03", "x")]}]}


def test_leak_extractor_on_synthetic_res():
    t = life_leak.leak_table(_res())
    assert t[("intact", 5)]["n"] == 3 and t[("intact", 5)]["leaks"] == 2     # the identity item "dan" is excluded
    assert t[("hippocampus-off", 5)]["n"] == 3 and t[("hippocampus-off", 5)]["leaks"] == 1   # "nova" excluded
    assert t[("hippocampus-off", 5)]["by_source"] == {"probe": [1, 0], "interview": [2, 1]}   # pruned + tagged reads out
    assert t[("hippocampus-off", 5)]["by_kind"] == {"probe:x": [1, 0], "interview:x": [2, 1]}
    assert ("both-off", 5) not in t
    assert life_leak.pooled(t, {5}) == {"n": 6, "leaks": 3, "share": 0.5}


def test_night_record_shares_and_strength_reference():
    from types import SimpleNamespace
    from brain.eval.life import replay_priority_record
    rows = [{"text": "User: Do you remember my name?\n\nAssistant: Of course, you're Dan, my friend.", "permanent": True,
             "strength": 3.0, "loss": 0.2}, {"text": "User: My cat is called Pixel.", "permanent": False, "strength": 1.0,
             "loss": 1.8}]
    for r in rows:
        r["weight"] = r["loss"]
    tot = sum(r["weight"] for r in rows)
    for r in rows:
        r["share"] = r["weight"] / tot
    led = {"pixel": {"cls": "H", "telling": "My cat is called Pixel."}, "dan": {"cls": "H", "telling": None}}
    rec = replay_priority_record(SimpleNamespace(last_replay_weights=rows), led)
    assert rec["shares_by_class"] == {"H": 0.9, "permanent": 0.1}
    assert rec["permanent_share"] == 0.1 and rec["max_share"] == 0.9 and rec["top6_share"] == 1.0
    assert rec["strength_permanent_share"] == 0.75 and rec["n"] == 2 and rec["mean_loss"] == 1.0
    assert replay_priority_record(SimpleNamespace(last_replay_weights=None), led) is None
