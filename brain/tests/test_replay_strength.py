# Capability: Memory strength and forgetting
# Record: brain/research/2026-11-27-fable-review-dg-port.md
# Expected-fail: none (mechanics only: draw counts, the night rule, off-switch identity, persistence; effects are screened offline)
"""Brain(replay_strengthen=, strength_setpoint=) and brain/replay_strength.py on the stub cortex."""
import importlib.util
import json
import os
import random
import subprocess
import sys

import pytest
import torch

import brain.brain as bb
import brain.sleep as sleep_mod
from brain import clock
from brain import replay_strength as RSM
from brain.brain import Brain
from brain.hippocampus import Episode, Hippocampus
from brain.tests.test_brain_key_switch import KW, ROOT, StubCortex
from brain.tests.test_keep_bank import DiffPlastic, _plastic_of as _diff_plastic_of
from brain.tests.test_use_scale import T0, _fake_plastic

KW60 = {**KW, "sleep_steps": 60}     # enough draws that told episodes are replayed beside the permanent ones
TELL = ("My cat is called Pixel.", "I drive a red Volvo.", "My sister's name is Marguerite.", "I work at the bakery.")


# ------------------------------------------------------------------ pure functions
def test_replay_counts_equal_the_real_sleep_loop(monkeypatch):
    """The counts are the episode draws the REAL brain/sleep.py loop makes (same RNG sequence), on unequal weights."""
    stub = StubCortex()
    p = _fake_plastic(stub)
    p.__class__ = DiffPlastic
    eps = [(f"User: fact number {j}.\n\n", w) for j, w in enumerate((3.0, 0.2, 1.1, 0.0005, 0.7))]
    gen = [f"Ordinary text {j}.\n\n" for j in range(3)]
    texts = {t: j for j, (t, _) in enumerate(eps)}
    seen = []
    real = sleep_mod._ids

    def spy(c, text):
        if text in texts:
            seen.append(texts[text])
        return real(c, text)
    monkeypatch.setattr(sleep_mod, "_ids", spy)
    sleep_mod.sleep(p, [torch.zeros(1)], eps, gen, steps=40, lr=1e-3, kl_weight=1.0, seed=17)
    got = [seen.count(j) for j in range(len(eps))]
    assert sum(got) == 40 and got == RSM.replay_counts([w for _, w in eps], len(gen), 40, 17)
    assert got != RSM.replay_counts([w for _, w in eps], len(gen), 40, 18)       # the seed matters (not vacuous)
    assert got != RSM.replay_counts([w for _, w in eps], len(gen) + 1, 40, 17) or len(gen) == 1   # randrange consumed
    with pytest.raises(ValueError):
        RSM.replay_counts([1.0], 0, 5, 0)


def _eps():
    return [Episode(text="a", delta=None, key=None, strength=2.0, last_used=0.0, created=0.0),
            Episode(text="b", delta=None, key=None, strength=0.5, last_used=0.0, created=0.0),
            Episode(text="p", delta=None, key=None, strength=1.0, last_used=0.0, created=0.0, permanent=True),
            Episode(text="c", delta=None, key=None, strength=1.0, last_used=0.0, created=0.0)]


def test_strengthen_counts_permanent_and_clock():
    eps = _eps()
    add = RSM.strengthen(eps, [3, 0, 5, 1], 0.125, now=86400.0)
    assert add == [0.375, 0.0, 0.0, 0.125]
    assert [e.strength for e in eps] == [2.375, 0.5, 1.0, 1.125]
    assert [e.last_used for e in eps] == [86400.0, 0.0, 0.0, 86400.0]     # unreplayed and permanent keep their clock
    eps = _eps()
    RSM.strengthen(eps, [3, 0, 5, 1], 0.125, now=86400.0, reset_clock=False)
    assert all(e.last_used == 0.0 for e in eps)


def test_scale_hits_setpoint_keeps_ratios_and_clamps():
    hc = Hippocampus()
    eps = _eps()
    now = 3 * 86400.0
    before = [e.strength for e in eps]
    rec = RSM.homeostatic_scale(eps, hc.effective_strength, 1.0, now)
    live = [e for e in eps if not e.permanent]
    assert sum(hc.effective_strength(e, now) for e in live) / 3 == pytest.approx(1.0)
    assert eps[0].strength / eps[1].strength == pytest.approx(before[0] / before[1])     # relative strength kept
    assert eps[2].strength == 1.0 and rec["n"] == 3
    eps = _eps()
    rec = RSM.homeostatic_scale(eps, hc.effective_strength, 100.0, now)
    assert rec["g"] == RSM.SCALE_MAX
    assert RSM.homeostatic_scale([eps[2]], hc.effective_strength, 1.0, now)["g"] == 1.0


def test_night_update_scaling_only_draws_nothing():
    hc = Hippocampus()
    eps = _eps()
    st = random.getstate()
    rec = RSM.night_update(eps, [1.0] * 4, 3, 50, 1, 86400.0, hc.effective_strength, boost=None, setpoint=1.0)
    assert "counts" not in rec and random.getstate() == st and all(e.last_used == 0.0 for e in eps)


# ------------------------------------------------------------------ the Brain switch
@pytest.fixture
def night(monkeypatch):
    seen = []

    def fake_replay(plastic, base, eps, generic, **k):
        seen.append({"eps": list(eps), "seed": k["seed"], "steps": k["steps"], "n_generic": len(generic)})
        return [(1.0, 0.0)] * k["steps"]
    monkeypatch.setattr(bb, "replay_into_cortex", fake_replay)
    monkeypatch.setattr(Brain, "_plastic", _diff_plastic_of)
    yield seen
    clock.set(None)


def _life(home, nights=2, **kw):
    ticks = iter(range(0, 10 ** 9, 3600))     # an hour per clock read: decay matters
    clock.set(lambda: T0 + next(ticks))
    b = Brain(home, cortex=StubCortex(), **KW60, **kw)
    for n in range(nights):
        for t in TELL[n % 2 * 2:n % 2 * 2 + 2]:
            b.hear(f"{t} ({n})")
        b.cortex.consolidated_on = False
        b.sleep()
        b.save()
    return b


def _read(home, f):
    return open(os.path.join(home, f), "rb").read()


def test_off_is_byte_identical(tmp_path, night):
    a = _life(str(tmp_path / "a"))
    b = _life(str(tmp_path / "b"), replay_strengthen=None, strength_setpoint=None)
    assert night[:2] == night[2:4]
    for f in ("meta.json", "episodes.json"):
        assert _read(str(tmp_path / "a"), f) == _read(str(tmp_path / "b"), f)
    assert a.last_strength_record is None and b.last_strength_record is None


def test_on_applies_the_rule_with_the_nights_own_draws(tmp_path, night):
    b = _life(str(tmp_path / "h"), nights=1)
    before = [(e.strength, e.last_used, e.permanent) for e in b.hippocampus.episodes]
    ticks = iter(range(0, 10 ** 9, 3600))
    clock.set(lambda: T0 + 10 ** 7 + next(ticks))
    b2 = Brain(str(tmp_path / "h"), cortex=StubCortex(), **KW60, replay_strengthen=0.25, strength_setpoint=None)
    for t in TELL[2:]:
        b2.hear(t)
    b2.cortex.consolidated_on = False
    pre = [(e.strength, e.last_used, e.permanent) for e in b2.hippocampus.episodes]
    b2.sleep()
    rec = b2.last_strength_record
    nt = night[-1]
    assert rec["counts"] == RSM.replay_counts([w for _, w in nt["eps"]], nt["n_generic"], nt["steps"], nt["seed"])
    assert sum(rec["counts"]) == nt["steps"] and len(rec["counts"]) == len(pre)
    assert any(c > 0 for c, (_, _, p) in zip(rec["counts"], pre) if not p)
    for e, (s, lu, p), c in zip(b2.hippocampus.episodes, pre, rec["counts"]):
        if p or c == 0:
            assert (e.strength, e.last_used) == (s, lu)
        else:
            assert e.strength == pytest.approx(s + 0.25 * c) and e.last_used > lu
    assert len(before) < len(pre)


def test_setpoint_on_hits_the_mean_and_persists(tmp_path, night):
    home = str(tmp_path / "h")
    b = _life(home, nights=2, replay_strengthen=RSM.REPLAY_BOOST, strength_setpoint=RSM.SET_POINT)
    rec = b.last_strength_record
    assert rec["setpoint"] == 1.0 and rec["n"] >= 2 and RSM.SCALE_MIN <= rec["g"] <= RSM.SCALE_MAX
    saved = json.load(open(os.path.join(home, "episodes.json")))
    rows = saved if isinstance(saved, list) else saved["episodes"]
    assert [r["strength"] for r in rows] == [e.strength for e in b.hippocampus.episodes]
    b.release()
    b2 = Brain(home, cortex=StubCortex(), **KW)
    assert [e.strength for e in b2.hippocampus.episodes] == [r["strength"] for r in rows]
    assert b2.replay_strengthen is None and "replay_strengthen" not in json.load(open(os.path.join(home, "meta.json")))


def test_switch_draws_no_rng(tmp_path, night):
    b = _life(str(tmp_path / "h"), nights=1, replay_strengthen=0.125, strength_setpoint=1.0)
    for t in TELL[2:]:
        b.hear(t)
    b.cortex.consolidated_on = False
    torch.manual_seed(3)
    random.seed(4)
    ts, rs = torch.get_rng_state(), random.getstate()
    b.sleep()
    assert torch.equal(ts, torch.get_rng_state()) and rs == random.getstate()


def test_off_equals_the_head_brain(tmp_path, night):
    """With the switches absent, the eps handed to replay and the saved files equal the pre-switch HEAD brain.py."""
    try:
        src = subprocess.run(["git", "-C", ROOT, "show", "HEAD:brain/brain.py"], capture_output=True, text=True,
                             check=True).stdout
    except Exception:
        pytest.skip("HEAD not available")
    if "replay_strengthen" in src:
        pytest.skip("HEAD already carries the switch")
    path = str(tmp_path / "_ref_brain_rs.py")
    open(path, "w").write(src)
    try:
        spec = importlib.util.spec_from_file_location("brain._ref_brain_rs", path)
        ref = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = ref
        spec.loader.exec_module(ref)
    finally:
        os.remove(path)
    seen_ref = []
    ref.replay_into_cortex = lambda plastic, base, eps, generic, **k: seen_ref.append((list(eps), k["seed"])) or [(1.0, 0.0)] * k["steps"]
    ref.Brain._plastic = _diff_plastic_of
    try:
        for name, cls in (("ref", ref.Brain), ("new", Brain)):
            ticks = iter(range(0, 10 ** 9, 3600))
            clock.set(lambda: T0 + next(ticks))
            b = cls(str(tmp_path / name), cortex=StubCortex(), **KW)
            for t in TELL[:2]:
                b.hear(t)
            b.cortex.consolidated_on = False
            b.sleep()
            b.save()
        assert len(seen_ref) == 1 and [(x["eps"], x["seed"]) for x in night] == seen_ref
        for f in ("episodes.json", "meta.json"):
            assert _read(str(tmp_path / "ref"), f) == _read(str(tmp_path / "new"), f)
    finally:
        sys.modules.pop("brain._ref_brain_rs", None)
