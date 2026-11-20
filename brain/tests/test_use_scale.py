# Capability: Lifelong learning
# Record: brain/research/2026-11-19-life-muse-mscale-prereg.md
# Expected-fail: none (mechanics only; the prereg's dev criteria are measured on lives, not here)
"""M-use (Hippocampus.USE_GATE / FREEZE_USE) and M-scale (Brain(sleep_downscale), PlasticCortex.downscale): mechanics only,
on a monkeypatched similarity and a fake adapter set. No model weights; no assertion reads transcript content or H counts."""
import importlib.util
import json
import os
import random
import subprocess
import sys

import pytest
import torch

import brain.brain as bb
from brain import clock
from brain.brain import Brain
from brain.hippocampus import Episode, Hippocampus
from brain.hippocampus_keys import KeyedHippocampus
from brain.plastic_cortex import PlasticCortex
from brain.tests.test_brain_key_switch import KW, ROOT, TALK, StubCortex

REF = "61c6ec728"   # the Hippocampus before M-use (pinned): the OFF path must equal it bit for bit
DAY = 86400.0
T0 = 1_000_000.0


@pytest.fixture(autouse=True)
def _clock():
    clock.set(lambda: T0)
    yield
    clock.set(None)


def _ref_module():
    try:
        src = subprocess.run(["git", "-C", ROOT, "show", f"{REF}:brain/hippocampus.py"], capture_output=True, text=True,
                             check=True).stdout
    except Exception:
        pytest.skip(f"{REF} not available")
    path = os.path.join(ROOT, "brain", "_ref_hippocampus_for_test.py")
    with open(path, "w") as fh:
        fh.write(src)
    try:
        spec = importlib.util.spec_from_file_location("brain._ref_hippocampus_for_test", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    finally:
        os.remove(path)
    return mod


def _fake(base):
    """A hippocampus of `base` whose similarities come from `.sims` (set per call)."""
    class Fake(base):
        sims = None

        def similarities(self, cortex, base_, cue_text, speaker="User"):
            return self.sims.clone()
    return Fake()


def _ep(cls, strength=1.0, last_used=T0, permanent=False):
    return cls(text="x", delta=[None], key=torch.zeros(1), strength=strength, created=last_used, last_used=last_used,
               permanent=permanent)


def _state(h):
    return [(e.strength, e.uses, e.last_used) for e in h.episodes]


def _recall(h, sims, **kw):
    h.sims = torch.tensor(sims, dtype=torch.float32)
    return h.recall(None, None, "cue", **kw)


# ---------------------------------------------------------------- M-use
def test_use_off_is_byte_identical_to_the_old_recall():
    ref = _ref_module()
    rng = random.Random(0)
    old, new = _fake(ref.Hippocampus), _fake(Hippocampus)
    n = 6
    for _ in range(n):
        s, age, perm = rng.choice([0.5, 1.0, 2.0, 6.0]), rng.uniform(0, 5) * DAY, rng.random() < 0.2
        old.episodes.append(_ep(ref.Episode, s, T0 - age, perm))
        new.episodes.append(_ep(Episode, s, T0 - age, perm))
    assert new.USE_GATE is None and not new.FREEZE_USE
    for t in range(200):
        now = T0 + t * 1800.0
        clock.set(lambda now=now: now)
        sims = [rng.uniform(-0.2, 0.6) for _ in range(n)]
        kw = {"question": rng.random() < 0.7, "exclude": rng.sample(range(n), rng.choice([0, 0, 1, 2]))}
        a, b = _recall(old, sims, **kw), _recall(new, sims, **kw)
        assert (a[0] is None) == (b[0] is None)
        if a[0] is not None:
            assert old.episodes.index(a[0]) == new.episodes.index(b[0])
        assert a[1] == b[1]
        assert _state(old) == _state(new)
    assert new.last_fetch is None            # off: nothing is logged


def _hub():
    h = _fake(Hippocampus)
    h.episodes = [_ep(Episode, 6.0, T0), _ep(Episode, 1.0, T0 - 2 * DAY)]     # H (strong, fresh), R (weak, 2 days old)
    return h


def test_hub_case_the_gate_denies_the_reward():
    h = _hub()
    ep, _ = _recall(h, [0.20, 0.30])
    assert ep is h.episodes[0] and h.episodes[0].strength == 6.5 and h.episodes[0].uses == 1      # off: the hub is rewarded
    h = _hub()
    h.USE_GATE = "raw_win"
    ep, _ = _recall(h, [0.20, 0.30])
    assert ep is h.episodes[0]                                       # fetched as before ...
    assert _state(h) == [(6.0, 0, T0), (1.0, 0, T0 - 2 * DAY)]       # ... but not rewarded, decay clock not reset
    lf = h.last_fetch
    assert lf["idx"] == 0 and lf["raw_winner"] == 1 and lf["weighted_only_winner"] and not lf["confirmed"]
    assert lf["sim_raw"] == pytest.approx(0.20) and lf["margin_raw"] == pytest.approx(-0.10)


def test_clear_raw_winner_is_boosted():
    h = _fake(Hippocampus)
    h.USE_GATE = "raw_win"
    h.episodes = [_ep(Episode), _ep(Episode)]
    clock.set(lambda: T0 + 5.0)
    ep, _ = _recall(h, [0.40, 0.30])
    assert ep is h.episodes[0] and (ep.strength, ep.uses, ep.last_used) == (1.5, 1, T0 + 5.0)
    assert h.last_fetch["confirmed"] and h.last_fetch["margin_raw"] == pytest.approx(0.10)


def test_boundary_and_exclude():
    def run(sims, **kw):
        h = _fake(Hippocampus)
        h.USE_GATE, h.MARGIN_USE = "raw_win", 0.25
        h.episodes = [_ep(Episode) for _ in sims]
        ep, _ = _recall(h, sims, **kw)
        return h, ep
    h, ep = run([0.75, 0.5])
    assert ep is h.episodes[0] and h.episodes[0].uses == 1             # margin exactly MARGIN_USE: granted
    h, ep = run([0.75, 0.625])
    assert ep is h.episodes[0] and h.episodes[0].uses == 0             # 0.125 < 0.25: weak retrieval
    h, ep = run([0.9, 0.5, 0.1], exclude=(0,))
    assert ep is h.episodes[1] and h.episodes[1].uses == 1             # the excluded 0.9 is not the raw runner-up
    assert h.last_fetch["raw_winner"] == 1 and h.last_fetch["margin_raw"] == pytest.approx(0.4)


def test_freeze_use_returns_the_same_fetch_and_changes_nothing():
    rng = random.Random(1)
    for _ in range(50):
        n = 5
        mk = lambda: [_ep(Episode, rng2.choice([0.5, 1.0, 3.0]), T0 - rng2.uniform(0, 3) * DAY) for _ in range(n)]  # noqa: E731
        seed = rng.random()
        rng2 = random.Random(seed)
        a = _fake(Hippocampus)
        a.episodes = mk()
        rng2 = random.Random(seed)
        b = _fake(Hippocampus)
        b.episodes = mk()
        b.FREEZE_USE = True
        sims = [rng.uniform(-0.2, 0.6) for _ in range(n)]
        before = _state(b)
        (ea, sa), (eb, sb) = _recall(a, sims), _recall(b, sims)
        assert sa == sb and (ea is None) == (eb is None)
        if ea is not None:
            assert a.episodes.index(ea) == b.episodes.index(eb)
        assert _state(b) == before                                   # frozen: nothing changed
        assert b.last_fetch is not None


def test_toy_loop_gate_bounds_strength_and_off_does_not():
    n, rng = 20, random.Random(3)
    stream = []
    for _ in range(400):
        s = [rng.uniform(0.0, 0.3) for _ in range(n)]
        s[0] = 0.30 + rng.uniform(-0.01, 0.01)           # the hub: always near-familiar, almost never the raw winner
        s[rng.randrange(1, n)] = 0.33
        stream.append(s)

    def run(gate):
        h = _fake(Hippocampus)
        h.USE_GATE = gate
        h.episodes = [_ep(Episode, 3.0 if i == 0 else 1.0) for i in range(n)]
        confirmed = [0] * n
        for s in stream:
            ep, _ = _recall(h, s)
            if ep is not None:
                i = h.episodes.index(ep)
                top = sorted(range(n), key=lambda j: -s[j])
                if top[0] == i and s[i] - s[top[1]] >= h.MARGIN_USE:
                    confirmed[i] += 1
        return h, confirmed
    on, c_on = run("raw_win")
    assert all(e.strength <= (3.0 if i == 0 else 1.0) + 0.5 * c_on[i] + 1e-9 for i, e in enumerate(on.episodes))
    off, c_off = run(None)
    assert any(e.strength > (3.0 if i == 0 else 1.0) + 0.5 * c_off[i] + 1e-9 for i, e in enumerate(off.episodes))
    assert max(e.strength for e in off.episodes) > max(e.strength for e in on.episodes)


def test_brain_plumbing_sall_and_meta(tmp_path):
    home = str(tmp_path / "h")
    c = StubCortex()
    b = Brain(home, cortex=c, **KW, key="S_all", use_gate="raw_win")
    assert isinstance(b.hippocampus, KeyedHippocampus) and b.hippocampus.USE_GATE == "raw_win"
    infos = [b.hear(t)[1] for t in TALK]
    assert all("fetch" in i for i in infos)                          # logged on every turn with the gate on (None when no recall ran)
    assert any(i["fetch"] is not None for i in infos)
    b.save()
    assert json.load(open(os.path.join(home, "meta.json")))["use_gate"] == "raw_win"
    b2 = Brain(home, cortex=c, **KW)                                 # the home keeps its switch
    assert b2.hippocampus.USE_GATE == "raw_win" and "use_gate" not in b2.meta
    b2.save()
    assert json.load(open(os.path.join(home, "meta.json")))["use_gate"] == "raw_win"
    # off: no meta key, no info["fetch"]
    off = Brain(str(tmp_path / "o"), cortex=c, **KW)
    assert all("fetch" not in off.hear(t)[1] for t in TALK)
    off.save()
    assert "use_gate" not in json.load(open(str(tmp_path / "o" / "meta.json")))
    with pytest.raises(ValueError):
        Brain(str(tmp_path / "bad"), cortex=c, **KW, use_gate="nope")


# ---------------------------------------------------------------- M-scale
def _fake_plastic(stub, rank=2, C=4, hidden=6):
    p = PlasticCortex.__new__(PlasticCortex)
    torch.nn.Module.__init__(p)
    g = torch.Generator().manual_seed(0)
    p.cortex, p.layers, p.enabled, p.z = stub, [0, 1], True, {"emb.weight": torch.zeros(1)}
    p.A_k, p.B_k, p.A_v, p.B_v = (torch.nn.ParameterDict() for _ in range(4))
    for i in p.layers:
        p.A_k[str(i)] = torch.nn.Parameter(torch.randn(C, rank, generator=g))
        p.B_k[str(i)] = torch.nn.Parameter(torch.zeros(rank, hidden))
        p.A_v[str(i)] = torch.nn.Parameter(torch.randn(hidden, rank, generator=g))
        p.B_v[str(i)] = torch.nn.Parameter(torch.zeros(rank, C))
    return p


@pytest.fixture
def night(monkeypatch):
    """Fake adapters and a fake replay (B += 0.05 per night); `seen` records B and the norm at the start of each replay."""
    seen = []

    def fake_replay(plastic, base, eps, generic, **k):
        seen.append({"norm": plastic.adapter_norm(), "B": [b.detach().clone() for b in plastic.B_k.values()], "steps": k["steps"]})
        with torch.no_grad():
            for d in (plastic.B_k, plastic.B_v):
                for b in d.values():
                    b += 0.05
        return [(1.0, 0.0)] * k["steps"]

    def plastic(self):
        if self.cortex.plastic is None:
            self.cortex.plastic = _fake_plastic(self.raw_cortex)
        return self.cortex.plastic
    monkeypatch.setattr(bb, "replay_into_cortex", fake_replay)
    monkeypatch.setattr(Brain, "_plastic", plastic)
    return seen


def _live_nights(home, nights, **kw):
    """Brain with 3 tellings a night; returns (per-night losses, the Brain)."""
    ticks = iter(range(10 ** 6))
    clock.set(lambda: T0 + next(ticks))
    b = Brain(home, cortex=StubCortex(), **KW, **kw)
    out = []
    for n in range(nights):
        for t in (f"My cat number {n} is Pixel.", f"I drive a red Volvo {n}.", f"My sister {n} is Marguerite."):
            b.hear(t)
        b.cortex.consolidated_on = False       # the fake adapters hold no real weights: speak with the pretrained path
        out.append(b.sleep())
        b.save()
    return out, b


def _read(home, name):
    with open(os.path.join(home, name), "rb") as fh:
        return fh.read()


def test_scale_off_is_byte_identical(tmp_path, night):
    la, a = _live_nights(str(tmp_path / "a"), 2)
    lb, b = _live_nights(str(tmp_path / "b"), 2, sleep_downscale=None)
    assert la == lb
    assert _read(str(tmp_path / "a"), "meta.json") == _read(str(tmp_path / "b"), "meta.json")
    sa, sb = a.cortex.plastic.state_dict(), b.cortex.plastic.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    meta = json.loads(_read(str(tmp_path / "a"), "meta.json"))
    assert not {"adapter_budget", "norm_pre_night", "downscale_g"} & set(meta)


def test_downscale_halves_the_norm_and_leaves_A():
    p = _fake_plastic(None)
    with torch.no_grad():
        for d in (p.B_k, p.B_v):
            for b in d.values():
                b.copy_(torch.randn_like(b))
    n0, (a0, b0) = p.adapter_norm(), p.factor_norms()
    A = {k: v.clone() for k, v in p.A_k.items()}
    p.downscale(0.5)
    assert p.adapter_norm() == pytest.approx(0.5 * n0, rel=1e-5)
    a1, b1 = p.factor_norms()
    assert a1 == a0 and b1 == pytest.approx(0.5 * b0, rel=1e-5)
    assert all(torch.equal(A[k], p.A_k[k]) for k in A)


def test_budget_is_the_post_night_one_norm_and_a_small_norm_is_untouched(tmp_path, night):
    home = str(tmp_path / "h")
    ticks = iter(range(10 ** 6))
    clock.set(lambda: T0 + next(ticks))
    b = Brain(home, cortex=StubCortex(), **KW, sleep_downscale=1.0)
    b.hear("My cat is called Pixel.")
    b.cortex.consolidated_on = False
    b.sleep()
    p = b.cortex.plastic
    assert b.meta["adapter_budget"] == pytest.approx(p.adapter_norm()) and b.meta["adapter_budget"] > 0
    assert b.meta["downscale_g"] == 1.0 and b.meta["norm_pre_night"] == 0.0
    with torch.no_grad():                     # night 2 starts at or under the budget: nothing is cut
        for d in (p.B_k, p.B_v):
            for x in d.values():
                x.mul_(0.9)
    before = [x.detach().clone() for x in p.B_k.values()]
    b.hear("I drive a red Volvo.")
    b.sleep()
    assert all(torch.equal(x, y) for x, y in zip(night[-1]["B"], before))
    assert b.meta["downscale_g"] == 1.0 and b.meta["norm_pre_night"] == pytest.approx(0.9 * b.meta["adapter_budget"])


def test_cut_before_the_replay_loop(tmp_path, night):
    home = str(tmp_path / "h")
    ticks = iter(range(10 ** 6))
    clock.set(lambda: T0 + next(ticks))
    b = Brain(home, cortex=StubCortex(), **KW, sleep_downscale=1.0)
    b.hear("My cat is called Pixel.")
    b.cortex.consolidated_on = False
    b.sleep()
    p, budget = b.cortex.plastic, b.meta["adapter_budget"]
    with torch.no_grad():
        for d in (p.B_k, p.B_v):
            for x in d.values():
                x.mul_(3.0)
    assert p.adapter_norm() == pytest.approx(3 * budget, rel=1e-5)
    b.hear("I drive a red Volvo.")
    b.sleep(steps=1)
    assert night[-1]["steps"] == 1
    assert b.meta["norm_pre_night"] == pytest.approx(3 * budget, rel=1e-5)
    assert night[-1]["norm"] == pytest.approx(budget, rel=1e-5)        # the replay starts from the budget
    assert b.meta["downscale_g"] == pytest.approx(1 / 3, rel=1e-5) and b.meta["adapter_budget"] == budget


def test_budget_survives_reopen_and_a_budgetless_home_is_not_cut(tmp_path, night):
    home = str(tmp_path / "h")
    _, b = _live_nights(home, 2, sleep_downscale=1.0)
    budget = b.meta["adapter_budget"]
    b.release()
    clock.set(lambda: T0 + 5000.0)
    b2 = Brain(home, cortex=StubCortex(), **KW, sleep_downscale=1.0)
    assert b2.meta["adapter_budget"] == budget
    b2.release()
    # a home with adapters made with the switch OFF, opened with it ON: budget = current norm, no downscale that night
    home2 = str(tmp_path / "h2")
    _, c = _live_nights(home2, 2)
    c.release()
    assert "adapter_budget" not in json.loads(_read(home2, "meta.json"))
    log = []
    clock.set(lambda: T0 + 9000.0)
    d = Brain(home2, cortex=StubCortex(), **KW, sleep_downscale=1.0, sleep_log=log.append)
    n = d.cortex.plastic.adapter_norm()
    d.cortex.consolidated_on = False
    d.hear("A new telling.")
    d.sleep()
    assert d.meta["adapter_budget"] == pytest.approx(n) and d.meta["downscale_g"] == 1.0
    assert night[-1]["norm"] == pytest.approx(n)                       # untouched before the replay
    assert any("no budget" in x for x in log)
