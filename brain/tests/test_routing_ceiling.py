# Capability: Episodic recall
# Record: brain/research/2026-11-25-routing-ceiling-prereg.md
# Expected-fail: none (harness plumbing on the stub cortex; no model weights)
"""routing_ceiling on the stub cortex: S0 is Hippocampus.similarities exactly, S_all is hippocampus_keys.centred_cos,
the DG code is binary top-k and the cached path equals the direct one, ranks are computed against the best right
episode, and on a stub life with night snapshots the ceiling run is sane (S0 top-1 = the replay's raw winner)."""
import json

import pytest
import torch

from brain import clock
from brain.brain import BIRTH
from brain.experiments import routing_ceiling as RC
from brain.experiments import routing_replay as RR
from brain.hippocampus import Hippocampus
from brain.hippocampus_keys import centred_cos
from brain.tests.test_brain_key_switch import StubCortex
from brain.tests.test_life import BKW, _env  # noqa: F401  (autouse fixture: fake night, 3 turns a day)
from brain.tests.test_routing_replay import CUES, TELLINGS, T


@pytest.fixture
def store():
    clock.set(lambda: T)
    cx = StubCortex()
    _, base = cx.read(BIRTH, None)
    hc = Hippocampus()
    for t in TELLINGS:
        hc.store(cx, base, t)
    yield hc, cx, base
    clock.set(None)


def test_S0_is_similarities(store):
    hc, cx, base = store
    K = torch.stack([e.key for e in hc.episodes]).float()
    for cue in CUES:
        _, st = cx.read(f"User: {cue}\n\n", base)
        q = cx.key_of(cx.matrix_delta(st, base))
        assert torch.allclose(RC.centred_cos_one(K, q), hc.similarities(cx, base, cue), atol=1e-6)


def test_S_all_matches_hippocampus_keys(store):
    hc, cx, base = store
    eps = hc.episodes
    _, st = cx.read(f"User: {CUES[0]}\n\n", base)
    dl = cx.matrix_delta(st, base)
    ls = [l for l, d in enumerate(dl) if d is not None]
    mine = sum(RC.centred_cos_one(torch.stack([e.delta[l].flatten().float() for e in eps]), dl[l].flatten().float())
               for l in ls) / len(ls)
    ref = centred_cos([torch.stack([e.delta[l].flatten().float() for e in eps]) for l in ls],
                      [dl[l].flatten().float()[None] for l in ls])[0]
    assert torch.allclose(mine, ref, atol=1e-5)


def test_dg_code_binary_and_cached_equals_direct(store):
    hc, cx, base = store
    K = torch.stack([e.key for e in hc.episodes]).float()
    _, st = cx.read(f"User: {CUES[1]}\n\n", base)
    q = cx.key_of(cx.matrix_delta(st, base))
    idx, sign = RC.dg_map(K.shape[1], 256, 42, fan_in=8)
    mu, h = RC.dg_store(K, idx, sign)
    for f in (0.02, 0.05):
        code = RC._binary(h, f)
        assert set(code.unique().tolist()) <= {0.0, 1.0} and (code.sum(1) == round(f * 256)).all()
        a, b = RC.dg_sims_cached(h, mu, q, idx, sign, f), RC.dg_sims(K, q, idx, sign, f)
        assert torch.equal(a, b) and (a >= 0).all() and (a <= 1).all()
    i2, s2 = RC.dg_map(K.shape[1], 256, 42, fan_in=8)
    assert torch.equal(idx, i2) and torch.equal(sign, s2)          # the map is fixed by its seed


def test_rank_of():
    s = torch.tensor([0.1, 0.9, 0.5, 0.7])
    assert RC.rank_of(s, [1]) == {"rank": 1, "margin": 0.0, "top": 1}
    r = RC.rank_of(s, [2, 0])
    assert r["rank"] == 3 and r["margin"] == pytest.approx(0.4) and r["top"] == 1
    tie = torch.tensor([0.5, 0.9, 0.9, 0.1])          # a wrong episode tied with the right one counts against it
    assert RC.rank_of(tie, [2])["rank"] == 2 and RC.rank_of(tie, [1, 2])["rank"] == 1


def test_sall_live_is_centred_concat_of_normalised_blocks(store):
    hc, cx, base = store
    eps = hc.episodes
    _, st = cx.read(f"User: {CUES[2]}\n\n", base)
    dl = cx.matrix_delta(st, base)
    ls = [l for l, d in enumerate(dl) if d is not None]
    blocks = {l: torch.stack([e.delta[l].flatten().float() for e in eps]) for l in ls}
    live = RC.sall_live(RC.live_store(blocks), dl, len(eps))
    K = torch.cat([torch.nn.functional.normalize(blocks[l], dim=1) for l in ls], dim=1)
    q = torch.cat([torch.nn.functional.normalize(dl[l].flatten().float(), dim=0) for l in ls])
    assert torch.allclose(live, RC.centred_cos_one(K, q), atol=2e-3)   # fp16 cache


def test_ceiling_on_a_stub_life(tmp_path, monkeypatch):
    from brain.experiments import life_kw

    monkeypatch.setattr(RC, "LAYERS", (1,))
    monkeypatch.setattr(RC, "DG_E", (128,))
    monkeypatch.setattr(RC, "DG_SEEDS", (42, 43))
    snaps = tmp_path / "snaps"
    argv = ["--seed", "7", "--days", "3", "--full-probe-nights", "1", "--frozen-probe", "--homes", str(tmp_path / "h"),
            "--out", str(tmp_path / "o" / "life_s7"), "--night-snapshots", str(snaps)]
    life_kw.main(argv, cortex=StubCortex(), brain_kw=BKW)
    res = json.load(open(str(tmp_path / "o" / "life_s7") + ".json"))
    rep = RR.replay(res, str(snaps), StubCortex(), str(tmp_path / "work"), weekend=False, nights=range(1, 4),
                    trunk=str(tmp_path / "h" / "trunk"), log=lambda *a: None)
    out = RC.run(res, rep, str(snaps), str(tmp_path / "h" / "trunk"), StubCortex(), str(tmp_path / "w2"),
                 log=lambda *a: None)
    nights = [n for n in out["nights"] if not n.get("void")]
    assert nights and all(n["sane"] for n in nights)
    told = [r for n in nights for r in n["rows"]]
    assert told, "the stub life should have told rows"
    for r in told:
        if r["right"]:
            assert {"S0", "S1", "S_all", "S0drift"} <= set(r["keys"]) and any(k.startswith("DG:") for k in r["keys"])
            assert r["keys"]["S0"]["rank"] > 1 or r["keys"]["S0"]["margin"] == 0.0
            assert "S_all_live" in r["keys"]
    tab = RC.table([dict(out, seed=s) for s in (1, 2, 3)])
    assert tab["verdict"]["K_sane"] == "ok"
    assert tab["pooled"]["S0"]["n"] == 3 * sum(1 for r in told if r["right"])
