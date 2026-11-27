"""Exact-verdict and plumbing tests for brain/experiments/mix_night_screen.py on synthetic records and a tiny fake adapter
forward (no model, no snapshots)."""
import copy
import glob
import os
import re

import pytest
import torch

from brain.experiments import mix_night_screen as MN
from brain.sleep import sleep as shipped_sleep

V = 40


class Pipe:
    def encode(self, text):
        return [ord(c) % V for c in text]


class Cx:
    pipe = Pipe()


class Fake(torch.nn.Module):
    """Differentiable stand-in for PlasticCortex: logits depend on the ids and on adapter parameters when enabled."""
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.cortex, self.enabled = Cx(), True
        self.A = torch.nn.Parameter(torch.randn(V, 4, generator=g) * 0.1)
        self.B = torch.nn.Parameter(torch.randn(4, V, generator=g) * 0.1)
        self.E = torch.randn(V, V, generator=g)

    def forward(self, ids, base):
        t = torch.tensor(ids)
        out = self.E[t]
        if self.enabled:
            out = out + self.A[t] @ self.B
        return out, None


EPS = [(f"episode number {i} says things about topic {i * 7}", 1.0 + (i % 3)) for i in range(9)]
GEN = [f"User: keep text {i} on a plain subject {i * 3}\n\n" for i in range(12)]


def run(loop="split", generic=GEN, steps=30, trace=None, **kw):
    p = Fake()
    losses = MN.mix_sleep(p, None, EPS, generic, steps=steps, lr=1e-2, kl_weight=3.0, seed=11, loop=loop, trace=trace, **kw)
    return losses, p


def test_legacy_loop_is_the_shipped_loop_exactly():
    p = Fake()
    ref = shipped_sleep(p, None, EPS, GEN, steps=30, lr=1e-2, kl_weight=3.0, seed=11)
    losses, q = run("legacy")
    assert losses == ref
    assert torch.equal(p.A, q.A) and torch.equal(p.B, q.B)
    assert float(q.B.abs().sum()) > 0 and losses[0] != losses[-1]       # not vacuous


def test_legacy_couples_keep_set_size_to_the_episode_stream_split_does_not():
    """randrange(n) rejects words >= n, so a keep-set size that is not a power of two re-times the shared generator and with
    it the episode draws (12 vs 40 here). Powers of two (the shipped 64 vs A3's 256) do not move the episode stream in the
    legacy loop but do change which keep text each word picks."""
    g12, g40 = GEN, GEN + [f"User: more keep {i}\n\n" for i in range(28)]
    t12, t40 = [], []
    run("legacy", g12, trace=t12)
    run("legacy", g40, trace=t40)
    assert [e for e, _ in t12] != [e for e, _ in t40]
    s12, s40 = [], []
    run("split", g12, trace=s12)
    run("split", g40, trace=s40)
    assert [e for e, _ in s12] == [e for e, _ in s40]
    g16, g64 = [f"User: k {i}\n\n" for i in range(16)], [f"User: k {i}\n\n" for i in range(64)]
    t16, t64 = [], []
    run("legacy", g16, trace=t16)
    run("legacy", g64, trace=t64)
    assert [e for e, _ in t16] == [e for e, _ in t64] and [k for _, k in t16] != [k for _, k in t64]


def test_split_streams_are_independent():
    base, a2, ce, ek, kk = [], [], [], [], []
    run(trace=base)
    run(trace=a2, keep_per_step=2)
    run(trace=ce, ce_keep_w=1.0)
    run(trace=ek, ep_add=1000)
    run(trace=kk, keep_add=1000)
    eps = lambda t: [e for e, _ in t]
    keeps = lambda t: [k for _, k in t]
    assert eps(base) == eps(a2) == eps(ce) == eps(kk)                  # no keep option moves the episode sequence
    assert keeps(base) == keeps(ce) == keeps(ek)                       # no episode option moves the keep sequence
    assert eps(ek) != eps(base) and keeps(kk) != keeps(base)
    assert all(len(k) == 2 for k in keeps(a2)) and all(len(k) == 1 for k in keeps(base))


def test_options_change_the_loss_and_defaults_do_not():
    l0, p0 = run()
    l1, p1 = run(ce_keep_w=1.0)
    l2, p2 = run(keep_per_step=2)
    l0b, p0b = run()
    assert l0 == l0b and torch.equal(p0.A, p0b.A)
    assert not torch.equal(p0.B, p1.B) and not torch.equal(p0.B, p2.B)


def test_legacy_refuses_arm_options_and_guards():
    with pytest.raises(ValueError):
        run("legacy", ep_add=1)
    with pytest.raises(ValueError):
        run("legacy", keep_per_step=2)
    with pytest.raises(ValueError):
        MN.mix_sleep(Fake(), None, EPS, GEN, steps=1, guard=["x"])
    with pytest.raises(ValueError):
        MN.mix_sleep(Fake(), None, EPS, [("t", [0])], steps=1)


def test_keep_set_is_nested_and_the_shipped_draw_is_its_prefix():
    import brain.brain as bb
    bank = [f"t{i}" for i in range(512)]
    ship = MN.shipped_subset(bank, 319, 12)
    import random
    assert ship == random.Random(319 + 12 + bb.KEEP_BANK_SEED_OFFSET).sample(bank, bb.KEEP_BANK_NIGHT)
    big = MN.keep_set(bank, 319, 12, 256)
    assert len(big) == 256 and len(set(big)) == 256 and big[:64] == ship
    assert MN.keep_set(bank, 319, 12, None) == ship and MN.keep_set(bank, 319, 12, 64) == ship
    assert MN.keep_set(bank, 319, 13, 256)[:64] != ship


def test_placebo_pool_excludes_and_assigns():
    pool, nex = MN.placebo_pool(["a\n\n", "b", "c", "d"], ["b", " c "], 5)
    assert sorted(pool) == ["a\n\n", "d"] and nex == 2
    out = MN.assign_placebo(5, pool)
    assert len(out) == 5 and all(not t.endswith("\n") for t in out) and set(out) == {"a", "d"}
    with pytest.raises(ValueError):
        MN.assign_placebo(2, [])
    assert MN.placebo_pool(["x", "y", "z"], [], 5)[0] == MN.placebo_pool(["x", "y", "z"], [], 5)[0]


def test_strata_and_sampling():
    assert [MN.stratum(a) for a in (0.0, 0.99, 1.0, 2.9, 3.0, 4.9, 5.0, 40.0, -0.2)] == \
        ["own", "own", "recent", "recent", "mid", "mid", "old", "old", "own"]
    now = 100 * 86400.0
    created = [now - d * 86400.0 for d in [0.2] * 30 + [2.0] * 5 + [9.0] * 30]
    s1, s2 = MN.sample_strata(created, now, 15), MN.sample_strata(created, now, 15)
    assert s1 == s2 and len(s1["own"]) == 20 and len(s1["recent"]) == 5 and s1["mid"] == [] and len(s1["old"]) == 20
    assert all(0 <= i < 30 for i in s1["own"]) and all(35 <= i for i in s1["old"])


def test_reply_flags():
    f = MN.reply_flags(["I'm sorry, I can't.", "Sure!", "My name is Nova.", "I don't know what you mean", "ok", "yes"])
    assert f == {"refusal": 2 / 6, "name": 1 / 6}
    assert MN.reply_flags([]) == {"refusal": 0.0, "name": 0.0}


def test_prompts_not_in_banks():
    assert len(MN.PROMPTS) == 24 and len(set(MN.PROMPTS)) == 24
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    blob = ""
    for f in glob.glob(os.path.join(root, "eval", "life_bank*.py")) + [os.path.join(root, "eval", "life_heldout.py")]:
        blob += open(f).read()
    assert not [q for q in MN.PROMPTS if q in blob]


def test_guard_needs_pool_node():
    for env in ({}, {"POOL_NODE": "0"}, {"BRAIN_ALLOW_W3090": "1"}):
        with pytest.raises(SystemExit):
            MN.guard(env)
    MN.guard({"POOL_NODE": "1"})
    with pytest.raises(SystemExit):
        MN.main(["run", "--seed", "319", "--arm", "A0", "--life", "x", "--snaps", "y", "--out-dir", "z",
                 "--offframe", "f", "--offframe-sha256", "0"])


def test_arm_table():
    assert set(MN.ARMS) == {"A0L", "A0", "A0r", "A0e", "A0k", "A1", "A2", "A3", "A4", "A5s", "A5f"}
    assert MN.arm_cfg("A0")["loop"] == "split" and MN.arm_cfg("A0L")["loop"] == "legacy"
    assert MN.arm_cfg("A0L")["nights"] == 1 and MN.arm_cfg("A0r")["nights"] == 1 and MN.arm_cfg("A3")["nights"] == 3
    with pytest.raises(ValueError):
        MN.arm_cfg("A9")


def test_estimate():
    e = MN.estimate(25, 10, 3)
    sleeps = 3 * (3 * 9 + 2)          # 9 three-night arms + A0L and A0r one night each, 3 seeds
    assert e["night_sleeps"] == sleeps == 87
    assert e["node_hours"] == round((87 * 35 + 3 * 10) / 60, 1) and e["wall_hours_on_nodes"] == round(e["node_hours"] / 3, 1)
    assert e["inputs"]["measured"] is False


# ---------------------------------------------------------------- exact verdicts on synthetic records
def rd(kl, off=0.10, ret=0.5, ref=0.0, name=0.0):
    return {"kl": kl, "off_kl": off, "ref": None, "refusal": ref, "name": name,
            "ret": {s: {"nll": ret, "n": 5, "under1": 1.0} for s in MN.P2_STRATA}}


def tab_of(a0=0.40, noise=0.01, x=0.20, **xkw):
    tab = {}
    for s in MN.SEEDS:
        tab.setdefault("A0", {})[s] = {13: rd(0.2), 15: rd(a0)}
        tab.setdefault("A0r", {})[s] = {13: rd(0.2)}
        tab.setdefault("A0L", {})[s] = {13: rd(0.2)}
        tab.setdefault("A0e", {})[s] = {15: rd(a0 + noise)}
        tab.setdefault("A0k", {})[s] = {15: rd(a0 - noise / 2)}
        for X in MN.CANDIDATES:
            tab.setdefault(X, {})[s] = {15: rd(x, **{"off": 0.05, **xkw})}
    return tab


def start_of(kl=0.10):
    return {s: rd(kl) for s in MN.SEEDS}


def test_pass():
    v = MN.candidate_verdict(tab_of(), "A3")
    assert v["status"] == "PASS" and v["P1"] and v["P1b"] and v["P2"] and v["P3"] and v["ratio"] == 0.5
    assert all(r["noise"] == pytest.approx(0.01) for r in v["per_seed"].values())


def test_p1_ratio_edge_and_noise_edge():
    # mean ratio exactly 0.75 passes P1; just above fails it (and, with every seed above the kill line, is a KILL only if
    # >= 2 seeds fail the per-seed clause; here the per-seed clause holds so it is INCONCLUSIVE)
    assert MN.candidate_verdict(tab_of(x=0.30), "A1")["status"] == "PASS"
    v = MN.candidate_verdict(tab_of(x=0.3001), "A1")
    assert not v["P1"] and v["status"] == "INCONCLUSIVE"
    # gap 0.02 against noise 0.01: needs > 2 x noise, so exactly equal fails
    v = MN.candidate_verdict(tab_of(a0=0.40, noise=0.01, x=0.38), "A1")
    assert not v["P1"] and v["status"] == "KILL"
    v = MN.candidate_verdict(tab_of(a0=0.40, noise=0.005, x=0.38), "A1")
    assert v["per_seed"][319]["p1_seed"]


def test_p1b_two_of_three():
    t = tab_of()
    t["A2"][319][15]["off_kl"] = 0.10                    # not below 0.85 x A0 off-frame on one seed
    assert MN.candidate_verdict(t, "A2")["status"] == "PASS"
    t["A2"][320][15]["off_kl"] = 0.10
    v = MN.candidate_verdict(t, "A2")
    assert not v["P1b"] and v["status"] == "INCONCLUSIVE"


def test_p2_and_p3_clauses():
    t = tab_of()
    t["A3"][321][15]["ret"]["old"]["nll"] = 0.5 + 0.10 + 1e-9
    assert not MN.candidate_verdict(t, "A3")["P2"]
    t = tab_of()
    t["A3"][321][15]["ret"]["old"]["nll"] = 0.5 + 0.10
    assert MN.candidate_verdict(t, "A3")["P2"]
    t = tab_of()
    t["A4"][320][15]["refusal"] = 0.05 + 1e-9
    assert not MN.candidate_verdict(t, "A4")["P3"]
    t["A4"][320][15]["refusal"] = 0.05
    t["A4"][320][15]["name"] = 0.0
    assert MN.candidate_verdict(t, "A4")["P3"]
    t["A4"][320][15]["ret"]["mid"] = {"nll": 99.0, "n": 1, "under1": 0.0}        # 'mid' is not a P2 stratum
    assert MN.candidate_verdict(t, "A4")["P2"]


def test_kill_when_p1_fails_on_two_seeds():
    t = tab_of()
    for s in (319, 320):
        t["A1"][s][15]["kl"] = 0.45
    assert MN.candidate_verdict(t, "A1")["status"] == "KILL"
    t = tab_of()
    t["A1"][319][15]["kl"] = 0.45
    assert MN.candidate_verdict(t, "A1")["status"] == "INCONCLUSIVE"


def test_missing_units_inconclusive():
    t = tab_of()
    del t["A0k"][321][15]
    assert MN.candidate_verdict(t, "A3")["status"].startswith("INCONCLUSIVE: missing")
    t = tab_of()
    del t["A3"][320]
    assert MN.candidate_verdict(t, "A3")["status"].startswith("INCONCLUSIVE: missing")


def test_determinism_gate_voids_candidates():
    t = tab_of()
    t["A0r"][320][13]["kl"] = 0.2 * (1 + 2e-3)
    v = MN.verdict(t, start_of())
    assert all(c["status"].startswith("INCONCLUSIVE: A0r differs") for c in v["candidates"].values())
    t["A0r"][320][13]["kl"] = 0.2 * (1 + 5e-4)
    assert MN.verdict(t, start_of())["candidates"]["A3"]["status"] == "PASS"


def test_k_sane_is_a_caveat_not_a_void():
    t = tab_of()
    v = MN.verdict(t, start_of(), banked={319: 0.2, 320: 0.2 / 1.2, 321: 0.2 / 1.3})
    ks = v["instrument"]["k_sane"]
    assert ks["A0:s319"] is True and ks["A0L:s320"] is True and ks["A0:s321"] is False
    assert v["candidates"]["A3"]["status"] == "PASS"


def placebo_tab(d_self, d_for, d_real=0.20, noise=0.01):
    t = tab_of(a0=0.10 + d_real, noise=noise)
    for s in MN.SEEDS:
        t["A5s"] = {**t.get("A5s", {}), s: {15: rd(0.10 + d_self[s])}}
        t["A5f"] = {**t.get("A5f", {}), s: {15: rd(0.10 + d_for[s])}}
    return t


def test_placebo_motion_content_bracketed():
    ds = lambda a, b, c: {319: a, 320: b, 321: c}
    v = MN.placebo_read(placebo_tab(ds(0.13, 0.13, 0.05), ds(0.2, 0.2, 0.2)), start_of())
    assert v["status"] == "MOTION-DOMINANT"                       # self >= 0.6 x real (0.12) on 2 seeds
    v = MN.placebo_read(placebo_tab(ds(0.13, 0.119, 0.13), ds(0.0, 0.0, 0.0)), start_of())
    assert v["status"] == "MOTION-DOMINANT"                       # 2 of 3 seeds is enough
    v = MN.placebo_read(placebo_tab(ds(0.119, 0.119, 0.119), ds(0.0, 0.0, 0.0)), start_of())
    assert v["status"] == "BRACKETED"                             # 0.595 x real: neither motion nor content
    v = MN.placebo_read(placebo_tab(ds(0.04, 0.045, 0.045), ds(0.045, 0.04, 0.045)), start_of())
    assert v["status"] == "CONTENT-DRIVEN"                        # both placebos <= 0.25 x real (0.05), clear of float edges
    v = MN.placebo_read(placebo_tab(ds(0.04, 0.05, 0.05), ds(0.2, 0.2, 0.2)), start_of())
    assert v["status"] == "BRACKETED"                             # self low but foreign high: the self bias cannot be ruled out
    v = MN.placebo_read(placebo_tab(ds(0.0, 0.0, 0.0), ds(0.0, 0.0, 0.0), d_real=0.015, noise=0.01), start_of())
    assert v["status"].startswith("UNREAD")                       # real drift within twice the noise


def test_placebo_missing_is_inconclusive():
    t = placebo_tab({s: 0.1 for s in MN.SEEDS}, {s: 0.1 for s in MN.SEEDS})
    del t["A5f"][319]
    assert MN.placebo_read(t, start_of())["status"].startswith("INCONCLUSIVE")


def test_score_reads_unit_files_and_flags_mixed_provenance(tmp_path):
    t = tab_of()
    for arm, by in t.items():
        for s, nights in by.items():
            for n, r in nights.items():
                MN.atomic_write(MN.unit_path(str(tmp_path), arm, s, n),
                                {**copy.deepcopy(r), "arm": arm, "seed": s, "night": n, "commit": "c1", "device": "cpu",
                                 "offframe_sha256": "h"})
    for s in MN.SEEDS:
        MN.atomic_write(MN.unit_path(str(tmp_path), "start", s, 12),
                        {**rd(0.1), "arm": "start", "seed": s, "night": 12, "commit": "c1", "device": "cpu",
                         "offframe_sha256": "h"})
        for arm in ("A5s", "A5f"):
            MN.atomic_write(MN.unit_path(str(tmp_path), arm, s, 15),
                            {**rd(0.12), "arm": arm, "seed": s, "night": 15, "commit": "c1", "device": "cpu",
                             "offframe_sha256": "h"})
    v = MN.score(str(tmp_path))
    assert v["candidates"]["A3"]["status"] == "PASS" and v["placebo"]["status"] == "CONTENT-DRIVEN"
    u = MN.unit_path(str(tmp_path), "A1", 319, 15)
    import json
    d = json.load(open(u))
    d["commit"] = "c2"
    MN.atomic_write(u, d)
    v = MN.score(str(tmp_path))
    assert v["mixed_provenance"] and v["candidates"]["A3"]["status"].startswith("INCONCLUSIVE: mixed")


def test_banked_kl_reads_trunk_lang_guard():
    res = {"nights": [{"line": "trunk", "night": 13, "lang_guard": {"kl": 0.3}},
                      {"line": "other", "night": 13, "lang_guard": {"kl": 9.0}}]}
    assert MN.banked_kl(res, 13) == 0.3 and MN.banked_kl(res, 14) is None


# ---------------------------------------------------------------- the real Brain path (tiny stand-in cortex)
def test_patched_brain_sleep_legacy_equals_unpatched_and_arms_change_it(tmp_path, monkeypatch):
    from brain import clock
    import brain.brain as bb
    from brain.tests import test_keep_bank as TK
    from brain.tests.test_use_scale import T0
    monkeypatch.setattr(bb.Brain, "_plastic", TK._plastic_of)
    monkeypatch.setattr(bb, "KEEP_BANK_N", 24)
    monkeypatch.setattr(bb, "KEEP_BANK_NIGHT", 5)

    def night(name, cfg=None, foreign=None):
        clock.set(lambda: T0)
        torch.manual_seed(3)
        b = TK._brain(str(tmp_path / name), keep_bank="prior")
        info = {}
        if cfg is None:
            b.sleep()
        else:
            with MN.patched(b, MN.arm_cfg(cfg), foreign, info):
                b.sleep()
        assert bb.replay_into_cortex.__module__ == "brain.sleep"            # restored
        b.save()
        clock.set(None)
        return TK._adapters(str(tmp_path / name)), info, b

    ref, _, _ = night("ref")
    leg, info, _ = night("legacy", "A0L")
    assert all(torch.equal(ref[k], leg[k]) for k in ref) and info["keep_n"] == 5
    a0, _, _ = night("a0", "A0")
    assert any(not torch.equal(ref[k], a0[k]) for k in ref)                  # split streams: a different (decoupled) night
    a3, info3, _ = night("a3", "A3")
    assert info3["keep_n"] == 24                                             # min(256, bank) extras: the whole 24-text bank
    pl, infop, _ = night("pl", "A5s")
    assert infop["placebo"] == "self" and infop["placebo_pool"] == 24 - 5
    pf, infof, _ = night("pf", "A5f", foreign=["foreign fact one", "foreign fact two"])
    assert infof["placebo"] == "foreign" and infof["placebo_pool"] == 2
