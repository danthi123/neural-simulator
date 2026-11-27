# Capability: Lifelong learning
# Record: brain/research/2026-11-27-mix-night-prereg.md
# Expected-fail: none (tests of an instrument)
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
    for env in ({}, {"POOL_NODE": "0"}, {"BRAIN_ALLOW_W3090": "0"}):
        with pytest.raises(SystemExit):
            MN.guard(env)
    MN.guard({"POOL_NODE": "1"})
    MN.guard({"BRAIN_ALLOW_W3090": "1"})
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


BANK = {s: 0.2 for s in MN.SEEDS}       # the lives' banked n13 KL (K-sane reads it)


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
    v = MN.verdict(t, start_of(), BANK)
    assert all(c["status"].startswith("INCONCLUSIVE: A0r differs") for c in v["candidates"].values())
    assert v["placebo"]["status"].startswith("INCONCLUSIVE: A0r differs")
    t["A0r"][320][13]["kl"] = 0.2 * (1 + 5e-4)
    assert MN.verdict(t, start_of(), BANK)["candidates"]["A3"]["status"] == "PASS"


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


PROV = {"host": "node-a", "commit": "c1", "device": "cpu", "precision": "fp32", "model": "m15", "offframe_sha256": "h"}
HOSTS = {319: "node-a", 320: "node-b", 321: "node-c"}


def write_units(d, t=None):
    import json
    t = tab_of() if t is None else t
    def w(arm, s, n, r):
        MN.atomic_write(MN.unit_path(d, arm, s, n),
                        {**copy.deepcopy(r), "arm": arm, "seed": s, "night": n, **PROV, "host": HOSTS[s]})
    for arm, by in t.items():
        for s, nights in by.items():
            for n, r in nights.items():
                w(arm, s, n, r)
    for s in MN.SEEDS:
        w("start", s, 12, rd(0.1))
        for arm in ("A5s", "A5f"):
            w(arm, s, 15, rd(0.12))
    lives = []
    for s in MN.SEEDS:
        f = os.path.join(d, f"life_{s}.json")
        with open(f, "w") as fh:
            json.dump({"seed": s, "nights": [{"line": "trunk", "night": 13, "lang_guard": {"kl": 0.2}}]}, fh)
        lives.append(f)
    return lives


def _edit(d, arm, s, n, **kw):
    import json
    u = MN.unit_path(d, arm, s, n)
    x = json.load(open(u))
    x.update(kw)
    MN.atomic_write(u, x)


def test_score_reads_unit_files(tmp_path):
    d = str(tmp_path)
    lives = write_units(d)
    v = MN.score(d, lives)
    assert v["candidates"]["A3"]["status"] == "PASS" and v["placebo"]["status"] == "CONTENT-DRIVEN"
    assert "mixed_provenance" not in v and v["instrument"]["void"] == []


@pytest.mark.parametrize("key,val", [("commit", "c2"), ("device", "cuda"), ("precision", "fp16"), ("model", "m04"),
                                     ("offframe_sha256", "h2"), ("host", "node-z"), ("commit", None)])
def test_score_mixed_provenance_each_key_alone(tmp_path, key, val):
    d = str(tmp_path)
    lives = write_units(d)
    _edit(d, "A1", 319, 15, **{key: val})
    v = MN.score(d, lives)
    assert v["mixed_provenance"]
    assert all(c["status"].startswith("INCONCLUSIVE: mixed provenance") for c in v["candidates"].values())
    assert v["placebo"]["status"].startswith("INCONCLUSIVE: mixed provenance")


def test_score_hosts_differ_by_seed_is_fine_but_not_within_a_seed(tmp_path):
    d = str(tmp_path)
    lives = write_units(d)
    assert "mixed_provenance" not in MN.score(d, lives)               # one host per seed, three hosts in all
    _edit(d, "start", 320, 12, host="node-a")
    assert MN.score(d, lives)["mixed_provenance"] == ["s320 units on several hosts ['node-a', 'node-b']"]


def test_score_without_lives_is_inconclusive(tmp_path):
    d = str(tmp_path)
    write_units(d)
    v = MN.score(d, [])
    assert all(c["status"].startswith("INCONCLUSIVE: K-sane unread") for c in v["candidates"].values())


# each instrument clause failing alone voids the screen (missing A0r, A0L, A0 n13, banked KL); a K-sane MISS does not
@pytest.mark.parametrize("drop,expect", [(("A0r", 320, 13), "INCONCLUSIVE: determinism unread"),
                                         (("A0L", 321, 13), "INCONCLUSIVE: K-sane unread"),
                                         (("A0", 319, 13), "INCONCLUSIVE: determinism unread"),
                                         (("bank", 320, None), "INCONCLUSIVE: K-sane unread")])
def test_missing_instrument_unit_is_inconclusive_not_skipped(drop, expect):
    t, bank = tab_of(), dict(BANK)
    arm, s, n = drop
    if arm == "bank":
        del bank[s]
    else:
        del t[arm][s][n]
    v = MN.verdict(t, start_of(), bank)
    assert all(c["status"].startswith(expect) for c in v["candidates"].values()), v["candidates"]["A3"]["status"]
    assert v["placebo"]["status"].startswith(expect)
    assert MN.verdict(tab_of(), start_of(), BANK)["candidates"]["A3"]["status"] == "PASS"     # the intact table passes


@pytest.mark.parametrize("n,nll,status", [(0, None, "INCONCLUSIVE: P2"), (4, 0.5, "INCONCLUSIVE: P2"), (5, 0.5, "PASS")])
def test_p2_empty_or_thin_stratum_never_passes(n, nll, status):
    for arm in ("A3", "A0"):
        t = tab_of()
        t[arm][320][15]["ret"]["recent"] = {"n": n, "nll": nll, "under1": None}
        v = MN.candidate_verdict(t, "A3")
        assert v["status"].startswith(status), (arm, v["status"])
        assert v["P2"] is (status == "PASS")
    t = tab_of()
    del t["A3"][319][15]["ret"]["old"]                                  # stratum absent from the record
    assert MN.candidate_verdict(t, "A3")["status"].startswith("INCONCLUSIVE: P2")


def test_p2_unread_does_not_hide_a_kill():
    t = tab_of()
    for s in (319, 320):
        t["A1"][s][15]["kl"] = 0.45
    t["A1"][321][15]["ret"]["own"] = {"n": 0, "nll": None, "under1": None}
    assert MN.candidate_verdict(t, "A1")["status"] == "KILL"


# ---------------------------------------------------------------- provenance, resume, off-frame set, strata, refusals
def test_provenance_reads_commit_file_on_pool_node(tmp_path):
    (tmp_path / "COMMIT").write_text("abc123\n")
    p = MN.provenance("cpu", "fp32", "m", env={"POOL_NODE": "1"}, repo=str(tmp_path))
    assert p["commit"] == "abc123" and p["commit_source"] == "COMMIT" and p["precision"] == "fp32" and p["model"] == "m"
    (tmp_path / "COMMIT").write_text("")
    with pytest.raises(SystemExit):
        MN.provenance("cpu", env={"POOL_NODE": "1"}, repo=str(tmp_path))
    os.remove(tmp_path / "COMMIT")
    with pytest.raises(SystemExit):
        MN.provenance("cpu", env={"POOL_NODE": "1"}, repo=str(tmp_path))
    g = MN.provenance("cpu", env={})
    assert g["commit_source"] == "git" and re.fullmatch(r"[0-9a-f]{40}", g["commit"])
    assert MN.precision_of("cpu", "fp16") == "fp32" and MN.precision_of("cuda", "fp16") == "fp16"


def test_resume_refuses_another_provenance(tmp_path):
    u = str(tmp_path / "u.json")
    prov = {k: v for k, v in PROV.items() if k != "offframe_sha256"}
    MN.atomic_write(u, {"arm": "A0", **PROV})
    assert MN._resume(u, prov, "h")["arm"] == "A0"
    with pytest.raises(SystemExit):
        MN._resume(u, {**prov, "commit": "c2"}, "h")
    with pytest.raises(SystemExit):
        MN._resume(u, prov, "other")


def test_offframe_set_is_registered_frozen_and_unframed():
    import hashlib
    import json
    raw = open(MN.OFFFRAME_FILE, "rb").read()
    assert hashlib.sha256(raw).hexdigest() == MN.OFFFRAME_SHA256
    texts = json.loads(raw)
    assert len(texts) == 12 and all(isinstance(t, str) for t in texts)
    assert not [t for t in texts if "User:" in t or "Assistant:" in t]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    blob = ""
    for f in glob.glob(os.path.join(root, "eval", "life_bank*.py")) + [os.path.join(root, "eval", "life_heldout.py")]:
        blob += open(f).read().lower()
    grams = lambda t: {" ".join(w[i:i + 6]) for w in [re.findall(r"[a-z']+", t.lower())] for i in range(len(w) - 5)}
    bank_words = " ".join(re.findall(r"[a-z']+", blob))
    assert not [g for t in texts for g in grams(t) if g in bank_words]


def test_ref_cache_on_demand_equals_cached():
    big = GEN + [f"User: extra keep {i}\n\n" for i in range(80)]           # > REF_CACHE_MAX: on demand by default
    lc, pc = run(generic=big, ref_cache=True)
    ld, pd = run(generic=big)
    lf, pf = run(generic=big, ref_cache=False)
    assert lc == ld == lf and torch.equal(pc.B, pd.B) and torch.equal(pc.A, pf.A)
    assert run(ref_cache=False)[0] == run()[0]                          # the shipped size too
    p = Fake()
    MN.mix_sleep(p, None, EPS, big, steps=3, lr=1e-2, seed=1)
    assert p.enabled is True


def test_foreign_check(tmp_path):
    with pytest.raises(SystemExit):
        MN.foreign_check(None)
    for k in (13, 14):
        (tmp_path / f"n{k}").mkdir()
        (tmp_path / f"n{k}" / "episodes.json").write_text("[]")
    with pytest.raises(SystemExit):
        MN.foreign_check(str(tmp_path))
    (tmp_path / "n15").mkdir()
    (tmp_path / "n15" / "episodes.json").write_text("[]")
    assert sorted(MN.foreign_check(str(tmp_path))) == [13, 14, 15]


def test_strata_counts_from_store_only(tmp_path):
    import json
    day = 86400.0
    for k in (12, 13, 14, 15):
        d = tmp_path / f"n{k}"
        d.mkdir()
        now = 100 * day + k * day
        rows = [{"text": f"e{i}", "created": now - a * day} for i, a in enumerate([0.5] * 6 + [2.0] * 3 + [4.0] * 2 + [8.0] * 30)]
        (d / "episodes.json").write_text(json.dumps(rows))
        (d / "meta.json").write_text(json.dumps({"last_sleep": now}))
    c = MN.strata_counts(str(tmp_path))
    assert c[15] == {"own": 6, "recent": 3, "mid": 2, "old": 20}
    out = MN.main(["strata", "--snaps", str(tmp_path)])
    assert out["p2_short_at_n15"] == ["recent"] and out["ok"] is False


def test_run_refuses_before_any_model_load(tmp_path, monkeypatch):
    monkeypatch.setenv("POOL_NODE", "1")
    base = ["run", "--seed", "319", "--life", "x", "--snaps", str(tmp_path / "none"), "--out-dir", str(tmp_path / "o"),
            "--offframe", MN.OFFFRAME_FILE, "--offframe-sha256", MN.OFFFRAME_SHA256]
    with pytest.raises(SystemExit, match="foreign"):
        MN.main(base + ["--arm", "A5f"], cortex=object())
    with pytest.raises(SystemExit, match="does not hold seed 319"):
        MN.main(base + ["--arm", "A0"], cortex=object())
    with pytest.raises(SystemExit, match="registered"):
        MN.main(base[:-1] + ["0" * 64, "--arm", "A0"], cortex=object())


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


# ---------------------------------------------------------------- pre-data review fixes (2026-11-27)
def _snap_tree(root, nights=(12, 13, 14, 15), bank=True):
    import json
    for k in nights:
        d = root / f"n{k}"
        d.mkdir(parents=True)
        for f in ("episodes.json", "keys.pt", "cortex_adapters.pt"):
            (d / f).write_text("[]")
        (d / "meta.json").write_text(json.dumps({"sleeps": k, "last_sleep": 1.0}))
        if bank:
            (d / "keep_bank.json").write_text("[]")


def test_chain_dir_refuses_a_snapshot_without_its_keep_bank(tmp_path):
    _snap_tree(tmp_path / "s")
    ad = tmp_path / "ad.pt"
    ad.write_text("x")
    d = MN.chain_dir(str(tmp_path / "s"), 13, str(ad), str(tmp_path / "w"), "t")
    assert os.path.islink(os.path.join(d, "keep_bank.json"))
    os.remove(tmp_path / "s" / "n14" / "keep_bank.json")
    with pytest.raises(SystemExit, match="keep_bank.json"):
        MN.chain_dir(str(tmp_path / "s"), 14, str(ad), str(tmp_path / "w"), "t")


def test_run_refuses_a_missing_keep_bank_before_model_load(tmp_path, monkeypatch):
    monkeypatch.setenv("POOL_NODE", "1")
    _snap_tree(tmp_path / "s")
    os.remove(tmp_path / "s" / "n15" / "keep_bank.json")
    with pytest.raises(SystemExit, match="does not hold seed 319"):
        MN.main(["run", "--seed", "319", "--life", "x", "--snaps", str(tmp_path / "s"), "--out-dir", str(tmp_path / "o"),
                 "--offframe", MN.OFFFRAME_FILE, "--offframe-sha256", MN.OFFFRAME_SHA256, "--arm", "A0"], cortex=object())


def test_strata_flags_a_missing_keep_bank(tmp_path):
    import json
    _snap_tree(tmp_path)
    for k in (12, 13, 14, 15):
        rows = [{"created": 1.0 - 86400 * a} for a in (0.1,) * 6 + (2,) * 6 + (4,) * 6 + (6,) * 6]
        (tmp_path / f"n{k}" / "episodes.json").write_text(json.dumps(rows))
    os.remove(tmp_path / "n13" / "keep_bank.json")
    out = MN.main(["strata", "--snaps", str(tmp_path)])
    assert out["no_keep_bank"] == [13] and out["ok"] is False


@pytest.mark.parametrize("r_kl,ok,rc", [(0.2, True, None), (0.2 * (1 + 0.9e-3), True, None),
                                        (0.2 * (1 + 1.1e-3), False, 3), (None, None, 3)])
def test_determinism_pair_and_detcheck_exit(tmp_path, r_kl, ok, rc):
    t = tab_of()
    if r_kl is None:
        del t["A0r"][320][13]
    else:
        t["A0r"][320][13]["kl"] = r_kl
    assert MN.determinism_pair(t, 320)["ok"] is ok
    d = str(tmp_path)
    write_units(d, t)
    if rc is None:
        assert MN.main(["detcheck", "--dir", d, "--seed", "320"])["ok"] is True
    else:
        with pytest.raises(SystemExit) as e:
            MN.main(["detcheck", "--dir", d, "--seed", "320"])
        assert e.value.code == 3


def test_p1_change_diag_is_report_only():
    t = tab_of()                                       # A0 n15 0.40, X 0.20, start 0.10, noise 0.01
    v = MN.verdict(t, start_of(), BANK)
    dg = v["candidates"]["A3"]["p1_change_diag"]
    assert dg["status"] == "READ" and dg["cut_of_change"] == pytest.approx(1 - 0.10 / 0.30)
    assert v["candidates"]["A3"]["status"] == "PASS"
    t2 = tab_of(a0=0.40, noise=0.2, x=0.20)            # A0's own drift 0.30 not above 2 x noise 0.2 -> UNREAD, KILL stands
    v2 = MN.verdict(t2, start_of(), BANK)["candidates"]["A3"]
    assert v2["p1_change_diag"]["status"].startswith("UNREAD") and v2["status"] == "KILL"
    t3 = tab_of()
    del t3["A0e"][321][15]
    assert MN.p1_change_diag(t3, start_of(), "A3")["status"].startswith("UNREAD: missing")


def test_threads_is_a_provenance_key(tmp_path):
    assert "threads" in MN.PROV_KEYS and MN.provenance("cpu", env={})["threads"] == torch.get_num_threads()
    d = str(tmp_path)
    lives = write_units(d)
    _edit(d, "A2", 321, 15, threads=1)
    assert any(r.startswith("threads differs") for r in MN.score(d, lives)["mixed_provenance"])


# ---------------------------------------------------------------- run_node.sh driven on stubs (no model, no node)
RUN_NODE = os.path.join(os.path.dirname(MN.__file__), "results", "2026-11-27-mix-night", "run_node.sh")
SHIP = os.path.join(os.path.dirname(RUN_NODE), "ship.sh")


def _node(tmp_path, fail_arm="", det_rc=0, total_gb=64):
    """A fake pool node under HOME=tmp_path: route_repo with COMMIT, stub mem_ok / memcap / python / GNU time."""
    R = tmp_path / "brain-lane" / "route_repo"
    (R / "tools").mkdir(parents=True)
    (R / "_scratch" / "route" / "kr" / "snaps_s319").mkdir(parents=True)
    (R / "COMMIT").write_text("c0ffee\n")
    for k in (13, 14, 15):
        (R / "_scratch" / "route" / "kr" / "foreign_s320" / f"n{k}").mkdir(parents=True)
        (R / "_scratch" / "route" / "kr" / "foreign_s320" / f"n{k}" / "episodes.json").write_text("[]")
    (R / "tools" / "mem_ok.sh").write_text("exit 0\n")
    (R / "tools" / "memcap.sh").write_text('while [ "$1" != -- ]; do shift; done; shift; exec "$@"\n')
    py = tmp_path / "brain-lane" / "venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text(f'''#!/bin/bash
echo "$*" >> {tmp_path}/calls.txt
case "$*" in *detcheck*) echo '{{"ok": {"true" if det_rc == 0 else "false"}}}'; exit {det_rc};; esac
A=""; O=""; N=15
while [ $# -gt 0 ]; do case $1 in --arm) A=$2;; --out-dir) O=$2;; --max-night) N=$2;; esac; shift; done
[ "$A" = "{fail_arm}" ] && exit 1
for k in 13 14 15; do [ $k -le $N ] && echo '{{}}' > $O/u_${{A}}_s319_n$k.json; done
exit 0
''')
    tm = tmp_path / "gtime"
    tm.write_text('''#!/bin/bash
shift; OUT=$2; shift 2; "$@"; RC=$?
A=$(echo "$*" | sed 's/.*--arm \\([A-Za-z0-9]*\\).*/\\1/'); R=1000000; [ "$A" = A2 ] && R=3000000
printf 'Maximum resident set size (kbytes): %s\\nElapsed (wall clock) time (h:mm:ss or m:ss): 0:01\\n' $R > $OUT
exit $RC
''')
    for f in (py, tm):
        f.chmod(0o755)
    env = {**{k: v for k, v in os.environ.items() if k not in ("MEMCAP_GB", "THREADS", "SEED")},
           "HOME": str(tmp_path), "MN_TEST_TIME": str(tm), "MN_TEST_TOTAL_GB": str(total_gb)}
    return R / "_scratch" / "route" / "kr" / "mix_night", env


def _sh(args, env):
    import subprocess
    return subprocess.run(["bash", RUN_NODE] + args, env=env, capture_output=True, text=True, timeout=60)


def test_run_node_measure_reads_a0r_and_a2_and_sizes_the_cap_from_the_max(tmp_path):
    O, env = _node(tmp_path)
    p = _sh(["measure"], env)
    assert p.returncode == 0, p.stdout + p.stderr
    m = (O / "measure_s319.txt").read_text()
    assert "maxrss_kB 3000000" in m and "maxrss_A0r_kB 1000000" in m and "det ok" in m
    calls = (tmp_path / "calls.txt").read_text()
    assert calls.index("--arm A0r") < calls.index("--arm A0 ") < calls.index("detcheck") < calls.index("--arm A2")
    assert sorted(os.listdir(O)).count("u_A2_s319_n13.json") == 1 and not (O / "u_A2_s319_n14.json").exists()
    p = _sh(["run", "A0L"], env)
    assert p.returncode == 0, p.stdout
    assert "cap 5G" in (O / "done_s319.txt").read_text()          # ceil(1.3 x 3000000 kB / 2^20) + 1 = 4 + 1


def test_run_node_determinism_miss_writes_no_measure_and_run_refuses(tmp_path):
    O, env = _node(tmp_path, det_rc=3)
    p = _sh(["measure"], env)
    assert p.returncode == 3 and "THREADS=1" in p.stdout and not (O / "measure_s319.txt").exists()
    assert "--arm A2" not in (tmp_path / "calls.txt").read_text()
    assert _sh(["run"], env).returncode == 2


def test_run_node_small_node_needs_explicit_memcap(tmp_path):
    O, env = _node(tmp_path, total_gb=15)
    p = _sh(["measure"], env)
    assert p.returncode == 2 and "MEMCAP_GB" in p.stdout
    assert _sh(["measure"], {**env, "MEMCAP_GB": "11"}).returncode == 0


def test_run_node_threads_must_match_measure(tmp_path):
    O, env = _node(tmp_path)
    assert _sh(["measure"], {**env, "THREADS": "1"}).returncode == 0
    p = _sh(["run", "A0L"], {**env, "THREADS": "2"})
    assert p.returncode == 2 and "THREADS=1" in p.stdout


@pytest.mark.parametrize("arm,stops", [("A0L", True), ("A0", True), ("A0e", True), ("A0k", True), ("A1", False)])
def test_run_node_a_failed_baseline_stops_the_node(tmp_path, arm, stops):
    O, env = _node(tmp_path)
    assert _sh(["measure"], env).returncode == 0
    (tmp_path / "brain-lane" / "venv" / "bin" / "python").write_text(
        (tmp_path / "brain-lane" / "venv" / "bin" / "python").read_text().replace('[ "$A" = "" ]', f'[ "$A" = "{arm}" ]'))
    p = _sh(["run"], env)
    ran = [c.split("--arm ")[1].split()[0] for c in (tmp_path / "calls.txt").read_text().splitlines() if "--arm" in c]
    if stops:
        assert p.returncode == 1 and ran[-1] == arm and "baseline" in p.stdout
    else:
        assert p.returncode == 0 and ran[-1] == "A5f"


def test_ship_refuses_a_live_driver_and_logs_launches():
    s = open(SHIP).read()
    assert "brain.experiments|mn_run_node" in s
    assert ">/dev/null 2>&1" not in s and "launch_$M.log" in s
