# Capability: Language
# Record: brain/research/2026-11-27-spiking-twin-e1-prereg.md
# Expected-fail: none (synthetic records and a tiny random cortex; no model weights)
"""E1 harness: host guard; the registered verdict on synthetic records (GO / carry / FLUENT_ONLY / NEGATIVE, and every
control clause failing alone gives VOID with exactly that reason); arms a and b draw the same data; a resumed training run
equals an uninterrupted one; eval units write atomically, carry provenance and are skipped when present; estimate sums."""
import copy
import json
import math
import os

import numpy as np
import pytest
import torch

from brain.experiments import spiking_twin_e1 as E
from brain.tests.test_source_tag import _Cortex

SEEDS = E.SEEDS


def _seed_rec(arm):
    ppl = {"wt103_nll": 3.0, "chat_nll": 2.0, "wt103_unigram_nll": 7.0}
    sleep = {a: {"recall": 0, "false_recall": 2} for a in E.SLEEP_ARMS}
    sleep["sleep_true"] = {"recall": 4, "false_recall": 2}
    ladder = {"in_context": 6, "reinst_4.0": 11, "sham_4.0": 1, "same_words": 12}
    if arm in ("a", "a2"):
        ppl = {"wt103_nll": 3.3, "chat_nll": 2.3, "wt103_nll_spikes_off": 3.6, "mean_count": 1.1, "silent_share": 0.2}
        sleep["sleep_true"] = {"recall": 3, "false_recall": 2}
        ladder = dict(ladder, **{"reinst_4.0": 9})
    if arm == "b":
        ppl = {"wt103_nll": 5.0, "chat_nll": 4.0, "wt103_nll_spikes_off": 5.2, "mean_count": 1.0, "silent_share": 0.3}
        sleep["sleep_true"] = {"recall": 1, "false_recall": 2}
    return {"ppl": ppl, "sleep": sleep, "ladder": ladder}


def _recs():
    return {arm: {s: _seed_rec(arm) for s in SEEDS} for arm in E.ARMS}


def _set(recs, arm, path, value):
    for s in SEEDS:
        d = recs[arm][s]
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = value
    return recs


def test_host_guard():
    with pytest.raises(SystemExit):
        E.host_guard({})
    with pytest.raises(SystemExit):
        E.host_guard({"POOL_NODE": "0", "BRAIN_ALLOW_W3090": "yes"})
    E.host_guard({"POOL_NODE": "1"})
    E.host_guard({"BRAIN_ALLOW_W3090": "1"})


def test_verdict_go_on_a():
    v = E.verdict(_recs(), SEEDS)
    assert v["verdict"] == "GO" and v["carry"] == "a" and v["claim"] and not v["stop_rule"] and v["reasons"] == []
    assert v["f2_assessable"] and v["b_status"] == []
    assert math.isclose(v["arms"]["a"]["ratio_wt103"], math.exp(0.3))
    assert math.isclose(v["arms"]["a"]["init_gap"], 1.7)


def test_verdict_carry_a2_only_when_chat_gap_smaller_by_005():
    r = _set(_recs(), "a2", ("ppl", "chat_nll"), 2.25)     # 0.05 smaller: carry a2
    assert E.verdict(r, SEEDS)["carry"] == "a2"
    r = _set(_recs(), "a2", ("ppl", "chat_nll"), 2.26)     # 0.04 smaller: stays a
    assert E.verdict(r, SEEDS)["carry"] == "a"


def test_verdict_go_on_a2_when_a_fails():
    r = _set(_recs(), "a", ("ppl", "wt103_nll"), 3.5)      # ratio exp(0.5) = 1.65 > 1.5
    r = _set(r, "a", ("ppl", "wt103_nll_spikes_off"), 3.8)
    v = E.verdict(r, SEEDS)
    assert v["verdict"] == "GO" and v["carry"] == "a2" and not v["arms"]["a"]["F1"]


def _both(path, value):
    return lambda r: [_set(r, X, path, value) for X in ("a", "a2")]


@pytest.mark.parametrize("clause,edit", [
    ("S", _both(("sleep", "sleep_true", "recall"), 1)),                 # < 0.5 x 4
    ("S", _both(("sleep", "sleep_shuffled", "recall"), 2.01)),        # margin 0.99 < 1.0 (and shuffled > 1.5)
    ("S", _both(("sleep", "sleep_true", "false_recall"), 2.51)),       # false recall > teacher 2 + 0.5
    ("R", _both(("ladder", "reinst_4.0"), 7)),
    ("R", _both(("ladder", "sham_4.0"), 3.01)),                        # the twin's own wrong-memory sham > 3
])
def test_verdict_fluent_only(clause, edit):
    r = _recs()
    edit(r)
    v = E.verdict(r, SEEDS)
    assert v["verdict"] == "FLUENT_ONLY" and v["stop_rule"] and not v["arms"]["a"][clause]


def test_verdict_negative_and_init_gap_bar():
    r = _recs()
    for X in ("a", "a2"):
        _set(r, X, ("ppl", "chat_nll"), 2.5)               # ratio exp(0.5) > 1.5
    v = E.verdict(r, SEEDS)
    assert v["verdict"] == "NEGATIVE" and v["stop_rule"]
    r = _set(_recs(), "b", ("ppl", "wt103_nll"), 4.2)      # init gap 0.9 < 1.0 for both
    r = _set(r, "b", ("ppl", "wt103_nll_spikes_off"), 4.4)
    v = E.verdict(r, SEEDS)
    assert v["verdict"] == "NEGATIVE" and not v["arms"]["a"]["F2"] and v["arms"]["a"]["F1"]


def test_verdict_sleep_must_beat_shuffled():
    r = _recs()
    for X in ("a", "a2"):
        _set(r, X, ("sleep", "sleep_shuffled", "recall"), 3)
    assert E.verdict(r, SEEDS)["verdict"] == "FLUENT_ONLY"


def test_sleep_margin_and_shuffled_bound_each_alone():
    # margin exactly 1.0 with shuffled 1.5 passes; margin 0.99 (shuffled within bound) fails S
    r = _recs()
    _both(("sleep", "sleep_true", "recall"), 2.5)(r)
    _both(("sleep", "sleep_shuffled", "recall"), 1.5)(r)
    assert E.verdict(r, SEEDS)["arms"]["a"]["S"]
    _both(("sleep", "sleep_true", "recall"), 2.49)(r)
    v = E.verdict(r, SEEDS)
    assert not v["arms"]["a"]["S"] and v["verdict"] == "FLUENT_ONLY"
    # shuffled 1.6 > 1.5 fails S even with a margin of 2.4
    r = _recs()
    _both(("sleep", "sleep_true", "recall"), 4.0)(r)
    _both(("sleep", "sleep_shuffled", "recall"), 1.6)(r)
    assert not E.verdict(r, SEEDS)["arms"]["a"]["S"]
    # false recall exactly teacher + 0.5 passes
    r = _recs()
    _both(("sleep", "sleep_true", "false_recall"), 2.5)(r)
    assert E.verdict(r, SEEDS)["verdict"] == "GO"


def test_twin_sham_bound():
    r = _recs()
    _both(("ladder", "sham_4.0"), 3.0)(r)
    assert E.verdict(r, SEEDS)["verdict"] == "GO"
    r = _recs()
    _both(("ladder", "sham_4.0"), 12)(r)                               # the review's case: 12/12 sham was GO before
    v = E.verdict(r, SEEDS)
    assert v["verdict"] == "FLUENT_ONLY" and not v["arms"]["a"]["R"] and not v["arms"]["a2"]["R"]


@pytest.mark.parametrize("edit,why", [
    (lambda r: r["b"][43]["ppl"].update(wt103_nll=float("nan")), "b non-finite s43"),
    (lambda r: r["b"][100]["sleep"]["no_sleep"].update(recall=None), "b non-finite s100"),
    (lambda r: _set(r, "b", ("ppl", "wt103_nll"), 6.01), "b not trained"),     # within 1 nat of unigram 7.0
])
def test_diverged_b_makes_f2_unassessable_not_void(edit, why):
    r = _recs()
    edit(r)
    v = E.verdict(r, SEEDS)
    assert v["verdict"] == "GO" and v["carry"] == "a" and not v["f2_assessable"]
    assert len(v["b_status"]) == 1 and v["b_status"][0].startswith(why)
    assert v["arms"]["a"]["F2"] is None and v["arms"]["a"]["init_gap"] is None
    r = _recs()
    edit(r)
    _both(("ladder", "reinst_4.0"), 7)(r)
    assert E.verdict(r, SEEDS)["verdict"] == "FLUENT_ONLY"              # F1 holds, F2 unassessable, R fails
    r = _recs()
    edit(r)
    _set(r, "b", ("ppl", "mean_count"), 0.0)                          # b's C5 is not read once b is out
    assert E.verdict(r, SEEDS)["verdict"] == "GO"


def test_b_exactly_one_nat_below_unigram_is_trained():
    r = _set(_recs(), "b", ("ppl", "wt103_nll"), 6.0)
    v = E.verdict(r, SEEDS)
    assert v["f2_assessable"] and v["arms"]["a"]["F2"]


CONTROL_CASES = [
    ("C1 teacher in-context", lambda r: _set(r, "c", ("ladder", "in_context"), 8)),
    ("C1 teacher in-context", lambda r: _set(r, "c", ("ladder", "in_context"), 4)),
    ("C1 teacher reinstatement", lambda r: _set(r, "c", ("ladder", "reinst_4.0"), 9)),
    ("C1 teacher same-words", lambda r: _set(r, "c", ("ladder", "same_words"), 10)),
    ("C2 teacher sleep recall", lambda r: _set(r, "c", ("sleep", "sleep_true", "recall"), 1)),
    ("C2 teacher shuffled", lambda r: _set(r, "c", ("sleep", "sleep_shuffled", "recall"), 2)),
    ("C3 a no-sleep", lambda r: _set(r, "a", ("sleep", "no_sleep", "recall"), 2)),
    ("C3 b no-sleep", lambda r: _set(r, "b", ("sleep", "no_sleep", "recall"), 2)),
    ("C3 c no-sleep", lambda r: _set(r, "c", ("sleep", "no_sleep", "recall"), 2)),
    ("C4 teacher wrong-memory sham", lambda r: _set(r, "c", ("ladder", "sham_4.0"), 4)),
    ("C5 a spikes not engaged", lambda r: _set(r, "a", ("ppl", "wt103_nll_spikes_off"), 3.305)),
    ("C5 a2 spikes not engaged", lambda r: _set(r, "a2", ("ppl", "silent_share"), 0.995)),
    ("C5 b spikes not engaged", lambda r: _set(r, "b", ("ppl", "mean_count"), 0.0)),
    ("C6 missing c s42", lambda r: r["c"].pop(42)),
    ("C6 missing a s101", lambda r: r["a"][101].pop("ladder")),
    ("C6 non-finite a2 s43", lambda r: r["a2"][43]["ppl"].update(wt103_nll=float("nan"))),
    ("C6 non-finite c s44", lambda r: r["c"][44]["sleep"]["sleep_true"].update(recall=None)),
    ("C6 non-finite a s42", lambda r: r["a"][42]["sleep"]["sleep_true"].update(false_recall=float("inf"))),
    ("C6 non-finite c s102", lambda r: r["c"][102]["ppl"].pop("wt103_unigram_nll")),
    ("C6 missing b s44", lambda r: r["b"].pop(44)),                    # a b that was never run still voids
]


@pytest.mark.parametrize("prefix,edit", CONTROL_CASES)
def test_each_control_failing_alone_voids(prefix, edit):
    r = _recs()
    assert E.verdict(copy.deepcopy(r), SEEDS)["verdict"] == "GO"
    edit(r)
    v = E.verdict(r, SEEDS)
    assert v["verdict"] == "VOID"
    assert len(v["reasons"]) == 1 and v["reasons"][0].startswith(prefix), v["reasons"]


def test_dev_seed_is_not_a_claim():
    r = {arm: {7: _seed_rec(arm)} for arm in E.ARMS}
    v = E.verdict(r, [7])
    assert v["verdict"] == "GO" and not v["claim"]


def test_estimate_four_arms():
    e = E.estimate(25000.0, refill_s=4.0)
    train_h = 3e8 / 25000 / 3600
    refills = math.ceil(E.total_steps() / E.CFG["regen_every"])
    assert E.total_steps() == 36622 and refills == 2289 and E.refill_size() == 64 and E.onpolicy_k() == 4
    assert math.isclose(e["a2_extra_h"], refills * 4.0 / 3600)
    assert math.isclose(e["per_seed_h"], 3 * train_h + e["a2_extra_h"] + 3 * 0.5 + 0.25)
    assert math.isclose(e["total_h"], 7 * e["per_seed_h"] + 0.25)
    slow = E.estimate(5000.0, refill_s=4.0)
    assert math.isclose(slow["a2_extra_h"], e["a2_extra_h"])          # sampling is a constant, not scaled by tok/s
    assert slow["per_seed_h"] > 3 * e["per_seed_h"]


TINY = dict(E.CFG, tokens_per_arm=6 * 16 * 2, seq=16, micro_b=2, accum=1, warmup=1, calib_seqs=2, ckpt_every=2,
            regen_every=1, prompt_len=4, sleep_steps=2)


def _pools():
    g = np.random.default_rng(0)
    return g.integers(32, 127, 4000).astype(np.uint16), g.integers(32, 127, 1000).astype(np.uint16)


def test_arms_a_and_b_draw_the_same_data():
    fw, chat = _pools()
    s1, s2 = E.Stream(fw, chat, 42, TINY), E.Stream(fw, chat, 42, TINY)
    assert [s1.micro() for _ in range(5)] == [s2.micro() for _ in range(5)]
    assert E.Stream(fw, chat, 43, TINY).micro() != E.Stream(fw, chat, 42, TINY).micro()


def test_train_resume_equals_uninterrupted_and_a2_b_run(tmp_path):
    c, pools = _Cortex(), _pools()
    d1, d2 = tmp_path / "full", tmp_path / "resumed"
    d1.mkdir(), d2.mkdir()
    r1 = E.train("a", 42, str(d1), "cpu", "fp32", cfg=TINY, cortex=c, pools=pools, log=lambda m: None)
    assert r1["steps"] == E.total_steps(TINY) == 6 and r1["provenance"]["device"] == "cpu"
    E.train("a", 42, str(d2), "cpu", "fp32", max_steps=3, cfg=TINY, cortex=c, pools=pools, log=lambda m: None)
    assert os.path.exists(d2 / "train_a_s42.partial.json") and not os.path.exists(d2 / "twin_a_s42.pt")
    E.train("a", 42, str(d2), "cpu", "fp32", cfg=TINY, cortex=c, pools=pools, log=lambda m: None)
    w1 = torch.load(d1 / "twin_a_s42.pt", weights_only=False)
    w2 = torch.load(d2 / "twin_a_s42.pt", weights_only=False)
    assert all(torch.allclose(w1["z"][k], w2["z"][k], atol=1e-6) for k in w1["z"])
    teacher = E.make_twin(c, spikes=False)
    assert any(not torch.equal(w1["z"][k], teacher.z[k].cpu()) for k in w1["z"] if w1["z"][k].numel())
    for arm in ("a2", "b"):
        r = E.train(arm, 42, str(d1), "cpu", "fp32", cfg=TINY, cortex=c, pools=pools, log=lambda m: None)
        assert r["steps"] == 6 and all(math.isfinite(x) for x in r["final_loss"][1:])


def test_eval_units_atomic_skip_and_provenance(tmp_path):
    c, pools = _Cortex(), _pools()
    E.train("a", 7, str(tmp_path), "cpu", "fp32", cfg=TINY, cortex=c, pools=pools, log=lambda m: None)
    ev = (pools[0][:200], pools[1][:200])
    for arm in ("a", "c"):
        E.evaluate_arm(arm, 7, str(tmp_path), "cpu", units=("ppl", "ladder", "sleep", "transcript"), cortex=c,
                       pools=ev, cfg=TINY, log=lambda m: None)
    rec = json.load(open(E.unit_path(str(tmp_path), "a", 7, "ppl")))
    assert rec["provenance"]["device"] == "cpu" and "host" in rec["provenance"] and "commit" in rec["provenance"]
    p = rec["result"]
    assert p["wt103_tokens"] > 0 and p["wt103_nll_spikes_off"] != p["wt103_nll"] and 0 < p["mean_count"] <= 4
    assert "mean_count" not in json.load(open(E.unit_path(str(tmp_path), "c", 7, "ppl")))["result"]
    lad = json.load(open(E.unit_path(str(tmp_path), "a", 7, "ladder")))["result"]
    assert set(lad) == {"in_context", "reinst_1.5", "reinst_2.5", "reinst_4.0", "sham_4.0", "same_words", "paraphrase"}
    sl = json.load(open(E.unit_path(str(tmp_path), "a", 7, "sleep")))["result"]
    assert set(sl) == set(E.SLEEP_ARMS) and sl["no_sleep"]["adapter_norm"] == 0 and sl["sleep_true"]["adapter_norm"] > 0
    assert not any(f.endswith(".tmp") for f in os.listdir(tmp_path))
    mt = os.path.getmtime(E.unit_path(str(tmp_path), "a", 7, "ppl"))
    E.evaluate_arm("a", 7, str(tmp_path), "cpu", units=("ppl",), cortex=c, pools=ev, cfg=TINY, log=lambda m: None)
    assert os.path.getmtime(E.unit_path(str(tmp_path), "a", 7, "ppl")) == mt
    recs = E.load_records(str(tmp_path), [7])
    assert set(recs) == {"a", "c"} and E.verdict(recs, [7])["verdict"] == "VOID"   # a2, b missing


def test_chat_pool_starts_at_marker_only(tmp_path):
    c = _Cortex()
    tw = E.make_twin(c, spikes=False)
    out = str(tmp_path / "chat.npy")
    E.chat_pool(tw, c.pipe, 2 * 16, 0, out, 16, batch=2)
    ids = np.load(out)
    mark = c.pipe.encode(E.USER_MARK)
    assert ids.size == 32 and list(ids[:len(mark)]) == mark and list(ids[16:16 + len(mark)]) == mark


def test_a2_resume_equals_uninterrupted_with_a_nonempty_buffer(tmp_path):
    # regen_every 4 with k = 1 per step: the step-3 checkpoint holds 1 unconsumed self-generated sequence and a
    # sampler that has advanced; both must survive the resume (review 11-27 item 4)
    cfg = dict(TINY, regen_every=4)
    assert E.onpolicy_k(cfg) == 1 and E.refill_size(cfg) == 4
    c, pools = _Cortex(), _pools()
    d1, d2 = tmp_path / "full", tmp_path / "resumed"
    d1.mkdir(), d2.mkdir()
    E.train("a2", 42, str(d1), "cpu", "fp32", cfg=cfg, cortex=c, pools=pools, log=lambda m: None)
    p = E.train("a2", 42, str(d2), "cpu", "fp32", max_steps=3, cfg=cfg, cortex=c, pools=pools, log=lambda m: None)
    ck = torch.load(d2 / "ckpt_a2_s42.pt", weights_only=False)
    assert len(ck["buf"]) == 1 and ck["gen_state"] is not None
    assert p["refill_s"] is not None and p["refill_s"] > 0          # the smoke times one a2 refill
    E.train("a2", 42, str(d2), "cpu", "fp32", cfg=cfg, cortex=c, pools=pools, log=lambda m: None)
    w1 = torch.load(d1 / "twin_a2_s42.pt", weights_only=False)
    w2 = torch.load(d2 / "twin_a2_s42.pt", weights_only=False)
    assert all(torch.equal(w1["z"][k], w2["z"][k]) for k in w1["z"])
    assert all(torch.equal(w1["theta"][k], w2["theta"][k]) for k in w1["theta"])
    assert not os.path.exists(d1 / "ckpt_a2_s42.pt") and not os.path.exists(d2 / "ckpt_a2_s42.pt")


def test_chat_draws_are_whole_dialogue_rows():
    fw, chat = _pools()
    cfg = dict(TINY, chat_share=1.0)
    st = E.Stream(fw, chat, 42, cfg)
    rows = {tuple(int(t) for t in chat[o:o + 16]) for o in range(0, len(chat) - 15, 16)}
    assert all(tuple(st.draw()) in rows for _ in range(50))


def test_arm_c_seed_free_units_are_shared(tmp_path):
    d = str(tmp_path)
    assert E.unit_path(d, "c", 7, "ppl") == E.unit_path(d, "c", 42, "ppl")
    assert E.unit_path(d, "c", 7, "ladder") == E.unit_path(d, "c", 42, "ladder")
    assert E.unit_path(d, "c", 7, "sleep") != E.unit_path(d, "c", 42, "sleep")
    assert E.unit_path(d, "a", 7, "ppl") != E.unit_path(d, "a", 42, "ppl")


def test_unigram_nll():
    ids = np.array([1, 2, 2, 3, 3, 3, 3, 3, 3], dtype=np.uint16)      # seq 4: targets ids[1:9] = 2,2,3,3,3,3,3,3
    p = np.array([2, 6]) / 8
    assert math.isclose(E.unigram_nll(ids, 4), float(-(p * np.log(p)).sum()))
