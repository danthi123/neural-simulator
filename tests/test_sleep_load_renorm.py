"""Load-dependent sleep renormalization (webapp/sleep_replay_capture.py r3, default-OFF BRAIN_SLEEP_LOAD_RENORM).

No brain build: a fake block store (build-time blocks + one block per told fact) whose `_block_role_scores` reads the
coherence of each block's current synapses with the pattern it was written with, and a fake linear D1 pool. Under test:
  * flag OFF -> the store is byte-identical to the pre-branch code, for the route alone AND for r2's constant
    downscaling (hashes computed on main at 9d1329c35's webapp/sleep_replay_capture.py and pinned here);
  * the night's amplitude is the measured fraction dW / W of the store's strength the preceding wake added;
  * a night after a day with nothing learned depresses nothing; later learning is what forces an older trace down,
    monotonically in the number of facts told; the load lesion reads the load and applies nothing;
  * with both sub-flags on, the measured amplitude replaces the constant;
  * the battery's fi groups are label-only, seven nights, the registered dose, and share no content word with the fact.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _REPO)

from webapp import da_tag_capture as T                     # noqa: E402
from webapp import da_tag_capture_chat as W                # noqa: E402
from webapp import sleep_replay_capture as S               # noqa: E402

FLAGS = ("BRAIN_DA_TAG_CAPTURE", "BRAIN_DA_TAG_CAPTURE_CLOCK", "BRAIN_SLEEP_REPLAY_CAPTURE",
         "BRAIN_SLEEP_REPLAY_CAPTURE_LESION", "BRAIN_DA_CAPTURE_LESION", "BRAIN_DA_ENCODING_LESION",
         "BRAIN_SLEEP_DOWNSCALING", "BRAIN_SLEEP_LOAD_RENORM", "BRAIN_SLEEP_LOAD_RENORM_LESION",
         "BRAIN_AWAKE_REPLAY_CAPTURE")
# sha256 of the scenario store below (facts_per_day=(2, 2), three nights), computed with the PRE-BRANCH
# webapp/sleep_replay_capture.py (main @ f793b6945, `git show main:webapp/sleep_replay_capture.py` swapped in, every
# other module identical): route only / route + r2 constant downscaling.
PRE_BRANCH_SHA = {"rc": "4275072c846c420b719dfef7f35742e44bb1af4312f3c4993f68f398f2f52c73",
                  "rc_shy": "2c4668b2c828bfc8dccdf66f2df5ed5bc0e40f87d62ba662cf2236dcc216d3ce"}


class FakeD1:
    def __init__(self, *a, **kw):
        self.a_go = self.read(T.prp_threshold())[0]

    def read(self, da):
        return float(np.clip((float(da) - T._DA_TONIC) / (S.DA_SWR_FULL - T._DA_TONIC), 0.0, 1.0)), None


class FakeComposer:
    def __init__(self, D=64, seed=0):
        self.D, self.store_conns, self.patterns = D, [], []
        self._rng = np.random.default_rng(seed)

    def store(self, g=1.0):
        u = np.exp(2j * np.pi * self._rng.random(self.D))
        i = len(self.patterns)
        self.patterns.append(u)
        trig = 1000 + i * (self.D + 1)
        self.store_conns += [(trig + 1 + k, trig, complex(g * u[k])) for k in range(self.D)]

    def coherence(self, i):
        w = np.array([complex(x[2]) for x in self.store_conns[i * self.D:(i + 1) * self.D]])
        return float(abs(np.mean(np.conj(self.patterns[i]) * w)) / max(1e-12, float(np.mean(np.abs(w)))))

    def _block_role_scores(self, i):
        c = self.coherence(i)
        return {"agent": ("a", 1.0, c, None), "action": ("b", 1.0, c, None), "patient": ("c", 1.0, c, None)}


class Inner:
    pass


class Chat:
    def __init__(self, comp):
        self.inner = Inner()
        self.inner.composer = comp


def _store_hash(comp):
    return hashlib.sha256(json.dumps([(p, q, complex(w).real, complex(w).imag)
                                      for (p, q, w) in comp.store_conns]).encode()).hexdigest()


def run_scenario(env, facts_per_day=(0, 0), n_build=3, g_fact=1.0, n_nights=3):
    """Tell one fact (neutral DA), then n nights; after night n (n < n_nights) the environment tells
    facts_per_day[n-1] other facts. Sets and restores os.environ itself (usable outside pytest)."""
    keep = {k: os.environ.get(k) for k in FLAGS}
    for k in FLAGS:
        os.environ.pop(k, None)
    os.environ.update({"BRAIN_DA_TAG_CAPTURE": "1", "BRAIN_DA_TAG_CAPTURE_CLOCK": "turn", **env})
    old_d1, old_off = W.SpikingD1Activation, W._WORLD_OFFSET_H
    W.SpikingD1Activation, W._WORLD_OFFSET_H = FakeD1, 0.0
    try:
        comp = FakeComposer()
        for _ in range(n_build):
            comp.store()
        chat = Chat(comp)

        def turn(da, g=None):
            chat._last_da_drives = {"da_level": da}
            W.observe_chat_turn(chat, seed=7)
            if g is not None:
                comp.store(g=g)
            W.after_store_chat(chat)
        for i, da in enumerate([0.5, 0.5, 0.5, 0.5, 0.5]):
            turn(da, g_fact if i == 4 else None)
        for n in range(1, n_nights + 1):
            W.advance_world_clock_h(24.0)
            W.tick_chat(chat)
            if n < n_nights:
                turn(0.5)                                   # the morning question (stores nothing)
                for _ in range(facts_per_day[n - 1] if n - 1 < len(facts_per_day) else 0):
                    turn(0.5, 1.0)
        return chat, getattr(chat, "_da_tag_capture", None), comp
    finally:
        W.SpikingD1Activation, W._WORLD_OFFSET_H = old_d1, old_off
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


RC = {"BRAIN_SLEEP_REPLAY_CAPTURE": "1"}
SHY = {"BRAIN_SLEEP_DOWNSCALING": "1"}
LR = {"BRAIN_SLEEP_LOAD_RENORM": "1"}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in FLAGS:
        monkeypatch.delenv(k, raising=False)
    yield


def _inc(cap, i=0):
    return float(np.mean(np.abs(cap.ledger.blocks[i]["inc"])))


# ── flag OFF: byte-identical to the pre-branch code ─────────────────────────────────────────────────────────────────
def test_flag_off_route_only_matches_pre_branch_hash():
    _c, cap, comp = run_scenario(dict(RC), facts_per_day=(2, 2))
    assert _store_hash(comp) == PRE_BRANCH_SHA["rc"]
    assert all("load" not in e and "shy_scale" not in e for e in cap._src.summary()["epochs"])
    assert "load_renorm" not in cap._src.summary()


def test_flag_off_constant_downscaling_matches_pre_branch_hash():
    _c, cap, comp = run_scenario({**RC, **SHY}, facts_per_day=(2, 2))
    assert _store_hash(comp) == PRE_BRANCH_SHA["rc_shy"]
    assert all("load" not in e for e in cap._src.summary()["epochs"])


def test_flag_explicit_zero_is_off():
    _c, _cap, comp0 = run_scenario({**RC, **SHY}, facts_per_day=(2, 2))
    _c, _cap, comp1 = run_scenario({**RC, **SHY, "BRAIN_SLEEP_LOAD_RENORM": "0"}, facts_per_day=(2, 2))
    assert _store_hash(comp1) == _store_hash(comp0)


def test_inert_without_the_sleep_route():
    _c, cap0, comp0 = run_scenario({}, facts_per_day=(2, 2))
    _c, cap1, comp1 = run_scenario(dict(LR), facts_per_day=(2, 2))
    assert _store_hash(comp1) == _store_hash(comp0) and getattr(cap1, "_src", None) is None


# ── flag ON ─────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_amplitude_is_the_measured_wake_fraction():
    _c, cap, comp = run_scenario({**RC, **LR}, facts_per_day=(0, 0), n_nights=1)
    e = cap._src.summary()["epochs"][0]
    ld = e["load"]
    assert ld["n_new_blocks"] == 1 and ld["t_since"] is None and not ld["lesioned"]
    assert 0.0 < ld["dW"] < ld["W"] and abs(ld["delta"] - ld["dW"] / ld["W"]) < 1e-8
    assert abs(e["shy_scale"][0] - (1.0 - ld["delta"] * (1.0 - e["R_eff"][0]))) < 1e-8


def test_store_total_and_wake_potentiation_read_the_store():
    comp = FakeComposer(D=8)
    comp.store(g=1.0)
    comp.store(g=2.0)
    assert abs(S.store_total_strength(comp) - 3.0) < 1e-12

    class L:
        blocks = [{"t_w": 0.0, "inc": np.full(4, 2.0 + 0j)}, {"t_w": 5.0, "inc": np.full(4, 3.0 + 0j)}]

        def weight_factor(self, blk):
            return np.full(4, 0.5)
    assert abs(S.wake_potentiation(L(), 1.0) - 1.5) < 1e-12 and abs(S.wake_potentiation(L(), -1.0) - 2.5) < 1e-12


def test_night_after_an_empty_day_costs_nothing():
    _c, cap, _comp = run_scenario({**RC, **LR}, facts_per_day=(0, 0), n_nights=3)
    eps = cap._src.summary()["epochs"]
    assert eps[0]["load"]["delta"] > 0.0
    for e in eps[1:]:
        assert e["load"]["dW"] == 0.0 and e["load"]["delta"] == 0.0 and e["shy_scale"] == [1.0]


def test_later_learning_forces_the_old_trace_down_in_dose_order():
    incs = {}
    for k in (0, 1, 3):
        _c, cap, _comp = run_scenario({**RC, **LR}, facts_per_day=(k, k), n_nights=3)
        incs[k] = _inc(cap, 0)
        eps = cap._src.summary()["epochs"]
        assert [e["load"]["n_new_blocks"] for e in eps] == [1, k, k]
    assert incs[0] > incs[1] > incs[3]


def test_constant_downscaling_ignores_the_dose():
    """r2's constant charges the old trace by nights, not by what was learned (the reason for r3)."""
    _c, cap0, _ = run_scenario({**RC, **SHY}, facts_per_day=(0, 0), n_nights=3)
    _c, cap3, _ = run_scenario({**RC, **SHY}, facts_per_day=(3, 3), n_nights=3)
    assert abs(_inc(cap0, 0) - _inc(cap3, 0)) < 1e-9


def test_load_lesion_reads_the_load_and_applies_nothing():
    _c, cap_rc, _ = run_scenario(dict(RC), facts_per_day=(3, 3))
    _c, cap, _ = run_scenario({**RC, **LR, "BRAIN_SLEEP_LOAD_RENORM_LESION": "1"}, facts_per_day=(3, 3))
    eps = cap._src.summary()["epochs"]
    assert all(e["load"]["lesioned"] and e["load"]["delta"] == 0.0 and e["load"]["delta_read"] > 0.0 for e in eps)
    assert all(all(s == 1.0 for s in e["shy_scale"]) for e in eps)
    assert abs(_inc(cap, 0) - _inc(cap_rc, 0)) < 1e-12


def test_load_replaces_the_constant_when_both_are_armed():
    _c, cap_lr, _ = run_scenario({**RC, **LR}, facts_per_day=(0, 0))
    _c, cap_both, _ = run_scenario({**RC, **LR, **SHY}, facts_per_day=(0, 0))
    e_lr, e_both = cap_lr._src.summary()["epochs"], cap_both._src.summary()["epochs"]
    assert [e["shy_scale"] for e in e_both] == [e["shy_scale"] for e in e_lr]
    assert e_both[1]["shy_scale"] == [1.0]                    # the constant would have charged this night


def test_replay_lesion_removes_the_protection_not_the_load():
    _c, cap, _ = run_scenario({**RC, **LR, "BRAIN_SLEEP_REPLAY_CAPTURE_LESION": "1"}, facts_per_day=(0, 0),
                              n_nights=1)
    e = cap._src.summary()["epochs"][0]
    assert e["R_eff"] == [0.0] and abs(e["shy_scale"][0] - (1.0 - e["load"]["delta"])) < 1e-8


# ── battery groups ──────────────────────────────────────────────────────────────────────────────────────────────────
def test_battery_fi_groups_are_label_only_seven_nights_and_the_registered_dose():
    from research.runners import onebrain_regression_battery as B
    from research.runners import _da_tag_capture_chat_probe as P
    from research.runners.load_bearing_fraction import turn_group
    probe_labels = {t[0] for t in B.PROBE_TURNS}
    facts_seen = set()
    for grp, k in P.FI_DOSE.items():
        lab = "%s_recall%d" % (grp, P.FI_NIGHTS)
        assert lab in B._TURN_BY_LABEL and lab not in probe_labels
        rows = turn_group(lab)
        assert sum(1 for x in rows if x in B._WORLD_STEPS) == P.FI_NIGHTS
        told = [x for x in rows if x.startswith(grp + "_d")]
        assert len(told) == k * (P.FI_NIGHTS - 1)
        rem = [x for x in rows if x.startswith(grp + "_remention")]
        assert len(rem) == len(P.FI_REMENTION.get(grp, ()))
        for x in told:
            msg = B._TURN_BY_LABEL[x][1]
            assert not ({"cat", "chase", "chases", "ball"} & set(msg.split()))
            facts_seen.add(msg)
    assert len(facts_seen) == max(P.FI_DOSE.values()) * (P.FI_NIGHTS - 1)     # no fact told twice in a group
    for n, lab, _env in P.FI_ARMS:
        assert lab in B._TURN_BY_LABEL
