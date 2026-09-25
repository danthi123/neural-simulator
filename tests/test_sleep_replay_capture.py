"""Sleep-replay-triggered synaptic capture (webapp/sleep_replay_capture.py, default-OFF BRAIN_SLEEP_REPLAY_CAPTURE).

No brain build: a fake block store whose `_block_role_scores` reads the coherence of each block's CURRENT synapses with
the fact it was written with (a stand-in for the composer's cleanup margin), and a fake D1 population whose activation
is linear between tonic (0.5) and its own ceiling (1.24). What is under test is the wiring and the ledger dynamics:
  * flag OFF -> byte-identical to the plain v3 ledger path (and the v3 scenario hash is the pre-branch value);
  * flag ON  -> an ordinary fact told just before sleep is replayed and captured; one whose trace had already decayed
    when sleep came is not replayed and still decays; the replay-edge lesion removes the rescue;
  * the capture lesion still blocks capture of a salient fact; the DA-encoding lesion blocks the sleep route too;
  * no sleep-depth idle -> no replay; one SWR epoch per sleep episode; a new turn starts a new episode.
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

V3_SCENARIO_SHA_PRE_BRANCH = "82f8856dc8465ca5eb25a6bb8ca380fc46134a552b11326aa30ad18d841248c8"
FLAGS = ("BRAIN_DA_TAG_CAPTURE", "BRAIN_DA_TAG_CAPTURE_CLOCK", "BRAIN_SLEEP_REPLAY_CAPTURE",
         "BRAIN_SLEEP_REPLAY_CAPTURE_LESION", "BRAIN_DA_CAPTURE_LESION", "BRAIN_DA_ENCODING_LESION")


class FakeD1:
    """a(DA) linear from tonic to the D1 pool's own ceiling; deterministic (no spiking noise)."""

    def __init__(self, *a, **kw):
        self.a_go = self.read(T.prp_threshold())[0]

    def read(self, da):
        return float(np.clip((float(da) - T._DA_TONIC) / (S.DA_SWR_FULL - T._DA_TONIC), 0.0, 1.0)), None


class FakeComposer:
    """Block-major store (D synapses per block) + a substrate read-back: the coherence of the block's current weights
    with the phasor pattern it was written with (high while the increment is expressed, ~1/sqrt(D) at baseline)."""

    def __init__(self, D=64, seed=0):
        self.D = D
        self.store_conns = []
        self.patterns = []
        self._rng = np.random.default_rng(seed)
        self.n_reads = 0

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
        self.n_reads += 1
        c = self.coherence(i)
        return {"agent": ("a", 1.0, c, None), "action": ("b", 1.0, c, None), "patient": ("c", 1.0, c, None),
                "attribute": ("x", 1.0, 0.0, None), "polarity": ("pos", 1.0, 1.0, None)}


class Inner:
    pass


class Chat:
    def __init__(self, comp):
        self.inner = Inner()
        self.inner.composer = comp


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in FLAGS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(W, "_WORLD_OFFSET_H", 0.0)
    monkeypatch.setattr(W, "SpikingD1Activation", FakeD1)
    yield


def _store_hash(comp):
    return hashlib.sha256(json.dumps([(p, q, complex(w).real, complex(w).imag)
                                      for (p, q, w) in comp.store_conns]).encode()).hexdigest()


def _conversation(monkeypatch, *, da_turns, fact_turn=2, night=True, n_build_blocks=2, g=1.0, rc=False, env=None):
    """Tell one fact inside a scripted conversation (turn clock), then (optionally) a 24 h night through the idle tick.
    Returns (chat, cap, comp)."""
    monkeypatch.setenv("BRAIN_DA_TAG_CAPTURE", "1")
    monkeypatch.setenv("BRAIN_DA_TAG_CAPTURE_CLOCK", "turn")
    if rc:
        monkeypatch.setenv("BRAIN_SLEEP_REPLAY_CAPTURE", "1")
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    comp = FakeComposer()
    for _ in range(n_build_blocks):
        comp.store()                                   # build-time knowledge: unmanaged (block_offset)
    chat = Chat(comp)
    for i, da in enumerate(da_turns):
        chat._last_da_drives = {"da_level": da}
        W.observe_chat_turn(chat, seed=7)
        if i == fact_turn:
            comp.store(g=g)
        W.after_store_chat(chat)
    if night:
        W.advance_world_clock_h(24.0)
        W.tick_chat(chat)
    return chat, chat._da_tag_capture, comp


NEUTRAL = [0.5, 0.5, 0.5, 0.5, 0.5]
SALIENT = [1.0, 1.1, 0.5, 1.1, 1.0]


def _fact_captured(cap, i=0):
    b = cap.ledger.summary()[i]
    return b["frac_synapses_z_gt_half"]


# ── flag OFF: byte-identical ────────────────────────────────────────────────────────────────────────────────────────
def test_v3_ledger_scenario_hash_is_the_pre_branch_value():
    from research.runners._da_tag_capture_chat_probe import _ledger_scenario_hash
    assert _ledger_scenario_hash(_REPO) == V3_SCENARIO_SHA_PRE_BRANCH


def test_flag_off_is_identical_to_plain_ledger_path(monkeypatch):
    chat, cap, comp = _conversation(monkeypatch, da_turns=NEUTRAL, rc=False)
    s_off = _store_hash(comp)
    assert not hasattr(cap, "_src")
    assert "sleep_replay_capture" not in W.after_store_chat(chat)
    assert all("h_rep" not in b for b in cap.ledger.blocks)
    assert comp.n_reads == 0                            # no substrate read is ever made with the flag off
    # the same conversation replayed through the v3 ledger alone (no ChatTagCapture, no sleep hook) -> same synapses
    comp2 = FakeComposer()
    comp2.store(); comp2.store()
    L = T.SynapticTagCaptureLedger(7, gamma=cap.gamma, d1=FakeD1(), block_offset=2)
    for i, da in enumerate(NEUTRAL):
        t = i * W.TURN_DRIVE_H
        L.sync_from_store(comp2, t); L.on_store(comp2, t); L.advance(comp2, t)
        L.observe_turn(t, W.TURN_DRIVE_H, da)
        if i == 2:
            comp2.store()
        L.sync_from_store(comp2, t); L.on_store(comp2, t)
    t_end = len(NEUTRAL) * W.TURN_DRIVE_H + 24.0
    L.sync_from_store(comp2, t_end); L.on_store(comp2, t_end); L.advance(comp2, t_end)
    assert _store_hash(comp2) == s_off


def test_flag_on_without_sleep_depth_idle_changes_nothing(monkeypatch):
    _c1, cap_off, comp_off = _conversation(monkeypatch, da_turns=NEUTRAL, night=False, rc=False)
    for k in FLAGS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(W, "_WORLD_OFFSET_H", 0.0)
    _c2, cap_on, comp_on = _conversation(monkeypatch, da_turns=NEUTRAL, night=False, rc=True)
    assert _store_hash(comp_on) == _store_hash(comp_off)
    assert cap_on._src.summary()["n_epochs"] == 0 and comp_on.n_reads == 0


def test_inert_without_the_tag_capture_ledger(monkeypatch):
    monkeypatch.setenv("BRAIN_SLEEP_REPLAY_CAPTURE", "1")
    comp = FakeComposer(); comp.store()
    chat = Chat(comp)
    chat._last_da_drives = {"da_level": 0.5}
    assert W.observe_chat_turn(chat, seed=7) is None and W.tick_chat(chat) is None
    assert not hasattr(chat, "_da_tag_capture") and comp.n_reads == 0


# ── flag ON: the route ──────────────────────────────────────────────────────────────────────────────────────────────
def test_ordinary_fact_forgotten_without_the_route(monkeypatch):
    _chat, cap, comp = _conversation(monkeypatch, da_turns=NEUTRAL, rc=False)
    assert _fact_captured(cap) == 0.0
    assert comp.coherence(2) < 0.3                     # back at baseline: the fact is gone


def test_ordinary_fact_replayed_and_captured(monkeypatch):
    _chat, cap, comp = _conversation(monkeypatch, da_turns=NEUTRAL, rc=True)
    src = cap._src.summary()
    assert src["n_epochs"] == 1 and src["episode_done"]   # the night's SWR epoch ran (the replay branch executed)
    e0 = src["epochs"][0]
    assert e0["R"][0] > 0.5 and e0["da_swr"] > T.prp_threshold()
    assert _fact_captured(cap) > 0.9
    assert comp.coherence(2) > 0.5                     # the fact's pattern is still expressed at 24 h


def test_unreplayed_decayed_fact_still_decays(monkeypatch):
    """Fact A told 4 h before sleep (its trace has decayed when the SWR comes -> the store's read-back is near noise),
    fact B told just before sleep: B is replayed and captured, A is not and decays to baseline."""
    monkeypatch.setenv("BRAIN_SLEEP_REPLAY_CAPTURE", "1")
    comp = FakeComposer()
    d1 = FakeD1()
    gamma = T.calibrate_gamma(d1.a_go)
    L = T.SynapticTagCaptureLedger(7, gamma=gamma, d1=d1, block_offset=0)
    src = S.SleepReplayCapture(7, d1)
    L.observe_turn(0.0, W.TURN_DRIVE_H, 0.5); comp.store(); L.on_store(comp, 0.0)          # fact A at t=0
    t_b = 4.0
    L.observe_turn(t_b, W.TURN_DRIVE_H, 0.5); comp.store(); L.on_store(comp, t_b)          # fact B at t=4 h
    src.catch_up(L, comp, t_b, 1, 24.0)
    L.advance(comp, 24.0)
    R = src.summary()["epochs"][0]["R"]
    assert R[1] > 0.5 and R[0] < R[1]
    s = L.summary()
    assert s[1]["frac_synapses_z_gt_half"] > 0.9       # replayed -> captured
    assert s[0]["frac_synapses_z_gt_half"] < 0.1       # not replayed (decayed) -> not captured
    assert comp.coherence(0) < 0.3 and comp.coherence(1) > 0.5


def test_replay_edge_lesion_removes_the_rescue(monkeypatch):
    _chat, cap, comp = _conversation(monkeypatch, da_turns=NEUTRAL, rc=True,
                                     env={"BRAIN_SLEEP_REPLAY_CAPTURE_LESION": "1"})
    src = cap._src.summary()
    assert src["n_epochs"] == 1 and comp.n_reads > 0                   # the reactivation reads still ran ...
    assert all(r == 0.0 for e in src["epochs"] for r in e["R_eff"])    # ... but their edge is cut
    assert all(e["da_swr"] == T._DA_TONIC for e in src["epochs"])
    assert _fact_captured(cap) == 0.0 and comp.coherence(2) < 0.3


def test_da_encoding_lesion_blocks_the_sleep_route(monkeypatch):
    _chat, cap, comp = _conversation(monkeypatch, da_turns=NEUTRAL, rc=True, env={"BRAIN_DA_ENCODING_LESION": "1"})
    assert all(e["da_seen_by_d1"] == T._DA_TONIC for e in cap._src.summary()["epochs"])
    assert _fact_captured(cap) == 0.0


def test_salient_fact_captured_awake_and_kept_without_the_replay_edge(monkeypatch):
    _chat, cap, comp = _conversation(monkeypatch, da_turns=SALIENT, rc=True,
                                     env={"BRAIN_SLEEP_REPLAY_CAPTURE_LESION": "1"})
    e0 = cap._src.summary()["epochs"][0]
    assert e0["replay_lesioned"] and e0["sum_R_eff"] == 0.0 and e0["a_eff_mean"] == 0.0   # no sleep PRP at all ...
    assert _fact_captured(cap) > 0.9                   # ... so the salient fact is kept by its waking DA capture alone


def test_capture_lesion_still_blocks_salient_capture(monkeypatch):
    _chat, cap, comp = _conversation(monkeypatch, da_turns=SALIENT, rc=True, env={"BRAIN_DA_CAPTURE_LESION": "1"})
    src = cap._src.summary()
    assert all(e["capture_lesioned"] for e in src["epochs"])
    assert _fact_captured(cap) == 0.0 and comp.coherence(2) < 0.3


def test_one_epoch_per_sleep_episode(monkeypatch):
    monkeypatch.setenv("BRAIN_SLEEP_REPLAY_CAPTURE", "1")
    comp = FakeComposer()
    d1 = FakeD1()
    L = T.SynapticTagCaptureLedger(7, gamma=T.calibrate_gamma(d1.a_go), d1=d1)
    src = S.SleepReplayCapture(7, d1)
    L.observe_turn(0.0, W.TURN_DRIVE_H, 0.5); comp.store(); L.on_store(comp, 0.0)
    onset = S.sleep_onset_h()
    assert src.catch_up(L, comp, 0.0, 1, onset - 1e-6) == 0                  # idle, not yet sleep-depth
    assert src.catch_up(L, comp, 0.0, 1, onset + 1e-6) == 1                  # sleep onset: the epoch runs ...
    assert abs(src.epochs[0]["t_h"] - onset) < 1e-9                          # ... at its own time
    assert src.catch_up(L, comp, 0.0, 1, 20.0) == 0                          # ... once per episode
    L.advance(comp, 20.0)
    assert src.catch_up(L, comp, 20.0, 2, 20.0 + onset / 2) == 0             # a new turn: a new episode, not yet due
    assert src.catch_up(L, comp, 20.0, 2, 20.0 + 2 * onset) == 1 and len(src.epochs) == 2


def test_reactivation_strength_reads_the_fact_roles_only():
    comp = FakeComposer(); comp.store()
    r = S.reactivation_strength(comp, 0)
    assert abs(r - comp.coherence(0)) < 1e-12          # attribute (0.0) and polarity ignored
    assert S.reactivation_strength(object(), 0) is None


def test_constants_are_reused_not_new():
    from research.runners import _da_write_gain_spiking_derisk as WG
    assert S.DA_SWR_FULL == WG._DA_CAL_HI
    assert S.SWR_SUBREAD_H == W.TURN_DRIVE_H and S.N_SWR_SUBREADS == 10
    assert S.SWR_BOUT_H == T.CAPTURE_PROTOCOL_MIN / 60.0
    from webapp.continuous_engine import SLEEP_IDLE_SEC
    assert S.sleep_onset_h() == SLEEP_IDLE_SEC / 3600.0


# ── r2 (research/sleep-replay-capture-r2): nightly epochs, the awake mark, sleep downscaling ────────────────────────
@pytest.fixture(autouse=True)
def _clean_env_r2(monkeypatch):
    monkeypatch.delenv("BRAIN_SLEEP_DOWNSCALING", raising=False)
    yield


def _nights(chat, n):
    for _ in range(n):
        W.advance_world_clock_h(24.0)
        W.tick_chat(chat)


def test_one_night_protocol_still_one_epoch_three_nights_three(monkeypatch):
    chat, cap, comp = _conversation(monkeypatch, da_turns=NEUTRAL, rc=True)       # one 24 h night
    assert cap._src.summary()["n_epochs"] == 1
    _nights(chat, 2)
    src = cap._src.summary()
    assert src["n_epochs"] == 3
    gaps = [b["t_h"] - a["t_h"] for a, b in zip(src["epochs"], src["epochs"][1:])]
    assert all(abs(g - S.NIGHT_PERIOD_H) < 1e-9 for g in gaps)


def test_awake_mark_moves_sleep_onset_and_an_old_fact_is_not_rescued(monkeypatch):
    chat, cap, comp = _conversation(monkeypatch, da_turns=NEUTRAL, rc=True, night=False)
    W.advance_world_clock_h(4.0)
    aw = W.mark_awake(chat)
    assert aw is not None and aw >= 4.0
    _nights(chat, 1)
    e0 = cap._src.summary()["epochs"][0]
    assert abs(e0["t_h"] - (aw + S.sleep_onset_h())) < 1e-9     # sleep began after the waking interval
    assert e0["R"][0] < 0.3                                     # the 4 h-old trace reads near baseline
    assert _fact_captured(cap) == 0.0 and comp.coherence(2) < 0.3


def test_mark_awake_without_a_ledger_is_a_noop():
    comp = FakeComposer(); comp.store()
    assert W.mark_awake(Chat(comp)) is None


def _two_fact_night(monkeypatch, shy, n_nights=3):
    """A strong (g=2.5) and a weak (g=0.3) fact told together just before sleep; n nights; flag ON."""
    monkeypatch.setenv("BRAIN_DA_TAG_CAPTURE", "1")
    monkeypatch.setenv("BRAIN_DA_TAG_CAPTURE_CLOCK", "turn")
    monkeypatch.setenv("BRAIN_SLEEP_REPLAY_CAPTURE", "1")
    if shy:
        monkeypatch.setenv("BRAIN_SLEEP_DOWNSCALING", "1")
    comp = FakeComposer()
    chat = Chat(comp)
    for g in (2.5, 0.3):
        chat._last_da_drives = {"da_level": 0.5}
        W.observe_chat_turn(chat, seed=7)
        comp.store(g=g)
        W.after_store_chat(chat)
    _nights(chat, n_nights)
    cap = chat._da_tag_capture
    return cap, comp, [float(np.mean(np.abs(b["inc"]))) for b in cap.ledger.blocks]


def test_downscaling_off_is_byte_identical(monkeypatch):
    _cap0, comp0, _ = _two_fact_night(monkeypatch, shy=False)
    h0 = _store_hash(comp0)
    monkeypatch.setattr(W, "_WORLD_OFFSET_H", 0.0)
    monkeypatch.setenv("BRAIN_SLEEP_DOWNSCALING", "0")
    cap1, comp1, _ = _two_fact_night(monkeypatch, shy=False)
    assert _store_hash(comp1) == h0
    assert all("shy_scale" not in e for e in cap1._src.summary()["epochs"])


def test_downscaling_fades_the_weakly_read_fact_and_spares_the_strong_one(monkeypatch):
    cap_off, comp_off, inc_off = _two_fact_night(monkeypatch, shy=False)
    monkeypatch.setattr(W, "_WORLD_OFFSET_H", 0.0)
    cap_on, comp_on, inc_on = _two_fact_night(monkeypatch, shy=True)
    assert all(b["frac_synapses_z_gt_half"] == 1.0 for b in cap_on.ledger.summary())   # both captured on night 1
    assert inc_on[1] / inc_off[1] < 0.75                                     # weakly read: > 25 % lost over 3 nights
    assert inc_on[0] / inc_off[0] > 0.95                                     # strongly read: < 5 % lost
    assert comp_on.coherence(1) < comp_off.coherence(1) - 0.05
    scales = [e["shy_scale"] for e in cap_on._src.summary()["epochs"]]
    assert len(scales) == 3 and all(s[1] < s[0] for s in scales)            # the weak block is depressed more each night


def test_downscaling_without_the_replay_edge_protects_nothing(monkeypatch):
    monkeypatch.setenv("BRAIN_SLEEP_REPLAY_CAPTURE_LESION", "1")
    cap, comp, _ = _two_fact_night(monkeypatch, shy=True, n_nights=1)
    assert cap._src.summary()["epochs"][0]["shy_scale"] == [round(1.0 - S.SHY_DELTA, 9)] * 2


def test_battery_r2_groups_are_label_only_and_world_steps_resolve():
    from research.runners import onebrain_regression_battery as B
    probe_labels = {t[0] for t in B.PROBE_TURNS}
    for lab in ("datl_recall", "d3w_recall", "d3c_recall", "d3r_recall"):
        assert lab in B._TURN_BY_LABEL and lab not in probe_labels
    assert B._WORLD_STEPS["datl_awake"] == "awake_4h" and B._WORLD_STEPS["datl_night"] == "overnight_24h"
    assert B._WORLD_STEPS["d3r_night3"] == "overnight_24h" and "d3r_remention1" not in B._WORLD_STEPS
    assert B._WORLD_STEPS["datc_night"] == "overnight_24h"                   # pre-r2 groups unchanged
