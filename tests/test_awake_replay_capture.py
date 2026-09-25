"""Awake-rest replay for the DA tag-and-capture route (webapp/awake_replay_capture.py, default-OFF
BRAIN_AWAKE_REPLAY_CAPTURE).

No brain build. The fake store's read-back is the Hill curve fitted to the composer margins the seed-42 brain smokes
measured (research/runners/_awake_replay_capture_design.py), applied to the block's CURRENT store synapses: the
expressed increment-to-baseline ratio read off `store_conns` (margin ~0 at baseline, ~0.44 for the fully expressed
neutral telling). The D1 pool is linear from tonic (a = 0) to its ceiling. What is under test is the wiring and the
ledger dynamics:
  * flag OFF -> byte-identical to the pre-branch idle tick (and the lesioned route writes exactly what OFF writes);
  * a fact told 4 h before sleep survives the night when the brain rests, not when the awake edge is cut, not when
    there is no idle tick, not when the route is off, not when the night's replay edge is cut (awake replay supplies
    no PRP), not under the DA-encoding lesion; a salient fact is kept on its waking capture alone;
  * bouts only on idle ticks while awake, at most one per AWAKE_BOUT_H; a turn never runs one.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import textwrap

import numpy as np
import pytest

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, _REPO)

from webapp import awake_replay_capture as A               # noqa: E402
from webapp import da_tag_capture as T                     # noqa: E402
from webapp import da_tag_capture_chat as W                # noqa: E402
from webapp import sleep_replay_capture as S               # noqa: E402

FLAGS = ("BRAIN_DA_TAG_CAPTURE", "BRAIN_DA_TAG_CAPTURE_CLOCK", "BRAIN_SLEEP_REPLAY_CAPTURE",
         "BRAIN_SLEEP_REPLAY_CAPTURE_LESION", "BRAIN_DA_CAPTURE_LESION", "BRAIN_DA_ENCODING_LESION",
         "BRAIN_SLEEP_DOWNSCALING", "BRAIN_AWAKE_REPLAY_CAPTURE", "BRAIN_AWAKE_REPLAY_CAPTURE_LESION")
HILL = (0.5093192874323118, 0.9222631904648466, 2.1069464979187074)   # the design sweep's fit (committed artifact)
R_FULL = 2.19                                                         # the neutral telling's ratio (seed 42, measured)
NEUTRAL = [0.5, 0.5, 0.5, 0.5, 0.5]
SALIENT = [1.0, 1.1, 0.5, 1.1, 1.0]
BOUT_H = 5.0 / 60.0


class FakeD1:
    def __init__(self, *a, **kw):
        self.a_go = self.read(T.prp_threshold())[0]

    def read(self, da):
        return float(np.clip((float(da) - T._DA_TONIC) / (S.DA_SWR_FULL - T._DA_TONIC), 0.0, 1.0)), None


class HillComposer:
    """Block-major store; its read-back is the fitted Hill curve of the expressed ratio read off the store synapses."""

    def __init__(self, D=64, seed=0):
        self.D, self.store_conns, self.n_reads, self.L = D, [], 0, None
        self._rng = np.random.default_rng(seed)

    def store(self, g):
        u = g * np.exp(2j * np.pi * self._rng.random(self.D))
        trig = 1000 + (len(self.store_conns) // self.D) * (self.D + 1)
        self.store_conns += [(trig + 1 + k, trig, complex(u[k])) for k in range(self.D)]

    def ratio(self, i):
        b = self.L.blocks[i - self.L.block_offset]
        w = np.array([complex(x[2]) for x in self.store_conns[i * self.D:(i + 1) * self.D]])
        d = b["inc"] / np.maximum(np.abs(b["inc"]), 1e-12)
        return float(abs(np.mean(np.conj(d) * (w - b["base"]))) / np.mean(np.abs(b["base"])))

    def margin(self, i):
        Rm, c, n = HILL
        x = self.ratio(i)
        return Rm * x ** n / (c ** n + x ** n)

    def _block_role_scores(self, i):
        self.n_reads += 1
        m = self.margin(i)
        return {"agent": ("a", 1.0, m, None), "action": ("b", 1.0, m, None), "patient": ("c", 1.0, m, None),
                "polarity": ("pos", 1.0, 1.0, None)}


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


def _tell(monkeypatch, da_turns=NEUTRAL, env=None, fact_turn=2):
    monkeypatch.setenv("BRAIN_DA_TAG_CAPTURE", "1")
    monkeypatch.setenv("BRAIN_DA_TAG_CAPTURE_CLOCK", "turn")
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    comp = HillComposer()
    comp.store(1.0)                                   # build-time knowledge: unmanaged (block_offset)
    chat = Chat(comp)
    for i, da in enumerate(da_turns):
        chat._last_da_drives = {"da_level": da}
        W.observe_chat_turn(chat, seed=7)
        if i == 0:
            comp.L = chat._da_tag_capture.ledger
        if i == fact_turn:
            b = comp.L._baseline(1, comp.D)
            comp.store(R_FULL * float(np.mean(np.abs(b))))
        W.after_store_chat(chat)
    return chat, chat._da_tag_capture, comp


def _rest(chat, hours=4.0, period_h=BOUT_H, tick=True):
    """The battery's awake-rest world step: per period the clock moves, the body is marked awake, one idle tick."""
    for _ in range(int(round(hours / period_h))):
        W.advance_world_clock_h(period_h)
        W.mark_awake(chat)
        if tick:
            W.tick_chat(chat)


def _awake_no_rest(chat, hours=4.0):
    W.advance_world_clock_h(hours)
    W.mark_awake(chat)


def _night(chat):
    W.advance_world_clock_h(24.0)
    W.tick_chat(chat)


def _kept(cap):
    return cap.ledger.summary()[0]["frac_synapses_z_gt_half"] > 0.9


RC = {"BRAIN_SLEEP_REPLAY_CAPTURE": "1"}
ARC = {"BRAIN_AWAKE_REPLAY_CAPTURE": "1"}


# ── flag OFF: byte-identical ────────────────────────────────────────────────────────────────────────────────────────
_PRE_BRANCH_TICK = textwrap.dedent('''
    def tick(self, chat):
        comp = store_composer(chat)
        t = max(self.now_h(), self.ledger.t)
        self._catch_up(comp, t)
        return t
''')


def _pre_branch_tick():
    ns = {"store_composer": W.store_composer}
    exec(_PRE_BRANCH_TICK, ns)
    return ns["tick"]


@pytest.mark.parametrize("rc", [False, True])
def test_flag_off_is_identical_to_the_pre_branch_idle_tick(monkeypatch, rc):
    env = RC if rc else {}
    chat, cap, comp = _tell(monkeypatch, env=env)
    _rest(chat); _night(chat)
    h_new = _store_hash(comp)
    assert not hasattr(cap, "_arc") and "awake_replay_capture" not in W.after_store_chat(chat)
    assert all("e_rep" not in b and "e_rep" not in s for b, s in zip(cap.ledger.blocks, cap.ledger.summary()))
    monkeypatch.setattr(W, "_WORLD_OFFSET_H", 0.0)
    monkeypatch.setattr(W.ChatTagCapture, "tick", _pre_branch_tick())       # the method as it was before this branch
    chat2, cap2, comp2 = _tell(monkeypatch, env=env)
    _rest(chat2); _night(chat2)
    assert _store_hash(comp2) == h_new


def test_the_early_expression_refactor_is_the_old_formula(monkeypatch):
    L = T.SynapticTagCaptureLedger(7, gamma=10.0)
    blk = {"t_w": 0.3, "z": np.array([0.0, 0.2, 1.0])}
    for t in (0.0, 0.3, 1.0, 5.0, 30.0):
        L.t = t
        e = np.exp(-max(0.0, t - 0.3) / T.TAU_EARLY_H)
        assert np.array_equal(L.weight_factor(blk), e + blk["z"] * (1.0 - e))


def test_awake_lesion_writes_exactly_what_the_flag_off_path_writes(monkeypatch):
    chat, cap, comp = _tell(monkeypatch, env=RC)
    _rest(chat); _night(chat)
    h_off = _store_hash(comp)
    monkeypatch.setattr(W, "_WORLD_OFFSET_H", 0.0)
    chat2, cap2, comp2 = _tell(monkeypatch, env={**RC, **ARC, "BRAIN_AWAKE_REPLAY_CAPTURE_LESION": "1"})
    n0 = comp2.n_reads
    _rest(chat2); _night(chat2)
    arc = cap2._arc.summary()
    assert arc["n_bouts"] == 48 and comp2.n_reads > n0                      # the reads ran ...
    assert all(r == 0.0 for x in arc["bouts"] for r in x["R_eff"])          # ... their edge is cut ...
    assert _store_hash(comp2) == h_off                                      # ... and the store is the OFF store


def test_inert_without_the_tag_capture_ledger(monkeypatch):
    monkeypatch.setenv("BRAIN_AWAKE_REPLAY_CAPTURE", "1")
    comp = HillComposer(); comp.store(1.0)
    chat = Chat(comp)
    chat._last_da_drives = {"da_level": 0.5}
    assert W.observe_chat_turn(chat, seed=7) is None and W.tick_chat(chat) is None
    assert not hasattr(chat, "_da_tag_capture") and comp.n_reads == 0


# ── flag ON: the rescue comes from rest ─────────────────────────────────────────────────────────────────────────────
def test_long_delay_fact_lost_without_the_route(monkeypatch):
    chat, cap, comp = _tell(monkeypatch, env=RC)
    _rest(chat); _night(chat)
    assert not _kept(cap) and comp.margin(1) < 0.01
    assert cap._src.summary()["epochs"][0]["R"][0] < 0.05                  # the Amendment-1 wall: unreadable at onset


def test_rest_keeps_a_long_delay_fact_capturable_and_the_night_captures_it(monkeypatch):
    chat, cap, comp = _tell(monkeypatch, env={**RC, **ARC})
    _rest(chat)
    arc = cap._arc.summary()
    assert arc["n_bouts"] == 48 and not any(x["no_reader"] for x in arc["bouts"])
    assert arc["bouts"][-1]["early_after"][0] > 0.8                         # still expressed after 4 h of rest
    _night(chat)
    ep = cap._src.summary()["epochs"]
    assert len(ep) == 1 and ep[0]["t_h"] > arc["bouts"][-1]["t_h"]         # one sleep epoch, after the last bout
    assert ep[0]["R"][0] > 0.3 and ep[0]["da_swr"] > T.prp_threshold()
    assert _kept(cap) and comp.margin(1) > 0.3


def test_no_idle_tick_no_rescue(monkeypatch):
    chat, cap, comp = _tell(monkeypatch, env={**RC, **ARC})
    _awake_no_rest(chat)
    _night(chat)
    assert cap._arc.summary()["n_bouts"] == 0 and cap._arc.summary()["n_ticks_asleep"] == 1
    assert not _kept(cap) and comp.margin(1) < 0.01


def test_awake_edge_lesion_removes_the_rescue(monkeypatch):
    chat, cap, comp = _tell(monkeypatch, env={**RC, **ARC, "BRAIN_AWAKE_REPLAY_CAPTURE_LESION": "1"})
    _rest(chat); _night(chat)
    assert not _kept(cap)


def test_awake_bouts_supply_no_prp_the_night_does(monkeypatch):
    chat, cap, comp = _tell(monkeypatch, env={**RC, **ARC, "BRAIN_SLEEP_REPLAY_CAPTURE_LESION": "1"})
    n_drive = len(cap.ledger.drive)
    _rest(chat)
    bouts = cap._arc.summary()["bouts"]
    assert len(cap.ledger.drive) == n_drive                                 # no D1 drive was added during rest
    assert all(b["n_drive_entries"] == n_drive for b in bouts)
    assert all(y["p_at_bout"] <= x["p_at_bout"] for x, y in zip(bouts, bouts[1:]))
    assert cap.ledger.summary()[0]["frac_synapses_z_gt_half"] == 0.0       # held in early phase, not captured awake
    _night(chat)
    assert not _kept(cap)                                                   # the night's replay edge cut -> lost


def test_da_encoding_lesion_blocks_the_rescue(monkeypatch):
    chat, cap, comp = _tell(monkeypatch, env={**RC, **ARC, "BRAIN_DA_ENCODING_LESION": "1"})
    _rest(chat); _night(chat)
    assert cap._src.summary()["epochs"][0]["da_seen_by_d1"] == T._DA_TONIC
    assert not _kept(cap)


def test_salient_fact_kept_on_its_waking_capture_with_the_night_edge_cut(monkeypatch):
    chat, cap, comp = _tell(monkeypatch, da_turns=SALIENT, env={**RC, **ARC, "BRAIN_SLEEP_REPLAY_CAPTURE_LESION": "1"})
    _rest(chat)
    assert cap.ledger.summary()[0]["frac_synapses_z_gt_half"] > 0.9        # captured while awake (waking DA)
    _night(chat)
    assert _kept(cap)


def test_hourly_rest_is_a_smaller_dose(monkeypatch):
    chat, cap, comp = _tell(monkeypatch, env={**RC, **ARC})
    _rest(chat, period_h=1.0)
    assert cap._arc.summary()["n_bouts"] == 4
    e_hourly = cap._arc.summary()["bouts"][-1]["early_after"][0]
    monkeypatch.setattr(W, "_WORLD_OFFSET_H", 0.0)
    chat2, cap2, comp2 = _tell(monkeypatch, env={**RC, **ARC})
    _rest(chat2)
    assert cap2._arc.summary()["bouts"][-1]["early_after"][0] > e_hourly


# ── scheduling ──────────────────────────────────────────────────────────────────────────────────────────────────────
def test_a_turn_never_runs_a_bout(monkeypatch):
    chat, cap, comp = _tell(monkeypatch, env={**RC, **ARC})
    assert not hasattr(cap, "_arc") and comp.n_reads == 0


def test_bouts_only_while_awake_and_at_most_one_per_bout_window(monkeypatch):
    comp = HillComposer()
    d1 = FakeD1()
    L = T.SynapticTagCaptureLedger(7, gamma=T.calibrate_gamma(d1.a_go), d1=d1)
    comp.L = L
    L.observe_turn(0.0, W.TURN_DRIVE_H, 0.5); comp.store(2.0); L.on_store(comp, 0.0)
    arc = A.AwakeReplayCapture(7)
    onset = S.sleep_onset_h()
    assert arc.maybe_bout(L, comp, 0.0, 0.01) is True                       # idle, awake -> a bout
    assert arc.maybe_bout(L, comp, 0.0, 0.01 + A.AWAKE_BOUT_H / 2) is False  # inside the 5-min window -> none
    assert arc.maybe_bout(L, comp, 0.0, onset + 1e-6) is False              # past sleep onset -> asleep, none
    assert arc.maybe_bout(L, comp, 1.0, 1.0) is True                        # awake mark at 1 h -> awake again
    s = arc.summary()
    assert s["n_bouts"] == 2 and s["n_ticks_rate_limited"] == 1 and s["n_ticks_asleep"] == 1


def test_selection_is_the_substrate_read_every_block_driven(monkeypatch):
    """An old fact and a fresh one: both triggers are driven; the fresh one reads high and is re-potentiated a lot,
    the old one reads near zero and is barely touched. No list or ranking chooses."""
    comp = HillComposer()
    d1 = FakeD1()
    L = T.SynapticTagCaptureLedger(7, gamma=T.calibrate_gamma(d1.a_go), d1=d1)
    comp.L = L
    for t in (0.0, 5.0):
        L.observe_turn(t, W.TURN_DRIVE_H, 0.5)
        comp.store(R_FULL * float(np.mean(np.abs(L._baseline(len(L.blocks), comp.D)))))
        L.on_store(comp, t)
    arc = A.AwakeReplayCapture(7)
    assert arc.maybe_bout(L, comp, 5.0, 5.05)
    x = arc.summary()["bouts"][0]
    assert comp.n_reads == 2 and x["R"][1] > 0.4 and x["R"][0] < 0.01 and x["R"][1] > 100 * x["R"][0]
    assert x["early_after"][1] > 0.95 and x["early_after"][0] < 0.05       # the old trace stays near baseline
    assert x["early_after"][0] - x["early_before"][0] < 0.01


def test_constants_are_reused_not_new():
    assert A.AWAKE_BOUT_H == S.SWR_BOUT_H == T.CAPTURE_PROTOCOL_MIN / 60.0
    assert A.reactivation_strength is S.reactivation_strength
    assert "e_rep" not in inspect.getsource(T.SynapticTagCaptureLedger._integrate)   # z / tag ODEs untouched


def test_battery_arc_groups_are_label_only_and_world_steps_resolve():
    from research.runners import onebrain_regression_battery as B
    from research.runners.load_bearing_fraction import turn_group
    probe_labels = {t[0] for t in B.PROBE_TURNS}
    for lab in ("datr_recall", "datcr_recall", "datq_recall", "datz_recall"):
        assert lab in B._TURN_BY_LABEL and lab not in probe_labels
    assert B._WORLD_STEPS["datr_rest"] == "awake_rest_4h" and B._WORLD_STEPS["datcr_rest"] == "awake_rest_4h"
    assert B._WORLD_STEPS["datq_rest"] == "awake_rest_hourly_4h"
    assert B._WORLD_STEPS["datz_awake"] == "awake_3h" and B._WORLD_STEPS["datz_rest"] == "awake_rest_1h"
    assert B._WORLD_STEPS["datl_awake"] == "awake_4h"                                  # r2 group unchanged
    ticks = {k: int(round(h / p)) for k, (h, p) in B._AWAKE_STEP_KINDS.items() if p}
    assert ticks == {"awake_rest_4h": 48, "awake_rest_hourly_4h": 4, "awake_rest_1h": 12}
    assert B._AWAKE_STEP_KINDS["awake_rest_4h"][1] == A.AWAKE_BOUT_H
    grp = turn_group("datz_recall")
    assert [x for x in grp if x in B._WORLD_STEPS] == ["datz_awake", "datz_rest", "datz_night"]
    from research.runners import _da_tag_capture_chat_probe as P
    assert all(P.ARC_BOUTS[lab] == {"datr_recall": 48, "datcr_recall": 48, "datq_recall": 4, "datz_recall": 12,
                                    "datl_recall": 0, "datni_recall": 0}[lab] for _n, lab, _e in P.ARC_ARMS)
