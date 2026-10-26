"""Tests for the C3 topic stop (L) and topic-locked phasic DA (P): research/runners/curiosity_topic_stop_organ.py and
webapp/curiosity_topic_stop_chat.py (packet research/coordination/c3-topic-stop-PACKET.md).

Fast tests build the 450-neuron topic-stop organ on numpy (about 0.6 s each) and drive `decide()` directly with a
stand-in conversation Context and a stand-in shared-pool judge result; they check that the per-topic state persists
within a conversation, is isolated between conversations, and depresses only the presented topic. The handler-turn
test (flag-off footprint on one real `/api/brain-chat` turn) builds the tiny-demo brain and is opt-in:
`C3TS_HANDLER_TEST=1`.
"""
from __future__ import annotations

import os
import types

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

import pytest  # noqa: E402

import webapp.curiosity_topic_stop_chat as TS  # noqa: E402
import research.runners.curiosity_topic_stop_organ as ORG  # noqa: E402

SEED = 7
NOVEL = 0.97
_FLAGS = ("BRAIN_CURIOSITY_TOPIC_STOP", "BRAIN_CURIOSITY_TOPIC_STOP_LESION", "BRAIN_CURIOSITY_PHASIC_DA",
          "BRAIN_CURIOSITY_PHASIC_DA_LESION")


def _judge(want=38.9):
    return {"want_hz": want, "threshold": 19.1, "curious": want >= 19.1,
            "calib": {"want_novel_hz": 38.9, "want_familiar_hz": 0.0}}


@pytest.fixture
def flags(monkeypatch):
    for k in _FLAGS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BRAIN_CURIOSITY_TOPIC_STOP", "1")
    return monkeypatch


def _turn(ctx, topic, novelty=NOVEL, lesioned=False):
    TS.note_turn(ctx)
    return TS.decide(ctx, None, topic, novelty, _judge(), SEED, lesioned)


def test_flags_default_off(monkeypatch):
    for k in _FLAGS:
        monkeypatch.delenv(k, raising=False)
    assert not TS.topic_stop_enabled() and not TS.topic_stop_lesioned()
    assert not TS.phasic_da_enabled() and not TS.phasic_da_lesioned()
    for v in ("1", "true", "on", "yes"):
        monkeypatch.setenv("BRAIN_CURIOSITY_TOPIC_STOP", v)
        assert TS.topic_stop_enabled()
    for v in ("0", "", "off", "no", "false"):
        monkeypatch.setenv("BRAIN_CURIOSITY_TOPIC_STOP", v)
        assert not TS.topic_stop_enabled()


def test_state_persists_within_a_conversation_and_stops_the_repeated_topic(flags):
    ctx = types.SimpleNamespace()
    recs = []
    for k in range(4):
        recs.append(_turn(ctx, "toad"))
        TS.note_turn(ctx)                     # an intervening control turn (the battery's c_dog_k)
    org = getattr(ctx, TS.CTX_ATTR).organ
    assert all(r["applied"] for r in recs)
    assert [r["stop"]["presentations"] for r in recs] == [1, 2, 3, 4]
    xs = [r["slice_pre"]["x_mean"] for r in recs]
    assert xs[0] == pytest.approx(1.0) and xs[0] > xs[1] > xs[2] > xs[3]
    assert recs[0]["curious"] is True
    assert recs[3]["curious"] is False         # stopped by the 4th presentation
    assert len({r["slot"] for r in recs}) == 1
    assert recs[3]["idles_run"] == 2            # one control turn + this turn's own inter-turn gap
    assert getattr(ctx, TS.CTX_ATTR).organ is org and recs[0]["held"]["ou_enabled"] is False
    assert recs[0]["held"]["stp_tau_d_ms"] == pytest.approx(ORG.TAU_HAB_MS)


def test_state_is_isolated_between_conversations(flags):
    a, b = types.SimpleNamespace(), types.SimpleNamespace()
    for _ in range(3):
        ra = _turn(a, "toad")
    rb = _turn(b, "toad")
    assert getattr(a, TS.CTX_ATTR).organ is not getattr(b, TS.CTX_ATTR).organ
    assert ra["stop"]["presentations"] == 3 and rb["stop"]["presentations"] == 1
    assert rb["slice_pre"]["x_mean"] == pytest.approx(1.0) and rb["curious"] is True
    assert ra["slice_pre"]["x_mean"] < 0.6


def test_presentation_depresses_only_the_presented_topic(flags):
    ctx = types.SimpleNamespace()
    _turn(ctx, "toad"); _turn(ctx, "toad")
    org = getattr(ctx, TS.CTX_ATTR).organ
    toad_slot = org._slot_of["toad"]
    before = {k: org.slice_state(k)["x_mean"] for k in range(ORG.N_SLOTS) if k != toad_slot}
    assert all(v == pytest.approx(1.0) for v in before.values())
    r = _turn(ctx, "eel")
    assert r["slot"] != toad_slot and r["slice_pre"]["x_mean"] == pytest.approx(1.0) and r["curious"] is True
    assert org.slice_state(toad_slot)["x_mean"] < 0.7
    others = [org.slice_state(k)["x_mean"] for k in range(ORG.N_SLOTS) if k not in (toad_slot, r["slot"])]
    assert all(v == pytest.approx(1.0) for v in others)


def test_familiar_topic_drives_nothing_and_is_not_depressed(flags):
    ctx = types.SimpleNamespace()
    r = _turn(ctx, "yak", novelty=0.0)
    assert r["want_hz"] == 0.0 and r["curious"] is False and r["afferent_spikes"] == 0
    assert r["slice_post"]["x_mean"] == pytest.approx(1.0)


def test_lesion_stop_keeps_asking_and_is_read_back(flags):
    flags.setenv("BRAIN_CURIOSITY_TOPIC_STOP_LESION", "1")
    ctx = types.SimpleNamespace()
    recs = []
    for _ in range(4):
        recs.append(_turn(ctx, "toad"))
        TS.note_turn(ctx)
    assert all(r["curious"] for r in recs)
    assert recs[3]["held"]["stp_tau_d_ms"] == pytest.approx(ORG.TAU_LESION_MS) and recs[3]["lesion_stop"] is True


def test_curiosity_lesion_cuts_the_topic_drive(flags):
    ctx = types.SimpleNamespace()
    r = _turn(ctx, "eel", lesioned=True)
    assert r["want_hz"] == 0.0 and r["curious"] is False and r["held"]["topic_ask_weight_sum"] == 0.0


def test_no_topic_keeps_the_production_decision(flags):
    ctx = types.SimpleNamespace()
    TS.note_turn(ctx)
    r = TS.decide(ctx, None, None, NOVEL, _judge(), SEED, False)
    assert r["applied"] is False and getattr(ctx, TS.CTX_ATTR).organ is None


def test_capacity_rule_recycles_the_least_recent_slice(flags):
    ctx = types.SimpleNamespace()
    topics = ["t%02d" % i for i in range(ORG.N_SLOTS - 1)]
    for t in topics:
        _turn(ctx, t)
    _turn(ctx, topics[0])                      # t00 is now the most recent; t01 is the least recent
    r = _turn(ctx, "newcomer")
    org = getattr(ctx, TS.CTX_ATTR).organ
    assert "t01" not in org._slot_of and org._slot_of["newcomer"] == r["slot"]
    assert r["slice_pre"]["x_mean"] == pytest.approx(1.0) and r["curious"] is True
    assert org.recycled[-1]["from"] == "t01"


def test_salience_from_judge_matches_the_organ_formula():
    assert TS.salience_from_judge(_judge(38.9)) == pytest.approx(1.0)
    assert TS.salience_from_judge(_judge(0.0)) == pytest.approx(0.0)
    assert TS.salience_from_judge({"want_hz": 5.0, "calib": {"want_novel_hz": 1.0, "want_familiar_hz": 1.0}}) == 0.0


def test_organ_selftest():
    ok, checks = ORG.selftest()
    assert ok, checks


@pytest.mark.skipif(os.environ.get("C3TS_HANDLER_TEST") != "1", reason="builds the tiny-demo brain; opt-in")
def test_flag_off_handler_turn_has_no_topic_stop_footprint(monkeypatch):
    """One real abstain turn through `/api/brain-chat` with every flag unset: no `curiosity_topic_stop` key, no
    Context attribute, and `decide`/`note_turn` are never reached (patched to raise)."""
    pytest.importorskip("fastapi"); pytest.importorskip("httpx")
    for k in _FLAGS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SIM_BACKEND", "numpy")

    def _boom(*a, **k):
        raise AssertionError("topic-stop code reached with the flag off")
    monkeypatch.setattr(TS, "decide", _boom)
    monkeypatch.setattr(TS, "note_turn", _boom)
    from fastapi.testclient import TestClient
    from webapp import server as srv
    client = TestClient(srv.app)
    sess = "pytest-c3ts-off"
    res = client.post("/api/brain-chat", json={"session": sess, "brain": "tiny-demo", "renderer": "stub",
                                               "message": "what does the eel chase", "reset": True})
    assert res.status_code == 200, res.text
    d = res.json()
    assert d["abstained"] is True
    assert "curiosity_topic_stop" not in (d.get("curiosity") or {})
    key = (sess, "tiny-demo", "stub")
    ctx = srv._CONTEXTS.get(key)
    assert ctx is None or not hasattr(ctx, TS.CTX_ATTR)
    client.post("/api/brain-chat/reset", json={"session": sess, "brain": "tiny-demo", "renderer": "stub"})


def test_phasic_da_feeds_the_gain_and_lesion_p_pins_the_phasic_afferent(flags):
    """P method B: at the floor tonic a fully salient topic's SNc read restores da_eff >= 0.5 (gain >= 1); lesion_P
    pins the phasic afferent to 0 (no reward_us spike fires, read back), da_eff falls to 0 and the gain to its clip."""
    flags.setenv("BRAIN_CURIOSITY_PHASIC_DA", "1")
    floor = {"_last_da_drives": {"da_level": ORG.DA_FLOOR_PROBE}}
    ctx = types.SimpleNamespace()
    TS.note_turn(ctx)
    r = TS.decide(ctx, types.SimpleNamespace(**floor), "eel", NOVEL, _judge(38.9), SEED, False)
    ph = r["phasic_da"]
    assert r["phasic_da_applied"] and ph["salience"] == pytest.approx(1.0) and ph["reward_us_spikes"] > 0
    assert ph["da_eff"] >= 0.5 and ph["da_crave_gain"] >= 1.0 and r["curious"] is True
    assert r["tonic_da"]["i_tonic_pa"] == pytest.approx(ORG.K_TONIC_PA * ORG.DA_FLOOR_PROBE)
    flags.setenv("BRAIN_CURIOSITY_PHASIC_DA_LESION", "1")
    ctx2 = types.SimpleNamespace()
    TS.note_turn(ctx2)
    r2 = TS.decide(ctx2, types.SimpleNamespace(**floor), "eel", NOVEL, _judge(38.9), SEED, False)
    ph2 = r2["phasic_da"]
    assert ph2["lesioned"] and ph2["i_phasic_pa"] == 0.0 and ph2["reward_us_spikes"] == 0
    assert ph2["da_eff"] == pytest.approx(0.0) and ph2["da_crave_gain"] == pytest.approx(0.25)
    assert r2["curious"] is False and r2["want_hz"] == pytest.approx(r["want_hz"])


def test_phasic_da_frozen_constants_match_the_design_artifact():
    import json
    d = json.load(open(os.path.join(ORG._REPO, ORG.DESIGN_FROZEN["artifact"])))
    assert d["chosen"]["S_DA_PA"] is None and ORG.DESIGN_FROZEN["S_DA_PA"] is None
    assert d["chosen"]["G_PHASIC_PA"] == ORG.DESIGN_FROZEN["G_PHASIC_PA"] == 600.0
