"""BRAIN_MULTIREF_FOCUS_BIND (the spiking referent->focus binding + pronoun resolution of the D6 multi-referent WM
organ). Private-organ tests (no full brain build; ~1 s each on numpy), dev seed 7 only.

Pre-registration: research/findings/2026-09-24-wm-referent-focus-bind-anaphor-probe-PREREGISTRATION.md."""
import os
import types

import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

import research.runners.d6_multiref_wm_production_organ as D6  # noqa: E402

SEED = 7   # a dev seed, outside the evaluation set {42, 43, 44, 100, 101, 102}


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv("BRAIN_MULTIREF_FOCUS_BIND", "1")
    monkeypatch.delenv("BRAIN_MULTIREF_LESION", raising=False)
    yield


def _wipe(org):
    """Another session / organ reset the whole bridge between turns."""
    from research.runners._d3_persistent_slot_derisk import _reset
    _reset(org.buf.sb)


def test_flag_off_is_inert(monkeypatch):
    monkeypatch.delenv("BRAIN_MULTIREF_FOCUS_BIND", raising=False)
    assert D6.multiref_focus_bind_enabled() is False
    org = D6.MultiReferentWMOrgan(seed=SEED, shared=None)
    res = org.load(["dog", "cat"])
    assert res["all_recovered"] is True
    assert not org.has_held_state()
    assert org.resolve_anaphor("it") is None
    assert org.read_held() is None
    chat = types.SimpleNamespace(_is_anaphor_token=lambda t: t == "it")
    assert D6.resolve_turn(chat, org, "what does it chase") is None
    assert not hasattr(chat, "_multiref_referent_override")


def test_held_state_survives_a_bridge_wipe_and_resolves_to_a_held_referent(flag_on):
    names = {}
    for order in (["dog", "cat"], ["cat", "dog"]):
        org = D6.MultiReferentWMOrgan(seed=SEED, shared=None)
        org.load(order)
        assert org.has_held_state()
        _wipe(org)
        r = org.resolve_anaphor("it")
        assert r["kind"] == "resolve" and r["resolved"] in order
        names[order[0]] = (r["resolved_register"], r["resolved"])
        h = org.read_held()
        assert sorted(h["recovered"].values()) == ["cat", "dog"]      # retrieval read the buffer, did not clear it
    # the SAME register wins in both sessions (same pools), so the order swap changes WHICH referent it holds
    assert names["dog"][0] == names["cat"][0]
    assert names["dog"][1] != names["cat"][1]


def test_dead_or_empty_buffer_resolves_nothing(flag_on, monkeypatch):
    org = D6.MultiReferentWMOrgan(seed=SEED, shared=None)
    org.ensure_built()
    org.buf.reset()
    org._stash(org.buf)
    assert org.resolve_anaphor("it")["resolved"] is None
    monkeypatch.setenv("BRAIN_MULTIREF_LESION", "1")
    les = D6.MultiReferentWMOrgan(seed=SEED, shared=None)
    assert les.load(["dog", "cat"], lesion=True)["hold_alive_min"] == 0.0
    r = les.resolve_anaphor("it", lesion=True)
    assert r["resolved"] is None and r["resolved_register"] is None


def test_resolve_turn_publishes_a_per_turn_override(flag_on):
    org = D6.MultiReferentWMOrgan(seed=SEED, shared=None)
    org.load(["dog", "cat"])
    chat = types.SimpleNamespace(_is_anaphor_token=lambda t: t in ("it", "they"))
    r = D6.resolve_turn(chat, org, "what does it chase")
    ovr = chat._multiref_referent_override
    assert ovr == {"question": "what does it chase", "pronoun": "it", "referent": r["resolved"]}
    assert D6.resolve_turn(chat, org, "what does the dog chase") is None     # no anaphor -> out of scope


def test_chatbrain_resolve_anaphora_honors_the_override_only_for_its_own_turn():
    from research.runners.brain_chat_tui import ChatBrain
    from webapp.gnw_bus_shadow import _multiref_resolved
    cb = types.SimpleNamespace(is_multiturn=False)
    cb._multiref_referent_override = {"question": "what does it chase", "pronoun": "it", "referent": "cat"}
    assert ChatBrain._resolve_anaphora(cb, "what does it chase") == "what does cat chase"
    assert ChatBrain._resolve_anaphora(cb, "what does it eat") == "what does it eat"   # a different turn's text
    assert ChatBrain._multiref_resolved_turn(cb) is True and _multiref_resolved(cb) is True
    cb._multiref_referent_override = {"question": "what does it chase", "pronoun": "it", "referent": None}
    assert ChatBrain._resolve_anaphora(cb, "what does it chase") == "what does it chase"
    assert ChatBrain._multiref_resolved_turn(cb) is False and _multiref_resolved(cb) is False
    plain = types.SimpleNamespace(is_multiturn=False)
    assert ChatBrain._resolve_anaphora(plain, "what does it chase") == "what does it chase"
    assert ChatBrain._multiref_resolved_turn(plain) is False and _multiref_resolved(plain) is False
