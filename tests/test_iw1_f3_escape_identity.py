"""F3 (integration wave 1): the explicit-off escape, asserted PIN-LOCALLY on the recorded turns.

RULING iw1-f3-1023 (2026-10-23) diagnosis: F3 as first scored compared arm=off at pin F (957c2752c) against
arm=default at pin P (049d4bb) -- a CROSS-PIN comparison. The recorded evidence shows the turns-7-12 sha_canon
delta is shared-code drift, not the packet: off@F vs off@P shows the SAME 7-12-only pattern with both packets
explicitly 0 at both pins, and off@P == default@P on every turn; between the pins the ungated body writers
moved (webapp/server.py affect at :5158-5175 and worldmodel at :5762/:5798-5819, fed by
affect_production_organ.py +27/-3 and sim/bridge.py +62, sim/kernels.py +156, sim/config.py +27).

What IS asserted here, on the recorded turns:
  (a) pin P: off == default, 13/13 sha_canon (the escape reproduced the pre-flip default on pre-flip code);
  (b) pin F: on:<flipped set> == default, 13/13 sha_canon (the flipped default IS its explicit-on arm);
  (c) cross-pin off@F vs off@P (opt-in --allow-cross-pin): every reply (answer/abstained/recalled_svo) identical
      on all 13 turns, sha_canon delta confined to turns 7-12 (valence/worldmodel turns), neutral 0-6 equal;
  (d) the same cross-pin comparison WITHOUT the opt-in is UNDEFINED (the guard that stops this FAIL recurring);
  (e) the flag gate itself: explicit 0/1 flips xedge's surprise-worldmodel reader whatever the default is.

NOT asserted: byte-identity of turns 7-12 across pins -- the re-anchoring of that expectation is Claude's call
(NEEDS [iw1-f3-reanchor]), and no packet defect is claimed (xedge's only pin-span change is its default line).
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
IDENT = os.path.join(REPO, "research", "findings", "raw", "_iw1", "identity")
PIN_P, PIN_F = "049d4bb", "957c275"
VALANCE_TURNS = set(range(7, 13))          # recorded turns where the pin-span drift shows (POS/NEG + expectation)
FLIPPED_AT_F = "on_d6-hebbian-write_xedge-surprise-worldmodel.json"
_XEDGE_FLAG = "BRAIN_ONEBRAIN_XEDGE_SURPRISE_WORLDMODEL"


def _load(pin, name):
    from research.runners import iw1_default_identity as t
    path = os.path.join(IDENT, pin, name)
    assert os.path.exists(path), "missing recorded arm: " + path
    return t.load_record(path)


def test_a_pin_p_off_equals_default_13_of_13():
    from research.runners import iw1_default_identity as t
    v = t.compare(_load(PIN_P, "off.json"), _load(PIN_P, "default.json"), "same")
    assert v["verdict"] == "IDENTICAL" and len(v["differing_turns"]) == 0
    assert v["confounded_cross_pin"] is False


def test_b_pin_f_flipped_default_equals_explicit_on_13_of_13():
    from research.runners import iw1_default_identity as t
    v = t.compare(_load(PIN_F, FLIPPED_AT_F), _load(PIN_F, "default.json"), "same")
    assert v["verdict"] == "IDENTICAL" and len(v["differing_turns"]) == 0


def test_c_cross_pin_replies_identical_on_all_turns_delta_confined_to_valence():
    from research.runners import iw1_default_identity as t
    v = t.compare(_load(PIN_F, "off.json"), _load(PIN_P, "off.json"), "differs", allow_cross_pin=True)
    assert v["confounded_cross_pin"] is True
    assert v["verdict"] == "DIFFERS"
    assert all(p for p in v["preconditions"].values())
    assert set(v["differing_turns"]) and set(v["differing_turns"]) <= VALANCE_TURNS
    assert v["reply_differing_turns"] == []       # the human-visible reply is byte-identical on all 13 turns


def test_d_cross_pin_same_is_undefined_without_opt_in():
    from research.runners import iw1_default_identity as t
    v = t.compare(_load(PIN_F, "off.json"), _load(PIN_P, "off.json"), "same")
    assert v["verdict"] == "UNDEFINED" and v["ok"] is None
    assert v["preconditions"]["pin_comparable"] is False
    assert v["confounded_cross_pin"] is True


def test_e_explicit_off_reader_beats_the_default(monkeypatch):
    from research.runners import onebrain_xedge_surprise_worldmodel_production as P
    assert callable(P.xedge_surprise_worldmodel_enabled)
    monkeypatch.delenv(_XEDGE_FLAG, raising=False)
    unset = P.xedge_surprise_worldmodel_enabled()
    monkeypatch.setenv(_XEDGE_FLAG, "0")
    assert P.xedge_surprise_worldmodel_enabled() is False      # explicit 0 always wins
    monkeypatch.setenv(_XEDGE_FLAG, "1")
    assert P.xedge_surprise_worldmodel_enabled() is True
    monkeypatch.setenv(_XEDGE_FLAG, "off")
    assert P.xedge_surprise_worldmodel_enabled() is False
    assert unset in (True, False)                               # unset follows the pin's default (either is legal)