"""Session 7 (2026-10-05): the false-belief chat's store-lesion knob and its silent-store abstain.

- `belief_store_read_ok`: an empty or tied store names no location, so the reply abstains (falls through) instead of
  answering the host argmax's slot-0 default, which on the classic probe IS the stale (false-belief) location.
- BRAIN_FALSE_BELIEF_STORE_LESION builds the belief store with its self-recurrence at 0; unset, the build is
  byte-identical to the GO'd bridge (belief_attractor_weight=None).
Prereg: research/findings/2026-10-05-fb-store-lesion-PREREGISTRATION.md.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

import webapp.false_belief_chat as F  # noqa: E402

PROBE = ("Sally puts the marble in the basket. Sally leaves the room. Anne moves the marble to the box. "
         "Where will Sally look for the marble?")


@pytest.mark.parametrize("rates,ok", [
    ({0: 0.333, 1: 0.0}, True),
    ({0: 0.0, 1: 0.0}, False),          # silent store: argmax would say slot 0
    ({0: 0.2, 1: 0.2}, False),          # tie: no single location
    ({}, True),                         # an organ that exports no rates: unchanged behaviour
    (None, True),
])
def test_belief_store_read_ok(rates, ok):
    assert F.belief_store_read_ok(rates) is ok


class _StubOrgan:
    def __init__(self, rates):
        self.rates, self.n_events = rates, 0

    def observe_event(self, loc, witnessed, lesion=False):
        self.n_events += 1
        return loc

    def query(self, lesion=False):
        top = max(self.rates, key=self.rates.get)
        return {"belief_loc": top, "world_loc": 1, "belief_rates": self.rates, "world_rates": {0: 0.0, 1: 0.1},
                "n_events": self.n_events}


@pytest.mark.parametrize("rates,answers", [({0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}, False),
                                           ({0: 0.33, 1: 0.0, 2: 0.0, 3: 0.0}, True)])
def test_silent_store_abstains_through_the_handler(monkeypatch, rates, answers):
    monkeypatch.setattr(F, "_new_organ", lambda seed, lesion: _StubOrgan(dict(rates)))
    monkeypatch.delenv("BRAIN_FALSE_BELIEF_CHAT", raising=False)
    F.reset_session("t-silent")
    info = F.observe_turn("t-silent", PROBE, seed=7, lesion=False)
    if answers:
        assert info["answer"] == "Sally will look in the basket for the marble." and info["reason"] == "queried"
        assert "store_lesioned" not in info, "the default body must not gain a key"
    else:
        assert not info.get("acted") and not info.get("answer") and info["reason"] == "belief_store_silent"


def test_store_lesion_knob_builds_the_cut_organ(monkeypatch):
    monkeypatch.setenv("BRAIN_FALSE_BELIEF_STORE_LESION", "1")
    assert F._new_organ(7, False).belief_attractor_weight == 0.0
    monkeypatch.delenv("BRAIN_FALSE_BELIEF_STORE_LESION")
    assert F._new_organ(7, False).belief_attractor_weight is None


def _conn(bridge):
    from sim.backend import to_host
    c = bridge.cp_connections
    return [np.asarray(to_host(x)) for x in (c.data, c.indices, c.indptr)]


def test_unset_knob_builds_the_identical_bridge():
    import research.runners._false_belief_register_derisk as FB
    a = FB.build_tom_bridge(seed=7)[0]
    b = FB.build_tom_bridge(seed=7, belief_attractor_weight=None)[0]
    for x, y in zip(_conn(a), _conn(b)):
        assert np.array_equal(x, y)


def test_store_lesion_engages():
    """The cut must do something measurable: the belief store's late-window rate falls (dev seed 7)."""
    from research.runners.tom_false_belief_chat_organ import FalseBeliefChatOrgan
    reads = []
    for w in (None, 0.0):
        o = FalseBeliefChatOrgan(seed=7, belief_attractor_weight=w)
        o.observe_event(0, True)
        o.observe_event(1, False)
        reads.append(max(o.query()["belief_rates"].values()))
    assert reads[1] < reads[0]


# ── the registered scorer (tools/fb_store_score.py) ──────────────────────────────────────────────────────────
import json as _json  # noqa: E402
import sys as _sys  # noqa: E402

_sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import fb_store_score as S  # noqa: E402

_BASKET = "Sally will look in the basket for the marble."


def _intact():
    return {"tom_fb": {"answer": _BASKET, "abstained": False, "false_belief_tom": {"acted": True, "answer": _BASKET}}}


def _write(tmp, seed, lesion_body, verdict):
    d = tmp / "tag" / ("s%d" % seed) / S.ROW
    d.mkdir(parents=True)
    (d / "lb.json").write_text(_json.dumps({"per_faculty": [{"faculty": S.ROW, "verdict": verdict,
                                                              "null_control_clean": True, "lesion_reproduced": True}]}))
    (d / "intact_a_tom_fb.json").write_text(_json.dumps(_intact()))
    (d / "intact_b_tom_fb.json").write_text(_json.dumps(_intact()))
    (d / "lesion_tom_false_belief_store.json").write_text(_json.dumps({"tom_fb": lesion_body}))


def test_scorer_not_load_bearing_needs_engagement(tmp_path):
    kept = {"answer": _BASKET, "abstained": False,
            "false_belief_tom": {"acted": True, "answer": _BASKET, "store_lesioned": True}}
    for s in S.SEEDS:
        _write(tmp_path, s, kept, "pass")
    res = S.score("tag", "p", base=str(tmp_path))
    assert res["verdict"] == "STORE-NOT-LOAD-BEARING" and res["preconditions"]


def test_scorer_unengaged_lesion_is_undefined(tmp_path):
    unmarked = {"answer": _BASKET, "abstained": False, "false_belief_tom": {"acted": True, "answer": _BASKET}}
    for s in S.SEEDS:
        _write(tmp_path, s, unmarked, "pass")
    assert S.score("tag", "p", base=str(tmp_path))["verdict"] == "UNDEFINED"


def test_scorer_store_load_bearing(tmp_path):
    silent = {"answer": "I don't know about that.", "abstained": True}
    for s in S.SEEDS:
        _write(tmp_path, s, silent, "regressed")
    assert S.score("tag", "p", base=str(tmp_path))["verdict"] == "STORE-LOAD-BEARING"
