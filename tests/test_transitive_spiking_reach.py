"""Packet `transitive-spiking-reach` (2026-10-01): the spiking reachability organ, its default-OFF wire-in in
webapp/reasoning_transitive_chat.py, and its LBF rows.

Pins: (1) flag off -> `resolve_transitive_query` takes the unchanged host-chase path and never touches the organ;
(2) the production-default anchor reads False; (3) the LBF row module's shape checks and registry pickup;
(4) the organ battery on ONE dev seed at op 0 (a small real build, a few seconds) returns the documented keys and
passes; (5) the CLI refuses gate seeds. The wire-in's reached / abstain / lesion / op routing is pinned against a stub
organ (no brain build), and `reach_from_composer` runs once on a tiny fake composer.
"""
import json
import os
import sys
import types

import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")
os.environ.setdefault("SIM_NO_PROVENANCE", "1")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import webapp.reasoning_transitive_chat as T  # noqa: E402
# imported at COLLECTION time: a later test file that puts tools/ on sys.path makes `experiment` resolve to
# tools/experiment.py, which breaks a first `import sim` inside a test (the triage's POLLUTION class)
import research.runners.transitive_spiking_reach_organ  # noqa: E402,F401
from research.runners.compositional_chain_route import ChainedSVO  # noqa: E402

FLAGS = ("BRAIN_TRANSITIVE_CHAT", "BRAIN_TRANSITIVE_SPIKING_REACH", "BRAIN_TRANSITIVE_REACH_LESION",
         "BRAIN_TRANSITIVE_REACH_OP", "BRAIN_TRANSITIVE_LESION")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for f in FLAGS:
        monkeypatch.delenv(f, raising=False)


class _FakeComposer:
    def __init__(self, facts):
        self.facts = dict(facts)
        self.kb = [({"agent": a, "action": act, "patient": p}, None) for (a, act), p in self.facts.items()]

    def query_patient(self, agent, action):
        return self.facts.get((agent, action))


class _FakeChat:
    def __init__(self, facts):
        self.inner = types.SimpleNamespace(composer=_FakeComposer(facts))


def _chain_facts():
    return {("cat", "precede"): "ball", ("ball", "precede"): "worm", ("worm", "precede"): "river"}


def _install_fake_chase(monkeypatch, path):
    """The host chase the flag-off path must still use: a fake keystone chase returning a fixed ADVANCE trace."""
    trace = [{"action": "ADVANCE", "committed": c} for c in path[1:]] + [{"action": "COMMIT", "committed": None}]
    mod = types.ModuleType("webapp.gnw_multistep_deliberation")
    mod.multistep_chase = lambda chat, agent, action, *, seed=42, lesion=False: (path[-1], {"trace": trace})
    monkeypatch.setitem(sys.modules, "webapp.gnw_multistep_deliberation", mod)


# ── (2) the anchor ───────────────────────────────────────────────────────────────────────────────────────────────
def test_anchor_reads_false():
    assert T._TRANSITIVE_SPIKING_REACH_DEFAULT_ON is False


@pytest.mark.parametrize("v,want", [(None, False), ("", False), ("0", False), ("1", True), ("TRUE", True),
                                    (" on ", True), ("yes", True), ("off", False), ("garbage", False)])
def test_flag_reader_while_anchor_off(monkeypatch, v, want):
    monkeypatch.setattr(T, "_TRANSITIVE_SPIKING_REACH_DEFAULT_ON", False)
    if v is not None:
        monkeypatch.setenv("BRAIN_TRANSITIVE_SPIKING_REACH", v)
    assert T.transitive_spiking_reach_enabled() is want


def test_lesion_and_op_readers(monkeypatch):
    assert T.transitive_reach_lesion_on() is False and T.transitive_reach_op() == 0
    monkeypatch.setenv("BRAIN_TRANSITIVE_REACH_LESION", "1")
    monkeypatch.setenv("BRAIN_TRANSITIVE_REACH_OP", "3")
    assert T.transitive_reach_lesion_on() is True and T.transitive_reach_op() == 3
    monkeypatch.setenv("BRAIN_TRANSITIVE_REACH_OP", "x")
    assert T.transitive_reach_op() == 0


# ── (1) flag off: the existing path, untouched ──────────────────────────────────────────────────────────────────
def test_flag_off_uses_host_chase_and_never_the_organ(monkeypatch):
    monkeypatch.setenv("BRAIN_TRANSITIVE_CHAT", "1")
    monkeypatch.setattr(T, "_spiking_reach_answer", lambda *a, **k: pytest.fail("organ route ran with the flag off"))
    _install_fake_chase(monkeypatch, ["cat", "ball", "worm", "river"])
    ok, svo = T.resolve_transitive_query(_FakeChat(_chain_facts()), "does cat precede river?", seed=7)
    assert ok is True and isinstance(svo, ChainedSVO) and list(svo) == ["cat", "precede", "river"]
    assert svo.derived_from == [["cat", "precede", "ball"], ["ball", "precede", "worm"], ["worm", "precede", "river"]]


def test_chat_flag_off_still_returns_none(monkeypatch):
    monkeypatch.setenv("BRAIN_TRANSITIVE_SPIKING_REACH", "1")      # the sub-flag alone never dispatches
    assert T.resolve_transitive_query(_FakeChat(_chain_facts()), "does cat precede river?") is None


def test_direct_fact_short_circuit_runs_before_the_organ(monkeypatch):
    monkeypatch.setenv("BRAIN_TRANSITIVE_CHAT", "1")
    monkeypatch.setenv("BRAIN_TRANSITIVE_SPIKING_REACH", "1")
    monkeypatch.setattr(T, "_spiking_reach_answer", lambda *a, **k: pytest.fail("organ ran on a direct fact"))
    ok, svo = T.resolve_transitive_query(_FakeChat(_chain_facts()), "does cat precede ball?")
    assert ok is True and svo == ["cat", "precede", "ball"] and not isinstance(svo, ChainedSVO)


# ── the wire-in with the flag on (stub organ, no brain build) ───────────────────────────────────────────────────
def _stub_organ(monkeypatch, result, calls):
    import research.runners.transitive_spiking_reach_organ as O

    def fake(composer, agent, relation, target, *, seed, lesion=False):
        calls.append({"agent": agent, "relation": relation, "target": target, "seed": seed, "lesion": lesion,
                      "op": dict(O.PARAMS)})
        if isinstance(result, Exception):
            raise result
        return dict(result, lesioned=lesion)

    monkeypatch.setattr(O, "reach_from_composer", fake)
    return O


def _on(monkeypatch):
    monkeypatch.setenv("BRAIN_TRANSITIVE_CHAT", "1")
    monkeypatch.setenv("BRAIN_TRANSITIVE_SPIKING_REACH", "1")


def test_flag_on_reached_returns_chained_svo_with_real_pairs(monkeypatch):
    _on(monkeypatch)
    calls = []
    _stub_organ(monkeypatch, {"reached": True, "detect_spikes": 40, "path": ["cat", "ball", "worm", "river"],
                              "n_concepts": 4}, calls)
    ok, svo = T.resolve_transitive_query(_FakeChat(_chain_facts()), "does cat precede river?", seed=5)
    assert ok is True and isinstance(svo, ChainedSVO) and list(svo) == ["cat", "precede", "river"]
    assert svo.derived_from == [["cat", "precede", "ball"], ["ball", "precede", "worm"], ["worm", "precede", "river"]]
    assert calls[0]["seed"] == 5 and calls[0]["lesion"] is False and T.last_reach_meta["path_has_target"] is True


def test_flag_on_not_reached_abstains(monkeypatch):
    _on(monkeypatch)
    _stub_organ(monkeypatch, {"reached": False, "detect_spikes": 0, "path": ["river"], "n_concepts": 4}, [])
    assert T.resolve_transitive_query(_FakeChat(_chain_facts()), "does river precede cat?") == (True, None)


def test_flag_on_path_without_target_is_recorded_and_falls_back(monkeypatch):
    _on(monkeypatch)
    _stub_organ(monkeypatch, {"reached": True, "detect_spikes": 33, "path": ["cat", "ball"], "n_concepts": 4}, [])
    ok, svo = T.resolve_transitive_query(_FakeChat(_chain_facts()), "does cat precede river?")
    assert ok is True and svo.derived_from == [["cat", "precede", "river"]]
    assert T.last_reach_meta["path_has_target"] is False and T.last_reach_meta["detect_spikes"] == 33


def test_flag_on_lesion_and_op_are_routed(monkeypatch):
    _on(monkeypatch)
    monkeypatch.setenv("BRAIN_TRANSITIVE_REACH_LESION", "1")
    monkeypatch.setenv("BRAIN_TRANSITIVE_REACH_OP", "2")
    calls = []
    O = _stub_organ(monkeypatch, {"reached": False, "detect_spikes": 0, "path": ["cat"], "n_concepts": 4}, calls)
    try:
        assert T.resolve_transitive_query(_FakeChat(_chain_facts()), "does cat precede river?") == (True, None)
        assert calls[0]["lesion"] is True and calls[0]["op"] == O.GRID[2]
    finally:
        O.set_operating_point(0)


def test_flag_on_never_raises(monkeypatch):
    _on(monkeypatch)
    _stub_organ(monkeypatch, RuntimeError("boom"), [])
    assert T.resolve_transitive_query(_FakeChat(_chain_facts()), "does cat precede river?") == (True, None)


def test_flag_on_self_query_abstains(monkeypatch):
    _on(monkeypatch)
    _stub_organ(monkeypatch, {"reached": True, "detect_spikes": 40, "path": ["cat"], "n_concepts": 4}, [])
    assert T.resolve_transitive_query(_FakeChat(_chain_facts()), "does cat precede cat?") == (True, None)


# ── (3) the LBF rows ──────────────────────────────────────────────────────────────────────────────────────────────
def test_lbf_row_selftest_shape():
    from research.runners.lbf_rows import transitive_spiking_reach as row
    res = row.selftest()
    assert res["ok"], res["checks"]


def test_lbf_row_turns_are_a_literal_list_the_battery_reads():
    import ast
    path = os.path.join(REPO, "research", "runners", "lbf_rows", "transitive_spiking_reach.py")
    tree = ast.parse(open(path, encoding="utf-8").read())
    lits = [ast.literal_eval(n.value) for n in tree.body
            if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "EXTRA_TURNS" for t in n.targets)]
    assert len(lits) == 1 and len(lits[0]) == 8 and all(len(t) == 6 for t in lits[0])
    from research.runners.onebrain_regression_battery import _TURN_BY_LABEL
    assert all(t[0] in _TURN_BY_LABEL for t in lits[0])


def test_lbf_rows_register_only_when_both_flags_set(monkeypatch):
    from research.runners import lbf_rows
    from research.runners.lbf_rows import transitive_spiking_reach as row
    lesions, probes = {}, []
    rep = lbf_rows.merge_lbf_rows(lesions, probes, _mods_override=[row])
    assert not lesions and not probes and len(rep["parked"]) >= 2
    monkeypatch.setenv("BRAIN_TRANSITIVE_CHAT", "1")
    monkeypatch.setenv("BRAIN_TRANSITIVE_SPIKING_REACH", "1")
    lesions, probes = {}, []
    rep = lbf_rows.merge_lbf_rows(lesions, probes, _mods_override=[row])
    assert set(lesions) == {"transitive-spiking-reach", "transitive-spiking-reversed-ctl"}
    assert {p[0] for p in probes} == set(lesions) and not rep["collisions"]


# ── (4) the organ battery, one dev seed, op 0 ─────────────────────────────────────────────────────────────────────
def test_organ_battery_dev_seed_op0():
    import research.runners.transitive_spiking_reach_organ as O
    assert O.PARAMS == O.GRID[0]
    res = O.organ_battery(1, 0)
    assert set(res) >= {"seed", "op_index", "params", "cases", "all_pass", "backend"} and res["backend"] == "numpy"
    assert set(res["cases"]) >= {"derived2", "derived3", "reversed", "unrelated", "branch", "lesion_derived3", "path3"}
    for c in res["cases"].values():
        assert "pass" in c and "detect" in c
    assert res["all_pass"] is True, {k: (v["detect"], v["pass"]) for k, v in res["cases"].items()}
    assert res["cases"]["path3"]["path"] == ["cat", "ball", "worm", "river"]
    assert res["cases"]["lesion_derived3"]["detect"] < O.DETECT_MIN <= res["cases"]["derived3"]["detect"]
    json.dumps(res, default=str)


def test_set_operating_point_copies_grid_into_module():
    import research.runners.transitive_spiking_reach_organ as O
    try:
        assert 6 <= len(O.GRID) <= 12
        O.set_operating_point(1)
        assert O.PARAMS == O.GRID[1] and O.GOAL_PA == O.GRID[1]["GOAL_PA"] and O.W_HOP == O.GRID[1]["W_HOP"]
    finally:
        O.set_operating_point(0)
    assert O.PARAMS == O.GRID[0]


def test_reach_from_composer_reads_kb():
    import research.runners.transitive_spiking_reach_organ as O
    comp = _FakeComposer({("cat", "precede"): "ball", ("ball", "precede"): "worm", ("dog", "chase"): "cat"})
    assert O.composer_edges(comp, "precede") == [("cat", "ball"), ("ball", "worm")]
    r = O.reach_from_composer(comp, "cat", "precede", "worm", seed=1)
    assert set(r) == {"reached", "detect_spikes", "path", "n_concepts", "lesioned"}
    assert r["reached"] is True and r["path"] == ["cat", "ball", "worm"] and r["n_concepts"] == 3
    l = O.reach_from_composer(comp, "cat", "precede", "worm", seed=1, lesion=True)
    assert l["reached"] is False and l["lesioned"] is True
    assert O.reach_from_composer(comp, "worm", "precede", "cat", seed=1)["reached"] is False
    assert O.reach_from_composer(comp, "cat", "eat", "worm", seed=1)["reached"] is False     # no such relation


# ── (5) gate-seed refusal ─────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("argv", [
    ["--seed", "42"], ["--seed", "100", "--out", "x.json"], ["--sweep", "--seeds", "1", "102"],
    ["--seed", "43", "--allow-gate-seed"], ["--seed", "44", "--prereg", "p.md"],
])
def test_cli_refuses_gate_seeds(argv):
    import research.runners.transitive_spiking_reach_organ as O
    with pytest.raises(SystemExit) as e:
        O.main(argv)
    assert "gate seed" in str(e.value)


def test_gate_seed_guard_passes_with_flag_and_prereg():
    import research.runners.transitive_spiking_reach_organ as O
    O._refuse_gate_seeds([42, 1], True, "research/findings/x-PREREGISTRATION.md")      # no raise
    O._refuse_gate_seeds([1, 2, 3], False, None)                                        # dev seeds never refused
