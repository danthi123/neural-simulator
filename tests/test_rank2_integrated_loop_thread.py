"""CI GUARD (scaffold-retirement backlog rank-2): `integrated_loop` (the OneBrainComposer spiking K-way cue-match
SEQUENCER, already GO 4/4 at production V=320 -- 2026-06-21-shortcut3-fold-integrated-loop-BUILD.md) is threaded
through the production webapp construction chain: `webapp.server._build_chat_brain` -> `brain_chat_tui.
_build_tiny_demo` / `developed_brain_io.load_developed_brain` -> `MultiTurnAgent` -> `BrainConversationalAgent` ->
`OneBrainComposer`. Before this rank, `MultiTurnAgent.__init__` had NO `integrated_loop` parameter at all and
`webapp/server.py` had ZERO references to `integrated_loop` -- there was no way to opt a live chat brain into the
sequencer. Every default stays OFF (byte-identical); this guard pins that AND that the parameter genuinely reaches
the composer, cheaply (small D/K -- the production-scale V=320/K=32 6-seed confirmation is the de-risk runner,
`research/runners/_rank2_integrated_loop_webapp_thread_derisk.py`, not this CI suite).
"""
import inspect
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.multi_turn_agent import MultiTurnAgent
from research.runners.brain_chat_tui import _build_tiny_demo
from research.runners.developed_brain_io import load_developed_brain


# ----------------------------------------------------------------------------------------------------------------
# Task 1: every new/extended signature defaults `integrated_loop` to the byte-identical OFF value. Pure
# introspection -- no composer build, so this stays instant even though the mechanism itself is not.
# ----------------------------------------------------------------------------------------------------------------
def test_multi_turn_agent_integrated_loop_defaults_false():
    sig = inspect.signature(MultiTurnAgent.__init__)
    assert "integrated_loop" in sig.parameters, "MultiTurnAgent must expose integrated_loop (the rank-2 thread)"
    assert sig.parameters["integrated_loop"].default is False


def test_build_tiny_demo_integrated_loop_defaults_false():
    sig = inspect.signature(_build_tiny_demo)
    assert "integrated_loop" in sig.parameters
    assert sig.parameters["integrated_loop"].default is False


def test_load_developed_brain_integrated_loop_defaults_false():
    sig = inspect.signature(load_developed_brain)
    assert "integrated_loop" in sig.parameters
    assert sig.parameters["integrated_loop"].default is False


def test_webapp_integrated_loop_env_default_off(monkeypatch):
    """`webapp.server._integrated_loop_enabled()`: unset -> False; explicit 1/true/on/yes -> True; explicit
    0/false/off/no -> False. Pure function -- no ChatBrain build, so this is cheap despite living in webapp.server
    (a heavier import than the runner modules above; test_webapp_server.py already pays this cost routinely)."""
    from webapp import server as webapp_server
    monkeypatch.delenv("BRAIN_INTEGRATED_LOOP", raising=False)
    assert webapp_server._integrated_loop_enabled() is False
    for on in ("1", "true", "True", "on", "yes"):
        monkeypatch.setenv("BRAIN_INTEGRATED_LOOP", on)
        assert webapp_server._integrated_loop_enabled() is True, f"{on!r} must enable"
    for off in ("0", "false", "False", "off", "no"):
        monkeypatch.setenv("BRAIN_INTEGRATED_LOOP", off)
        assert webapp_server._integrated_loop_enabled() is False, f"{off!r} must stay OFF"


# ----------------------------------------------------------------------------------------------------------------
# Task 2: MultiTurnAgent.integrated_loop genuinely reaches OneBrainComposer's constructor -- a pure WIRING check.
# The REAL substrate capability (answer-identity + moat at production V=320, 6 seeds) is already GO
# (2026-06-21-shortcut3-fold-integrated-loop-BUILD.md) and is re-confirmed through this SAME new plumbing by
# research/runners/_rank2_integrated_loop_webapp_thread_derisk.py Part B (GPU) -- repeating that here would just
# make every commit pay for a ~90-180s onebrain build (BrainConversationalAgent's onebrain branch always sizes
# OneBrainComposer's sequencer fabric at k_max=32 regardless of the caller's vocab, so there is no "small-K" cheap
# lane through this branch the way test_onebrain_integrated_loop_fold.py gets by calling OneBrainComposer
# directly). So: monkeypatch OneBrainComposer (+ LearnedAssocGraph, which the onebrain branch also always builds)
# with instant recorders, and assert the KWARG each one receives -- this is what a "does parameter X reach
# constructor Y" claim needs, no spiking required.
# ----------------------------------------------------------------------------------------------------------------
def _spy_construct(monkeypatch, integrated_loop):
    captured = {}

    class _FakeOneBrainComposer:
        def __init__(self, *a, **kw):
            captured["integrated_loop"] = kw.get("integrated_loop")

        def hear(self, *a, **kw):    # hasattr(composer, "hear") == True -> the agent skips its own separate parser
            pass

        def store(self, *a, **kw):
            pass

    class _FakeLearnedAssocGraph:
        def __init__(self, *a, **kw):
            pass

    import research.runners.one_brain_composer as obc_mod
    import research.runners.onebrain_merge_production as merge_mod
    import research.runners.learned_assoc_graph as lag_mod
    monkeypatch.setattr(obc_mod, "OneBrainComposer", _FakeOneBrainComposer)
    monkeypatch.setattr(lag_mod, "LearnedAssocGraph", _FakeLearnedAssocGraph)
    # BRAIN_COMPOSER_MERGE's pool-1 composer (default ON, an unrelated feature) would otherwise route construction
    # through make_pool1_onebrain_composer instead of the plain OneBrainComposer this spy targets; force it off so
    # this WIRING check is deterministic regardless of that separate flag's default.
    monkeypatch.setattr(merge_mod, "composer_merge_enabled", lambda: False)

    MultiTurnAgent(referent_concepts=["dog", "cat"], concepts={"dog": None, "cat": None, "chase": None}, seed=42,
                   composer_kind="onebrain", integrated_loop=integrated_loop, enable_neural_render=False,
                   defer_planner=True)
    return captured


def test_multi_turn_agent_integrated_loop_thread_reaches_composer(monkeypatch):
    for want in (False, True):
        captured = _spy_construct(monkeypatch, want)
        assert captured["integrated_loop"] is want, f"integrated_loop={want} did not reach OneBrainComposer"
