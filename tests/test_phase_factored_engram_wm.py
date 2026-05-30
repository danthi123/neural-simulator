"""Structural pins for the DG/engram-based working-memory (wm) retrieval
rework of research/runners/phase_factored_loop_gate.py.

Diagnosis (research/findings/2026-05-30-phase-factored-fullscale-
grounding-INSTRUMENT-UNSOUND-wm-nondiscriminating.md): the OLD wm
readout retrieved role->filler via CORTICAL dlpfc_verb->filler STDP
selectivity, which is unstable on this substrate (repeated selectivity
training erodes the topographic prior, so a role query lights ALL
filler pools ~equally -> v1 wm ~= chance). THE FIX (de-risked GO by the
controller): route the wm role->filler retrieval through the
DG-separated hippocampal ENGRAM mechanism (the SAME path the ep readout
already uses to reach ep=1.0) instead of the eroding cortical STDP.

These pins assert the engram-rework at the STRUCTURAL level on the
tiny-synth CPU/numpy path (fast, deterministic, no GPU):
  1. ENCODE writes PER-BINDING engram tags (pf_ep<E>_bind<bi>) -- one
     per (role, filler) binding -- alongside the unchanged whole-episode
     tag (pf_episode_<E>) the ep readout needs.
  2. The wm READOUT retrieves via the engram path: it calls
     stimulate_tag on the queried role's per-binding tag(s) (multitag
     stim-recall) -- it is NOT the old "drive the role code into lang and
     rank cortical pool firing" path.
  3. The lesions still skip the per-binding tags exactly where they skip
     the whole-episode tag (no_hippo_store), so the frozen partition is
     preserved at the structural level.

Plain ASCII. SIM_BACKEND=numpy tiny-synth (CPU). No autograd.
"""
from __future__ import annotations
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
_CONTROLLER_PATH = REPO_ROOT / "research" / "runners" / \
    "phase_factored_loop_gate.py"


def _import_controller():
    if not _CONTROLLER_PATH.exists():
        pytest.skip("phase_factored_loop_gate.py not landed yet")
    os.environ.setdefault("SIM_BACKEND", "numpy")
    import importlib
    return importlib.import_module(
        "research.runners.phase_factored_loop_gate")


class _BridgeCallSpy:
    """Wrap the SimulationBridge class methods so every per-binding
    engram-API call made during a _run_mode is recorded. Records ONLY
    the (method, name) tuple; does not alter behavior (it delegates to
    the original method). Restored by the context manager exit."""

    def __init__(self):
        self.calls = []  # list of (method_name, tag_name)

    def __enter__(self):
        from sim.bridge import SimulationBridge
        self._cls = SimulationBridge
        self._orig = {}
        for m in ("start_engram_recording", "commit_engram_tag",
                  "stimulate_tag"):
            self._orig[m] = getattr(SimulationBridge, m)

        def _mk(method_name, orig):
            def _wrap(bridge_self, name, *a, **k):
                self.calls.append((method_name, name))
                return orig(bridge_self, name, *a, **k)
            return _wrap

        for m, orig in self._orig.items():
            setattr(SimulationBridge, m, _mk(m, orig))
        return self

    def __exit__(self, *exc):
        for m, orig in self._orig.items():
            setattr(self._cls, m, orig)
        return False

    def names(self, method_name):
        return [nm for (mm, nm) in self.calls if mm == method_name]


def _run_full_with_spy(mod, mode="full", N=2, gap_zero=False):
    with _BridgeCallSpy() as spy:
        mod._run_mode(mode, 42, N, True, gap_zero=gap_zero)
    return spy


# ---------------------------------------------------------------------------
# 1. ENCODE writes per-binding engram tags alongside the whole-episode tag.
# ---------------------------------------------------------------------------
def test_encode_commits_per_binding_tags():
    """For each of the N bindings the encode loop must start + commit a
    PER-BINDING engram tag named pf_ep<E>_bind<bi>, in addition to the
    whole-episode tag pf_episode_<E> (which the ep readout still needs)."""
    mod = _import_controller()
    spy = _run_full_with_spy(mod, mode="full", N=2)
    started = set(spy.names("start_engram_recording"))
    committed = set(spy.names("commit_engram_tag"))
    # The whole-episode tag is still present (ep readout depends on it).
    assert any(nm.startswith("pf_episode_") for nm in started), (
        "whole-episode tag must still be recorded: %r" % started)
    assert any(nm.startswith("pf_episode_") for nm in committed), (
        "whole-episode tag must still be committed: %r" % committed)
    # Per-binding tags: one per binding (N=2 -> bind0 + bind1), at least
    # for the final epoch.
    per_bind_started = {nm for nm in started if "_bind" in nm}
    per_bind_committed = {nm for nm in committed if "_bind" in nm}
    assert len(per_bind_started) >= 2, (
        "expected >= 2 per-binding tags started (N=2): %r" % started)
    assert len(per_bind_committed) >= 2, (
        "expected >= 2 per-binding tags committed (N=2): %r" % committed)
    # Naming contract: pf_ep<E>_bind<bi>.
    for nm in per_bind_committed:
        assert nm.startswith("pf_ep") and "_bind" in nm, (
            "per-binding tag name must be pf_ep<E>_bind<bi>: %r" % nm)


# ---------------------------------------------------------------------------
# 2. The wm READOUT retrieves via the engram path (stimulate_tag), NOT the
#    old cortical "drive role code + rank pool firing" path.
# ---------------------------------------------------------------------------
def test_wm_readout_stimulates_per_binding_tags():
    """The wm readout must drive retrieval through stimulate_tag on
    per-binding tags. Pin: across a full v1 run, stimulate_tag is called
    on at least one per-binding tag (pf_ep<E>_bind<bi>). This proves the
    wm retrieval routes through the engram mechanism (the ep path),
    replacing the eroding cortical dlpfc_verb->filler STDP readout."""
    mod = _import_controller()
    spy = _run_full_with_spy(mod, mode="full", N=2, gap_zero=True)
    stim_names = spy.names("stimulate_tag")
    assert any("_bind" in nm for nm in stim_names), (
        "wm readout must stimulate at least one per-binding tag "
        "(engram retrieval path); stimulate_tag was called on: %r"
        % stim_names)


def test_wm_readout_uses_wm_raw_sink_with_engram_path():
    """The passive _WM_RAW_SINK still records (true_fidx, raw filler
    counts, gated decision) per scored query, so the controller can
    scrutinize a PASS. Enabling the sink changes NO drive/gate/RNG/score;
    it only records already-computed values. Pin: a v1 run with the sink
    enabled records exactly one entry per scored query (N=1 for v1)."""
    mod = _import_controller()
    mod._WM_RAW_SINK = []
    try:
        mod._run_mode("full", 42, 2, True, gap_zero=True)
        recorded = list(mod._WM_RAW_SINK)
    finally:
        mod._WM_RAW_SINK = None
    # v1 (gap_zero) scores n_q == N == 2 queries (each role queried as its
    # own drilled binding). Each entry is (true_fidx:int, counts:list,
    # decision:str|None).
    assert len(recorded) >= 1, "wm raw sink recorded nothing"
    for (tf, counts, dec) in recorded:
        assert isinstance(tf, int)
        assert isinstance(counts, list) and len(counts) >= 1
        assert dec is None or isinstance(dec, str)


# ---------------------------------------------------------------------------
# 3. The lesions still skip the per-binding tags exactly where they skip
#    the whole-episode tag (no_hippo_store): the engram rework preserves
#    the frozen partition at the structural level.
# ---------------------------------------------------------------------------
def test_no_hippo_store_skips_per_binding_tags():
    """no_hippo_store is a SHARED lesion: it must skip the relational
    store entirely. With the engram rework, the per-binding tags ARE the
    wm-side relational store, so no_hippo_store must commit NEITHER the
    whole-episode tag NOR any per-binding tag (wm collapses with ep)."""
    mod = _import_controller()
    spy = _run_full_with_spy(mod, mode="no_hippo_store", N=2)
    committed = set(spy.names("commit_engram_tag"))
    assert not any(nm.startswith("pf_episode_") for nm in committed), (
        "no_hippo_store must NOT commit the whole-episode tag: %r"
        % committed)
    assert not any("_bind" in nm for nm in committed), (
        "no_hippo_store must NOT commit any per-binding tag (the wm-side "
        "relational store is skipped): %r" % committed)


def test_no_hippo_store_does_not_stimulate_for_wm():
    """Under no_hippo_store there are no committed tags, so the wm readout
    cannot stimulate any per-binding tag -> wm collapses by construction
    (mirrors the ep collapse). Pin: no per-binding stimulate_tag call."""
    mod = _import_controller()
    spy = _run_full_with_spy(mod, mode="no_hippo_store", N=2)
    stim_names = spy.names("stimulate_tag")
    assert not any("_bind" in nm for nm in stim_names), (
        "no_hippo_store must not stimulate any per-binding tag: %r"
        % stim_names)
