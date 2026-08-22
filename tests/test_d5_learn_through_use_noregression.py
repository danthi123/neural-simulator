"""Fast, no-GPU no-regression GUARD for the D5 learn-through-use default-on flip (board #71).

The heavy end-to-end proof (the graded read RISES through the real substrate consolidation loop) is the GPU runner
`research/runners/_d5_learn_through_use_noregression.py`. THIS test locks the two invariants that must hold at the
CONVERSATION-VISIBLE surfacing layer and can be checked deterministically with NO brain — so they run on every commit:

  A. OFF (`BRAIN_D5_CONSOLIDATE` unset) is byte-identical to HEAD: `recall_disclosure` emits NO recall-strength clause
     (the exact HEAD reply text), and `consolidate_used_memory` short-circuits to None (no store mutation).
  B. ON (`BRAIN_D5_CONSOLIDATE=1`) leaves the moat / abstain UNCHANGED: the honest-abstain line for a not-in-memory
     record is byte-identical off vs on; ON only APPENDS the graded strength to a record the binary gate already
     admitted (in_memory=True), the completion text (the moat-carrying part) is unchanged, and the surfaced number is
     the record's real graded `depth_hold` (never invented). A not-in-memory record NEVER surfaces a strength, on or off.

`recall_disclosure` is a PURE function of (record, flag): the flag gates SURFACING + consolidation only, never the
binary `in_memory` gate (which `recall` computes without ever reading the flag). So these invariants fully characterise
the disclosure-layer behaviour of the flip without a substrate.
"""
import os

import pytest

from research.runners.d5_episodic_production_organ import (
    recall_disclosure, SURFACED_GRADED_READ, GRADED_READS)
from webapp import continuous_engine as CE

_FLAG = "BRAIN_D5_CONSOLIDATE"


def _record(topic, in_memory, cue=0.43, depth_hold=15.0):
    """A synthetic recall record with the SAME shape EpisodicDapMemory.recall emits (binary gate + graded reads)."""
    g = {r: (depth_hold if r != "soft" else 0.6) for r in GRADED_READS}
    z = {r: 0.0 for r in GRADED_READS}
    return {"topic": topic, "slot": 0, "formed": in_memory, "in_memory": bool(in_memory),
            "apical_cue": float(cue) if in_memory else 0.0, "apical_perm": 0.0, "apical_nocue": 0.0,
            "lesioned": False, "reason": "spiking-dap-completion",
            "graded_cue": (g if in_memory else z), "graded_perm": z, "graded_nocue": z}


@pytest.fixture
def flag_off(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    assert not CE.d5_consolidate_enabled()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")
    assert CE.d5_consolidate_enabled()


# ── CLAIM A — OFF byte-identical to HEAD ──────────────────────────────────────────────────────────────────────────
def test_A_off_formed_disclosure_is_exact_head_text(flag_off):
    rec = _record("dog", in_memory=True, cue=0.43)
    disc = recall_disclosure(rec, content=None)
    expected = ("Earlier you brought up dog — my hippocampal readout completes its assembly for it "
                "(dendritic dAP completion 0.43).")
    assert disc == expected
    assert "recall strength" not in disc


def test_A_off_abstain_disclosure_is_exact_head_text(flag_off):
    rec = _record("cat", in_memory=False)
    disc = recall_disclosure(rec, content=None)
    expected = ("I don't recall us discussing cat — no assembly completes for that cue "
                "(a genuine spiking completion failure, so I won't make something up).")
    assert disc == expected


def test_A_off_consolidate_short_circuits_to_none(flag_off):
    # OFF returns None BEFORE touching any store (the byte-identity anchor). True even with a dummy organ present.
    assert CE.consolidate_used_memory(("t", 0), object()) is None
    assert CE.consolidate_used_memory(("t", 0), None) is None


# ── CLAIM B — ON leaves the moat / abstain UNCHANGED ──────────────────────────────────────────────────────────────
def test_B_on_appends_strength_completion_text_preserved(monkeypatch):
    rec = _record("dog", in_memory=True, cue=0.43, depth_hold=15.0)
    monkeypatch.delenv(_FLAG, raising=False)
    disc_off = recall_disclosure(rec, content=None)
    monkeypatch.setenv(_FLAG, "1")
    disc_on = recall_disclosure(rec, content=None)
    assert "recall strength" not in disc_off
    assert "recall strength 15.0 mV" in disc_on            # the surfaced number IS the record's depth_hold
    # the moat-carrying completion text is unchanged — ON only APPENDS inside the same lead
    frag = "dendritic dAP completion 0.43"
    assert frag in disc_off and frag in disc_on


def test_B_on_abstain_line_identical_off_vs_on(monkeypatch):
    rec = _record("cat", in_memory=False)
    monkeypatch.delenv(_FLAG, raising=False)
    disc_off = recall_disclosure(rec, content=None)
    monkeypatch.setenv(_FLAG, "1")
    disc_on = recall_disclosure(rec, content=None)
    assert disc_on == disc_off                              # the honest-abstain reply never changes with the flag


def test_B_on_surfaced_strength_equals_record_read(flag_on):
    rec = _record("dog", in_memory=True, cue=0.50, depth_hold=22.4)
    surfaced = float(rec["graded_cue"][SURFACED_GRADED_READ])
    disc = recall_disclosure(rec, content=None)
    assert f"recall strength {surfaced:.1f} mV" in disc


def test_B_not_in_memory_never_surfaces_strength_even_on(flag_on):
    rec = _record("cat", in_memory=False)
    disc = recall_disclosure(rec, content=None)
    assert "recall strength" not in disc                   # a not-completing memory surfaces NOTHING, on or off
    assert "I don't recall us discussing cat" in disc
