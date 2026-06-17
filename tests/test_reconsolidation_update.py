"""Reconsolidation (prediction-error-gated in-place fact update) on the PRODUCTION RFPhasorComposer.

Asserts the de-risked capability (`research/findings/2026-06-17-reconsolidation-update-derisk-GO.md`, 6/6 GO) on
the production class itself: a corrective utterance UPDATES the cued fact in place (no contradictory duplicate),
the prediction-error boundary condition is real (a re-statement re-stabilizes unchanged, NOT last-write-wins), and
the no-confab moat holds (correcting a never-stored subject abstains). CPU/numpy; reuse-by-import; no sim/ edit.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.rf_phasor_composer import RFPhasorComposer

VOCAB = ["dog", "cat", "bird", "fish", "elephant",
         "go", "run", "fly", "swim", "north", "south", "east", "west"]
BASE = [("dog", "go", "north"), ("cat", "run", "south"), ("bird", "fly", "east"), ("fish", "swim", "west")]


def _build(seed):
    c = RFPhasorComposer(seed=seed, D=128, vocab=VOCAB)
    for a, ac, p in BASE:
        c.store(a, ac, p)
    return c


@pytest.mark.parametrize("seed", [42, 43])
def test_baseline_recovers(seed):
    """The substrate must recover the stored facts before any correction (else the test is confounded)."""
    c = _build(seed)
    for a, ac, p in BASE:
        assert c.query_patient(a, ac) == p


@pytest.mark.parametrize("seed", [42, 43])
def test_reconsolidate_updates_in_place(seed):
    """RECONSOLIDATE: 'dog go north' -> corrected 'dog go south' updates IN PLACE (query=south, exactly ONE fact),
    and an untouched fact is preserved."""
    c = _build(seed)
    res = c.update_on_mismatch("dog", "go", "south")
    assert res["action"] == "rewrite" and res["wrote"] is True
    assert c.query_patient("dog", "go") == "south"        # corrected
    assert c.count_facts("dog", "go") == 1                # no contradictory duplicate
    assert c.query_patient("cat", "run") == "south"       # collateral preserved


@pytest.mark.parametrize("seed", [42, 43])
def test_c1_prediction_error_boundary(seed):
    """C1 (the decisive boundary condition): re-stating the SAME fact (PE~0) must NOT change the memory -- proving
    this is prediction-error-gated reconsolidation, not last-write-wins."""
    c = _build(seed)
    res = c.update_on_mismatch("dog", "go", "north")      # same patient -> PE below the gate
    assert res["action"] == "restabilize" and res["wrote"] is False
    assert c.query_patient("dog", "go") == "north"        # unchanged
    assert c.count_facts("dog", "go") == 1


@pytest.mark.parametrize("seed", [42, 43])
def test_c2_moat_never_stored_abstains(seed):
    """C2 (no-confab moat): correcting a NEVER-stored subject must abstain -- update a reactivated trace, never
    fabricate a missing one."""
    c = _build(seed)
    res = c.update_on_mismatch("elephant", "go", "west")
    assert res["action"] == "abstain" and res["wrote"] is False
    assert c.query_patient("elephant", "go") is None
    assert c.count_facts("elephant", "go") == 0


@pytest.mark.parametrize("seed", [42, 43])
def test_naive_append_contrast(seed):
    """The current production path (store the correction) yields TWO contradictory facts answered stale-first --
    the behavior reconsolidation replaces."""
    c = _build(seed)
    c.store("dog", "go", "south")                          # naive append (no update)
    assert c.count_facts("dog", "go") == 2                 # duplicate
    assert c.query_patient("dog", "go") == "north"         # stale first-match


def test_pe_gate_separation():
    """The calibrated labilization gate sits cleanly between same-fact and different-fact prediction errors."""
    c = _build(42)
    idx, fact, comp = c._find_cued_fact("dog", "go")
    pe_same = c._patient_prediction_error(comp, "north")
    pe_diff = c._patient_prediction_error(comp, "south")
    gate = c._calibrate_pe_labile()
    assert pe_same < gate < pe_diff


# --- MultiTurnAgentV2 correction-turn hook (the conversational entry point) ---
from research.runners.multi_turn_agent_v2 import MultiTurnAgentV2

REFERENTS = ["dog", "cat", "bird", "fish", "elephant"]


def _agent(seed=42):
    return MultiTurnAgentV2(referent_concepts=REFERENTS, concepts={w: None for w in VOCAB}, seed=seed)


def test_agent_correct_updates_in_place():
    """A correction turn updates the cued fact in place; hear() stays append-only (the byte-preserved path)."""
    a = _agent()
    a.hear("dog go north")
    res = a.correct("actually dog go south")
    assert res["wrote"] is True and res["action"] == "rewrite"
    assert a.what_does("dog", "go") == "south"
    assert a.agent.composer.count_facts("dog", "go") == 1


def test_agent_correct_pronoun_resolves():
    """'actually it go south' resolves the pronoun agent from the discourse buffer and updates that fact."""
    a = _agent()
    a.hear("dog go north")                       # foregrounds 'dog' (north is not a referent)
    res = a.correct("actually it go south")      # 'it' -> dog
    assert res["wrote"] is True
    assert a.what_does("dog", "go") == "south"


def test_agent_correct_restatement_no_change():
    """C1 boundary at the agent level: re-stating the same fact re-stabilizes unchanged (not last-write-wins)."""
    a = _agent()
    a.hear("dog go north")
    res = a.correct("actually dog go north")
    assert res["wrote"] is False and res["action"] == "restabilize"
    assert a.what_does("dog", "go") == "north"


def test_agent_correct_never_stored_abstains():
    """C2 moat at the agent level: correcting a never-stored subject abstains (no fabrication)."""
    a = _agent()
    a.hear("dog go north")
    res = a.correct("actually elephant go west")
    assert res["action"] == "abstain" and res["wrote"] is False
    assert a.agent.composer.count_facts("elephant", "go") == 0
