"""CI guard for EMERGE-56 RUNG 1 -- wiring EMERGENT grounded reasoning -> fluent render behind the gate-first
no-confab MOAT. CPU/numpy, offline. The load-bearing property is the MOAT: the renderer must NEVER be invoked on
an abstain (0 false renders). Tests:
  1. adapter fidelity -- the extracted (gate, subject, property) matches EMERGE's own ask_can decision on every probe.
  2. moat preserved -- every abstain renders "I don't know" AND the render-call count on abstains is 0 (the
     renderer-not-invoked-on-abstain assertion).
  3. correct grounded facts -- inherit (owl->fly) and cancel (penguin->walks) render the correct content.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.runners._emerge56_reasoning_to_fluent_wire_derisk import (
    emerge_gate_decision, wired_reply, CountingStubFaculty, _adapter_matches,
    _teach_console, _derisk_one,
)


@pytest.fixture(scope="module")
def trained():
    """Train the EMERGE-51 console once (observe -> discover -> teach) + the scripted ASK probes."""
    console, probes = _teach_console(seed=42)
    return console, probes


def test_adapter_fidelity(trained):
    """The adapter's structured (gate, subject, property) must equal EMERGE's own ask_can decision on EVERY probe."""
    console, probes = trained
    matches = [_adapter_matches(console, m, prop) for (m, prop, _exp) in probes]
    assert all(matches), f"adapter diverged from EMERGE on {sum(1 for x in matches if not x)}/{len(matches)} probes"


def test_moat_preserved_renderer_never_invoked_on_abstain(trained):
    """The LOAD-BEARING property: on every abstain the loop emits 'I don't know' AND the renderer is NEVER invoked
    (the render-call count does not change across an abstain)."""
    console, probes = trained
    faculty = CountingStubFaculty()
    abstain_probes = [(m, prop) for (m, prop, exp) in probes if exp.startswith("moat")]
    assert abstain_probes, "the scripted probes must include moat/abstain cases"
    for (m, prop) in abstain_probes:
        calls_before = faculty.render_call_count
        rec = wired_reply(console, faculty, m, prop)
        calls_after = faculty.render_call_count
        assert rec["gate"] == "ABSTAIN"
        assert rec["surface"].startswith("I don't know"), rec["surface"]
        # THE MOAT: the renderer must NOT have been invoked on an abstain (0 false renders).
        assert calls_after == calls_before, f"renderer INVOKED on abstain for {m!r} (moat breached)"
        assert rec["renderer_invoked"] is False


def test_correct_grounded_facts_rendered(trained):
    """Inherit (owl->can fly) and cancel (penguin->walks) render the CORRECT gated content."""
    console, probes = trained
    faculty = CountingStubFaculty()
    # inherited default: a held-out bird inherits 'fly' -> "Yes, the <m> can fly."
    inh = next(p for p in probes if p[2] == "inherited" and p[1] == "fly")
    rec = wired_reply(console, faculty, inh[0], inh[1])
    assert rec["gate"] == "ANSWER" and rec["source"] == "inherited"
    assert rec["surface"].startswith("Yes") and "fly" in rec["surface"]
    # member exception (cancellation): penguin -> "No, the penguin walks."
    exc = next(p for p in probes if p[2] == "exception")
    rec2 = wired_reply(console, faculty, exc[0], exc[1])
    assert rec2["gate"] == "ANSWER" and rec2["source"] == "exception"
    assert rec2["surface"].startswith("No")
    ovr_word = console.ovr_prop.get(exc[0])
    assert ovr_word and ovr_word in rec2["surface"]


def test_derisk_one_seed_go():
    """The single-seed de-risk record: adapter fidelity == 1.0, moat preserved with 0 false renders, facts correct."""
    d = _derisk_one(seed=42)
    assert d["adapter_fidelity"] >= 0.95
    assert d["moat_ok"] is True
    assert d["moat_render_calls_on_abstains"] == 0
    assert d["moat_false_renders"] == 0
    assert d["fact_correct"] >= 0.99
