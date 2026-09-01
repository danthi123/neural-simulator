"""Confidence-gated finer-period decode-escalation (#66 seed-44 recall hole, 2026-09-01).

ROOT CAUSE it closes: the RF phase readout `((period - spike_step) % period)/period` quantizes the recovered
phase to 1/period (= 0.005 at period=200), coarser than the real inter-word cleanup margin for some facts, so a
fact's stored cue role occasionally argmax-decodes to the WRONG vocab word by a razor-thin margin and
`_scan_first_match` drops a fact it genuinely holds (a false abstain -> what_does None / ask_yes_no unknown). The
fix re-examines a near-tie MATCH candidate at a finer resonate period (a longer-integrated, more faithful neural
readout) before dropping it, and accepts the match iff the finer decode now argmaxes to the cued value.

These are the byte-identical-when-OFF + moat-safe + safe-when-aggressive ASSERTIONS (the constraint: make the
byte-identity claim a test, not a comment). The end-to-end hole recovery on the real 78,857-fact bundle is the
integration proof in `_knowledge_scale_100k_production_verify.py` (oracle-parity gate) + the finding.
"""
import os

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.rf_phasor_composer import RFPhasorComposer

VOCAB = ["dog", "cat", "bird", "fish", "elephant", "horse", "lion", "wolf",
         "go", "run", "fly", "swim", "eat", "see", "chase", "hunt",
         "north", "south", "east", "west", "river", "tree", "mouse", "deer"]
FACTS = [("dog", "go", "north"), ("cat", "run", "south"), ("bird", "fly", "east"),
         ("fish", "swim", "west"), ("elephant", "eat", "tree"), ("horse", "see", "river"),
         ("lion", "chase", "deer"), ("wolf", "hunt", "mouse")]
ABSENT_PATIENT = ("lion", "fly")           # a never-stored (agent, action) cue
UNKNOWN_AGENT = ("zzz_unknown_xq", "go")   # an out-of-vocabulary agent (the moat cue)


def _build(seed, **kw):
    c = RFPhasorComposer(seed=seed, D=128, vocab=VOCAB, **kw)
    for a, ac, p in FACTS:
        c.store(a, ac, p, polarity="AFFIRM")
    return c


def test_escalation_default_is_off():
    c = RFPhasorComposer(seed=42, D=64, vocab=["a", "b"])
    assert c.enable_decode_escalation is False       # default OFF -> the extra branch never runs


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_escalation_on_is_byte_identical_to_off_when_no_hole(seed):
    """With the small clean codebook there is no thin-margin hole, so escalation ON -- even with the margin turned
    up so it fires on EVERY candidate -- must return byte-identical answers to escalation OFF for every query
    (escalation only ever RECOVERS a genuine hole; it never changes a cleanly-decoded answer)."""
    off = _build(seed)
    on = _build(seed, enable_decode_escalation=True, decode_escalate_margin=2.0, decode_escalate_period=2000)
    for a, ac, p in FACTS:
        assert on.query_patient(a, ac) == off.query_patient(a, ac) == p
        assert on.query_agent(ac, p) == off.query_agent(ac, p) == a
        assert on.ask_yes_no(a, ac, p) == off.ask_yes_no(a, ac, p) == "yes"
    # abstain / moat cues agree too (None / unknown), on both paths
    assert on.query_patient(*ABSENT_PATIENT) == off.query_patient(*ABSENT_PATIENT) is None
    assert on.query_patient(*UNKNOWN_AGENT) == off.query_patient(*UNKNOWN_AGENT) is None
    assert on.ask_yes_no("dog", "go", "south") == off.ask_yes_no("dog", "go", "south")   # wrong patient


def test_escalation_moat_preserved_even_when_aggressive():
    """Escalation fired on every candidate (margin=2.0) must NOT manufacture a match for an unknown agent, an
    unknown relation, or an unknown patient -- an out-of-vocabulary cue value is never in `concepts`, so escalation
    is skipped and the abstain path is unchanged; an in-vocab-but-unstored cross cue still abstains because the
    finer readout converges to the ideal representation (a fact that does not encode the cue is never promoted)."""
    on = _build(42, enable_decode_escalation=True, decode_escalate_margin=2.0, decode_escalate_period=2000)
    assert on.query_patient("zzz_unknown_agent_xq", "go") is None          # unknown agent
    assert on.query_patient("dog", "zzz_unknown_relation_xq") is None      # known agent, unknown relation
    assert on.query_patient("lion", "fly") is None                         # in-vocab cross cue, never stored
    assert on.ask_yes_no("zzz_unknown_agent_xq", "go", "north") == "unknown"
    assert on.ask_yes_no("dog", "go", "south") in ("no", "unknown")        # stored dog-go-north, NOT south
    assert on.ask_yes_no("bird", "fly", "north") in ("no", "unknown")      # bird flies EAST, not north


def test_finer_period_unbind_decodes_stored_roles():
    """The finer-period unbind path (`_unbind_phases(..., period=)`, the escalation's 'second look') recovers each
    stored role correctly -- a sanity check that the period override threads through `_resonate` and the readout is
    at least as accurate as the default period."""
    c = _build(42)
    fact, comp = c.kb[0]                       # ('dog','go','north')
    comp = np.asarray(comp)
    for role, expect in (("agent", "dog"), ("action", "go"), ("patient", "north")):
        fine = c._unbind_phases(comp, role, period=2000)
        assert c._cleanup(fine) == expect


def test_period_override_default_matches_self_period():
    """`_resonate(..., period=None)` uses self.period -> byte-identical to the pre-change call (the OFF path)."""
    c = _build(42)
    fact, comp = c.kb[2]                        # ('bird','fly','east')
    comp = np.asarray(comp)
    default = c._unbind_phases(comp, "action")               # period=None -> self.period (200)
    explicit = c._unbind_phases(comp, "action", period=None)
    assert np.array_equal(default, explicit)
    assert c._cleanup(default) == "fly"
