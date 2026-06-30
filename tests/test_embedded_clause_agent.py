"""CI guard for embedded-clause PARSING wired into the production BrainConversationalAgent behind the default-OFF
`enable_embedded_clause` flag (the last host-constructed `Clause` replaced by a parsed one).

De-risk GO: 2026-06-19-embedded-clause-parse-derisk.md (mean 0.951) + 2026-06-19-embedded-clause-redundancy-polish.md
(redundancy lever -> 1.000 6/6). Tier-1 close-out item 3b. Mechanism source:
`_phaseB_embedded_clause_parse_derisk.py` (EmbeddedClauseParser.parse_nested + RedundantEmbeddedReadout), reused by
import into `BrainConversationalAgent.hear_nested` (mirroring the enable_attributed / enable_multiframe opt-in pattern).

What this asserts (CPU/numpy-runnable):
  * THE GAP (a nested embedded clause the FLAT parser drops): the production agent's flat `hear()` parser reads a
    relative-clause stream as a 3-word SVO and gets WRONG roles (or stores garbage), so `what_does` cannot recover
    the embedded clause.
  * THE FIX (enable_embedded_clause=True): `hear_nested("dog that chase cat run")` SEGMENTS the depth-1 relative,
    stores the matrix fact ("dog run (dog chase cat)") with the parsed embedded Clause as its patient, and
    `what_does("dog","run")` decodes the embedded clause -> "dog chase cat". Both subject- and object-relatives.
  * MOAT (never weakened): an unparseable/garbled stream -> abstain (hear_nested returns None, stores nothing);
    a never-stored query -> None.
  * DEFAULT BYTE-IDENTITY: `enable_embedded_clause` defaults OFF -> the parser is never constructed and hear_nested
    asserts-off; the existing test_brain_conversational_agent.py (incl. the host-constructed-clause test) passes
    verbatim.

These run on the rf composer with an explicit vocab so the `denoise64` cache is not needed (CPU/numpy). The
GPU 6-seed redundancy re-confirm of the redundancy default on the agent path is flagged for the controller (the
de-risk's 6-seed GPU GO at 1.000 already exists).
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

import pytest

from research.runners.brain_conversational_agent import BrainConversationalAgent, Clause
from research.runners._phaseB_embedded_clause_parse_derisk import NOUNS, VERBS

# the de-risk's probe lexicon (NOUNS disjoint from VERBS; 'that'/'which'/'who' are the relativizers).
VOCAB = sorted(set(NOUNS + VERBS))
SEED = 42  # the de-risk's clean GO seed


def _agent(enable_embedded_clause):
    """An rf-composer agent over the de-risk vocab; `enable_embedded_clause` toggles the parse_nested path.
    enable_neural_render=False keeps the answer the plain composer decode (no spiking serial-order render needed)."""
    return BrainConversationalAgent(seed=SEED, concepts={w: None for w in VOCAB},
                                    enable_neural_render=False,
                                    enable_embedded_clause=enable_embedded_clause)


def test_default_off_hear_nested_asserts():
    """DEFAULT (enable_embedded_clause OFF): the parser is never constructed and hear_nested asserts-off (the flag is
    required). This pins the byte-identical default behavior -- the embedded-clause path is entirely inert. The flat
    hear() path is unchanged (the full byte-identity guard is test_brain_conversational_agent.py)."""
    a = _agent(enable_embedded_clause=False)
    assert a.enable_embedded_clause is False
    assert a._embedded_parser is None                       # the parser is never built when the flag is off
    with pytest.raises(AssertionError):
        a.hear_nested("dog that chase cat run")             # hear_nested needs the flag ON
    with pytest.raises(AssertionError):
        a.query_nested("dog", "run")                        # query_nested also needs the flag ON
    assert a._embedded_parser is None                       # still never built (no side effect from the asserts)


def test_hear_nested_subject_relative_recovers_embedded_clause():
    """THE FIX (flag ON): a subject-relative 'dog that chase cat run' parses to matrix (dog run) + embedded
    Clause(dog chase cat); what_does('dog','run') decodes the embedded clause."""
    a = _agent(enable_embedded_clause=True)
    assert a.enable_embedded_clause is True
    a.composer.kb = []
    parsed = a.hear_nested("dog that chase cat run")
    assert parsed is not None and parsed["nested"] is True
    assert parsed["matrix"][:2] == ("dog", "run")           # matrix subject + verb (the suspended head)
    assert a.what_does("dog", "run") == "dog chase cat"     # the decoded embedded clause


def test_hear_nested_object_relative_recovers_embedded_clause():
    """Object-relative 'cat that dog chase run' -> embedded Clause(dog chase cat) (head=cat is the embedded
    PATIENT), matrix (cat run). The NO-SEGMENTATION flat reader provably CANNOT segment an object-relative -> this
    is the decisive case the parser must handle."""
    a = _agent(enable_embedded_clause=True)
    a.composer.kb = []
    parsed = a.hear_nested("cat that dog chase run")
    assert parsed is not None and parsed["nested"] is True
    assert parsed["matrix"][:2] == ("cat", "run")
    assert a.what_does("cat", "run") == "dog chase cat"     # head cat is the embedded patient


def test_hear_nested_moat_garbled_abstains():
    """A garbled / unparseable stream (no relativizer, no clean SVO) -> abstain (None), stores nothing. The
    no-confab moat is preserved."""
    a = _agent(enable_embedded_clause=True)
    a.composer.kb = []
    assert a.hear_nested("dog cat fish bird") is None       # no relativizer + not a clean SVO -> abstain
    assert a.hear_nested("dog that zzz cat run") is None     # unknown token -> abstain
    # nothing was stored -> a query still abstains (the moat)
    assert a.what_does("dog", "run") is None


def test_hear_nested_flat_svo_unregressed():
    """A flat (non-nested) SVO routed through hear_nested stores the plain fact (nested=False) and answers normally
    -- the parse_nested path does not regress flat comprehension."""
    a = _agent(enable_embedded_clause=True)
    a.composer.kb = []
    parsed = a.hear_nested("dog chase cat")
    assert parsed is not None and parsed["nested"] is False
    assert a.what_does("dog", "chase") == "cat"
