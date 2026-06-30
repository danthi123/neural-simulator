"""CI guard for the CONTENT-GRADED bias wired into the production MultiTurnAgent behind the default-OFF
`graded_bias` knob.

De-risk GO: 2026-06-30 tier1-closeout-assessment (item 3a) + 2026-06-19-multireferent-graded-bias-polish.md
(GO-arm 6/6, bias-LESION breaks 6/6, moat 6/6). Mechanism source: `_phaseB_biased_competition_graded_derisk.py`
(reused by import into `MultiTurnAgent._resolve_biased`).

What this asserts (CPU/numpy-runnable):
  * THE GAP (graded OFF = the production default / fixed 2500 pA): the seed-100 extreme-intrinsic-asymmetry case
    MIS-RESOLVES — a 2-referent {cat, ball} buffer at seed 100, `roll` (selects inanimate) resolves to `cat`
    (WRONG; cat is animate) because the fixed bias cannot lift the intrinsically-dominated ball past cat. This is
    the documented pre-registered miss the graded bias closes (the abstain/mis-resolve is moat-preserving — NOT a
    confabulation — but the named fix is the graded steer).
  * THE FIX (graded ON): the deficit-scaled bias lifts ball past cat -> `roll` resolves CORRECTLY to `ball`, while
    the easy `eat` case (cat already competitive) stays correct (deficit=0 -> base magnitude -> no over-steer).
  * MOAT (graded ON, never weakened): empty WM -> abstain (None); content-silent query verb -> abstain (None).
  * DEFAULT BYTE-IDENTITY: `graded_bias` defaults OFF, so the production path is byte-identical to the fixed-bias
    behavior (asserted here against the seed-100 mis-resolve; the full byte-identity guard is
    test_multi_turn_agent.py + test_multireferent_biased_competition.py, which pass verbatim).

The buffer is built over the EXACT 2-referent {cat, ball} registry the de-risk uses (n=600, pattern_size=40 = the
MultiTurnAgent defaults), so the agent-path competition dynamics match the validated de-risk.
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.multi_turn_agent import MultiTurnAgent

# The de-risk's exact 2-referent setup: cat (animate) vs ball (inanimate); 'eat' selects animate, 'roll' inanimate.
NOUNS = ["cat", "ball"]
# the composer vocab needs the verbs AND the patient words of the stored facts (river/worm) so store() can encode them.
VOCAB = NOUNS + ["eat", "roll", "river", "worm"]
SEED = 100  # the pre-registered extreme-intrinsic-asymmetry miss (ball ~0 sel vs cat under 'roll')


def _agent(graded_bias):
    """A 2-referent MultiTurnAgent with biased competition ON; `graded_bias` toggles the deficit-scaled magnitude.
    Both cat and ball get a 'roll' fact, so the turn's answer is decided by WHICH referent resolves (resolving
    wrongly returns a different non-None answer), not by fact availability."""
    a = MultiTurnAgent(referent_concepts=NOUNS, concepts={w: None for w in VOCAB}, seed=SEED,
                       enable_biased_competition=True, graded_bias=graded_bias)
    a.agent.composer.store("ball", "roll", "river")   # if 'it'->ball (correct for 'roll'), answer = river
    a.agent.composer.store("cat", "roll", "worm")     # if 'it'->cat (wrong for 'roll'), answer = worm
    return a


def test_fixed_bias_default_mis_resolves_seed100_roll():
    """DEFAULT (graded_bias OFF == fixed 2500 pA): seed-100 'roll' MIS-RESOLVES to cat (the documented miss the
    graded bias closes). This pins the byte-identical default behavior — if this assertion ever flips, the default
    path changed."""
    a = _agent(graded_bias=False)
    assert a._graded_bias is False                       # default-OFF knob
    a._write_referent("cat")
    a._write_referent("ball")
    assert a._held_set() == ["ball", "cat"]
    # fixed bias cannot lift the intrinsically-dominated ball -> mis-resolves to cat (wrong for 'roll')
    assert a._resolve_biased("roll") == "cat"


def test_graded_bias_closes_seed100_roll():
    """THE FIX (graded_bias ON): the deficit-scaled bias resolves seed-100 'roll' correctly to ball, and the full
    turn answers via the resolved referent (river, not worm).

    NB the biased-competition READ advances the spiking bridge state (it re-drives the held assemblies + the WTA),
    so each assertion uses a FRESH agent and a SINGLE resolution read — successive reads of the SAME buffer can
    differ (the marginal seed-100 case), which is a property of the live spiking substrate, not a wiring bug. The
    first read on a fresh buffer is deterministic."""
    a = _agent(graded_bias=True)
    assert a._graded_bias is True
    a._write_referent("cat")
    a._write_referent("ball")
    # GRADED: 'roll' selects inanimate -> ball; the scaled bias lifts the dominated ball past cat (single read).
    assert a._resolve_biased("roll") == "ball"

    # the full turn answers via the resolved referent (river, not the wrong cat-fact worm). Fresh agent so this is
    # also a first-and-only resolution read.
    b = _agent(graded_bias=True)
    b._write_referent("cat")
    b._write_referent("ball")
    assert b.what_does("it", "roll") == "river"


def test_graded_bias_no_over_steer_on_easy_case():
    """The easy case ('eat' favors the already-competitive cat) stays correct under the graded bias (deficit -> 0
    -> base magnitude -> no over-steer)."""
    a = _agent(graded_bias=True)
    a._write_referent("cat")
    a._write_referent("ball")
    assert a._resolve_biased("eat") == "cat"             # animate favored, already competitive


def test_graded_bias_moat_empty_wm_abstains():
    """Empty WM -> abstain (None). The no-confab moat is preserved under the graded bias."""
    a = _agent(graded_bias=True)
    assert a._resolve_biased("roll") is None
    assert a.what_does("it", "roll") is None


def test_graded_bias_moat_content_silent_abstains():
    """Two held referents but a query verb with NO selectional restriction ('see') -> content silent -> abstain."""
    a = _agent(graded_bias=True)
    a._write_referent("cat")
    a._write_referent("ball")
    assert a._resolve_biased("see") is None
