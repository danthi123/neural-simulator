"""CI guard for multi-referent pronoun disambiguation via WTA biased competition, wired into MultiTurnAgent
behind the default-OFF `enable_biased_competition` flag.

De-risk GO: 2026-06-19-multireferent-biased-competition-derisk.md (GO-arm 5/6, all anti-cheat controls 6/6).
Integration: 2026-06-19-multireferent-integration-multiturnagent.md.

What this asserts (CPU/numpy-runnable):
  * CAPABILITY (flag ON): a bare pronoun over >=2 held referents of OPPOSING content features ({cat=animate,
    ball=inanimate}) resolves to the CONTENT-favored referent — 'it'+'eat' -> cat (animate), and the feature-flip
    'it'+'roll' -> ball (inanimate). The full turn answers via the resolved referent. The answer is content-
    determined, NOT fact-availability: BOTH cat and ball have an 'eat' fact, so resolving wrongly would return a
    DIFFERENT (also non-None) answer.
  * MOAT (flag ON, never weakened): empty WM -> abstain (None); content-silent query verb (no selectional
    restriction) over >=2 held referents -> abstain (None). Zero confabulation.
  * FLAG-OFF byte-identity: with the flag default-OFF the biased-competition buffer is never constructed, and the
    existing test_multi_turn_agent.py passes verbatim (that file is the byte-identity guard; here we additionally
    assert the buffer is not built and the single-referent anaphora answer is unchanged).

These are the validated 2-referent decisive case. The seed-100 extreme-intrinsic-asymmetry case ABSTAINS (moat-
preserving, NOT a clean win) and the all-compatible-referent case are the two named follow-ons (see the finding).
"""
import os

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.multi_turn_agent import MultiTurnAgent

# Opposing-feature referents: cat (animate) vs ball (inanimate). 'eat' selects animate, 'roll' selects inanimate.
NOUNS = ["dog", "cat", "fish", "bird", "worm", "ball"]
VOCAB = NOUNS + ["chase", "eat"]
SEED = 42  # the validated GO seed (the de-risk's clean 3/3 on 42/43/44)


def _bc_agent():
    """A MultiTurnAgent with biased competition ON. BOTH cat and ball get an 'eat' fact, so the turn's answer is
    decided by WHICH referent the content bias resolves to (resolving wrongly returns a different non-None
    answer), not by fact availability — a stronger anti-cheat than a single-fact setup."""
    a = MultiTurnAgent(referent_concepts=NOUNS, concepts={w: None for w in VOCAB}, seed=SEED,
                       enable_biased_competition=True)
    a.agent.composer.store("cat", "eat", "fish")     # if 'it'->cat (correct for 'eat'), answer = fish
    a.agent.composer.store("ball", "eat", "worm")    # if 'it'->ball (wrong for 'eat'), answer = worm
    return a


def test_biased_competition_resolves_content_favored_referent():
    """Two held referents of opposing animacy; the query verb's selectional restriction picks the right one."""
    a = _bc_agent()
    a._write_referent("cat")
    a._write_referent("ball")
    assert a._held_set() == ["ball", "cat"]                 # both referents co-present
    # 'eat' selects animate -> cat -> (cat eat fish)
    assert a._resolve_biased("eat") == "cat"
    assert a.what_does("it", "eat") == "fish"
    # FEATURE-FLIP: 'roll' selects inanimate -> ball (proves the steer is content, not magnitude/recency)
    assert a._resolve_biased("roll") == "ball"


def test_biased_competition_content_not_recency():
    """Write-order flip: ball introduced first, cat most-recent. 'eat' (animate) must STILL resolve to cat —
    the bias is content, not recency (the prior NEGATIVE that recency could not solve)."""
    a = _bc_agent()
    a._write_referent("ball")     # older
    a._write_referent("cat")      # most recent
    assert a._resolve_biased("eat") == "cat"     # content wins over recency


def test_biased_competition_moat_empty_wm_abstains():
    """Empty WM -> no antecedent to confabulate -> abstain (None). The no-confab moat is preserved."""
    a = _bc_agent()
    assert a._resolve_biased("eat") is None
    assert a.what_does("it", "eat") is None


def test_biased_competition_moat_content_silent_abstains():
    """Two held referents but the query verb has NO selectional restriction ('see' not in VERB_SELECTS) ->
    content is silent -> abstain (the agent refuses to pick by intrinsic attractor strength). Moat preserved."""
    a = _bc_agent()
    a._write_referent("cat")
    a._write_referent("ball")
    assert a._resolve_biased("see") is None


def test_flag_off_buffer_not_built_and_anaphora_unchanged():
    """Flag default-OFF: the biased-competition buffer is never constructed, and single-referent anaphora answers
    exactly as the plain agent (byte-identical path). The full byte-identity guard is test_multi_turn_agent.py."""
    a = MultiTurnAgent(referent_concepts=NOUNS, concepts={w: None for w in VOCAB}, seed=SEED)  # flag default OFF
    assert a.enable_biased_competition is False
    assert a.bcw is None                                     # buffer never built when the flag is off
    a.agent.composer.store("cat", "eat", "fish")
    a.hear("dog chase cat")                                  # one referent ('cat') -> plain single-attractor path
    assert a.what_does("it", "eat") == "fish"
