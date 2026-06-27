"""CI guard for Tier 1.1 — the entity-instance / discourse-referent layer (the KEYSTONE).

Guards the production research/runners/entity_instance_layer.py: allocate two same-type instances as
pattern-separated barcode tokens (DG separation), attach distinguishing facts to the INSTANCE, resolve
"which boy?" by the biased-competition WTA, render the clarification + answer, pattern-complete a pronoun to the
held referent, and the no-confab moat (0 false-accepts). The load-bearing anti-cheats — the MERGE LESION (alpha=0
collapses the instances -> disambiguation breaks) and the BINDING LESION (sever instance->fact binding -> abstain)
— are asserted. All CPU/numpy. See research/findings/2026-06-27-tier1-entity-instances-GO.md.
"""
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

os.environ.setdefault("SIM_BACKEND", "numpy")

from research.runners.rf_phasor_composer import RFPhasorComposer  # noqa: E402
from research.runners.entity_instance_layer import (  # noqa: E402
    EntityInstanceLayer, phase_cos, ALPHA_DEFAULT)

VOCAB = ["boy", "girl", "dog", "cat", "go", "eat", "chase", "see",
         "park", "apple", "bone", "ball", "river"]


def _two_boys(seed=42, alpha=ALPHA_DEFAULT):
    """Allocate boy#1 (go->park) + boy#2 (eat->apple) on a fresh composer; return (layer, b1, b2)."""
    comp = RFPhasorComposer(seed=seed, D=128, vocab=VOCAB)
    L = EntityInstanceLayer(comp, alpha=alpha)
    b1, b2 = L.allocate("boy"), L.allocate("boy")
    L.store_fact(b1, "go", GOAL="park")
    L.store_fact(b2, "eat", patient="apple")
    return L, b1, b2


@pytest.fixture(scope="module")
def two_boys():
    return _two_boys(seed=42)


# --- GATE (a): PATTERN SEPARATION (instances stay separable, type-linked) ----------------------------------------
def test_instances_are_separated(two_boys):
    L, b1, b2 = two_boys
    z1, z2, zt = L.comp.concepts[b1], L.comp.concepts[b2], L.comp.concepts["boy"]
    inst_inst = phase_cos(z1, z2)
    inst_type = 0.5 * (phase_cos(z1, zt) + phase_cos(z2, zt))
    # the two same-type instances are MORE separated from each other than from the type they share (the DG
    # pattern-separation signature) and near the ~0 random floor for D=128.
    assert inst_inst < inst_type, f"inst-inst {inst_inst:+.3f} !< inst-type {inst_type:+.3f}"
    assert inst_inst < 0.2, f"instances not decorrelated: inst-inst {inst_inst:+.3f}"
    assert inst_type > 0.05, "instances lost their type linkage entirely"


def test_merge_lesion_breaks_disambiguation():
    """Anti-cheat (load-bearing): alpha=0 collapses the two instances to identical type codes -> the agent-binding
    cannot individuate -> the system cannot recover the distinct (boy#1, boy#2) pair."""
    Lm, mb1, mb2 = _two_boys(seed=42, alpha=0.0)
    assert phase_cos(Lm.comp.concepts[mb1], Lm.comp.concepts[mb2]) > 0.99  # collapsed to one code
    park, _ = Lm.which("boy", action="go", GOAL="park")
    apple, _ = Lm.which("boy", action="eat", patient="apple")
    merged_distinct_correct = (park == mb1 and apple == mb2)
    assert not merged_distinct_correct, "merge lesion failed to break disambiguation (separation not load-bearing)"


# --- GATE (b): RIGHT REFERENT ("which boy went to the park?" -> boy#1) -------------------------------------------
def test_which_resolves_right_referent(two_boys):
    L, b1, b2 = two_boys
    assert L.which("boy", action="go", GOAL="park")[0] == b1
    assert L.which("boy", action="eat", patient="apple")[0] == b2


def test_answer_and_clarification_text(two_boys):
    L, b1, b2 = two_boys
    tok, ans = L.answer_which("boy", action="go", GOAL="park")
    assert tok == b1
    assert ans == "the boy that went to the park"
    text, n = L.clarify_which("boy")
    assert n == 2
    assert "went to the park" in text and "ate the apple" in text and " or " in text


def test_binding_lesion_breaks_disambiguation(two_boys):
    """Anti-cheat (load-bearing): ignore the instance->fact binding -> both instances match the cue equally ->
    abstain (the cue no longer individuates)."""
    L, b1, b2 = two_boys
    assert L.which("boy", sever_binding=True, action="go", GOAL="park")[0] is None


def test_pronoun_resolves_to_held_referent_and_abstains_when_empty(two_boys):
    L, b1, b2 = two_boys
    L.reset_discourse()
    L._held = [b2]                                  # discourse just mentioned boy#2
    assert L.resolve_pronoun(type_name="boy") == b2
    L.reset_discourse()
    assert L.resolve_pronoun(type_name="boy") is None   # empty file-card -> no antecedent -> abstain


# --- GATE (c): MOAT, 0 FALSE-ACCEPTS -----------------------------------------------------------------------------
def test_moat_abstains_on_unstored(two_boys):
    L, b1, b2 = two_boys
    assert L.which("boy", action="chase", patient="cat")[0] is None      # no boy chased the cat
    assert L.which("girl", action="go", GOAL="park")[0] is None          # no girl ever allocated
    assert L.comp.query_patient(b1, "chase") is None                     # composer moat on an unstored predicate


# --- robustness: 3+ same-type instances; a genuine tie abstains (the moat IS the clarification trigger) ----------
def test_three_instances_tie_abstains_unique_resolves():
    comp = RFPhasorComposer(seed=42, D=128, vocab=VOCAB)
    L = EntityInstanceLayer(comp)
    c1, c2, c3 = L.allocate("boy"), L.allocate("boy"), L.allocate("boy")
    L.store_fact(c1, "go", GOAL="park")
    L.store_fact(c2, "eat", patient="apple")
    L.store_fact(c3, "go", GOAL="park")            # c3 ALSO went to the park -> ambiguous
    assert L.which("boy", action="go", GOAL="park")[0] is None      # two match -> tie -> abstain
    assert L.which("boy", action="eat", patient="apple")[0] == c2   # unique -> resolves


def test_multiseed_separation_and_disambiguation():
    """The GO claim: separation + right-referent + moat across 6 seeds."""
    for seed in (42, 43, 44, 100, 101, 102):
        L, b1, b2 = _two_boys(seed=seed)
        z1, z2, zt = L.comp.concepts[b1], L.comp.concepts[b2], L.comp.concepts["boy"]
        assert phase_cos(z1, z2) < 0.5 * (phase_cos(z1, zt) + phase_cos(z2, zt)), f"seed {seed} not separated"
        assert L.which("boy", action="go", GOAL="park")[0] == b1, f"seed {seed} wrong referent"
        assert L.which("boy", action="chase", patient="cat")[0] is None, f"seed {seed} moat breach"
