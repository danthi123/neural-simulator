"""Tests for PhasorAssociativeMemory -- the LEARNED-code foundation of phasor substrate unification.

Where the nesting agent uses CONSTRUCTED phasor codes, this memory LEARNS the map from a grounded sparse
word cue to a phasor concept code via online weight-bounded spike-timing plasticity (validated cheap-first:
research/findings/2026-06-03-spiking-STDP-learns-phasor-map-RESOLVES-algorithmic.md, incl. the grounded
vocab_to_drive_pattern encoder). These tests pin: learning + recall, abstention on the unlearned, and that
the learned codes compose (bind/unbind).
"""

from research.runners.phasor_associative_memory import PhasorAssociativeMemory

WORDS = ["apple", "river", "dog", "cat", "big", "small", "hot", "cold"]


def _mem(seed=42, **kw):
    m = PhasorAssociativeMemory(seed=seed, **kw)
    for w in WORDS:
        m.learn(w)
    return m


def test_recall_learned_word():
    m = _mem()
    assert m.recall("dog") == "dog"            # a learned word recalls itself


def test_recall_all_learned_words():
    m = _mem()
    assert all(m.recall(w) == w for w in WORDS)   # every learned word recalls correctly


def test_abstain_on_unlearned_word():
    m = _mem()
    assert m.recall("zebra") is None           # never learned -> abstain (no confabulation)


def test_learned_codes_compose():
    # the learned (not constructed) codes must bind/unbind through roles -- the migration's whole point
    m = _mem()
    bundle = m.bundle([m.bind("AGENT", "dog"), m.bind("PATIENT", "cat")])
    assert m.unbind_cleanup("AGENT", bundle) == "dog"
    assert m.unbind_cleanup("PATIENT", bundle) == "cat"


def test_multi_seed_recall_robust():
    ok = 0
    for seed in (42, 43, 44):
        m = _mem(seed)
        ok += int(all(m.recall(w) == w for w in WORDS))
    assert ok == 3, f"recall robust on {ok}/3 seeds"


def test_recall_confidence_separates_learned_from_unlearned():
    m = _mem()
    learned_conf = m.recall_confidence("dog")
    unlearned_conf = m.recall_confidence("zebra")
    assert learned_conf > unlearned_conf       # learned words read out more confidently than novel ones
