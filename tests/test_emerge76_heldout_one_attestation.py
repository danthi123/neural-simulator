"""CI for EMERGE-76 -- ONE attestation of a fully-HELD-OUT construction frame's OWN distinctive element (function word /
inflection / does<not internal order) recovers its distinctive slot + order in one shot (one-shot / fast-mapping),
closing the EMERGE-63/64/65 held-out DISTINCTIVE-slot residual as a SINGLE-EXEMPLAR DATA residual (NOT a wall).

CPU/numpy, offline. Small-stream smoke of: the zero-attestation held-out baseline does NOT recover the distinctive
slot (the EMERGE-63/64 residual, reproduced as the zero control); ONE attestation of the held frame's own canonical
sentence recovers it (inventory + order + exact spiking render); the permuted-attestation input-destruction control
collapses; the F_INTR reader-unreadable-verb sub-residual is the honestly-named inherited morphology-reader limit; and
the gate-first no-confab moat holds (0 producer invocations on abstains).
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("SIM_BACKEND", "numpy")
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from research.runners._emerge59_spiking_broca_frame_slots_derisk import (  # noqa: E402
    FRAME_NAMES, build_heldout_facts,
)
from research.runners._emerge62_discover_function_words_derisk import (  # noqa: E402
    build_stream, compute_stats, discover_closed_class,
)
from research.runners._emerge63_corpus_taught_slot_order_derisk import split_sentences  # noqa: E402
from research.runners._emerge76_heldout_one_attestation_derisk import (  # noqa: E402
    recover_held_frame, build_heldout_corpus_sentences, _readable_intr_verbs, _unreadable_intr_verbs,
    _held_frame_of_sentence,
)


def _prep(seed=42, n=6000):
    tokens = build_stream(seed, n_sentences=n)
    base_sents = split_sentences(tokens)
    words, freq, cover, _ = compute_stats(tokens)
    closed, *_ = discover_closed_class(words, freq, cover)
    facts = build_heldout_facts(seed, n=4)
    return base_sents, closed, facts


def test_zero_attestation_is_the_heldout_residual():
    """K=0 (no attestation of the held frame -- the EMERGE-63/64 held-out baseline): the held frame's DISTINCTIVE slot +
    order are NOT recovered (its exemplars are absent, only the OTHER two frames attest the shared backbone). This is the
    load-bearing zero control -- the recovery at K=1 must NOT be smuggled from the other frames."""
    base_sents, closed, facts = _prep(42)
    for held in FRAME_NAMES:
        rec = recover_held_frame(base_sents, closed, held, 42, 0, facts)
        assert rec["recovered"] is False, f"held-out {held} should NOT recover at K=0 (the residual)"
        assert rec["exact"] == pytest.approx(0.0), f"held-out {held} K=0 exact should be 0 (not attested)"


def test_one_attestation_recovers_distinctive_slot_and_order():
    """K=1 (a SINGLE well-formed attestation of the held frame's own canonical sentence): the held frame's DISTINCTIVE
    slot (F_MODAL's can+position / F_NEGMOD's does,not + does<not order / F_INTR's 3sg) + its order are recovered EXACTLY
    and it renders EXACT on spikes -- the one-shot / fast-mapping claim (Carey-Bartlett; CLS one-exposure)."""
    base_sents, closed, facts = _prep(42)
    for held in FRAME_NAMES:
        rec = recover_held_frame(base_sents, closed, held, 42, 1, facts)
        assert rec["inventory_recovered"] is True, f"held-out {held} inventory not recovered at K=1"
        assert rec["order_acc"] == pytest.approx(1.0), f"held-out {held} order not recovered at K=1"
        assert rec["exact"] == pytest.approx(1.0), f"held-out {held} did not render exact at K=1"
        assert rec["recovered"] is True, f"held-out {held} not recovered at K=1 (single attestation)"


def test_one_attestation_beats_zero_residual():
    """The single attestation is LOAD-BEARING: the K=1 exact render clears the K=0 residual by a clear margin for every
    held-out frame (else the recovery would be smuggled from the OTHER frames, not the attestation)."""
    base_sents, closed, facts = _prep(42)
    for held in FRAME_NAMES:
        one = recover_held_frame(base_sents, closed, held, 42, 1, facts)["exact"]
        zero = recover_held_frame(base_sents, closed, held, 42, 0, facts)["exact"]
        assert one - zero >= 0.30, f"held-out {held}: K=1 ({one:.2f}) does not clear K=0 ({zero:.2f}) by >=0.30"


def test_permuted_attestation_collapses():
    """PERMUTED-ATTESTATION (K=1 with the attestation's word order shuffled): the distinctive slot's POSITION / the
    does<not internal order is destroyed -> the recovery collapses (the recovery must come from the attestation's WORD
    ORDER, not merely its token presence). The decisive input-destruction control."""
    base_sents, closed, facts = _prep(42)
    for held in FRAME_NAMES:
        one = recover_held_frame(base_sents, closed, held, 42, 1, facts)["exact"]
        perm = []
        for j in range(4):
            srng = np.random.default_rng(2000 + j)
            perm.append(recover_held_frame(base_sents, closed, held, 42, 1, facts,
                                            shuffle_attest=True, shuffle_rng=srng)["exact"])
        perm_exact = float(np.mean(perm))
        assert one - perm_exact >= 0.30, \
            f"held-out {held}: permuted-attestation did not collapse (one {one:.2f} vs perm {perm_exact:.2f})"


def test_heldout_negmod_does_not_internal_order_recovered_by_one_attestation():
    """The hardest case (EMERGE-63's named residual): a fully-held-out F_NEGMOD's does<not INTERNAL order -- unlearnable
    from the OTHER two frames -- is recovered EXACTLY by ONE attestation of F_NEGMOD's own canonical sentence."""
    base_sents, closed, facts = _prep(42)
    zero = recover_held_frame(base_sents, closed, "F_NEGMOD", 42, 0, facts)
    one = recover_held_frame(base_sents, closed, "F_NEGMOD", 42, 1, facts)
    assert zero["order_acc"] < 0.999, "held-out F_NEGMOD does<not order should NOT be recoverable at K=0 (the residual)"
    assert one["order_acc"] == pytest.approx(1.0), "one attestation should recover F_NEGMOD's does<not internal order"


def test_intr_reader_unreadable_verb_is_named_residual():
    """The honestly-named inherited sub-residual: a SINGLE F_INTR attestation whose 3sg surface the EMERGE-64 morphology
    reader (single-`s` strip) CANNOT parse (lurk/wait/sit/sleep -> mis-read as bare) fails to recover F_INTR -- so the
    one-shot claim is about a WELL-FORMED (reader-parseable) exemplar; the reader's lexicon coverage is the next data
    signal. Reported, NOT a wall. (Skipped only if this stream/domain leaves the unreadable pool empty.)"""
    assert _readable_intr_verbs, "there should be reader-parseable intransitive verbs"
    if not _unreadable_intr_verbs:
        pytest.skip("no reader-unreadable intransitive verbs in this domain")
    base_sents, closed, facts = _prep(42)
    good = recover_held_frame(base_sents, closed, "F_INTR", 42, 1, facts, unreadable_intr=False)
    bad = recover_held_frame(base_sents, closed, "F_INTR", 42, 1, facts, unreadable_intr=True)
    assert good["recovered"] is True, "a reader-parseable F_INTR attestation should recover"
    assert bad["recovered"] is False, "a reader-unreadable F_INTR attestation should NOT recover (the named sub-residual)"


def test_moat_holds_through_recovery():
    """The gate-first no-confab MOAT is untouched: after ONE attestation recovers the held frame, 3 abstains invoke the
    producer 0 times and an answer is produced (the moat holds by construction)."""
    base_sents, closed, facts = _prep(42)
    for held in FRAME_NAMES:
        rec = recover_held_frame(base_sents, closed, held, 42, 1, facts)
        assert rec["moat_calls_on_abstain"] == 0, f"held-out {held}: producer invoked on abstain (moat breach)"
        assert rec["answer_produced"] is True, f"held-out {held}: answer not produced"


def test_attestation_added_only_for_held_frame():
    """The corpus construction is honest: build_heldout_corpus_sentences WITHHOLDS the held frame's base exemplars and
    adds K attestations of ONLY the held frame -- so the held frame appears exactly K times (its own attestations), and
    the OTHER frames are untouched (their base exemplars remain)."""
    base_sents, closed, facts = _prep(42)
    held = "F_NEGMOD"
    sents = build_heldout_corpus_sentences(base_sents, closed, held, 42, 1)
    held_count = sum(1 for s in sents if _held_frame_of_sentence(s, closed) == held)
    assert held_count == 1, f"held frame should appear exactly once (its single attestation), got {held_count}"
    # the other frames still present (their base exemplars were not withheld)
    other = [f for f in FRAME_NAMES if f != held]
    for f in other:
        cnt = sum(1 for s in sents if _held_frame_of_sentence(s, closed) == f)
        assert cnt > 0, f"other frame {f} should still be attested in the held-out corpus"
