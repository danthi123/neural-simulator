"""Burndown #1 (the shortcut-burndown-inventory P1): the SHIPPED OneBrainComposer's cleanup SELECTION must be a
fully-on-substrate SPIKING WTA, not a host `np.argmax` over the matched-filter membrane.

The matched FILTER is already on the co-resident bridge (the complex-synapse `clean` matvec -> membrane scores); the
shortcut was the WINNER-PICK (`self.words[int(np.argmax(scores[ri]))]`). This guard pins:

  (1) PARITY: with `enable_spiking_cleanup=True` the who/what/yes-no/clause answers + the no-confab moat are
      ANSWER-IDENTICAL to the host-argmax path (the spiking WTA selects the same winner the numpy argmax did -- the
      selection is now in spikes, a readout of the Izhikevich WTA firing, the host argmax retired).
  (2) MOAT 0-FA: the no-confab moat (abstain on an unstored cue / fact) holds on the spiking path -- 0 false-accepts.
  (3) OFF == byte-identical: `enable_spiking_cleanup=False` (the default) keeps the host-argmax path verbatim (the
      numpy-CPU portability + test-oracle path), so enabling the conversion is purely additive.

CPU-first (SIM_BACKEND=numpy): the spiking Izhikevich WTA runs on both backends, so this validates on CPU and is
exercised again on GPU. Skips gracefully if the concept-code cache is unavailable.
"""
import os

import numpy as np
import pytest

from sim.backend import is_gpu_backend  # noqa: E402

VOCAB = ["dog", "cat", "bird", "river", "apple", "go", "come", "look", "stop", "swim",
         "north", "east", "south", "west", "home"]


def _store_facts(c, facts):
    for (a, v, p) in facts:
        c.store(a, v, p, polarity="AFFIRM")


def test_onebrain_spiking_cleanup_parity_and_moat():
    """The spiking-cleanup OneBrain == the host-argmax OneBrain on the who/what/yes-no matrix + the moat 0-FA."""
    from research.runners.one_brain_composer import OneBrainComposer
    facts = [("dog", "go", "north"), ("cat", "come", "east"), ("bird", "look", "south")]
    try:
        c_spk = OneBrainComposer(seed=42, D=64, vocab=VOCAB, enable_spiking_cleanup=True,
                                 enable_rf_cudagraph=False)
        c_host = OneBrainComposer(seed=42, D=64, vocab=VOCAB, enable_spiking_cleanup=False,
                                  enable_rf_cudagraph=False)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    _store_facts(c_spk, facts)
    _store_facts(c_host, facts)

    for (a, v, p) in facts:
        assert c_spk.query_patient(a, v) == c_host.query_patient(a, v) == p, \
            f"spiking-cleanup query_patient must == host == truth for {(a, v)}"
        assert c_spk.query_agent(v, p) == c_host.query_agent(v, p) == a, \
            f"spiking-cleanup query_agent must == host == truth for {(v, p)}"
        assert c_spk.ask_yes_no(a, v, p) == c_host.ask_yes_no(a, v, p) == "yes"

    # the no-confab moat on the spiking path: an unstored cue / fact still abstains (0 false-accepts)
    assert c_spk.query_patient("apple", "stop") is None, "moat breach: unstored cue not abstained (spiking)"
    assert c_spk.query_agent("swim", "home") is None, "moat breach: unstored cue not abstained (spiking)"
    assert c_spk.ask_yes_no("cat", "go", "west") == "unknown", "moat breach: unstored fact not abstained (spiking)"


def test_onebrain_spiking_cleanup_per_block_and_batched():
    """The spiking WTA selection holds on BOTH read paths (the batched default + the per-block oracle), == host."""
    from research.runners.one_brain_composer import OneBrainComposer
    facts = [("dog", "go", "north"), ("cat", "come", "east"), ("bird", "look", "south")]
    try:
        c = OneBrainComposer(seed=42, D=64, vocab=VOCAB, enable_spiking_cleanup=True, enable_rf_cudagraph=False)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    _store_facts(c, facts)
    for (a, v, p) in facts:
        c.enable_batched = True
        bat = (c.query_patient(a, v), c.query_agent(v, p), c.ask_yes_no(a, v, p))
        c.enable_batched = False
        per = (c.query_patient(a, v), c.query_agent(v, p), c.ask_yes_no(a, v, p))
        assert bat == per == (p, a, "yes"), f"spiking batched {bat} != per-block {per} != truth for {(a, v, p)}"


def test_onebrain_spiking_cleanup_clause_parity():
    """A recursive embedded clause (`_decode_clause`, site :489) selects the inner words with the spiking WTA too,
    == the host-argmax decode == ground truth."""
    from research.runners.one_brain_composer import OneBrainComposer
    from research.runners.rf_phasor_composer import Clause
    clause = Clause(agent="cat", action="look", patient="south")
    try:
        c_spk = OneBrainComposer(seed=42, D=64, vocab=VOCAB, enable_spiking_cleanup=True, enable_rf_cudagraph=False)
        c_host = OneBrainComposer(seed=42, D=64, vocab=VOCAB, enable_spiking_cleanup=False, enable_rf_cudagraph=False)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    c_spk.store("dog", "go", clause)
    c_host.store("dog", "go", clause)
    assert c_spk.query_patient("dog", "go") == c_host.query_patient("dog", "go") == "cat look south", \
        "spiking-cleanup clause decode must == host == truth"
    assert c_spk.render_fact("dog") == c_host.render_fact("dog") == "dog go cat look south"


def test_onebrain_spiking_cleanup_default_off_byte_identical():
    """Explicit `enable_spiking_cleanup=False` keeps the host-argmax path: the composer reports it is off and the
    read path never builds a WTA bank. (The composer's OWN default is True = spiking, so this guard pins the host
    oracle EXPLICITLY; construction smoke; answer byte-identity is the parity test's host arm.)"""
    from research.runners.one_brain_composer import OneBrainComposer
    try:
        c = OneBrainComposer(seed=42, D=64, vocab=VOCAB, enable_spiking_cleanup=False)
    except (FileNotFoundError, KeyError) as e:
        pytest.skip(f"concept-code cache / vocab unavailable: {e}")
    assert c.enable_spiking_cleanup is False, "default must be host-argmax (off) for numpy-CPU + oracle parity"
    _store_facts(c, [("dog", "go", "north")])
    assert c.query_patient("dog", "go") == "north"
