"""Regression gate for issue #184 (2026-09-01): `TieredFactStore.last_trace` DID NOT PROPAGATE the LTM tier's own
match trace when the buffer sub-call abstained and the LTM tier answered -- `composer.last_trace` (what
`webapp/server.py`'s metacog confidence read consults) stayed on the buffer's own ABSTAIN record even though a
real, matched fact was returned. Root cause was TWO-PART:

  1. `ShardedPhasorStore` (the LTM) emitted NO `last_trace` at all -- each shard is an independent
     `RFPhasorComposer` with its OWN `.trace`/`.last_trace`, and nothing armed a shard's flag or read its result
     back up to the store level.
  2. `TieredFactStore._tiered()` never copied the answering tier's trace to where callers read it
     (`composer.last_trace` -> `__getattr__` -> `self.buffer.last_trace`).

See `research/FAILURE_LOG.md` (2026-09-01 entry) and
`research/findings/2026-09-01-confidence-kb-relation-realtraffic.md` for the real-traffic measurement this
unblocks. Candidate gate this failure log entry named: "a regression test building a `TieredFactStore` with an
LTM-only fact, calling `query_patient`, and asserting `composer.last_trace['abstained'] is False`" -- this file.
"""
from __future__ import annotations

import pytest

from research.runners.rf_phasor_composer import RFPhasorComposer
from research.runners.sharded_phasor_store import ShardedPhasorStore
from research.runners.tiered_fact_store import TieredFactStore

VOCAB = ["dog", "go", "north", "cat", "run", "south", "asimov_isaac", "employer", "university_of_boston"]


def _tiered(seed=42):
    buffer = RFPhasorComposer(seed=seed, D=64, period=200, vocab=VOCAB)
    ltm = ShardedPhasorStore(n_shards=4, seed=seed, D=64, vocab=VOCAB, period=200)
    return TieredFactStore(buffer, ltm), buffer, ltm


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_ltm_answered_turn_propagates_a_real_trace(seed):
    """THE #184 GATE. A fact stored ONLY in the LTM tier (the buffer never sees it, so the buffer's own
    `query_patient` genuinely abstains): the tiered read must (a) still return the correct answer (unchanged
    moat/recall behavior) and (b) leave `composer.last_trace` holding the LTM's OWN match trace -- NOT the
    buffer's abstain record. Before the fix, (b) failed: `last_trace['abstained']` read True (the buffer's own
    abstain) even though the overall call answered correctly."""
    tiered, buffer, ltm = _tiered(seed)
    ltm.store("asimov_isaac", "employer", "university_of_boston")

    tiered.trace = True         # mirrors webapp/server.py's per-turn `_composer.trace = True`
    tiered.last_trace = None    # mirrors the per-turn reset

    ans = tiered.query_patient("asimov_isaac", "employer")
    assert ans == "university_of_boston"

    trace = tiered.last_trace
    assert trace is not None, "#184 regression: an LTM-answered turn left last_trace empty"
    assert trace["abstained"] is False, "#184 regression: last_trace still reports the buffer's own abstain"
    assert trace.get("roles"), "the LTM's trace must carry real per-role decode chips, not an empty shell"
    # the confidence is a REAL read off the matched block (`_cleanup_all_score_stats`), never fabricated --
    # every role that produced a decode carries a numeric confidence.
    decoded_roles = [r for r in trace["roles"] if r.get("word") is not None]
    assert decoded_roles, "expected at least one decoded role in the LTM's trace"
    for r in decoded_roles:
        assert isinstance(r.get("confidence"), float)


def test_buffer_answered_turn_is_byte_identical_to_before_the_fix():
    """A buffer-answered turn must be UNCHANGED by this fix: the buffer's own trace already flowed correctly
    (this was never the #184 bug), and the fix must not touch that path."""
    tiered, buffer, ltm = _tiered(42)
    buffer.store("dog", "go", "north")
    ltm.store("cat", "run", "south")   # a decoy fact in the OTHER tier -- must not interfere

    tiered.trace = True
    tiered.last_trace = None
    ans = tiered.query_patient("dog", "go")
    assert ans == "north"

    trace = tiered.last_trace
    assert trace is not None
    assert trace["abstained"] is False
    # the buffer's own composer trace shape is untouched (composer == 'rf', from RFPhasorComposer._trace_scan)
    assert trace.get("composer") == "rf"


def test_both_tiers_abstain_stays_a_clean_abstain_no_confab():
    """The no-confab moat: an agent unknown to BOTH tiers must abstain (return None) and the propagated trace
    must honestly report abstained=True -- the fix must never manufacture a match out of two genuine misses."""
    tiered, buffer, ltm = _tiered(42)
    buffer.store("dog", "go", "north")
    ltm.store("cat", "run", "south")

    tiered.trace = True
    tiered.last_trace = None
    ans = tiered.query_patient("totally_unknown_entity", "go")
    assert ans is None

    trace = tiered.last_trace
    assert trace is not None
    assert trace["abstained"] is True


def test_ltm_none_stays_byte_identical():
    """`TieredFactStore(buffer, ltm=None)` degrades to exactly the buffer (the class's own documented contract)
    -- this fix must not disturb that. Setting `.trace` must not raise with no LTM present."""
    buffer = RFPhasorComposer(seed=42, D=64, period=200, vocab=VOCAB)
    buffer.store("dog", "go", "north")
    tiered = TieredFactStore(buffer, ltm=None)

    tiered.trace = True   # must not raise despite ltm is None
    tiered.last_trace = None
    assert tiered.query_patient("dog", "go") == "north"
    assert tiered.last_trace["abstained"] is False

    assert tiered.query_patient("cat", "run") is None
    assert tiered.last_trace["abstained"] is True


def test_sharded_phasor_store_trace_property_propagates_to_every_shard():
    """`ShardedPhasorStore.trace = True` must arm EVERY shard's own `.trace` flag (any shard may be the one a
    routed query lands on) -- and default False leaves every shard's tracing off (byte-identical for every
    caller that never opts in)."""
    ltm = ShardedPhasorStore(n_shards=8, seed=42, D=64, vocab=VOCAB, period=200)
    assert ltm.trace is False
    assert all(sh.trace is False for sh in ltm.shards)

    ltm.trace = True
    assert all(sh.trace is True for sh in ltm.shards)

    ltm.trace = False
    assert all(sh.trace is False for sh in ltm.shards)
