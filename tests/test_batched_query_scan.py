"""Batched fact-store scan on RFPhasorComposer: the O(K) per-fact query loop replaced by ONE batched resonate +
one codebook matched-filter (the resonator-network pattern). Asserts the batched path is ANSWER-IDENTICAL to the
per-fact loop (and correct + abstaining), so it is a pure performance change. CPU/numpy; no sim/ edit.
"""
import os
import time

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
ABSENT = ("lion", "fly")   # not a stored (agent, action) cue


def _build(seed):
    c = RFPhasorComposer(seed=seed, D=128, vocab=VOCAB)
    for a, ac, p in FACTS:
        c.store(a, ac, p)
    return c


def _perfact(c, fn, *args):
    """Run a query with the batched fast-path forced OFF (the per-fact loop)."""
    orig = c._can_batch_scan
    c._can_batch_scan = lambda: False
    try:
        return fn(*args)
    finally:
        c._can_batch_scan = orig


@pytest.mark.parametrize("seed", [42, 43])
def test_query_patient_batched_equals_perfact_and_groundtruth(seed):
    c = _build(seed)
    assert c._can_batch_scan()
    for a, ac, p in FACTS:
        batched = c.query_patient(a, ac)
        perfact = _perfact(c, c.query_patient, a, ac)
        assert batched == perfact == p          # batched == per-fact == the stored ground truth
    # abstention: a never-stored cue -> None on BOTH paths (the no-confab moat preserved)
    assert c.query_patient(*ABSENT) is None
    assert _perfact(c, c.query_patient, *ABSENT) is None


@pytest.mark.parametrize("seed", [42, 43])
def test_query_agent_batched_equals_perfact(seed):
    c = _build(seed)
    for a, ac, p in FACTS:
        batched = c.query_agent(ac, p)
        perfact = _perfact(c, c.query_agent, ac, p)
        assert batched == perfact == a
    assert c.query_agent("fly", "mouse") is None    # never-stored (action, patient)


def test_ask_yes_no_batched_equals_perfact():
    c = RFPhasorComposer(seed=42, D=128, vocab=VOCAB)
    c.store("dog", "go", "north", polarity="AFFIRM")
    c.store("cat", "run", "south", polarity="NEGATE")
    assert c.ask_yes_no("dog", "go", "north") == _perfact(c, c.ask_yes_no, "dog", "go", "north") == "yes"
    assert c.ask_yes_no("cat", "run", "south") == _perfact(c, c.ask_yes_no, "cat", "run", "south") == "no"
    assert c.ask_yes_no("bird", "fly", "east") == "unknown"     # never stored -> abstain


def test_batched_speedup_smoke(capsys):
    """Not a hard assertion (timing is environment-dependent) -- prints the batched-vs-per-fact query time so the
    perf win is visible. The batched scan should be at least not-slower; at larger KB it is much faster."""
    c = _build(42)
    a, ac, _p = FACTS[-1]                       # the LAST fact -> the per-fact loop scans the whole store
    c.query_patient(a, ac)                       # warm
    t = time.time()
    for _ in range(5):
        c.query_patient(a, ac)
    batched_ms = (time.time() - t) / 5 * 1000
    t = time.time()
    for _ in range(5):
        _perfact(c, c.query_patient, a, ac)
    perfact_ms = (time.time() - t) / 5 * 1000
    with capsys.disabled():
        print(f"\n  [batched scan] query (KB={len(FACTS)}, worst-case last fact): "
              f"batched {batched_ms:.1f} ms vs per-fact {perfact_ms:.1f} ms "
              f"({perfact_ms / max(batched_ms, 1e-9):.1f}x)")
    assert batched_ms <= perfact_ms * 1.5        # never meaningfully slower
