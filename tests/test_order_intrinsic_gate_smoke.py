"""Import / signature smoke + pure-helper CPU unit tests for the
order-intrinsic Task 7 pre-registered multi-seed capability gate.

Per the implementation plan (project pattern): the gate's integration
is validated by the pre-registered multi-seed GPU run itself, NOT by a
contrived orchestration unit test here. This file pins ONLY:

  1. the module imports cheaply (heavy sim.*/bridge imports are lazy
     inside main()) and exposes `main` + the two PURE helpers; and
  2. real CPU unit tests for the two PURE helpers this gate adds
     (`_freeze_propositions`, `_production_top_rate`) -- they are
     anti-cheat-load-bearing (the frozen disjoint train/held-out
     split + THE one top-rate aggregation), so a regression must be
     caught at CPU speed. order_intrinsic_core is NOT touched -- these
     new pure helpers + their tests live in the gate module's own
     test file.
"""


def test_gate_module_exposes_expected_api():
    # Must import without pulling sim.*/cupy (heavy imports are lazy
    # inside main()) -> the smoke is instant.
    import research.runners.order_intrinsic_gate as og

    for fn in ("main", "_freeze_propositions", "_production_top_rate"):
        assert hasattr(og, fn), fn


def test_freeze_propositions_is_pure_disjoint_and_resume_stable():
    from research.runners.order_intrinsic_gate import (
        _freeze_propositions,
    )

    train, heldout = _freeze_propositions(42)
    # both non-empty; held-out has >= 2 props (aggregate_multiseed
    # requires every seed to contribute >= 1; we freeze 3).
    assert train and heldout
    assert len(heldout) >= 2

    # DISJOINT: a held-out prop must NEVER appear in train (else it
    # could tune the frozen floor -- anti-cheat invariant 4).
    train_set = {tuple(p) for p in train}
    heldout_set = {tuple(p) for p in heldout}
    assert train_set.isdisjoint(heldout_set)

    # every prop is an ordered pair of DISTINCT direction words from
    # the substrate's fixed 4-direction vocab.
    vocab = {"north", "east", "south", "west"}
    for p in train + heldout:
        assert len(p) == 2 and p[0] != p[1]
        assert set(p) <= vocab

    # PURE / resume-stable: same seed -> byte-identical split (the
    # frozen sidecar relies on this on resume).
    t2, h2 = _freeze_propositions(42)
    assert t2 == train and h2 == heldout

    # different seeds -> (generally) different held-out sets, so the
    # split is genuinely seed-derived, not constant.
    _t43, h43 = _freeze_propositions(43)
    assert h43 != heldout

    # every held-out prop has a non-identity permuted-ORDER control
    # (a length-2 ordered prop's reversal) -> the load-bearing
    # anti-cheat always applies.
    for p in heldout:
        assert [p[1], p[0]] != p


def test_production_top_rate_is_the_one_aggregation():
    from research.runners.order_intrinsic_gate import (
        _production_top_rate,
    )

    # max over the position-sweep's per-slot max concept rate.
    per_pos = [
        {"north": 0.01, "east": 0.05},
        {"north": 0.12, "east": 0.02},
        {"north": 0.00, "east": 0.00},
    ]
    assert abs(_production_top_rate(per_pos) - 0.12) < 1e-12

    # empty position dicts contribute 0.0; empty sweep -> 0.0.
    assert _production_top_rate([{}, {}]) == 0.0
    assert _production_top_rate([]) == 0.0

    # a single slot -> its own max; deterministic / pure.
    assert abs(_production_top_rate([{"a": 0.3, "b": 0.7}]) - 0.7) < 1e-12
