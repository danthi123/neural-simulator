# Adaptive experiment design

`tools/adaptive_experiment.py` proposes the next bounded parameter batch. It does not run simulations or replace
the experiment harness. Every proposed candidate must still be added as an arm to a preregistered experiment spec,
sealed by `tools/experiment.py`, expanded into its normal jobs, and executed through those digest-bound job
contracts.

## What a design contains

- A repository-relative `experiment.spec_path`. Its declared backends and partitions are authoritative.
- Continuous, discrete, and categorical parameters with finite bounds or choices.
- Hard constraints expressed as JSON predicates. Each constraint requires a biological source or rationale.
- Objectives covering physiology, behavior, robustness, compute cost, and scaffold penalty. Each has a direction,
  weight, meaningful normalization range, and optional success target.
- Exactly three ordered fidelity tiers: NumPy CPU screening, CuPy GPU evaluation, and CuPy replication. Calibration
  and replication partitions are allowed; any held-out partition is rejected.
- Completed observations from non-held-out partitions and a deterministic policy seed.

Constraints use `lt`, `le`, `gt`, `ge`, `eq`, `ne`, `in`, `not_in`, `and`, `or`, and `not`. Numeric sides may be a
`{"param": "name"}`, a `{"value": 1.0}`, or an `add`, `sub`, `mul`, `div`, `min`, or `max` expression with `args`.

## Selection policy

The initial design uses a seeded scrambled Sobol sequence from SciPy, filters hard constraints, removes duplicates,
and greedily selects points farthest from prior observations. After enough CPU observations, a regularized radial-
basis surrogate predicts normalized weighted utility. The acquisition score adds a declared exploration weight to
model uncertainty. Strong CPU results can advance to GPU, and strong GPU results can advance to replication;
promotion ranking uses utility per declared fidelity cost, and no candidate advances directly from CPU to
replication.

Each batch includes deterministic sensitivity and pair-interaction diagnostics when enough data exists, otherwise
it marks them `insufficient_data`. It also reports a separate observed Pareto set for each fidelity and the
feasible-space fraction.

The decision is one of:

- `propose`: create candidate points and hand them to the normal preregistration/seal/job workflow.
- `stop`: replicated targets, observation budget, or a declared plateau condition has been reached.
- `escalate_to_research`: constraints leave too little feasible space, the finite space is exhausted without a
  promotable result, or a low-uncertainty surrogate predicts no material improvement.

## Usage

```bash
python tools/adaptive_experiment.py research/specs/my-adaptive-design.json \
  --output research/queue/my-next-batch.json
```

The output is created with exclusive-create semantics, made read-only, and carries a self-digest. Re-running with
the same output path fails rather than overwriting evidence.

## Current limits

This first layer uses a scalarized multi-objective utility for acquisition; the Pareto report is diagnostic rather
than a full multi-objective optimizer. Its RBF uncertainty is useful for ranking bounded searches but is not a
calibrated probability. It assumes completed observations report every objective, does not model censored or failed
runs, and proposes values only; materializing candidate arms remains an explicit preregistration step so the layer
cannot weaken `tools/experiment.py` seals or job contracts.
