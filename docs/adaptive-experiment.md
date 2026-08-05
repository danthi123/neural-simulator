# Adaptive experiment design

`tools/adaptive_experiment.py` proposes the next bounded parameter batch. It does not run simulations or replace
the experiment harness. Treatment, control, and lesion remain experiment arms. Proposed parameter vectors are a
separate candidate dimension that must be materialized into a derived preregistration before sealing. The sealed
harness then expands candidate by arm by seed and emits the only executable digest-bound job contracts.

## What a design contains

- A repository-relative `experiment.spec_path`. Its declared backends and partitions are authoritative.
- Continuous, discrete, and categorical parameters with finite bounds or choices.
- Hard constraints expressed as JSON predicates. Each constraint requires a biological source or rationale.
- Objectives covering physiology, robustness, compute cost, scaffold penalty, and at least one domain outcome:
  behavior for integrated tasks or mechanism for isolated mechanism calibration. A design may include both. Each
  objective has a direction, weight, meaningful normalization range, and optional success target. Existing designs
  with a behavior objective remain valid without changes.
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

python tools/experiment_controller.py research/specs/my-adaptive-design.json \
  --output research/queue/my-controller-plan.json \
  --owner-root research/queue \
  --materialized-spec-output research/specs/my-materialized-experiment.json
```

The outputs use exclusive-create semantics and carry self-digests where applicable. Re-running with the same paths
fails rather than overwriting evidence. The derived spec records the controller-plan and design digests, embeds the
exact candidate parameters and backend/partition mappings, adds candidate-specific output paths, and passes a
canonical effective-parameter document to the runner. Commit that spec, create the normal experiment seal, and use
`expand_experiment_jobs`; the executor does not add or mutate candidates after sealing. A successful adaptive
output must echo the exact candidate digest and effective parameters before its receipt can be accepted.

## Ingesting completed results

`tools/experiment_observation.py` closes the mechanical gap between the durable executor and the next adaptive
proposal. Its preregistered contract binds one adaptive design and one exact executor manifest, maps each objective
to a named experiment arm and direct scalar JSON path, declares the exact non-held-out seeds, and maps executor
backend/partition pairs to fidelity tiers. The compiler then:

- reauthenticates every successful executor receipt, output digest, and provenance sidecar;
- requires the complete treatment/control/lesion by seed evidence set for each candidate and fidelity;
- rejects held-out seeds, engineering-only output, changed artifacts, hard-constraint violations, and incomplete or
  widened evidence;
- applies only the preregistered arithmetic mean reducer; and
- writes a create-only, self-digested observation document with `scientific_verdict: null`.

```bash
python tools/experiment_observation.py \
  --contract research/specs/my-observation-contract.json \
  --executor-manifest research/queue/my-executor-manifest.json \
  --receipt research/queue/my-state/receipts/job-1.json \
  --receipt research/queue/my-state/receipts/job-2.json \
  --output research/findings/raw/my-observations.json \
  --repository-root "$PWD"
```

The resulting rows exactly match the adaptive design's observation shape; a separate top-level `evidence` ledger
binds each row to its receipts. `tools/adaptive_design_update.py` authenticates that binding, rejects duplicate
parameter/fidelity cells, writes a create-only next design version, and emits a self-digested lineage receipt. It
does not issue a scientific verdict or launch the next batch.

```bash
python tools/adaptive_design_update.py \
  --design research/specs/my-adaptive-design-v1.json \
  --observations research/findings/raw/my-observations.json \
  --output research/specs/my-adaptive-design-v2.json \
  --receipt-output research/findings/raw/my-adaptive-design-v2.update.json \
  --new-id my-adaptive-design-v2 \
  --repository-root "$PWD"
```

## Resumable supervision

`tools/adaptive_campaign_supervisor.py` advances exactly one lifecycle transition per invocation. It creates or
authorizes the next controller plan, candidate spec, seal, sealed expansion, executor state, exact job, observation
compilation, or design update. Every transition is a create-only, self-digested state-chain record. Repeated calls
with unchanged evidence return the same authorization rather than duplicating work.

```bash
python tools/adaptive_campaign_supervisor.py \
  --design research/specs/my-adaptive-design-v1.json \
  --campaign-dir research/queue/my-campaign-v1 \
  --repository-root "$PWD" \
  --observation-contract research/specs/my-observation-contract.json \
  --next-design-id my-adaptive-design-v2
```

The supervisor does not run commands itself, choose retry policy, reconcile remote queue results, open held-out
partitions, or issue scientific verdicts. It reauthenticates the deterministic proposal, sealed handoff,
materialization, executor manifest, and every receipt before authorizing the next transition. A worker loop may
execute the emitted `authorized_command` and call the supervisor again; failed jobs and unreconciled queued jobs
stop for explicit handling.

## Current limits

This first layer uses a scalarized multi-objective utility for acquisition; the Pareto report is diagnostic rather
than a full multi-objective optimizer. Its RBF uncertainty is useful for ranking bounded searches but is not a
calibrated probability. Failed and incomplete runs remain blocked evidence rather than being statistically modelled.
Objective ingestion supports direct scalar paths and an arithmetic mean over exact seeds; complex trace analysis
must first produce a sealed scalar artifact. Candidate materialization and design-version updates remain explicit,
so neither proposal nor observation ingestion can weaken `tools/experiment.py` seals or job contracts.
