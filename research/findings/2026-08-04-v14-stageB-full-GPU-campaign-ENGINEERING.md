---
status: qualified
type: finding
---

# V14 Stage B full GPU campaign: engineering screen complete

## Scope

This is engineering screening evidence, not a scientific physiology verdict.
The GPU backend is used to remove weak candidates cheaply. Any survivor must
be rerun with the authenticated NumPy runner before it can enter source-bound
scoring.

## Executed campaign

- Candidate design: exact seed-free 512-point, 24-dimensional Sobol manifest.
- Candidate manifest file SHA-256:
  `31bc6ecf263b6ce107a54e14e1dbccbacc0065572e1fac9f4084aae905ebb801`.
- Materialization: 512 independently compiled and verified packet/policy
  releases, grouped into 40 declarations of 64 candidates.
- Materialization wall time: 61.156 seconds on the local 20-thread host. <!--derived-->
- Execution: five arms for every candidate, 2,560 compact traces total.
- Arms: intact, Nap lesion, Cav2.2 lesion, SK lesion, and HCN lesion.
- Runtime storage is local and regenerable under
  `research/experiment-runtime/`; it is excluded from Git because it contains
  5,000+ compiled files and about 1.6 GB of traces.

The first 64-candidate Nap batch completed in 8.785 seconds. <!--derived--> The remaining
seven Nap batches were run concurrently. Harder 101-spike-or-timeout batches
were also fanned out as independent GPU processes; seven intact batches took
roughly two minutes together versus an estimated eight minutes sequentially.

## Strict triage result

The triage artifact has self SHA-256
`f78ada09406b967176602e2ac25c4a696da305cfeae5c5a8a13f60bcb6d25136`.
It authenticates every compact trace before recomputing the five resolved
engineering checks.

Artifact: `research/experiment-runtime/v14-stageB-sobol-512-results/triage.json`
(local regenerable engineering state; intentionally excluded from Git).

- `engineering_fail`: 409 candidates.
- `engineering_inconclusive`: 101 candidates.
- `engineering_pass`: 2 candidates.
- Selected for NumPy confirmation:
  `v14-stageB-sobol-284-b89bd2d48cda` and
  `v14-stageB-sobol-404-1676dc9e28ab`.

Per-check coverage:

| Check | Pass | Fail | Unavailable |
|---|---:|---:|---:|
| Nap lesion spike count equals zero | 347 | 165 | 0 |
| Cav2.2 lesion ISI CV exceeds intact | 17 | 214 | 281 |
| SK lesion ISI CV exceeds intact | 221 | 58 | 233 |
| HCN lesion remains active | 363 | 149 | 0 |
| HCN baseline rate changes at most 20% | 267 | 5 | 240 |

Timeouts are unavailable, not failures. No check compensates for another and
no ranking or scientific `GO` is produced.

## Infrastructure corrections made before accepting the screen

- Candidate manifests are regenerated from the filed template before
  aggregation; a shorter or hand-selected manifest is rejected.
- Aggregation recomputes the scorer output from authenticated NumPy runner
  observations instead of trusting supplied `passed` fields.
- Pool source attestation now includes JSON scorer fixtures.
- Multi-candidate runtime bindings authenticate one policy per candidate.
- GPU traces are published as deterministic digest-bound compact archives.

## Next action

Run both selected candidates across all five arms on authenticated NumPy CPU
workers, preferably split between the local host and freshly provisioned pool
nodes. Score only those recomputed observations. Then decide whether the 101
GPU-inconclusive candidates justify a bounded authority follow-up; do not
reinterpret timeout as physiology failure.
