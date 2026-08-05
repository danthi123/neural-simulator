---
status: unavailable
type: finding
date: 2026-08-05
---

# V14 Stage B two-survivor NumPy confirmation: resolved checks pass, full contract unavailable

## What ran

The exact two engineering-pass candidates from the filed 512-point GPU screen
were rerun through all five Stage B arms with the authoritative NumPy backend.
Each candidate stayed on one machine for intact, Nap lesion, Cav2.2 lesion, SK
lesion, and HCN lesion execution.

| Candidate | Host | Candidate SHA-256 | Confirmation receipt SHA-256 |
|---|---|---|---|
| Sobol 284 | `pool40` | `bcac603bfe489d11f0e1999843de380e73447cf775ce41d01a61a06dc4d79f2b` | `3810e07fb71dccc539b36d3bb30691d60a7b45d9774104d9c66941db637e2b17` |
| Sobol 404 | `pool41` | `422d15fb00b3d9d0973ef2961faa003331d6f69b5a36e4faa4f3e95ec306cf41` | `12cb9e639e7edb2b00162c96fa102c069392587ca010e7714e541d3cc92383a6` |

Both ran from immutable Git archive revision
`5917295bc51281e5d9eb694a12ad99ca06ba1d23`, complete source manifest
`263d1578987c3ce6b4081004ae4025b14ad615dd24241a2a2a102b37dbaf23ae`,
Python 3.10, NumPy 2.2.6, SciPy 1.15.3, h5py 3.16.0, and PyYAML 6.0.3.
`pool42` was provisioned identically as the declared recovery host but was not
needed.

Runtime artifacts (local, authenticated, and intentionally excluded from Git):

- `research/experiment-runtime/v14-stageB-numpy-confirmation/results/v14-stageB-sobol-284-b89bd2d48cda/confirmation-receipt.json`
- `research/experiment-runtime/v14-stageB-numpy-confirmation/results/v14-stageB-sobol-404-1676dc9e28ab/confirmation-receipt.json`
- `research/experiment-runtime/v14-stageB-numpy-confirmation/job-plan.json`

## Result

Both candidates passed every check that is currently defined and computable:

- complete Nap lesion produced zero spikes;
- Cav2.2 lesion increased inter-spike-interval variability;
- SK lesion increased inter-spike-interval variability;
- HCN lesion preserved firing; and
- HCN lesion changed baseline rate by less than the 20% operational limit
  (4.83% for Sobol 284 and 3.65% for Sobol 404). <!--derived-->

The strict full result is nevertheless `UNAVAILABLE` for both candidates.
Neither is a scientific physiology GO. Four required evidence contracts remain
undefined or absent: a source-bound stable-baseline Nap voltage estimator, an
event-aligned medium-AHP window, paired HCN hyperpolarized current-step traces,
and a sealed 12-cell SK depolarization-block cohort.

## Authentication and independent replay

The confirmation controller bound candidate, job, host, source revision,
complete source file set, numerical environment, five-arm artifact set, and
remote receipt. Collection used a temporary directory followed by atomic local
promotion. Local verification authenticated every collected file and
recomputed each strict score from the bound scorer input; both recomputations
were exactly equal to the remote scores.

This replay exposed and fixed a transport defect in compact traces. The old
loader re-compressed a valid ZIP and demanded byte identity, although deflate
output can differ across zlib versions. The corrected loader still requires
the external archive digest, exact members, exact member digests, canonical
NumPy payloads, canonical JSON, member order, and fixed ZIP metadata, but does
not mistake compressor implementation bytes for scientific content.

## Decision and next action

Do not tune the two survivors or open reserved scientific partitions. The
highest-value next slice is to resolve the four unavailable biology contracts
from primary sources, then implement their exact stimulation, measurement, and
cohort protocols. The 101 GPU-timeout candidates do not justify authority CPU
follow-up until those protocols can distinguish a full Stage B pass from the
same partial pass already obtained here.
