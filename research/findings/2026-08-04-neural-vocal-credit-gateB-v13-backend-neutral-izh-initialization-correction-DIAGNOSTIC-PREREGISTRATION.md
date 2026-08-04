---
type: preregistration
status: locked-not-executed
date: 2026-08-04
mechanism: gateB-v13-backend-neutral-izh-initialization-correction
spec: research/specs/v13_backend_neutral_izh_initialization_diagnostic.json
promotion_value: none
---

# V13 backend-neutral Izhikevich initialization correction diagnostic

## Why this diagnostic exists

The completed state-transplant aggregate is immutable evidence. Its NumPy- and
CuPy-origin bundles have identical topology, weights, traits, and fixed
Izhikevich parameters, but differ in `C`, `a`, `b`, `d`, and the stale initial
neuron type IDs. Replaying either fixed bundle also reveals a separate small
backend arithmetic divergence. This correction addresses only the population
initialization difference; it cannot earn a V13 scientific verdict or erase
the remaining arithmetic result.

Source aggregate:

- `research/findings/raw/v13_backend_state_transplant/aggregate.json`
- file SHA-256 `2f94905eebfddec69e4264ccae75f618bbffc720eb95126be9270d6a3aa5a8c6`
- embedded artifact SHA-256 `3fc135870cf42434078ea4129364f15be10100abb5f5ed07d89ae935d21d1ae4`
- sealed in commit `a3edf28abfc4a85d978142326274689a8660f28f`

The aggregate and its completed diagnostic artifacts must not be modified.

## Locked correction

Add `CoreSimConfig.backend_neutral_izh_initialization`, default `False`. When
enabled for an Izhikevich population, draw on the host with
`numpy.random.RandomState`, cast to the declared NumPy dtype, and transfer the
finished array once to the active backend. Keep the main population stream and
the pre-existing separately seeded heterogeneity stream distinct.

The complete initialization-time random-call inventory in scope is:

1. generic trait assignment, which feeds the initial neuron type IDs;
2. structured-profile trait shuffling;
3. per-neuron firing-threshold uniform draws;
4. every configured Gaussian or lognormal Izhikevich heterogeneity draw, not
   only the four arrays that differed in the completed aggregate;
5. initial 3D positions, because legacy NumPy shares its global stream with
   backend-native draws while CuPy does not.

Connectivity generation, runtime noise, simulation arithmetic, HH, and AdEx
are outside this narrowly named correction. The local, sparse, event-driven
architecture is unchanged. With the flag off, the existing statements and RNG
sequence remain the default path.

The opt-in path fails closed on an invalid flag type, unsupported neuron model,
invalid seed, unavailable parameter target, unknown or malformed distribution,
or non-finite output.

## Diagnostic seed

Use one paired diagnostic-only seed: `6556023`. It was derived before any run
from SHA-256 of:

`V13_BACKEND_NEUTRAL_INITIALIZATION_CORRECTION_DIAGNOSTIC_V1|aggregate_sha256=3fc135870cf42434078ea4129364f15be10100abb5f5ed07d89ae935d21d1ae4|role=paired_initialization`

The digest is
`b636ce6a34b79424f4ee62125ca9cd9a94336e01f66eacfdd50b9676ac3fd841`.
Using the locked rule `2000000 + (integer(first 12 hex digits, 16) mod
7000000)` gives `6556023`. A repository-wide exact-number check found no prior
use. It is neither formal nor held out. Tests must not execute it, and no V13
formal or held-out seed may be substituted.

## Future execution contract

A future create-only runner must initialize the same frozen 20-neuron
inhibitory-source plus 40-neuron GPi/SNr configuration once on NumPy and once
on CuPy, with the correction flag enabled and no simulation steps. It must bind
both cells to one source manifest and compare dtype, shape, and bytes for:

- traits, neuron type IDs, firing thresholds, and 3D positions;
- `C`, `k`, `vr`, `vt`, `vpeak`, `a`, `b`, `c_reset`, and `d_increment`;
- initial membrane potential and recovery state.

The diagnostic passes only if every listed array is byte-identical and the
source/config identities match. Any missing array, stale output path, source
drift, or partial backend result is an undefined diagnostic, not a pass.

No scientific experiment is authorized by this preregistration. The seed stays
sealed until a create-only runner, receipts, and source manifest are frozen.
