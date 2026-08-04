---
type: plan
status: active
date: 2026-08-04
---

# V14 Stage A Direct-Output Performance Amendment

## Why The Protocol Changed

The first performance attempt timed out before producing a timing result. The
simulation was launch-bound: the GPU spent most of its time receiving many
small operations rather than doing arithmetic. The owner therefore required a
kernel-launch optimization before another long run.

The replacement path reuses CuPy's already compiled SNr fusion graph and binds
its eight outputs directly to seven persistent state arrays plus one reusable
current buffer. This removes seven separate state-copy launches without
changing the equations or arithmetic graph. The established path remains the
default and is selected whenever the narrow dispatch guard is not satisfied.

## Evidence Required Before Timing

The optimized path must pass all of the following before the matrix runs:

1. Every HH and SNr state is byte-identical for 64 randomized GPU steps.
2. Persistent state-array addresses do not change.
3. The first uncached call and subsequent cached calls are both exact.
4. Independent bridges produce identical spike rasters and state over 64 steps.
5. Save-and-restore continuation remains exact.
6. Unsupported models, CPU execution, conductance noise, and extended HH
   currents refuse the optimized dispatch and retain the established path.

## Frozen Performance Matrix

The replacement matrix contains four cells, each repeated three times in a
precomputed randomized process order:

- `prechange-control-default`: detached source revision, feature absent.
- `candidate-default`: candidate source, optimization disabled.
- `candidate-active-unfused`: active SNr conductances on the established path.
- `candidate-active`: the same active conductances with direct outputs enabled.

The existing default compatibility threshold remains `1.02`, and the active
candidate/default threshold remains `1.25`. The new optimization must reduce
both median host time and median CUDA-event time to at most `0.85` of the
matching active-unfused cell. These values are fixed before replacement timing.
The projected runtime rises from 4,800 to 6,400 seconds because the matrix grew
from nine to twelve isolated worker processes.

## Scope And Risk

This is an execution optimization, not evidence for SNr physiology and not
permission to inspect sealed scientific seeds. It relies on a private CuPy
fusion-cache interface, isolated behind a fail-fast adapter. A CuPy cache-shape
change must raise an error rather than silently change arithmetic or fall back
inside a measured active cell.
