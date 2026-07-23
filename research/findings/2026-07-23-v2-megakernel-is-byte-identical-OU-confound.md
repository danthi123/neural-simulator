# The v2 step-megakernel is BYTE-IDENTICAL to the reference — the "diverges at scale" belief was an OU-noise-reseed confound (2026-07-23)

## Headline
The general step-megakernel **v2** (`enable_step_megakernel_v2`, the in-kernel CSR-matvec fused path) is **byte-identical**
to the Python/cuSPARSE reference step — spike **raster bit-identical AND `max|Δv|=max|Δu|=0.00`** — at n=200/2000/5000,
**dispatch-asserted** (guard passed + RawKernel compiled) and **OU-noise-controlled**. It is ~**4.3× faster than
`read_only_fast_step`** in the launch-bound regime. It is now **DEFAULT-ON** (owner directive: performance improvements
on by default), alongside `fast_spike_reset` (byte-identical, Izhikevich-scoped) and `read_only_fast_step` (the ~3× sync
removal, committed earlier). Set `enable_step_megakernel_v2=False` for the exact-Python reference path.

## What was wrong before
The prior record (this session's summary + `docs/plans/2026-07-23-general-step-megakernel-design.md` + the config
comment) held that v2 **"diverges at scale (n=2000, raster diverges at step 18, 1 neuron) — double-accum matvec differs
from cuSPARSE, NOT bit-identical at scale."** That was **false**, and produced by a confounded instrument.

## Root cause — the OU-noise-reseed confound (a reusable methodology lesson, like the seed gotcha)
The bridge **reseeds the OU background-noise RNG at build time** (deterministic per `cfg.seed`). So:
- Building bridge A, building bridge B, **then** stepping both (`A.step(); B.step()`) makes A draw the OU sample from the
  RNG state left by **B's build**, and B draw the **next** sample — so A and B see **different noise**. On a chaotic
  spiking network a per-step noise difference of ~1 mV avalanches; a neuron near threshold flips a few steps in. Every
  ad-hoc "v2 vs reference" check this session used this build-both-then-step-both shape and mis-attributed the **noise**
  difference to v2's kernel. (Three confounded instruments in a row: post-reset-`v`; build-all-then-step; build-both-then-step.)
- The **clean** comparison builds+steps **each bridge in its own isolated scope**, so each reseeds OU to the **same**
  state → identical noise → the only remaining difference is the kernel. Result: **byte-identical** (Δv=0), both at
  n=2000 and n=5000, with `_step_megakernel_can_dispatch()==True` and the RawKernel compiled (so it genuinely ran — not
  a silent fall-through to Python).

**Decisive control:** two DEFAULT-path bridges with **bit-identical init substrates** (weights/thresholds/`v` all equal)
drift **1.59 mV after one step** when built-then-both-stepped — i.e. the "divergence" magnitude is present between the
reference and **itself**; ref-vs-v2 (1.46 mV) sat inside that same envelope. And `run_steps`-style build+step-each shows
DEFAULT run-to-run identical AND v2 run-to-run identical → both paths are individually deterministic; the confound was
purely the cross-bridge noise-draw order.

**Rule for future byte-identity comparisons:** never compare two bridges by building both then stepping both. Either
build+step each in isolation (so OU reseeds identically), or feed both the identical externally-supplied noise sequence.
This is the OU analogue of the `cfg.seed` substrate-seeding gotcha and belongs in the same silent-failure class.

## Evidence (GPU, cupy, seed 42, read-only Izhikevich inference regime)
- `research/findings/raw/gpubench/v2_clean_final.log`: n=2000 & n=5000 → `ref-vs-v2 raster-identical=True, max|Δv|=0.00, first-diff-step=-1`.
- `research/findings/raw/gpubench/v2_dispatch_verified.log`: n=2000 → `v2 CAN_DISPATCH=True, kernel COMPILED=True, raster-identical=True, max|Δv|=0.00e+00` (closes the "did v2 actually run?" rigor gap).
- `research/findings/raw/gpubench/v2_determinism_map.log`: n=2000 → DEFAULT run-to-run identical AND v2 run-to-run identical.
- Perf (n=1000, concurrent-with-training so absolute numbers are contended, but the ratio is stark):
  `baseline 89 → read_only 2602 (29×) → +v2 11186 (4.3× over read_only)`.

## Why v2 turns out byte-identical (mechanism note)
Empirically the in-kernel matvec reproduces cuSPARSE's float32 result exactly for the E/I-split CSR patterns the guard
admits (Izhikevich, read-only, no NMDA, no per-step CSR rebuild). The design doc's predicted "double-accum FMA residual"
did not materialize; the raster + `v/u` match to the bit. The existing GPU tests (`test_v2_onpath_equivalence_raster_identical_gpu`,
`test_v2_matches_v1_raster_gpu`) already asserted dispatch + raster-identity + `dv<1e-4`; the measured `dv` is 0.

## Changes shipped
- `sim/config.py`: `enable_step_megakernel_v2 = True`, `fast_spike_reset = True` (both default-on; comments corrected).
  `read_only_fast_step = True` was committed earlier (`5e80afd3`).
- `tests/test_step_megakernel.py::test_default_flags_and_fields_exist` + `tests/test_fast_spike_reset.py::test_fast_spike_reset_default_on`: default-pins updated (v1 stays off, v2 on).
- Validation: numpy byte-identity + determinism 18 pass; GPU megakernel suite 7 pass (incl. the v2 dispatch + byte-identity tests).

## Guard (blast radius bounded)
v2 activates ONLY on: GPU backend + IZHIKEVICH + read-only inference (no learning/STP/homeostasis/structural/neuromod/
recording) + `fast_spike_reset` + no NMDA + `effective_connections IS cp_connections` (no per-step CSR rebuild). Any
guard-failing config (all training runs, HH/AdEx, learning) falls through to the unchanged Python step. So the default
flip speeds read-only inference byte-identically and cannot touch learning runs.
