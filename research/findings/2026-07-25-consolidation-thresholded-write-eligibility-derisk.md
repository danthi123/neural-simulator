# Consolidation write-side thresholded eligibility de-risk — NO-GO: a hard k-WTA gate on the BTSP presynaptic eligibility does NOT lift the ca1→slot own/other (stays flat ~1.0 vs the ~2.5 analytic target); the write-side threshold alone is insufficient → the RECALL-side (dendritic spike-count) read is load-bearing too (2026-07-25)

**Bounded de-risk of the "two-sided spike-count-threshold" surpass named in
`2026-07-25-consolidation-boundary-REATTRIBUTED-dense-CA1-code-not-the-write.md`** (its last two sections). That doc's
decisive result: the write's own/other is a bilinear self/cross overlap of the CA1 rate code; the DENSE any-spike/fire-count
code the write+recall actually read has ceiling ~1.45 (< the 2.5 gate); the sparse >25%-of-max BINARY core has ceiling ~8.0
but is NOT the operative set. The named surpass is a **two-sided** spike-count threshold gating BOTH the write eligibility
AND the recall read to the sustained-firing core (ceiling 8.0); a **write-only** threshold was analytically predicted to
land ~2.5 (core_rate/halo_rate ≈ 10/4, marginal at the gate). **This de-risk tested the write-only half. Result: it does
NOT reach ~2.5 — it stays flat at ~1.0.**

## The exact edit (additive, default-off, byte-identical verified)

Added a HARD-threshold k-WTA gate on the BTSP presynaptic eligibility, applied right after `etilde_bt` is formed (and
after the existing `btsp_elig_exponent` supralinear normalize), before it feeds the potentiation set:

- **`sim/config.py:359`** — new field `btsp_elig_hard_thresh: float = 0.0` (default 0.0 ⇒ byte-identical).
- **`sim/bridge.py:8071-8081`** — after the `elig_exp` block:
  ```python
  _elig_hthresh = float(getattr(cfg, "btsp_elig_hard_thresh", 0.0))
  if _elig_hthresh > 0.0 and etilde_bt.size > 0:
      _ehmax = etilde_bt.max()
      etilde_bt = cp.where(etilde_bt >= cp.float32(_elig_hthresh) * _ehmax,
                           etilde_bt, cp.float32(0.0))
      self._btsp_elig_survivor_n = (etilde_bt > cp.float32(1e-6)).sum()   # observability only (no behavior change)
      self._btsp_elig_total_n = int(etilde_bt.size)
  ```
  Only synapses whose PRESYNAPTIC eligibility is ≥ `thresh·peak` keep their eligibility; the rest are zeroed, so (in the
  default pure-potentiation path, `active_bt` requires `etilde_bt > 1e-6`) only the sustained-firing core writes.
- **`research/runners/nmda_compositional_consolidation.py:364`** — `cfg.btsp_elig_hard_thresh = getattr(args, "comp_btsp_elig_hard_thresh", 0.0)` threads it through `build_substrate`.
- **`research/runners/_consol_decoupled_plateau_probe.py`** — threaded `elig_hard_thresh` through `run_seed`/CLI (`--elig-hard-thresh`), added the survivor-count read (`elig_gate`) + a `conn_hash` of the post-write weights.

## Byte-identical-default verification (TESTED, not asserted)

GPU end-to-end bit-identity is **impossible to demonstrate** here: the consolidation write is inherently non-deterministic
on GPU (atomic scatter-adds). Two IDENTICAL-code runs at `thresh=0.0` differ (dw 6.926 vs 6.956; distinct `conn_hash`),
and even with `CUBLAS_WORKSPACE_CONFIG=:4096:8` set (dw 7.00 vs 6.95). So byte-identity was verified by isolating the edit
from the sim's stochasticity:

1. **Operation-level bit-exact test** (deterministic, on the actual cupy expression): at `thresh=0.0` the guard is False ⇒
   `etilde_bt` is a pure pass-through — input hash == output hash (`81db4bbf...`), `(out==in).all()`. At `thresh>0` the op
   is an exact k-WTA: survivors are precisely `{e ≥ thresh·max}`, non-survivors are exactly `0.0`, and the set is sparse
   (26/20/20 of 200 at thresh 0.25/0.4/0.6). PASS.
2. **Executable control-flow confirmation:** at `thresh=0.0` the probe reports `elig_gate=None` — the survivor attribute is
   never set, i.e. the guarded block never executed. The default path is untouched by construction.
3. **End-to-end corroboration within the noise floor:** OLD (git-stashed pre-edit `bridge.py`) vs NEW at `thresh=0.0`, same
   seed → identical `thr_hash` (`ee7fcf106bea`, substrate seeded identically); dw 6.888 (old) vs 6.926/6.933/6.956 (new
   repeats) — the old-vs-new gap (0.045) is within the identical-code run-to-run gap (0.030); rate-weighted own/other old
   `[1.012,0.974,1.013]` vs new `[1.009,0.974,1.012]` (≈ exact); dense ceiling old 1.365 vs new 1.329 (within noise).
4. **No regression:** `tests/test_onbridge_btsp.py` = 3 passed / 2 failed on the NEW bridge — and **identically** on the
   git-stashed OLD bridge (the 2 failures are pre-existing, not caused by this edit).

## The threshold-fires check (survivor fraction — must be sparse, not 0/not all)

`ELIG-GATE FIRES` (surviving synapses of 5,465,567 total network synapses, on the final write step), seed 42, `--commit-top-k 15`,
`--hippo-izh-type IZH2007_STRIATAL_MSN --hippo-izh-regions dg,ca3,ca1`:

| thresh | survivors | frac | dw |
|-------:|----------:|-----:|-----:|
| 0.25 | 5,465,567 | **1.000** | 7.00 |
| 0.40 | 5,464,775 | 0.9999 | 7.57 |
| 0.60 | 3,890,066 | 0.712 | 6.07 |
| 0.80 | 1,892,404 | 0.346 | 0.83 |
| 0.90 | 1,174,859 | 0.215 | 0.41 |

**Key diagnostic:** at thresh 0.25 **100.0%** of synapses survive — the eligibility MINIMUM is ≥ 25% of its max. The
seconds-long BTSP eligibility (`btsp_elig_tau_ms=1000`, a low-pass integrating across the whole multi-fact write) is
**compressed into a narrow (~4×) band** across all cells that ever fired, so a magnitude threshold cannot isolate a sharp
per-fact firing core — it only starts cutting appreciably above thresh 0.6, and then it removes the write (dw collapses).

## own/other results (seed 42 sweep)

RATE-WEIGHTED own/other (the CAPABILITY metric = the recall read; target ≥2.5, own-is-max 3/3) and the ceilings:

| thresh | rate-weighted own/other | mean | own-is-max | dense ceiling | sparse-core ceiling @>0.25max (n_active) |
|-------:|:------------------------|-----:|:----------:|:-------------:|:-----------------------------------------|
| 0.00 (baseline) | [1.009, 0.974, 1.012] | 0.999 | 1/3 | 1.329 | 9.40  (1, 11, 22) |
| 0.25 | [1.016, 0.969, 1.005] | 0.996 | 1/3 | 1.339 | 5.39  (1, 9, 27) |
| 0.40 | [1.013, 0.973, 1.011] | 0.999 | 1/3 | 1.312 | 7.19  (1, 9, 22) |
| 0.60 | [1.008, 0.979, 1.002] | 0.997 | 0/3 | 1.372 | 5.42  (1, 6, 26) |
| 0.80 | [0.957, 1.017, 1.019] | 0.998 | 1/3 | 1.397 | — |
| 0.90 | [0.955, 0.938, 0.998] | 0.964 | 0/3 | 1.403 | — |

own/other **never rises above ~1.0** at ANY threshold — it is flat (or degrades: 0.964 at thresh 0.9, 0/3 own-is-max).
Per the task protocol ("run seeds 43,44 only if a thresh lifts own/other above ~1.5 at seed 42") **no seed-42 threshold
reached 1.5, so the 3-seed confirmation is not triggered** (and the bounding mechanism below is seed-independent).

## Why it's flat (the mechanism, consistent with the record's bilinear proof)

The rate-weighted own/other is the RECALL read `Σ_k fire_i[k]·w[k→slot_j]`, bounded by the CODE-OVERLAP CEILING (~1.3–1.4,
the DENSE fire-count code). Two compounding reasons the write-side threshold can't lift it:
1. **The threshold barely gates** (frac 1.0 at 0.25) — the seconds-long BTSP eligibility low-pass compresses core-vs-halo
   firing-RATE differences into a near-uniform accumulated eligibility, so a magnitude cut can't isolate a per-fact core.
2. **Even when it gates hard** (thresh 0.8–0.9, ≤35% survive), own/other stays ~1.0 or worse — the surviving
   high-eligibility cells are the network's *overall*-strongest-firing cells (shared across facts), not each fact's
   DISTINCT core, so the write does not become fact-specific; and dw collapses (0.41), weakening the write without adding
   selectivity. This is the record's decisive point from the other side: no WRITE-side change can exceed the code overlap
   the RECALL reads.

## Sparse-core ceiling — still ≫ gate (the separable structure exists but is not operative)

The >25%-of-max BINARY-core ceiling remains **~5–9** (facts 1&2: 3.8–20.4; fact-0's size-1 core drags the mean), consistent
with the record's ~8.0. The near-disjoint sparse core is REAL and present — but neither the write-side threshold (this
de-risk) nor prior write-side levers (supralinear `elig_exp`, heterosynaptic depression) make it OPERATIVE, because both
the write eligibility and the recall read the dense fire-count code in which the core is drowned by the weak halo.

## VERDICT — NO-GO (write-side threshold alone insufficient)

vs the ~2.5 analytic prediction: **NO-GO.** A hard-thresholded WRITE eligibility does not lift own/other above the flat
~1.0 baseline at any threshold (never near ~2.5, never near the 1.5 promotion bar). This is a real finding that **sharpens
the boundary exactly as anticipated: the write-side threshold alone is insufficient — the RECALL-side read is load-bearing
too.** The analytic ~2.5 assumed the threshold would isolate each fact's core; empirically the seconds-long BTSP eligibility
is too temporally compressed for a magnitude k-WTA to do so, and even a perfect write-side core isolation is bounded by the
recall reading the dense code (the record's bilinear ceiling).

**What the full two-sided (recall-side) read would additionally require** (to reach the confirmed ceiling 8.0): a
per-cell nonlinear **spike-count-threshold READ** — the RECALL must also gate the `ca1→slot` activation to CA1 cells that
fired above a sustained-firing threshold in the read window (a dendritic branch thresholding on its inputs' spike-count),
so the dense weak halo does not contribute to the read. That is the deferred **dendritic per-cell nonlinear read** the
re-attribution names — NOT dendrites-for-decorrelation (the sparse core is already separable), but dendrites-for-the-
nonlinear-read of an existing sparse core. It must be applied to BOTH the write eligibility AND the recall; the write-only
half tested here is confirmed insufficient on its own.

## Shipped infra (reusable, additive, default-off)
`btsp_elig_hard_thresh` (config + bridge, byte-identical default, with a survivor-count observability read) ·
`comp_btsp_elig_hard_thresh` thread-through in `nmda_compositional_consolidation.build_substrate` ·
`_consol_decoupled_plateau_probe.py --elig-hard-thresh` (+ `elig_gate` survivor report + `conn_hash`). Raw:
`research/findings/raw/consol_opsweep_gpu/decoupled_vt-25_eht{0.25,0.4,0.6,0.8,0.9}_seed42.json`.

## Provenance
Bounded write-side de-risk, seed-42 indicator (the bounding mechanism — bilinear code-overlap ceiling + eligibility
compression — is seed-independent; 3-seed not triggered per protocol as no threshold reached 1.5). Op-level byte-identical
test + git-stash old-vs-new corroboration + no-regression check on `tests/test_onbridge_btsp.py`. NO protected-behavior
change (additive, default-off).
