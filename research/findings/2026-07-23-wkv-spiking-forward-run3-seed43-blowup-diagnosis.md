# run3 WKV spiking-forward: the seed-43 read blowup was an id()-reuse CACHE-ALIASING bug, NOT a substrate limit (2026-07-23)

**TL;DR.** The 2-seed cheap-first RF-on-bridge forward of run3's 83.17M chunked-WKV LM reported seed 42 PERFECT
(`ppl_ratio=1.000000`, `rf_max_read_err=3.68e-6`) but seed 43 BROKEN (`ppl_ratio=130`, `logit_fid=0.29`,
`rf_max_read_err=14.05`). The blowup is **not** a numerical/overflow limit of the RF graded read and **not**
intrinsic to seed 43's data. It is a **Python `id()`-reuse aliasing bug** in the runner's RF-read weight->CSR
cache that fires ONLY when a second seed runs after the first in one process. Root-caused, fixed at the
runner level (NO `sim/` edit), and re-verified.

## The symptom (the ORIGINAL 2-seed cheap-first, `run3_rf_2seed_cheapfirst.json`, the BEFORE)
| seed | ann_ppl | spk_ppl | ppl_ratio | logit_fid(spearman) | rf_max_read_err |
|------|---------|---------|-----------|---------------------|-----------------|
| 42   | 31.12   | 31.12   | **1.000000** | 0.99999999996    | **3.68e-6** (PERFECT) |
| 43   | 45.03   | 5862.0  | **130.17**   | 0.2923            | **14.05** (BROKEN) |

The two seeds differ ONLY in the rng-sampled val-window offset (`load_val_ids(seed, ...)`), so a per-seed
14.05 read error looked like a per-input numerical blowup. It is not.

## Diagnosis (what it is NOT, then what it IS)

**Not magnitude / overflow.** Instrumented the ANN forward on both seeds across all 8 runner windows: the
activation magnitudes are essentially IDENTICAL (`max|input|~=90` at the head, `max|output|~=30`; block
matvecs `<=~21`). A 14.05 abs error on a ~30-magnitude output is a ~50% relative error — impossible from
float32 precision. Confirmed by a direct RF-read magnitude sweep: the read error scales cleanly as
`~8e-8 * |ref|` (relative float32 precision) even at input scale 1e5 (`|ref|~1.2e5 -> read_err 1.05e-2`). To
reach 14.05 by precision alone you'd need `|ref|~1.4e8`; the model's outputs are bounded ~30. LayerNorm bounds
every matvec input regardless of the residual-stream scale, and the SSM time-mix is a convex combination — so
the architecture CANNOT blow up an intermediate. The read is a linear op with error `~1e-7*|ref|`.

**Not intrinsic to seed 43.** Re-ran the RF forward for seed 43 **alone** over ALL 8 runner windows: worst
read error **4.96e-6** (clean, == seed 42). The blowup does NOT reproduce when seed 43 runs by itself.

**It IS cross-seed `id()`-reuse cache aliasing.** The RF read installs each weight `W` as a complex CSR and
caches it in the MODULE-LEVEL `_WEIGHT_CSR_CACHE` (in `_genseq_loopstep3_full_genf_generate_derisk.py`) keyed
by `(id(Wf), n_neurons)` — where `Wf` is a float32 cast of `W`. The runner's `make_rf_graded_read` memoized
those float32 casts in a `_w32` dict that was **created FRESH inside each `run_full_one_seed` call**. So:
1. seed 42 runs first (empty cache) -> caches ~49 CSRs under `id(Wf_42)` keys -> **clean**.
2. `run_full_one_seed(42)` returns -> its `_w32` (and every `Wf_42` array) is GC'd.
3. seed 43 allocates ~49 NEW `Wf_43` arrays. CPython **reuses the freed `id()`s** (numpy-array wrappers are
   same-size PyObjects -> aggressive id reuse).
4. `_set_rf_weights(bridge, Wf_43)` computes `key=(id(Wf_43), n)`, finds a STALE seed-42 entry at that id, and
   installs **seed 42's CSR for a DIFFERENT weight matrix** -> a wrong matvec -> arbitrary large read error
   (14.05) -> corrupted logits -> ppl 5862.

This exactly explains why seed 42 is always clean (runs first, empty cache) and seed 43 (or any later seed)
blows up. A standalone deterministic reproduction of the aliasing (the runner's per-seed fresh-`_w32` pattern
with distinct weights) produced **2/3 false cache hits** — the new seed installed the old seed's CSR for a
different weight. The runner's own docstring already stated the intent ("the CSR cache hits across ALL
windows/seeds") — the per-seed fresh `_w32` silently violated it.

## The fix (runner-level, NO `sim/` edit; `sim_edit = NONE`)

Persist the float32-weight-cast cache (`w32`) and the RF bridge cache ACROSS seeds so every `Wf` stays alive
for the whole run -> `id(Wf)` is stable -> the CSR cache is valid AND genuinely hits across seeds (the
documented intent + a cross-seed perf win). Three additive, default-preserving edits in
`research/runners/_wkv_spiking_forward_derisk.py`:
- `make_rf_graded_read(..., w32=None)` — use the injected persistent dict when given (`None` = old behavior).
- `run_full_one_seed(..., bridges=None, w32=None)` — thread the shared caches through.
- `run_full` — create ONE `_shared_bridges` + `_shared_w32` and pass them to every seed.

No `sim/` edit. `make_rf_graded_read` is used only by this runner. (The module-level `_WEIGHT_CSR_CACHE`
keyed by `id()` is a latent landmine for ANY caller that pairs a per-seed transient weight cast with the
persistent cache; the transformer arc avoids it by building its bridges once over stable model-weight ids.)

## Before/after (the runner's OWN printed lines)

BEFORE (unfixed code, `--seeds 42 43 --n-windows 2`) -- a CONTROLLED reproduction (the runner's own lines):
```
[wkv-spk-fwd:FULL seed=42] ann_ppl=26.2484 spk_ppl=26.2484 ppl_ratio=1.000000 logit_fid=1.0000 rf_read_err=3.3e-06
[wkv-spk-fwd:FULL seed=43] ann_ppl=38.2955 spk_ppl=37123.10  ppl_ratio=969.385640 logit_fid=0.0372 rf_read_err=1.4e+01
[wkv-spk-fwd:FULL] mean ppl_ratio=485.192820 mean logit_fid=0.5186  VERDICT NEGATIVE
```
The DECISIVE control: seed 43 run **alone** at n=2 (no seed 42 before it) is CLEAN (worst read err 4.5e-6),
but seed-42-THEN-43 at the same n=2 blows up (14.0) -> the blowup is CROSS-SEED, not intrinsic to seed 43.

AFTER (fixed code), 2-seed cheap-first `--seeds 42 43 --n-windows 8` (the ORIGINAL config that produced the
14.05; the runner's own lines):
```
[wkv-spk-fwd:FULL seed=42] ann_ppl=31.1216 spk_ppl=31.1216 ppl_ratio=1.000000 logit_fid=1.0000 rf_read_err=3.7e-06
[wkv-spk-fwd:FULL seed=43] ann_ppl=45.0331 spk_ppl=45.0331 ppl_ratio=1.000000 logit_fid=1.0000 rf_read_err=5.0e-06
[wkv-spk-fwd:FULL] mean ppl_ratio=1.000000 mean logit_fid=1.0000  VERDICT GO
```
seed 43's `rf_read_err` went **14.05 -> 5.0e-06** and `ppl_ratio` **130.17 -> 1.000000** with the fix; BOTH
seeds now read at the RF-faithful ~5e-6 level == seed 42. Direct before/after at the identical config.

AFTER (fixed code), FULL 6-seed `--seeds 42 43 44 100 101 102 --n-windows 16` -> `run3_rf_6seed.json`:
```
per-seed ppl_ratio = 1.0000 all six; rf_max_read_err = 4.24 / 4.51 / 3.69 / 5.01 / 4.08 / 4.30 e-6
mean ppl_ratio = 1.000000   mean logit_fid(spearman) = 1.000000   VERDICT: GO
```
Every seed reads at the RF-faithful ~4-5e-6 level (incl. seed 43, the one the id-reuse bug had blown to 14.05).

## Verdict — GO (6-seed)
The "NEGATIVE" was a **harness bug (id()-reuse cache aliasing), not a substrate limit**. With the fix, the FULL
6-seed at production n_windows=16 is **GO: mean ppl_ratio 1.000000, mean logit_fid 1.000000, all six seeds
rf_read_err ~4-5e-6**. run3's **83.17M chunked-WKV LM runs as a faithful RF spiking-graded-read forward == the
ANN** — the project's largest TRAINED generative model validated as spiking-consolidatable (gap#1 "fully-spiking
on one brain" prerequisite for the trained LM). NO `sim/` edit. Follow-on: the 267M's spiking-forward once it has a
converged checkpoint (same runner, --ckpt run4_d2048).

## Knobs recorded
- ckpt `bridges/lmtrain/run3/ckpt/best.pt` (83.17M, d=1024, L=16, V=16000, chunk_c=16, step 902000).
- RF read: `RF_PERIOD=100000`, `RF_LAMBDA=0.0`, `nsteps=8`; block_size=256; n_logit_pos=16.
- backend `rf-bridge` (cupy, local 3090); RF complex state dtype = float32 (`cp_membrane_potential_v/u`).
- GO gate (runner): `mean ppl_ratio <= 1.05 AND mean logit_fid_spearman >= 0.99`.
