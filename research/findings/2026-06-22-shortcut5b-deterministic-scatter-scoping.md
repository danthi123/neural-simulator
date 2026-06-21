# Shortcut #5b — deterministic-scatter scoping for the secondary SNc-burst δ residual (2026-06-22)

**Task:** scope the cheapest genuine close for the ONE remaining #5b item — the secondary SNc-burst δ-read,
which holds 2/3 after R1 itself was CLOSED 3/3 (`research/findings/2026-06-22-shortcut5b-R1-deltabar-3of3-close.md`,
`3dd640e6`). The named fix was a "deterministic-scatter place→critic matvec (a `sim/` change)." This is a
read-only scope of that fix.

## VERDICT — this is a CLEAN, CHEAP determinism fix, NOT a deeper rewrite NOR a substrate limit. The deterministic-scatter SpMV is ALREADY BUILT, byte-identity-proven, and applied at every relevant matvec site; the only change needed is to KEEP it ON through the value-train and δ-read windows (the runner currently restores it to OFF after STEP-1 self-org). Rank-1 = a guarded, default-OFF runner/config scoping that holds `deterministic_transpose_matvec` ON past STEP-1.

The residual is **not** an R1 failure and **not** a point-neuron substrate boundary — it is the documented
**CuPy transpose-SpMV atomic-scatter non-determinism** leaking into the read-time critic firing rate (17–292 Hz
spread across seeds → the SNc GABA_B subtraction over-clamps on the hot seed). The deterministic remedy for that
exact op is the existing `cfg.deterministic_transpose_matvec` flag, which is currently scoped to STEP-1 only.

---

## 1. The exact non-deterministic op (file:function)

**Op:** the per-step **transpose sparse mat-vec `Wᵀ @ fired`** (source-fired → drive targets), inside
`SimulationBridge._run_one_simulation_step` in `sim/bridge.py`.

On CuPy, `csr.T` is a `csc_matrix`, so `csc @ v` routes to `cusparse.spmv(transa=True)`, whose **atomic scatter**
sums contributions in a thread-race order that varies run-to-run (FP summation-order variance). `CUBLAS_WORKSPACE_CONFIG`
(`--deterministic`) pins only cuBLAS dense GEMM — it does NOT pin this cuSPARSE atomic scatter. (Documented in
`sim/config.py:293-299` and `research/findings/2026-06-10-N9-placecode-reproducibility-*`.)

The place→critic value path passes through THREE of these transpose-SpMV sites every step (all restricted matvecs
re-cast from `cp_connections` against the same indices/indptr, masked to the routed synapses):

| site | `sim/bridge.py` line | what it drives (the place→critic path) | gated by `deterministic_transpose_matvec`? |
|---|---|---|---|
| **coincidence drive** `_co_matT @ prev_firing` | 6361–6364 | the place→value coincidence current (count or weighted) | **YES** (6362) |
| **graded-plateau drive** `_gp_matT @ prev_firing` | 6418–6421 | the place→value **weighted-plateau READ** (the stage-B value read-out — the over-driving toggle on the hot seed) | **YES** (6419) |
| **GABA_B drive** `_gb_matT @ prev_firing` | 6465–6471 | the value→SNc GABA_B/GIRK subtraction (the δ-gap read) | **YES** (6469) |

The two general conductance matvecs that build the rest of the critic's input are also gated:
- E/I conductance `_eff_cT @ fired_2col` — 6068–6071 (gated, 6069)
- graded-transmission `_WgT @ a_cont` — 6123–6133 (gated, 6124 / 6131)

So **every matvec that determines the critic's firing rate is already wired to the deterministic path.** The
non-determinism is not in a missing scatter op — it is that the deterministic path is **switched OFF** during the
windows where the critic rate is set.

**Why it leaks despite `--deterministic-selforg`:** the runner (`g11_bg_runner.py`) toggles
`cfg.deterministic_transpose_matvec = True` **only for the STEP-1 place-field self-org**, then **restores it to
the saved value (off)** immediately after, to "bound the per-step `.tocsr()` cost to STEP-1":

- `g11_bg_runner.py:5510-5512` — `_saved_detmv = getattr(...); if deterministic_selforg: cfg.deterministic_transpose_matvec = True`
- `g11_bg_runner.py:5548` — `bridge.core_config.deterministic_transpose_matvec = _saved_detmv   # restore (bound cost to STEP-1)`

The value-training (`_run_place_value_training`, `g11_bg_runner.py:5783`) and the subsequent δ-read run **after**
that restore — i.e. with the flag **OFF**. So the place FIELDS (the learned place→value weights) are reproducible
(same seed → same weights, the R-A anti-cheat), but the **read-time volley STRENGTH** (`Wᵀ @ prev_firing` summation
order) is **not** — that is the 17–292 Hz critic-rate spread the R1 doc root-caused. The hot seed's strong volley
drives the weighted-plateau read over the critic threshold to ~290 Hz → the GABA_B subtraction over-clamps (the
δ-gap inverts to 0.0).

**This precisely confirms the R1 doc's framing:** "the documented `--deterministic-selforg` already fixes the
FIELDS, but the volley STRENGTH still varies (the transpose-SpMV atomic scatter). A deterministic-scatter SpMV for
the place→critic matvec ... would normalize the read-time critic rate across seeds."

## 2. The deterministic options, ranked cheap-first

### Rank 1 (RECOMMENDED) — keep `deterministic_transpose_matvec` ON through the value-train + δ-read windows

The deterministic-scatter SpMV is the EXISTING, byte-identity-proven `cfg.deterministic_transpose_matvec`
(`.tocsr()`-materialize the transpose → a one-thread-per-output-row non-transpose SpMV, numerically allclose to the
csc product). It is already applied at every place→critic matvec site (table above). The only change is to **not
restore it to OFF after STEP-1** for the δ-read regime.

- **`sim/`-edit size:** ZERO new `sim/` op (the deterministic branch already exists at all five sites). The change
  is to the **scoping of when it is on** — the cleanest realization is a runner/config-level scope:
  - Option 1a (no `sim/` edit at all): the standalone probe (`_n5_grid_frontend_onbridge_probe.py`) sets
    `bridge.core_config.deterministic_transpose_matvec = True` and leaves it on for the value-train + read (it
    already owns the probe argv; it can set the cfg directly via the existing `_patched_init` hook). This makes
    the δ-read deterministic with **zero protected-file change** — the cheapest possible close to validate the
    mechanism, and the right first step.
  - Option 1b (small runner edit, the durable deployment form): add a `deterministic_read: bool = False` runner
    kwarg (or reuse `deterministic_selforg` with an extended scope) that holds the flag ON from STEP-1 through the
    value-train + read instead of restoring at 5548. ~3–6 lines in `g11_bg_runner.py`, default-OFF (the CLI default
    keeps the documented STEP-1-only behavior byte-identical).
  - Option 1c (optional belt-and-suspenders, a true `sim/`-level guarded flag): add a sibling
    `cfg.deterministic_transpose_matvec_persistent` (or simply document that callers may leave
    `deterministic_transpose_matvec=True` for the whole run) — no new op, just the existing branch left on. A
    one-line addition closes the lone ungated transpose-matvec at **`sim/bridge.py:6294`** (the NMDA-recurrent
    Wang-attractor drive `_nr_mat.T @ prev_firing`) so the deterministic mode is comprehensive; that site is NOT
    on the place→critic value path (it is the CA3/dlPFC attractor pathway, `enable_nmda_recurrent`), so it does not
    affect this δ, but gating it keeps "deterministic mode" honest. Flag this `sim/` one-liner as the only genuine
    protected edit, guarded by the existing flag, default-OFF byte-identical.
- **De-risk:** with the flag ON through the read, the place→critic volley strength becomes seed-stable (same seed →
  same `Wᵀ @ prev_firing`, the same property `--deterministic-selforg` already proves for STEP-1). The critic rate
  then lands in a single regime across seeds 42/43/44 → the SNc GABA_B subtraction is in-range on every seed → the
  SNc-burst δ-gap holds 3/3 under a SINGLE config (no per-seed g_gabab cap / graded-strength trade — the
  irreconcilable gentle-vs-hot tension in the R1 doc was a SYMPTOM of the seed-variable rate, which this removes).
  Re-run the R1 doc's all-arms battery on all three seeds with the read-time flag on and confirm gabab_gap True 3/3
  at one cap setting (0 or a single small cap).
- **Anti-cheat:** (i) **R1 stays selective** — V near/far must remain 4.35×/13.40×/5.04× (the deterministic matvec
  is numerically allclose, so the learned place→value selectivity is unchanged; only the read-time scatter order is
  pinned). (ii) **The full control battery must still collapse** — render (NEGATIVE) 1.0, scramble (LESION) → no
  learned V, no_learn (floor), graded-OFF lesion, and the in-arm g_gabab-mask-zeroed lesion must ALL still collapse
  the δ-gap to ~1.0 (the deterministic read must not manufacture a gap for a non-functional arm — it only stabilizes
  the rate of a genuinely-learned value). (iii) **Default-OFF byte-identity** — with the flag off the expression is
  the unchanged `.T @` (a pure extract-to-variable refactor), so every non-deterministic-mode run is byte-identical
  (the existing 6-seed nav byte-identity already proves this for the STEP-1 toggle). (iv) **The no-confab moat is
  untouched** — a nav-only probe; the place/critic/SNc state arrays (`cp_connections` / `cp_conductance_g_*`) are
  array-disjoint from the composer's complex `cp_rf_w_*` synapses, preserved by construction.

### Rank 2 — host-side deterministic accumulation for the small critic afferent

If the per-step `.tocsr()` cost were prohibitive at the full network scale, the critic's afferent is SMALL (a few
place→critic synapses), so its restricted matvec could be accumulated on the host (or via a CuPy segmented
sort-then-reduce) deterministically while the rest of the network keeps the fast path.

- **`sim/`-edit size:** larger — a new restricted-accumulation code path for the critic afferent only, with its own
  index bookkeeping. More surface area than rank 1 for the same outcome.
- **De-risk / anti-cheat:** same δ-3/3 target and same control battery as rank 1.
- **Why ranked below 1:** rank 1 already achieves a deterministic critic afferent with ZERO new op (the
  `.tocsr()`-materialized non-transpose SpMV IS a deterministic reduction). Rank 2 is only warranted if a
  per-step-cost profile shows rank 1 is too slow on the deployment — and the R1 doc already notes the runner toggles
  the deterministic path on for STEP-1 without a cost problem, and the critic read window is short. Rank 2 is the
  fallback, not the first move.

### Rank 3 — CuPy deterministic-mode env flags (`CUPY_*` / cuSPARSE determinism)

There is no general CuPy env flag that makes `cusparse.spmv(transa=True)`'s atomic scatter deterministic (unlike
PyTorch's `use_deterministic_algorithms`, CuPy exposes no such global for the transpose SpMV; the documented
mechanism is precisely why the project built `deterministic_transpose_matvec` rather than relying on an env flag).

- **Why ranked last:** speculative — no such flag is known to pin this specific op, and the project already
  characterized `CUBLAS_WORKSPACE_CONFIG` as insufficient for it. Not a reliable close.

## 3. Honest framing — clean cheap fix, not a deeper rewrite, not a substrate limit

**This is a determinism fix, NOT a substrate limit.** The R1 doc already established the V near/far value is
robustly SELECTIVE on every seed (4.35–13.40×) through the over-clamp — the place value IS computed correctly; only
the SNc **somatic readout** saturates on the seed whose critic over-fires. The over-firing is caused by the
read-time atomic-scatter non-determinism, a NUMERICAL reproducibility issue, not a point-neuron-substrate
expressivity wall. (Same family as `--deterministic-selforg` for the fields — this just extends the same fix to the
read window.)

**It is a CLEAN cheap close, not a deeper rewrite.** The deterministic-scatter SpMV is already implemented and
byte-identity-proven, and already wired into all three place→critic matvec sites. The CSR-materialized non-transpose
SpMV is deterministic by construction (order-fixed, one-thread-per-output-row). The only thing missing is to keep it
ON past STEP-1 — at minimum a zero-`sim/`-edit probe change (option 1a), and for deployment a ~3–6 line default-OFF
runner scope (option 1b). The lone ungated transpose-matvec at `sim/bridge.py:6294` (NMDA-recurrent, off the
critic path) is an optional one-line guarded addition for completeness, not required for this δ.

**Recommended rank-1 + its de-risk, in one line:** hold `cfg.deterministic_transpose_matvec` ON through the
value-train + δ-read (option 1a probe-only first, then option 1b runner scope), which pins the place→critic volley
strength → seed-stable critic rate → the SNc-burst δ-gap holds 3/3 under a single config; gated by the existing
default-OFF flag (byte-identical when off), R1 selectivity and the full control-battery collapse re-asserted, the
no-confab moat untouched by construction.

## Files referenced (read-only)
- `sim/bridge.py` — `_run_one_simulation_step`: the five gated transpose-SpMV sites (6068, 6123/6131, 6361, 6418,
  6465) + the lone ungated NMDA-recurrent matvec (6294).
- `sim/config.py:293-300` — the `deterministic_transpose_matvec` flag + its documentation.
- `research/runners/g11_bg_runner.py:5510-5512, 5548` — the STEP-1-only toggle + restore (`_saved_detmv`); 5783 —
  `_run_place_value_training` (runs after the restore, i.e. flag OFF).
- `research/runners/_n5_grid_frontend_onbridge_probe.py:128, 200-208` — the volley-non-determinism note + the probe
  argv (`--deterministic-selforg`, the STEP-1-only determinism).
- `research/findings/2026-06-22-shortcut5b-R1-deltabar-3of3-close.md` (`3dd640e6`) — R1 CLOSED 3/3; the secondary
  δ residual root-caused to the read-time volley non-determinism, naming this fix.
