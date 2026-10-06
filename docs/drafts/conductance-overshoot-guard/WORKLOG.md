# Conductance-overshoot-guard WORKLOG (2026-09-29)

Flag-gated, default-OFF fix for the explicit-step conductance overshoot defect. Branch
`research/conductance-overshoot-guard`, worktree
`/home/dant123/Projects/sim/.claude/worktrees/conductance-overshoot-guard`, branched from `main`
at `3c2907b85`.

## The defect (recap)

`sim/kernels.py` computed the synaptic current from the PREVIOUS step's membrane potential,
`I_syn = g_e*(E_e-v) + g_i*(E_i-v)`, then advanced `v` by an explicit-Euler step. When the summed
conductance is large relative to `C/dt`, one step drives `v` far PAST the reversal potential
instead of toward it. Measured: a basal-ganglia build with four weight-40 GABA-A projections
converging on a 'commit' pool drove `v` to -377 mV within 60 steps; a -3000 pA clamp drove a
thalamic relay to -175 mV into rebound firing.

## The fix

New flag `cfg.enable_bounded_conductance_integration` (`sim/config.py`, default `False`). When
True, three conductance-current kernels in `sim/kernels.py` are routed through an
exponential-Euler bounded sibling that solves the local linear ODE for that conductance term
EXACTLY over `dt`, so one step can only move `v` toward its own reversal potential, never past
it:

```
I_syn_eff = (C/dt) * (E_eff - v) * (1 - exp(-g_tot*dt/C))
```

which reduces to the unbounded `g_tot*(E_eff-v)` at small `g_tot*dt/C` and saturates at
`(C/dt)*(E_eff-v)` (pins `v_new` exactly at `E_eff`) as `g_tot*dt/C -> inf`.

## Files and line regions changed

- `sim/config.py` (~L954-L979, right after `fast_spike_reset`): new field
  `enable_bounded_conductance_integration: bool = False`, documented in the file's own style
  (why/what/byte-identical-when-off/scope).
- `sim/kernels.py`:
  - After `fused_conductance_decay_and_current` (original ~L1109-1117): added
    `fused_bounded_conductance_decay_and_current` (the AMPA/GABA-A pair, `g_tot=g_e+g_i`) and the
    dispatcher `conductance_decay_and_current(..., bounded=False, C=None, dt=None)` that calls the
    ORIGINAL kernel unchanged when `bounded=False` (mirrors the existing
    `izhikevich2007_dynamics_update` / `backend_neutral_izh_arithmetic` dispatch idiom already in
    this file).
  - After `fused_gabab_decay_and_current` (original ~L1175-1183): added
    `fused_bounded_gabab_decay_and_current` (single-conductance case, own `E_gabab`) and dispatcher
    `gabab_decay_and_current`.
  - After `fused_nmda_update_and_current` (original ~L1186-1207): added
    `fused_bounded_nmda_update_and_current` (uses `g_eff*mg_block` as the effective conductance,
    same `E_nmda`) and dispatcher `nmda_update_and_current`. Covers BOTH call sites that reuse this
    kernel (global NMDA + slow-NMDA-recurrent).
  - `fused_readonly_izh_step` (the step-megakernel's fused inference kernel, ~L1119+) was left
    UNTOUCHED -- see "megakernel" below.
- `sim/bridge.py`:
  - Import list (~L88-112): added `conductance_decay_and_current`, `gabab_decay_and_current`,
    `nmda_update_and_current` alongside the existing `fused_*` imports (kept, since the dispatchers
    call them).
  - New method `_bounded_conductance_C(self, cfg)` (added just before `_run_one_simulation_step`):
    resolves `self.cp_izh_C` (per-neuron, IZHIKEVICH) or `cfg.adex_C` (scalar, ADEX); raises
    `RuntimeError` for any other `neuron_model_type` when the flag is on.
  - `_run_one_simulation_step` (~L8717): resolves `_bounded_ci` / `_bounded_ci_C` once per step
    right after `dt = cfg.dt_ms`.
  - Four call sites updated to route through the new dispatchers with `bounded=_bounded_ci,
    C=_bounded_ci_C, dt=dt`: the main `fused_conductance_decay_and_current` call (section 2, AMPA
    E/I), the global-NMDA call (section 2.3), the slow-NMDA-recurrent call (section 2.3a), and the
    GABA_B call (the GABA_B block). Each site's arithmetic is UNCHANGED when `_bounded_ci=False`
    (default) -- the dispatcher passes straight through to the original `fused_*` function with the
    original arguments.
  - `_step_megakernel_can_dispatch` (~L8098): added
    `if getattr(cfg, "enable_bounded_conductance_integration", False): return False` right after
    the `NeuronModel.IZHIKEVICH` guard, so `enable_step_megakernel`/`_v2` fall through to the
    (bounded) Python path instead of running the megakernel's still-unbounded
    `fused_readonly_izh_step`. Chose fall-through over a hard error: the guard-failure convention
    this function already uses everywhere else, and it still HONORS the flag correctly (just on the
    slower path) rather than refusing to run.

## Why each conductance term is bounded INDEPENDENTLY, not folded into one shared g_tot

AMPA/GABA-A, GABA_B and NMDA are computed at three DIFFERENT points in
`_run_one_simulation_step` (GABA_B and NMDA read `total_input_current_pA` that AMPA/GABA-A's
`I_syn` has already been added into). Folding them into one shared `g_tot`/`E_eff` would need the
step reordered to sum every conductance before any bound is applied -- out of scope for a
flag-gated, additive fix. Bounding each term independently against its OWN reversal guarantees no
SINGLE pathway can drive `v` past ITS OWN reversal in one step (the measured defect -- GABA-A
alone drove `v` to -377 mV), at the cost of not catching an AGGREGATE overshoot from several
simultaneously-large terms adding up across pathways. Documented in `sim/config.py`'s field
docstring and in each bounded kernel's own docstring.

## Tests (`tests/test_bounded_conductance_integration.py`), numpy backend, `cfg.seed=42`

All 4 pass on `SIM_BACKEND=numpy PYTHONPATH=$PWD .venv/bin/python -m pytest
tests/test_bounded_conductance_integration.py -v` (run from the worktree; venv from the primary
checkout, worktree has none):

```
tests/test_bounded_conductance_integration.py::test_byte_identical_when_off PASSED
tests/test_bounded_conductance_integration.py::test_overshoot_reproduced_flag_off PASSED
tests/test_bounded_conductance_integration.py::test_bounded_flag_on PASSED
tests/test_bounded_conductance_integration.py::test_small_conductance_equivalence PASSED
4 passed in 1.52s
```

- **(a) byte-identical off**: a 6-region network (commit pool + 4 GABA-A sources + GABA_B source +
  excitatory source, `enable_nmda=True` on commit, `enable_gabab=True`, OU on) run 60 steps with
  the flag explicitly `False` and left at its dataclass default both hash
  (`sha256` of the stacked per-step firing raster + final membrane array) to
  `c14523058fe9ba091f033be2c68a4600683fce94bee798d96e87c72419790462`. That hash was
  ALSO captured by running the identical builder script directly against `sim/` on the PRIMARY
  checkout (`/home/dant123/Projects/sim`, branch `main`, unmodified, commit `3c2907b85` --
  the exact tip this worktree branched from) -- i.e. the true pre-change code path, not a
  monkeypatch. All three (default, explicit-False, main) match exactly.
- **(b) overshoot reproduced (flag off)**: the literal reproducer -- 4 single-neuron GABA-A pools
  (weight 400 each, driven by a 3000 pA direct current so they fire reliably) converging on a
  1-neuron Izhikevich-2007 RS 'commit' pool with NO excitatory pathway. Flag off: **v_min =
  -247.69 mV** within 80 steps (dt=1ms), and the commit pool FIRES despite its excitatory drive
  being silent (the driving-force-sign-flip artifact the defect report describes).
- **(c) bounded (flag on)**: the SAME build, flag on: **v_min = -72.24 mV** (E_i=-75 mV default,
  so the floor is -76 mV -- held for every one of the 80 steps), and the commit pool NEVER fires.
  A same-test cross-check re-runs the flag-off build tracking the running min (not just the final
  `v`, which can mask the overshoot behind a post-spike reset to `c_reset`) and confirms it still
  overshoots past the floor -- so this is a real fix, not a network that happens to never overshoot.
- **(d) small-conductance equivalence**: weight 2.0 (vs 400 above) + a real excitatory drive + OU
  noise on an 8-neuron commit pool, 200 steps. Spike rasters: 16 spikes total (flag off),
  **match_fraction = 0.98125** between flag-off and flag-on (threshold asserted: >= 0.90).

Weight-40-exactly reproduction of the ORIGINAL -377 mV magnitude was not attempted (that build used
a much larger, unpublished basal-ganglia topology); weight 400 on this minimal 1-neuron-target
reproducer already clears the requested `< -150 mV` bar by a wide margin (-247.69 mV) while staying
a fast, deterministic unit test.

Sanity checks also run (not part of the new test file): `python -c "import ast; ...
ast.parse(...)"` on all three modified files; `tools/check_docs.py` (W1/W2, both 0 violations);
existing `tests/test_step_megakernel.py`, `tests/test_fast_spike_reset.py`,
`tests/test_adex_fast_spike_reset.py` (11 passed, 4 GPU-only skipped -- unaffected by the new
`enable_bounded_conductance_integration` guard added to `_step_megakernel_can_dispatch`); manual
check that `_bounded_conductance_C` raises `RuntimeError` for `NeuronModel.HODGKIN_HUXLEY`.

## GPU / cupy

NOT run. `bash tools/gpu_queue.sh status` at the time of this work: `state: running`, 5 queued, a
brain-loading job already active (`host-share-baselines` worktree) plus an untracked GPU-resident
process. Per the task instructions ("If the GPU queue is busy, skip cupy and say so"), cupy
variants of tests (a)-(c) were skipped. The kernel implementations use only `cp.where` / `cp.exp` /
`cp.clip` / elementwise arithmetic already used throughout `sim/kernels.py`'s other `@fuse()`
kernels (e.g. the `C_param == 0.0` guard in `fused_izhikevich2007_dynamics_update`, the `exp_arg`
clip in `fused_adex_dynamics_update`), so they are expected to fuse under `cp.fuse()` the same way,
but this is UNVERIFIED on an actual GPU.

## Open questions / explicitly out of scope

1. **Coincidence-detection / graded-dendritic-plateau currents** (`fused_coincidence_plateau`,
   `fused_graded_dendritic_plateau`, `sim/kernels.py` ~L1210+) are NMDA-like (voltage-gated,
   `(E_e - v)` driving force) but were NOT named in the task's explicit call-site list and were not
   given a bounded sibling. If a build enables both `enable_bounded_conductance_integration` and
   `enable_coincidence_detection`/graded plateaus, that current is still unbounded and can still
   contribute to an aggregate overshoot (see the "independent terms" caveat above).
2. **Hodgkin-Huxley and resonate-and-fire models** are explicitly out of scope
   (`_bounded_conductance_C` raises `RuntimeError` rather than silently no-op). HH's current is in
   `uA/cm^2` against `C_m` in `uF/cm^2`, not the `pA`/`pF` this bound's formula assumes; RF has no
   membrane-capacitance/synaptic-conductance-current term at all.
3. **Legacy Izhikevich kernel** (`fused_izhikevich_legacy_dynamics_update`, `sim/kernels.py` ~L61)
   appears to be DEAD CODE -- grep found no call site anywhere outside `sim/kernels.py` itself. Not
   touched; `NeuronModel` has no enum value that would select it (only `IZHIKEVICH`, i.e. the 2007
   model, `HODGKIN_HUXLEY`, `ADEX`, `RESONATE_AND_FIRE`).
4. **Aggregate overshoot across simultaneously-large independent terms** (AMPA+GABA_A+GABA_B+NMDA
   all large at once) is not caught by this fix even with the flag on -- each term is individually
   bounded against its own reversal, but summing several already-bounded currents into the same
   explicit-Euler membrane step can still, in principle, push `v` past where any single term alone
   would stop it. Not measured or reproduced here; flagged as a known limitation in both
   `sim/config.py`'s field docstring and the GABA_B/NMDA kernel docstrings.
5. **NMDA recurrent's own reproducer**: test (a)'s reference network exercises the GLOBAL NMDA call
   site (`enable_nmda=True` on the commit region) but not `enable_nmda_recurrent` /
   `cp_conductance_g_nmda_recurrent` (a separate, more involved wiring requiring `exc_receptor=
   "nmda_slow"`-routed synapses). Both call sites share the identical `nmda_update_and_current`
   dispatcher and `fused_bounded_nmda_update_and_current` kernel, so this is a coverage gap in the
   test's reach, not a code-path difference.

## Commit status

Staged (`git add sim/config.py sim/kernels.py sim/bridge.py
tests/test_bounded_conductance_integration.py
docs/drafts/conductance-overshoot-guard/WORKLOG.md`). `git commit` attempted once and BLOCKED by
the pre-commit hook -- output:

```
⛔ COMMIT BLOCKED — failure-class gate(s):
   [corpus-check-required] raw/g.json: ... ACCEPTED via fallback evidence ...   (informational, not a failure)
   consumer-hardware-reference: ... 22.66 GiB headroom (5.6% of budget used)     (informational, not a failure)
     ⛔ lane-starvation        1 problem(s)  -- 5 of 5 disjoint CPU lanes unserved (A-E)
     ⛔ compute-idle-persistent 1 problem(s) -- dedicated GPU/pool compute idle-with-ready-work
                                                for 5609 min straight (budget 45)
```

Both firing gates are unrelated to this change's correctness (repo-wide compute-utilization
gates, not a check on the diff) and both auto-expire in 6h per their own messages. Per the task
instructions ("If a commit is blocked ONLY by compute-idle-persistent or lane-starvation, leave
the work staged... and stop"), this is exactly that case: `git log -1` on the worktree still shows
`3c2907b85` (the branch point) as HEAD, nothing was committed, `--no-verify` was NOT used, and no
waiver file was written. `HEAD` unchanged; all five files remain staged (`git status --short`
shows `A`/`M` for all of them). No push, no merge.

## 2026-10-01 07:18 (local model, Hermes) — commit landed, cupy queued

- **numpy test re-run (this session):** `SIM_BACKEND=numpy .venv/bin/python -m pytest -q
  tests/test_bounded_conductance_integration.py` from the worktree -> **4 passed in 1.10s**
  (byte-identical-off, overshoot-reproduced-flag-off, bounded-flag-on, small-conductance-equivalence).
- **Commit landed:** the earlier block (lane-starvation + compute-idle-persistent, both
  repo-wide compute-utilization gates, auto-expiring) has cleared. `git commit` with the
  prescribed message now passes the gates (no `--no-verify`, no waiver).
  HEAD = `91608ce47` on `research/conductance-overshoot-guard` (5 files: sim/config.py,
  sim/kernels.py, sim/bridge.py, tests/test_bounded_conductance_integration.py,
  docs/drafts/conductance-overshoot-guard/WORKLOG.md).
- **cupy test queued (GPU busy):** enqueued via the sanctioned `tools/local_llm/gpu_add.sh`
  (the only GPU-queue path allowed to the local model; `gpu_queue.sh add` itself stays denied).
  Job: `cd <worktree> && SIM_BACKEND=cupy .venv/bin/python -m pytest -q
  tests/test_bounded_conductance_integration.py`. Queue depth 1 at enqueue. The 3090 was
  busy (~22/24 GB used, below the 3000 MiB free threshold) so the job is WAITING; it will run
  in the next yield. **rc: PENDING — read from `research/queue/gpu_queue.log` on W in the next
  yield and record it here.**
- **No push, no merge:** per the `push-credential` ruling (S has no push credential; the branch
  lives in this W checkout where Claude reviews and merges) and the wake's "never push". The
  branch `research/conductance-overshoot-guard @ 91608ce47` is ready for Claude to review/merge.
