---
type: finding
status: live
date: 2026-09-24
lane: Vikunja #203 (frozen synapses drift under use) -- flip-prep
mechanism: PRE-REGISTRATION for flipping `enforce_plastic_mask_in_hebbian` (sim/config.py:456) /
  `BRAIN_ENFORCE_PLASTIC_MASK` (sim/bridge.py `_hebbian_plastic_mask_enforced`, ~line 1250) from
  additive default-OFF (merge 509c7137b) to default-ON. No new mechanism; this document only fixes
  the GO/NO-GO criteria and the exact commands for the 6-seed evidence the flip decision needs.
seeds: []
verdict: PRE-REGISTRATION only. No GO/NO-GO claimed here. The seed-42 smoke (below) is a single-seed
  sanity check, explicitly NOT a generalization claim (feedback_6seed_validation). The 6-seed drift
  battery and the 6-seed load-bearing no-regression battery are prepared as exact commands but NOT
  queued -- the owner queues them after reading this document and the review report.
runner: research/findings/raw/_plastic_mask_flip_prep/prod_drift_probe.py
builds_on:
  - research/findings/raw/_read_isolation_audit_29/audit_29runners.json (the original 13.8->56.1
    comprehension-organ read-driven-drift measurement over 30 reads; the bug
    enforce_plastic_mask_in_hebbian fixes -- see sim/config.py:445-456 for the code-side citation)
  - merge 509c7137b (the additive default-OFF fix + tests/test_enforce_plastic_mask.py)
  - commit 511899f28 (research/plastic-mask-hebbian-variant-tests, cherry-picked onto this branch:
    per-Hebbian-variant SHA/freeze tests for rate-window/BCM/branchless)
---

# Plastic-mask enforcement flip: PRE-REGISTRATION (2026-09-24)

Branch `research/plastic-mask-flip-prep`, from `origin/main` at `54cf42ccb`. Vikunja #203 residual:
"a brain connection marked 'never change' can still quietly drift just from being used."

## What is already banked (not re-litigated here)

- `enforce_plastic_mask_in_hebbian` (sim/config.py:456) / `BRAIN_ENFORCE_PLASTIC_MASK`
  (`sim/bridge.py::_hebbian_plastic_mask_enforced`, ~line 1250): additive, default-OFF. When OFF,
  every masked branch is skipped -> byte-identical to the historical (buggy) permissive Hebbian
  path. `tests/test_enforce_plastic_mask.py` verifies BOTH directions on the numpy backend: OFF is
  byte-identical to the genuine pre-fix code (SHA compare against a throwaway pre-fix worktree),
  and ON eliminates the read-driven drift that IS present OFF (>10.0 -> <1e-3 on the scenario's own
  scale). Re-run this session: **2 passed** (`tests/test_enforce_plastic_mask.py`, CPU/numpy,
  `CUDA_VISIBLE_DEVICES= SIM_BACKEND=numpy`).
- Commit `511899f28` (branch `research/plastic-mask-hebbian-variant-tests`, never merged, never
  reviewed before now) adds three more SHA/freeze tests for the Hebbian variants the main test
  above does not drive: rate-window (non-BCM), BCM, and branchless
  (`enable_branchless_plasticity`). Cherry-picked onto this branch clean (commit `40a3b7d8e`).

## Review verdict on 511899f28

**SOUND in scope; the tests were written but never executed under cupy (self-disclosed in the
commit message: "the actual cupy freeze measurement has not been executed and is pending a
gpu_queue run").** A code read against current `main` confirms:

1. **Coverage is by write-site, not by literal config flag, and that is the correct unit.** The
   runtime Hebbian LTP/decay/clip block has exactly three delta-computation write sites gated by
   `_hebbian_plastic_mask_enforced`: the causal/symmetric `else` branch (`hebbian_symmetric` only
   changes which mask selects `pre_fired`, not the write site -- already covered by
   `tests/test_enforce_plastic_mask.py`'s default-path test), the rate-window `elif _rate_win:`
   branch (covers plain rate-window AND its BCM sub-branch AND its Oja/mean-subtract sub-variants,
   since the mask is applied to `delta_weights`/`_dw_b` AFTER those modifiers, at ONE call site
   each), and `_apply_branchless_hebbian` (a separate function). 511899f28 exercises all three
   distinct sites (rate_window, bcm, branchless); the shared decay/clip masking
   (bridge.py ~10094-10159) that runs after every branch is exercised by all of them, including
   the pre-existing default-path test. No gap found by inspection.
2. **Test design matches the existing STDP precedent correctly**: subprocess-isolated cupy (backend
   resolves once per process at `sim.bridge`'s first import), SHA/freeze assertion (`max_change ==
   0.0`, not "small"), and a false-freeze control (a companion all-plastic net must move by
   `>0.01`). This mirrors `tests/_plastic_mask_stdp_scenario.py` exactly, as claimed.
3. **Gap found by this review, not by 511899f28's author**: these tests, like the pre-existing
   `test_plastic_mask_freezes_fixed_synapses` (STDP, already on `main`), hard-require an actual
   CUDA device (`SIM_BACKEND=cupy` forced inside the subprocess env, then `cp.random.uniform`/
   memory-pool calls that touch the device). `pytest.importorskip("cupy")` only checks the package
   imports, not that a device is reachable, so under strict CPU-only
   (`CUDA_VISIBLE_DEVICES=`) these tests **error** (`CUDARuntimeError: cudaErrorNoDevice`), not
   skip. Reproduced this session against the PRE-EXISTING `test_plastic_mask_freezes_fixed_synapses`
   on `main` before touching 511899f28 at all, so this is a pre-existing gap in the STDP-test
   pattern the new commit inherited, not a regression it introduced. No fix applied (out of this
   branch's scope; flagged for a separate small fix to the `importorskip` idiom across both files).
4. **No fixes were needed to 511899f28's own code.** Cherry-picked as-is.

Actual cupy execution of the three new tests plus the two pre-existing STDP-scenario tests was
queued this session via `tools/gpu_queue.sh add` (shared singleton queue; 4 jobs already ahead at
queue time) rather than run inline, per this branch's GPU-lane rule. Result pending at the time of
this document; the owner should check `tools/gpu_queue.sh status` / the queued job's log before
treating 511899f28's cupy-side correctness as verified end-to-end (the code-read in point 1-2 above
stands regardless).

## Production drift smoke (seed 42, CPU/numpy)

`research/findings/raw/_plastic_mask_flip_prep/prod_drift_probe.py` builds the SAME production
`ChatBrain` the live webapp chat uses (tiny-demo, `composer_kind='rf'`, routed through
`webapp.brain_reply.reply_over_chat` -- the identical shared full-faculty pipeline `run_repl`
uses), drives 6 chat turns (five 3-content-token transitive assertions matching the tiny-demo's
own facts, plus one question), and reports per-named-pathway `max|dw|` on
`research.runners.comprehension_production_organ`'s `SpikingRoleCompetition` bridge -- the organ
the board's 13.8->56.1 read-driven-drift finding named
(`research/findings/raw/_read_isolation_audit_29/audit_29runners.json`, also cited in
`sim/config.py:445-456`). Its
`sel_{role}->sel_FS_{role}` / `sel_FS_{role}->sel_{other}` pathways are
`RegionPathway(plastic=False)` with NO named `plasticity_gate` (confirmed by reading
`research/runners/_phaseB_multicue_competition_spiking_derisk.py` lines ~296-309) -- exactly the
exposed shape this flag targets, and `comprehension_enabled()` is default-ON in live chat
(`webapp/server.py` `_get_comprehension_organ`), so this organ is genuinely load-bearing on
production turns, not a research-only fixture.

Run as two separate processes (arm=off, arm=on), each building the organ ONCE, gated by
`bash tools/mem_ok.sh` and run under `bash tools/memcap.sh` per this branch's memory rule (actual
measured RSS ~460MB, well under any plausible cap -- 3GB was used, not the suggested 12GB, because
this is a `tiny-demo`/`rf`-composer smoke, not a full developed-bundle build, and 12GB was refused
by `mem_ok.sh` against concurrent sessions' RAM use at run time).

**Numbers (both arms completed; see
`research/findings/raw/_plastic_mask_flip_prep/smoke_off_s42.json` /
`..._smoke_on_s42.json`).** Both arms report `frozen_max_abs_dw: 0.0` across EVERY
`declared_plastic: false` named pathway, including the four the board's finding names
(`sel_agent->sel_FS_agent`, `sel_patient->sel_FS_patient`, `sel_FS_agent->sel_patient`,
`sel_FS_patient->sel_agent`, each 288 synapses). The two arms' `frozen_max_abs_dw` (0.0 vs 0.0),
`plastic_max_abs_dw` (0.7016666531562805 vs 0.7016666531562805, identical to 16 significant
figures), and all 6 turn-by-turn replies (byte-identical text, `abstained: false` on every turn in
both arms) match exactly. `judged_any: true` in both (the comprehension gate did fire).

**This is NOT the clean confirmation it looks like, and should NOT be read as "the flag has zero
effect, so the flip is trivially safe."** Two things need to be separated:

1. **The `cue_*->sel_*` learned pathways' zero drift is EXPLAINED, not surprising.**
   `research/runners/_spiking_comprehension_monitor_derisk.py::_build_comp` calls
   `comp.freeze_all_cue_plasticity()` immediately after construction
   (`research/runners/_phaseB_multicue_competition_spiking_derisk.py::freeze_all_cue_plasticity`,
   which sets each `cue_{c}` NAMED plasticity gate to 0.0 via the pre-existing
   `set_plasticity_gate` mechanism). These pathways are `declared_plastic: true` with a real
   `named_gate` (e.g. `"cue_position"`) -- they were already frozen by the OLD, already-correct
   named-gate path this Vikunja fix does not touch. Their zero drift in both arms is expected and
   uninformative about the flag.
2. **The `sel_*->sel_FS_*` / `sel_FS_*->sel_*` pathways' zero drift is UNEXPLAINED and is the
   actual open question.** These are the exact pathways
   `research/findings/raw/_read_isolation_audit_29/audit_29runners.json` measured drifting from
   13.8 to 56.1 max-weight-delta starting from the FIRST read and continuing over 30 successive
   reads on a process-shared organ -- i.e. that audit's own baseline read was, like this probe's,
   taken after `ensure_built()` (build + the same calibration battery), so "calibration already
   saturated the weights before this probe's snapshot" is not an obvious explanation: the original
   audit saw substantial drift starting from THAT SAME post-calibration point. This smoke drove far
   fewer comprehension-gated reads (5 transitive-assertion turns, three of them repeats of the same
   two sentences, vs. the audit's 30 read_margin() calls) and got exactly 0.0 in the OFF arm, where
   the historical code should reproduce SOME nonzero drift as a positive control. Two explanations
   are open and neither is verified: (a) something about this organ's current build (weight
   values, `sel_recurrent_density`, drive strength) or this probe's specific turns no longer
   reproduces the original bug's firing pattern on these pathways, or (b) `corg.judge()`'s live-chat
   call path takes a different route through the substrate than the original audit's direct
   `read_margin()`/calibration-battery calls for a REPEATED identical sentence (memoization,
   caching, or an early-return this document has not traced).

**This blocks treating this smoke as evidence for the flip.** Before the 6-seed battery below is
queued, the missing step is a POSITIVE CONTROL: reproduce SOME nonzero `frozen_max_abs_dw` on the
`sel_*` pathways with the flag OFF (ideally by mirroring the original audit's own 30-read
`read_margin()` protocol directly against `corg.comp`, bypassing the full chat pipeline), to prove
the instrument can detect the bug it is meant to detect, before trusting either arm's zero as
meaningful. Without that, this smoke shows the pipeline runs end-to-end and produces a well-formed
artifact -- a real and non-trivial result on its own -- but says nothing yet about whether the flag
changes anything for this organ.

## Broader exposure survey (grep-only, NOT built or measured -- scope note for the flip's blast radius)

Comprehension is not the only default-ON production organ with this shape. A grep for
`plastic_internal *= *False` / `plastic *= *False` (no gate check, just presence) across the
`_derisk.py` modules the other `webapp/server.py`-wired production organs import-and-reuse found:
`_stageA_full_integration_derisk.py` (affect organ) -- 31 occurrences, `_curiosity_seek_learn_onbridge_derisk.py`
(curiosity) -- 7, `_spiking_expectation_rpe_derisk.py` (surprise) -- 6, `_affective_world_model_derisk.py`
(worldmodel) -- 5, `_second_order_metacog_monitor_derisk.py` (metacog) -- 2. None of these were built or
measured this session (out of scope for a seed-42 smoke of one organ); each occurrence still needs the
same check comprehension got (is there a co-located named `plasticity_gate`, or is it bare structural
`plastic=False`) before anyone can say the flip's effect is comprehension-only. The 6-seed
`load_bearing_fraction` battery in the GO criteria below exercises all of these faculties already, so a
regression in any of them under `BRAIN_ENFORCE_PLASTIC_MASK=1` would surface there even without a
per-organ drift probe -- but a battery PASS is not the same claim as "no other organ drifts today with
the flag off"; that broader baseline (does drift already exist elsewhere, flag off) was not measured and
is a gap this document does not close.

## GO / NO-GO criteria for the flip (6 seeds: 42, 43, 44, 100, 101, 102)

**GO** iff, on EVERY one of the 6 seeds, both hold:

1. **Zero frozen-synapse drift with the flag ON.** `frozen_max_abs_dw == 0.0` (exact, SHA-style --
   not "small") for every named non-plastic pathway the drift probe reports (at minimum
   `sel_agent->sel_FS_agent`, `sel_FS_agent->sel_patient`, `sel_patient->sel_FS_patient`,
   `sel_FS_patient->sel_agent`), AND the companion plastic-pathway control
   (`plastic_max_abs_dw`, the `cue_*->sel_*` learned edges) moves by more than 0 (proves the run
   fired and the flag did not silently freeze everything).
2. **No regression in the load-bearing battery with the flag ON.** The 6-seed
   `load_bearing_fraction` battery (exact command below) run with
   `BRAIN_ENFORCE_PLASTIC_MASK=1` must not reduce `robust_core_n` / raise the count of faculties
   whose `verdict` flips from `pass`/`regressed` to a worse category, relative to the existing
   production-default (flag-unset) 6-seed baseline aggregate already on record for this battery
   tag family. A faculty whose result is `UNRELIABLE` (per `load_bearing_fraction`'s own
   `null_control_clean` field) on either arm is inconclusive for that faculty, not a pass.

**NO-GO / hold** if either seed shows nonzero frozen-synapse drift ON, or the battery shows a
regression -- in which case the METHOD (this specific mask/gate wiring) is banked as insufficient
for some additional write site, not the CAPABILITY (closing the drift bug) abandoned, per this
repo's standing rule.

One seed (42, the smoke above) is explicitly NOT sufficient for either criterion
(`feedback_6seed_validation`); it exists only to catch a gross wiring error before spending 6-seed
compute.

## Exact commands (prepared, NOT launched)

### 6-seed drift-probe (pool jobs; prepared, NOT queued -- the owner queues these)

For `SEED` in `42 43 44 100 101 102`, TWO jobs per seed (off/on), pinned to this branch's head SHA
via a pool isolated-revision checkout (`~/derisk-pool/revisions/<HEAD_SHA>`, provisioned by
`tools/pool_provision.sh` if not already present -- not done by this document):

```
bash tools/pool_queue.sh add 'cd ~/derisk-pool/revisions/<HEAD_SHA> && SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python research/findings/raw/_plastic_mask_flip_prep/prod_drift_probe.py --seed=SEED --arm=off --out=research/findings/raw/_plastic_mask_flip_prep/pool_smoke_off_s<SEED>.json'
bash tools/pool_queue.sh add 'cd ~/derisk-pool/revisions/<HEAD_SHA> && SIM_BACKEND=numpy CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=. BRAIN_ENFORCE_PLASTIC_MASK=1 .venv/bin/python research/findings/raw/_plastic_mask_flip_prep/prod_drift_probe.py --seed=SEED --arm=on --out=research/findings/raw/_plastic_mask_flip_prep/pool_smoke_on_s<SEED>.json'
```

(`<HEAD_SHA>` = this branch's head at queue time, e.g. the SHA reported alongside this document in
the review report -- substitute literally, do not queue against a moving branch tip.)

### 6-seed load-bearing no-regression battery (job-generation command; prepared, NOT launched)

```
.venv/bin/python tools/lb_shard.py jobs --seeds 42 43 44 100 101 102 \
    --tag plasticmask0924 --root ~/derisk-pool/revisions/<HEAD_SHA> \
    --extra-env BRAIN_ENFORCE_PLASTIC_MASK=1
```

This prints one shell job line per (seed, faculty); pipe to a file and dispatch via the pool /
`tools/pool_queue.sh add` per line, then `tools/lb_shard.py aggregate --tag plasticmask0924` once
all shards land. `--root` again needs that revision provisioned on the pool first.
A guarded alternative exists (`tools/lb_shard_guarded_jobs.sh`) for a `--no-fixes`
production-default framing that statically rejects undeclared `BRAIN_*` overrides; not used here
because this run intentionally carries one declared override
(`BRAIN_ENFORCE_PLASTIC_MASK=1`) on top of the default fix set, which is the opposite of what that
guard checks for.

## What would block a flip

- **The seed-42 smoke's unexplained instrument gap (see above): the OFF arm did not reproduce ANY
  drift on the `sel_*->sel_FS_*` pathways the original audit named, so this probe has not yet been
  shown capable of detecting the bug it targets.** Run the positive-control check (mirror the
  original 30-read `read_margin()` protocol against `corg.comp` directly, flag OFF) BEFORE queuing
  the 6-seed battery; if it also reads 0.0, the wall-check applies (bank this specific probe/wiring
  reading as insufficient, do not conclude the capability is closed) rather than treating either
  arm's zero as a GO signal.
- Any nonzero `frozen_max_abs_dw` on any seed with the flag ON (the capability isn't actually
  closed by this wiring for some pathway shape not yet found -- e.g. a fourth Hebbian write site,
  or a non-Hebbian plasticity rule this flag doesn't touch: recall STDP/BDSP/BTSP already
  unconditionally respect the mask, per sim/config.py's own comment, so this should be Hebbian-only
  by construction, but the 6-seed run is what actually checks that rather than the comment).
- A load-bearing regression in the 6-seed battery.
- The queued cupy per-variant test run (511899f28's tests, gpu_queue) coming back FAILING --
  would mean the write-site code read in this document's review section missed something live.
- `mem_ok.sh`/`memcap.sh` availability on whatever host actually runs the 6-seed battery (both
  exist and were exercised at small scale this session; not re-verified at battery scale).
