---
type: preregistration
status: preregistered
date: 2026-09-24
lane: scaffold-retirement (VSA composer -> learned) + consumer-hardware-reference
mechanism: BRAIN_COMPOSER_KIND=slotbinder through the REAL production chat path (webapp.server._build_chat_brain ->
  developed_brain_io.load_developed_brain -> MultiTurnAgent -> BrainConversationalAgent -> SlotBinderComposer)
seeds: [7 (dev), 42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTRATION only. No cupy production-scale result exists yet.
---

# SlotBinder PRODUCTION composer gate: pre-registration (6-seed, real chat path, SIM_BACKEND=cupy)

## Why this gate, and why it differs from the L3 latency de-risk

`research/findings/2026-09-24-slotbinder-l3-cupy-latency-derisk-PREREG.md` (and its 6-seed GO,
`2026-09-24-slotbinder-l3-gpu-latency-GO-6seed.md`) measured `SlotBinderComposer(...)` constructed DIRECTLY, at 2
facts/seed, on the GPU: ~1 s/query, recall 1.0, moat/mismatch passing. It never called `load_developed_brain` or
`webapp.server._build_chat_brain`, so it never exercised `developed_brain_io._restore_facts`'s REAL production
fact-restore semantics (for `composer_kind='slotbinder'`, `_restore_facts` re-`store()`s -- re-teaches -- EVERY fact
in the loaded bundle, since `SlotBinderComposer` has no `.kb` composite fast-path). A default-OFF production route
already exists: `BRAIN_COMPOSER_KIND=slotbinder` routes `webapp/server.py`'s developed-brain loader through exactly
that path (`research/findings/2026-09-05-slotbinder-L3-wirein-derisk-NOGO-perstep-cost-dominates-latency.md`).

This gate runs the SAME production entry point `/api/brain-chat` uses (`_build_chat_brain`) end to end, at the
largest fact count the real corpus actually has (see FACT SCALE below), and requires the gate to be able to FAIL
(an ablation that zeroes the built SlotBinder bridge's own synapses must collapse recall -- a composer answering via
a host shortcut, not the taught synapses, would be unaffected).

Runner: `research/runners/_slotbinder_production_gate.py` (this commit). Read its module docstring for the full
per-arm protocol; this document states only the run command, the fact scale, and the criteria fixed BEFORE any
`SIM_BACKEND=cupy` result exists.

## FACT SCALE (measure-and-state, decided from the runner's own sizing note, not a new claim)

The real day_33 bundle carries K=2020/KF=1195 (404 facts, 788-word vocab) -- fixed, because
`load_developed_brain` always sizes `slotbinder_max_facts=len(facts)` from whatever bundle it is given, so there is
no partial-teach mode against the full bundle. `fanout=32` (this gate's default) pins the effective per-slot
candidate set at 32 regardless of vocab size, so nnz -- and VRAM -- scales with `K=5N` synapses alone
(~14,160 synapses/fact), trivial against a 24 GB consumer GPU at any N this 404-fact corpus can produce. **The
largest fact count that fits is therefore the WHOLE corpus: N=404** -- there is no smaller-than-full VRAM wall to
size against at this topology, so "largest that fits" and "the full real bundle" coincide. If the dev-seed run
below measures otherwise (VRAM pressure, or a wall-clock budget that makes N=404 impractical per seed), this
document is AMENDED with the measured ceiling before the 6-seed battery is queued, not silently substituted.

## Run (fixed)

```
SIM_BACKEND=cupy .venv/bin/python -m research.runners._slotbinder_production_gate \
    --seed <SEED> --n-facts 404 --fanout 32 --renderer stub \
    --out research/findings/raw/_slotbinder_production_gate/seed<SEED>.json
```
Dev seed 7 first, un-queued (small enough to run to completion and read by hand). Queued through
`bash tools/gpu_queue.sh add` (one brain-loading GPU process at a time), each invocation wrapped in
`until bash tools/mem_ok.sh 12 4; do sleep 30; done` per the lane brief, for the six evaluation seeds
(42 43 44 100 101 102) once the dev seed reads sane. `--check-flagoff` is passed for the dev seed only (cheap,
once), not for the 6-seed battery.

## Criteria (written before any cupy result)

A seed is **GO** iff every one of `verdict_criteria` in the runner's own output JSON is `true`:
- `recall_ge_fhrr`: SlotBinder's recall accuracy over the sampled facts is `>=` the FHRR (`rf`) reference arm's.
- `parity_1_0`: SlotBinder's answer matches the FHRR arm's answer on every sampled question (`parity_rate == 1.0`).
- `slotbinder_moat_pass` / `fhrr_moat_pass`: a never-taught (agent, action) pair abstains on both composers.
- `slotbinder_mismatch_pass` / `fhrr_mismatch_pass`: cross-fact agent/action pairing does not leak the wrong patient.
- `ablation_falsifies_intact_pass`: zeroing the built SlotBinder bridge's synapses (`cp_connections.data[:] = 0`)
  COLLAPSES its recall accuracy below the intact value -- the falsifiability requirement. A composer that is
  unaffected by this ablation is a NO-GO regardless of every other criterion (it would mean recall is not coming
  from the taught synapses).

The gate's headline verdict ("SlotBinder production composer gate GO N/6") requires all of the above `true` on
every one of the six evaluation seeds (42 43 44 100 101 102); seed 7 is dev-only per `feedback_6seed_validation`.
Per-turn latency (p50/p95 vs the FHRR composer on the same GPU) and the `--check-flagoff` resolved-class check are
reported alongside as measurements, not gating criteria (latency was already gated at L3; flag-off byte-identity is
established by `git diff main` over wiring files, not by this smoke -- see the runner docstring's point 6 and the
2026-09-24 FAILURE_LOG entry on `--check-flagoff`'s known limitation against the tiny synthetic dev-seed bundle).

## Known residuals going in (not gated by this document; carried from the runner's own docstring + FAILURE_LOG)

1. `SlotBinderComposer`'s recall composer uses exact-inverse FHRR bind/unbind arithmetic -- host arithmetic, a
   declared open item from the L1-L3 findings, unchanged by this gate.
2. `research/runners/brain_chat_tui.py`'s `ChatBrain._refresh_facts()` read `comp.kb` unconditionally and crashed
   for `composer_kind='slotbinder'` (no `.kb`, facts live in `.facts`) -- found smoke-testing this gate's script
   logic on CPU/numpy, CLOSED in this commit (`_refresh_facts` now falls back to `comp.facts` when `.kb` is
   absent; a no-op by inspection for every composer kind that already has `.kb`). See the FAILURE_LOG row dated
   2026-09-24.
3. `--check-flagoff` against the tiny synthetic (`n_facts=3`) dev-seed sample bundle raises `KeyError:
   'onebrain_composer'` building the production-default (`onebrain`) arm -- a bundle-structure gap in the ad hoc
   sample bundle, not in the SlotBinder path this gate is about. OPEN, not gating this document's criteria. See
   the FAILURE_LOG row dated 2026-09-24.

## AMENDMENT 1 (2026-09-25 03:10, orchestrator) -- measured wall-clock ceiling, before any 6-seed battery

The dev-seed run registered above (`--seed 7 --n-facts 404 --fanout 32 --check-flagoff`, cupy, GPU queue, worktree
`agent-ad44199e6b8cdcc9b`) started 2026-09-24 21:00:38 and printed `[seed 7] running arm=slotbinder ...` after a 266.9 s
staging build. It was still inside that FIRST arm (of three: slotbinder, the FHRR reference, flag-off) at 03:08, after 6 h
08 min of GPU time, with no further output: the runner prints nothing per query, so per-query latency and progress
were not observable. It was stopped (SIGTERM, rc 143) to free the shared GPU for three queued jobs. No artifact was written.

This is the case the sizing note above anticipates: N=404 is impractical per seed on this path, against the L3 de-risk's
~1 s/query for a directly constructed composer (so the production `load_developed_brain` path costs orders of magnitude more
per query, consistent with 2026-09-05-slotbinder-L3-wirein-derisk-NOGO-perstep-cost-dominates-latency). Before any 6-seed
battery: (1) add per-fact progress and per-query latency lines to the runner; (2) re-run seed 7 at small N (e.g. 8, 32, 128)
to measure the per-query cost on this path; (3) register the largest N whose three arms finish within a stated wall-clock
budget as a further amendment. The gate criteria themselves are unchanged.

## AMENDMENT 2 (2026-09-25, orchestrator) -- instrumentation landed + small-N runs QUEUED; item (3) is a DRAFT, values TBD

**Item (1), instrumentation -- DONE, committed in this section's own commit.**
`research/runners/_slotbinder_production_gate.py` now prints (all `flush=True`, matching the file's existing print
style):
- a per-fact TEACH progress line during the slotbinder arm's build (`[seed S] teach fact i/N ... fact_seconds=...
  elapsed_s=... avg_s_per_fact=... eta_s=...`), emitted by a monkeypatch on `SlotBinderComposer.store`
  (`_progress_instrumented_slotbinder_store`, scoped to a `with`-block around the `_build_chat_brain` call for that
  one arm, restoring the original method on exit) -- this is the exact call AMENDMENT 1 found silent for 6h08m
  (`_build_chat_brain` -> `load_developed_brain` -> `developed_brain_io._restore_facts` -> `comp.store()` once per
  fact, since `SlotBinderComposer` has no `.kb` fast path). A cheap JSON progress sidecar
  (`<out>.progress_<arm>.json`) is refreshed on the same cadence (one small `json.dump` per fact);
- a per-query latency line for every per-fact query, the moat probe, and the mismatch probe, in both arms
  (`[seed S] arm=<kind>: query i/N ... query_latency_s=...`, `... moat probe ... query_latency_s=...`, `...
  mismatch probe ... query_latency_s=...`);
- a build-start/build-done line per arm (`build starting (n_facts=N) ...` / `build done in Xs`).

All additive: the monkeypatch is entered only inside `run_arm`'s own build call and restores the original
`SlotBinderComposer.store` in a `finally`, so nothing about `store()`'s behavior, return value, or call signature
changes, and the 'rf'/flag-off arms (whose builds normally take `_restore_facts`'s direct-set-from-persisted-
composites fast path, never calling `.store()`) are unaffected -- the wrapper is simply never invoked for them.
`--out`, `--seed`, `--n-facts`, `--fanout` and every existing CLI flag are unchanged.

**Working hypothesis for WHY the dev run stalled (to be CONFIRMED or REFUTED by item (2)'s data, not yet a
measured claim), WITH a topology caveat this amendment states up front:**
`research/findings/2026-09-05-slotbinder-L3-wirein-derisk-NOGO-perstep-cost-dominates-latency.md` and
`2026-09-24-slotbinder-l3-gpu-latency-GO-6seed.md` (both read before this amendment was written) measured
per-fact TEACH cost (CPU, ~8.8-22.9 CPU-hours extrapolated for 404 facts) and per-query cost (GPU, ~1 s/query)
respectively at a topology HELD FIXED at the full production size (K=2020/KF=1195, `n_neurons=64,324`) while only
2 real facts were ever actually stored per seed -- those findings deliberately DECOUPLE network size from fact
count. **This runner does not do that**: `build_sample_bundle` constructs a genuinely SMALLER real sub-bundle of N
facts, so `load_developed_brain` sizes `slotbinder_max_facts=len(facts)=N` from THAT sub-bundle -- network size
(`K=5N`, vocab, `n_neurons`) grows WITH N here, unlike the L1-L3 methodology. The N=404 dev-seed run therefore
matches the L1-L3 findings' own full topology (N=404 IS the whole corpus, so K=2020 either way), but the new
N=8/32/128 runs will each build a SMALLER network than that, not the same K=2020 network fed fewer facts. Reading
across: `SlotBinderComposer.store()` runs 5 `_store_pair` calls per fact (agent/action/patient/polarity/attribute
slots), each running `teach_steps=40` simulation steps, so per-fact teach cost is expected to depend on BOTH the
number of `_store_pair` calls (linear in N regardless) AND the per-step cost at that N's own network size (which
the L1-L3 findings only measured at the one, full-scale, K=2020 point) -- so whether the N=8/32/128 trend is
linear in N, or grows faster because per-step cost itself rises with K, is an open empirical question this
amendment does NOT prejudge. **This is stated so item (3)'s eventual numbers are read against the RIGHT
methodology, not assumed to replicate the L1-L3 fixed-topology regime** -- the small-N runs below are what will
show which effect (call count, per-step-at-N cost, or both) actually dominates.

**Item (2), small-N runs -- QUEUED, not yet landed.** Seed 7 (dev seed only, per `feedback_6seed_validation` --
this is sizing, not a multi-seed accuracy claim), N in {8, 32, 128}, fanout=32, `SIM_BACKEND=cupy`, default
ablation and no `--check-flagoff` (that flag's own known `KeyError` residual at tiny synthetic N, see this
document's "Known residuals" #3, is orthogonal to timing and would only add noise to a sizing run), one GPU job
running the three N values in sequence via `tools/gpu_queue.sh`, each writing its own artifact under
`research/findings/raw/_slotbinder_production_gate/sizing/seed7_n<N>.json`. See the commit history for the exact
queued command line.

**Item (3), the largest-N-within-budget registration -- DRAFT, TO BE COMPLETED WHEN THE SMALL-N RUNS LAND.** This
paragraph is a placeholder marking what the completed amendment will contain, not yet a result:
- per N in {8, 32, 128}: `build_seconds` (the real teach cost for that arm), mean/max per-fact teach seconds (from
  the new progress lines/sidecar), mean per-query latency (slotbinder and FHRR arms), and whether the teach-cost
  trend across N is linear or worse-than-linear in N (a superlinear trend would mean the per-fact cost itself
  grows with corpus size already taught, a qualitatively different residual from the NOGO's own per-fact-constant
  extrapolation);
- a stated wall-clock budget (TBD once the N=8/32/128 numbers exist -- candidates to choose between at that time
  include a single-session bound and a queued-overnight bound; this document will say which and why, not assume
  one now) and the largest N whose three arms (slotbinder + FHRR + the ablation re-query) are projected to finish
  within it, extrapolated from the measured per-fact trend;
- an explicit statement of whether the measured GPU per-fact teach cost confirms, refutes, or refines the working
  hypothesis above (i.e., whether the per-step simulation cost the 2026-09-05 finding identified on CPU is in fact
  the dominant GPU cost too, or whether something else -- e.g. GPU kernel-launch overhead specific to this
  network's small size -- turns out to dominate instead).

**This section is committed BEFORE the N=8/32/128 runs it registers are dispatched**, per this project's
prereg-amendments-before-runs discipline; the runs are queued by the same commit, not run ahead of it.
