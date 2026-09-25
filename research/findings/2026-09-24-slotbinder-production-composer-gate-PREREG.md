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

**Item (3), the largest-N-within-budget registration -- COMPLETE (2026-09-25).** The N=8/32/128 seed-7 runs
item (2) queued landed on the GPU 2026-09-25 04:28-05:25 (rc 0, all three). Artifacts are copied into this repo
(not left in the run's scratch worktree) at `research/findings/raw/_slotbinder_production_gate/sizing/seed7_n8.json`,
`research/findings/raw/_slotbinder_production_gate/sizing/seed7_n32.json`,
`research/findings/raw/_slotbinder_production_gate/sizing/seed7_n128.json` (each with its `.prov.json` sidecar;
the two smaller runs also carry a `.progress_slotbinder.json` sidecar), cited throughout this item.

**Why N=128 reads NOT-YET -- a gate criterion, not a missing arm and not a timeout.** Both arms (slotbinder, rf)
ran to completion and wrote a full result; six of the seven `verdict_criteria` in `seed7_n128.json` read `true`.
The ONE `false` is `parity_1_0`: `parity.parity_rate` is 0.9765625 <!--derived--> (125/128 <!--derived--> rows
matched the FHRR reference arm's answer). Reading the three mismatching rows in `seed7_n128.json:parity.rows`
(filtered to `match: false`): `(agent=atom, action=share)`, expected `electron` -- SlotBinder answered `electron`
(correct) while FHRR abstained (`None`); and `(agent=man, action=place)`, expected `penis`, sampled twice --
SlotBinder answered `penis` (correct) both times while FHRR answered `atom` (wrong, cross-fact leakage) both
times. **In all three mismatches SlotBinder's own answer was the correct one and the FHRR reference arm was
wrong** -- `recall_ge_fhrr` still reads `true` (SlotBinder 0.7421875 >= FHRR 0.71875, both from repeated
(agent,action) keys mapping to different patients in the real sampled corpus, an ambiguity in the facts
themselves and not a composer defect). But `parity_1_0` is written as an EXACT match to the reference arm's
own answer on every question (see the runner's `_parity`/`verdict_criteria` above), so a case where the
*reference* arm degrades at this larger vocab/fact scale still reads NOT-YET under this gate's fixed wording.
This is a genuine, measured criterion failure specific to N=128 (parity held 1.0 at both N=8 and N=32) -- not a
missing arm (both ran) and not a wall-clock timeout (the run finished; see the table below).

**Per-N wall-clock, all three arms (bundle-staging the sample bundle + the slotbinder arm [build, N per-fact
queries, moat probe, mismatch probe, the zeroed-synapse ablation re-query] + the rf reference arm), read
directly from the three artifacts' `bundle_staging_build_seconds` and per-arm `build_seconds`/`wall_clock_s`:**

| N | bundle_staging_s | slotbinder build_s | slotbinder wall_s | rf build_s | rf wall_s | TOTAL (3 arms) | parity_rate | verdict |
|---|---|---|---|---|---|---|---|---|
| 8 | 50.934 | 2.5693 | 8.668 | 0.002573 | 3.6573 | 63.260 s <!--derived--> (1.05 min <!--derived-->) | 1.0 | GO |
| 32 | 94.054 | 20.618 | 111.135 | 0.006879 | 12.020 | 217.210 s <!--derived--> (3.62 min <!--derived-->) | 1.0 | GO |
| 128 | 238.833 | 99.528 | 2645.566 | 0.087416 | 82.554 | 2966.953 s <!--derived--> (49.45 min <!--derived-->) | 0.9765625 | NOT-YET |

**The per-query cost curve is accelerating, not constant or linear.** Mean per-fact query latency
(`latency_slotbinder_per_fact_query_s.mean` in each artifact) is 0.292 / 1.109 / 6.037 s at N=8/32/128; its
p95 (`...p95`) is 0.456 / 1.8784 / 14.201 s. Fitting TOTAL wall-clock (the table above) as a power law over all
three points gives `total(N) ~= 2.804 * N^1.388` seconds <!--derived-->, but the LOCAL slope between the two
LARGEST measured points (32->128) is steeper still: quadrupling N multiplied TOTAL wall-clock by ~13.66x
<!--derived--> (local exponent ~=1.886 <!--derived-->), versus only ~3.43x <!--derived--> (local exponent
~=0.890 <!--derived-->, i.e. SUB-linear) when quadrupling N from 8 to 32. This is convex, accelerating growth in
log-log space, not a single power law -- consistent with fixed per-process overhead (CUDA context, bundle load,
kernel warm-up) dominating at small N and shrinking as a fraction of the total as N grows, while a cost that
itself scales with the built network's own size (`K=5N`) takes over at larger N. **The global 3-point fit
therefore UNDERSTATES cost near and past N=128 and must not be used to extrapolate beyond the measured range.**

**This confirms, in a refined form, AMENDMENT 2's working hypothesis.** Per-fact TEACH cost (`build_seconds`)
does NOT hold the trend implied by "linear in `_store_pair` call count alone": its local exponent falls from
~=1.502 <!--derived--> (8->32) to ~=1.136 <!--derived--> (32->128) -- decelerating TOWARD linear as N grows, the
opposite of what a network-size-dependent per-step cost acting on the TEACH path alone would predict. The
RECALL side is where the network-size effect shows up: mean per-query latency's local exponent RISES from
~=0.963 <!--derived--> (8->32, near-linear) to ~=1.222 <!--derived--> (32->128, clearly super-linear), and its
tail is worse -- p95's local exponent rises from ~=1.021 <!--derived--> to ~=1.459 <!--derived-->. A fixed
per-call/kernel-launch overhead would keep per-query latency roughly FLAT in N; it instead accelerates, so the
per-step simulation cost at the built network's own size is the dominant driver of the recall-side growth,
exactly as the 2026-09-05 CPU finding's per-step-cost mechanism predicts -- refined here to show it is the
QUERY/recall path, not the teach path, where that cost currently bites hardest on this GPU path.

One further, UNEXPLAINED cost asymmetry is noted here (not gating this document, flagged for a future
amendment): the post-ablation re-query loop costs MORE than the intact per-fact query loop, increasingly so --
computed as slotbinder `wall_clock_s` minus `build_seconds` minus the summed `per_fact` query latencies minus
the moat/mismatch probe latencies, the implied ablation-loop time is ~3.077 <!--derived-->/ ~51.642
<!--derived-->/ ~1757.377 <!--derived--> seconds at N=8/32/128 -- at N=128 this is ~66.4% <!--derived--> of the
slotbinder arm's OWN wall-clock (larger than build + intact queries combined) and ~59.2% <!--derived--> of the
seed's TOTAL (3-arm) wall-clock. Why zeroed synapses would make queries slower, not faster or unchanged, is an
open question this amendment does not resolve.

**Wall-clock budget and the chosen N.** Budget: <=10 minutes (600 s) per seed for the three arms combined
<!--derived-->, chosen so the registered 6-seed battery finishes inside a single `tools/gpu_queue.sh` session
rather than requiring an overnight reservation -- this is a SIZING decision (pick the largest N that still
exercises a genuine multi-fact real sub-corpus with comfortable margin), not a push toward N=404 (already
measured impractical by AMENDMENT 1's 6h08m stall). Against that budget: N=8 (63.260 s <!--derived-->) and N=32
(217.210 s <!--derived-->) both fit comfortably; N=128 (2966.953 s = 49.45 min <!--derived-->) does not.
Independently of the budget, N=128 is ALSO excluded on gate-criteria grounds (`parity_1_0: false`, above) --
registering a 6-seed battery at an N that already fails one of the gate's own fixed criteria on the dev seed
would not be a genuine test of the battery, it would be re-running a known failure mode six more times.

**Chosen N = 32.** It is the largest of the three measured sizes that is BOTH within the stated budget AND a
full dev-seed GO (all seven `verdict_criteria` true, `parity_rate` 1.0, `ablation_falsifies_intact_pass` true).
Projected 6-seed wall-clock at N=32, sequential through `tools/gpu_queue.sh`: 6 x 217.210 s <!--derived--> ~=
1303.26 s <!--derived--> (~21.72 min <!--derived-->) of GPU time total, plus each seed's own bundle-sampling
variance (a different 32-fact sample per seed) -- comfortably inside a single queue session.

**This section is committed BEFORE the 6-seed battery it registers is queued**, per this project's
prereg-amendments-before-runs discipline; the queued commands are added by `tools/gpu_queue.sh add` right after
this commit, not run ahead of it. (The N=8/32/128 sizing runs item (2) registered were themselves queued and
landed before this completion -- see the artifact paths and timestamps above.)

## AMENDMENT 3 (2026-09-25, review fix round) -- the N=404 dense-step model + 3090 time projection (item 4), and
the prepared sparse_activity_step smoke/6-seed commands (item 5)

**Why this section exists.** research/FAILURE_LOG.md's 2026-09-25 row and this runner's own comments/help text
(`_slotbinder_production_gate.py` around the `run_arm` docstring and the `--sparse-step` CLI flag) already CITED
"AMENDMENT 3" before this section was written -- an independent review of the event-driven-step branch caught
the gap (the citation pointed at nothing, and at the WRONG document: FAILURE_LOG's parenthetical named
`research/findings/2026-09-25-slotbinder-event-driven-step-bit-identical-numpy.md`, a different finding, not
this prereg). This section is what those citations now resolve to; the stale parenthetical is corrected in the
same commit.

**Item (4): the N=404 dense-step count and its 3090 time projection.**
`research/runners/_slotbinder_gate_step_model.py` counts the slotbinder arm's simulation steps EXACTLY from the
fixed protocol (5 `_store_pair` calls/fact x `teach_steps`=40 for teach; `SlotBinderComposer._match`'s read
count x `retr_steps`=40 for each intact query, the moat probe and the mismatch probe; N reads x 40 for the
post-ablation re-query, since every ablated query scans the whole corpus) and cross-checks it against the
measured N=8/32/128 cupy sizing artifacts (`research/findings/raw/_slotbinder_production_gate/sizing/seed7_n{8,
32,128}.json`) by dividing each artifact's measured phase seconds by the model's step count for that phase --
agreement within the teach/query per-step-cost difference across all three sizes confirms the model. Run and
committed this section (`research/findings/raw/_slotbinder_sparse_step/step_model_seed7.json`,
`.prov.json` sidecar):

```
CUDA_VISIBLE_DEVICES="" SIM_BACKEND=numpy .venv/bin/python -m research.runners._slotbinder_gate_step_model
```

At **N=404** (the full corpus, this document's fixed FACT SCALE) the model gives **teach 80,800 / intact-query
1,877,000 / post-ablation-re-query 6,560,960 / total 8,518,760** simulation steps (all read from
`step_model_seed7.json:per_n.404`, no decimals to round) -- teach is under 1% of the arm's own steps, matching
FAILURE_LOG's reading of the AMENDMENT-1 stall as "most likely in the query loops", not the teach. Applying the
mean per-step cost measured on cupy at this exact topology (K=2020, the L3 latency de-risk's 6-seed GO artifacts
`research/findings/raw/_slotbinder_l3_latency_derisk_cupy/latency_f32_s{42,43,44,100,101,102}.json`, each
seed's moat-probe latency over its 80 steps): 8.81 / 9.2 / 7.85 / 8.31 / 8.51 / 8.86 ms/step, mean ~8.59 ms/step
<!--derived-->, `step_model_seed7.json:l3_ms_per_step_mean`. Projected slotbinder-arm-alone wall-clock at N=404
on the CURRENT (dense, unchanged) step, on a 3090: **~20.3 h** <!--derived--> (`total_steps x l3_ms_per_step_mean`,
read directly as `step_model_seed7.json:n404_dense_cupy_projection_h`), of which the teach
phase alone projects to ~11.6 min <!--derived--> (`n404_dense_cupy_teach_projection_min`). AMENDMENT 1's kill
(6 h 08 min = 368 min, still inside the slotbinder arm, no artifact written) is therefore consistent with the
run having reached roughly 30% of its own projected total when stopped -- a live-but-slow run inside the query
phase, not evidence of a hang. This projection is a MEASURE-AND-STATE of the CURRENT dense step; it does not by
itself make N=404 practical, and does not change this document's chosen N=32 for the 6-seed battery (AMENDMENT
2) or its GO criteria.

**Item (5): the event-driven step exists; the smoke and 6-seed commands with it are PREPARED, not yet run.**
Independently of this gate, `cfg.sparse_activity_step` (default off; sim/config.py, sim/bridge.py) was verified
BIT-IDENTICAL to the unchanged dense step on numpy for the SlotBinder's own bridge topology
(`tests/test_slotbinder_sparse_step_equivalence.py`;
`research/findings/2026-09-25-slotbinder-event-driven-step-bit-identical-numpy.md`), and wired into this runner
as `--sparse-step` (`run_arm(..., sparse_step=True)` sets `BRAIN_SLOTBINDER_SPARSE_STEP=1` for the slotbinder
arm only). **No cupy equivalence or timing run for this flag exists yet** -- that finding's own open issue,
unchanged by this amendment. Two commands are PREPARED here (their exact invocations, not their results) so
the next GPU session can run them without re-deriving the protocol; neither has been run against cupy yet, and
neither is a criterion this document gates on:

```
# (a) extend the bit-identity check past numpy, at a size cheap enough to run inline before trusting the flag
# on GPU (SIM_BACKEND is read at import time by both the equivalence runner and the production gate -- set it
# in the environment, do not rely on the module's numpy default):
N=8
SIM_BACKEND=cupy .venv/bin/python -m research.runners._slotbinder_sparse_step_equivalence \
    --n-facts $N --out research/findings/raw/_slotbinder_sparse_step/equivalence_seed7_n${N}_cupy.json

# (b) a single-seed GPU smoke of the production-gate slotbinder arm WITH the flag, at the already-chosen N=32
# (AMENDMENT 2), dev seed 7, before spending any 6-seed budget on it:
N=32
SIM_BACKEND=cupy .venv/bin/python -m research.runners._slotbinder_production_gate \
    --seed 7 --n-facts $N --fanout 32 --renderer stub --sparse-step \
    --out research/findings/raw/_slotbinder_production_gate/sparse_step/seed7_n${N}.json

# (c) the 6-seed battery this document already registers (AMENDMENT 2), WITH --sparse-step, held pending (a)
# and (b) both passing -- this is the first proposed cupy use of the flag, so it is not queued by this
# amendment, only written down for whoever runs it next:
for SEED in 42 43 44 100 101 102; do
  until bash tools/mem_ok.sh 12 4; do sleep 30; done
  bash tools/gpu_queue.sh add "SIM_BACKEND=cupy .venv/bin/python -m research.runners._slotbinder_production_gate \
      --seed $SEED --n-facts 32 --fanout 32 --renderer stub --sparse-step \
      --out research/findings/raw/_slotbinder_production_gate/sparse_step/seed${SEED}_n32.json"
done
```

Command (c) is deliberately NOT queued by this commit (same discipline as AMENDMENT 2's own battery: registered
here, queued separately, not run ahead of the prereg) -- it additionally waits on (a)/(b), which AMENDMENT 2's
original (non-sparse) battery did not need to.
