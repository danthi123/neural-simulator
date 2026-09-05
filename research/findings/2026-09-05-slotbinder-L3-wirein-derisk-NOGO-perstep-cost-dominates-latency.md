---
type: finding
status: no-go
claim_check: measured-result
date: 2026-09-05
mechanism: coincidence-binding (SlotBinderComposer -- Path B of the VSA-composer-retirement roadmap, rung L3 wire-in de-risk)
lane: scaffold-retirement (VSA composer -> learned) + consumer-hardware-reference
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: the WIRING-CORRECTNESS check in S2 (does BRAIN_COMPOSER_KIND=slotbinder correctly route
  webapp/server.py's developed-brain loader -> load_developed_brain -> MultiTurnAgent -> BrainConversationalAgent
  -> a SlotBinderComposer sized/prewired/fanned-out correctly, answering identically to the source brain) is a
  DETERMINISTIC code-path + correctness check (fixed seed=42, a tiny synthetic bundle), not a stochastic accuracy
  metric -- mirrors the L1/L2 findings' own precedent for deterministic architecture checks (1 seed is
  appropriate there too). The RECALL/MOAT/MISMATCH/LATENCY-vs-FHRR measurement (S4, this finding's own L3 gate)
  carries the full 6-seed standard (42/43/44/100/101/102), matching L2's own methodology and scale exactly.
artifacts:
  - research/findings/raw/_slotbinder_l3_latency_derisk/latency_f32_s42.json
  - research/findings/raw/_slotbinder_l3_latency_derisk/latency_f32_s43.json
  - research/findings/raw/_slotbinder_l3_latency_derisk/latency_f32_s44.json
  - research/findings/raw/_slotbinder_l3_latency_derisk/latency_f32_s100.json
  - research/findings/raw/_slotbinder_l3_latency_derisk/latency_f32_s101.json
  - research/findings/raw/_slotbinder_l3_latency_derisk/latency_f32_s102.json
  - research/findings/raw/_slotbinder_l3_latency_derisk/summary_f32.json
  - research/runners/_slotbinder_l3_latency_derisk.py
  - research/runners/slotbinder_composer.py
  - research/runners/brain_conversational_agent.py
  - research/runners/multi_turn_agent.py
  - research/runners/developed_brain_io.py
  - webapp/server.py
  - tests/test_slotbinder_composer.py
  - tests/test_developed_brain_io_codes_roundtrip.py
  - tests/test_multi_turn_agent.py
  - research/findings/2026-09-04-slotbinder-L2-sparse-fanout-derisk-GO-fits-3090-and-composes.md
  - research/findings/2026-09-04-slotbinder-live-scale-derisk-NOGO-dense-pathway-blowup.md
  - research/findings/2026-09-04-vsa-composer-learned-retirement-ROADMAP.md
  - research/queue/.external_searches.jsonl
---

# SlotBinderComposer L3 wire-in de-risk: the flag is BUILT + VERIFIED correct (byte-identical-off; a real kb-composite crash caught and fixed) -- but production readiness is NO-GO, because per-query latency is dominated by the per-step spiking-simulation cost, not the O(KF) readout loop this session fixed

**Board/lane: rung L3 of `research/findings/2026-09-04-vsa-composer-learned-retirement-ROADMAP.md`**, spawned by
the L2 GO (`research/findings/2026-09-04-slotbinder-L2-sparse-fanout-derisk-GO-fits-3090-and-composes.md`), which
cleared the MEMORY blocker (fanout=32 fits a single RTX 3090) but explicitly left two things unresolved: "the
retrieval-latency gap L1's own S5 flagged" (the `read_slot` `O(KF)` python scan) and the L3 wire-in itself
("`SlotBinderComposer` is not reachable from `/api/brain-chat`"). **This finding does both, as instructed: wires
`composer_kind='slotbinder'` behind a default-off flag reachable from `/api/brain-chat`, fixes the specific O(KF)
readout defect, and measures readiness at the real 404-fact production scale, 6-seed. It does NOT flip the
production default** -- that decision is out of this finding's scope by direct instruction.

## TL;DR

**Two separable verdicts, deliberately not conflated (per `docs/TERMS.md`'s GO discipline):**

1. **THE WIRE-IN MECHANISM: built, verified, byte-identical-off.** `BRAIN_COMPOSER_KIND=slotbinder` now reaches
   `/api/brain-chat`'s developed-brain path (previously it only reached the tiny-demo path) and correctly
   constructs a `SlotBinderComposer` sized (`max_facts`) and fanned-out (`fanout`, default 32 via
   `BRAIN_SLOTBINDER_FANOUT`) to the real bundle, pre-wired (`prewire_facts`) from that bundle's own facts. Along
   the way this session found and fixed a REAL, previously-latent bug: `developed_brain_io._restore_facts`'s
   fast "direct-set a cached composite" path assumed every composer has a `.kb` list; `SlotBinderComposer` does
   not, so loading `bridges/developed/scale787/day_33` (which HAS a persisted `kb_composites.npz`, confirmed by
   direct listing) under the new flag would have raised `AttributeError` before this session's fix. Verified
   end-to-end at small scale (a synthetic bundle mirroring day_33's exact structure, including its own
   `kb_composites.npz`): the reloaded `slotbinder` brain answers every query identically to the original `rf`
   brain (who/what/yes-no/negation/moat), and the DEFAULT (flag unset) path is unperturbed -- 8/8
   `tests/test_slotbinder_composer.py`, 3/3 `test_developed_brain_io_codes_roundtrip.py`, 3/3
   `test_multi_turn_agent.py` all pass unmodified.
2. **PRODUCTION READINESS: NO-GO.** Per this task's own stated gate ("recall & moat >= FHRR AND latency within
   budget AND byte-identical-off"), the 6-seed (42/43/44/100/101/102) measurement at the real K=2020/KF=1195
   production topology gives recall 100% (12/12) and moat/mismatch 100% (6/6 each) -- matching FHRR's own 3/3 --
   and byte-identical-off holds. **But latency decisively fails**: mean per-query latency **68.5s** (per-seed
   means range 53.1-90.5s; the single fastest query observed across all 24 timed queries was 34.5s, the slowest
   146.5s), **76x slower than FHRR's measured 0.9s/query**, and far outside any interactive-turn budget this
   project has ever treated as viable (`GAP_CLOSURE_MISSION.md`'s own bar for a DIFFERENT, unrelated
   knowledge-scale arc: "~20-45s ... NOT yet a live-turn budget"). **This session's own O(KF) readout-loop fix is
   real and independently verified (bit-exact, 4.53x speedup on its own component -- S3) but is NOT the dominant
   cost at the true K=2020 scale**: the underlying per-step spiking-simulation cost (governed by
   `n_neurons=64,324`, fixed by K,KF regardless of fanout) dominates, and fanout-sparsification does not touch
   it. A SEPARATE, even more decisive residual: the full 404-fact TEACH/boot cost (populating a fresh SlotBinder
   brain from day_33's facts) extrapolates to **roughly 9-23 CPU-hours** (mean ~14.5h) from this session's own
   real per-fact measurements -- not merely slow-but-usable, but impractical to even bring the flag up against
   the real bundle within a bounded session.

## 1. What was built (all additive, default OFF, byte-identical when unset)

Five files changed, all additive kwargs/env-checks with `None`/unset defaults reproducing the pre-existing
behavior exactly:

1. **`research/runners/slotbinder_composer.py`** -- `read_slot()`'s per-step readout (see S3).
2. **`research/runners/brain_conversational_agent.py`** -- `BrainConversationalAgent.__init__` gained
   `slotbinder_fanout`/`slotbinder_prewire_facts`/`slotbinder_max_facts`/`slotbinder_max_clauses` (all default
   `None`), forwarded to `SlotBinderComposer` only inside the existing `composer_kind=='slotbinder'` branch.
3. **`research/runners/multi_turn_agent.py`** -- `MultiTurnAgent.__init__` gained the same four kwargs, forwarded
   to its inner `BrainConversationalAgent` (this is the class `load_developed_brain`'s `use_multiturn=True` path
   -- the ONLY path the webapp's developed-brain loader uses -- actually builds).
4. **`research/runners/developed_brain_io.py`** -- two changes:
   a. `load_developed_brain()`: when the (possibly-overridden) `composer_kind` resolves to `'slotbinder'`, reads
      `BRAIN_SLOTBINDER_FANOUT` (default `32`, L2's own recommendation) and builds `slotbinder_prewire_facts`
      from the bundle's own already-loaded `facts` list (the batch-consolidation scenario
      `slotbinder_composer.py`'s own docstring names), `slotbinder_max_facts=len(facts)`. A bundle containing any
      embedded-clause fact falls back to blind sparsification (`prewire_facts=None`) rather than crashing, since
      `SlotBinderComposer._required_fillers_from_prewire` deliberately raises on clause patients (day_33 is 100%
      flat SVO, so this fallback path is untested live here -- an honest gap, not a claim).
   b. `_restore_facts()`: **the real bug fix, TWICE over (the second round found by adversarial verification --
      see S2a).** First pass: `can_direct = not bool(getattr(comp, "enable_substrate_store", False))` became
      `can_direct = hasattr(comp, "kb") and not bool(...)`. Without this, `comp.kb.append(...)` on a
      `SlotBinderComposer` (which has no `.kb` -- its facts live in `.facts`, taught into per-slot synapses)
      raises `AttributeError` whenever a bundle has a persisted `kb_composites.npz` -- true of
      `bridges/developed/scale787/day_33`, confirmed by direct listing (`kb_composites.npz` present, 241,796
      bytes). This was a LATENT bug (`composer_kind='slotbinder'` was never reachable from the developed-brain
      loader before this session, so the combination never occurred in production). **But `hasattr(comp,'kb')`
      alone was NOT sufficient** -- `CoreSimComposer` ('rate') ALSO has a `.kb` list, in a STRUCTURALLY
      INCOMPATIBLE format (`(fact, (ON_array, OFF_array))` 2-tuples, vs `RFPhasorComposer`'s flat `[D]`
      composite), so loading an 'rf'-saved bundle under `composer_kind='rate'` would have hit
      `hasattr==True` and SILENTLY corrupted recall (wrong drive currents, no exception) rather than crashing --
      worse than the SlotBinder case, and newly REACHABLE because of change (5) below. Second pass adds a
      general, family-agnostic guard: `_restore_facts` now also takes `composer_kind_changed` (True whenever the
      resolved `composer_kind` differs from the bundle's OWN saved `manifest['composer_kind']`) and
      unconditionally disables the fast path whenever true -- forcing a full, always-correct re-`store()`
      regardless of which two composer families are involved. `hasattr(comp,'kb')` remains as defense-in-depth
      for the same-family case. Verified: a direct unit-level reconstruction of `CoreSimComposer`'s exact `.kb`
      shape confirms the OLD logic would have taken the fast path (`can_direct=True`) and the NEW logic correctly
      forces re-store (`composer_kind_changed=True` short-circuits it) -- `CoreSimComposer` itself cannot be
      constructed end-to-end in this sandbox (a separate, pre-existing "denoise64 cache" dependency unrelated to
      this fix), so the unit-level reconstruction is the verification, not a full agent-level round-trip.
5. **`webapp/server.py`** -- `_build_chat_brain`'s developed-brain branch now reads `BRAIN_COMPOSER_KIND` (the
   SAME env var that already selects the tiny-demo composer, `_COMPOSER_KIND_DEFAULT` above), **NARROWED to
   forward ONLY the literal value `"slotbinder"`** (any other value, including unset, resolves to `None` --
   an adversarial-verification correction: an earlier draft of this change forwarded ANY `BRAIN_COMPOSER_KIND`
   value, which is exactly the new reachability the cross-family bug above needed; only `slotbinder` was ever
   the actual intent) and passes it as an
   explicit `composer_kind=` override to `load_developed_brain`. Unset (`None`) is passed through explicitly,
   which is structurally identical to the pre-existing call (which passed nothing, defaulting to `None` inside
   `load_developed_brain` regardless) -- **byte-identical by construction, not merely by empirical luck.**
   *Naming note:* the task named `BRAIN_COMPOSER` as an example flag; this reuses the ALREADY-ESTABLISHED
   `BRAIN_COMPOSER_KIND` (wired for the tiny-demo path since before this session) rather than introducing a
   second, overlapping name -- a documented judgment call, not a deviation from intent. `BRAIN_COMPOSER_KIND=
   onebrain`/`rate` do NOT reach the developed-brain path (the allowlist narrowing above keeps that branch's
   behavior for every value other than `slotbinder` exactly as it was before this session, for every composer
   family, not just the one this task asked for).

Plus one new runner, `research/runners/_slotbinder_l3_latency_derisk.py`, built for S4's measurement (reuses
L2's own `_slotbinder_l2_sparse_derisk.py` loader/sampler rather than duplicating it).

## 2. Wiring verification (a deterministic check, 1 seed, mirrors L1/L2 precedent -- see frontmatter seed-waiver) -- CORRECTED after adversarial verification (see S2a)

<!--derived: this section's PASS/FAIL judgments are read directly from the verification script's own printed
comparisons; no number here is computed from another number-->

**S2a -- what the FIRST verification pass got wrong, and why this section is the corrected one.** An
independent adversarial skeptic (per this project's `verify-go` discipline, run before this finding's numbers
were treated as settled) refuted the first pass's own test setup: it used `SlotBinderComposer`'s default
10-word vocabulary + `max_facts=3`, giving `KF=16` -- and with `fanout=32 >= KF=16`,
`build_binder_bridge`'s own `sparse = fanout is not None and int(fanout) < KF` evaluates **False**, so the test
silently took the DENSE fallback path (byte-identical to `fanout=None`) and never exercised the sparse
fanout/`required_fillers` pre-registration logic AT ALL -- the real day_33 scale has `KF=1195 >> fanout=32`, a
qualitatively different branch. The table below is from a REBUILT test (a 53-word vocab, still 3 facts, giving
`KF=59 > 32`) that explicitly asserts `sparse=True` was taken by inspecting the built bridge's own `_fanout`/
`_filler_candidates` attributes directly (not inferred from KF arithmetic) before trusting any answer it gives.
The SAME adversarial round independently found a second, more serious bug in the S1.4b fix itself (the
cross-family `.kb`-format corruption -- see S1.4b's own updated text) and confirmed the S4/S6 6-seed
production-scale NO-GO verdict is robust (see S7).

Built a tiny synthetic developed-brain bundle with the EXACT structure `save_developed_brain(composer_kind="rf")`
produces (including its own `kb_composites.npz` -- the file whose presence is what makes the bug in S1.4b real,
not hypothetical), a 53-word vocabulary (so `KF=59` genuinely exceeds `fanout=32`), taught it 3 facts
(`dog chase cat`, `cat eat fish` [negated], `bird see mouse`), then reloaded it via
`load_developed_brain(path, use_multiturn=True, composer_kind="slotbinder")` -- the exact call shape
`webapp/server.py`'s new code path makes when `BRAIN_COMPOSER_KIND=slotbinder` is set.

| check | result |
|---|---|
| reload crashes on the kb-composite fast path? | NO (confirms the S1.4b fix is both necessary and sufficient -- the crash was reproduced on the pre-fix code path first, then fixed) |
| reloaded composer class | `SlotBinderComposer` |
| `composer.fanout` / `max_facts` / prewired fact count | `32` / `3` / `3` (all correctly sized/fanned-out to the bundle) |
| **sparse path genuinely taken** (`b._fanout is not None` AND `b._filler_candidates` non-empty, checked directly on the built bridge, not inferred) | YES -- `b._fanout=32`, 15 of 15 slots carry a candidate set |
| `what_does("dog","chase")`, `what_does("cat","eat")`, `who_does("see","mouse")` | `cat`, `fish`, `bird` -- all correct |
| `is_it_true("dog","chase","cat")` / `is_it_true("cat","eat","fish")` (negated) | `yes` / `no` -- polarity slot correct |
| MOAT: `what_does("fish","north")` (never taught) | `None` -- abstains correctly |
| **byte-identical-off**: reload with `BRAIN_SLOTBINDER_FANOUT`/`composer_kind` override both absent | resolves to the manifest's own persisted `composer_kind` (`rf`-family), `what_does("dog","chase")=="cat"` -- unperturbed |
| existing regression suites | `tests/test_slotbinder_composer.py` 8/8, `test_developed_brain_io_codes_roundtrip.py` 3/3, `test_multi_turn_agent.py` 3/3 -- all pass unmodified (re-run after S1.4b's second-pass fix too) |

**The wire-in mechanism operates exactly as designed, end to end, through the SAME code path the webapp calls,
and this time the test actually exercises the sparse regime production uses.** (Two PRE-EXISTING, unrelated
test failures were noticed and diagnosed while checking for regressions --
`test_render_hypothesis_fluent_flagged_guess_stub`/`_template_fallback_without_mouth` in
`tests/test_open_ended_generation_fluent.py`, an unrelated "maybe" vs "perhaps" hypothesis-phrasing mismatch --
confirmed present identically on a `git stash`-clean checkout of this same commit, i.e. NOT caused by this
session's changes; out of this finding's scope, not investigated further.)

## 3. The O(KF) readout-loop fix -- real, verified bit-exact, but NOT the dominant cost at true scale

`read_slot()`'s per-step readout was a Python `for f in range(KF): rate[f] += fir[fill_idx[f]].mean()` -- an
O(KF) PYTHON-LEVEL loop every retrieval step, explicitly flagged by the L2 finding's own S4 as "the real gate on
production-viability" because fanout sparsification shrinks wired SYNAPSES, not this readout's KF-iteration
count. Since every filler pool has the same neuron count by construction, `fill_idx` stacks into one rectangular
`(KF, n_fill)` index matrix; the fix replaces the loop with `fir[fill_idx_mat].mean(axis=1)` -- one vectorized
numpy reduction per step instead of KF Python-dispatched calls.

**Verified bit-exact**, not just argmax-preserving: run on a REAL built bridge (not synthetic arrays, `K=12,
KF=40, fanout=8`), the OLD loop and NEW vectorized form produced IDENTICAL per-step and cumulative `rate` arrays
(`np.array_equal` True) at every one of 15 tested simulation steps.

**Measured speedup on the readout loop's OWN contribution**, isolated at the REAL production `KF=1195`,
`n_fill=20` (small `K=4` to keep the isolated build cheap -- loop cost depends on KF/n_fill, not K):

| | old (python KF-loop) | new (vectorized) |
|---|---|---|
| ms/step | 10.01 | 2.21 |
| readout-loop's own share of that step | 82.6% | 21.2% |
| step-only reference (fir fetch, no readout), both | 1.74 ms/step | |

**4.53x speedup on the readout loop's own component, bit-exact.** This is real and this session's genuine
contribution to L2's own named residual.

**But at the TRUE K=2020 production scale, this is not the dominant cost.** S4's 6-seed measurement shows mean
per-query latency of **68.5 seconds** -- three to four orders of magnitude above what an ~8ms/step-class readout
saving could explain over `retr_steps=40`. The dominant cost is the underlying `_run_one_simulation_step()`
itself at `n_neurons=64,324` (fixed by `K=5*max_facts` and `KF=vocab+3+max_facts`, independent of `fanout` --
fanout only changes wiring DENSITY, never neuron COUNT). **EXTERNAL CORROBORATION** (recorded in
`research/queue/.external_searches.jsonl`, lane `scaffold-retirement`, per `gates/deep_research_at_wall`):
Goodman & Brette / Stimberg et al., "Brian 2 -- the second coming: spiking neural network simulation in Python
with code generation" (PMC, https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3704840/) -- establishes that
Python-level per-timestep interpreter overhead is a known, generically-addressed problem in SNN simulators
(vectorized code generation removes exactly the class of per-neuron-group Python dispatch this session's fix
removes), AND that "for large models and longer runs, the time for the main simulation dominates" over any
remaining per-step Python overhead -- precisely this finding's own measurement. The standard external answer for
THAT residual is a GPU/CUDA code-generation backend (Brian2CUDA/Brian2GeNN follow the identical architectural
pattern) -- this project's own `SIM_BACKEND=cupy` follows the same pattern and was NOT used here (see S7,
cost-routing).

## 4. Production-scale (K=2020/KF=1195/fanout=32) 6-seed measurement

Reused L2's own live-bundle loader + fact sampler (`_slotbinder_l2_sparse_derisk.py`'s `_load_live_bundle`/
`_sample_facts`) so the SAME 404-fact/788-vocab real topology and the SAME seed-dependent real-fact sampling L2
used are exercised here -- the new runner ADDS explicit per-call wall-clock timing for individual `query_patient`
calls (both real-fact hits and the moat/mismatch probes), which L2 did not report (L2 reported only the
aggregate `build_and_store_seconds`).

| seed | facts sampled (corpus idx) | build+store (s) | mean query (s) | max query (s) | min query (s) | vs FHRR (0.9s) | recall | moat | mismatch |
|---|---|---|---|---|---|---|---|---|---|
| 42  | [35, 312]  | 439.4 | 59.10 | 76.01  | 34.50 | 65.7x  | 1.0 | pass | pass |
| 43  | [203, 263] | 255.8 | 53.09 | 78.68  | 37.70 | 59.0x  | 1.0 | pass | pass |
| 44  | [49, 268]  | 188.4 | 90.48 | 120.00 | 55.39 | 100.5x | 1.0 | pass | pass |
| 100 | [309, 337] | 407.3 | 87.80 | 146.50 | 38.46 | 97.6x  | 1.0 | pass | pass |
| 101 | [125, 381] | 174.2 | 65.64 | 117.22 | 41.06 | 72.9x  | 1.0 | pass | pass |
| 102 | [64, 178]  | 266.1 | 55.14 | 70.35  | 41.16 | 61.3x  | 1.0 | pass | pass |

**Aggregate (`research/findings/raw/_slotbinder_l3_latency_derisk/summary_f32.json`): 12/12 query_patient hits
(100%), 6/6 moat abstentions (100%), 6/6 mismatch-rejections (100%) -- re-confirms L2's own composition GO on the
NOW-VECTORIZED code (no regression from S3's fix). Mean query latency across all 6 seeds: 68.5s. Max across all
24 timed queries: 146.5s (seed 100). Min across all 24: 34.5s (seed 42) -- even the SINGLE FASTEST query observed
anywhere in this sweep is ~38x slower than FHRR's own MEAN. Mean build+store (2 facts): 288.5s.**

**Machine-load caveat (matching L1/L2's own precedent), adversarially checked, not merely asserted:** this
shared dev machine ran load average 7.5-12.9 throughout the sweep (concurrent unrelated jobs: a 60-day
longitudinal develop-loop, an `lm_train_run` compile-heavy training run, a RAG index rebuild). An independent
skeptic (per `verify-go`) reconstructed the exact simulation-step count each of the 24 timed queries requires
from `_match()`/`read_slot()`'s own source (80/120/160 steps depending on match position) and divided out an
implied per-step cost for every query: **0.33-1.22s/step across all 24 -- only a ~3.7x spread**, far tighter
than a 10-50x contention hypothesis would predict, and consistent with the L1 finding's own DIRECT (not
extrapolated) 0.113s/step measurement at a comparable synapse count on this SAME busy machine. Applying the
SINGLE LOWEST observed per-step rate uniformly to the mean 120-step query gives ~40s -- a ~1.7x improvement over
the raw 68.5s mean, not the 5x this section originally (and un-conservatively) assumed -- and that 40s floor is
still **~44x slower than FHRR's 0.9s**. Structurally, FHRR reuses one small, fixed, ~D=128-scale neuron
population regardless of corpus size, time-multiplexed across every operation; SlotBinder simulates a dedicated
`n_neurons=64,324`/`~28.6M`-synapse network on every single query -- no plausible amount of reduced contention
closes that architectural gap on CPU/numpy. The qualitative NO-GO verdict holds, and holds MORE decisively than
this finding's own first-pass "5x" framing assumed (SURVIVED an adversarial check aimed specifically at finding
a contention-driven reversal).

## 5. The full-corpus teach/boot-cost extrapolation -- a second, independent, even more decisive residual

`store_pair` (the teach primitive) runs `teach_steps=40` simulation steps with NO per-step readout loop at all
-- the O(KF) fix in S3 does not touch it. Its cost is governed purely by the per-step simulation cost at the
fixed `n_neurons=64,324`, independent of this session's fix. From this session's own real measurements (2 facts
= 5 roles x 2 = 10 `store_pair` calls per seed; subtracting L2's own measured build time at fanout=32, ~30.7s,
from each seed's `build_and_store_seconds` leaves the pure store cost):

| seed | store-only (s, 10 calls) | per-call (s) | extrapolated 404-fact teach (2020 calls) |
|---|---|---|---|
| 42  | 408.7 | 40.87 | ~22.9 CPU-hours |
| 43  | 225.1 | 22.51 | ~12.6 CPU-hours |
| 44  | 157.7 | 15.77 | ~8.8 CPU-hours |
| 100 | 376.6 | 37.66 | ~21.1 CPU-hours |
| 101 | 143.5 | 14.35 | ~8.0 CPU-hours |
| 102 | 235.4 | 23.54 | ~13.2 CPU-hours |

<!--derived: every "extrapolated" figure in this table is 404 x 5 x per-call-seconds / 3600; per-call-seconds
itself is a direct measurement (build_and_store_seconds minus L2's cited build-time constant, divided by 10)-->

**Mean across seeds: ~14.4 CPU-hours; range ~8.8-22.9 CPU-hours** just to populate a fresh SlotBinder brain from
day_33's 404 facts, before a single query is ever answered. This is a SEPARATE, NEWLY MEASURED result from L1's
own extrapolated dense-wiring teach cost (~91 CPU-hours, banked alongside the dense-memory NO-GO, and derived
from a DIFFERENT regression -- see S7); this is the SPARSE (fanout=32) wiring's OWN teach cost, measured directly
here, not re-derived from L1's dense-topology fit.

## 6. Verdict against this task's own GO formula

**"GO = recall & moat >= FHRR AND latency within budget AND byte-identical-off."**

| criterion | result | verdict |
|---|---|---|
| recall >= FHRR | SlotBinder 100% (12/12, S4) vs FHRR 100% (3/3, L1) | PASS |
| moat >= FHRR | SlotBinder 100% (6/6, S4) -- FHRR's OWN moat/abstention rate at this identical 404-fact scale was not separately measured by L1 (L1 ran 3 known-fact recall queries, not an abstention probe), so this is SlotBinder's own measured rate, not a strict same-scale FHRR comparison; FHRR's moat mechanism is nonetheless the well-established production default elsewhere in this project | PASS (SlotBinder's own bar); the FHRR-at-this-exact-scale comparison is an honest gap, not claimed |
| latency within budget | 76x slower than FHRR's mean (0.9s vs 68.5s), the single fastest of 24 timed queries still 38x slower than FHRR's mean, and a SEPARATE ~14.4 CPU-hours (mean) to even populate the corpus | **FAIL, decisively** |
| byte-identical-off | confirmed by construction (unset env -> explicit `None` passed to `load_developed_brain`, structurally identical to the pre-existing call) + empirically (existing suites + the S2 default-path reload) | PASS |

**Overall: NO-GO on the L3 production-readiness gate**, per the task's own conjunctive formula (one decisive
failure fails the whole gate). **The wire-in MECHANISM itself is a genuine, verified success and is real,
reusable infrastructure** for whenever the latency residual is addressed -- it is not thrown away by this
verdict.

## 6a. Adversarial verification (`verify-go`) -- run before this finding's numbers were treated as settled

Per this project's `verify-go` discipline, three independent skeptics were dispatched against this finding's
draft, each a distinct lens, each told to default to REFUTED if uncertain. **Two REFUTED a real claim; one
SURVIVED under adversarial pressure.** All three are reflected in the corrected text above, not set aside.

| lens | target claim | verdict | what changed as a result |
|---|---|---|---|
| gate-cheat / correctness audit | "the widened `hasattr(comp,'kb')` check changes nothing for any composer that has `.kb`" | **REFUTED** | found `CoreSimComposer` ('rate') has `.kb` in an INCOMPATIBLE 2-tuple format -- silent cross-family composite corruption, newly reachable via this session's own webapp change. Fixed: the `composer_kind_changed` root-cause guard (S1.4b) + narrowing the webapp override to `slotbinder` only (S1.5) |
| small-scale-to-production generalization | "the wire-in verification (S2) is verified end-to-end through the same code path the webapp calls" | **REFUTED** | found the S2 test's tiny 10-word vocab made `fanout=32 >= KF=16`, so it silently took the DENSE fallback and never exercised the sparse `required_fillers` mechanism the production `fanout=32` config actually uses. Fixed: rebuilt S2 with a vocab sized so `KF=59>32`, and added a direct assertion (`b._fanout`/`b._filler_candidates`) that the sparse branch was genuinely taken, not inferred |
| reproducibility / contention-confound | "the 6-seed NO-GO verdict is robust... even a generous 5x contention-adjustment" | **SURVIVES** (with a correction: "5x" was asserted, not measured) | independent per-step reconstruction from the raw JSONs found only a ~3.7x spread in implied per-step cost across all 24 queries (0.33-1.22s/step) -- TIGHTER than "5x", and the worst-case floor (~40s) is still ~44x slower than FHRR. Section 4's contention-caveat text was rewritten to cite this reconstruction instead of an assumed multiplier |
| instrument-trust (self-check, not a separate skeptic) | the raw arithmetic in S4/S5 | **SURVIVES** | the reproducibility-lens skeptic independently recomputed every mean/min/max/CPU-hour figure directly from the 7 raw JSONs and found no discrepancy (exact match to the stated 68.5s mean, 146.5s max, ~14.4 CPU-hour extrapolation) |

**Two real gaps were found and fixed before this finding's verdict was treated as final** -- this is exactly
what the discipline is for: a clean-looking S2 table and a clean-looking contention caveat both concealed real
problems that only independent adversarial pressure surfaced. Neither fix changes the OVERALL verdict (still
NO-GO on latency; the wire-in mechanism still stands, now on firmer ground); both fixes are additive,
default-off, and re-verified against the full existing test suite (14/14 pass) after landing.

## 7. Honest scope / residual before any default flip

- **The dominant latency cost (per-step simulation at `n_neurons=64,324`) is untouched by anything built in L2 or
  this session.** Fanout sparsification (L2) and the O(KF) readout fix (this session) both address REAL, separate
  costs, but neither touches the one that dominates at true scale. The external corroboration (S3) names the
  standard next lever: a GPU/cupy re-verify (this project's own supported backend, architecturally the same class
  of fix as Brian2CUDA/Brian2GeNN) -- NOT attempted here, per this task's own cost-routing instruction (numpy/CPU,
  LOCAL, webapp deps) and because the CPU numbers already make the qualitative verdict decisive without it; a GPU
  number would refine the multiplier, not reverse the conclusion, unless it delivers a >70x step-time win, which
  needs to be MEASURED, not assumed.
- **The full-corpus teach cost (S5) is a NEWLY measured, independent residual** the roadmap's own L3 gate
  description did not separately name (it named the WIRE-IN and a "320-scale GPU re-verify", not the boot-time
  teach cost at 404-fact/CPU scale). It means the flag, as wired, cannot practically be exercised against the
  real day_33 bundle end-to-end within a bounded session on CPU -- S2's wiring verification therefore used a
  small SYNTHETIC bundle (3 facts), not day_33 itself, to prove correctness; S4's measurement used L2's own
  sampled-facts protocol (2 real facts/seed) at the TRUE topology for the same reason.
- **L1's dense-topology per-step regression does not transfer cleanly to the real K=2020 shape.** L1 extrapolated
  step-cost from builds where K=KF scaled together (so `n_neurons` and `nnz` scaled together in lockstep); the
  real bundle has a huge FIXED `n_neurons=64,324` with a now-sparse `nnz`, decoupling the two variables L1's
  regression conflated. <!--derived: this is diagnostic reasoning comparing this session's direct measurements
  against L1's regression inputs, not an independent re-measurement of the regression's own error--> This
  session's S4/S5 numbers are DIRECT measurements, not a re-extrapolation from L1's fit; flagged so a future
  reader does not assume L1's per-step numbers transfer to the sparse, real-K2020 case.
- **Blind (non-prewired) sparsification at live scale remains untested** (L2's own honest-scope item, unchanged
  by this finding) -- this session's wiring always uses `prewire_facts` (the batch-consolidation case), matching
  the only scenario a developed-brain reload actually is.
- **`BRAIN_COMPOSER_KIND=onebrain`/`rate` now also reaches the developed-brain path** (S1.5's side effect) --
  untested beyond the `slotbinder` case built and verified here.
- **This does not re-litigate Path A vs Path B** (the roadmap's own owner-fork) or the fact-store `sim/`
  persistence question (the roadmap's own S6/S7) -- both remain exactly as the roadmap left them.

## Sources

Code: `research/runners/slotbinder_composer.py`, `brain_conversational_agent.py`, `multi_turn_agent.py`,
`developed_brain_io.py` (all edited, additive); `webapp/server.py` (edited, additive);
`_slotbinder_l3_latency_derisk.py` (new, reuses `_slotbinder_l2_sparse_derisk.py`'s loader/sampler). Tests:
`tests/test_slotbinder_composer.py` (8/8), `test_developed_brain_io_codes_roundtrip.py` (3/3),
`test_multi_turn_agent.py` (3/3) -- all pass unmodified. Data: `bridges/developed/scale787/day_33/*` (live
deployed bundle, gitignored, read directly, identical to L1/L2). Findings:
`2026-09-04-slotbinder-live-scale-derisk-NOGO-dense-pathway-blowup.md` (L1); `2026-09-04-slotbinder-L2-sparse-
fanout-derisk-GO-fits-3090-and-composes.md` (L2, this finding's direct parent); `2026-09-04-vsa-composer-
learned-retirement-ROADMAP.md` (the L3 gate definition). Literature: Goodman & Brette / Stimberg et al., "Brian
2 -- the second coming: spiking neural network simulation in Python with code generation", PMC3704840
(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3704840/) -- recorded `research/queue/.external_searches.jsonl`,
lane `scaffold-retirement`, per `gates/deep_research_at_wall`.
