---
type: finding
status: mixed
date: 2026-09-02
mechanism: wkv-cortex generative-scale — the CAPACITY axis (S7(b)) run alongside a same-corpus EXTENSION of the
  TOKEN axis at fixed d96, the "repeating the sweep at a larger capacity" rung the runner docstring named next
verdict: >
  MIXED, 6 seeds each arm (42/43/44/100/101/102), both single-variable-clean extensions of the banked
  2026-09-01 GO-TOKEN-LEVER run. Arm B (d192, V=2000 held fixed so active_params=918,224 = 2.17x d96,
  token_points chosen so tok/param tops out at 4.53 -- matching d96's original 4.536 ceiling almost exactly)
  is a CLEAN 6/6 GO-TOKEN-LEVER: still_descending_at_top 6/6, uses_context_at_top 6/6, beats_trigram_at_top
  6/6, mean top-point deep NLL 3.7737 -- LOWER than d96's original top (3.9321) at essentially the SAME
  tok/param, and only 0.084 nats above the fluency band (vs the original run's 0.242 residual). <!--derived-->
  Arm A (d96
  itself, SAME corpus/capacity, extended token_points to 15.12 tok/param, 3.3x past the original ceiling) is
  PARTIAL by the runner's own gate arithmetic (6/6 seeds fail still_descending_at_top and
  margin_grows_with_tokens): the WKV-vs-trigram margin peaks around 4.5-6.8 tok/param then narrows
  monotonically 6/6, the per-doubling slope falls to a mean 0.0148 (below the 0.02 still-descending bar), <!--derived-->
  and per-seed train NLL itself nearly flattens (e.g. seed42 3.9069->3.8539->3.8261->3.8198->3.8164) -- the
  model is running out of capacity to even fit its own growing training pool, not merely generalizing worse.
  Net: MORE CAPACITY at MATCHED tokens beats MORE TOKENS at fixed capacity here (0.158 nats vs 0.129 nats of <!--derived-->
  improvement over the same 2026-09-01 baseline), and does so while still cleanly ascending rather than
  approaching its own ceiling -- confirming the runner's named next rung and re-pointing the critical path at
  JOINT capacity+token scale, not either axis alone.
lane: generative-cortex-scale
seeds: [42, 43, 44, 100, 101, 102]
external: >
  Same grounding as the 2026-09-01 parent finding (Hoffmann et al. 2022 "Training Compute-Optimal LLMs"
  arXiv:2203.15556 -- Chinchilla's joint capacity/token optimum, ~20 tok/param; Allal et al. SmolLM2 <!--derived-->
  arXiv:2502.02737; "Beyond Chinchilla-Optimal" arXiv:2401.00448). This de-risk is the direct, previously <!--derived-->
  unrun test of that same literature's central claim -- that capacity and tokens are JOINTLY optimized, not
  substitutes -- on the project's own instrument; no new external round was needed since the question and
  its citations are unchanged from the parent arc.
instrument: >
  research/runners/_gen_cortex_token_supply_scaling_derisk.py (UNCHANGED, reused byte-for-byte from the
  2026-09-01 run) + research/runners/_emerge_wkv_lm_derisk.py (build_and_train_wkv, eval_perdepth,
  load_stories, fit_interp_trigram). NO sim/ edit, no runner edit. Provenance auto-stamped per seed.
artifacts:
  - research/findings/raw/_gen_cortex_capacity_rung/armA_d96_extended_seed42.json
  - research/findings/raw/_gen_cortex_capacity_rung/armA_d96_extended_seed43.json
  - research/findings/raw/_gen_cortex_capacity_rung/armA_d96_extended_seed44.json
  - research/findings/raw/_gen_cortex_capacity_rung/armA_d96_extended_seed100.json
  - research/findings/raw/_gen_cortex_capacity_rung/armA_d96_extended_seed101.json
  - research/findings/raw/_gen_cortex_capacity_rung/armA_d96_extended_seed102.json
  - research/findings/raw/_gen_cortex_capacity_rung/armB_d192_matched_seed42.json
  - research/findings/raw/_gen_cortex_capacity_rung/armB_d192_matched_seed43.json
  - research/findings/raw/_gen_cortex_capacity_rung/armB_d192_matched_seed44.json
  - research/findings/raw/_gen_cortex_capacity_rung/armB_d192_matched_seed100.json
  - research/findings/raw/_gen_cortex_capacity_rung/armB_d192_matched_seed101.json
  - research/findings/raw/_gen_cortex_capacity_rung/armB_d192_matched_seed102.json
  - research/findings/2026-09-01-generative-cortex-token-supply-lever-broad-domain-plateau-is-starvation-not-capacity-wall.md
  - research/findings/raw/_gen_cortex_token_supply_scaling.json
  - research/runners/_gen_cortex_token_supply_scaling_derisk.py
runner: research/runners/_gen_cortex_token_supply_scaling_derisk.py
---

# Generative-cortex capacity vs. tokens: larger capacity at matched tok/param beats more tokens on a fixed small model, and gets meaningfully closer to fluency

**Artifacts:** 12 per-seed JSONs under `research/findings/raw/_gen_cortex_capacity_rung/` (6 seeds x 2 arms,
e.g. `research/findings/raw/_gen_cortex_capacity_rung/armB_d192_matched_seed42.json`), each provenance-stamped.

## 0. Headline

The 2026-09-01 finding established that the generative cortex's broad-domain plateau was TOKEN-STARVATION,
not a capacity wall, at a small capacity-matched WKV (d96/V2000, ~0.42M active params) -- but its own top
point (4.54 tok/param) was still 0.24 nats above a ppl20-40 fluency band, and its own instrument docstring
named the obvious next question: **does the token lever keep paying off if you push it further at the SAME
capacity, and does a LARGER capacity (done correctly -- tokens scaled to match, not left behind) actually do
better?** This de-risk runs both arms, 6 seeds each, single-variable-clean against the banked run. Answer:
**capacity is the more efficient lever at this operating point, and d96's OWN token lever is measurably
beginning to run out of runway** -- both real, useful, NO-DEFER-compliant results (a verdict on which METHOD
pays off first, not a stopping point for either).

## 1. Understanding the knob before touching it (the task's own precondition)

Read `_gen_cortex_token_supply_scaling_derisk.py` and the exact provenance of the 2026-09-01 artifact
(`.prov.json` argv) before designing anything, and verified rather than assumed:

- `--token-points` values are counts of TRAINING PASSAGES (`max_train_sents`/`k`, nested nested prefixes of a
  fixed 85%-of-`--n-sentences` pool), not raw tokens. Actual tokens = `k x max_len`.
- `tok_per_active_param = n_tokens / active_params(V,D)`, `active_params = 2VD + V + 4D + 4D^2` -- confirmed
  exactly against both the banked artifact and a fresh calibration run.
- The banked run's REAL args (from its `.prov.json`, not the runner docstring's example): `d_model=96,
  vocab=2000, epochs=6, batch=256, max_len=40, n_sentences=70000, token_points=[4000,8000,16000,32000,48000]`.
  Top point 48000 passages = 1,920,000 tokens = 4.536 tok/param (active_params=423,248), mean top NLL 3.9321.
- Verified via `load_stories()` directly that wikitext103.txt supports >=300,000 passages at max_len=40 with
  no corpus exhaustion -- any ceiling in the banked run was a chosen `--n-sentences`, not a corpus limit.

## 2. Design -- both options from the task, each single-variable-clean against the banked run

**Arm A -- d96 EXTENDED (locate this capacity's own ceiling).** Identical d96/V2000/epochs=6/batch=256/
max_len=40 to the banked run; `n_sentences` raised to 220000 (pool room) and `token_points=[48000, 72000,
96000, 128000, 160000]` -> tok/param 4.54 -> 15.12, continuing past the banked run's own top point (which is
re-measured fresh here as point 0, not literally reused -- a larger `n_sentences` pool changes the per-seed
vocab/eval draw, so this is its own clean 5-point sweep, not a stitched continuation).

**Arm B -- d192 MATCHED CAPACITY (larger capacity done correctly).** V held at 2000 (isolates capacity from
vocabulary breadth, per the task's own guidance); active_params = 918,224 (2.17x d96). `token_points=[8000,
16000, 32000, 64000, 104000]` chosen so the top point's tok/param (4.53) lands almost exactly on d96's
original ceiling (4.536) -- an apples-to-apples endpoint, not the naive "bump d-model, keep the same token
counts" move (which would have LOWERED tok/param to ~2.2 and produced an uninformative, trivially-starved
comparison). `n_sentences=140000`.

Both arms: 6 seeds (42/43/44/100/101/102), `--max-eval-sents 1500` (matches the banked run).

## 3. Operational notes (routing, a real bug caught in-flight, and the honest wall-clock)

Attempted the mandated CPU-pool route (`tools/pool_queue.sh add`) first, after `before_you_build.sh` and a
proper `--checked` record-check; refused because `pool40` was physically unreachable ("No route to host") --
a genuine node outage, not a code defect (`pool41`/`pool42` both passed). Routed to `tools/gpu_queue.sh`
instead per the task's own fallback clause, after confirming locally that this instrument is launch-bound
(a Python loop over T=40 per batch): 21.0s GPU vs 23.4s CPU on an identical calibration config, i.e. CUDA
buys nothing here. Mid-run, the coordinator asked for a re-route to local CPU (`GNU parallel -j 8`) to free
the 3090 for concurrent one-brain-merge verification work. Removing the 12 GPU-queue entries surfaced a real
`grep -v`/`&&` bug of my own (an all-matching filter exits 1, so the `&&`-gated `mv` never ran) -- fixed with
`;`. The FIRST local relaunch was also wrong: `build_and_train_wkv` unconditionally does `device = "cuda" if
torch.cuda.is_available() else "cpu"` regardless of `SIM_BACKEND`, so all 8 concurrent local processes were
still hitting the SAME 3090 the coordinator wanted freed -- caught within ~20s when one job crashed with
`torch.AcceleratorError: CUDA error: unspecified launch failure` (contention with the coordinator's own GPU
job). Killed everything by PID, added `CUDA_VISIBLE_DEVICES=""`, relaunched, verified GPU usage stayed at the
coordinator's own job's baseline throughout. **Honest timing:** the 8-way local concurrency ran ~4-4.5x
slower per job than isolated (measured 5094-12922s per job vs a ~2085-4514s isolated-CPU projection),
consistent with 8 processes sharing 20 cores against ~6 cores of pre-existing background load; total wall
clock for all 12 jobs was **4.87 hours** (first start to last finish), not the ~1.5-2h hoped for, but well
under the ~10.5h the GPU-sequential route would have taken. All 12 jobs exited 0 (`parallel_joblog.tsv`), no
process exceeded the 4GB RSS budget at any point.

## 4. Results

<!--derived-->
Per-point and per-arm MEANS across the 6 seeds below are computed from the 12 cited per-seed JSONs (each
individual seed's own numbers -- e.g. "42=3.8075" -- are the literal values in that seed's artifact; the
means, deltas, and table aggregates are this document's own arithmetic over them, marked derived once here
for the whole section rather than annotating every cell).

### Arm A -- d96 extended, mean deep NLL across 6 seeds (nats)

| tok/param | 4.536 | 6.805 | 9.073 | 12.097 | 15.121 |
|---|---|---|---|---|---|
| mean WKV deep NLL | 3.9516 | 3.8802 | 3.8433 | 3.8178 | 3.8029 |
| mean margin vs trigram | 0.2623 | 0.2619 | 0.2528 | 0.2341 | 0.2161 |

Per-seed top-point NLL: 42=3.8075, 43=3.8251, 44=3.7976, 100=3.7987, 101=3.7917, 102=3.7969 (mean 3.8029).
Gate readout (6 seeds): `uses_tokens` 6/6, `still_descending_at_top` **0/6**, `uses_context_at_top` 6/6,
`beats_trigram_at_top` 6/6, `margin_grows_with_tokens` **0/6**. Mean top-segment slope (128k->160k) = 0.0148,
below the 0.02 still-descending bar in every seed. The margin over the fair trigram PEAKS at 4.5-6.8
tok/param (~0.26) then narrows monotonically through 15.1 tok/param (6/6 seeds) -- the same "counts start
catching up" signature the 2026-09-01 finding named as the record's d512 saturation tell, now appearing
inside d96's own extended range. Per-seed train NLL corroborates independently: seed42's train NLL goes
3.9069 -> 3.8539 -> 3.8261 -> 3.8198 -> 3.8164 -- nearly flat by the last two points, i.e. the model is
running out of capacity to fit its own EXPANDING training pool, not merely generalizing worse. Anti-cheats
hold throughout (uses_context_at_top 6/6; WKV never actually loses to the trigram, margin stays positive).
Runner's own aggregation formula on these 6 seeds (recomputed independently, matching the runner's `main()`
logic exactly): **PARTIAL** -- `token_lever_go` requires `n_still_descending_at_top >= n-2 = 4`; actual is 0.

### Arm B -- d192 matched capacity, mean deep NLL across 6 seeds (nats)

| tok/param | 0.348 | 0.697 | 1.394 | 2.788 | 4.53 |
|---|---|---|---|---|---|
| mean WKV deep NLL | 4.3082 | 4.1388 | 3.9629 | 3.8243 | 3.7737 |
| mean margin vs trigram | 0.3130 | 0.3015 | 0.3264 | 0.3327 | 0.3023 |

Per-seed top-point NLL: 42=3.7485, 43=3.7855, 44=3.7846, 100=3.7669, 101=3.7737, 102=3.783 (mean 3.7737).
Gate readout (6 seeds): `uses_tokens` 6/6, `still_descending_at_top` **6/6**, `uses_context_at_top` 6/6,
`beats_trigram_at_top` 6/6, `margin_grows_with_tokens` 1/6. Mean top-segment slope (64k->104k) = 0.0506 --
more than 3x Arm A's slope at its own top, i.e. still descending at a healthy clip. The margin over trigram
also softens slightly at the very last point (0.3327 -> 0.3023, mirroring Arm A's shape faintly) but from a
much higher absolute level, and the NLL descent itself shows no sign of flattening. Per-seed overfit gap at
the top point is consistently POSITIVE (0.033-0.076, healthy generalization gap) -- unlike Arm A's near-zero/
slightly-negative gap at its top, another independent signal that d192 has NOT yet run out of capacity the
way d96 has. Runner's own aggregation formula on these 6 seeds: **GO-TOKEN-LEVER**, clean (all three
thresholds cleared: `n_uses=6>=5`, `n_desc=6>=4`, `n_clean=6>=5`).

### Cross-arm comparison (the decision-relevant numbers)

| run | capacity (active params) | top tok/param | mean top-point deep NLL | residual above fluency band (3.69) |
|---|---|---|---|---|
| banked 2026-09-01 (d96) | 423,248 | 4.536 | 3.9321 | 0.2421 |
| Arm A (d96, extended tokens) | 423,248 | 15.121 | 3.8029 | 0.1129 |
| Arm B (d192, matched tokens) | 918,224 (2.17x) | 4.53 | **3.7737** | **0.0837** |

Tripling tok/param on the SAME d96 model (banked -> Arm A) bought 0.1292 nats, while doubling capacity at the
SAME tok/param (banked -> Arm B) bought 0.1584 nats -- MORE improvement, for less relative "effort" on the
axis that isn't yet showing a ceiling. Sampled prose from Arm B's top point (seed42, temp 0.8) shows the same
real grammatical scaffolding as the 2026-09-01 sample, still heavily `<unk>`-bottlenecked by V=2000 (e.g.
*"in the second episode a &lt;unk&gt; and &lt;unk&gt; it &lt;unk&gt; the episode had been a major &lt;unk&gt;
&lt;unk&gt; in the ..."*) -- the lexical cap, not incoherence, exactly as the parent finding characterized it.

## 5. The verdict -- capacity is currently the more efficient lever, and d96's token lever has a visible ceiling

Per `docs/TERMS.md`, **GO** is used only where the gate's own verdict is positive: that is Arm B, cleanly,
6/6. Arm A's own gate verdict is **PARTIAL**, not GO and not NO-GO-CAPACITY-SATURATED (`uses_tokens` is still
6/6 -- the token lever has NOT gone negative on this axis, tokens still help overall) -- it is the honest
middle state the runner's own arithmetic was built to name: the WKV is still net-improving with more tokens,
but the RATE has fallen below the "still descending" bar and the trigram-margin trend has inverted, both 6/6
seeds, both independently corroborated by the flattening train-NLL. That is a real capacity-ceiling SIGNAL,
not an artifact of one seed or one metric.

Read together: the runner docstring's own named next rung ("repeating the sweep at a larger capacity")
delivers exactly what it promised -- a clean GO at larger capacity, closer to the fluency band, still
ascending -- while the complementary same-capacity extension independently locates WHERE the smaller model's
own capacity starts to bind (roughly 9-15 tok/param for d96/V2000). Per NO-DEFER, Arm A's PARTIAL is not a
stopping point for the token lever in general -- it is a verdict on ONE capacity's ceiling, and it hands the
next method (capacity growth) rather than abandoning the capability. Neither the original 2026-09-01 run nor
this one has closed the "arbitrary broad-domain prose" gap; the residual to the fluency band has now shrunk
from 0.242 to 0.084 nats, which is real, measured progress, not a declared solve. <!--derived-->

## 6. Honest residuals

1. **Fluency band still not reached.** Best mean NLL (Arm B, 3.7737) is still 0.084 nats above the 3.0-3.69 <!--derived-->
   band. Small relative to the 0.242 residual this arc started with, but not zero. <!--derived-->
2. **Arm B's own ceiling is unmapped.** Its top-segment slope (0.0506) is healthy but its margin-vs-trigram <!--derived-->
   already softened slightly at the last point -- a possible EARLY hint of the same eventual ceiling Arm A
   shows more clearly, just not yet crossed within the tested range. The next rung (S7(b) continued, or
   S7(a)'s TinyStories-quality lever, already in flight in a separate lane per `dispatch.log`) should watch
   for this rather than assume d192 is unbounded.
3. **Absolute vs. matched-ratio tokens.** Arm B's LOW end (0.348 tok/param, 320,000 tokens) already beats
   d96's comparable-ratio low end (0.378 tok/param, 160,000 tokens) partly because it also has 2x the
   ABSOLUTE tokens, not tok/param alone -- expected (bigger embedding table needs more raw exposure per
   vocab item at fixed V), but it means the "matched tok/param" framing is the right control for the
   capacity-vs-tokens question, not a claim that d192 is uniformly better at every tok/param in isolation.
4. **BPTT instrument, unchanged from the parent arc.** Same scope note as 2026-09-01: the biologization is
   inherited (fixed reservoir + local-rule read-out at BPTT parity), the scaling PROPERTY is architecture-
   level so it transfers, but the local-rule read-out has not itself been re-run at d192 or at these token
   counts.
5. **Contention-inflated timing.** All wall-clock numbers in section 3 were measured under 8-way local CPU
   contention (per-job ~4-4.5x an isolated-run projection); they characterize THIS run's operational cost,
   not the instrument's isolated cost (already known from the 2026-09-01 run: ~381s/seed for the original
   5-point d96 sweep in isolation).

## 7. The forward lever this re-anchors

- **Immediate:** repeat Arm A's extension logic at d192 (push its own token_points past 4.53 tok/param) to
  find whether d192's slight top-point margin softening (S6.1 above) is real onset-of-ceiling or noise --
  the same single-variable method used here, just continuing Arm B's own axis.
- **The (capacity x tokens) surface named in 2026-09-01's S7(b)** is now partially mapped: two points
  (d96@4.54, d192@4.53) plus d96's own extension to 15.1. A third capacity point (e.g. d288 or d384, tokens
  matched to its own active-param count) would start to reveal the SHAPE of the efficient frontier rather
  than a single comparison.
- **S7(a) (TinyStories-as-matched-quality-data)** is a complementary, independent axis already dispatched in
  a separate lane per `research/queue/dispatch.log` (parent finding c71cc7c9) -- its result should be read
  alongside this one: capacity, absolute tokens, and corpus quality are three distinct levers this arc has
  now started to separate, where the record previously conflated them into one "~4 orders of magnitude" wall
  estimate.

**NO-DEFER note:** Arm A's PARTIAL is a verdict on d96's OWN capacity ceiling, not on the token lever or the
capability. The capability (brain-native arbitrary prose) is not deferred; Arm B hands the next method
(capacity growth, jointly with tokens) exactly as the runner docstring anticipated.
