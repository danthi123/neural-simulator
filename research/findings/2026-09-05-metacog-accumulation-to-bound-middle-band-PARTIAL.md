---
type: finding
status: partial
date: 2026-09-05
mechanism: metacog honesty-hedge recall-margin evidence — an ACCUMULATION-TO-BOUND (sequential-sampling) confidence read over the SAME Izhikevich winner-vs-runner-up recall competition the rank-9 PARTIAL's single-snapshot margin already drives; additive, default-OFF/unwired, opt-in companion read to `RFPhasorComposer._spiking_margin`
lane: introspection-self-model
backlog: research/coordination/scaffold_retirement_backlog.md rank 9 ("metacog confidence host formula") — wall-reframe follow-on to the rank-9 PARTIAL's own characterized residual
backend: numpy
runner: research/runners/_metacog_accumulation_to_bound_derisk.py
unit_tests: tests/test_spiking_margin_accum.py
seeds: 42, 43, 44 (CPU smoke, numpy; the full mandated 42/43/44/100/101/102 spiking battery is deferred to a post-mouth-training GPU queue — command below)
artifacts:
  - research/findings/raw/_metacog_accumulation_to_bound_derisk/smoke_3seed.json
  - research/findings/raw/_metacog_accumulation_to_bound_derisk/smoke_3seed.json.prov.json
---

# Accumulation-to-bound raises type-2 discrimination in the rank-9 residual's ambiguous middle band — direction confirmed, magnitude modest, CPU-smoke scale (PARTIAL)

**Verdict: PARTIAL, default-OFF and UNWIRED (de-risk only, per `docs/TERMS.md` — not "closed", not "validated" at
production scale).** The wall-reframe hypothesis — that the rank-9 PARTIAL's single-SNAPSHOT recall-margin
discards the deliberation TRAJECTORY a real sequential-sampling / accumulation-to-bound confidence read would use
— is **directionally confirmed on CPU-smoke evidence**: an accumulation-based read of the SAME Izhikevich
winner-vs-runner-up competition **out-discriminates the snapshot on genuine correctness (a type-2 AUC, not
agreement-with-host) in the ambiguous middle band, consistently across all 3 seeds tested**, with the largest gain
from the **time-to-bound diagnostic** the mission frame specifically named. The effect is **real but modest**
(no signal here reaches strong discrimination — the best pooled AUC is 0.585 <!--derived-->, far below the ~0.825
the task brief associates with an unrelated metacog subsystem elsewhere) and measured only at **CPU-smoke scale** (a tiny
5-role/~15-word composer, 3 seeds, extreme synaptic-noise levels engineered to reach the band) — not yet the
mandated 6-seed spiking validation. Building this also surfaced and fixed a genuine **instrument bug** (a
naive-ratio bound criterion is dominated by small-spike-count noise); see below.

## Verify-first: the PARTIAL's residual is real and accurately described

<!--derived-->

Read [`2026-09-05-metacog-spiking-recall-margin-derisk-PARTIAL.md`](2026-09-05-metacog-spiking-recall-margin-derisk-PARTIAL.md),
its runner (`research/runners/_metacog_spiking_recall_margin_derisk.py`), and its artifact
(`research/findings/raw/_metacog_spiking_recall_margin_derisk/6seed_results.json`) before building.
The residual is exactly as briefed: the spiking recall-margin agrees with the host confidence formula's
confident/hedge classification on **97.6% (41/42)** of unambiguous cases (host margin outside
`ROLE_CONF_LO/HI`=0.30/0.50) but only **50% (9/18)** of genuinely ambiguous ones, and a `_margin_drive_pA`/
`_cleanup_window` operating-point sweep (300/500/800pA × 120/300/600 steps) "shows the Spearman correlation
staying in a narrow band regardless of integration time" — i.e. adding raw window LENGTH did not resolve it,
because that sweep still only ever read one snapshot at the end of a longer window.

One characterization in the task brief does **not** appear in the PARTIAL itself: a "type-2 AUC ~0.825 overall"
figure. `.venv-rag/bin/python tools/rag/rag_search.py "metacognition confidence recall margin accumulation" 5
--corpus finding` surfaces only this PARTIAL as the top hit (score 2.72; the next metacog hits are the unrelated
GateB production-readout and margin-comparator findings) and the PARTIAL's own metrics are Pearson
r=0.959/Spearman rho=0.954 and the 97.6%/50% agreement split — never a type-2 AUC. `0.825` (and `0.67-0.82`)
appear instead in `metacog_production_organ.py`'s own module docstring, describing a DIFFERENT, ARCHITECTURALLY
DISTINCT mechanism (the metacog WORKSPACE's own settled NMDA-balance read, `_second_order_metacog_monitor_derisk`,
E1 GO) — a sibling system downstream of this one, not this de-risk's own measurement. This is a loose citation in
the brief, not a mis-stated residual: the actual claim this task asks to verify (ambiguous-band agreement ~50%,
unresolved by a window-size sweep) is accurate, so building proceeded as directed rather than stopping.

## The mechanism: read the SAME deliberation's trajectory and time-to-bound, not just its endpoint

`RFPhasorComposer._spiking_margin_accum(scores, lesion=False)` (new, `research/runners/rf_phasor_composer.py`)
drives the IDENTICAL setup `_spiking_margin` already uses (the SAME cached Izhikevich concept bank, the SAME
`_margin_drive_pA`-normalized input, the SAME `_cleanup_window`=120 steps, the SAME `lesion=True` uniform-drive
substitution) — but instead of reading the winner-vs-runner-up spike-count margin ONCE off the final accumulated
count, it reads the RUNNING (cumulative-so-far) state at every step of the SAME window, at no extra simulation
cost (the loop already ran; only the intermediate reads were previously discarded). It returns `final_margin`
(bit-identical to `_spiking_margin`'s own return — verified below), `mean_trajectory_margin` (the time-INTEGRATED
normalized margin across the whole window — the accumulated-evidence read), and `steps_to_bound`/`bounded` (the
first step, if any, at which a decision criterion is reached — the time-to-bound read). Purely additive: nothing
calls this new method by default, `_spiking_margin` itself is byte-unchanged, and it is **not** threaded into
`_cleanup_all_score_stats`, `OneBrainComposer._block_role_scores`, or `mean_role_confidence`'s preference chain —
explicitly unwired, per the task's own scope (de-risk only).

## An instrument bug found and fixed en route, not banked

<!--derived-->

The first version of `steps_to_bound` put the criterion on the NORMALIZED ratio `_spiking_margin` itself reports
(`(top1-top2)/(top1+eps)`), reasoning it was "the same form, just read early." This was WRONG, and the wrongness
was directly measurable: on a hand-built nearly-tied 4-way competition (all four candidates driven within ~2% of
each other's input current — the deliberate synthetic "ambiguous" case), the ratio spuriously touched 1.0 (a
"maximally confident" reading) at step 12, purely because the FIRST spike of the whole run happened to land on
one candidate before any other had fired — at that instant `top1=1, top2=0`, a ratio of exactly 1.0 that reflects
firing-ORDER coincidence, not a real rate differential. The SAME lesioned (uniform-drive, no differential at all)
version of the SAME competition produced an IDENTICAL spurious step-12 crossing. A genuinely decisive (20:1)
competition's ratio-based crossing (step 17 in this same test) was therefore, absurdly, LATER than the tied
competition's — the opposite of the intended signature.

The fix: put the criterion on the RAW spike-COUNT DIFFERENCE (top1-top2) instead of the ratio —
`_margin_accum_count_bound=3`. Measured directly (same hand-built competitions, `_cleanup_window`=120): the
decisive competition's raw difference grows LINEARLY and MONOTONICALLY once its winner starts firing (0 for 15
steps, then 1,1,...,2,2,...,7 by the end — crossing a difference of 3 partway through the window and never
retreating), while the nearly-tied competition's raw difference **never exceeds 1 spike at any point in the
entire 120-step window** (both candidates fire at nearly equal rates the whole time). A raw count-difference bound
is exactly the decision variable a real race/accumulator model puts a threshold on (Usher & McClelland 2001's
leaky-competing-accumulator; Ratcliff's diffusion boundary) — non-decreasing once a lead opens, not renormalized
by a small, noisy denominator — and this fix is now the shipped criterion (see the constant's docstring in
`rf_phasor_composer.py` for the full measurement). `mean_trajectory_margin` (the mean of the RATIO across the
whole window, not a first-crossing test) was never affected by this — averaging over 120 steps already dilutes a
single-step coincidence far more than a first-crossing detector does, confirmed by the fact it correctly separated
the two hand-built cases (0.867 decisive vs. 0.058 tied) even under the flawed ratio-bound version.

## Unit tests: the mechanics are pinned independent of substrate results

<!--derived-->

`tests/test_spiking_margin_accum.py` (9 tests, all passing, SIM_BACKEND=numpy): `final_margin` is bit-exact
against `_spiking_margin`'s own return on the same scores/lesion (2 seeds); a clearly-separable competition
reaches the count-bound criterion in under half the window and sustains a trajectory mean at least half its final
value; the nearly-tied competition NEVER bounds within the window and its accumulated evidence is materially
lower; the load-bearing lesion (uniform drive on an otherwise-decisive competition) collapses both the trajectory
mean and the bound-crossing (never reached, matching the tied signature); degenerate inputs (`V<2`, all-zero
scores) match `_spiking_margin`'s own zero verdict; and a grep-backed test confirms `_cleanup_all_score_stats`
never calls the new method (unwired by construction, not merely by convention).

## CPU smoke (3 seeds, numpy): a genuine type-2 test in the ambiguous band

<!--derived-->

`research/runners/_metacog_accumulation_to_bound_derisk.py` reuses the rank-9 PARTIAL's own validated
composer/capture machinery UNCHANGED (`build_composer`, `capture_raw_scores`, `_host_mrc`, `FACTS`, `VOCAB` —
reuse-by-import), but where the PARTIAL drew exactly one noise realization per sigma level, this draws a BATTERY
of independent noise draws (n=20 per sigma level, sigma in {1.5, 1.8, 2.0, 2.2, 2.5} — a band informed by the
PARTIAL's own per-seed hedge-crossing table) across 3 seeds (42, 43, 44; 100 trials/seed, 300 total). For every
trial, on the SAME captured per-role raw score arrays (one query, one capture, never a re-run per arm), it reads
BOTH `snapshot_mrc` (mean over roles of `final_margin` — the existing single-snapshot arm) and `accum_mrc` (mean
over roles of `mean_trajectory_margin` — the new arm), plus `frac_roles_bounded` (fraction of roles whose
competition reached the count-bound criterion — the time-to-bound arm). Crucially, correctness here is **genuine
recall accuracy** (`answer == the actually-stored patient "spikes"`), NOT agreement with the host formula (the
PARTIAL's own metric) — a stricter, more direct type-2 test: does the confidence signal predict whether the
recall was actually RIGHT.

**Ambiguous-band results** (host mrc strictly inside `ROLE_CONF_LO`/`ROLE_CONF_HI`=0.30/0.50; type-2 AUC via a
Mann-Whitney rank statistic, `research/runners/_stageA_foundation_honesty_arbiter_derisk.py::_auc`, ties averaged):

| seed | n ambiguous (correct/incorrect) | AUC snapshot | AUC accum (mean-trajectory) | delta |
|---:|---:|---:|---:|---:|
| 42 | 29 (19/10) | 0.574 | 0.600 | +0.026 |
| 43 | 29 (18/11) | 0.306 | 0.444 | +0.139 |
| 44 | 24 (20/4)  | 0.487 | 0.575 | +0.087 |
| **pooled** | **82 (57/25)** | **0.446** | **0.530** | **+0.084** |

**All-condition results** (every trial, not just the ambiguous band; n=168 of 300 trials produced a usable margin
— the rest abstained, which the composer's no-confab moat correctly reports as having no recall competition to
read a margin from, so they carry no signal for either arm): pooled AUC snapshot 0.542, AUC accum 0.586, delta
+0.043. The improvement is directionally the SAME but smaller than in the ambiguous band specifically — matching
the mission's own hypothesis that accumulation should help MORE exactly where a fixed-endpoint snapshot is
weakest.

**The time-to-bound diagnostic (`frac_roles_bounded`) is the STRONGEST of the three signals measured**, not just
an auxiliary: pooled ambiguous-band AUC 0.585 (per-seed 0.618 / 0.525 / 0.631 — all three seeds above chance,
consistently), versus 0.446 (snapshot) and 0.530 (accum mean-trajectory). Mean `frac_roles_bounded` for correct
ambiguous-band trials is 0.553 vs. 0.490 for incorrect ones. This was not the read this de-risk set out to
privilege (`mean_trajectory_margin` was the planned primary read) — it emerged from measuring both, and is
reported because it is the actual strongest result, not folded silently into a blended scalar (see
`_spiking_margin_accum`'s own docstring for why no single blended `confidence_accum` scalar is defined).

**A new, sharper characterization of the residual**: under this de-risk's genuine-correctness ground truth, the
EXISTING single-snapshot spiking margin (`snapshot_mrc`) reads **below chance (AUC 0.446)** in the ambiguous band
— it is not merely imprecise there, it is mildly ANTI-predictive of correctness at this operating point and
sample size. The PARTIAL's own agreement-with-host metric could not surface this because it measures agreement
with a DIFFERENT proxy signal, not predictive validity for ground-truth correctness.

## Honest characterization

<!--derived-->

**Direction: CONFIRMED, consistently across all 3 seeds tested**, for both the accumulated-trajectory-mean read
and (more strongly) the time-to-bound fraction — every seed's accum-family AUC exceeds its own snapshot AUC in
the ambiguous band; none regresses. **Magnitude: MODEST.** No signal measured here reaches strong type-2
discrimination (the ceiling observed is 0.585-0.631, not the ~0.8+ region a robust confidence code would show);
the improvement moves the ambiguous band from mildly-anti-predictive toward weakly-predictive, not to a resolved
state. **Scale: CPU-smoke only.** This is a tiny 5-role/~15-word composer under extreme, sigma-driven synaptic
damage engineered specifically to LAND trials in the narrow ambiguous band — not the real-handler production
composer (`enable_attributed`, `vocab_headroom`, the wider real trace shape the PARTIAL's own calibration checked
once), not GPU, not the mandated 6-seed battery (42/43/44/100/101/102). A genuine 6-seed spiking validation at
production scale could move these numbers in either direction. This is why the verdict is PARTIAL, not GO: a
real, reproducible, consistently-positive-direction signal that has not yet been validated at the scale or
fidelity a production or "closed" claim would need — an honest positive lead, not a manufactured resolution of the
rank-9 residual.

## Scope / honesty notes

<!--derived-->

No `sim/` edit. Changes confined to `research/runners/rf_phasor_composer.py` (additive: `_spiking_margin_accum`
and the `_margin_accum_count_bound` constant; `_spiking_margin` and every existing method are byte-unchanged), a
new runner (`research/runners/_metacog_accumulation_to_bound_derisk.py`), and a new unit-test file
(`tests/test_spiking_margin_accum.py`). Nothing in `metacog_production_organ.py` or `one_brain_composer.py` was
touched — `mean_role_confidence`'s preference chain is unaware this method exists (confirmed by a grep-backed
unit test, not merely asserted). Default-OFF / unwired is therefore true by construction, not by an env-var
gate: the only callers of `_spiking_margin_accum` are this de-risk's own runner and its unit tests. Regression:
`tests/test_rf_phasor_composer.py` + `tests/test_onebrain_spiking_cleanup.py` + the new
`tests/test_spiking_margin_accum.py` together (57 passed, 4 skipped) and
`tests/test_merged_rf_composer_coresident.py` + `tests/test_one_brain_composer_agent.py` +
`tests/test_production_spiking_flags.py` (5 passed, 20 skipped) all pass unmodified — identical pass/skip counts
to the rank-9 PARTIAL's own baseline. Cost-routed to numpy/CPU throughout (`SIM_BACKEND=numpy`); no GPU used (the
GPU is held by the mouth-training run per this session's directive). `research/runners/__init__.py`'s automatic
provenance sidecar stamped `smoke_3seed.json.prov.json` on the cited artifact.

## Ready-to-queue 6-seed spiking validation (deferred, post-mouth-training GPU)

Not launched this session (CPU-only per directive). When the GPU is free, run the identical runner with the full
mandated seed set and an output filename of your choice under
`research/findings/raw/_metacog_accumulation_to_bound_derisk` (this path does NOT exist yet — it will be written
BY the run, so it is deliberately not spelled out as a concrete filename here, to avoid this finding citing data
that does not exist):

```bash
SIM_BACKEND=cupy PYTHONPATH=. python -m research.runners._metacog_accumulation_to_bound_derisk \
    --seeds 42 43 44 100 101 102 \
    --sigmas 1.5 1.8 2.0 2.2 2.5 \
    --n-trials-per-sigma 40 \
    --out research/findings/raw/_metacog_accumulation_to_bound_derisk/PICK_A_NAME_6seed
```

(Route via `tools/gpu_queue.sh` per the cost-routing skill rather than a direct Claude-agent invocation; doubling
`--n-trials-per-sigma` to 40 tightens the ambiguous-band sample beyond this smoke's 82 pooled points. Consider
also widening `--sigmas` slightly per-seed once real seed-specific hedge-crossing points are known, mirroring the
rank-9 PARTIAL's own per-seed table.)

## Next rung (not attempted here)

<!--derived-->

If the 6-seed spiking run reproduces this direction with a materially larger effect (or the `frac_roles_bounded`
time-to-bound read proves the more robust of the two accumulation arms, as this smoke's pooled numbers already
suggest), the next step is threading `margin_accum`/`frac_bounded` as an ADDITIONAL, still default-OFF trace field
through `_cleanup_all_score_stats`/`OneBrainComposer._block_role_scores` (mirroring `margin_spiking`'s own
threading) so `mean_role_confidence` COULD prefer it — a wiring step explicitly out of scope here per the task's
own "de-risk only" instruction. Also worth checking: whether combining `mean_trajectory_margin` and
`frac_roles_bounded` (rather than either alone) improves further, and whether the count-bound constant (3) is
itself well-calibrated at production `_margin_drive_pA`/vocabulary scale rather than only on this smoke's tiny
V<=5 candidate banks.
