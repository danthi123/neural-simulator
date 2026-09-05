---
type: finding
status: go
date: 2026-09-05
mechanism: vision-cluster-grouping-decision-spiking-wta (RANK-23 backlog item, decision sub-piece only)
lane: scaffold-retirement backlog rank 23 (research/coordination/scaffold_retirement_backlog.md)
verdict: GO (pure de-risk; no live consumer, nothing wired). A 2-pool spiking lateral-inhibition WTA circuit
  (board #86's reused, unmodified N-pool motif) reproduces the EMERGE-36 vision-identity pipeline's host
  argmax-over-averaged-dendritic-voltages GROUPING decision -- parity 1.00 mean (min 1.00/seed), mis-route
  shuffle flips the reported category as predicted 1.00 of clear-winner trials, a symmetric-drive
  decision-level control abstains 0.90 of trials (no baked-in pool bias), and 84.4% of the winner-margin is
  attributable to the real dendritic signal vs. the symmetric control -- 6 seeds (42/43/44/100/101/102).
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/runners/_rank23_vision_cluster_spiking_wta_derisk.py
  - research/findings/raw/_rank23_vision_cluster_spiking_wta.json
runner: research/runners/_rank23_vision_cluster_spiking_wta_derisk.py
---

# RANK-23 "vision cluster" grouping decision: a spiking lateral-inhibition WTA circuit CAN replace the host argmax (de-risk GO, no live consumer, nothing wired)

## The diagnosis this de-risks (scaffold-retirement backlog rank 23)

`research/coordination/scaffold_retirement_backlog.md` RANK-23 ("vision cluster", LOW/no live consumer/mixed
readiness) groups FOUR distinct host residuals inside `vision_identity_production_organ.py` /
`_emerge36_spiking_perception_pipeline_derisk.SpikingPerceptionProbe` — the fully-spiking "spiking HMAX"
bird-vs-fish identity pipeline:

1. `encode_v1` (the Gabor/V1 front end) is a host numpy matmul (`retina @ W.T`).
2. Feature-selection is a host `argsort` + 70th-percentile cut (building `self.OF`, each object's "active
   feature" set).
3. **The final bird-vs-fish DECISION is a host `argmax` over two averaged apical-dendrite voltages**
   (`SpikingPerceptionProbe.infer`: `dr = {c: mean(vap[PROP[c]]) for c in (0,1)}; best = max(dr, key=dr.get)`).
4. V1 receptive-field structure is a hand-written Gabor formula (self-organization, tracked separately as
   `BRAIN_V1_SELFORG`, itself flagged BLOCKED / must-stay-OFF by the map).

**Verified against current code before building anything** (`bash tools/before_you_build.sh`; RAG corpus check;
direct read of `_emerge36_spiking_perception_pipeline_derisk.py` lines 71-77 and 143-146, 2026-09-05): all four
are still present, byte-for-byte as the map described. Not already fixed.

**Scope decision.** This de-risk attacks item 3 ONLY. Item 1 and item 4 are assigned by the map's own
`retirement_mechanism` text to different, much larger mechanisms (a bridge-resident Gabor pathway + a real
image/camera transport; self-organized RF learning). Item 2 is assigned by the SAME text to "the satdiv/num-den
arc [already] in flight" — board #135's semi-saturating divisive-normalization work
(`research/findings/2026-09-0[1-3]-vision-*satdiv*.md`) — re-attacking that host statistic here would duplicate
that arc's own territory rather than open a clean independent lane. Item 3 is the one the parent task's own
framing names ("the vision cluster/grouping step ... reuse existing visual-cortex + WTA machinery") and the one
with a directly reusable, already-validated WTA primitive sitting one import away: assigning a perceived object
to ONE of the two learned category clusters is exactly a 2-way winner-take-all read.

**No live consumer, verified (not assumed).** `vision_identity_production_organ.py` is wired into
`webapp/server.py` behind `BRAIN_VISION_IDENTITY` (default-ON since 2026-08-26), but that wiring block "ONLY
fires on a visual query that CARRIES a `percept` field (`req.percept`)" — grepped: no caller anywhere in
`webapp/` ever populates `req.percept`; there is no image/camera transport. The default-ON flag is structurally
unreachable today. Per the task, this is a pure de-risk of whether the substrate CAN do the grouping; an honest
negative would have been a fine outcome. **Nothing here is wired into any live path.**

## What was built (additive; `research/runners/_rank23_vision_cluster_spiking_wta_derisk.py`; NO `sim/` edit, NO edit to any existing file)

A brand-new, standalone research module. It imports `SpikingPerceptionProbe` (EMERGE-36, unmodified) for the
upstream pipeline, and `_affect_marker_wta_derisk._build_bridge` / `_pool_rates` / `DRIVE_BASE_PA` /
`DRIVE_GAIN_PA` (board #86, unmodified) for the WTA circuit itself — literal reuse of "existing visual-cortex +
WTA machinery", not a new implementation:

- **`VisionDecisionWTA`** builds a private 2-pool bridge via `_build_bridge(seed, n_pools=2, "vis_dec")` — the
  SAME excitatory-assembly + dedicated fast-spiking-interneuron reciprocal cross-inhibition motif already
  6-seed-GO'd for the affect-marker SELECTION (board #86, 2026-08-28) and, before that, the 2-channel
  SPEAK-vs-STAY-SILENT basal-ganglia race (`_vocal_action_selector_gate`) — same pool sizes (24 exc + 12 FSI per
  pool), same cross-inhibition weight, same `DRIVE_BASE_PA=150 / DRIVE_GAIN_PA=1200` operating point, same
  warmup(60)/washout(40)/run(60)-step protocol via the unmodified `_pool_rates`.
- **The drive.** Each held-out object's two PROP-population averaged apical voltages (`dr[0]`, `dr[1]`) are
  reconstructed by literally replaying `SpikingPerceptionProbe`'s own `_codon` + `_prime_from_winners` steps
  (`infer()` computes this internally but does not expose it before its trailing host comparison), then linearly
  rescaled ([-65, +20] mV -> [150, 1350] pA, calibrated from the pipeline's own observed range: an untaught/
  uncharged PROP dendrite reads ~-61.7 mV, a taught/charged one reads ~+12.7 to +14.0 mV) into a drive current
  per pool.
- **The read.** After washout+warmup+run, the per-pool spike rate is read; the winner must clear the runner-up
  by a dead margin (0.05, the SAME constant board #86 calibrated for this identical pool architecture) or the
  circuit reports "no clean winner". The winning pool's FIXED, never-permuted label (pool 0 = category 0,
  pool 1 = category 1) is the reported grouping decision — replacing ONLY the host `best = max(dr, key=dr.get)`
  line; everything upstream of `dr` (Gabor/V1, the coincidence-column pooler, the codon->inheritance) is
  untouched.

## Verification (`_rank23_vision_cluster_spiking_wta_derisk.py`, 6 seeds {42,43,44,100,101,102}, numpy backend, epochs=40 matching production default)

**(A) Parity (PASS, 1.00 mean, 1.00 min/seed).** On every held-out object across all 6 seeds, the spiking WTA's
decision matched the host `argmax`'s decision (36/36 held-out reads: `host_acc = wta_acc = parity = 1.00` on
every one of the 6 seeds).

**(B) Mis-route / shuffle anti-cheat (PASS, 1.00 of clear-winner trials).** Swapping which physical pool
receives which category's drive (pool 0 <- `dr[1]`, pool 1 <- `dr[0]`) at the SAME fixed, never-permuted
pool->category readout label flips the reported category to the predicted opposite on every one of the 36
clear-winner trials, 6/6 seeds. This is the proof the decision genuinely rides which physical assembly won the
spiking race, not a host formula secretly blind to the mis-wiring.

**(C) Symmetric-drive decision-level control (PASS, 0.90 mean abstain rate; per-seed 0.70/0.90/1.00/1.00/0.90/0.90).**
Feeding the WTA an artificial `dr` with both categories forced EQUAL (10 trials/seed, values drawn across the
full calibrated range) — no signal to break symmetry with — produces a clean "no winner" verdict on 90% of
trials on average, with NO seed showing a majority false-winner rate. This is the analog of board #86's own
decision-level lesion, scoped correctly: the pipeline's POOLER-level lesion (coincidence detection off) already
collapses `_codon()` to an empty set BEFORE `dr` is ever computed — upstream of the step this runner changes —
so re-testing it would exercise nothing new; the symmetric-drive control is the one that actually stresses the
WTA circuit itself for a pool-0-always-wins bias, and finds none.

**(D) Attribution (PASS, `tools.lab.attributable_to`, the gap#5 discipline).** Mean real-signal winner-margin
across 6 seeds = 0.152546; mean symmetric-control margin = 0.023808 -> **84.4% of the winner-margin is
attributable to the real dendritic signal**, 15.6% also present in the symmetric control (a small residual
driven by per-pool heterogeneity draws, expected and non-dominant — well clear of the `attributable_to` 50%
floor that would flag a proxy-dominated measurement).

**Verdict artifact:** `research/findings/raw/_rank23_vision_cluster_spiking_wta.json`, GO on all four
preconditions defined BEFORE the run: parity mean >=0.90 (got 1.00) AND min/seed >=0.75 (got 1.00) AND
shuffle-flip mean >=0.85 (got 1.00) AND symmetric-abstain mean >=0.60 (got 0.90) AND `attributable_to` >=0.5
(got 0.844).

## Honest residuals and scope (named, not claimed closed)

1. **This is a de-risk GO, not "closed"/"wired"/"integrated" (per `docs/TERMS.md`).** RANK-23 as a whole is
   untouched in production: no flag was added, nothing in `vision_identity_production_organ.py` or
   `webapp/server.py` changed. There is genuinely no live consumer to wire this into today (verified above) —
   the honest status is "a runner-level GO", nothing more.
2. **Only item 3 of RANK-23's 4-part cluster is addressed.** Items 1 (`encode_v1` host matmul), 2 (argsort +
   70th-percentile feature-selection — deliberately left to the in-flight satdiv/board-#135 arc so as not to
   duplicate it), and 4 (hand-written Gabor RF / `BRAIN_V1_SELFORG`, BLOCKED) remain host/unresolved. "Vision
   cluster" as a whole is NOT retired by this finding.
3. **The task the pipeline solves is an easy, well-separated 2-way distinction by construction** (the trained
   winner's dendritic read is ~+13 mV vs. the loser's fixed ~-61.7 mV, a ~75 mV gap) — EMERGE-36's own 6-seed GO
   already required held-out accuracy >=0.85 with a comfortable margin over its scramble/lesion controls. This
   de-risk answers "can a spiking WTA replace the host comparison on THIS existing, already-solved decision",
   which is the correct question for a scaffold-retirement item; it does not newly demonstrate spiking WTA
   solving a harder discrimination than the host pipeline already does.
4. **`DR_LOW`/`DR_HIGH`/`DECISION_DEAD_MARGIN` are calibrated to THIS pipeline's own observed dendritic range**
   (a one-time seed-42 measurement, both intact and pooler-lesioned), not independently re-derived per seed —
   consistent with how board #86's own `DEAD_MARGIN` was calibrated once for its pool architecture and reused.
5. **No image/camera transport exists** — the deeper, larger item the map calls "the true ceiling" for vision
   overall. Nothing in this de-risk moves that forward; it was explicitly out of scope for RANK-23's own item 1.

## Reproduce

```
SIM_BACKEND=numpy .venv/bin/python -m research.runners._rank23_vision_cluster_spiking_wta_derisk --demo --seed 42
.venv/bin/python -m research.runners._rank23_vision_cluster_spiking_wta_derisk \
    --seeds 42 43 44 100 101 102 --out research/findings/raw/_rank23_vision_cluster_spiking_wta.json
```
