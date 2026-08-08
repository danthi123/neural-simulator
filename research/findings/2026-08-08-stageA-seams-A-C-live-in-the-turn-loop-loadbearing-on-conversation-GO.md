---
type: finding
status: contributing
date: 2026-08-08
mechanism: stageA-conversation-integration-seams-A-C-live-in-turn-loop
lane: E-language
runner: research/runners/_stageA_full_integration_derisk.py
artifacts:
  - research/findings/raw/lanes/stageA/stageA_full_integration_seamslive_s42.json
  - research/findings/raw/lanes/stageA/stageA_seamslive_6seed_aggregate.json
---

# Stage-A seams A (forward-model world-model) + C (graded-affect ladder) routed LIVE in the multi-turn loop — load-bearing ON the conversation, regression stays GO (single-seed smoke)

The banked structural integration (main `d37e3f48`) put the two seams on the ONE merged bridge
DEFAULT-OFF byte-identical and load-bearing only on their OWN isolated reads. This finding does the
FUNCTIONAL wiring the verifier declared as the next step: with `co_resident_forward_model=True` +
`co_resident_affect_ladder=True`, EACH TURN of `run_multi_turn_loop` is routed through the seams, so A and C
are load-bearing ON THE CONVERSATION OUTPUT — proven by a conversation-lesion battery (lesion each faculty →
the turn output changes vs a matched sham). The sacred 6/6 GO regression is preserved: moat 475/475, FM4 0
law-flips, default-off byte-identity True. NO `sim/` edit.

Runner: `research/runners/_stageA_full_integration_derisk.py` (seams live by default; `--no-seam-a` /
`--no-seam-c` roll a seam back to OFF). Seam byte-identity probe:
`research/runners/_stageA_seam_integration_probe.py`. Artifacts:
`research/findings/raw/lanes/stageA/stageA_full_integration_seamslive_s42.json` (canonical single-seed run,
provenance-sidecarred) and `research/findings/raw/lanes/stageA/stageA_seamslive_6seed_aggregate.json`
(the 6-seed verdict summary).

## SEAM-A LIVE — the forward-model world-model on a NOVEL turn
<!--derived-->
On a novel `(s,a)` turn (a compositional (state,action) query the moat abstains on): the agent drives the
co-resident `fm_reservoir` slice with the turn's `(s,a)` (a fixed per-word sensory embedding → the fixed-random
`W_in` → the reservoir), reads the per-neuron spike-COUNT off `cp_firing_states`, and a ridge read-out (a
DECLARED host shortcut — the reservoir SPIKES are the brain-based content) trained over the STORED facts decodes
a predicted `s'` + a top1–top2 read-out MARGIN. The margin folds into `g_eff` TIGHTENING-ONLY
(`g_eff = max(g_eff, g0 + k*(1-margin))`, `g0=0.06`, `k=0.30`) — a low-confidence forward model can only make
the brain MORE cautious, never less. The decoded `s'` is offered as a certainty-TAGGED simulation channel
(`"my forward model predicts 'X' for this novel case (margin M); I have not observed it"`) that NEVER enters the
cue-match candidate set and NEVER writes `cp_rf_w_*` — so the no-confab moat still ABSTAINS on the unstored
factual cue (475/475 by construction). Seed-42 transcript:
- T2 `novel_query`: `"what does big run ? -- my forward model predicts 'south' for this novel case (margin 0.04); I have not observed it"` (moat abstains; arb_ask wins).
- T4 `novel_query`: `"what does look run ? -- my forward model predicts 'river' for this novel case (margin 0.12); I have not observed it"`.

## SEAM-C LIVE — GRADED-affect coloring on every turn (replaces the binary latch)
<!--derived-->
Each turn drives the staggered-bistable ladder from the turn's appraised valence (host-fed on the shared
appraisal bus — the inherited honest-negative) and reads the tone/forthcomingness NEURALLY as
`rate(aff_pos_readout) - rate(aff_neg_readout)` through the `affect_out` gate. The GRADED differential sets a
MULTI-LEVEL tone (a warmth staircase L−3…L+3), REPLACING the P0.3 binary latch. Seed-42: the positive-mood
known turns read at tone level **L3** (`"warmly, gladly apple big cat ; also big, cat"`), and the graded
differential (>0) persists across the intervening neutral novel turn (affect persistence via the ladder's
within-pool NMDA latches). Neutral novel turns read differential ~0 → neutral tone.

## The conversation-lesion battery (the acceptance test) — real vs matched sham, ON THE TURN OUTPUT
<!--derived-->
Deltas are measured on the TURN OUTPUT (predicted-content / tone-level), not the isolated read; the matched sham
is an off-target intervention of the same kind on the OTHER faculty's (array-disjoint) pathway.
- **Faculty A** — REAL lesion (silence the `fm_reservoir`): predicted `'cat'` → `None` (the predicted-content
  channel vanishes; the turn reverts to plain abstention; `g_eff` reverts to the `g0` floor). SHAM
  (off-target: clamp `affect_out=0`, which the fm content path does not traverse): predicted `'cat'`
  UNCHANGED. The moat abstains under both (invariant).
- **Faculty C** — REAL lesion (clamp `affect_out=0`): tone **L3 → L0** (flat/ungraded) while the ANSWER `'cat'`
  is UNCHANGED. SHAM (off-target: silence the fm, which the ladder read does not traverse): tone **L3**
  UNCHANGED.
- `battery_ok = True`: each REAL lesion changes the turn output; each matched SHAM does not.

## Regression stays GO with the seams LIVE (single seed 42, numpy/CPU)
<!--derived-->
- (a) SINGLE-BRIDGE — `composer._merged is bridge`; all faculties + both seam slices co-resident in ONE process. GO.
- (b) COMPOSES-LIVE — 4-turn transcript: graded-colored honest answers on known turns, curiosity wh-asks +
  forward-model predictions on novel turns, moat holds, affect persists. `graded_tone=True`,
  `fm_content_on_novel=True`. GO.
- (c) FM4 LIVE — yoked high-arousal affect mis-colored tone on 15/15 below-assert candidates, flipped **0** to
  assert under the g_eff law; the naive path flipped 15/15 (teeth present). GO.
- (d) MOAT LIVE 475/475 — abstains on every unstored cue under a positive high-arousal mood; 0 false-accepts,
  0 manufactured. GO.
- (e) NO-PIECE-BREAKS-ANOTHER — every pairwise interaction holds; the 3-way arbiter arbitrates (3 distinct
  correct winners; contention collapses on the inhibition lesion, 90.7% attributable). GO.
- (f) DEFAULT-OFF BYTE-IDENTITY — faculty slices appended after the composer rf slice; composer-index
  thresholds byte-identical. GO.

## The rf_w byte-identity sub-check was vacuous — now genuine
<!--derived-->
The seam-probe's `_rf_w_sha` returned `None` (wrong attribute names `cp_rf_store_*`, AND the moat store is a
SPARSE CSR passed to a dense-array hasher that raised into a swallowing `try`) → `rf_w_identical=None` →
treated as a PASS. Fixed: the moat store is `cp_rf_w_re` / `cp_rf_w_im` (written by `rf_set_complex_weights`,
allocated only after a fact is stored), so the probe now stores the SAME facts on both builds FIRST and hashes
the CSR restricted to the pre-existing `[:n, :n]` block (the appended-LAST slice grows N, so a full-matrix hash
differs even though every stored synapse sits at an rf index < n — the `_conn_block_sha` mechanism). With the
fix the check GENUINELY runs: `moat_store_written=True`, `rf_w=True` for both seam A (`n_off 24961 → 25261`)
and seam C (`24961 → 25531`). A byte difference in the moat store would now fail the check.

## 6-seed regression WITH seams LIVE — 6/6 GO (numpy/CPU, seeds 42 43 44 100 101 102)
<!--derived-->

| seed | verdict | moat | FM4 law-flips | FM4 naive-flips | byte-identical | battery_ok | graded-tone | fm-content-on-novel |
|-----:|:-------:|:----:|:-------------:|:---------------:|:--------------:|:----------:|:-----------:|:-------------------:|
| 42   | GO | 475/475 | 0 | 15 | True | True | True | True |
| 43   | GO | 475/475 | 0 | 13 | True | True | True | True |
| 44   | GO | 475/475 | 0 | 11 | True | True | True | True |
| 100  | GO | 475/475 | 0 | 16 | True | True | True | True |
| 101  | GO | 475/475 | 0 | 13 | True | True | True | True |
| 102  | GO | 475/475 | 0 | 16 | True | True | True | True |

The sacred 6/6 GO holds with both seams routed LIVE: moat 475/475 every seed, FM4 g_eff-law abstain→assert
flips = 0 every seed (the naive affect-into-confidence path flips 11–16 = the check has teeth), default-off
byte-identity True every seed, and the conversation-lesion battery passes every seed. No seam had to be rolled
back. Command:
`for s in 42 43 44 100 101 102; do SIM_BACKEND=numpy python -m research.runners._stageA_full_integration_derisk --seed $s --out research/findings/raw/lanes/stageA/stageA_full_integration_seamslive_s$s.json; done`

## Honest-negatives (declared)
- A's content decode is a ridge argmax over reservoir spike-counts — a DECLARED HOST SHORTCUT (same status as
  the composer render / `OnBridgeLSM._fit_slots`); the brain-based content is the reservoir SPIKES; the spiking
  synaptic read-out is the target to biologize.
- A's per-word `(s,a)` token embedding is a host-provided sensory encoding (same status as `W_in` / a retinal
  render), not the read-out.
- C's appraisal is host-fed on the shared bus (the inherited STEP-2 boundary); the tone/forthcomingness TOKEN
  render is host (the STEP-3 boundary). The affect SIGNAL is the neural ladder differential through `affect_out`.
- Single-seed SMOKE; the parent runs the 6-seed sweep (seeds 42 43 44 100 101 102).
