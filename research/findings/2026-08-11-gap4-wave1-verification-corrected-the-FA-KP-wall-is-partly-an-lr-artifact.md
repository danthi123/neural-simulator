---
type: finding
status: contributing
date: 2026-08-11
mechanism: deep-credit-on-spikes
lane: gap#4 ALL-IN (wave-1 verification synthesis)
verdict: ADVERSARIAL-VERIFY CORRECTION of the gap#4 wave-1 "first cracks". The enter-the-regime results (FF, DECOLLE) are REAL and triply reproduced — a local, transport-free rule builds selective features in every layer of a deep spiking net and beats the OPTIMAL frozen reservoir. But the "cracks a wall where top-down credit COULD NOT enter" framing is RETRACTED — at a FAIR per-arm learning rate the chained transport-free FA/KP ALSO enter at N=3/4 (FA 0.84–0.93, KP 0.84–0.90). The 2026-08-02 "chained FA/KP collapse to majority-class at N≥3" wall is therefore PARTLY an lr-divergence artifact at the shared lr, NOT (only) a property of the credit rule. Combined with Q5 (obligatory-depth-3 unconstructible) and depth_contributes=False on XOR, NOTHING in wave-1 demonstrates DEEP credit — everything tested is depth-2-solvable and everything (local rules AND fair-lr FA/KP) solves it. The gap#4 deep-credit question is still OPEN and now REDIRECTED to a fittable genuinely-deep task + a per-arm-tuned FA/KP baseline.
artifacts:
  - research/findings/2026-08-11-gap4-forward-forward-local-contrastive-ENTERS-the-deep-spiking-learning-regime-where-topdown-credit-could-not-6seed.md
  - research/findings/2026-08-11-gap4-DECOLLE-local-readouts-deep-spiking-GO-6seed-enters-the-learning-regime-where-topdown-collapses.md
  - research/findings/raw/_gap4_ff/aggregate_gap4_ff.json
  - research/findings/raw/_gap4_decolle/decolle_N3_s42.json
verification: workflow wrufiei6u (4 adversarial skeptics + adjudicator), MERGE-WITH-EDITS for both GOs
---
<!--derived--> This is a SYNTHESIS/verification finding: every number below is QUOTED from the two wave-1 findings
cited above (and their raw artifacts) or from the `wrufiei6u` verification's re-runs. All values are derived/quoted.

# gap#4 wave-1 verification — the "first cracks" are real as enter-the-regime results, but do NOT crack the deep-credit wall; and the wall itself is partly an lr artifact

## Why this finding exists

The gap#4 ALL-IN assault's wave-1 produced two convergent GOs — Q1 spiking **Forward-Forward** and Q4 **DECOLLE** —
each reported as "the first crack in the deep-spiking credit wall: a LOCAL transport-free rule gets a deep (N=3,4)
spiking net into the learning regime WHERE the top-down chained FA/KP rule collapses to majority-class." Before merging,
a 4-skeptic adversarial verification (workflow wrufiei6u) reproduced both and probed the crux. Verdict:
**MERGE-WITH-EDITS** — the enter-the-regime result is real; the wall/uniqueness framing is inflated. This finding is the
authoritative corrected record. **It corrects claims I had already reported to the owner as "first cracks" — the
correction is the deliverable, per the honesty boundary.**

## What is CONFIRMED (real, triply reproduced — do NOT re-derive)

On the SAME trainable LIF-SNN forward + SAME depth-2 XOR→threshold task, both local transport-free rules genuinely
reshape the deep hidden layers and beat the OPTIMAL-ridge frozen reservoir (the strong floor, not the weak frozen):

<!--derived-->
| candidate | held-out (N=3 / N=4) | vs OPTIMAL-ridge reservoir | deep-layer signature | anti-cheats |
|---|---|---|---|---|
| Q1 Forward-Forward | 0.780 / 0.771 | **+0.16** (0.623 / 0.615 floor) | every layer above majority (`n_weak_coupling=0`) | permuted→chance; only hidden weights update |
| Q4 DECOLLE | 0.926 / 0.941 | **+0.30 / +0.33** (0.623 / 0.615 floor) | every hidden layer ridge-selectivity ~0.95 vs reservoir ~0.55 | shuffled-target→chance; no readout-transport |

Both are byte-identically reproducible, both bite their anti-cheats, neither edits `sim/`. **The real, bankable claim:
a local, transport-free rule (per-layer contrastive goodness, or per-layer fixed-random readout targets) builds
task-useful class-selective features in EVERY layer of a deep spiking net and beats the optimal random reservoir.** That
is a genuine R3-reframe result (local rules > fixed reservoir on a depth-2 task).

## What is RETRACTED (the load-bearing inflation the verification caught)

1. <!--derived--> **"Cracks a wall where top-down credit COULD NOT enter" / "uniquely capable" — RETRACTED.** A skeptic re-ran the
   chained transport-free FA and KP arms through the SAME `_train_snn_arm` machinery at a FAIR per-arm learning rate
   (lr 0.01–0.02 instead of the shared 0.05 both findings used) and **they too enter the regime at N=3 and N=4**: FA
   0.84–0.93, KP 0.84–0.90, both beating the reservoir. The reported "chained FA/KP collapse to majority-class (modal
   single-class) at N≥3" was an **lr-divergence artifact at the shared lr**, not the credit rule failing. So neither FF
   nor DECOLLE is uniquely capable, and the "first crack where top-down could not" framing does not hold.

<!--derived-->
2. **"Deep credit / depth is obligatory" — RETRACTED.** FF's own numbers: at N=3 the BEST single hidden layer reads
   0.789 while the full accumulated stack reads only 0.780 (N=4: 0.782 vs 0.771) → `depth_contributes=False`, a single
   layer is as good as the stack. DECOLLE's own scope note: XOR is depth-2-obligatory, the extra layers are redundant.
   And DECOLLE trains each layer by its OWN local target — no credit flows THROUGH depth at all. So neither
   demonstrates credit through genuinely-obligatory depth.

## The bigger implication — the 2026-08-02 "chained FA/KP wall" is partly an lr artifact

The wall this assault was built to surpass — `2026-08-02-gap4-depth-rescue-untestable-on-spikes-...` ("the chained
multi-hop transport-free FA/KP does NOT leave majority-class at N≥3; FA/KP byte-identical at chance 0.45–0.54 — the
degenerate-dynamics fingerprint") — was measured at a shared/untuned learning rate. The wave-1 verification shows that
at a fair per-arm lr the SAME chained FA/KP **do** leave majority-class and beat the reservoir at N=3/4 on XOR. So the
"located wall" is, on this task, **substantially a per-arm-lr-tuning artifact**, not a hard property of chained
transport-free credit on spikes. This is the most consequential output of wave-1: it says the assault's foundational
premise ("top-down chained credit has no purchase on deep spikes") was over-stated by an instrument/tuning miss — the
exact "the instrument is part of the emulation / a proxy we replaced with a constant" failure class the workflow warns
about (the constant here = a single shared lr across arms with very different scale). It does NOT prove the wall is
entirely absent — it proves the wall was not fairly measured, and must be re-measured with per-arm tuning.

## Honest synthesis with Q5 — nothing in wave-1 tests DEEP credit

- **Q5** (`2026-08-12-gap4-obligatory-depth3-...NEGATIVE`): an obligatory-depth-3 task is NOT constructible on this
  substrate — whenever a depth-3 model generalises, a matched-width depth-2 matches it. Depth is never OBLIGATORY here.
- **Wave-1 (this finding):** on the (depth-2-obligatory) XOR task, FF/DECOLLE enter + beat the reservoir, but so does
  fair-lr FA/KP, and depth does not contribute.
- **⇒ Everything tested in wave-1 is depth-2-solvable, and everything solves it.** The gap#4 DEEP-credit question —
  credit assignment THROUGH a genuinely-deep (obligatory ≥3) spiking net with a local transport-free rule — is
  **untested**, not cracked. The "cracks" are real enter-the-regime results; they do not bear on deep credit.

## The redirect (what wave-2 must do — this is the corrected next action)

1. **Per-arm-tuned FA/KP baseline across tasks + depths (HIGHEST value).** Re-measure the 2026-08-02 wall with a fair
   per-arm lr sweep for every arm. Does the chained transport-free FA/KP wall survive fair tuning, or was it an
   artifact throughout? This decides whether gap#4 has a wall at all on the current substrate. Runner:
   parameterize `_gap4_bptt_snn_chained_fa_transport_free_derisk.py::_train_snn_arm` with a per-arm `--lr` grid.
2. **A fittable genuinely-deep task.** Q5 showed obligatory-depth-3 is unconstructible as a *matched-width
   generalisation* gate. Build a task where surrogate-BPTT CAN fit AND depth is load-bearing (a fan-in-2 compositional
   hierarchy avoiding the parity/fan-in traps), then run the FF/DECOLLE-vs-tuned-FA/KP comparison there.
3. **The Izhikevich on-bridge port** (`2026-08-02-gap4-FA-convergence-is-the-onbridge-credit-root-cause-...`): the
   FA-convergence root cause differs on point-neuron Izhikevich — the wall may be real THERE even if it is an artifact
   on the LIF surrogate net. This is where a genuine substrate wall (if any) most likely lives.

De-prioritized by this correction: running more local rules (SoftHebb, CwComp) on XOR — they will all "enter" a
depth-2 task, which no longer distinguishes anything. Keep any in-flight ones as data points; do not queue more.

## Artifacts (the raw records these quoted numbers derive from)

- `research/findings/raw/_gap4_ff/aggregate_gap4_ff.json` — the FF 6-seed aggregate (held-out, per-layer, reservoir floor).
- `research/findings/raw/_gap4_decolle/decolle_N3_s42.json` — a DECOLLE per-seed record (selectivity, reservoir, arms).
- The two wave-1 findings (cited in frontmatter) carry the full per-seed tables + their own raw-artifact lists; the
  `wrufiei6u` per-arm-lr FA/KP re-runs are the verification workflow's transcript (scratchpad `verify_gap4_ffdecolle.js`).

## Brain-based + scope

All arms' weight updates are host-computed bookkeeping (the SAME shortcut status as BPTT/FA/KP), but the candidate
rules are LOCAL + transport-free (genuine spiking/synaptic biologization candidates). Runner-side; NO sim/ edit;
`cfg.seed`/pure-numpy seeding verified. This finding VOIDS the "uniqueness/first-crack" clauses of the two wave-1
findings (their correction blocks point here); it does NOT void their enter-the-regime + reservoir-beat measurements,
which stand.
