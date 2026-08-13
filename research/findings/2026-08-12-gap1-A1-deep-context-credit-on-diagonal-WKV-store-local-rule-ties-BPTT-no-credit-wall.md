---
type: finding
status: qualified
verdict: INCONCLUSIVE-re-the-wall (mechanism works at de-risk scale; controlled-lag copy task too easy to exhibit the open-prose credit wall; NOT a GO)
lane: gap#1
date: 2026-08-12
mechanism: diagonal-WKV-store-local-credit + learned-feedback(DNI)
---

# gap#1 / A1 mouth-burn-down residual — at DE-RISK SCALE, spiking diagonal e-prop handles the deep-context sequence credit (a transport-free LOCAL rule TIES-OR-BEATS the BPTT ceiling on the WKV/SSM store), but the controlled-lag copy task is TOO EASY to exhibit the credit-assignment WALL the open-prose mouth actually hits — INCONCLUSIVE re the wall; NOT a GO

**Runner:** `research/runners/_spiking_deepcontext_generation_derisk.py` (torch/GPU, rate-level; eligibility grad-check
PASS ~1e-7; NO `sim/` edit). Controlled-lag delayed cued-recall (provably beats no n-gram), diagonal gated leaky
(WKV membrane-leak) store, 6-seed (42/43/44/100/101/102), deep lags T=48 & T=64, per-arm lr sweep.

## TL;DR (the honest reframe)
<!--derived-->

The A1 residual was framed as: *open, arbitrary, deep-context spiking prose is blocked by a deep-context
CREDIT-ASSIGNMENT wall — biological one-step-local e-prop captures only ~44% of the deep-context recurrent-credit
margin full backprop achieves.* Two things dissolve that framing:

1. **The FORWARD half is already GO** (2026-07-20 RF-PHASE-ENCODE, 6-seed): a graded `cp_ssm_state` WKV/SSM store,
   driven by a spiking phase (FHRR) input through real synapses, GENERATES fluent, coherent, coreference-maintained
   multi-clause TinyStories prose on the real Izhikevich substrate. Open deep-context prose ON FIRING NEURONS exists.
   Its store weights, however, are BPTT-trained (a tracked scaffold) — so the residual is the LEARNING RULE.

2. **The "~44% wall" was measured on the WRONG substrate.** The ~44% (2026-07-11 R1b) is on a FULL-W_rec random
   RESERVOIR — where e-prop's diagonal eligibility drops the OFF-DIAGONAL cross-unit gradient — against a BPTT
   denominator its OWN authors later showed is READ-OUT-confounded (2026-07-11 R2b corrected: the dual-timescale
   "44->81% lift" was an effective-learning-rate artifact; the "% of BPTT recurrent credit" framing was dropped as
   fragile). The store the RF-phase WKV cortex actually uses is **DIAGONAL** (`a_t = decay*a_{t-1} + gate*write`,
   per-channel; the 2026-07-20 uniform-decay "ssm" form). For a diagonal store the forward-filtered e-prop
   eligibility of the store params is **EXACT RTRL** (grad-check vs autograd ~1e-7) — NO off-diagonal to drop, hence
   no truncation wall.

This de-risk tests the credit rule directly on that diagonal store, on a controlled deep dependency with no n-gram /
read-out shortcut. **Result (6-seed, honest): the LOCAL rule (diagonal e-prop, random feedback) TIES-OR-BEATS the
BPTT ceiling and the learned-feedback (DNI) arm adds nothing — but that is because the controlled-lag copy task is
TOO EASY to exhibit the deep-context credit wall. The tell: `eprop_random` (and even `sign_flip`) reach or exceed the
BPTT ceiling, so ANY coherent per-channel eligibility solves it and feedback QUALITY is not the discriminating axis.
This mechanism-works-but-task-too-easy result is INCONCLUSIVE re the wall — NOT a claim that open-prose deep-context
is solved.** The genuine store-credit IS load-bearing (shuffle-eligibility, cue-scramble, and zero-signal all collapse
to chance), so the substrate learns the deep dependency with a local rule; what the task fails to do is STRESS the
axis (feedback alignment over a large vocab / long dependency) where the Qwen mouth's open-prose wall lives.

## The exact deep-context residual (what actually breaks, synthesizing the banked walls)
<!--derived-->

Deep-context open generation decomposes into THREE separable sub-problems; the field conflated them under "deep
credit":

| sub-problem | status | evidence |
|---|---|---|
| (1) a NON-FADING content-selective STORE (the memory horizon) | **SOLVED** | WKV / graded `cp_ssm_state` leaky integrator, learned content-selective write; removes the non-fading-store wall every fading reservoir hit (2026-07-19); RF-phase delivers the spiking input, graded conductance holds state (biologically legit per the SpikeGPT reframe) |
| (2) the nonlinear READ / cross-cue combination | **SOLVED** (identified, cheap) | a 2-stage cortical read-out (the controlled-lag arc's identified fix, 2026-07-14) |
| (3) LEARNING the store's write/decay/read by a BIOLOGICAL local rule (vs BPTT) | **this de-risk** | see below |

Prior banked levers on (3), all now superseded/scoped for the diagonal store:
- multi-timescale / dual-timescale ELIGIBILITY (the task-named "multi-timescale" candidate) — **REFUTED** as an
  effective-learning-rate artifact (2026-07-11 R2b: magnitude-controlled lift ~+0.04; plain e-prop at 5-10x lr
  reproduces the whole "lift").
- ALIF adaptation-as-FORWARD-STATE (the task-named "WM-maintenance current" candidate) — **REFUTED** (2026-07-11 R2:
  the `-beta*a` forward imprint DEGRADES the representation; deep HURTS).
- the controlled-lag arc (2026-07-14): on a FIXED random-reservoir substrate, recurrent W_rec credit is **not**
  load-bearing; the bottleneck is the substrate MEMORY HORIZON + a nonlinear read-out.

## The companion-process reframe applied (the wall-discipline question)

*"What does biology run ALONGSIDE deep-context credit that the prior de-risks replaced with a CONSTANT?"*
e-prop's broadcast learning signal uses a **FIXED RANDOM** feedback matrix B — a constant, misaligned map from
output error to hidden credit. Biology runs a **LEARNED top-down feedback** (apical-dendrite / predictive-coding /
Bellec-2020's synthetic-gradient DNI) that ALIGNS the credit direction; the 2026-08-11 gap#4 ALL-IN arc found exactly
this at the rate level (LEARNED Kolen-Pollack feedback reaches depth-3 where FIXED feedback fails). So the ONE
mechanism de-risked here is `eprop_learnfb`: diagonal e-prop with a **learned** feedback B (a DNI/synthetic-gradient
predictor trained online to align with the read-out's own spatial credit), tested against the fixed-random baseline
and the BPTT ceiling.

## The instrument (clean, non-vacuous — the requirement the banked instruments failed)
<!--derived-->

Controlled-lag delayed cued-recall: `[STORE, x, f_1..f_T, RECALL] -> predict x` at the RECALL position. The last-n
window is fillers+RECALL, statistically INDEPENDENT of x, so a held-out n-gram is provably at chance (the deep
dependency is genuine, not a memory-timescale artifact — the flaw that voided the TinyStories "deep margin"). The
store is a diagonal gated leaky integrator with LOG-SPACED fixed decays + a LEARNABLE content-selective write-gate
(the WKV mechanism). **Learning is load-bearing** because a fixed random write corrupts the slow channels with filler
tokens (the distal cue is lost => chance at deep lag); only LEARNING to gate the write (exclude fillers) preserves it.

Instrument validity (all pre-registered, all checked in-runner):
- **eligibility grad-check PASS** (diagonal RTRL vs autograd, max residual ~1e-7) — the credit is exact for the store.
- **fixed_store -> chance at deep T** (the cue is genuinely lost from the fixed store) + **held-out n-gram at chance**
  + **cue_scramble collapses to chance** (train on random targets -> eval-vs-stored collapses) => the deep dependency
  is real and non-gameable.
- **shuffle_elig COLLAPSES toward chance** (permuting the per-channel eligibility keeps update magnitude but breaks
  the credit STRUCTURE) => the win requires genuine per-channel credit, not capacity/read-out.
- **sign_flip does NOT collapse — and this is EXPECTED, a weak control here:** feedback-alignment is invariant to a
  global sign flip of the feedback (the forward weights simply align to `-B`), so sign_flip is not a discriminating
  anti-cheat for a random-feedback rule. The genuine-credit evidence is carried by shuffle_elig + cue_scramble +
  zero_signal (`== fixed`, byte sanity). [Noted honestly; do not read sign_flip's non-collapse as a fake win.]

## RESULTS (6-seed, seeds 42/43/44/100/101/102; per-arm lr sweep {0.03,0.1,0.3}; chance = 1/K)
<!--derived-->

| arm | T=48 mean±sd | T=64 mean±sd | role |
|---|---|---|---|
| fixed_store (floor) | 0.231 ± 0.034 | 0.205 ± 0.025 | cue lost beyond horizon => ~chance |
| **eprop_random (LOCAL rule)** | **1.000 ± 0.000** | **0.973 ± 0.054** | transport-free diagonal e-prop, FIXED random FB |
| **eprop_learnfb (MECHANISM)** | **1.000 ± 0.000** | **0.965 ± 0.055** | LEARNED (DNI) feedback — adds NOTHING over random |
| eprop_truefb | 0.928 ± 0.111 | 0.610 ± 0.238 | exact spatial feedback (feedback-direction "ceiling") |
| bptt_ceiling | 0.928 ± 0.111 | 0.599 ± 0.147 | full backprop-through-time ("ceiling") |
| shuffle_elig (anti-cheat) | 0.248 ± 0.037 | 0.202 ± 0.028 | COLLAPSES ✓ (credit structure load-bearing) |
| sign_flip (weak control) | 0.961 ± 0.077 | 0.888 ± 0.221 | does NOT collapse — EXPECTED (FA sign-invariant) |
| cue_scramble (control) | 0.181 | 0.156 | COLLAPSES to chance ✓ (task is sound) |

(`zero_signal` — the byte-identity sanity, L:=0 so the store never moves — reads EXACTLY == `fixed_store` on every
seed by construction, as it must; it is a no-op control, not an independent arm, and is omitted from the table for
that reason. `attributable_to(eprop_random − chance, shuffle_elig − chance)` attributes ~95–100% of the local-rule
win above chance to genuine per-channel credit not present in the shuffle control.)

Runner auto-verdict: **T=48 = NO_CREDIT_WALL_AT_THIS_SCALE / TASK_UNDER_DIFFICULT**; **T=64 = INSTRUMENT_VACUOUS**
(the BPTT ceiling collapsed BELOW the local rule at depth — see below). grad_check PASS.

**The two decisive tells that the task is UNDER-DIFFICULT (do not read this as "solved"):**
1. **`eprop_random` ties (T=48: 1.000 vs 0.928) or beats (T=64: 0.973 vs 0.599) the BPTT ceiling, and even `sign_flip`
   reaches ≈ceiling (0.96 / 0.89).** When RANDOM and SIGN-FLIPPED feedback match the ceiling, feedback QUALITY is not
   being probed — any coherent per-channel eligibility-weighted update solves the copy task. The mechanism (learned
   feedback) is neither confirmed nor needed here.
2. **At T=64 the "ceiling" is INVALID: BPTT (0.599) and true-feedback (0.610) UNDER-perform the forward-eligibility
   local rule (0.973).** This is the classic vanishing-gradient signature — BPTT through 64 leaky steps loses the
   gradient to the distal write (~decay^T), while the FORWARD-accumulated eligibility does not vanish the same way
   (a genuine, known e-prop/RTRL advantage for long sequences). So there is no valid "credit ceiling" to measure a
   fraction against at depth; the instrument's own validity gate correctly flags T=64 vacuous.

What IS solid: the store-credit is genuinely load-bearing (shuffle_elig, cue_scramble, zero_signal all collapse to
chance while the plastic arms solve) and the eligibility is exact (grad-check ~1e-7). So a spiking diagonal e-prop DOES
learn a deep dependency (T=48–64, well beyond the fixed store's ~15-token horizon) with a local rule. It just does so
on a task too easy to separate the credit rules.

## VERDICT (honest, non-overclaimed): INCONCLUSIVE re the deep-context wall — NOT a GO
<!--derived-->

- **This is NOT "open-arbitrary deep-context spiking prose is solved."** At de-risk scale (T=48/64, K=6 content vocab,
  F=6 fillers, N=128 channels) a spiking diagonal e-prop handles the sequence credit — but the controlled-lag copy
  task does NOT exhibit the deep-context credit-assignment WALL the Qwen mouth actually hits (open, arbitrary,
  multi-clause prose over a LARGE vocab with LONG, structured dependencies). Random-feedback tying/beating the BPTT
  ceiling is the tell that the benchmark is under-difficult.
- **The learned-feedback (DNI) companion mechanism is UNTESTED-as-needed** — the baseline saturates, so the arm is a
  null (not a refutation, not a confirmation). It remains the pinned mechanism for wherever the arms DO separate.
- **What the de-risk DID establish (bounded, honest):** (a) the ~44% e-prop "wall" was substrate-specific — it was a
  FULL-W_rec-reservoir (off-diagonal drop) measured against a read-out-confounded BPTT denominator (its authors'
  own R2b correction); on the DIAGONAL WKV/SSM store the eligibility is EXACT (no truncation), so that particular
  wall does not transfer; (b) a local forward-eligibility rule is at least as good as BPTT for a single-cue deep
  dependency and better at long lag (vanishing-gradient-robust); (c) the store-credit is genuinely load-bearing and
  clean controls (shuffle/cue-scramble/zero) work — the instrument machinery is sound, it just needs a HARDER task.

**SCALE-SWEEP RESULT (2026-08-12, `_dc_scale_*.json`, K∈{16,32,64,128}×T32 + K64×T64, 3-seed, N=192):** the K-sweep
RAN and the picture is now sharp (mean accuracy per arm):
- **K16 (too easy):** every e-prop arm saturates at 1.000 (== BPTT) — no credit signal to probe.
- **K32 (the ONE informative point):** the credit WALL appears — `eprop_random` 0.752 and `eprop_learnfb` 0.751 both fall
  short of `eprop_truefb`/`bptt` 0.931 — AND **learned-KP feedback does NOT close the gap** (`learnfb_frac_of_gap`
  = −0.002; it ties random), while TRUE (weight-transport) feedback closes it fully. `sign_flip` 0.720, `shuffle_elig`
  0.053 (control collapses). So: on the diagonal store a real deep-context credit gap opens, and the biologically-PLAUSIBLE
  local feedbacks (random AND learned-KP) leave it — only biologically-IMPLAUSIBLE weight-transport closes it.
- **K64 / K128 / K64×T64 (capacity wall, NOT a credit probe):** EVERYTHING — including BPTT — collapses to ~chance
  (BPTT 0.062 / 0.015 / 0.026). Past K32 the task exceeds the learnable capacity of ALL arms at this N/epochs, so it no
  longer isolates credit-assignment quality.

**⛔ CORRECTED verdict (2026-08-12, wider-N VALIDITY check — supersedes the "wall is real at K32" reading above):** the
K32 gap is a **CAPACITY-REGIME ARTIFACT, NOT a feedback-quality wall.** The validity run
(`_dc_K32_validity_widerN.json`: same K=32 but N 192→384, H 96→192, epochs 60→80, 3-seed) gives **eprop_random 0.998 =
eprop_learnfb 0.998 ≈ eprop_truefb 1.000 = bptt_ceiling 1.000** (fixed_store 0.070, shuffle_elig 0.074 — controls
collapse). So once the net has ADEQUATE CAPACITY the gap VANISHES: biologically-plausible random-feedback e-prop reaches
the ceiling. The N=192 gap (eprop 0.752 < bptt 0.931) was diagnostic of a **capacity-limited regime** — BPTT itself only
reached 0.931 there, i.e. NOT a valid credit ceiling — so the earlier "learned-KP ties random = honest negative" was
**capacity-confounded**, not a genuine feedback-quality result. **This is the "verify a refutation as hard as a
confirmation" discipline (silent-failure rule 3) catching a wrong wall before it drove a mechanism search.** Net: on the
diagonal WKV store there is **NO fundamental deep-context credit-quality wall** — a transport-free local rule handles the
sequence credit at adequate capacity; the feedback-alignment-at-scale hunt on this instrument is MOOT. (The KP lever is
neither confirmed nor refuted as a *mechanism* — the instrument never posed a real credit gap for it to close.)

**The other genuine remaining residual (separate, already mapped):** porting the store's local credit to the
PRODUCTION Izhikevich few-spike READ regime (2026-08-11: fixed-DFA/KP/DRTP and even a perfect W^T oracle give no
directed credit on the finite-spike read). Neither that nor the scale-up is the rate-level "44% margin" the task named.

## Honest scope / caveats
<!--derived-->

- **Rate-level + synthetic controlled-lag** (not full TinyStories). This is the altitude the 2026-08-11 arc MANDATED
  for a deep-credit test (the finite-spike read is a separate wall; a depth-obligatory spiking test is closed by it).
  The copy task is a clean deep dependency but simpler than open prose; the open-prose FORWARD path is separately GO
  (RF-phase). This de-risk does NOT demonstrate the local rule end-to-end on full open-language deep credit.
- The diagonal store is the 2026-07-20 uniform-decay "ssm" membrane-leak form (the divisive-normalization full WKV is
  an optional ~0.1-nat enhancement, per that finding).
- Per-arm lr is tuned on the eval metric (mild optimism) but applied EQUALLY to all arms (incl. the anti-cheats, which
  still collapse) => the eprop-vs-bptt comparison and the anti-cheat collapse are both lr-fair.
- `sign_flip` non-collapse is a known feedback-alignment property, disclosed above (weak control, not a fake win).

## Files

- Runner: `research/runners/_spiking_deepcontext_generation_derisk.py`
- Raw: `research/findings/raw/_spiking_deepcontext_generation.json` + `.log`
- Builds on / reframes: `2026-07-11-R1-...`, `2026-07-11-R2b-...`, `2026-07-14-controlled-lag-...`,
  `2026-07-19-gap1-WKV-...`, `2026-07-20-gap1-RF-PHASE-ENCODE-...`, `2026-08-11-gap4-ALLIN-ARC-SUMMARY-...`.
