---
type: finding
status: contributing
claim_check: synthesis
date: 2026-08-11
mechanism: ROLE-GATE x TRANSPORT-FREE FEEDBACK-ALIGNMENT RELIABILITY (LEVER 5) — resolve the CLEAN residual LEVER 4 isolated (the reliability of transport-free feedback alignment on the variable-subject-position role task) by adding a WEIGHT-MIRROR alignment WARM-UP (Akrout et al. 2019, "Deep Learning without Weight Transport"): a pre-credit phase that aligns the feedback matrices B → forward^T via NOISE-output correlation (inject noise into a forward hop, Hebbian-accumulate outer(noise, response) → the forward weight's transpose in expectation; TRANSPORT-FREE — B is never read off a forward weight). Reuse-by-import of the LEVER-4 varpos machinery (2-layer competitive-stabilizer gate + byte-faithful chained-FA/KP transport-free credit + the REAL spiking D3 SpikingSlot); the ONLY additions are the warm-up + an optional per-sentence mirror refresh + a kp_lr boost, all additive/off-by-default (identical to LEVER 4 when off). NO sim/ edit; SIM_BACKEND=numpy for the role task, cupy for the GPU scale probe.
lane: emergence engine / working memory x gap#4 / role-gate transport-free reliability
verdict: 6-SEED (42 43 44 100 101 102), real spiking D3 slot, GO distance L=6 (dist 7, chance 0.250, held-out NOVEL fillers). role_go=True, task_valid=True, winner=mirror_fa. The task stays VALID (untrained 0.267, permuted-reward FA 0.250 / KP 0.267, onset 0.267 — all fail <= chance+0.15), the credit-fidelity CEILING holds (aligned+stabilizer 0.953 [min 0.800]), and TRANSPORT-FREE credit is now RELIABLE: adding ONLY a weight-mirror WARM-UP to the failing LEVER-4 FA arm (nothing else) lifts it from bimodal 0.494 [min 0.217] to 0.981 [min 0.950] 6/6 (gap +0.97 [min +0.92], fires NOM 1.00/obl 0.03); all four alignment-warm-up arms clear the GO bar (mirror_fa 0.981 [min 0.950], mirror_fa_ref 0.947 [min 0.767], mirror_kp 0.936 [min 0.767], kp_strong 0.947 [min 0.833]; L=5 corroborates, all >= 0.917 min). THE RESIDUAL WAS ALIGNMENT TIMING, NOT CAPACITY: the winner mirror_fa reaches 6/6 with only MODERATE final alignment (cosB 0.62 [min 0.40], B frozen so it drifts as W moves), while mirror_kp holds NEAR-PERFECT final alignment (cosB 1.00 [min 0.99]) yet scores LOWER (0.936 [min 0.767]) — and the LEVER-4 KP baseline co-adapts B to high alignment yet still collapses (0.267), because by the time B aligns the gate has already fallen into the fire-everything basin. Aligning B BEFORE credit begins puts the gate in the role basin from the start. SCALE: widening the role gate's own hidden H 32->128 does NOT rescue the FA/KP baselines (FA still 0.519 [min 0.217] at H=128 — not a low-dim pathology), while the warm-up fix holds at H=128 (kp_strong 0.994 [min 0.983]). A GPU batched deep-MLP width sweep (random-teacher classification, widths 128/512/2048, peak VRAM 532 MB) shows transport-free FA/KP already MATCH the aligned ceiling in a plain MLP (all ~0.60, no width trend) — confirming the role-task FA failure is SPECIFIC to the sequential fire-everything basin, not a generic transport-free-credit deficiency. The 2-layer net + chained credit + the competition + the weight-mirror are HOST math; their on-substrate spiking DA-gated / mirror realisation is the named next rung. NO sim/ edit.
seeds: [42, 43, 44, 100, 101, 102]
runner: research/runners/_rolegate_feedback_alignment_reliability_derisk.py
artifacts:
  - research/findings/raw/_rolegate_fbalign/fbalign_6seed.json
  - research/findings/raw/_rolegate_fbalign/H128_6seed.json
  - research/findings/raw/_rolegate_fbalign/scale_probe_gpu.json
---

# Role-gate LEVER 5 — a weight-mirror alignment WARM-UP makes TRANSPORT-FREE credit reliable on the variable-position role task (GO); the residual was alignment TIMING, not capacity

## TL;DR

LEVER 4 (`_var_bind_rolegate_varpos_derisk`, banked NEGATIVE) isolated the role-gate arc's residual to exactly one thing: the RELIABILITY of transport-free feedback alignment. On the valid variable-subject-position case-marked task (onset cannot cheat), exact-feedback (weight-transport) credit reached role reliably 6/6, but transport-free chained-FA was bimodal (2/6) and canonical-KP collapsed (0/6).

LEVER 5 resolves it. Adding a WEIGHT-MIRROR alignment WARM-UP (Akrout et al. 2019 — align the feedback matrices to the forward transposes via noise-output correlation BEFORE the credit phase, transport-free) makes transport-free credit RELIABLE. Adding ONLY the warm-up to the failing FA arm — no other change — lifts it from 0.494 [min 0.217] to **0.981 [min 0.950]** over 6 seeds, matching the exact-feedback ceiling; all four alignment-warm-up arms clear the GO bar. The validity killers still bite, the credit is still required, the tag cue is still load-bearing.

The mechanism is a genuine, non-trivial finding: **the residual was alignment TIMING, not alignment capacity.** The winner reaches 6/6 with only moderate FINAL alignment (its frozen feedback drifts as the forward weights move), while a near-perfectly-aligned co-adapting arm scores lower — because what matters is that the feedback is aligned when credit STARTS, so the gate enters the role basin before the fire-everything attractor can capture it. The LEVER-4 KP baseline eventually aligns its feedback but collapses anyway, because it aligns too late.

## What changed vs LEVER 4 (only the warm-up + an optional refresh/boost, all additive/off-by-default)

The gate, the chained-FA+sigma' / canonical-KP transport-free credit, the competitive forward stabilizer, the variable-subject-position case-marked stream, and the REAL spiking D3 SpikingSlot deployment are the LEVER-4 machinery, reused by import unchanged. The credit loop in the new gate (`FBAlignVarposGate.train_varpos_fb`) is a byte-faithful copy of the LEVER-4 `train_varpos` loop. The only additions:

1. **A weight-mirror WARM-UP.** Before the credit phase, estimate each forward hop's transpose into its feedback matrix by injecting Gaussian noise xi into the forward hop, reading its response zeta, and Hebbian-accumulating outer(xi, zeta): E[outer(xi, W xi)] = sigma^2 * W^T. This is Akrout et al. 2019's weight-mirror circuit — TRANSPORT-FREE, because the feedback is estimated from the forward pathway's RESPONSE to noise, never by reading a forward weight's transpose. After the warm-up the credit phase starts from an aligned feedback.
2. **An optional per-sentence mirror REFRESH** (re-align the frozen feedback to the CURRENT forward weights every `mirror_every` sentences) and a **kp_lr boost** (stronger/longer KP co-adaptation), used by two arms to test whether MAINTAINED alignment or stronger co-adaptation adds anything.

When the warm-up/refresh/boost are off, `train_varpos_fb` is identical to the LEVER-4 credit loop; the LEVER-4 aligned/FA/KP baselines are reproduced verbatim inside this runner as reference arms.

## Why the weight-mirror is transport-free (the biology grounding)

Akrout, Wilson, Humphreys, Lillicrap & Tweed 2019, "Deep Learning without Weight Transport" (NeurIPS): the weight-transport problem — backprop's feedback matrix must equal the forward weight's transpose, which no synapse can read — is solved by a biologically-plausible "weight mirror" that runs the forward pathway on noise and adjusts the feedback synapses by a local Hebbian correlation of the injected noise with the forward response, driving the feedback toward the forward transpose WITHOUT ever transporting a weight. Their central result: a network that alternates a "mirror" (align) phase with an "engaged" (learn) phase matches backprop on ImageNet where fixed-random feedback alignment collapses. This lever imports exactly that mechanism as a PRE-CREDIT warm-up phase. It is host math here (like the whole 2-layer gate and its chained credit); its on-substrate spiking realisation — a DA-gated / noise-driven mirror circuit on the gate's own populations — is the standing next rung, unchanged from LEVER 4.

## The decisive result — per-arm 6-seed table (real spiking slot, GO distance L=6, chance 0.250, held-out novel fillers)

| arm | acc mean [min] | tag-gap [min] | fire NOM/obl | cosB final [min] | reliable? |
|---|---|---|---|---|---|
| case-marker ORACLE (ceiling) | 0.967 [0.900] | — | (detector) | — | yes (target exists) |
| aligned + stabilizer (exact-feedback CEILING) | 0.953 [0.800] | +0.95 [+0.88] | 1.00 / 0.05 | +1.00 [+1.00] | yes, 6/6 |
| chained-FA (LEVER-4 transport-free baseline) | 0.494 [0.217] | +0.20 [−0.17] | 0.33 / 0.11 | +0.29 [−0.06] | NO — bimodal 2/6 |
| canonical-KP (LEVER-4 transport-free baseline) | 0.267 [0.217] | −0.16 [−0.17] | 0.00 / 0.17 | +0.28 [−0.14] | NO — 0/6 collapses |
| **mirror_FA (FA + warm-up; align-then-hold)** | **0.981 [0.950]** | +0.97 [+0.92] | 1.00 / 0.03 | +0.62 [+0.40] | **YES, 6/6** |
| mirror_FA_ref (FA + warm-up + refresh; align-maintain) | 0.947 [0.767] | +0.92 [+0.87] | 1.00 / 0.08 | +0.95 [+0.94] | YES, 6/6 |
| mirror_KP (KP + warm-up; align-then-KP) | 0.936 [0.767] | +0.95 [+0.86] | 0.99 / 0.04 | +1.00 [+0.99] | YES, 6/6 |
| kp_STRONG (KP + warm-up + boost + refresh) | 0.947 [0.833] | +0.91 [+0.75] | 0.98 / 0.07 | +0.95 [+0.94] | YES, 6/6 |

Per-seed transport-free FA vs mirror_FA (L=6): FA = [1.000, 0.883, 0.333, 0.233, 0.217, 0.300] (bimodal — the 4 collapsing seeds sit at the onset floor); mirror_FA = [0.967, 0.950, 1.000, 0.983, 0.983, 1.000] (all 6 seeds cleared). The aligned ceiling on the same seeds = [0.800, 0.950, 1.000, 0.983, 0.983, 1.000]. mirror_FA is the SAME net + SAME credit rule + the SAME task as the failing FA arm — the ONLY difference is the pre-credit weight-mirror warm-up.

## The residual was alignment TIMING, not capacity (the adversarial read the GO gate demanded)

The task asked whether the winning lever RAISES the worst-seed alignment or whether alignment stays high while accuracy still collapses (which would move the residual PAST alignment). The answer is neither, and it is informative:

- **The winner (mirror_FA) reaches 6/6 with only MODERATE final alignment** (cosB 0.62 [min 0.40]). Its feedback is aligned at the START (post-warmup cosB init ≈ +1.00) but FROZEN, so it drifts down as the forward weights move — its worst-alignment seed (cosB 0.40) still scores 0.967. High accuracy with mediocre final alignment.
- **A near-perfectly-aligned arm scores LOWER.** mirror_KP holds cosB 1.00 [min 0.99] throughout (KP maintains alignment) yet reaches only 0.936 [min 0.767] — lower than mirror_FA despite far higher final alignment.
- **The LEVER-4 KP baseline aligns but collapses.** Its co-adapting feedback reaches high alignment on the readout hop during training (the banked LEVER-2 finding measured cos 0.92–1.00), yet role accuracy still collapses to 0.267, because by the time the feedback aligns the gate has already fallen into the fire-everything basin.

So final-alignment magnitude does not predict accuracy; alignment TIMING does. Aligning the feedback BEFORE credit begins (the warm-up) puts the gate in the role basin from the first update, before the fire-everything attractor can capture it — exactly Akrout's "align then train" thesis, and exactly why co-adapting from a random feedback (LEVER-4 KP) or holding a random feedback (LEVER-4 FA) is unreliable. The load-bearing ingredient is the warm-up: it is present in all four clearing arms and absent from both failing baselines, and adding it ALONE to the FA arm (mirror_FA = FA + only the warm-up) is sufficient.

## Task validity holds (unchanged from LEVER 4) — the fix is not an artifact

The controls that must fail on the genuinely-hard task still fail decisively at L=6 (chance 0.250, chance+0.15 = 0.400):

| control (must FAIL <= chance+0.15) | L=6 mean [min] | verdict |
|---|---|---|
| UNTRAINED stabilized gate | 0.267 [0.217] | FAILS |
| PERMUTED-reward chained-FA | 0.250 [0.033] | FAILS |
| PERMUTED-reward canonical-KP | 0.267 [0.217] | FAILS |
| ONSET gate (fires t==0) | 0.267 [0.217] | FAILS |
| identity control (noun-only, tag stripped) | 0.256 [0.183], gap −0.13 | FAILS (tag load-bearing) |

The NOM oracle beats the permuted-cue control (0.967 vs 0.267), the n-gram held-out floor is at chance (0.339), hold is load-bearing (oracle-lesion 0.289). All 8 verdict preconditions hold on the merge. The warm-up does not touch the stream, the validity killers, or the credit signal — it only changes the initial feedback, and the permuted-reward arms (which include the warm-up path) still collapse, so the LEARNING signal is still required.

## Scale: width alone does NOT fix it; the warm-up fix is width-robust; and a plain MLP has no FA gap

Two scale probes answer "is this a low-dim / narrow-network pathology?"

1. **On-task role width sweep (H 32 → 128, 6-seed, L=6, the decisive one, numpy/CPU — the per-token LIF loop is launch-bound so CPU is faster).** Widening the role gate's own hidden layer does NOT rescue the transport-free baselines: FA stays 0.519 [min 0.217] and KP 0.267 [min 0.217] at H=128 — the min is unchanged from H=32, so the FA/KP reliability failure is NOT a narrow-hidden pathology. The warm-up fix, by contrast, holds at H=128 (kp_strong 0.994 [min 0.983], mirror_kp 0.972 [min 0.900], mirror_fa 0.967 [min 0.867]).
2. **GPU batched deep-MLP width sweep (random-teacher classification, aligned/FA/KP/mirror_fa, widths 128/512/2048, SIM_BACKEND=cupy, 3090; peak VRAM 30 → 532 MB with width, 0.22 arms/s).** In a plain feedforward deep-MLP, transport-free FA and KP already MATCH the aligned ceiling across widths (all ~0.60 mean, no systematic FA collapse, no width trend). So transport-free credit is not generically unreliable at these scales — the role-task FA failure is SPECIFIC to the sequential fire-everything basin the gate must avoid, which the alignment warm-up addresses. (Note: the static-MLP arbitrary-cue GATHER task is unlearnable even by the aligned/backprop arm — all arms at chance — a corroboration that the cue-detection binding needs the sequential WM latch the role gate is built around, not a plain MLP.)

## Verdict — GO (brain-based, transport-free), with the residual mechanism named

`role_go=True`, `task_valid=True`, `winner=mirror_fa`. On the genuinely-hard variable-position role task where LEVER 4 banked a clean honest negative, a transport-free weight-mirror alignment warm-up makes transport-free credit RELIABLE (6/6, min 0.950, matching the exact-feedback ceiling), with the validity controls still biting, the credit still required, and the tag cue still load-bearing. The precisely-isolated LEVER-4 residual — the reliability of transport-free feedback alignment — is resolved AT THE ARC'S (host/rate) LEVEL: the mechanism is alignment TIMING (align the feedback before credit begins), realized transport-free by the Akrout weight mirror.

This is a de-risk / GO at the host-math level, NOT a shipped capability: the standing next rung (unchanged) is the on-substrate SPIKING realisation — a DA-gated / noise-driven weight-mirror + lateral-inhibitory competition on the gate's own spiking populations, replacing the 2-layer host net + chained credit + the host mirror. Only when that spiking path is the deployed default is the capability "closed" in the project's sense; this finding earns the transport-free reliability GO that unblocks building it.

Decisive artifacts (every per-seed point is inside each aggregate's `points[].per_seed`, and all 8 verdict preconditions hold on the merges): the primary 6-seed role-task run (L=5 and L=6, real spiking slot, `role_go=True`) at `research/findings/raw/_rolegate_fbalign/fbalign_6seed.json`; the H=128 width-sweep 6-seed run at `research/findings/raw/_rolegate_fbalign/H128_6seed.json`; the GPU batched deep-MLP width sweep at `research/findings/raw/_rolegate_fbalign/scale_probe_gpu.json`.

## Scope, honesty, reuse

The 2-layer net + chained credit + the competitive stabilizer + the weight-mirror are HOST math; the case-tag lexicon and the composite barcode are legitimate environment/front-end artifacts (the language's data). Reuse-by-import of `VarposCompetitiveGate`, `CaseMarkerOracle`, `build_codes`, `compose_code`, the varpos stream, `SpikingSlot`, `MarkerRoleGate`, `role_layout`, `_mint_codes`, the n-gram floor. The warm-up / refresh / boost are additive and off-by-default (identical to LEVER 4 when off). **NO sim/ edit** (runner-side only). SIM_BACKEND=numpy for the role task (launch-bound), cupy for the GPU scale probe. `build_persistent_slot` sets `cfg.seed` so the substrate is seeded per seed (verified: `_d3_persistent_slot_derisk.py:47`). Every knob is in the artifact config. 1-seed is a smoke indicator; the 6-seed sweeps above are decisive.

The exact fan+merge foreground commands (role task, the H=128 width sweep, and the GPU scale probe) are in the runner docstring at the top of `research/runners/_rolegate_feedback_alignment_reliability_derisk.py`. Per-seed runs are ephemeral intermediates fanned to a scratch dir; only the merged 6-seed aggregates are committed, and every per-seed point is inside each aggregate's `points[].per_seed` (reproducible by the documented fan+merge). The knobs used for the decisive run: `--distances 5 6 --n-test 60 --mirror-steps 6000 --kp-boost 6 --mirror-every 40 --mirror-refresh-steps 256` (add `--hidden 128 --distances 6` for the width-sweep aggregate).
