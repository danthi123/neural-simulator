# Generator P — predictive-coding top-down: honest TERMINAL NEGATIVE (the generative-production line concludes; the validated asset stands)

## TL;DR

P — the evidence-justified deep branch (a net-new top-down
predictive-coding layer that *predicts the next concept given the
sequence-so-far*, the order-sensitive signal the recognition-only
substrate provably lacks) — **FAILED its pre-registered held-out
gate.** Per the design's explicit pre-registration, this is the
**terminal, decision-relevant** conclusion of the self-contained
generative-production line: across six maxed-integrity, honestly
propagated attempts (Inc-1/2/3, G1, G1.5, P), self-contained,
locally-trained generative *production* — the sim emitting novel
order-correct propositions through its own substrate, judged by its
own comprehension, with no external teacher/corpus/templates — does
not work on this substrate/hardware under the no-cheating/local
constraints, even with the most biologically-principled mechanism
that directly targeted the precisely-diagnosed cause. The project's
robust, multi-seed, anti-cheat-validated **trustworthy grounded
continual memory with no-confabulation abstention is untouched,
no-harm-re-proven throughout, and stands as the deliverable.** This
is an honest scientific boundary, not a project failure.

## The pre-registered gate result (FIXED bars, never touched)

`song_g1_gate.py --mode p` on the trained P checkpoint (epoch 59),
held-out props only, sidecar-frozen P-regime floor `g1_abstain=85.0`
(NOT 650, NOT G1's 72.0, NOT trajectory's 46.0, NOT recomputed),
mode MATCH asserted, `meta_smoke=False`, bars
`_G1_MARGIN=0.10`/`_G1_ABS_FLOOR=0.5` untouched:

| held-out prop | true | best_perm | gate_cleared | top_rate | verdict |
|---|---|---|---|---|---|
| int=4 `old hard` | 0.000 | 0.000 | N (39.0 < 85.0) | 39.0 | FAIL |
| int=5 `ride smell` | 0.000 | 0.000 | N (47.0 < 85.0) | 47.0 | FAIL |

Aggregate (`g1_verdict` on the means): mean_true 0.000,
mean_best_perm 0.000, 0/2 cleared → **GATE: FAIL**.

## What P showed that was genuinely new (and why it still does not generalize)

P is **not** another flat-zero negative — it is the first attempt in
the arc with a real order-learning mechanism, and that mechanism
demonstrably works *in isolation*:

- **Pure-core proof:** the `PredictiveCoder` learns order-correct
  generation from its own internal prediction error — `rollout`
  reproduces a learned ordered proposition `[2,6]` (FD-verified
  exact gradients; reviewed SOUND). G1/G1.5's controllers could not
  do this even on paper.
- **First nonzero training signal in the entire arc:** over the
  60-epoch protocol, P's `mean_reward` took values {0.0, 0.25, 0.50}
  and `n_gate_cleared` reached 2 — the trained predictor *sometimes*
  produced order-correct sequences that survived substrate
  realization and cleared the frozen floor. Inc-1/2/3, G1, G1.5 were
  *only ever* 0.0000 / capacity-bound / memorization.

But the training signal was **sparse and unstable** (0 at most
checkpointed epochs; 0.25 at epoch 49; final epoch 59 back to 0.0),
and — decisively — on the **held-out** propositions the trained P
produces top-rates 39/47, *below* the control-calibrated frozen
floor 85.0, with true-order score 0.000. The order-learning the
predictor achieved on TRAIN propositions **does not generalize to
unseen propositions through substrate realization.** This is the
Inc-3 lesson recurring at a deeper level: an intermittent training
signal is not held-out generalization. P closed the most ground of
any approach and is the right principled mechanism, but the learned
in-predictor order does not survive being realized through the
write-only substrate and read back by the substrate's own
comprehension at the pre-registered bar.

## The converged conclusion (across the whole arc)

Six attempts, one consistent, precisely-diagnosed cause:

- **Inc-1/2/3** (char-level BPTT): capacity not the bottleneck;
  memorization ≠ generalization.
- **G1** (songbird controller): the substrate's self-comprehension
  judge cannot distinguish order (Step-0 AUC 0.775).
- **G1.5** (trajectory readout): worse (AUC 0.40); its calibration
  also falsified the cold-start hypothesis (G1.6) for free.
- **P** (predictive-coding top-down): the order signal *can* be
  learned in a dedicated predictor (proven in isolation) but does
  not generalize through substrate realization to the held-out gate.

The recognition-only G.20 substrate does not encode, and cannot be
made via these mechanisms to express, recoverable generative
sequence *order* under self-contained/local/no-cheating constraints.
P was the deepest, most biologically-correct, evidence-indicated
mechanism (Rao-Ballard/Friston/Bastos); its honest FAIL on the
pre-registered held-out gate is the terminal, decision-relevant
boundary the design pre-registered. Continuing to spin further
generative variants would be the garden-of-forking-paths /
config-cranking the project's anti-cheat discipline explicitly
forbids; the honest action is to record this boundary and stop the
generative-production line here.

## Anti-cheat discipline (maxed-integrity terminal negative)

Every gate bar (`_G1_MARGIN=0.10`/`_G1_ABS_FLOOR=0.5`) never tuned.
650 never used. The P-regime abstention floor was pre-registered
control-max/AUC-calibrated, frozen to an isolated `song_g1.pc.*`
sidecar, and **never recomputed at gate time**; cross-mode/readout/
smoke sidecar reuse hard-refused. `--mode songbird`/`--readout final`
kept byte-identical (G1/G1.5 reproducible). The LOAD-BEARING no-harm
re-proof PASSED (13/13 validated-known, band excess +0.0, abstention
moat holds) — P's top-down write into concept pools does **not**
regress the validated comprehension path. The pure core was
independently reviewed (no Critical/Important; gradients
FD-verified). A TDD-caught latent spec bug (start-of-sequence
unlearnable) was fixed pre-data at root, the subagent correctly
refusing to fake-pass. The predictor was never config-cranked; the
full pre-registered 60-epoch protocol ran to completion. This is the
strongest possible form of an honest negative: a maxed-integrity,
principled, pre-registered terminal FAIL.

## The robust validated asset is the deliverable (untouched, no-harm-re-proven)

The project's genuinely validated, multi-seed, anti-cheat
contribution stands entirely intact: the **trustworthy grounded
continual memory with no-confabulation abstention** — G.20 sparse
distributed ensemble (160 concepts @ 100% / 320 @ 98.4%,
multi-seed), no catastrophic forgetting (Marr/McClelland CLS),
cross-bridge associative recall, and the no-confabulation moat that
*refuses to make things up* (a property the original target, a small
LLM, does not have). Every P step explicitly re-proved (no-harm
probe, 13/13) that this asset is not regressed. It is the honest,
robust deliverable. The generative-production line is concluded; the
memory line is validated and shippable.

## Files

- `sim/predictive_coding.py` (+ `tests/test_predictive_coding.py`,
  6 tests, FD-verified, reviewed SOUND)
- `research/runners/song_g1_{ignite,train,gate}.py` (`--mode p`
  threaded through the proven isolation/cross-mode-refusal/
  sidecar-frozen-floor machinery; write-only `ignite_prediction`)
- Evidence: `research/findings/raw/g11_bg/song_g1_pc_train.log`,
  `song_g1.pc.ckpt.npz` (+ `.meta.json`: smoke=False, mode=p,
  g1_abstain=85.0), `song_g1_pc_gate.json`, `song_g1_noharm.json`
  (Task-7 re-proof PASS)
- Design/plan: `docs/plans/2026-05-16-generator-P-predictive-coding-{topdown-design,implementation}.md`
- Prior arc: `2026-05-16-generator-increment{1,2,3}-*.md`,
  `2026-05-16-generator-G1-songbird-NEGATIVE.md`,
  `2026-05-16-generator-G1.5-trajectory-readout-NEGATIVE.md`
