---
type: finding
status: no-go
date: 2026-08-07
mechanism: gateB-stage2o-learning-gated-FIXF-commit-integration-threshold
backend: numpy
runner: research/runners/_vocal_gateb_stage2o_learning_gated_commit.py
builds-on: 2026-08-07-gateB-stage2n-accumulate-then-commit-NMDA-integration-closes-730705.md
grounded-in:
  - research/findings/2026-08-07-gateB-stage2n-accumulate-then-commit-NMDA-integration-closes-730705.md
  - research/findings/2026-08-07-gateB-stage2m-bg-output-homeostat-inverts-thalamus-but-necessary-not-sufficient.md
  - research/findings/2026-08-07-gateB-stage2k-commit-WTA-release-selects-730705-but-cannot-express-at-test.md
artifacts:
  - research/findings/raw/gateb_stage2o_learning_gated_commit/smoke_730705_numpy.json
  - research/findings/raw/gateb_stage2o_learning_gated_commit/sweep_probe_730705.json
  - research/findings/raw/gateb_stage2o_learning_gated_commit/sweep_fixe_730705.json
  - research/findings/raw/gateb_stage2o_learning_gated_commit/sweep_fullreset_xinh1.0.json
---

# Gate B Stage 2o: a learning-gated FIX-F commit-integration threshold CANNOT close 730705 — the commit WTA is a hard bistable latch with two learning-BLIND regimes, and the (real, learning-gated) thal-ordering signal sits BELOW its flip margin. 730705 is a conclusively-characterized heterogeneity boundary; Gate B stands at ≥5/6.

## Verdict (NO-GO — outcome (ii): no separating window exists; a first-class boundary result)

The Stage-2o hypothesis was: the drive from a LEARNED str_d1_1 (~286) and a woken-UNLEARNED
str_d1_1 (~124) differ ~2.3×, so instead of Stage-2n's binary cross-inhibition de-latch (which
unmasked the unlearned channel = a shortcut), a LEARNING-GATED FIX-F THRESHOLD — raise the commit
NMDA-reverberation ignition threshold (scale both commit pools' `cp_izh_k` down, target-blind) —
would let the LEARNED drive cross the commit-integration bound while the UNLEARNED drive does not.
**Measured: no such threshold exists, for a structural reason, and neither do the two adjacent
readout-stage surpasses (an onset entry-state reset; any veto strength).**

## Why the commit threshold cannot separate (argmax scale-invariance — measured)

The commit winner is `argmax(motor_0, motor_1)`. Scaling BOTH commit pools' excitability down
(`commit_k_scale`<1) shrinks both channels but PRESERVES the relative ordering, so it cannot flip
the winner. Cascade sweep on 730705 (fix_e off, xinhib×0.1, `sweep_probe_730705.json`):

| commit_k | LEARNED commit / motor / act1 | UNLEARNED commit / motor / act1 | separates |
|---|---|---|---|
| 1.0 | [184,504] / [378,759] / 8/8 | [233,416] / [463,624] / 7/8 | no |
| 0.5 | [15,88] / [36,246] / 7/8 | [15,102] / [34,282] / 7/8 | no |
| 0.3 | [4,27] / [15,125] / 7/8 | [4,37] / [14,164] / 7/8 | no |

The UNLEARNED bridge picks action 1 at EVERY threshold (7/8), identically to LEARNED — at
commit_k 0.5 the unlearned commit_1 (102) even EXCEEDS the learned (88). The de-latch, not a
threshold, decides the winner.

## The learning-gated signal is REAL but lives at the THALAMUS, below the WTA flip margin

With FIX E (Stage-2m BG-output homeostat) the thalamic aggregate separates by learning
(`sweep_fixe_730705.json`): LEARNED thal=[215,**228**] (thal_1 leads → correct), UNLEARNED
thal=[215,**198**] (thal_1 < thal_0 → favors action 0). The learned D1 advantage IS present at
the BG output — a +26-spike thal_1 lead. **But the commit WTA cannot read it out gradedly.**

## The commit WTA is a hard bistable latch — two regimes, both learning-blind (the structural fact)

Sweeping the veto strength (xinhib) WITH FIX E and even WITH a full selection-circuit onset
entry-state reset (equalise thal, gpi, commit AND commit_fs membrane at onset — the Stage-2m
closing-stack lead, now CODED as a TRN-like reset), on the trained-vs-untrained cascade:

| operating point | LEARNED commit / act1 | UNLEARNED commit / act1 |
|---|---|---|
| xinhib 1.0 (full veto) + reset | [734,8] / **0/8** | [724,9] / **0/8** |
| xinhib 0.5 + reset | [723,0] / 0/8 | [729,0] / 0/8 |
| xinhib 0.1 (de-latch) + reset | [26,163] / 7/8 | [32,174] / 7/8 |

- **Strong veto (xinhib ≥0.5), even with the entry-state head-start removed:** commit_0 ignites
  first and LATCHES → action 0 wins on BOTH the learned and the unlearned bridge (0/8), despite
  the learned thal_1 (228) leading thal_0 (200). The +26 lead is below the WTA's bistable flip
  margin.
- **De-latch (xinhib ≤0.1):** commit_1 always wins on BOTH bridges (the Stage-2n shortcut).
- **No intermediate regime tracks the thal ordering.** The onset entry-state reset does NOT create
  one — resetting thal/gpi/commit/commit_fs to their per-channel means leaves commit_0's structural
  ignition advantage intact.

So across the full readout-stage knob set — commit-integration threshold (`commit_k_scale`), onset
entry-state reset, AND veto strength — there is NO operating point where the commit winner reflects
the learning-gated thal ordering. The commit competition is bistable; its flip is decided by the
veto strength (a target-blind constant), not by the +26-spike learned advantage.

## Anti-cheat outcomes at the best swept operating point (full train→test pipeline, numpy)

<!-- FILLED FROM smoke_730705_numpy.json -->

## Anti-cheat that PASSES

**Byte-identical when off.** FIX-O OFF (`accum_on=False`) reproduces the Stage-2k base
(fix_c+fix_d) exactly on the byte-seeds (mismatch `{}`), so the Stage-2j/2k GO is unaffected.

## Banked method + honest boundary

BANKED (refuted): a learning-gated commit-integration THRESHOLD (raising the commit
NMDA-reverberation ignition bound) cannot close 730705, because the commit winner is a
scale-invariant argmax over a hard bistable WTA — shrinking both commit pools preserves the winner,
and no veto strength or onset entry-state reset makes the winner track the (real, +26-spike)
learning-gated thal-ordering signal. The two commit regimes are learning-BLIND: strong veto →
always action 0, de-latch → always action 1.

**730705 is a conclusively-characterized heterogeneity boundary.** Its str_d1 policy is correctly
learned (str_d1_1 ~286 ≫ str_d1_0 ~104) and FIX E surfaces that advantage at the thalamus
(thal_1 > thal_0 by 26 spikes), but the extreme initial-condition asymmetry of this specific seed
leaves that advantage below the cortical commit WTA's bistable flip margin — and the WTA cannot be
made to read it out gradedly without a de-latch that also passes the unlearned channel (the
Stage-2n shortcut). We have now legitimately exhausted the readout-stage options (Stage 2k WTA
release, 2l soft-WTA, 2m BG-output homeostat, 2n NMDA accumulate-then-commit, 2o learning-gated
threshold + entry-state reset + veto sweep). **Gate B stands at ≥5/6 — a first-class result.**

## Reproduce (numpy, orphan-proof)

```bash
export PYTHONPATH=$PWD SIM_BACKEND=numpy
# Full train→test smoke (byte-identity + sweep + full-pipeline + acq-lesion legitimacy):
.venv/bin/python -m research.runners._vocal_gateb_stage2o_learning_gated_commit --mode smoke \
  --smoke-seeds 730705 --byte-seeds 730703 730705 --commit-k-grid 1.0 0.5 0.3 \
  --out research/findings/raw/gateb_stage2o_learning_gated_commit/smoke_730705_numpy.json
# The decisive cheap instrument (does a threshold window exist?) with FIX E + entry-state reset:
.venv/bin/python -m research.runners._vocal_gateb_stage2o_learning_gated_commit --mode sweep \
  --legit-seed 730705 --fix-e --onset-reset --xinhib-scale 1.0 --commit-k-grid 1.0
# Byte-identity when off (all_byte_identical must be true -> GO protected):
.venv/bin/python -m research.runners._vocal_gateb_stage2o_learning_gated_commit --mode byte
```
