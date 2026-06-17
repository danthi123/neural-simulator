# The learned binder's weights can be learned by a LOCAL rule (not host Adam) — 6-seed unanimous GO

**Date:** 2026-06-17 (the biology-faithfulness frontier: remove the last host shortcut from the on-bridge learned binder)
**Status:** **GO, 6 seeds unanimous** (42, 43, 44, 100, 101, 102). A biologically-plausible **local** learning rule —
a **fixed-random encoder + a local delta-rule (Widrow-Hoff/LMS) decoder**, the Neural-Engineering-Framework
principle — matches (slightly beats) host gradient descent for the binder read-out, with no backprop through the
bundle. CPU/numpy; reuse-by-import; NO `sim/` edit.
**Runner:** `research/runners/_phaseB_localrule_readout_derisk.py`
**Raw:** `research/findings/raw/_phaseB_localrule_readout.json`

## The question

The on-bridge learned spiking binder (Steps 1–2, 6-seed GO) has a brain-based *forward* path (LIF ON/OFF bind,
spiking composite, retrieval, abstention), but its weights — the filler projection `W_F` and the read-out
cleanup `W_O` — were trained off-substrate by **Adam** (a non-local gradient method the brain does not run). This
de-risk isolates the *learning rule* question (is it biologically plausible?) from the on-substrate question: can
a **local** rule replace Adam and still reach parity?

The biology-grounded candidate (Eliasmith-Anderson Neural Engineering Framework; the same principle as the
project's spiking NEF cleanup): a **fixed-random encoder** `W_F` (a random projection = tuning curves, *not*
learned) + a **local delta-rule decoder** `W_O` learned by `dW_O = −lr·outer(act, err)` — a three-factor
`pre × post-error` rule, the on-substrate-realizable form. No backprop through the bundle; only the final linear
read-out is learned, by a local error-correcting rule.

## Result — 6 seeds, D_h=256, the bundled systematicity protocol

| arm | bundled held-out (6-seed mean) | per-seed |
|---|---|---|
| **FIXED-Wf + DELTA-Wo (NEF, LOCAL)** | **1.000** | [1.000 × 6] |
| ADAM-both (the reference) | 0.974 | [1.000, 0.889, 1.000, 1.000, 1.000, 0.952] |
| FIXED-Wf + ADAM-Wo (control) | 0.810 | [0.967, 0.907, 0.648, 0.651, 0.878, 0.811] |

- **local ≥ 0.85× Adam: 6/6** (GO bar). The NEF local rule reaches **103%** of the Adam read-out, unanimous.
- **Systematicity holds:** the NEF single-binding held-out is **1.000** (it generalizes to held-out (role,
  filler) combinations, not memorization).

## Reading it

- **The host-Adam shortcut is removed for the binder read-out, at the learning-rule level.** A fixed-random
  encoder + a local delta-rule decoder — both biologically standard (random tuning curves + a three-factor
  error-correcting synapse) — learns the binder as well as backprop, with no non-local gradient.
- **The local rule is not a compromise — it's better-conditioned here.** The fixed-encoder + **Adam**-decoder
  control lands at only 0.810, well below the fixed-encoder + **delta**-decoder's 1.000. For a linear read-out
  with a random encoder (the NEF/reservoir-computing sweet spot), the local LMS rule converges faster and cleaner
  than Adam at the same budget. (This is also why the earlier trimmed-budget run showed joint-Adam collapsing
  while NEF held — Adam needs more steps; the delta rule does not.)
- **Encoder learning is not needed.** Fixing `W_F` to a random projection (no encoder learning) loses nothing —
  the NEF result equals or beats full joint-Adam. So only the **decoder** must be learned, by the local rule.

## Honest scope — what's de-risked vs what remains

- **De-risked here:** the *learning rule* is biologically plausible (local, no backprop) and reaches parity. This
  removes the conceptual blocker that "the binder only works if trained by backprop."
- **Remaining (the deeper piece):** *realizing* the delta rule **on the spiking substrate** — the three-factor
  `pre × post-error` update at the synapse (the bridge already has reward-modulated / three-factor machinery). The
  rule is now known to work; wiring it on-substrate (so the weights are learned in spikes, not numpy) is the next
  step. Single-attribute learned binding already works on real spikes (2026-06-16, 0.833), so the read-out (a
  supervised linear cleanup) is the tractable on-substrate target.

⇒ Together with the binding build (Steps 1–2 GO: the forward path is brain-based) and this result (the weights
can be learned by a local rule), the on-bridge learned binder's idealization is nearly fully removed: brain-based
binding + retrieval + abstention, and a biologically-plausible local learning rule. The last residual is the
on-substrate realization of that rule.

## Reproduce
```bash
SIM_BACKEND=numpy python -u -m research.runners._phaseB_localrule_readout_derisk \
    --dh 256 --seeds 42,43,44,100,101,102
```
