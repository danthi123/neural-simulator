# On-bridge learning — burstprop off-bridge probe: the EXACT rule locates the shared-readout wall (FA/clean-gradient is the de-risk)

**Date:** 2026-07-20 · **Status:** informative characterization (the exact E-gated burstprop rule faces the shared-
readout wall on a large-vocab classifier; the clean-gradient FA is the working de-risk of the local rule). Off-bridge,
NO `sim/` edit. First cheap-first probe of the on-bridge fully-spiking local-rule learning frontier.

## Context

The grounded render is learnable OFF-bridge by a biological local rule (FA/KP + SGD, 0.86-0.91 == BPTT; finding
`2026-07-20-wkv-cortex-biological-learning-CLOSE-...`). The on-bridge fully-spiking realization was research-gated:
the RULE is committed (`fused_bdsp_update` = burstprop, verified local + no-transport, `sim/kernels.py:461-493`); the
gate's biggest risk = the shared-readout wall (`2026-07-15-...RUNG3-BOUNDARY`), with the named escape = the GRADED
state + a REDUCED read-out. This probe tested the EXACT committed rule off-bridge before the `sim/` realization.

## The probe — the EXACT burstprop rule form over the graded state (`--credit burstprop`)

Added `--credit burstprop` to `_gap_grounded_wkv_local_readout.py`: the OUTPUT credit is the burst-rate DEVIATION
`dev = B - Pbar*E = E*(P - Pbar)` (the committed kernel's form), `P = sigmoid(logit(P0) - gain*error)`, FA-routed to
the hidden read-out layers. Verify-first (loss must descend) is mandatory.

**Result — the exact rule locates the wall (large-vocab classifier read-out, V=4002):**
- **Non-E-gated** (`dev = P - Pbar`, my first impl): descends one step but DIVERGES over training (ppl → inf) — the
  credit is not zero-sum (unlike the softmax gradient `p - onehot`), so it drifts. An incomplete rule form.
- **E-gated** (`dev = E*(P - Pbar)`, `E ≈ softmax p`, the CORRECT committed form): the `E`-gating (the multiplexing
  invariant — credit only to ACTIVE units, 0 at rest = the P0 moat) shrinks the credit by ~V× (p ~ 1/V for a
  V=4002 classifier), so the usable lr band is NARROW: lr 5e-4 too slow, lr 1.0 ASCENDS (the verify-first assertion
  correctly aborted it). ⇒ the exact E-gated burstprop credit on a LARGE-VOCAB classifier read-out is delicate — this
  IS the shared-readout-wall mechanism (the biological E-gating suppresses credit for a large-vocab softmax).

## Read-out — the honest frontier map

- **The LOCAL RULE is de-risked (via FA/clean-gradient):** my off-bridge close reached grounded 0.86-0.91 with the
  CLEAN gradient (non-E-gated FA/KP) — which escapes the wall exactly because it is not E-gated. The mission-relevant
  capability (biologically-learnable grounded render) stands.
- **The EXACT E-gated committed rule faces the shared-readout wall on a large-vocab classifier** — confirming the
  gate's biggest risk. This is precisely why the gate's rung (ii) specifies a **REDUCED read-out** (the grounded task
  needs only ~50 curriculum words, not the full 4002-way softmax) + the graded/phasor escape.
- **⇒ the on-bridge fully-spiking realization is the genuine DEEP sub-frontier** (the gate's (b)/(c) verdict), NOT a
  naive burstprop port of the full-vocab read-out. The well-specified next steps (a fresh focused arc): (1) a REDUCED
  read-out (curriculum-vocab, ~50-way) where the E-gating is tractable; (2) the additive default-off `sim/` mechanism
  (propagate `cp_ssm_state` as synaptic read-out drive + eligibility); (3) the graded/phasor read-out coordinates
  (RUNG-3's named surpass) if the reduced-vocab E-gated rule still stalls. The clean-gradient FA remains the
  fallback/upper-bound (it works; it is a less-E-gated, slightly-less-faithful credit).

`--credit burstprop` (+ `--burst-sampled` / `--burst-gain` / `--burst-p0`) retained as the tool. Runner:
`_gap_grounded_wkv_local_readout.py`. Result: `research/findings/raw/_local_bp_*_grounded.json`.
