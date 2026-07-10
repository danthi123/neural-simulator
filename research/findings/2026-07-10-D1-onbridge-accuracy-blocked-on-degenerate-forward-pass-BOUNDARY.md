# D1 on-bridge accuracy — the credit mechanism is fixed, but the runner's spiking FORWARD PASS is degenerate: the input never reaches a discriminative hidden representation. Characterized boundary.

**Date:** 2026-07-10
**Runners:** `research/runners/_d1_onbridge_accuracy_sweep.py`, `_d1_onbridge_forward_propagation_probe.py` (numpy; NO `sim/` edit this cycle).
**Verdict:** BOUNDARY — the on-bridge learning-to-accuracy cannot be demonstrated through this runner's forward pass; the block is forward propagation, not the credit rule (which is validated).

## The chain, and where it actually breaks
The apical-coupling `sim/` edit surpassed the boundary at the MECHANISM level (directed credit restored; separation 1.33x
-> 20x at a sparse regime; byte-identical when off). But the end-to-end held-out accuracy on emerge1 (a VALID task:
numpy oracle 0.958, floor 0.543, chance 0.549) stays at **chance for every arm** — boundary, surpass, wrong-sign,
apical-lesion all 0.549, at 30-80 epochs, both gains. The train-accuracy curve is FLAT at chance too (not undertraining).

Read the substance instead of assuming: the on-bridge forward **readout is a hard constant** `[pool0=12, pool1=24]` for
**every input**, class 0 or 1, trained or untrained (**std = 0.00** across inputs). And upstream, the **hidden-layer
firing std across inputs is ~0.003** at every config swept (output_bias 40-350, in_hi 750-900, fwd_wmean 6-14). **The
input does not propagate to a discriminative hidden representation.**

## Root cause
The tonic bias that makes the hidden/output neurons fire (needed for bursts to exist — the BDSP credit requires somatic
spikes) **swamps the input-driven synaptic current**. So hidden firing is set by the bias, not the input; the hidden
representation is input-independent; the output readout is a constant; and no amount of directed credit can produce a
discriminative readout through a forward pass that doesn't carry the input. Bias reduction does not fix it (tested to
bias=40 with in_hi=900/fwd_w=14 -> hidden std still 0.003). This is the tonic-drive-vs-signal tension: enough tonic drive
to fire (bursts) vs. letting the input modulate the firing.

## What this does and does NOT indict
- **Does NOT indict the credit mechanism.** My apical-coupling edit is validated on its own terms (B rises with the
  apical; directed-credit separation 20x; moat clean in the sparse regime; byte-identical when off). That result stands.
- **Does indict the subagent's runner forward architecture.** Its "wired correctly" check passed only the numpy oracle +
  the no-weight-transport arms; it never verified the on-bridge forward readout reflects the input. It does not.

## the surpass (named; a runner rebuild, not a sim/ or mechanism wall)
The on-bridge net needs an input-DRIVEN forward pass, not a tonic-driven one: the hidden/output neurons must fire
*because of* the input (strong, structured input->hidden->output synaptic drive) rather than a class-independent tonic
bias, while still producing bursts. The project already has validated spiking-forward machinery that propagates an input
to a discriminative representation (the 88.6M spiking-forward, the concept-pool architecture, the sparse-expansion Marr
codon). The fix is to rebuild the runner's forward pass on that machinery (input-driven firing + a burst-supporting
operating point), then re-run the accuracy gate with the coupling. This is a substantial runner build, not a config tweak.

## Honest scope / strategic note
The D1 on-bridge learning-to-accuracy is a DEEP composite: input propagation + a discriminative hidden representation +
directed credit (FIXED) + a clean moat (regime, characterized) + a discriminative readout. Three of five are done; the
two open ones (input propagation + discriminative forward rep) are a spiking-forward-network build. This is a legitimate
frontier and remains queued with the surpass named. It is instrumental to the fully-spiking end state of the register's
TRANSITION learning; the register itself already works at rate (97%/73%) + spiking for memory/gates, so the communication
capability does not depend on closing this.

## Files
`research/runners/_d1_onbridge_accuracy_sweep.py`, `_d1_onbridge_forward_propagation_probe.py`; the mechanism surpass
`2026-07-10-D1-apical-soma-coupling-sim-edit-restores-directed-credit-PARTIAL-surpass.md`; the boundary
`2026-07-10-D1-onbridge-BDSP-apical-decoupled-from-soma-BOUNDARY-root-caused.md`.

## Cheap-first confirmation: it's near-SILENCE, not bias-swamping (the rebuild is genuinely deep)
I had only swept forward weights to fwd_wmean=14. The honest cheap-first extends it: even at **fwd_wmean=250, in_hi=1800,
bias=0** the hidden firing std stays ~0.005 AND the **hidden MEAN firing rate is ~0.006-0.016 (approx 1%)** — the hidden
neurons barely fire *at all*, at any drive strength (`_d1_onbridge_strong_forward_probe.py`). So the deeper truth: the
runner's net sits in a **near-silent regime** — no firing -> no bursts -> no BDSP credit -> a constant readout. It is not
that the tonic bias swamps the input; it is that neither the bias nor a very strong input->hidden pathway drives the
hidden neurons into a firing regime that is BOTH active (bursts exist) AND input-dependent. This confirms the surpass is a
genuine rebuild of the operating point + the forward wiring (using the project's validated spiking-propagation machinery,
where a driven input pool reliably makes a downstream pool fire), NOT a hyperparameter tweak. The credit mechanism (the
apical-coupling `sim/` edit) remains validated and independent of this forward-net issue.

## CORRECTION: the propagation is NOT a fundamental wall -- a minimal AMPA net DOES propagate; the runner's near-silence is a CONFIG issue
The "deep rebuild" framing above is corrected by two more a0 reads:
- **The input encoding works** (`_d1_onbridge_input_propagation_probe.py`): the runner's input pool fires input-DEPENDENTLY (mean 0.079, **std 0.025**). Different inputs -> different input firing. The bug is downstream (input->hidden).
- **A minimal 2-region AMPA input->hidden net propagates fine** (`_d1_ampa_vs_nmda_propagation_probe.py`): 20 input -> 60 hidden, AMPA, fw=40, in_hi=1200 -> hidden rate mean 0.039, **std 0.0275** (the hidden differentiates inputs). Counter to my NMDA hypothesis, NMDA is WORSE (mean 0.079 but std 0.0029 -- temporal summation SATURATES, killing discrimination). So AMPA propagation is the right substrate.

⇒ **the spiking-forward propagation is NOT a fundamental wall.** A minimal AMPA net propagates input-dependently; the runner's near-silent hidden (std 0.005) is a **specific mis-configuration** (operating point / the 2nd input->hidden->output hop / pool size), NOT a deep rebuild. The narrowed fix: match the runner's config to the working minimal AMPA recipe (proper drive level so the hidden fires ~4-8% input-dependently, then the same for hidden->output), keep AMPA (not NMDA), + the coupling. This DE-ESCALATES the frontier from "deep spiking-forward-net rebuild" to "fix the runner's operating point." (My 'deep rebuild' claim was too pessimistic -- a careful minimal-net read corrected it.)
