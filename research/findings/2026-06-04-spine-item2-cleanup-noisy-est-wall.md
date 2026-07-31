---
type: finding
status: superseded
superseded_by: research/findings/2026-06-05-composer-cleanup-NEF-GO.md
date: 2026-06-04
mechanism: cleanup
---

# Spine item 2: the spiking cleanup's wall is the composer's NOISY est, not M=320 — and the lever is integration time — 2026-06-04

**One line:** A spiking matched-filter cleanup on the core bridge replaces numpy `argmax` *perfectly* at 320 concepts
when the cue is clean (1.00 at cue-cosine 0.87), but the composer's real unbind produces a **noisy estimate**
(cue-cosine ~0.35) at which it falls to ~0.49 while numpy stays 1.00. The wall is the noisy `est`, not the
320-way comparison — and the principled lever is **integration time** (more accumulation steps make the spiking
firing rates approach the exact dot-products numpy computes).

## The investigation (cheap-first, de-risk before any composer rewrite)

**Standalone mechanism (`_spiking_cleanup_core_probe.py`), M=32:** concept codes as synaptic receptive fields +
optional lateral inhibition; operates on the ON/OFF channels. On **decorrelated** codes it tracks numpy (0.99,
graceful degradation); on **correlated cos-0.80** codes it collapses (0.17) — the spiking matched filter is not
common-mode invariant, so it needs decorrelated codes (separate finding
`2026-06-04-spine-item2-spiking-cleanup-needs-decorrelation.md`).

**On the composer's REAL est (`_spiking_cleanup_on_real_est_probe.py`), V=320 production codes (cos-0.05):** captured
the real unbind estimate via `composer._unbind_onoff` for each role, then cleaned it up with numpy vs the spiking
bridge.
- The real est has **cue-cosine ~0.35** (the spiking unbind is itself noisy). numpy `argmax` recovers it
  **45/45** — near-orthogonal codes make argmax robust to low SNR.
- The spiking matched filter peaks at **~0.47** (bias −300; collapses to 0 by −800). Lateral inhibition (WTA) at the
  optimal bias **hurts** (0/45 — inhibition + high threshold suppresses everything).

**M-vs-fidelity diagnostic (standalone, M=320, clean synthetic cue):** PERFECT at high cue-cosine —
1.00 at cos 0.87, 0.98 at 0.66, 0.88 at 0.51, **0.49 at 0.33**. So **M=320 is NOT the wall** (the matched filter
discriminates 320 concepts fine when the cue is clean); the wall is the *noisy est* (cue-cosine 0.35 ≈ the
diagnostic's 0.33 point ≈ the real-est 0.47).

## The lever — integration time (helps, then plateaus at ~0.78 → saturation)

numpy `argmax` is instantaneous and infinite-precision; a spiking readout trades precision for **time** — more
accumulation steps average the firing-rate noise. Run-steps sweep at the real-est noise level (M=320, cue-cosine
0.335): recovery **0.475 (80 steps) → 0.759 (300) → 0.775 (800)**. So longer integration helps substantially (the
rate-noise component), but **plateaus at ~0.78**, ~22pp below numpy's 1.00. The flat 300→800 is a SYSTEMATIC floor,
not rate noise — consistent with **saturation**: at the tuned gain, the true concept and a few competitors all drive
*past* the neuron's saturation, so their firing rates tie and `argmax` cannot separate them. The fix is to keep the
population in its responsive (non-saturated) range — **lower match gain and/or divisive normalization** (the
canonical cortical gain-control; Carandini-Heeger). Lower gain alone does NOT close it (w_match 20 → 0.71, 15 →
0.47; both ≤ the w_match-40 0.78) — gain and signal scale together, so anti-saturation needs the actual
normalization *circuit*, not just a smaller gain.

**So the full spiking cleanup is the canonical cortical computation:** matched filter (synaptic dot-product) +
decorrelation (common-mode invariance) + temporal integration (precision) + divisive normalization (anti-saturation
gain control). Each numpy convenience (`argmax`'s infinite precision, instantaneity, common-mode invariance) maps to
a concrete cortical mechanism the brain must spend resources on.

## Conclusion + decision

The numpy `argmax` cleanup is **a thin, high-precision linear readout** — the same category as the already-disclosed
linear inter-phase ops (superposition sum, ON/OFF opponency). Removing it with a fully-spiking cleanup is reachable
in principle (the matched filter is perfect at M=320 on clean cues) but requires the **complete cortical cleanup
circuit** — decorrelation + temporal integration + divisive normalization — to overcome the composer's noisy `est`,
and a partial version (~0.78 on cue-cosine-0.35 est) would **regress the validated capability matrix** (which numpy
holds at 1.00). For a thin readout, that is a poor trade.

**Decision:** characterize the `argmax` cleanup as a **disclosed high-precision readout** (not a hidden cheat) with
the biology mapping captured below, rather than ship a sub-parity spiking version that loses capability. The load-
bearing nonlinearity (the bind/unbind coincidence) IS already spiking — the hard part is done. The full cortical
cleanup circuit is a legitimate **future sub-project** (Carandini-Heeger normalization on the core bridge), not the
highest-value next step. Move to grounding (item 3) / one-bridge (B), revisiting the cleanup circuit when the agent
is otherwise complete. NO capability bar is changed; nothing is weakened — the numpy readout stays, now honestly
documented as to *why* a cheap spiking replacement is not yet warranted.

## Biology-translatable insight (independent of the run-steps outcome)

numpy `argmax` cleanup was quietly relying on two things real neurons do not get for free: **infinite precision**
and **common-mode invariance**. Biology earns the first with **integration time** (longer settling → finer rate
discrimination) and the second with **decorrelation** (efficient coding). So the existence of the `argmax`
"shortcut" maps onto two concrete biological mechanisms — temporal integration and decorrelation — that the brain
must spend resources on. That is the deliverable whether or not the spiking cleanup reaches full parity.

## Files
- `research/findings/raw/_spiking_cleanup_core_probe.py` (matched-filter + WTA cleanup; `--rho`, `--w-inh`,
  `--run-steps`, `--sigma`, numpy baseline)
- `research/findings/raw/_spiking_cleanup_on_real_est_probe.py` (cleanup on the composer's REAL est, per category)
