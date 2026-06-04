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

## The lever — integration time (PENDING result)

numpy `argmax` is instantaneous and infinite-precision; a spiking readout trades precision for **time** — more
accumulation steps average out the firing-rate noise, so the rates converge to the exact match scores numpy uses,
and recovery should climb toward parity. Run-steps sweep at the real-est noise level (M=320, cue-cosine 0.33,
run-steps 80 → 800): **PENDING**.

- If recovery climbs toward ~0.9+ at longer integration, item 2 succeeds with an honest **compute cost** (the spiking
  cleanup needs several-fold more integration than the bind to match numpy precision — biologically reasonable; the
  brain's ~100 ms recognition latency *is* exactly this precision-time tradeoff).
- If it plateaus below parity, there is a systematic (non-averagable) error at low cue-cosine and the cleanup needs a
  recurrent denoising attractor (capacity ~0.14·D → D ≳ 2560 for 320) or a cleaner unbind.

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
