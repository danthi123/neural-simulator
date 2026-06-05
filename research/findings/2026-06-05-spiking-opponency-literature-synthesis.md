# Robust spiking opponency — literature + catalog/Kandel synthesis (the NEGATIVE-branch fix) — 2026-06-05

Owner-requested prep (research-first, the proven pattern). Deep pass on the composer's `onoff(bon−boff)` opponency
(the last numpy in the compute path, which the simple-accumulator de-risk + likely the NEF de-risk fail to make
spiking). Mined the LOCAL catalog (`docs/biology.md` Retina) + Kandel 6e Ch 22 (the retina, `references/textbooks/`)
+ the VSA/spiking literature. The synthesis is decisive and reframes the problem.

## The load-bearing structural insight: ON/OFF is a TRANSPORT code, not the computation
`bind_fact` returns `(bon, boff)`, but EVERY downstream consumer (`_filler_signed`, `unbind`, `_render_filler`)
immediately recomputes `e_on − e_off`. The real object is the SIGNED `s = bon − boff`; ON/OFF rectification exists
only because spike rates are non-negative. The failure is trying to recover a SMALL SIGNED DIFFERENCE from TWO
separately-read NOISY NON-NEGATIVE channels. Biology and the VSA literature both say: don't represent it that way.

## What biology actually does (Kandel 6e Ch 22 "Low-Level Visual Processing: The Retina", pp 521-544)
The retina computes EXACTLY this (opponent common-mode removal) and does it ROBUSTLY — but NOT by subtracting one
noisy correlated channel from another:
1. **The surround subtracts a SMOOTH low-pass PREDICTION** (a spatial weighted-mean via horizontal cells), not the
   raw channel (Kandel p536; difference-of-Gaussians p535). The subtracted quantity is DENOISED (a sum → high SNR).
2. **The result is a two-cell PUSH-PULL**, each half-wave rectified (ON cell for +, OFF cell for −, p530-531) — but
   biology never reduces it back to a single subtracted number; the common mode was already removed upstream.
3. **Common-mode rejection is the explicit function** (Fig 22-10 p532: uniform center+surround → small response;
   "discard ambient intensity, keep reflectance" p543). Color opponency = the same trick on correlated cone channels.
4. **THE DEEPEST FINDING (p543):** these gain changes / common-mode subtractions are done with **GRADED (analog)
   signals BEFORE action potentials**, because "firing rates vary over only two orders of magnitude" — i.e. **biology
   REFUSES to do common-mode subtraction in spike RATES; it does it in the analog pre-spiking stage.** This is the
   root explanation of the de-risk failure: it tries in spike-rates what evolution moved to the analog stage.

## The SNR argument (why rate-coded subtraction of a small correlated difference is FUNDAMENTAL)
Subtracting two cos-ρ correlated channels amplifies relative noise by ~√(2/(1−ρ)) ≈ **4.3× at ρ=0.89**: a 3%
per-channel spiking read → ~13% error on the difference, matching the observed collapse (channel cos 0.97 → signed
cos 0.41). Holding signed cos ≥ 0.99 needs per-channel ≈ 0.999 ≈ **~100× more spikes** — not available at biological
rates. So Option D (honest boundary) is itself biology-faithful.

## Recommended fixes (ranked) — STRUCTURALLY REMOVE the small-signal-of-correlated-rates object
- **Option B (CHEAP, keeps the bind — recommended FIRST if NEF is NEGATIVE): bipolar threshold the bundle (MAP-B /
  Binary Spatter Codes).** Keep the ±1 Hadamard bind + the validated spike-sum accumulator (cos 0.97); then a
  per-dimension WTA between the ON and OFF lines (a 2-neuron competitive microcircuit per dim = biology's push-pull
  as a DECISION) → binary ±1. The small-signal problem only exists for GRADED `bon−boff`; a per-dim argmax(ON,OFF) is
  a SIGN read — the most noise-robust thing a spiking pair can compute (tolerates the full 3% noise; only the sign
  matters). Bipolar VSA (Gayler MAP-B, Kanerva BSC) is canonical + robust. Cost: quantization (recoverable by raising
  D, the project's validated lever). SMALL, LOCAL change reusing the entire bind/store/unbind.
- **Option A (STRATEGIC, biggest leverage): pivot the bound-vector representation to spiking-phasor FHRR.** Every
  component unit-magnitude (info in PHASE, no common mode, no `bon−boff`); bind=phase-add, unbind=conjugate,
  bundle=phase-of-complex-sum — **the opponency simply doesn't exist in this algebra.** The repo HAS the reference
  (`research/runners/spiking_phasor_fhrr.py`, Orchard-Jarvis 2023; genuinely spiking via resonate-and-fire,
  phase=spike_time/T). Readout SNR ≈ 2N/M (a DIMENSION dial, not a fragile subtraction) — dovetails with the project's
  "cost of correlation is DIMENSIONAL" law. Gives the F=3 two-attribute resonator (which ±1 provably can't do) for
  free. Tradeoff: biggest rework (the whole bind/store/unbind move to phase/timing); decision = promote the existing
  FHRR reference from "numpy ceiling" to "the substrate."
- **Option C (keep the in-flight NEF, but fix the subtracted quantity): predictive coding (Srinivasan-Laughlin-Dubs
  1982).** Subtract a SMOOTH/low-rank PREDICTION of the common mode `c = ½(bon+boff)` (estimated in a separate
  well-averaged population → high SNR), with the surround gain SET BY THE OPERATING SNR (Atick-Redlich 1992: sharp at
  high SNR, diffuse at low SNR — the de-risk used the high-SNR op at low SNR). Sidesteps divisive shunting (subtract
  in the represented value) AND subtracts a denoised common mode. Use only if the NEF read returns NEGATIVE + a phasor
  pivot is deferred; still SNR-limited at heart.
- **Option D (honest boundary):** if all fail, the rate-coded small-signal common-mode removal is an SNR boundary
  (biology avoids it by staying analog pre-spiking) → the two LINEAR glue ops stay numpy DISCLOSED, both DEEP
  shortcuts cleared, pivot to the grounded run.

## Recommendation
Lead with **Option B (bipolar threshold) as the cheap immediate fix** and **Option A (FHRR phase-code) as the
strategic fix** — both STRUCTURALLY remove the small-signal object, which the biology (Kandel Ch 22) + the SNR
argument both say is the only robust spiking path. (My earlier Option-0 "is opponency needed?" is subsumed: the
consumers recompute `e_on−e_off`, so the signed object is unavoidable WITHOUT a representation change — B or A IS the
representation change.) Reserve C if the NEF is NEGATIVE + A/B deferred; D as the honest verdict.

## Citations
- Kandel 6e Ch 22 (retina ON/OFF, center-surround common-mode rejection, color opponency, "graded before spikes"
  p543), `references/textbooks/kandel-pns-6e/full-book.pdf` pp 565-592; `docs/biology.md` §Retina.
- Srinivasan, Laughlin & Dubs (1982) *Proc. R. Soc. B* 216:427 (predictive coding, SNR-set surround); Atick-Redlich
  (1992) *Neural Comp.* 4:196 (whiten/smooth by SNR).
- Frady & Sommer (2019) *PNAS* 116:18050 (TPAM spiking phasors, SNR≈2N/M); Orchard-Jarvis (2023) ICONS (spiking-phasor
  FHRR — the repo's reference); Schlegel-Neubert-Protzel (2021) (VSA comparison, FHRR + bipolar MAP-B/BSC robustness).
- Plate 1994/2003 (HRR); Gayler 2003 (MAP-B); Kanerva 2009 (BSC); Eliasmith-Anderson 2003 (NEF, Option C basis).
