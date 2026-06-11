# Dual / CLS ON-SUBSTRATE GATE — **BOUNDARY** (the spiking-DG sub-reproducibility destroys the round-trip; the learned decode does NOT rescue it)

**Status:** BOUNDARY. The numpy ARCHITECTURE-PROOF passed (+0.877 round-trip Pearson on a synthetic
graded codebook with a DETERMINISTIC sparse encode); this ON-SUBSTRATE gate replaces that
deterministic encode with the **real spiking dentate-gyrus (DG)** pattern-separation and re-runs the
load-bearing round-trip. **It collapses to the permuted-baseline floor.**
**Date:** 2026-06-11. **Backend:** `SIM_BACKEND=cupy` (GPU, the real bridge); numpy tiny-smoke first.
**Probe:** `research/runners/dual_cls_onsubstrate_gate_probe.py`.
**Raw:** `research/findings/raw/_dual_cls_onsubstrate_multiseed.json` (seeds 42/43/44, full scale),
`_dual_cls_onsubstrate_seed42.json` (seed-42 timing run), `_dual_cls_onsubstrate_smoke.json`
(numpy harness check).

## Headline (the load-bearing number, front-and-centre)

**On-substrate round-trip Pearson(S, S') = −0.000 / +0.044 / +0.015 (mean +0.020) — IDENTICAL to the
permuted-S baseline (−0.000 / +0.044 / +0.015, mean +0.019), against the numpy proof's +0.877.** The
learned CA1→cortex ridge decode **does NOT compensate** for the real spiking DG: at NO read-sparsity k,
on NO seed, does the on-substrate Pearson rise above its own permuted baseline. The graded similarity
is GONE by the time it has passed through the spiking DG and the attractor.

**The documented risk materialised, in the exact shape predicted.** The reason is the
separation-vs-reproducibility tension from `2026-06-11-cortex-dg-ratekwta-cleanup-NEGATIVE.md`, now
shown to **propagate through the full round-trip**:

| | numpy proof (deterministic encode) | ON-SUBSTRATE (real spiking DG), mean 3 seeds |
|---|---|---|
| round-trip Pearson(S,S') | **+0.877** | **+0.020** (= permuted +0.019) |
| permuted-S baseline | ≈ 0 (−0.06) | +0.019 |
| binding identity (noised cue) | 1.000 | **0.033** (≈ chance 0.025) |
| DG/expansion between-cos | 0.009 (decorrelated) | 0.168–0.205 decorrelated **OR** 0.84–0.89 reproducible (never both) |

| Gate (all 3 seeds) | result | verdict |
|---|---|---|
| DG decorrelates the graded codes (some k) | between-cos 0.168–0.205 < 0.40 (sparsity 6.7%) | **PASS** |
| DG same-input reproducible (the documented risk) | **0.18–0.57** at the decorrelated k | **FAIL** (reported, not buried) |
| Binding on the real DG codes | identity 0.025–0.05 ≈ chance | **FAIL** |
| **C2 — learned decode recovers the round-trip** | **Pearson +0.020 = permuted +0.019** | **FAIL** (load-bearing) |
| A1 graded generalizes on-substrate (DG-space vote) | 0.19–0.26 ≈ chance 0.25 | **FAIL** |
| A2 orthogonal collapses (contrast) | 0.19–0.29 ≈ chance | PASS (vacuous — graded also at chance) |
| A3 permuted collapses (anti-cheat) | 0.21–0.32 ≈ chance | PASS |

⇒ **BOUNDARY. The dual architecture is VIABLE in shape (numpy proof) but the SPIKING-DG ENCODE is the
broken link on the real substrate. The encode needs a more-reproducible mechanism — NOT the noisy
spiking DG.**

## The decisive evidence — the full per-k sweep (multi-seed, seed-stable)

The probe sweeps the DG read-sparsity k and, **at each k**, measures DG decorrelation + reproducibility
→ binds on the real DG codes → learns the CA1→cortex decode → computes the round-trip Pearson. The
operating point is selected by the **highest on-substrate Pearson** (the compensation hypothesis' best
shot), so a GO cannot be missed by a bad k choice and a BOUNDARY means **no k works**.

```
seed 42                                                              seed 43/44 identical shape
 k    between  repro  spars  bind   Pearson  perm   clean-DG
 40   +0.171   0.250  0.067  0.025  -0.002  -0.008  1.000   <- DECORRELATES (6.7% sparse) but repro 0.25
 80   +0.539   0.570  0.133  0.050  -0.000  -0.000  1.000
150   +0.721   0.802  0.250  0.025  -0.009  -0.009  1.000
300   +0.839   0.930  0.500  0.025  -0.004  -0.003  1.000   <- REPRODUCIBLE (0.93) but between-cos ≈ raw
```

Three facts make the boundary airtight:

1. **The DG read decorrelates OR reproduces, never both** — exactly the tension of the prior NEGATIVE,
   now confirmed on graded (not just raw-denoise64) input. At **k=40** the DG genuinely
   pattern-separates the graded codes (between-cos +0.17, sparsity 6.7% — biologically valid) but the
   same-input read is only 0.18–0.33 reproducible. At **k=300** the read is 0.93–0.95 reproducible but
   between-cos +0.84–0.89 ≈ the raw input cosine — zero decorrelation gain. There is no k in between
   where both hold.

2. **The round-trip Pearson sits EXACTLY on the permuted baseline at every k, every seed.** Whether the
   DG is decorrelated-but-noisy (k=40) or reproducible-but-overlapping (k=300), the learned decode
   recovers nothing — Pearson and its permuted control are equal to 3 decimal places. The graded
   structure does not survive into a form the decode can read.

3. **The decode itself is PERFECT — the clean-DG ceiling = +1.000 at every k/seed.** When the round-trip
   is run on the CLEAN DG codes (no cue noise, no attractor), the learned ridge map inverts the DG back
   to the cortex codebook with Pearson +1.000. **This isolates the failure unambiguously: the decode is
   not the problem; the spiking DG read is.** The compensation hypothesis ("the learned decode adapts to
   whatever the DG produces") is FALSE here because the DG produces something *non-reproducible* —
   driving the same input twice gives a different DG code (repro 0.18–0.57), so there is no stable
   target for the learned map to compensate toward, and a noised cue settles to the wrong (or no)
   attractor (binding 0.025–0.05 ≈ chance).

## Why the numpy proof passed and the substrate does not — the precise gap

The numpy proof's encode was a **deterministic** fixed random projection + top-k WTA: the SAME input
always produces the SAME sparse expansion (reproducibility = 1.000 by construction), and at K/N = 0.0125
it decorrelated to between-cos 0.009 while staying perfectly reproducible. **Deterministic decorrelation
+ perfect reproducibility is exactly what the round-trip needs**, and the numpy proof had both for free.

The real spiking DG has neither jointly. Its EC→DG input-driven signal sits **below the threshold noise
floor** (`2026-06-11-cortex-dg-ratekwta-cleanup-NEGATIVE.md`: ~15 spikes / 600 DG neurons; OU noise,
not the input, determines which cells cross threshold). So:
- a SPARSE read (small k, below the spike count) is decorrelated but noise-determined → non-reproducible;
- a DENSE read (large k) captures "who fired at all," which is the same noise-determined background for
  every concept → reproducible but un-separated (between-cos ≈ raw).

The deterministic numpy encoder had no noise floor; the spiking DG is dominated by it. **The
architecture proof de-risked the SHAPE; this gate shows the SUBSTRATE's encode does not realise that
shape.** (This is exactly the carried-forward caveat #3 of the architecture-proof GO: "the DG may behave
differently on graded structured input than on the random sparse input it was P1-validated on… that
integration risk is now front-loaded." It is now resolved: the DG behaves the same broken way on graded
input as on raw denoise64 — the input's structure is irrelevant because the read is noise-determined.)

## Decision logic (stated explicitly)

GO required: DG decorrelates ∧ learned decode recovers the round-trip (C2 Pearson ≥ 0.7 ≫ permuted) ∧
binding ∧ generalization, all multi-seed.
- DG decorrelates: **PASS** (at k=40).
- C2 round-trip recovered: **FAIL** — Pearson +0.020 = permuted +0.019, vs the 0.7 bar; not rescued.
- Binding on-substrate: **FAIL** — 0.033 ≈ chance.
- Generalization on-substrate: **FAIL** — graded(DG) at chance (the similarity does not survive the
  encode, so the DG-space vote has no graded neighbours to exploit).

⇒ **BOUNDARY** (probe verdict string `BOUNDARY_binding_fails_onsubstrate`). The spiking-DG
sub-reproducibility destroys the round-trip even with the learned decode, precisely as the risk warned.
**No banking** — reported exactly as found.

## The brain-based-vs-reproducible tension (named, per the brief)

The honest core: a **brain-faithful spiking DG** at this scale is, by the project's three multiply-
confirmed NEGATIVEs (instantaneous spiking, rate-accumulated k-WTA, OU-off; + this round-trip gate),
**not reproducible enough** to serve as the CLS encode. The reproducible alternatives are all
non-spiking:
- the numpy proof's **deterministic random-projection + top-k** (reproducible by construction, works) —
  but that is a host/deterministic encode, not the brain's DG firing;
- a **learned cortex→DG-target map** (a fixed/learned assignment of each cortex code to a sparse DG
  ensemble) — reproducible, and the genuinely brain-plausible answer (developmental wiring assigns a
  stable sparse code per concept), but it is a *learned/structural* encode, not the stochastic spiking
  DG read this probe used.

So the encode-mechanism question is the fork: **the round-trip works the moment the encode is
reproducible (clean-DG ceiling +1.000 proves it), and the spiking-DG read is the only thing that makes
it non-reproducible.** The CLS architecture does not need a *stochastic* DG; it needs a *reproducible
sparse separator*, which biology realises through **learned/structural** DG wiring (stable place-/
concept-fields), not through the moment-to-moment spike lottery this read samples.

## The decisive next step

**Pivot the encode from the stochastic spiking DG READ to a REPRODUCIBLE sparse cortex→DG assignment
(deterministic-or-learned), then re-run THIS gate.** Concretely the cheap next probe is: replace
`rate_kwta_dg_read` with a fixed/learned cortex→DG-ensemble map (each cortex code → a stable K-of-N DG
pattern — the numpy proof's deterministic encoder IS this, and `generate_sparse_patterns` already
produces such stable codes), keep everything else (the real CA1→cortex learned decode, the Hopfield
bind, the on-substrate generalization vote) identical, and confirm the round-trip Pearson returns toward
+0.877. That isolates whether **a reproducible encode** closes the gate — which the clean-DG ceiling
(+1.000) strongly predicts it will. If it does, the open engineering target is then narrowly defined:
**a spiking mechanism that produces a reproducible sparse DG code** (much higher EC→DG signal-to-noise,
or a learned/structural DG assignment matured through development), not the stochastic read.

Do **not** re-attempt the stochastic spiking-DG read for the CLS encode — that path is now four-times
NEGATIVE on the same root cause (the three cleanup NEGATIVEs + this round-trip gate). The honest label:
**"the CLS round-trip is viable and the learned decode is sufficient; the open piece is a *reproducible*
sparse encode, which the brain realises structurally, not via the noise-dominated spiking-DG read."**

## Anti-cheats (all present + clean)

- **Permuted-S baseline (the headline anti-cheat):** the round-trip decoder fit on a row-shuffled cortex
  codebook gives Pearson +0.019 — and the TRUE Pearson (+0.020) is equal to it, which is itself the
  NEGATIVE result (a real signal would clear the permuted floor by ≥ 0.3; it does not).
- **Orthogonal contrast (A2):** generalization on the DG-encoded orthogonal codes is at chance — but so
  is graded(DG), so the contrast is vacuous here (the point of A2 is to show graded ≫ orthogonal; both
  collapse on-substrate because the encode destroys the graded structure — reported honestly, not spun
  as a PASS).
- **DG same-input reproducibility reported FRONT-AND-CENTRE** (0.18–0.57 at the decorrelated k) — the
  documented risk, not buried; it is the mechanistic cause of the boundary.
- **Validated DG read convention used** (accumulated-rate-over-window top-k, the project's P1/DG-CA3
  standard), not a single-spike read; graded-input codebook unit-check asserted (`is_graded=True`, within
  0.40 ≫ between −0.001 every seed).
- **Clean-DG ceiling (+1.000)** is the internal positive control proving the decode + scoring pipeline is
  correct — so the +0.020 round-trip is a real substrate failure, not a probe artifact.
- **No `sim/` edits.** The DG read is a readout operation on the bridge's spike state. Substrate is
  `build_biological_brain_regions(enable_hippocampus_consolidation=True)`, identical to the validated P1
  trisynaptic loop and the DG-CA3 NEGATIVE. Multi-seed 42/43/44; ~575 s total on GPU.

## Reproduce

```bash
# numpy tiny-smoke (harness check, ~9 s)
SIM_BACKEND=numpy python -m research.runners.dual_cls_onsubstrate_gate_probe \
    --smoke --seeds 42 --out research/findings/raw/_dual_cls_onsubstrate_smoke.json
# full GPU multi-seed (~575 s, 3 seeds, 1912-neuron bridge / 265K synapses each)
SIM_BACKEND=cupy python -m research.runners.dual_cls_onsubstrate_gate_probe \
    --seeds 42,43,44 --out research/findings/raw/_dual_cls_onsubstrate_multiseed.json
```

## Sources / cross-references

- The architecture-proof this gates: `research/findings/2026-06-11-dual-CLS-architecture-proof-GO.md`
  (+0.877 numpy round-trip; carried-forward caveat #3 = exactly this integration risk).
- The spiking-DG sub-reproducibility root cause:
  `research/findings/2026-06-11-cortex-dg-ratekwta-cleanup-NEGATIVE.md` (repro ≈ sep, ~15 spikes/600,
  noise floor); `research/findings/2026-06-10-cortex-DG-CA3-cleanup-NEGATIVE.md` (same-input DG cosine
  0.04–0.15).
- Reused harnesses: `research/runners/dual_cls_architecture_proof_probe.py` (synthetic graded codebook,
  generalization, permuted controls, decode/Pearson machinery);
  `research/runners/cortex_sparse_attractor_poscontrol_probe.py` (Hopfield bind + noised cue);
  `research/runners/validate_trisynaptic_loop.py` / `text_minimal_isolation.build_biological_brain_regions`
  (the real EC→DG→CA3→CA1 substrate + the validated accumulated-rate DG read).
- CLS theory: McClelland-McNaughton-O'Reilly 1995; Kumaran-Hassabis-McClelland 2016. Catalog D.12 (DG
  separation), D.13 (CA3 completion).
