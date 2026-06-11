# Dual / CLS STRONG-ENCODE DE-RISK — **BOUNDARY** (strong drive FIXES the encode tension, but the round-trip still does not close — the bottleneck has moved DOWNSTREAM, to the bind→settle→decode step)

**Status:** BOUNDARY — but a *different, more-useful* boundary than the prior one. The literal de-risk
question is **ANSWERED YES**: a strong, stable sparse encode breaks the reproducible-vs-decorrelated
tension the weak DG could not (co-occurrence repro 1.000 **AND** decorr ≈ 0 at sparse k, all 3 seeds).
**But the round-trip Pearson still does NOT close** (+0.189, vs the +0.7 bar / +1.000 clean ceiling) — and
the **DETERMINISTIC reference (a perfect, reproducible-by-construction encode) gives the IDENTICAL +0.189**,
which **proves the round-trip failure is NOT the encode**. The bottleneck is now localized to the
**bind→settle→decode** step (the Hopfield attractor's settled-state shape loss), which is *independent* of
the encode.
**Date:** 2026-06-11. **Backend:** `SIM_BACKEND=cupy` (GPU, the real bridge); numpy tiny-smoke first.
**Probe:** `research/runners/dual_cls_strong_encode_derisk_probe.py`.
**Raw:** `research/findings/raw/_dual_cls_strong_encode_multiseed.json` (seeds 42/43/44, full P1 scale),
`_dual_cls_strong_encode_smoke.json` (numpy harness check).

## The exact question (and the precise answer)

The prior on-substrate gate (`2026-06-11-dual-CLS-onsubstrate-gate-BOUNDARY.md`) found the encode was the
broken link: the **weak** spiking EC→DG drive (~15 spikes / 600 DG, below the OU noise floor) gave a DG read
that **decorrelated OR reproduced, never both** — at sparse k=40 between-cos 0.17 but repro 0.18–0.57; at
dense k=300 repro 0.93 but between-cos 0.84. **This de-risk asked: does a STRONG, STABLE sparse encode (the
input, not noise, determines the DG winners — like real strongly-firing concept/place cells) break that
tension at sparse k?**

**Answer: YES, the encode tension breaks — unanimously, multi-seed.** And **NO, that is not enough to close
the round-trip** — a second bottleneck downstream of the encode is now the binding constraint.

## Headline (front-and-centre)

| | weak DG (prior gate) | **SPIKING strong-DG (this probe)** | DETERMINISTIC ref (perfect encode) | clean-decode ceiling |
|---|---|---|---|---|
| **co-occur repro≥0.9 AND decorr≤0.1 at sparse k** | **NEVER** | **YES — all 3 seeds** | YES (repro 1.000 by construction) | — |
| DG repro at sparse k=40 | 0.18–0.57 | **1.000** | 1.000 | — |
| DG between-cos at k=40 | 0.17 | **−0.001 (mean)** | −0.001 | — |
| binding identity (noised cue) | 0.025–0.05 ≈ chance | **1.000** | 1.000 | — |
| **round-trip Pearson(S,S')** | +0.020 (= permuted) | **+0.189** (per-seed +0.141/+0.173/+0.253) | **+0.189 (IDENTICAL)** | **+1.000** |
| permuted-S baseline | +0.019 | +0.021 | +0.021 | — |

The **load-bearing co-occurrence** the weak DG could not achieve **is achieved by strong drive**: at k=40,
the spiking strong-DG read is *both* perfectly reproducible (same input → cosine 1.000) *and* decorrelated
(between-concept cosine ≈ 0, sparsity 6.7%). The strength sweep confirms the mechanism — even the *lowest*
drive tested (800 pA, ~1100–1450 DG spikes vs the weak path's ~15) is already enough to lift the read above
the noise floor into the reproducible+decorrelated regime; 5000 and 12000 pA are byte-identical at k=40 (the
input fully dominates). **The encode is fixed.**

**But the round-trip Pearson is +0.189 — far below the +0.7 bar and the +1.000 clean ceiling — and the
deterministic reference, which has a literally perfect encode (reproducible by construction, decorrelated),
fails at the EXACT same +0.189.** That equality is the decisive evidence: **the encode is no longer the
constraint.** Strong drive did its job (the encode now matches the deterministic reference exactly); the
round-trip still does not close because the limit has moved to the next stage.

## The decisive evidence — the strength × k sweep (multi-seed, seed-stable)

```
seed 42 (43/44 identical shape)
 drive   k    between-cos  repro   sparsity  spikes   bind_id  Pearson  perm
 800     40    -0.001      1.000    0.067    1454     1.000    +0.173  -0.008   <== CO-OCCUR (repro AND decorr)
 800     80    +0.074      0.627    0.133    1455     0.075    +0.000  +0.000
 800    150    +0.417      0.814    0.250    1458     0.025    +0.007  +0.007
 5000    40    -0.001      1.000    0.067    4026     1.000    +0.173  -0.008   <== CO-OCCUR (drive-saturated; == 800 at k=40)
 5000    80    +0.067      0.594    0.133    4026     0.075    -0.004  -0.003
```

Three facts make the boundary precise:

1. **Strong drive co-achieves reproducible + decorrelated at sparse k** (the thing the weak DG never did).
   At k=40 every drive ≥ 800 pA gives repro 1.000 AND between-cos ≈ 0. The de-risk's literal question is
   answered YES, multi-seed (per-seed co-occur = [True, True, True]).

2. **The encode is no longer the bottleneck — proven by the deterministic reference.** The DETERMINISTIC
   `generate_sparse_patterns` reference (repro = 1.000 *by construction*, between-cos ≈ 0) gives the
   **identical** round-trip Pearson +0.189 as the spiking strong-DG. A perfect encode and the spiking
   strong-DG fail *equally*. So whatever caps the round-trip at +0.189 is **independent of the encode's
   reproducibility/decorrelation**.

3. **The bottleneck is the bind→settle→decode step (the Hopfield settled-state shape loss).** Isolating the
   round-trip stages on the deterministic reference at full scale (seed 42):
   - **clean-decode ceiling = +1.000** — decoding the *clean* DG codes recovers S perfectly (encode + the
     learned CA1→cortex ridge decode are both fine).
   - **settled-state vs clean-DG cosine = 0.71 mean (0.64 min) even at flip=0.0 (NO cue noise).** The
     Hopfield attractor over the orthogonal sparse DG codes settles to a state with the *right identity*
     (binding 1.000) but only ~71% aligned with the clean pattern's real-valued *shape*.
   - decoding that 0.71-degraded settled state → round-trip Pearson +0.13–0.14.

   The round-trip reads the *settled attractor state* (a faithful round-trip must — that is what a recall
   reconstructs), and that settled state has lost the graded structure even with a perfect encode and zero
   cue noise. **The constraint moved from "the encode is not reproducible" (weak DG) to "the attractor's
   settled real-valued state does not preserve graded similarity" (independent of the encode).**

## Why this is the RIGHT boundary to find (the localization, per the brief)

The brief asked to localize the brain-based-vs-reproducible tension. The localization is now sharp and
**reassuring for the encode, redirecting for the round-trip**:

- **Encode (the literal de-risk target): SOLVED by strong drive.** The spiking substrate *can* produce a
  reproducible + decorrelated sparse code at sparse k — it just needs the input to strongly, stably drive
  the right sparse cells (≥ ~800 pA into the assigned ensemble), which is exactly what a matured strong
  learned sparse encode (or strong concept cells) would do. The four prior spiking-DG NEGATIVEs were all
  the *weak-drive* regime; strong drive escapes it. This is the part that bears on "should we build the
  learned strong encode" — and the encode itself is no longer the blocker.

- **Round-trip (newly exposed): the binding/recall reconstruction is the new constraint, NOT the encode.**
  The deterministic reference proves a *perfect* encode does not close the round-trip with this
  bind→settle→decode pipeline. The graded similarity is destroyed in the Hopfield settle (orthogonal sparse
  codes → near-binary attractor whose settled real-valued shape is only 0.71-faithful), before the decode
  even runs.

## Decision logic (stated explicitly)

GO required: strong drive co-achieves repro (≥0.9) AND decorr (≤0.1) at sparse k, **AND** the round-trip
closes (Pearson ≥ 0.7 ≫ permuted), **AND** binding + generalization, all multi-seed.

- Co-occurrence (repro AND decorr at one sparse-k point): **PASS** — all 3 seeds, at k=40.
- Round-trip Pearson closes: **FAIL** — +0.189 mean (vs 0.7 bar, +1.000 ceiling), and equal to the
  deterministic reference's +0.189 → not an encode failure.
- Binding identity at operating point: **PASS** — 1.000 (the *identity* round-trips; the graded *shape*
  does not).
- Generalization on-substrate (graded-DG vote): **FAIL** — 0.8–1.3× chance (the sparse pattern-separated
  DG is decorrelated by design, so a DG-space similarity vote has no graded neighbours; the orthogonal A2
  and permuted A3 controls correctly collapse, so the contrast is honest, not spuriously passing).

⇒ **BOUNDARY** (probe verdict string `BOUNDARY_cooccur_but_roundtrip_not_closed`). **Not a GO** — per the
strict bar (do not call GO unless the round-trip closes). **No banking.** But the boundary is *materially
different and more actionable* than the prior one: the encode is fixed; a separable downstream stage is the
new blocker.

## What this means for the months-scale learned-embedding build (the decision this de-risk gates)

**Honest, precise reading:**
- **The encode half of the months-scale build is DE-RISKED.** The spiking substrate can carry a reproducible
  + decorrelated sparse code at sparse k under strong, stable drive — so "learn a strong sparse encode" is a
  viable target (the learned embedding would produce exactly the strong stable drive this probe injected). The
  four prior DG NEGATIVEs do **not** condemn the spiking encode; they condemn the *weak perforant-path drive*.
- **But the build is NOT fully justified yet, because closing the round-trip needs a SECOND fix that this
  de-risk just exposed and that is independent of the encode:** the binding/recall reconstruction must
  **preserve graded similarity**, which the current orthogonal-sparse Hopfield does not (settled shape only
  0.71-faithful even with a perfect encode + zero cue noise). The clean-decode ceiling (+1.000) shows the
  similarity *is* recoverable from the DG code — the loss is entirely in the attractor's settled real-valued
  state. So the round-trip is gated on the **recall/bind representation**, not the encode.

The CLS shape remains viable (clean ceiling +1.000); the open engineering target is now **two-part and
correctly separated**: (1) a strong reproducible sparse *encode* — **de-risked** here; (2) a *recall/binding*
mechanism whose settled state preserves graded similarity (a graded/real-valued attractor or a decode that
reads the recovered-identity's *clean* code rather than the degraded settled state) — **the newly-exposed
open piece**.

## The decisive next step

**Re-run THIS round-trip with a recall step that preserves the graded shape — the cheap next probe is to
decode the RECOVERED-IDENTITY's CLEAN DG code rather than the degraded settled state.** Binding identity is
already 1.000 at the strong operating point, so the attractor reliably recovers *which* concept; decoding
that concept's clean stored DG code (a content-addressable lookup, the biological "pattern completion → read
the completed pattern" rather than "read the half-settled trajectory") should return the round-trip toward
the +1.000 clean ceiling. If it does, the round-trip is closed and the months-scale build is justified
end-to-end (strong learned encode + content-addressable recall). If even the recovered-identity's clean code
fails to round-trip the *graded* similarity (it should not — the clean ceiling is +1.000), the limit is the
sparse pattern-separated representation itself destroying gradation, and the dual architecture needs the
graded similarity carried on a *separate* (cortex-side) channel, recombined after recall — which is, in fact,
the dual/CLS design's intent (cortex graded + hippocampal decorrelated, *linked*), so the round-trip metric
may simply be testing the wrong path (it should test the cortex-channel similarity post-recall, not the
hippocampal settled state).

Do **not** re-attempt the *weak* spiking-DG read — strong drive conclusively escapes that regime (this
probe + the smoke + two independent seed-42 runs all agree). The encode question is settled; the recall
representation is the next fork.

## Anti-cheats (all present + clean)

- **The co-occurrence is reported as a CO-OCCURRENCE** — repro ≥ 0.9 AND decorr ≤ 0.1 at *one* operating
  point (k=40), not one without the other. This is the exact tension the brief flagged; it is broken, and
  reported as such, with the full strength × k surface showing *where* (k=40 only; k≥80 loses decorrelation
  as sparsity rises, exactly the prior tension shape).
- **Permuted-S baseline** for the round-trip Pearson: +0.021 (the true +0.189 clears it by only ~0.17, well
  below the +0.3 bar → the round-trip does NOT close; reported honestly).
- **Orthogonal contrast (A2)** and **permuted-S (A3)** for generalization both collapse to chance — and so
  does graded(DG), so the contrast is vacuous *and reported as such* (the pattern-separated DG has no graded
  neighbours by design; not spun as a pass).
- **Deterministic reference labelled explicitly** as the reproducible-by-construction ceiling/sanity vs the
  SPIKING strong-DG (the real test). Their identical +0.189 is the load-bearing localization, not a hidden
  conflation.
- **Validated DG read convention** (accumulated-rate-over-window top-k k-WTA, the project P1/DG-CA3
  standard); graded-codebook unit-check asserted (`is_graded=True`, within 0.39–0.40 ≫ between ≈ 0).
- **Clean-decode ceiling (+1.000)** is the internal positive control proving the decode + scoring pipeline
  is correct — so the +0.189 round-trip is a real substrate-pipeline limit (the settle step), not a probe
  artifact.
- **No `sim/` edits.** The strong DG drive sets `cp_external_input_current` on the DG slice (the input
  current the neural DG receives — the world/body drive analogue) and the DG read is a readout operation.
  Substrate is `build_biological_brain_regions(enable_hippocampus_consolidation=True)`, identical to the P1
  trisynaptic loop and the prior gate. Multi-seed 42/43/44; 448 s total on GPU.

## Reproduce

```bash
# numpy tiny-smoke (harness check, ~11 s)
SIM_BACKEND=numpy python -m research.runners.dual_cls_strong_encode_derisk_probe \
    --smoke --seeds 42 --out research/findings/raw/_dual_cls_strong_encode_smoke.json
# full GPU multi-seed (~448 s, 3 seeds, 1912-neuron / 264K-synapse bridge each)
SIM_BACKEND=cupy python -m research.runners.dual_cls_strong_encode_derisk_probe \
    --seeds 42,43,44 --drive-list "800,5000" --k-list "40,80" --window 150 \
    --out research/findings/raw/_dual_cls_strong_encode_multiseed.json
```

## Sources / cross-references

- The prior boundary this de-risks: `research/findings/2026-06-11-dual-CLS-onsubstrate-gate-BOUNDARY.md`
  (weak DG: decorrelate OR reproduce never both; round-trip +0.020 = permuted; clean ceiling +1.000).
- The architecture this gates: `research/findings/2026-06-11-dual-CLS-architecture-proof-GO.md` (+0.877
  numpy round-trip; the SHAPE proof).
- The spiking-DG weak-drive root cause: `research/findings/2026-06-11-cortex-dg-ratekwta-cleanup-NEGATIVE.md`
  (~15 spikes/600, noise floor); `2026-06-10-cortex-DG-CA3-cleanup-NEGATIVE.md`.
- Reused harnesses: `research/runners/dual_cls_architecture_proof_probe.py` (graded codebook,
  generalization, decode/Pearson); `research/runners/dual_cls_onsubstrate_gate_probe.py` (the on-substrate
  round-trip harness this extends — only the ENCODE changed: weak EC→DG read → strong stable DG-ensemble
  drive + sweep); `research/runners/cortex_sparse_attractor_poscontrol_probe.py` (Hopfield bind + noised
  cue); `research/runners/concept_pool_sparse_distributed.py` (`generate_sparse_patterns` — the project's
  reproducible strong sparse code mechanism, used both as the stable DG-ensemble assignment and the
  deterministic reference); `text_minimal_isolation.build_biological_brain_regions` (the real EC→DG→CA3→CA1
  substrate + the validated accumulated-rate DG read).
- CLS theory: McClelland-McNaughton-O'Reilly 1995; Kumaran-Hassabis-McClelland 2016. Catalog D.12 (DG
  separation), D.13 (CA3 completion).
