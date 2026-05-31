# DG-biologization line: FUNDAMENTAL BOUNDARY. The hippocampal DG pattern-separation MECHANISM is confirmed (overlapping concept activity 0.82 -> 0.18), but turning the substrate's activity into a separated-AND-STABLE compositional symbol is blocked by a FUNDAMENTAL separation-vs-reliability tradeoff: separation needs few competitive k-WTA winners (sparse -> unstable within-concept); reliability needs many stable winners (denser -> unseparated). No DG SIZE threads both (800-sparse: separated 0.27 / unstable 0.24; 4000-sparse: stable 0.6-0.8 / unseparated 0.66-0.76 -- two points on the SAME tradeoff curve, the sweet-spot is never reached), and CA3 completion collapses separation further. The oracle lookup's orthogonality is IRREDUCIBLE on this substrate. Honest biology-translatable boundary = the deliverable. PIVOT to the validated conversational capability.

**Date:** 2026-05-31
**Status:** Decisive BOUNDARY for biologizing the oracle lookup (the last engineered shortcut) via the hippocampal trisynaptic loop. The DG separation mechanism is confirmed and reproducible; the assembly into a composable symbol is fundamentally blocked. Closes the trisynaptic-symbol-source line cleanly. The night's three-arc convergence on DG pattern-separation identified the right mechanism but the wrong expectation that it could yield a usable compositional symbol on this substrate.

## The decisive evidence

The DG-composition NULL localized the failure to within-concept reliability (a concept's storage/query sparse DG codes are near-disjoint). The graded-sparse CPU model predicted a LARGER DG (more active neurons at the same sparsity fraction) would be more stable. The corrected 4000-low-drive test (sparsifying from the excitation side, 0 silent words) confirmed the prediction AND revealed the fundamental tradeoff:

| DG | sparsity | BETWEEN (separation) | WITHIN (reliability) | silent |
|---|---|---|---|---|
| 800 | 0.041 | **0.265** (separated) | **0.235** (unstable) | 4 |
| 4000 | 0.068 | 0.657 | 0.591 | 0 |
| 4000 | 0.046 | 0.758 | **0.816** (stable) | 0 |

So the larger DG FIXED reliability (within 0.59-0.82 vs 0.235, 0 silent) but LOST separation (between 0.66-0.76 vs 0.27). These are two points on the SAME monotonic tradeoff curve: more active neurons -> more within-concept stability (overlapping winners survive the storage/query noise) BUT less between-concept separation (concepts share more of the larger active set). Interpolating, at within >= 0.6 (reliable) the between is >= ~0.6 (unseparated); the curve never reaches the sweet-spot (between <= 0.5 AND within >= 0.6 simultaneously).

## Why it is fundamental (the mechanism)

Separation and reliability are produced by the SAME competitive k-WTA mechanism with OPPOSITE sparsity demands:
- SEPARATION requires FEW, strongly-competitive winners (high sparsity / small active set): different concepts then pick different winners -> low between-concept cosine. But few near-threshold winners are MAXIMALLY SENSITIVE -- a concept's two noisy observation-halves flip the winners -> low within-concept cosine (unstable). (800-sparse.)
- RELIABILITY requires MANY, stably-co-active winners (lower sparsity / large active set): the same winners survive the storage/query noise -> high within-concept cosine. But many co-active neurons are SHARED across concepts -> high between-concept cosine (unseparated). (4000-sparse.)
DG size moves along this curve; it does not escape it. And CA3 completion (the trisynaptic attractor that would add stability) makes it WORSE for separation: the CA3 diagnostic showed CA3 settling to dense, overlapping ensembles (between-concept ~0.90) -- the attractor amplifies overlap. So neither DG-size nor CA3 threads separation AND reliability for this substrate's concept activity.

## The conclusion (honest, biology-translatable)

The substrate's overlapping concept-pool activity (between-concept cosine 0.82) CANNOT be transformed into a separated-AND-stable compositional symbol by the hippocampal pattern-separation/completion machinery at this scale. The oracle lookup's value -- a CLEAN, ORTHOGONAL, STABLE code per concept -- is irreducible on this substrate: the substrate's own activity does not contain orthogonality that survives the stability requirement. This is WHY all the biologize-the-oracle-lookup attempts (activity grounding, temporal integration, attractor cleanup, DG separation, larger DG, CA3 completion) fail -- they all hit this one tension. It is a deep, clean, biology-translatable boundary: a compositional symbol must be both well-separated and stable, and a single competitive-sparse-coding stage cannot deliver both from overlapping inputs.

This does NOT close composition -- it closes ONE route (biologizing the oracle lookup via the hippocampal loop on this substrate). The validated phase-coded FHRR composition (with the oracle lookup as an engineering component) still works (identity-level 0.96-0.99), and the validated grounded-memory conversational capability (multitag 90% / engram 87.5% / G.20 160-320 concepts) is unaffected and is the project's working artifact.

## Disposition: PIVOT to P4 (the validated conversational capability)

Per the pre-staged decision logic and the worth-GPU-time frame: the DG-biologization line is BANKED as this characterized fundamental BOUNDARY (the deliverable). Pivot to advancing the WORKING conversational stack -- the G.20 multitag bridges (160 + 320 concepts) exist on disk and are instantly runnable. Candidate high-value extensions: (a) multi-hop reasoning over stored associations (a known open gap), (b) scale toward 640 concepts (D8 infra scaffolded), (c) a cleaner interactive chat surface. The biologization NEGATIVE/BOUNDARY characterization across the night (integrated-loop VOID -> ceiling audit -> denoiser NEGATIVE -> 3-arc DG convergence -> DG gate PASS -> DG-composition NULL -> this fundamental separation-vs-reliability boundary) is a coherent, honest, biology-translatable scientific deliverable.

## Discipline

Throwaway probes only; no protected/frozen/moat/sim/runner/builder module modified (all reuse-by-import byte-unchanged). No bars moved. No autograd. The boundary was reached only after the controller's own mis-tuning (raised drive in the first 4000 re-run) was caught and CORRECTED (4000-low-drive, 0 silent, clean sparse) -- the boundary is NOT an artifact of mis-tuning; it is the clean tradeoff curve. The graded-model prediction (larger DG more stable) was confirmed, which is what makes the simultaneous separation LOSS decisive. Honest BOUNDARY = the deliverable.
