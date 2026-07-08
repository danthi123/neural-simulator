# The KNOWLEDGE half of breadth, rung 2 (GO on spikes, 6-seed): property inheritance over REAL-corpus-discovered categories realized ON THE SPIKING SUBSTRATE — a held-out member of a TinyStories-discovered category inherits its class property via the EMERGE-42 competitive pooler + the committed HTM coincidence kernel, read from `cp_v_apical`. Lesion/derangement/permuted-features all collapse. NO `sim/` edit.

**Date:** 2026-07-08
**Runner:** `research/runners/_realcorpus_inheritance_rung2_spiking_derisk.py` (reuse-by-import: EMERGE-42's spiking machinery — `apply_kernel_update` = the committed `sim.kernels.fused_htm_permanence_update` three-term rule, `_prime_from_winners`, `_host`, the EMERGE-38 competitive pooler; + the breadth discovery `discover_vocab`/`learn_stream_codes`). numpy backend, offline. NO `sim/` edit.
**Verdict:** rung-2 GO (spiking) at K=1024; PARTIAL at K=256 (probe-resolution-limited). Realizes rung-1's rate inheritance ON SPIKES — the biology-purity step of the KNOWLEDGE-half ladder.

## Why this ran (the rate→spike ladder step)
Rung 1 (`2026-07-08-knowledge-half-inheritance-rides-real-corpus-discovered-breadth-rung1-GO.md`) showed the real-corpus RATE codes support held-out inheritance (associative-memory read). The mission's non-negotiable is FULLY SPIKING on one brain, so this rung realizes it on the spiking substrate: the real-corpus co-occurrence codes → the EMERGE-42 competitive pooler → a CLASS property taught on the committed HTM kernel on a real `SimulationBridge` → a held-out member's category read from the spiking apical drive.

## The mechanism (on spikes)
- **Input:** each probe word's real-corpus co-occurrence code → an SDR = its top-T=50 most-active hubs.
- **Pooler:** the EMERGE-38 competitive self-organizing pooler (numpy k-WTA learning; the spiking k-WTA realization is EMERGE-40/41) maps each word's SDR → a category codon (shared columns for same-category words).
- **Teach (SPIKING):** each category's CLASS property is bound on its NON-HELD member codons via `apply_kernel_update` — the committed `sim.kernels.fused_htm_permanence_update` three-term rule on a real Izhikevich `SimulationBridge` with two-compartment dendritic-plateau (dAP) coincidence detection.
- **Inherit (SPIKING read):** a HELD-OUT member (never taught the class property) is primed (`_prime_from_winners`) on its pooler codon + identity ensemble; its category is read as the argmax of the apical compartment drive (`cp_v_apical`) over the category property cells — inheriting the class property via the shared codon.

## The result — 6-seed (42/43/44/100/101/102), TinyStories (3.9M tokens), T=50
| scale | SPIKING held-out inherit | deranged-labels | permuted-features | lesion (coincidence off) | chance |
|---|---|---|---|---|---|
| **K=1024** (8 cats, 31 held-out) | **0.490 ± 0.142** (3.9× chance) | 0.094 | 0.073 | 0.000 | 0.125 |
| K=256 (4 cats, 8 held-out) | 0.458 ± 0.138 (1.8× chance) | 0.062 | 0.062 | 0.000 | 0.250 |

**GO at K=1024** — every seed beats chance AND all three controls by ≥0.15 margin; the held-out member inherits its class property ON SPIKES at 3.9× chance. **K=256 is PARTIAL** — the mechanism works (aggregate 1.8× chance, all controls collapse) but the strict per-seed gate fails on the weakest seed because the 8-item held-out probe quantizes accuracy to eighths (the K=1024 31-item probe resolves this — the fix was probe resolution, not the mechanism).

**Every control collapses (all seeds):**
- **LESION (coincidence detection off) → 0.000:** the committed HTM/dAP spiking kernel cannot bind without coincidence detection → inheritance is genuinely the spiking mechanism, not a host computation. (The single most load-bearing control.)
- **PERMUTED-features (random SDRs) → 0.073:** the pooler cannot discover categories from random features → the real category structure is load-bearing.
- **DERANGED-labels (random category grouping) → 0.094:** teaching random groupings → held-out cannot inherit its real category → the real grouping is load-bearing.

## Root-cause fix along the way (systematic debugging)
The first pass was NEGATIVE (held-out 0.125 == permuted-features 0.125): the pooler was not discovering categories from the real-corpus SDRs. Root cause (measured, not guessed): SDR-ification at T=12 collapsed the continuous code's within/between cosine margin (+0.134) to a within/between SDR-jaccard of +0.047 — EMERGE-42's pooler was tuned for SYNTHETIC SDRs with ~0.5 within-category overlap, but real-corpus SDRs at T=12 share only ~7.5% of hubs (7× too sparse for the competitive pooler to lock onto shared columns). Single-variable fix: T=50 (within-jaccard 0.154) restores enough overlap → clean pooler discovery + spiking inheritance. T=150 over-densifies (deranged rises → less discriminative). T=50 is the swept operating point.

## Honest scope
- The INHERITANCE is spiking (committed HTM kernel + dAP coincidence on a real bridge + apical read; the lesion→0.000 confirms it). The pooler competitive-LEARNING is the EMERGE-38 numpy k-WTA (its spiking realization is EMERGE-40/41) — same scope as EMERGE-42's own probe.
- Absolute accuracy (0.490) is well above chance + all controls but lower than rung-1's rate read (0.656 at K=1024) — the on-substrate apical single-shot read is noisier (the documented spiking read-out cost; population coding is the field's lever to close it).
- The taught property is a synthetic per-category target vector (the mechanism, not real facts yet) — rung 3 mines real class properties from the corpus.

## What this establishes + the next rung
The KNOWLEDGE half of breadth now rides real-corpus-discovered structure ON THE SPIKING SUBSTRATE: discover a broad vocab from a real corpus (breadth thread) → a held-out member inherits its class property on spikes via the competitive pooler + committed HTM kernel (this). Next rungs: (3) mine REAL class properties from the corpus (not a synthetic target); (4) wire the spiking inherited answer into a conversational turn (EMERGE-59..73 speaks it on spikes); and the population-coded read-out to lift the absolute spiking accuracy.

## Files
`research/runners/_realcorpus_inheritance_rung2_spiking_derisk.py`; 6-seed `research/findings/raw/_rc_spk_s{42,43,44,100,101,102}.json` (K=256) + `_rc_spk_k1024_s*.json` (K=1024) + the T-sweep `_rc_spk_T{50,100,150}.json`. Prior: rung 1 `2026-07-08-knowledge-half-inheritance-rides-real-corpus-discovered-breadth-rung1-GO.md`; EMERGE-42 `2026-07-02-emerge42-pooler-discovered-categories-reason-GO.md`.
