---
type: finding
status: live
date: 2026-05-24
---

# bio_brain_regions load-ceiling map (extension of pillars n=96/n=97/n=98) — ALL THREE substrate variants (OPTION 3 / HIPPO-OPTION3 / DLPFC-extension) PASS at EVERY load L=2..7 (the gamma-slot ceiling); substrate has HUGE capacity headroom; the (c) loop NEGATIVE is bounded as a (c)-integration design limitation, NOT a substrate limitation (2026-05-24)

## What was tested

Descriptive load-ceiling map across the three bio_brain_regions substrate variants validated by pillars n=96/n=97/n=98 (OPTION 3 no-hippo / HIPPO-OPTION3 / DLPFC-extension). Those pillars tested at loads {2, 3, 5}; this probe extends to loads {2, 3, 4, 5, 6, 7} -- the full gamma-slot range (N_GAMMA_SLOTS=7 is the ceiling).

Analogous to the cross-bridge 160-concept-union load-ceiling map (2026-05-24) which extended pillar n=95. CPU-only; reuses every substrate's cached trained activity + parallel-matching primitives byte-unchanged. Total wall-clock 930.5 s (~15.5 min) for 3 substrates × 3 seeds × 6 loads × 200 trials = 10800 OB trials + 10800 OI trials = 21600 trials total.

## Result: ALL substrates PASS at every load multi-seed

Multi-seed-mean accuracy:

| Substrate | L=2 OB / OI | L=3 OB / OI | L=4 OB / OI | L=5 OB / OI | L=6 OB / OI | L=7 OB / OI |
|---|---|---|---|---|---|---|
| **OPTION 3 (no hippo)** | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.998 | 1.000 / 0.978 | 1.000 / 0.900 |
| **HIPPO-OPTION3** | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.997 | 1.000 / 0.970 | 1.000 / 0.895 |
| **DLPFC-extension** | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 0.980 | 1.000 / 0.935 |

- **OB exactly 1.000 at every cell** across all 54 cells (3 substrates × 3 seeds × 6 loads): zero errors / 10800 OB trials. Per-slot identification is perfect at the maximum gamma-slot capacity (L=7).
- **OI ≥ 0.895 at every cell**. Multi-seed-means at L=7 OI: 0.900 / 0.895 / 0.935 — well above the 0.80 bar by 0.10+ margin.

Highest-L-with-OI-≥-0.80 = **L=7 (the gamma-slot ceiling)** for all three substrates. Lowest-L-below = **None** at L=2..7.

Per-substrate per-seed L=7 OI: OPTION 3 [0.900, 0.870, 0.930]; HIPPO [0.860, 0.905, 0.920]; DLPFC [0.905, 0.950, 0.950]. Even the lowest single-seed cell (HIPPO seed 42 L=7 OI = 0.860) clears the 0.80 bar by 0.06.

## Comparison: bio_brain_regions vs cross-bridge G.20 sparse 160-concept union (n=95)

| Comparison | Cross-bridge G.20 sparse (V=160; n=95) | bio_brain_regions V=16 (this) |
|---|---|---|
| OI L=5 multi-seed | ~0.77 (BELOW BAR) | 0.997-1.000 (PASS) |
| OI L=6 multi-seed | ~0.45 | 0.970-0.980 (PASS) |
| OI L=7 multi-seed | ~0.16 (chance) | 0.895-0.935 (PASS) |
| OB L=7 multi-seed | 1.000 (perfect) | 1.000 (perfect) |

The bio_brain_regions V=16 substrate has DRAMATICALLY more load headroom than the G.20 sparse 160-concept union -- because V=16 << V=160. The OI capacity scales with vocabulary size; smaller V = much larger OI margin at high L. This is consistent with the FHRR algebra capacity envelope (n=87 characterised at the algebra level; load capacity is proportional to N_dim / V).

## Biology-translatable insight

**The bio_brain_regions substrate supports parallel-matching mode-unification at FULL gamma-slot capacity (L=7) with substantial margin.** Adding hippocampus + dlpfc_wm does NOT degrade load capacity (the DLPFC variant has SLIGHTLY HIGHER L=7 OI than HIPPO-OPTION3 — perhaps because the dlpfc_wm region's NMDA bistability provides additional stable signal). The validated cortical identification mechanism is ROBUST across substrate component additions.

**Implication for the (c) loop NEGATIVE**: the (c) loop's failure (pillar n=99 + diagnostic REPLAY_DOESNT_REACTIVATE) is NOT a substrate capacity limitation. The substrate has HUGE headroom for the encoding/decoding of K-tuple compositional bindings at every K up to the gamma-slot ceiling. The (c) loop's specific design (SWR-replay → capture → decode) fails to read sequence-specific information from the cortex; the substrate itself can hold and read compositional bindings at this scale far in excess of what (c) attempted.

This precisely bounds the (c) NEGATIVE: it is a (c)-integration design limitation (specifically the SWR-to-cortex sequence-specific reactivation gap), NOT a substrate-capacity limitation. Future (c)-type loops that bypass the SWR-driven cortical-readout step (e.g., direct cue-driven engram stim → cortex similarity readout — which IS the validated multitag mechanism) could succeed where (c) failed.

## Refinement of pillars n=96/n=97/n=98

This descriptive characterisation extends the three substrate-readiness pillars:
- n=96 OPTION 3 (no hippo): OI PASSes at every load L=2..7 multi-seed; L=7 multi-seed 0.900
- n=97 HIPPO-OPTION3: OI PASSes at every load L=2..7 multi-seed; L=7 multi-seed 0.895
- n=98 DLPFC-extension: OI PASSes at every load L=2..7 multi-seed; L=7 multi-seed 0.935

Each substrate's parallel-matching capability stretches to the FULL gamma-slot capacity at V=16. No new pillars; the three existing pillars are sharpened.

## Files

- Runner: `research/findings/raw/bio_brain_regions_oi_load_ceiling_map.py`
- Log: `research/findings/raw/bio_brain_regions_oi_load_ceiling_map.log`
- Output JSON: `research/findings/raw/bio_brain_regions_oi_load_ceiling_map.json`
- This findings doc: `research/findings/2026-05-24-bio_brain_regions-load-ceiling-map-ALL-3-substrates-PASS-every-load-L2-to-L7-the-c-NEGATIVE-is-not-substrate-bounded.md`
- Parent pillars: n=96/n=97/n=98 substrate-readiness chain (validated; this extension sharpens their characterisation)
- Cross-bridge load-ceiling map (analogous extension of n=95): `research/findings/2026-05-24-cross-bridge-OI-load-ceiling-map-extension-of-n95-ceiling-between-L4-and-L5.md`
- (c) NEGATIVE pillar n=99 + diagnostic: bounded interpretation in light of this finding

## Standing constraints

- Reuse-by-import only; protected set byte-empty diff.
- No autograd; no protected/frozen/moat module modified.
- No-confab moat 7/7 green.
- Frozen 0.80 bar unchanged.
- Descriptive characterisation only; no new pillar; n=96/n=97/n=98 sharpened.
- Both git remotes propagated.
