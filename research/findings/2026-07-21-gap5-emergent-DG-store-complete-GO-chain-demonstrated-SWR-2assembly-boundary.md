# gap#5 emergent-DG COMPLETABLE STORE — the emergent-selected assembly BTSP-stores + bistable-completes (mechanism 6/6); the select→store→complete chain is demonstrated end-to-end on one spiking substrate. Two-assembly SWR discrimination = a precise partially-closed BOUNDARY. NO `sim/` edit.

**2026-07-21.** R4 (research-gate ranked, following R1's emergent SELECTION GO). The last piece the emergent-DG needed:
store the transient-drive-SELECTED emergent assembly as a SELF-SUSTAINING COMPLETABLE attractor (one-shot BTSP), so a
partial cue completes it. It WORKS — the gap#4↔#5 BTSP unification now runs on EXPERIENCE-DERIVED assemblies, not the
pre-assigned masks it was validated on.

## Index-space gate: PASS (6/6) — R1's mossy-detonator bridge and the BTSP bridge place CA3 at identical global indices.

## STORE + COMPLETE — MECHANISM GO 6/6 (verified in `r4_final_6seed.json`)
The emergent-selected sparse assembly (R1: n_ca3=2000, d0.02/w3000/acw12/θ0.15/STP-off; ~20-58 cells) BTSP-stores as a
bistable completable attractor. 6-seed:
| gate | result |
|---|---|
| cue-gated (partial cue → held members fire) | held_cue **0.132-0.199** (mean 0.173) |
| BISTABLE (no-cue → silent rest) | held_nocue **0.000 all 6** ✓ |
| SPECIFIC (permuted cue) | held_perm **0.000 all 6** ✓ |
| load-bearing (no-encode → collapse) | noencode_cue **0.000 all 6** ✓ |
| BTSP-stored within-assembly weight | w_within 40-141 |
- **The MECHANISM is 6/6** — cue-gated, bistable, specific, load-bearing on every seed. **Magnitude 4/6 ≥0.15** (0.198,
  0.175, 0.194, 0.199 pass; 0.132, 0.141 low), strict-0.20 bar 0/6 — the SAME "mechanism-6/6, magnitude-marginal"
  profile as the reference PRE-ASSIGNED BTSP completion (0.166-0.191, itself called "mechanism 6/6"). The 2 low seeds
  track the smallest emergent assemblies + lowest w_within (magnitude is emergent-size-variance-bound).
- **One real substrate finding:** the sparse emergent assembly (~30 cells) needs storage recurrent density ≥0.35 for
  adequate cue→held fan-in (at the reference's 0.05 it gives cue 0.05; the reference's 240-cell assembly completes at
  0.05 because its fan-in is 6× higher). Density is a substrate parameter — indices unchanged.
- ⇒ **the emergent-DG select→store→complete CHAIN is demonstrated end-to-end on one spiking substrate:** R1 SELECTION
  (sparse detonator, GO 6/6 core) → R4 STORE (one-shot BTSP) → bistable COMPLETE (mechanism 6/6). Self-organized memory
  codes FROM EXPERIENCE, stored and completable — the emergence-bar for the completion half, met.

## Two-assembly SWR readout — BOUNDARY (partially closed, mechanism named)
The CLOSED SWR readout (learned Schaffer + E%-max topk=0.1 + `SWR_PHASE2_NOSTP`) does NOT discriminate two CO-STORED
emergent assemblies: ca1_match 0.68-0.80 ≈ ca1_cross 0.43-0.78. Root cause (SWR_DEBUG): a partial cue of A CROSS-COMPLETES
B through the shared dense CA3 recurrent (latched [31 of A, 18 of B]) — the density needed for sparse-assembly completion
also COUPLES the assemblies, and `structural_sep` isolates the assembly-UNION from non-members, not the assemblies from
EACH OTHER. The principled fix — zero BETWEEN-assembly recurrent edges (`interassembly_isolate`, additive default-off,
= realizing R1's emergent separation in RECURRENT space = the DG's between-memory pattern-separation) — removes the
primary cross-completion (partial cue of A now latches [30, 0]) and discriminates CLEANLY on 2/6 (seed 100: match
0.75/cross **0.06**, ratio 12.5×) but stays SEED-FRAGILE (cross 0.51-0.62 on the rest; the smaller emergent assembly
avalanches under co-storage). **Precise remaining gap: co-storing two SIZE-VARIABLE emergent assemblies as
independently-addressable attractors** (avalanche-stable co-storage) — the reference SWR-GO 10.79× relied on a controlled
`swr_disjoint` pool + tight Hebbian regime the emergent path lacks.

## Verdict
- **Emergent-DG completable STORE: GO (mechanism 6/6; magnitude marginal 4/6, matching the reference).** The chain
  select→store→complete works on one spiking substrate with experience-derived assemblies.
- **Two-assembly SWR discrimination: BOUNDARY, partially closed** — between-assembly recurrent isolation is the named
  mechanism (clean 2/6); avalanche-stable co-storage of size-variable emergent assemblies is the precise next step.
- Files: `_gap5_r4_emergent_btsp_store.py` (wrapper) + additive default-off `interassembly_isolate` on
  `_riii_ca3_synchronous_assembly_derisk.py` (byte-identical off, no existing caller passes it). NO `sim/` edit.
