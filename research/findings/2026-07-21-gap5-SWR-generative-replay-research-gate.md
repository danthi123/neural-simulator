# gap#5 SWR generative-replay loop — research gate (a-1 RAG + biology): the open piece is SPONTANEOUS no-cue reactivation; RANK 1 = single-assembly spontaneous reactivation on the CLOSED bistable store (cheapest, NO `sim/` edit, anti-cheats pre-mapped by the 3 retractions)

**2026-07-21.** GPU-free research gate (during the production LM run). The board's open gap#5 piece: "DG-selected assembly
→ SWR replay loop → console" — the SPONTANEOUS (no-cue) generative replay / imaginative replay.

## a-1 RAG reconciliation (drift-#12 — do NOT re-derive the retracted confounds)
- **3 RETRACTED confounds** (all faked "a stored attractor that reactivates genuinely"): (1) the self-sustaining artifact
  (an attractor that never turned off — the completion TRILEMMA, a point soma can't be bistable); (2) the Wang-NMDA
  plasticity+noise confound (frozen+noise-off gave 0.000; OU-noise-ON gave 0.5 EVERYWHERE = uniform noise, not basin-
  selective); (3) the `_hard_silence` dendritic-reset bug (a latched plateau persisted through "silence"). Plus older
  decisive negatives: the on-bridge generative-replay loop at CHANCE ("the SWR trigger does not drive sequence-specific
  cortical activity"); the naive "add a CA3 drive" fix REFUTED (post-seed window → chance); v16 sequence-storage BOUNDARY;
  reverse-replay NULL (both pre-bistable-keystone).
- **CLOSED machinery to build ON:** CA3 completion CLOSED (intrinsic dendritic bistability + KIR down-state = a GENUINE
  silent rest state — the missing ingredient that makes a non-artifact spontaneous-reactivation test possible); emergent-DG
  SELECTION GO 6/6; emergent completable STORE (BTSP) GO; SWR→CA1 readout "6/6 given cues" but AUDIT-NARROWED (completion-
  driving-CA1 UNdemonstrated — no completion-lesion/permuted-cue CA1 control).
- **Honest state:** every existing SWR result is CUE-driven completion→CA1; the genuinely-open piece (spontaneous no-cue
  reactivation) has ONLY been tested in the 3 retracted/confounded runs. FHRR-algebra imaginative recombination is GO on
  numpy (the capability at the algebra grain); the SWR loop is its biology-purity spiking realization (open per THE LAW).

## Biology reframe (Ecker 2022 eLife = the buildable template)
SWR generation is a SPONTANEOUS recurrent buildup (nonspecific noise + the learned attractor basin + the bistable down→up
transition + PV-basket disinhibition; Schlingloff-Buzsáki 2014). Ecker et al. eLife 2022 (71850): a spiking CA3 with
recurrent weights structured by a SYMMETRIC-STDP rule AUTONOMOUSLY generates SWRs + replays sequences FORWARD/REVERSE from
a nonspecific drive — "sufficiently strong, properly structured recurrent excitation" is necessary+sufficient (our exact
template). Order stored at encoding via theta-compression/phase-precession → asymmetric forward links (BTSP) + symmetric
(reverse). Imagination = the SAME spontaneous machinery traversing a NOVEL path through the learned assembly graph
(Gupta 2010; Dragoi-Tonegawa preplay; George-Barry-Behrens 2023). **Ladder (cannot skip):** (i) single-assembly
spontaneous reactivation → (ii) sequence replay (forward/reverse) → (iii) generative/novel-path replay.

## RANK 1 (recommended cheap-first, NO `sim/` edit) — single-assembly SPONTANEOUS reactivation on the CLOSED bistable store
The FIRST genuinely-untested spontaneous piece. Additive default-off "rest" phase: FREEZE plasticity + reset dendritic
state (the committed `_hard_silence` fix) → run ~1500 rest steps with WEAK NON-SPECIFIC background (`enable_ou`, low σ; NO
cue/`recall_drive`) → detect reactivation EVENTS (sliding-window CA3 co-firing bursts) → score member-fraction +
assembly-specificity. Config = the CLOSED completion GO_CFG (`n_ca3=2000, ca3_density=0.05, assembly_frac=0.12,
encode_drive=3000, no_sync=True, recall_k_thresh=110, bistable=True, structural_sep=1, selective_inhib=True,
plateau_self_regen=0.15, apical_kir_g=3.0, apical_gc=1.0, apical_gc_read=5.0`). Driver: new
`_gap5_spontaneous_reactivation_derisk.py` (reuse `run()` from `_riii_ca3_synchronous_assembly_derisk.py`).
- **6-seed GO gate:** discrete spontaneous events OCCUR (rate>0) + assembly-SPECIFIC (member-fraction ≫ chance, spec>margin)
  on ≥5/6 + the net RESTS silent between events (low duty cycle, discrete events NOT a continuous ON state).
- **Anti-cheats (each retires a named retracted confound):** NO-ENCODING→no events (noise-artifact); PERMUTED-ASSEMBLY→
  chance (non-specific completion); FROZEN plasticity MANDATORY (Wang confound); NO-NOISE→silent (self-sustaining artifact);
  SHUFFLED within-assembly WEIGHTS→no specific events (OU-uniform confound); dendritic-reset verified (`_hard_silence` bug).
- **BONUS:** feeding SPONTANEOUS (not cued) reactivations into the CLOSED CA1 readout also closes the 8-skeptic audit's
  missing completion-lesion + permuted-cue CA1 controls (two birds).
- **Failure levers (all characterized, no new mechanism):** noise never ignites (OU σ / `apical_gc_read`); avalanche
  (`selective_inhib`/`recall_k_thresh`); non-specific (needs emergent-DG separated codes).

## RANK 2 (moderate) sequence replay (asymmetric-BTSP chain, Ecker) · RANK 3 (deep frontier) generative/novel-path replay.
## VERDICT: RANK 1 first — near-term, cheapest, NO `sim/` edit, anti-cheats pre-mapped, advances the loop WITHOUT re-deriving retracted work. RANK 2 moderate (re-de-riskable on the bistable substrate; old v16 boundary + R4 avalanche wall). RANK 3 deep (proven only at FHRR grain). GPU-preferred (n_ca3=2000 bistable); numpy-CPU smoke at n_ca3=500-1000 valid pre-check.

## RANK 1 CPU PRE-CHECK RESULT (2026-07-22) — PROMISING/PARTIAL: spontaneous basin-selective reactivation demonstrated (NOT the retracted artifacts), but the LEARNED-weight contribution not yet cleanly isolated
Built `research/runners/_gap5_spontaneous_reactivation_derisk.py` (NO `sim/` edit): BTSP-encode the bistable store → REST
phase (freeze plasticity + `_hard_silence` + weak NON-SPECIFIC background, NO cue) → detect discrete CA3 co-firing events.
CPU pre-check (n_ca3=2000, n_mem=2, seed 42, Poisson r=0.015/pa=1500/dur=10):
| condition | events | member_frac | random | cross-asm | duty | notes |
|---|---|---|---|---|---|---|
| **GO (pa 1500)** | 3 (2 specific) | **0.319** | 0.048 | 0.091 | **0.079** | 6.6× random, 3.5× competing assembly; discrete (silent between) |
| **NO-NOISE (acid)** | **0** | 0.000 | — | — | 0.000 | somatically SILENT (pop 0.0002) → NOT the self-sustaining artifact ✓ |
| **NO-ENCODE** | **0** | 0.000 | 0.048 | — | 0.000 | encoded store necessary → not the noise-artifact ✓ |
| SHUFFLED-W | 3 | **0.213** | 0.048 | — | 0.079 | PARTIAL collapse (still 4.4× random) — the caveat |
- **Two load-bearing acid tests PASS** (verified, not asserted): NO-NOISE→silent (retires the self-sustaining artifact),
  NO-ENCODE→0 (retires the noise-artifact); FROZEN plasticity byte-hash-verified (retires the Wang confound); dendritic
  reset clean. ⇒ **the first genuine (non-artifact) spontaneous basin-selective reactivation in the project.**
- **HONEST CAVEAT (subagent self-flagged): the SHUFFLED-within-weights control only PARTIALLY collapses** (0.319→0.213) —
  the selectivity is mostly carried by the assembly's STRUCTURAL wiring (`structural_sep` + `selective_inhib` survive a
  weight-shuffle), the LEARNED within-assembly weights add only ~33%. So "the LEARNED attractor selectively reactivates"
  is NOT cleanly demonstrated at single-seed. VERDICT: **PROMISING/PARTIAL, not a clean GO.**
- Config corrections banked: the completing store is the BTSP encode (NOT plain Hebbian, which never completes); needs
  n_mem=2 + n_ca3=2000 (500/1000 too weak). Op-point narrow (pa=3000 → 0 events).
- **NEXT (queued behind the live production run):** (1) TIGHTEN the shuffle control — add a structure-matched control that
  ALSO removes `structural_sep`+`selective_inhib`, so the shuffle must drop member_frac toward random (0.05), isolating the
  learned-weight contribution; (2) 6-seed GPU confirm: `SIM_BACKEND=cupy python -m
  research.runners._gap5_spontaneous_reactivation_derisk --seeds 42 43 44 100 101 102 --n-ca3 2000 --n-mem 2 --noise
  poisson --poisson-rate 0.015 --poisson-pa 1500 --poisson-dur 10 --rest-steps 1500`; (3) fix the NO-NOISE bridge-reuse
  numerical artifact (own fresh bridge). ⇒ RANK 1 is the honest cheapest first move demonstrated (spontaneous reactivation
  real, not the confounds); the learned-weight-isolation + 6-seed are the close. RANK 2 (sequence) / RANK 3 (imagination)
  follow. Per THE LAW: a promising-partial with the next control named, NOT a wall.
