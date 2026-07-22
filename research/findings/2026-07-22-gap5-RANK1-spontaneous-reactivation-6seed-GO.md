# gap#5 RANK 1 — SPONTANEOUS single-assembly reactivation is 6-SEED GO: a stored bistable CA3 assembly self-reactivates under weak non-specific noise, biologically, with every retracted-confound control clean

**2026-07-22 (autonomous, CPU/numpy, coexisting with the production LM run — zero GPU risk).** RANK 1 of the SWR
generative-replay ladder (the first rung of gap#5 CA3-completion/imaginative-replay): does a stored, closed, bistable
CA3 assembly SPONTANEOUSLY reactivate under weak non-specific background (NO cue), specifically and discretely? Earlier
this was a 1-seed clean GO; this is the **6-seed confirmation** at the standing rule.

## Result — 6/6 GO (seeds 42, 43, 44, 100, 101, 102)
`_gap5_spontaneous_reactivation_derisk.py`, n_ca3=2000, n_mem=2, Poisson background (rate 0.015, 1500 pA, dur 10),
rest 1500 steps, `--skip-poscontrol`:

| seed | memb (overlap w/ stored) | spec (vs random) | events | rate/1k | NO-NOISE (acid) |
|------|--------------------------|------------------|--------|---------|-----------------|
| 42   | 0.319 | +0.271 | 3 | 2.00 | 0 events ✓ |
| 43   | 0.328 | +0.278 | 5 | 3.33 | 0 events ✓ |
| 44   | 0.328 | +0.279 | 4 | 2.67 | 0 events ✓ |
| 100  | 0.308 | +0.260 | 5 | 3.33 | 0 events ✓ |
| 101  | 0.305 | +0.255 | 6 | 4.00 | 0 events ✓ |
| 102  | 0.323 | +0.271 | 5 | 3.33 | 0 events ✓ |

Every seed => GO on the PRIMARY gate + all retracted-confound anti-cheats:
- **specific_events** (the reactivated pattern overlaps the STORED assembly, not random: memb 0.31-0.33 vs rand ~0.05);
- **discrete** (the net rests silent between events — not a self-sustaining runaway);
- **acid_noise_off** — NO-NOISE → 0 events EVERY seed (the decisive test: retires the self-sustaining-artifact confound
  that sank prior sequence-replay attempts — with no noise, there is no reactivation, so reactivation is noise-TRIGGERED
  completion, not endogenous runaway);
- **frozen_ok** (plasticity byte-frozen during rest — retires the plasticity+noise/Wang confound);
- **dendrite_reset_ok** (retires the `_hard_silence` bug);
- **noencode_retired** (NO-ENCODE → 0 events: the completing basin is the learned store, not the substrate);
- **shuffle_retired** (SHUFFLED-W → specific=0: the LEARNED weight structure, not the weight budget, carries it);
- **permuted_retired**.

**Honest nuance:** the SECONDARY `learned_weight_carries` diagnostic (NO-STRUCT: does the learned attractor carry the
selectivity even without the structural_sep/selective_inhib scaffolding?) is **5/6** — True for 42/43/44/100/102, False
for seed 101. The primary GO (specific noise-triggered reactivation + all confound controls) is **6/6**; only this
one secondary robustness check varies at one seed. Not GO-changing; reported for completeness.

## Meaning
The imagination line's first rung is SOLID at the standing rule: a stored memory spontaneously "comes back" on its own
under the brain's ongoing background activity — the substrate of SWR replay / imaginative reactivation — realized fully
on the bistable CA3 substrate (dendritic-plateau two-compartment neurons + the committed BTSP store), biologically
(noise-triggered attractor completion, no host shortcut), with the artifact confounds that killed prior attempts all
retired. RANK 2 (ORDERED sequence replay A→B→C) builds on this: its forward-chain mechanism is already validated
(asym=+2.53) and its within-reactivation blocker is isolated to a `_prepare` encode-reuse fix (separate finding, in
progress). NO `sim/` edit. Runner: `research/runners/_gap5_spontaneous_reactivation_derisk.py`.
