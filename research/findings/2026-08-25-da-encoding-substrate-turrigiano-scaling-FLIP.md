---
type: finding
status: contributing
date: 2026-08-25
mechanism: da-gated-encoding
lane: integration
integration_faculty: da-gated-encoding
seeds: [42, 43, 44, 100, 101, 102]
verdict: GO
instrument: the CONTROL-ARM decomposition is OFF-arm (encoding-gate disabled, write gain pinned 1.0) vs ON-arm
  (RAW DA gate + the on-substrate Turrigiano rule) recall at MATCHED read-damage sigma, on the production
  OneBrainComposer magnitude store. THREE instrument properties carried per (seed,sigma): (1) the moat is decomposed
  into encoding-INTRODUCED leaks (GO requires 0) vs BASELINE read-floor artifacts (excluded); (2) the ON-arm derivation
  is cross-checked byte-equal (max|dw|=0.0) to a REAL from-scratch encoding_gain_fn + apply_homeostatic_scaling build;
  (3) NEW 2026-08-25 -- the stress-net recall is TARGET-BLOCK ATTRIBUTED: a recall OFF produces via a FOREIGN engram's
  damaged decode that coincidentally matches the cue (a confabulation the no-confab moat exists to suppress) is
  reported (confab_off_total) and EXCLUDED from the genuine stress net; a GENUINE regression is a fact OFF recalls via
  its OWN target engram that ON then loses.
runner: research/runners/_da_encoding_leansoak.py --substrate-scaling
artifacts:
  - research/findings/raw/_da_encoding_leansoak/soak_substrate.json
external: Turrigiano 2008 "The Self-Tuning Neuron" (Cell 135(3):422-435) -- homeostatic synaptic scaling: a
  MULTIPLICATIVE scaling of a neuron's synapses toward an activity SET-POINT that PRESERVES their relative strengths
  while preventing runaway weakening (Turrigiano & Nelson 2004 Nat Rev Neurosci 5:97-107; Turrigiano 2012 CSH Perspect
  Biol a005736). DA-gate anchor unchanged: Lisman-Grace hippocampal-VTA loop / Kandel D.16. Reuses the in-repo I-7-b
  gain slope (k_DA=2.0) and the #76/#79 spiking DA-mode read.
supersedes: none -- EXTENDS + RESOLVES
  research/findings/2026-08-21-da-gated-encoding-flip-gate-UNDEFINED-moat-leak-and-sigma-dependent-benefit.md and the
  lever-2 host-proxy preregistration (research/findings/raw/_da_encoding_lever2/HARVEST_NOTES_preregistration.md). The
  UNDEFINED verdict named two next levers -- (1) verify the instrument at the leak points; (2) an on-substrate
  homeostatic-scaling companion -- and this finding delivers BOTH: the host-proxy homeostat is replaced by a genuine
  synaptic-scaling rule, and the residual raw stress-net violation is proven to be a foreign-block confabulation.
---
# DA-gated-encoding default-ON flip gate: the on-substrate Turrigiano synaptic-scaling rule + the instrument fix (6-seed)

## Verdict

**GO (6-seed cupy, decisive).** The magnitude-store no-regression flip gate is cleared by an ON-SUBSTRATE Turrigiano
synaptic-scaling rule (NOT the host-arithmetic proxy), and the prior UNDEFINED verdict is resolved: its raw stress-net
violations are FOREIGN-BLOCK CONFABULATIONS in the OFF control arm, not ON memory regressions.

Artifact `research/findings/raw/_da_encoding_leansoak/soak_substrate.json` (backend cupy, per its `.prov.json`).
Reproduce: `SIM_BACKEND=cupy python -m research.runners._da_encoding_leansoak --substrate-scaling`. GO bar (all three
met): `preconditions_summary.moat_introduced_total == 0` (met, =0) AND `outcome.stress_net_genuine_violations == 0`
(met, =0, target-attributed) AND OFF byte-identical (by construction; the ON path is entirely inside
`apply_homeostatic_scaling`, called only when enabled). The ON-arm derivation is cross-checked byte-equal
(`derivation_cross_check.byte_equal == true`, max|dw|=0) to a REAL from-scratch encoding_gain_fn +
apply_homeostatic_scaling build. Confirmed independently on a numpy 6-seed de-risk (same verdict + same per-seed
decomposition), and the GO is ROBUST to the down-regulation exponent (beta_down=0, the pure recall-safe floor with no
down-regulation, is also GO -- the floor does the safety work, so the result is not a tuned operating point).

<!--derived--> (the per-sigma salience aggregates below are 6-seed sums over the artifact's per_seed sweep, of 54 = 9 facts x 6 seeds.)
The rule does not merely avoid regression -- it IMPROVES recall robustness across the damage knee (ON vs OFF of 54):
sigma 0.75 -> 46 vs 43; 1.0 -> 39 vs 33; 1.5 -> 24 vs 13; 2.0 -> 19 vs 8. The only raw violations are at sigma 4.0/6.0
(the noise floor, OFF recall 1-2 of 54), both target-attributed confabulations (genuine OFF recall 1/0; ON abstains).

## What was built -- the mechanism, on the substrate, not host arithmetic

The lever-2 homeostat (`webapp/da_encoding_drives_chat.homeostatic_step`) was a documented HOST PROXY: a feed-forward
`multiply + clip` on the DA scalar at write time (a running EMA `mu` of the DA-derived salience `r`). It reduced the
low-sigma regression but stayed UNDEFINED, and the honest next target it named was the on-substrate spiking
synaptic-scaling rule.

`OneBrainComposer.apply_homeostatic_scaling()` (research/runners/one_brain_composer.py) is that rule -- a FEEDBACK rule
on the synaptic STATE, not on the DA reading:

<!--derived--> (the readout-activity constant below is a local numpy diagnostic-probe measurement (seed 43), not a soak-artifact value; it establishes that the neural read is unconfounded by phase pattern and linear in encoding strength.)
1. **SENSE (neural read).** `_measure_block_readout(i)` kicks each stored engram's trigger, resonates, and reads the
   mean `|Z|` over that engram's D readout neurons off the bridge membrane -- a genuine measurement of the engram's
   postsynaptic readout activity (verified: CONSTANT ~208 for a unit write, independent of the fact's phase pattern;
   exactly LINEAR in the encoding strength, `A_i = A_unit * g_i`).
2. **SET-POINT.** `_homeo_setpoint()` = the readout activity of a UNIT-magnitude engram (measured from a
   unit-normalized reference) -- the intrinsic functional level, independent of the stored DA distribution, so a tonic
   fact maps to `s=1` (byte-safe).
3. **ACTUATE (synaptic scaling).** Multiplicatively rescale each engram's store synapses toward the set-point: a WEAK
   engram (a low-DA fact the DA gate wrote at g<1) is scaled UP to the set-point -- the recall-safe FLOOR, now EMERGENT
   from measured activity, NOT a host `g_floor` clip; a STRONG engram is partially down-regulated by
   `(A*/A_i)^beta_down` (beta_down=0.25) which PRESERVES the relative DA-salience ORDER while pulling the extreme toward
   the set-point.

The sensed variable is postsynaptic activity; the actuator is the synaptic weight -> a faithful Turrigiano synaptic
scaling (multiplicative, activity-set-point, relative-strength-preserving). This is what clears the "host shortcut"
standard: the scaling factor is COMPUTED FROM MEASURED NEURAL ACTIVITY on the substrate, not from a Python EMA of the
DA scalar. Effective ON gains under the rule (beta_down=0.25) are reported in the soak artifact's
`config.substrate_scaling.effective_on_gains` (high-DA down-regulated, low-DA floored to unit, tonic unchanged).

DEFAULT-OFF is byte-identical by construction: `homeostatic_scaling=False` -> `apply_homeostatic_scaling` is never
called -> `store_conns` is untouched (confirmed: a fresh default build's store is unchanged).

## The instrument fix -- the UNDEFINED verdict was a foreign-block confabulation

<!--derived--> (this seed-43 / sigma-6.0 mechanism trace is a local numpy diagnostic reproduction identifying WHICH engram produced OFF's answer -- not a soak-artifact aggregate; the soak carries the confab decomposition as `confab_off_total` + per-seed `regressed_genuine_idx`/`confab_off_idx`.)

The prior UNDEFINED finding's named lever #1 was "verify the instrument at the leak points first." Done. The single raw
stress-net violation (seed 43, sigma 6.0, the noise floor where OFF recall is 1 of 9) decomposes on inspection of WHICH
engram produced OFF's answer:

- **OFF:** block 6's own trace (`dog see cat`) is destroyed at sigma 6.0; a DIFFERENT engram -- block 0 (`dog eat
  grass`, unit magnitude) -- mis-decodes under damage to `(dog, see, cat)`, spuriously matching the cue `(dog, see)`
  and returning `cat`, which COINCIDENTALLY equals block 6's true patient. OFF "recalls" via a foreign-block
  confabulation.
- **ON:** the DA gate boosts block 0 -> higher SNR -> it decodes CORRECTLY (`dog eat grass`) -> no longer spuriously
  matches `(dog, see)` -> the store correctly ABSTAINS.

So the "regression" is DA-gating REMOVING a lucky confabulation (ON abstaining honestly) -- the no-confab moat working
BETTER, not a memory regression. The soak now carries TARGET-BLOCK attribution (`_select_and_read`): `genuine[i]` = the
fact's OWN engram (block i) was selected AND decoded right. The GO is on `stress_net_genuine_violations`; the RAW
stress-net + `confab_off_total` are reported alongside. NB: the host-proxy homeostat, under this corrected instrument,
is ALSO a genuine GO on seed 43 (raw=1, genuine=0, confab=1) -- the whole UNDEFINED verdict was that one confabulation.

## 6-seed table (cupy, decisive)

<!--derived--> (per-seed rows aggregate the artifact's per_seed sweep; the TOTAL row's fields are the artifact's `preconditions_summary` / `outcome` scalars.)

| seed | clean_reg | stress_net_genuine | stress_net_raw | confab_off |
|------|-----------|--------------------|----------------|-----------|
| 42   | 0 | 0 | 0 | 0 |
| 43   | 0 | 0 | 1 | 1 |
| 44   | 0 | 0 | 0 | 0 |
| 100  | 0 | 0 | 0 | 0 |
| 101  | 0 | 0 | 0 | 0 |
| 102  | 0 | 0 | 1 | 3 |
| **TOTAL** | 0 | **0** | 2 | 4 |

`moat_introduced_total = 0` (moat_baseline_total = 1, a control-arm read-floor artifact, excluded). The GENUINE
stress-net (target-attributed) is clean on every seed; the 2 raw violations are all confabulations, decomposed per
seed.

## What was flipped (and what was deliberately NOT)

- **FLIPPED: the on-substrate homeostat is now the default when the coupling runs.** `BRAIN_DA_ENCODING_SUBSTRATE`
  defaults to on (`webapp/da_encoding_drives_chat.da_encoding_substrate_enabled`); `install_encoding_gain` arms
  `OneBrainComposer.homeostatic_scaling`, the per-write gain carries the recall-safe floor, and
  `apply_substrate_homeostasis(chat)` runs the genuine synaptic-scaling consolidation pass. `=0` falls back to the
  lever-2 host-proxy EMA. This is ZERO blast radius: it only changes WHICH homeostat runs WHEN encoding is enabled.
- **NOT flipped: `da_encoding_enabled()` stays default-OFF.** The soak clears the flip gate, so the faculty default-ON
  (ledger `on_by_default`) is now the UNBLOCKED owner product decision -- but flipping THAT default silently breaks the
  byte-identical baseline (it moves from UNSET to `=0`, so `_da_encoding_wired_verify` + `_wave4_composed_flip_noregression`
  OFF arms, which rely on unset==off, must pin `BRAIN_DA_ENCODING=0` first) and changes the response shape of every
  turn. That is a coordinated change for the owner/parent, not a silent subagent flip. The wire-in verifier remains GO
  with these changes (OFF byte-identical, ON g_high 2.48 > g_low 1.0, lesion severs).

## Next mechanism (no-defer)

The substrate rule is validated on the BATCH magnitude-store soak (store the battery, apply the scaling pass once, read
under damage) -- the flip gate. Turrigiano scaling is biologically SLOW/OFFLINE (hours-days, during sleep), so its live
realization is a CONSOLIDATION-time pass (`apply_substrate_homeostasis`), not a per-turn call (a per-turn full re-scale
would compound toward unit). The two open rungs, both banked not deferred: (1) wire the consolidation TRIGGER (which
event fires the pass live) -- the per-write recall-safe floor keeps facts safe in the meantime; (2) pin the two OFF-arm
verifiers to `=0` and flip the faculty default-ON (the owner product call the soak has now unblocked).
