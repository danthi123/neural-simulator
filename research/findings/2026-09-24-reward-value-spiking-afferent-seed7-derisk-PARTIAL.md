---
type: finding
status: partial
lane: load-bearing
date: 2026-09-24
mechanism: A10 (midnight plan S15c) seed-7 capability-gate de-risk of the surprise-organ-driven SNc reward/context afferent for da-mode-drives-response, per research/findings/2026-09-24-reward-value-spiking-afferent-PREREGISTRATION.md
seeds: [7]
artifacts:
  - research/findings/raw/_reward_value_afferent_derisk/s7.json
---

# A10 reward/value afferent: seed-7 de-risk result -- (A) GO, (B) GO, (C) UNDEFINED at the pre-registered threshold

Governed by `research/findings/2026-09-24-reward-value-spiking-afferent-PREREGISTRATION.md` (committed first).
Runner: `research/runners/_reward_value_afferent_derisk.py`. Seed 7 only -- a dev/calibration seed, **not** a
6-seed gate verdict. Verdict object: `research/findings/raw/_reward_value_afferent_derisk/s7.json`.

## Headline (exact wording, per docs/TERMS.md -- "GO" is never lifted out of a run whose gate read UNDEFINED)

The runner's own gate reads **UNDEFINED**, not GO: (A) and (B) pass; (C), the pre-registered LESION criterion
(`differential < 1e-6`), does **not** hold. The lesion removes most, not all, of the confirm-vs-contradict
differential.

## (A) OFF byte-identity -- GO

No `reward_value` key on either turn; `da_drives_confirm` and `da_drives_contra` are (byte-for-byte) the SAME
dict (`afferent_pA=427.6363636363635`, `mode="neutral"`, `lead=""`) -- confirming, as a side-effect, the residual this
lane targets: the pre-existing host `engagement_of()` scalar cannot distinguish "the dog chase the cat" from
"the dog chase the fish" (both are 3 fresh content tokens -> identical novelty+richness), so
`da-mode-drives-response`'s afferent (and hence its mode/suffix) is IDENTICAL on a CONFIRM vs a CONTRADICT turn
when this module is off.

## (B) ON, load-bearing -- GO

`reward_value.source == "surprise"` on both turns. CONFIRM: `surprise_hz=0.3472222222222222`,
`normalized=0.0646029609690444`, `pa=90.44414535666216`, `surprised=false`. CONTRADICT:
`surprise_hz=5.150462962962964`, `normalized=0.9582772543741588`, `pa=1341.5881561238223`,
`surprised=true`. The differentiation `engagement_of()` could not produce (A) is produced cleanly by the
surprise-organ read, and `surprised` flips exactly as the organ's own de-risk predicts.

## (C) LESION -- does NOT fully collapse (the pre-registered criterion is UNDEFINED, not disproven)

Under `sorg.judge(..., lesion=True)` (the organ's own per-call prediction-edges-zeroed twin):
CONFIRM `normalized=0.8613728129205921`, CONTRADICT `normalized=0.9690444145356663` -- both HIGH (as expected:
losing the top-down inhibition, both reads are dominated by direct excitation), but NOT IDENTICAL. The residual
differential is `0.10767160161507416`, versus `0.8936742934051144` live -- `tools.lab.attributable_to` reports a
`lesion_attribution_fraction` of `0.8795180722891565` (about 88%) removed by the lesion; about 12% survives it.

### Why (a real mechanism, not a bug in this module -- named, not hidden)

`surprise_production_organ.py`'s own block assignment gives the STORED patient ("cat") a CUE-ADDRESSABLE,
homeostat-gain-EQUALIZED block (`_block_for(cue_addressable=True)`, one of the `n_trained=8` blocks the
per-block prediction-gain equalizer tunes at build time), while a NOVEL asserted patient ("fish") gets a SPARE
block (`_block_for(cue_addressable=False)`) that never runs through that equalizer. With the top-down
INHIBITORY prediction edge zeroed (this lesion), each block's surprise read is driven by direct excitation
alone, and the two block POPULATIONS are not perfectly matched in gain/heterogeneity -- a small, honest
residual survives that has nothing to do with prediction/inhibition, purely the trained-vs-novel block
asymmetry the homeostat was never asked to equalize in this direction. This is a property of the ALREADY-GO
surprise organ (`research/findings/raw/_surprise_organ_homeostat/summary.json`, 6/6-GO) that this module
inherits, not a new defect this wiring introduces.

### What this changes about the claim

The pre-registration's (C) criterion (`< 1e-6`) was too strict an operationalization of "collapses" -- it
assumed the ONLY source of the confirm/contradict differential is the prediction/inhibition pathway. The 88%
attribution says the coupling IS substantially (not exclusively) attributable to the live prediction-mediated
surprise read. Per `docs/TERMS.md`, this is reported as **UNDEFINED at the pre-registered threshold**, not
rounded up to GO and not called a NO-GO (the mechanism plainly differentiates confirm/contradict in (B), and
most of that differentiation is lesion-sensitive).

## Next rung (named, not attempted tonight -- RAM-tight box, other lanes' compute contention)

A CLEANER lesion control: assert `p_asserted` into a SECOND cue-addressable (trained) block instead of a
NOVEL/spare one (so both arms of the confirm/contradict comparison sit in equally-homeostat-equalized
populations), isolating the prediction/inhibition contribution from the trained-vs-novel population asymmetry.
Alternatively, tighten the runner's own criterion to `< residual observed at a matched-block control`, which
requires that control to exist first. Either fix stays inside the ALREADY-GO surprise organ's existing lesion
machinery -- no new lesion mechanism, no scope creep into A10's own coupling.

## Scope reminder (unchanged from the pre-registration)

`BRAIN_REWARD_VALUE_AFFERENT` / `BRAIN_REWARD_VALUE_LESION` remain default-OFF. This is a seed-7 de-risk, not a
6-seed gate; per the midnight plan (S15c), the 6-seed capability gate (seeds 42/43/44/100/101/102) runs at the
frozen SHA `F` in B2b, and default-ON additionally needs a SOUND independent (opus) review -- not run here (see
the pre-registration's declared residual #5; this finding does not change that).
