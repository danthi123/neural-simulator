---
type: finding
status: verified
date: 2026-09-23
mechanism: bank two ALREADY-DECISIVE 6-seed runs of `--readout attention-gated-soft` (the graded/divisive-norm
  attention-gain readout, `_attention_gated_soft_class_read` in `_vision_lindiscrim_readout_derisk.py`) at two
  different front-end operating points, correcting a same-day pre-registration's false "never decisively run"
  premise and non-blind band-writing order, and closing this readout as a NO-GO method (not the capability).
integration_faculty: perception (vision object readout)
lane: perception (vision configural / position-invariant readout)
seeds: [42, 43, 44, 100, 101, 102]
verdict: NO-GO for `--readout attention-gated-soft`, confirmed on BOTH front-end operating points this lane has
  tested it at (12 of 12 seed-runs total). `LEARNED_spkwta_held` (the fully-spiking class-population WTA
  prediction) is at exact chance (0.25) on 11 of the 12 seed-runs, with ONE exception (default front end, seed
  101, 0.2604 -- still statistically indistinguishable from chance at this scale, not a second data point of
  real signal). `LEARNED_linscore_held` (the identical fitted discriminant's raw signed score, read from a
  SEPARATE, never-gated pathway -- `_lin_score_pred`, unaffected by any `--readout` mode) retains real signal on
  11 of the 12 (0.32-0.68); the exception is default front end seed 100 at 0.3229, BELOW this runner's own
  `nogo_floor` of 0.34 -- so that one seed's linscore is not clearly "real signal" by the runner's own threshold
  either, though it is still far above the spkwta collapse on the same seed.
  `scramble_null_pass=1.0` holds on all 12 (the anti-cheat precondition is intact). This is a NO-GO on the
  READOUT METHOD, not a capability abandonment: the signed linear discriminant clearly separates classes
  (linscore), and the plain `--readout linear` spiking port at the SAME satdiv-GO front end already clears this
  same floor decisively (`LEARNED_spkwta_held` mean 0.5729 <!--derived--> across 6 seeds, per the competitive-selection-at-
  satdiv-GO baseline this lane already banked) -- so the fully-spiking WTA port is NOT broken in general, only
  when driven by attention-gated-soft's specific per-class, per-trial gain+normalization stage. See "What
  retraction/correction this makes" below and the follow-up pre-registration for the next mechanism.
runner: research/runners/_vision_lindiscrim_readout_derisk.py
artifacts:
  - research/findings/raw/lanes/perception/conjbind_attngatedsoft_n1152_heldoutpos_scramblenull_6seed.json
    (2026-09-17, default `--s2-norm z` front end, `--ridge 0.5 --n-glimpses 2`; untracked in the primary
    checkout until this session, banked here with a backfilled `.prov.json`)
  - research/findings/raw/lanes/perception/conjbind_attngatedsoft_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s42.json
  - research/findings/raw/lanes/perception/conjbind_attngatedsoft_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s43.json
  - research/findings/raw/lanes/perception/conjbind_attngatedsoft_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s44.json
  - research/findings/raw/lanes/perception/conjbind_attngatedsoft_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s100.json
  - research/findings/raw/lanes/perception/conjbind_attngatedsoft_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s101.json
  - research/findings/raw/lanes/perception/conjbind_attngatedsoft_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s102.json
    (the satdiv-GO front end, `--s2-norm satdiv --s2-satdiv-sigma 8 --s2-satdiv-scale 760 --ridge 1.0
    --n-glimpses 6`; seed 42 was run+committed by the prior round, seeds 43/44/100/101/102 finished on the pool
    after the review landed and are banked here for the first time)
external: NO-EXTERNAL-NEEDED for this finding itself -- it is a BANKING/correction of already-run data (no new
  mechanism lever is proposed here; the retracted prereg's own external citation, Reynolds & Heeger 2009, already
  covers the mechanism being banked). The follow-up pre-registration for the NEW mechanism carries its own,
  genuinely new, external grounding.
builds_on:
  - research/findings/2026-09-23-vision-configural-binding-attention-gated-soft-readout-PREREGISTERED.md
    (⛔ RETRACTED, this commit -- see docs/RETRACTED.md)
  - research/findings/2026-09-16-vision-nglimpses-temporal-evidence-integration-lifts-capability-2of6-to-4of6.md
  - research/findings/2026-09-17-vision-satdiv-divisive-norm-is-the-decisive-lever-for-position-invariant-readout-binding-not-load-bearing-GO.md
  - research/findings/2026-09-09-vision-configural-binding-attention-gated-readout-NEXT-MECHANISM-PREREGISTERED.md
  - research/biology/attention-gated-readout.md (registry entry; this finding's landing updates its
    `current_finding`/`current_status`)
---

# Attention-gated-soft readout: banked NO-GO on 12 of 12 seed-runs across two operating points

## What this corrects (an adversarial review, this session, caught before any commit landed on `main`)

The pre-registration committed at `7a3bcccaa` (2026-09-23) claimed the `--readout attention-gated-soft` readout
"had NEVER been decisively run" and that `before_you_build.sh` + `git log --all --grep` + a filesystem search
"all came back empty for any decisive artifact." **That premise was false.**
`research/findings/raw/lanes/perception/conjbind_attngatedsoft_n1152_heldoutpos_scramblenull_6seed.json`
(mtime 2026-09-17T03:43) is a 6-seed decisive run of this exact readout, on the default `--s2-norm z` front end.
It sat **untracked** in the primary checkout -- never committed, never linked from a finding -- which is exactly
why the filesystem/git search missed it (those searches only see committed/known paths; a raw artifact that was
produced but never banked is invisible to them). `research/queue/dispatch.log` line 20311 confirms the exact
command and dispatch timestamp; a `.prov.json` sidecar is backfilled from that log line in this commit's
predecessor, since the artifact itself carried none.

That 2026-09-17 run's own verdict, unread until now: `LINDISCRIM-READOUT-NOGO`, `capability_go` 0 of 6,
`LEARNED_spkwta_held` at exact chance (0.2517 mean, individual seeds 0.25/0.25/0.25/0.25/0.2604/0.25) while
`LEARNED_linscore_held` carries real signal (0.4462 mean). This directly REFUTES the prereg's second claim too --
that the seed-42 collapse at the satdiv-GO front end was "a genuinely NEW combined operating point this
mechanism's own 2026-09-16 de-risk smoke never tested," implying the collapse might be specific to stacking two
divisive normalizations. The 2026-09-17 run used the DEFAULT `z`-norm front end (no satdiv stacking at all) and
shows the SAME collapse. What IS established, measured directly across both runs: the spiking-port collapse is
not a property of either specific front-end operating point (it reproduces identically on both). WHAT stage of
this readout's net-drive path the collapse is a property of remains a HYPOTHESIS, not yet isolated by an
ablation -- see "Why," below, and its registered test.

**Ordering.** The prereg's bands (a paired-t gate against a fixed `t_crit=2.571` <!--derived-->) were written in the SAME commit
as the seed-42 satdiv-GO run they discuss in prose -- the bands are stated to be evaluated once all 6 seeds land,
but the commit's own prose already describes and interprets the seed-42 number before registration closed. This
finding does not attempt to salvage that gate: with the underlying premise refuted (this is a re-run of an
already-decisively-NO-GO'd method, not an unrun one) and the collapse now confirmed 12/12 seeds across two
operating points, reading the paired-t gate against the satdiv-GO baseline would answer a question ("does this
readout beat the plain linear readout at this one front end") that a clean, un-gated look at the raw numbers
already answers directly and more informatively: it does not, on either front end, ever.

## The full picture: 12 of 12 seed-runs, both operating points

| front end | seed | `LEARNED_spkwta_held` | `LEARNED_linscore_held` | `RATE_lin_ceiling_held` | `spkport_cost` | `capability_go` |
|---|---|---|---|---|---|---|
| default (z-norm, ridge0.5, glimpses2) | 42 | 0.25 | 0.4479 | 0.4062 | 0.1979 | false |
| default | 43 | 0.25 | 0.4688 | 0.4688 | 0.2188 | false |
| default | 44 | 0.25 | 0.4167 | 0.4167 | 0.1667 | false |
| default | 100 | 0.25 | 0.3229 | 0.2917 | 0.0729 | false |
| default | 101 | 0.2604 | 0.5625 | 0.5312 | 0.3021 | false |
| default | 102 | 0.25 | 0.4583 | 0.4583 | 0.2083 | false |
| satdiv-GO (satdiv sig8/sc760, ridge1.0, glimpses6) | 42 | 0.25 | 0.5208 | 0.4896 | 0.2708 | false |
| satdiv-GO | 43 | 0.25 | 0.6458 | 0.625 | 0.3958 | false |
| satdiv-GO | 44 | 0.25 | 0.5104 | 0.4896 | 0.2604 | false |
| satdiv-GO | 100 | 0.25 | 0.4688 | 0.4479 | 0.2188 | false |
| satdiv-GO | 101 | 0.25 | 0.6771 | 0.5625 | **0.4271** | false |
| satdiv-GO | 102 | 0.25 | 0.6042 | 0.5312 | 0.3542 | false |

`spkport_cost = LEARNED_linscore_held - LEARNED_spkwta_held`, this runner's own diagnostic for exactly this gap.
**Correcting the retracted prereg's unsupported claim:** it called seed 42's `spkport_cost=0.2708` "the largest
such gap seen anywhere in this lane's history for this seed/bank." That is false even within the same battery --
satdiv-GO seed 101 reaches 0.4271 -- and this table now gives the actual largest gap on record for this
mechanism: **0.4271, seed 101 at the satdiv-GO operating point.**

`RATE_lin_ceiling_held` (the host-ridge diagnostic, unaffected by `--readout`) tracks `LEARNED_linscore_held`
closely at both operating points, confirming the front end and conjunction bank are untouched by the readout
choice -- the collapse is downstream of both.

## Why (mechanism-level, not just "the LIF port is broken") -- HYPOTHESIS, not yet tested by an ablation

The plain `--readout linear` spiking port is NOT broken at the satdiv-GO operating point: the already-banked
competitive-selection-at-satdiv-GO baseline gets `LEARNED_spkwta_held` mean 0.5729 <!--derived--> across the same 6 seeds, real
signal, well above chance. So a signed-linear-discriminant LIF class-population WTA port CAN carry signal at this
exact front end. The difference is what `_attention_gated_soft_class_read` does BEFORE the sign-split: for each
class `c` separately, it computes a per-class top-down-gained drive `bd = r * A_c**exponent`, then independently
runs THAT class's own version through `_apply_s2_norm`'s host satdiv ratio, with a per-trial semi-saturation
constant computed from `bd`'s OWN mean. Because this normalization is applied INDEPENDENTLY PER CLASS, each
class's channel is pulled toward a similar internal scale REGARDLESS of how strongly its own top-down template
favored the trial's actual evidence -- a per-class self-normalizing step can compress exactly the BETWEEN-class
contrast that a WTA argmax over `net_c` depends on, while leaving each class's OWN linear score (computed from
the raw, never-gated `r` by an entirely separate function, `_lin_score_pred`) untouched.

**This paragraph is a HYPOTHESIS, not an established diagnosis -- no ablation has been run to isolate it.** The
per-class-satdiv explanation was written by code inspection, not measurement. This finding's own data point to a
COMPETING explanation that has not been ruled out: `LEARNED_spkwta_TRAIN` (the identical WTA port's readout on
the TRAINING trials, not held-out) is pinned at EXACTLY 0.25 on every one of the 6 satdiv-GO seeds (verified
directly against the artifacts, `decode_means.LEARNED_spkwta_train`), which is the signature this runner's own
docstring already names for a different failure mode entirely -- `_attention_gated_soft_class_read`'s docstring
(`research/runners/_vision_lindiscrim_readout_derisk.py:1542`) describes `const` (the trial-INDEPENDENT class
bias) dominating `net` and "collapsing every trial's prediction to a single class" when a semi-saturation
constant saturates the ratio toward a near-constant fraction. A port that outputs the SAME class regardless of
trial is a constant-output/degenerate-port failure, not necessarily a between-class-contrast-compression failure
-- both produce chance-level `LEARNED_spkwta_held`, and this finding's own measurements do not distinguish them.

**The registered test of this hypothesis** is the follow-up mechanism's gain-only arm (`--readout
attention-gated-soft-fbgain --fb-strength 0.0`, research/findings/2026-09-23-vision-configural-binding-spiking-
feedback-divisive-gain-control-readout-PREREGISTERED.md): it removes the per-class host satdiv stage entirely
(`gated = bd` goes raw into the sign-split) while holding the top-down gain template fixed. If the per-class-
satdiv hypothesis above is correct, removing that stage alone should lift `LEARNED_spkwta_held` off chance even
with `fb_strength=0` (no spiking feedback loop yet). If the collapse persists at `fb_strength=0`, the competing
constant-output explanation (or some other stage) is more likely responsible, and the per-class-satdiv diagnosis
above should be considered refuted, not merely unconfirmed.

**Declared host shortcuts in this readout's own read path** (CLAUDE.md boundary, per the build-lane checklist;
not previously declared in this finding): the cross-class mean-centering `net = net - net.mean(axis=1,
keepdims=True)` (line 1494) is itself a POOLED host normalization computed across all classes at once, before
the LIF read -- the same SHAPE of operation (a host formula normalizing across the class population) as the
per-class satdiv step this finding's hypothesis blames, just a different stage. `read_gain`/`read_bias` (line
1495) are fixed host constants, calibrated once and reused unchanged across operating points; they can rescale
the whole population's drive but cannot restore contrast an earlier stage already discarded. Neither the mean-
centering nor the gain/bias affine step is done by neurons or synapses; both are host numpy arithmetic on `net`
before it reaches `lif_spike_read`.

## What retraction/correction this makes

`research/findings/2026-09-23-vision-configural-binding-attention-gated-soft-readout-PREREGISTERED.md` is
retracted (`docs/RETRACTED.md`, this commit) for its "never decisively run" premise and its non-blind band-
writing order. Its measured seed-42 data point (`LEARNED_spkwta_held=0.25`, `LEARNED_linscore_held=0.5208`,
`spkport_cost=0.2708`) is NOT retracted -- it is correct and is folded into the table above. `research/biology/
attention-gated-readout.md`'s registry entry is updated (this commit) to point here and to record the readout as
BANKED NO-GO, not "in flight."

## Next rung (not this finding's job to run, only to name)

**HYPOTHESIS, not yet tested by an ablation** (see "Why," above): that the residual is specifically the
per-class, host-computed, one-shot satdiv normalization stage -- not the top-down gain template, not the LIF
class-population port itself, and not the front end. A COMPETING explanation (the constant-output/degenerate-
port signature named above, `LEARNED_spkwta_TRAIN` pinned at exactly 0.25 on every satdiv-GO seed) has not
been ruled out. A follow-up pre-registration (`research/findings/2026-09-23-vision-configural-binding-spiking-
feedback-divisive-gain-control-readout-PREREGISTERED.md`) targets the per-class-satdiv stage with a genuinely
different method -- replacing the host-computed normalization with a SPIKING, pooled-across-classes, feedback
divisive gain-control loop realized inside the LIF read itself (Wilson & Cowan 1972; Heeger 1992), a companion
process the fixed host formula substituted with a constant, per this project's standing wall question ("what
does the real system run alongside this, that we replaced with a constant?") -- and its `--fb-strength 0.0`
gain-only arm is the REGISTERED TEST of this hypothesis, not an assumption that it is already confirmed.
