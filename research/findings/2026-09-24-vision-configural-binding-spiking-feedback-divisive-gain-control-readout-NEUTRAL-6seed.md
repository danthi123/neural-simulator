---
type: finding
status: verified
date: 2026-09-24
mechanism: score the 6-seed decisive run of the SPIKING FEEDBACK DIVISIVE GAIN-CONTROL readout
  (`--readout attention-gated-soft-fbgain`, `_attention_gated_soft_fbgain_class_read` +
  `lif_spike_read_fbgain` in `research/runners/_vision_lindiscrim_readout_derisk.py`) against the
  bands fixed in the same-lane pre-registration, and read the per-seed diagnostic data those bands
  do not themselves surface.
lane: vision (D-perception configural binding / position-invariant readout)
seeds: [42, 43, 44, 100, 101, 102]
verdict: NEUTRAL, exactly (band 4 of the pre-registration). `mean(d) = 0.0` and `SD(d) = 0.0`
  ACROSS ALL 6 SEEDS (not merely "not significant" -- an exact, zero-variance null): the full-mechanism
  arm (`--fb-strength 1.0`) and the gain-only control arm (`--fb-strength 0.0`) produce
  bit-identical `LEARNED_spkwta_held` on every one of the 6 seeds (0.25, exact chance for 4 classes,
  on all 12 runs). `capability_go` is 0/6 for both arms, unchanged from the gain-only arm's own 0/6
  baseline, per the pre-registration's own band-4 definition. `scramble_null_pass` holds 1.0 on all
  12 runs (the precondition for reading the bands at all is satisfied). This is a NO-GO on THIS
  method (the spiking feedback loop does not rescue the collapse), not a capability abandonment --
  see "What this seed data actually rules out" below for why the fix must be sought further upstream
  than either arm tested here.
runner: research/runners/_vision_lindiscrim_readout_derisk.py
artifacts:
  - research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s42.json
  - research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s43.json
  - research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s44.json
  - research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s100.json
  - research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s101.json
  - research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s102.json
  - research/findings/raw/lanes/perception/conjbind_fbgain_full_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s42.json
    (committed by the pre-registration's own commit; the other 11 arms above/below are new this commit,
    pulled from the pool at the pinned revision the pre-registration staged)
  - research/findings/raw/lanes/perception/conjbind_fbgain_full_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s43.json
  - research/findings/raw/lanes/perception/conjbind_fbgain_full_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s44.json
  - research/findings/raw/lanes/perception/conjbind_fbgain_full_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s100.json
  - research/findings/raw/lanes/perception/conjbind_fbgain_full_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s101.json
  - research/findings/raw/lanes/perception/conjbind_fbgain_full_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s102.json
  - all 12 `.prov.json` sidecars, pulled from the pool in the same commit, each recording
    `git_sha: 464d970e2d3e707ee01e04aeb750a3fb6762ddca`, `git_dirty: false`,
    `source_kind: git_archive`, `source_manifest_verified_at_start/exit: true` -- the pinned,
    isolated pool revision the pre-registration named, verified here by direct read of every
    sidecar, not assumed.
external: NO-EXTERNAL-NEEDED for this finding itself -- it scores an already-externally-grounded
  pre-registration (Wilson & Cowan 1972; Heeger 1992, both on file, lane-tagged, within this gate's
  window: `research/queue/.external_searches.jsonl`, 2026-09-24T01:30:30Z entry). No new mechanism
  lever is proposed in THIS commit; the follow-up next-method note below is a diagnosis + direction,
  not a new pre-registration (that is left as this lane's explicit next action, not built or run
  here, to avoid staging a design that has not itself been vetted).
builds_on:
  - research/findings/2026-09-23-vision-configural-binding-spiking-feedback-divisive-gain-control-readout-PREREGISTERED.md
    (the pre-registration this finding scores; its own seed-42 sanity run is confirmed byte-identical
    here, ignoring `elapsed_seconds`, to the pool's independent re-run of the same seed at the pinned
    revision)
  - research/findings/2026-09-23-vision-attention-gated-soft-readout-spiking-port-collapse-NOGO-banked.md
    (the banked NO-GO this mechanism targeted; see below for what this run does and does not resolve
    about its two named competing explanations)
  - research/biology/attention-gated-readout.md (registry entry; this finding's landing updates its
    `current_finding`/`current_status`)
---

# Spiking feedback divisive gain-control readout: NEUTRAL, exact null across all 6 seeds

## The scored result

Pulled all 6 seeds of both pre-registered arms from the pool (`bash tools/pool_sync.sh`; e.g.
`research/findings/raw/lanes/perception/conjbind_fbgain_full_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s43.json`
and its gain-only counterpart
`research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s43.json`
-- see the full 12-artifact list in the frontmatter `artifacts:` field). All 12 artifacts' `.prov.json`
sidecars confirm `git_sha: 464d970e2d3e707ee01e04aeb750a3fb6762ddca`, `git_dirty: false`,
`source_kind: git_archive` -- the exact pinned, isolated revision the pre-registration named, not a
later or dirty checkout.

| seed | full (`fb_strength=1.0`) `LEARNED_spkwta_held` | gain-only (`fb_strength=0.0`) `LEARNED_spkwta_held` | `d = full - gainonly` | `capability_go` full | `capability_go` gainonly | `scramble_null_pass` (both) | `LEARNED_linscore_held` (both, identical) |
|---|---|---|---|---|---|---|---|
| 42  | 0.25 | 0.25 | 0.0 | false | false | 1.0 | 0.5208 |
| 43  | 0.25 | 0.25 | 0.0 | false | false | 1.0 | 0.6458 |
| 44  | 0.25 | 0.25 | 0.0 | false | false | 1.0 | 0.5104 |
| 100 | 0.25 | 0.25 | 0.0 | false | false | 1.0 | 0.4688 |
| 101 | 0.25 | 0.25 | 0.0 | false | false | 1.0 | 0.6771 |
| 102 | 0.25 | 0.25 | 0.0 | false | false | 1.0 | 0.6042 |

`mean(d) = 0.0`, `SD(d) = 0.0`, `SE(d) = 0.0`, `t` is degenerate (0/0 in the exact-zero-variance
case, reported honestly as `t = 0.0` rather than an inflated or undefined value -- the ONLY
principled reading when every per-seed difference is identically zero: there is no evidence of
*any* effect, in either direction, at any magnitude, not merely "not significant at this n").

**Precondition check (must hold on all 6 seeds of both arms or the read is UNDEFINED):**
`scramble_null_pass == 1.0` on all 12 runs -- holds.

**Band read (bands fixed in the pre-registration, NEGATIVE evaluated first; `t_crit = 2.571` <!--derived-->
is the pre-registration's own fixed standard statistical constant, `scipy.stats.t.ppf(0.975, 5)` <!--derived-->, not a
measurement made here):**
1. NEGATIVE (`mean(d) < 0` and `|t| >= 2.571` <!--derived-->) -- does not fire (`mean(d) = 0`).
2. GO (`mean(d) > 0` and `|t| >= 2.571` <!--derived--> and `capability_go >= 5/6`) -- does not fire (`mean(d) = 0`).
3. PARTIAL (`mean(d) > 0` and (`|t| < 2.571` <!--derived--> or `capability_go` improves short of 5/6)) -- does not
   fire (`mean(d) = 0`, not `> 0`).
4. **NEUTRAL (`|t| < 2.571` <!--derived--> and `capability_go` unchanged from gain-only's own 0/6) -- FIRES.**
   `|t| = 0 < 2.571` <!--derived-->; `capability_go` is 0/6 for the full arm, identical to 0/6 for the gain-only
   arm and to the gain-only arm's own standing baseline from the banked NO-GO finding.
5. RESIDUAL -- not reached (band 4 fires cleanly, no leftover case).

**Verdict: NEUTRAL.** The spiking feedback divisive gain-control loop makes no measurable difference,
in any direction, to this readout's collapse.

## What this seed data actually rules out (beyond the banded verdict)

The pre-registration's own seed-42 sanity run flagged a competing explanation for the banked
collapse -- a constant-output / degenerate-port signature -- and said the remaining 5 seeds would
show whether it persisted. It persists, and more strongly than "persists": on EVERY ONE of the 12
runs scored here, `LEARNED_spkwta_held`, `LEARNED_spkwta_train`, `RANDOM_spkwta_held`, and
`scramble_learned_held` are ALL exactly 0.25 -- the trained population's held-out accuracy, its own
TRAINING accuracy, an UNTRAINED random population's accuracy, and a label-scrambled control's
accuracy are bit-identical to each other and to chance, on every seed, in both arms. A readout that
were merely noisy or weakly discriminating would not produce this pattern; a readout emitting a
CONSTANT prediction (the same class on every trial, regardless of input) does, exactly, on 4
balanced classes.

This is informative about *where* the fix is not, not just where it might be. The gain-only arm
(`--fb-strength 0.0`) already removes the host `_apply_s2_norm` per-class satdiv stage entirely --
`gated = bd` (the raw, gain-multiplied drive) goes straight into the excitatory/inhibitory sign-split,
with NO downstream normalization of any kind. It collapses identically to the full mechanism, which
adds the spiking feedback divisive gain-control loop back on top. Reading
`_attention_gated_soft_fbgain_class_read` (`research/runners/_vision_lindiscrim_readout_derisk.py:1690-1724`)
alongside `_spiking_class_read` (the plain `--readout linear` mode, which does NOT collapse at this
front end -- see the banked finding's own baseline, mean 0.5729 <!--derived--> across 6 seeds, quoted from
research/findings/2026-09-23-vision-attention-gated-soft-readout-spiking-port-collapse-NOGO-banked.md, not
re-derived from an artifact cited in this finding): the one step present
in the gain-only arm that is absent from plain linear is the top-down per-unit gain multiply itself,
`gated = r * A_c**attn_gain_exponent`, applied BEFORE the class's already-fitted discriminant
(`w = V/sd`, fit on the UN-gained `r`) is applied via the dot product. Neither this finding's own
scored run nor the banked finding instruments `net`'s pre-spike value directly, so the following is a
CODE-READING-DERIVED HYPOTHESIS, not a measurement made here: rescaling each unit of `r` by a
per-class template the linear discriminant was never calibrated against is a plausible mechanism for
driving `net` (the pre-spike class score) to be dominated by the trial-independent `const[c]` term
rather than the trial's own signal, which would produce exactly the constant-output signature
observed -- and, if correct, explains why NEITHER of this lane's two competing explanations (a
per-class-satdiv artifact, or a downstream normalization/feedback gap) is the whole story: the
collapse point precedes both. This has NOT been isolated by an ablation and is reported as a
hypothesis for the next lever, not a conclusion this run establishes.

## What this closes and what it does not

**Closes:** `--readout attention-gated-soft-fbgain` as a NO-GO/NEUTRAL method for this readout's
collapse, on the SAME footing as the plain `attention-gated-soft` mode it was built to fix (both are
now measured, at decisive scale, to leave the collapse exactly where the gain-only ablation alone
leaves it). Per this project's standing law, this is a verdict on the METHOD (a downstream,
post-gain normalization/feedback stage), not a license to abandon the capability (attention-gated,
per-class read-time modulation of a shared conjunction bank).

**Does not close:** the capability itself, nor the diagnosis of *why* the top-down gain multiply
collapses the readout (a hypothesis above, not yet tested by an ablation that holds the gain template
fixed while removing only the per-unit rescale-before-dot-product step -- e.g. applying the top-down
template as a per-class SCALAR response gain on the already-computed linear score, in the spirit of
Reynolds & Heeger's (2009) own distinction between contrast-gain (input-reweighting, what every arm
in this lane has tried so far) and response-gain (output-rescaling) forms of attentional modulation,
rather than another downstream normalization retune). Per THE LAW (a wall is a verdict on a method,
never the capability), this lane's next registered lever should target that upstream gain-calibration
step specifically, and should be pre-registered and selftested in its own commit before any decisive
run, per this project's standing build discipline -- not done in this commit, to avoid staging an
under-vetted design under time pressure.

## Registry update

`research/biology/attention-gated-readout.md`'s `current_finding`/`current_status` are updated
alongside this commit to point here and record the NEUTRAL/exact-null result plus the sharpened
diagnosis (the collapse point precedes the per-class satdiv step AND the feedback-loop stage; the
top-down per-unit gain multiply itself is now the leading suspect, code-reading-derived, not yet
ablated).
