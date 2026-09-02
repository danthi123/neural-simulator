---
type: finding
status: negative
date: 2026-09-02
mechanism: crossedge-surprise-metacog-error-gated-port-ablation
board: 108 / one-brain integration program (C2 rung)
seeds: [42, 43, 44, 100, 101, 102]
artifact: research/findings/raw/_crossedge_surprise_metacog_errorgated_numpy6seed.json
---

# C2 (D2-surprise -> E1-metacog confidence) error-gated port: a SECOND, controlled negative — the port REGRESSES robustness, it does not fix it

**2026-09-02, numpy, 6 seeds (42/43/44/100/101/102), same runner
(`research/runners/_crossedge_surprise_metacog_derisk.py`), same backend, same seeds — an `--ablation gated|plain`
A/B kept live in the committed runner.**

## Background

The 2026-09-02 harvest finding
(`research/findings/2026-09-02-integration-program-6seed-harvest-singlepool-GO-C1-GO-C2-NOGO.md`) recorded C2
(D2-surprise -> E1-metacog confidence) at 3/6 NO-GO on cupy
(`research/findings/raw/_crossedge_surprise_metacog_6seed.json`), and named the next rung: port the sibling C1
edge's (D2-surprise -> E2-world-model, 6/6 GO,
`research/findings/raw/_crossedge_surprise_worldmodel_6seed.json`) prediction-error-GATED three-factor update
into C2's plain rate-Hebbian training, on the hypothesis that gating would make growth more robust across seeds.

## What was ported

C1's mechanism (`research/runners/_crossedge_surprise_worldmodel_derisk.py`, `_gated_update_step`/
`_calibrate_gate`): each training turn first READS the D2-surprise firing rate for that trial (learning OFF);
the Hebbian window opens only if the rate clears a threshold calibrated from THIS seed's own expected-vs-violated
firing gap (`GATE_FRAC=0.35` into the gap); the update runs only when gated open.

This was ported into C2 verbatim in shape: `SurpriseMetacogPool._read_surprise_hz` + `_calibrate_conf_gate`
(mirroring C1's `_read_surprise`/`_calibrate_gate`) and `train()`'s per-episode gate (mirroring
`_gated_update_step`), gated on `GATE_FRAC=0.35` into C2's own CONFIRM-vs-CONTRADICT surprise-firing gap. Kept
live in the SAME runner as an `--ablation gated|plain` flag (default `plain`, the original mechanism) rather
than a throwaway duplicate module, so the comparison below is reproducible from one committed file.

## The controlled result

Full 6-seed numpy runs of BOTH ablations, from the same runner, same seeds:

- `plain` (the ORIGINAL, unconditional rate-Hebbian mechanism): `GO: true`, `n_go: 6/6` —
  `research/findings/raw/_crossedge_surprise_metacog_plain_numpy6seed_control.json`.
- `gated` (the ported third-factor gate): `GO: false`, `n_go: 3/6` (seeds 42/43/44 GO; 100/101/102 NO-GO) —
  `research/findings/raw/_crossedge_surprise_metacog_errorgated_numpy6seed.json`.

Both artifacts show `emergence.PASS: true` on all 6 seeds in both ablations (the edge grows from `W0=0.05` to
`~0.61-0.68`, `runs[*].emergence.grown.surprise_to_metacog_conf`) and `byte_off.PASS: true` on all 6 in both. The
entire difference is in `interaction.PASS` — the same failure MODE as the original cupy NO-GO
(`_crossedge_surprise_metacog_6seed.json`): a wrong-sign/oversized `interaction.per_condition.high.delta_lesion`
(the lesioned pool's own CONTRADICT-vs-CONFIRM read gap, which should sit near zero once the cross-edge is
zeroed, instead exceeds `lesion_ratio=0.34 * |delta_intact|`).

Per-seed `delta_lesion` (`runs[*].interaction.per_condition.high.delta_lesion`), plain vs gated, identical seeds
— table values are the cited artifacts' own numbers rounded to 4dp for readability:

<!--derived-->
| seed | plain `delta_lesion` | gated `delta_lesion` | plain GO | gated GO |
|---|---|---|---|---|
| 42  | -0.0051 | -0.0245 | true | true |
| 43  | -0.0018 | +0.0153 | true | true |
| 44  | +0.0034 | +0.0123 | true | true |
| 100 | -0.0131 | +0.0431 | true | false |
| 101 | -0.0332 | -0.0638 | true | false |
| 102 | -0.0016 | +0.0950 | true | false |

Mean `|delta_lesion|` across the 6 seeds: plain 0.0097, gated 0.0423 (mean of the |delta_lesion| column above,
each value read from the two cited JSONs' `runs[*].interaction.per_condition.high.delta_lesion`) — roughly 4.3x
larger under the gated mechanism, and the confound GREW (not shrank) on 5 of 6 seeds. The gate did not reduce
the lesion-baseline confound; it enlarged it.

`gate_opens`/`gate_n_episodes` (`runs[*].gate_opens`, `runs[*].gate_n_episodes`,
`_crossedge_surprise_metacog_errorgated_numpy6seed.json`) reads `80/80` for EVERY one of the 6 seeds — the gate
never actually closed, on any seed, at any point in the 80-episode training run.

## Diagnosis: why the port doesn't transfer

C1's gate is self-limiting because the thing being trained (`state -> pred`) directly feeds the surprise units
that gate it (`pred_{pos,neg}` inhibits `surprise_{pos,neg}` — a closed loop; C1's own module docstring:
"surprise falls silent -> the gate self-closes"). C2's D2-surprise circuit (`cue -> patient_expected -> surprise`)
is FIXED/non-plastic and structurally independent of the cross-edge being trained (`surprise ->
metacog.workspace member[1]`) — growing the cross-edge has zero effect on `surprise`'s own firing. With no
feedback loop, the calibrated threshold (`gate_calib.threshold`, ~5 Hz, well below the CONTRADICT trial's own
~14-15 Hz firing, `gate_calib.violated_hz`, in `_crossedge_surprise_metacog_errorgated_numpy6seed.json`) is
cleared on every episode of every seed — the gate never actually discriminates. What it adds instead is two
extra un-learned read passes per episode (a full hard-reset + settle + read cycle before each teach step) that
perturb the pool's own homeostatic threshold state (`enable_homeostasis`, which is NOT reset by
`SurpriseMetacogPool._hard_reset()` — only membrane/conductance/firing/coactivity-trace arrays are), producing a
real but harmful divergence from the plain trajectory: pure added noise, no added selectivity.

The actual C2 defect — visible identically in the original cupy NO-GO and in this numpy control, under BOTH
ablations — is a LESION-BASELINE confound: the CONFIRM and CONTRADICT trials drive DIFFERENT `patient_asserted`
blocks (`cue_c` vs the seed's randomly-assigned `assert_cp`, from the shared `_assign_blocks` anti-cheat), and
those blocks' own (non-learned, random) base connectivity into the metacog workspace read differs by chance
per seed/block — independent of the cross-edge, and therefore un-fixable by changing HOW the cross-edge learns
(plain vs gated, or any other Hebbian variant). This orthogonality is why the "fix the learning rule" hypothesis
failed on a controlled test: the crux lives in the READ's isolation from a structural base-connectivity leak,
not in the plasticity rule.

## Verdict — a second, real negative on the METHOD, not the capability

Per NO-DEFER this is a verdict on a METHOD (error-gating C2's Hebbian rule), not the capability (D2-surprise
lowering metacog confidence) — emergence and the base "surprise lowers confidence, intact" effect are robust in
BOTH ablations, 6/6. The gated port is **not** queued for a 6-seed cupy verify: the numpy control shows it
regresses (not fixes) robustness relative to the already-committed plain mechanism, so spending cupy compute on
it would very likely reproduce (or worsen) the original 3/6 NO-GO. `--ablation plain` (the original mechanism)
remains the runner's default; `--ablation gated` is banked in the same file for reproducibility, not
recommended.

**NEXT MECHANISM (not deferred):** a normalized/competitive READ or WRITE that controls for the per-block
baseline leak, e.g. (a) a baseline-subtracted read — measure each seed's own no-edge (or pre-training)
CONTRADICT-vs-CONFIRM gap and subtract it from the intact read before scoring the drop, rather than assuming the
lesioned baseline is ~0; or (b) a competitive/divisive cross-edge write (an explicit inhibitory counterpart from
the CONFIRM-condition block, or a normalization stage at the read) that cancels the block-specific leak
structurally rather than statistically. Both target the READ's isolation from the confound, not the Hebbian
rule — the lesson banked here is that repeating the "adopt the sibling edge's learning-rule form" lever a second
time, without first checking whether the sibling's mechanism structurally applies (a closed feedback loop, here
absent), reproduces the failure rather than resolving it.
