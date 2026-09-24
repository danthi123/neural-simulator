---
type: preregistration
status: preregistered
date: 2026-09-23
mechanism: SPIKING FEEDBACK DIVISIVE GAIN-CONTROL readout (`--readout attention-gated-soft-fbgain
  --attn-gain-exponent 1.0 --fb-strength 1.0`, `_attention_gated_soft_fbgain_class_read` +
  `lif_spike_read_fbgain` in `research/runners/_vision_lindiscrim_readout_derisk.py` /
  `_vision_hmax_spiking_derisk.py`), stacked on the lane-best `--conj-select competitive` conjunction bank,
  AT the satdiv-GO front-end operating point (`--s2-norm satdiv --s2-satdiv-sigma 8 --s2-satdiv-scale 760
  --s2-satdiv-n 2.0 --ridge 1.0 --n-glimpses 6`). Replaces attention-gated-soft's HOST-computed, per-class,
  pre-spike satdiv normalization with a SPIKING, pooled-across-classes, post-spike feedback divisive
  gain-control loop realized inside the LIF class-population read itself. Grounded in Wilson & Cowan
  (1972), Biophys J 12:1-24, and Heeger (1992), Visual Neuroscience 9:181-197 (both recorded this session,
  research/queue/.external_searches.jsonl, lane "vision (D-perception configural binding / position-
  invariant readout)").
lane: vision (D-perception configural binding / position-invariant readout)
seeds: [42, 43, 44, 100, 101, 102]
verdict: PREREGISTERED -- no decisive result in this finding. Seed 42 to be run locally below (sanity/
  first data point only, NOT a verdict, IF `tools/mem_ok.sh` passes at commit time), then seeds 43/44/100/
  101/102 staged (pool if reachable from this session, otherwise their exact commands are recorded here so
  they can be dispatched by whichever session next has pool access). A separate finding reads the merged
  6-seed result against the bands fixed here.
artifacts:
  - research/findings/raw/lanes/perception/vlin_fbgain_frac0_smoke.json (byte-identical-off proof, tiny
    scale, both levers disabled)
  - research/findings/raw/lanes/perception/vlin_fbgain_gainonly_smoke.json (tiny scale, gain ON / feedback
    OFF -- the pre-registered OFF-arm control for isolating the feedback loop's own causal contribution)
  - research/findings/raw/lanes/perception/vlin_fbgain_smoke.json (tiny scale, full mechanism, non-crash
    smoke)
  - research/findings/raw/lanes/perception/vlin_linear_frac0_comparison_smoke.json (a fresh `--readout
    linear` run at the identical tiny scale, committed alongside the smokes, used to VERIFY -- not just
    assert -- that the frac0 smoke is byte-identical to `--readout linear` except `elapsed_seconds`)
  - all four `.prov.json` sidecars, committed in the prior (code) commit
  - the decisive-scale seed-42 artifact, added in a SEPARATE commit AFTER this one (per
    `gates/prereg_before_run`)
external: Heeger (1992), "Normalization of cell responses in cat striate cortex," Visual Neuroscience
  9(2):181-197 -- divisive normalization REALIZED at the circuit level by RECURRENT/SHUNTING inhibition, a
  pooled inhibitory signal computed DYNAMICALLY from the circuit's own ongoing activity, distinct from the
  already-on-file Reynolds & Heeger (2009) algebraic gain-then-normalize FORM this lane already used (and
  which is what collapsed -- see `builds_on`). Wilson & Cowan (1972), "Excitatory and inhibitory
  interactions in localized populations of model neurons," Biophysical Journal 12(1):1-24 -- the canonical
  justification for a population's own recent activity, low-pass filtered through a time constant, acting
  as a dynamical state variable that shapes its own ongoing input drive (the `r_fb` construction this
  mechanism implements). Both recorded via `tools/record_external_search.sh`, lane-tagged, this session.
builds_on:
  - research/findings/2026-09-23-vision-attention-gated-soft-readout-spiking-port-collapse-NOGO-banked.md
    (BANKED NO-GO, 12 of 12 seed-runs, two front-end operating points -- the collapse this mechanism
    targets, and the diagnosis that it is specific to attention-gated-soft's per-class host satdiv step,
    not the front end or the LIF port in general)
  - research/biology/attention-gated-readout.md (mechanism registry entry; this finding's landing updates
    its `current_finding`/`current_status`)
review_corrections_applied:
  - "the pre-registration must be committed BEFORE any run artifact it governs" (`gates/prereg_before_run`,
    added to `main` this session after the prior round's violation): this finding's own commit stages NO
    `research/findings/raw/**` artifact except the ALREADY-committed (prior commit) tiny smokes, which no
    gate below reads for its verdict -- see the escape declaration at the end of this section.
  - "a wall's own diagnosis must be a genuinely DIFFERENT method, not another retune of the same host
    formula" (this session's harness instruction): this mechanism does not touch `attn_satdiv_sigma_frac`/
    `attn_satdiv_scale_mult`/`attn_satdiv_n` (the retracted mechanism's own tunables) at all -- it REMOVES
    the host satdiv step from this readout's own code path entirely and replaces it with machinery that
    did not exist in this file before this session (`lif_spike_read_fbgain`).
  - "exhaustive bands including a residual band" (the prior round's review correction): see the gate below
    -- five bands, collectively exhaustive over `(mean(d), |t|, capability_go)`, with an explicit RESIDUAL
    band for the leftover case.

prereg-same-commit: the tiny smokes cited above were committed in the PRIOR commit (the code commit), not
  this one -- this finding's own commit adds no `research/findings/raw/**` artifact, so `gates/
  prereg_before_run`'s escape line is declared defensively (a markdown file with "PREREG" in its name is
  itself sometimes miscounted) even though the literal trigger condition (a raw artifact staged in the
  SAME commit as this prereg) does not apply here.
---

# Spiking feedback divisive gain-control readout: pre-registration (before any decisive-scale run)

**This is a pre-registration only.** It fixes the mechanism, the exact command, the GO/NEGATIVE/PARTIAL/
NEUTRAL/RESIDUAL bands, and the anti-cheats BEFORE any decisive-scale seed is read for verdict purposes.
The mechanism and its selftest were committed in the PRIOR commit (code, no decisive-scale run); this
commit adds no run artifact.

## Why this lever, and why a genuinely different method (not another retune)

`bash tools/before_you_build.sh "spiking readout LIF port collapses to chance under a per-class host
normalization while the raw linear score separates"` (this session) surfaced this lane's own diagnosis
(the just-banked NO-GO finding) and nothing that already answers "replace the normalization stage with a
spiking feedback loop" -- `git log --all --oneline --grep` for `fbgain`/`feedback.*gain`/`shunt.*inhibit`
in this lane returns nothing before this session's own commits.

**The wall question, asked first, per this project's standing rule.** "What does the real system run
alongside this readout that we replaced with a constant?" The banked NO-GO's diagnosis: attention-gated-
soft's `read_gain`/`read_bias` are FIXED HOST CONSTANTS, calibrated once and reused unchanged across front-
end operating points, and its per-class divisive normalization (`_apply_s2_norm` satdiv) is a PRE-spike,
ONE-SHOT host formula computed independently per class -- nothing in this file lets the SPIKING stage's own
realized activity feed back and correct that. A real cortical circuit's divisive normalization is not
computed once and frozen; Heeger (1992) shows it is REALIZED by recurrent/shunting inhibition, a pooled
signal computed dynamically from the circuit's own ongoing activity (Wilson & Cowan 1972's population-
activity state variable). This mechanism builds exactly that missing companion process, INSIDE the LIF
read, instead of retuning the same host satdiv formula's own parameters (which the banked finding's own
diagnosis rules out as the fix: the per-class independence of that step, not its sigma/scale constants, is
what compresses between-class contrast).

## The mechanism (already built + selftested, this finding's prior commit; unchanged by this commit)

Identical top-down template and continuous multiplicative gain step to `_attention_gated_soft_class_read`
(`A_c = |w_c| / mean(|w_c|)`; `bd = r * A_c ** attn_gain_exponent`). The difference is entirely downstream:
this mode does NOT run `bd` through `_apply_s2_norm`'s host satdiv ratio at all -- `gated = bd` (or `= r`
when `attn_gain_exponent <= 0`) goes RAW into the excitatory/inhibitory sign-split, and the divisive gain
control is instead realized by `lif_spike_read_fbgain`: every simulated millisecond, AFTER real spikes are
drawn, a leaky low-pass trace `r_fb` of the class-population's OWN realized output-spike fraction (pooled
across the WHOLE row -- every class, for a given trial) is updated, and the NEXT step's input current is
divided by `(1 + fb_strength * r_fb)` -- a population-level, activity-dependent, dynamically self-
correcting gain, computed by the spiking stage itself rather than a host formula computed once before any
spike is drawn. Full derivation in `lif_spike_read_fbgain`'s own docstring (`research/runners/
_vision_hmax_spiking_derisk.py`) and `_attention_gated_soft_fbgain_class_read`'s (`research/runners/
_vision_lindiscrim_readout_derisk.py:1615-1690`, approx.).

**Host shortcut declared** (CLAUDE.md boundary, per the build-lane checklist): the top-down template `A_c`
and the gain multiply `bd = r * A_c**exponent` remain host numpy formulas, unchanged from every other arm
in this file. This finding does not close that shortcut -- only the DOWNSTREAM normalization/gain-control
shortcut (previously a host satdiv ratio, now a spiking feedback loop). `read_gain`/`read_bias` remain the
same fixed host constants, applied upstream of `lif_spike_read_fbgain`'s own `gain=` argument; this
mechanism is additive to them, not a replacement.

## Newly-added automated selftest (committed in the prior commit; 5 tests, all pass)

`tests/test_vision_fbgain_readout_selftest.py`:
1. `test_fbgain_disabled_reproduces_lif_spike_read_exactly` -- `fb_strength<=0` (0.0, -1.0, -5.0 checked)
   delegates to `lif_spike_read` verbatim (identical counts AND first-spike times, identical RNG draws).
2. `test_feedback_trace_is_driven_by_real_spikes_not_a_constant` -- a controlled strong-drive-row vs weak-
   drive-row contrast: turning feedback on must COMPRESS the strong/weak spike-count ratio (divisive, self-
   normalizing), never widen it, and must change the counts at all (rules out a disconnected `r_fb`).
3. `test_readout_disabled_reproduces_linear_exactly` -- both levers off reproduces `_spiking_class_read`
   (`--readout linear`) bit-for-bit, including a negative-exponent check.
4. `test_fbgain_actually_changes_output_vs_gain_only` -- gain ON + feedback ON must differ from gain ON +
   feedback OFF on a synthetic non-degenerate problem (isolates the feedback loop's own causal
   contribution, holding the gain template fixed).
5. `test_class_read_dispatcher_routes_fbgain` -- pins the `_class_read` dispatch table.

Verified locally: `.venv/bin/python -m pytest tests/test_vision_fbgain_readout_selftest.py -v` -> 5 passed.

**Runner-level byte-identical-off, verified by DIRECT DIFF, not construction.** `vlin_fbgain_frac0_smoke.
json` (both levers disabled, tiny scale) is identical to `vlin_linear_frac0_comparison_smoke.json` (a
fresh `--readout linear` run at the SAME tiny-scale flags, generated and committed in the same session,
not reused/assumed from an older run) except `elapsed_seconds` (direct JSON diff, both artifacts committed
in the prior commit; closes the prior round's "commit the comparison artifact or drop the claim" item with
an actual comparison artifact, not a repeated assertion).

**`fb_strength`'s default (1.0) is chosen on PRINCIPLE, not by a sweep.** A coarse 0.5/1.0/2.0/5.0/10.0
sweep was run at the tiny smoke scale (`n_ex=2`, effectively no data) during development and showed no
reliable monotonic signal -- that scale is too data-starved to calibrate anything on, and picking whichever
value looked best there would be exactly the "argmax over the evaluation metric" the build-lane checklist
forbids. `fb_strength=1.0` is instead the dimensionless "unit-strength" default: at full-population
saturation (`r_fb=1`, every class spiking every step), the drive is exactly halved -- a natural, scale-free
choice requiring no tuning against any evaluation data, and structurally analogous to `attn_gain_exponent
=1.0` already being this file's own convention for "the untuned, literal form of the cited paper's
equation" rather than a hand-fit constant.

## Pre-registered GO gate (fixed BEFORE the decisive 6-seed run)

**Primary (mechanism-attribution) statistic -- PAIRED per-seed differences**, isolating the feedback loop's
own causal contribution with the bank AND the gain template held fixed (the SAME paired-difference
discipline the retracted prereg introduced and that review round confirmed was correctly designed, applied
here against a DIFFERENT, non-retracted control -- the gain-only arm, not the plain-linear arm, since the
question this mechanism answers is "does the feedback loop fix the gain template's own collapse," not "is
gain-plus-something better than no-gain-at-all"):

```
d_s = LEARNED_spkwta_held(attention-gated-soft-fbgain, --fb-strength 1.0, seed s)
    - LEARNED_spkwta_held(attention-gated-soft-fbgain, --fb-strength 0.0 [gain-only], seed s)
```

Both arms run at the IDENTICAL satdiv-GO front end, IDENTICAL competitive bank, IDENTICAL `--attn-gain-
exponent 1.0` -- only `--fb-strength` differs, so the two arms share the same trained discriminant, C2
code, bank membership, and top-down gain template for a given seed. No width-matched null is needed for
the same reason the retracted prereg gave (not retracted): this lever changes neither the bank, its width,
nor the population size, only whether one downstream normalization step is host-computed or spiking-
feedback-computed.

Report `mean(d)`, `SD(d)`, `SE(d) = SD(d)/sqrt(6)`, and the paired t-statistic `t = mean(d)/SE(d)` with
`df=5`. `t_crit = 2.571` <!--derived--> (`scipy.stats.t.ppf(0.975, 5)`, a standard statistical constant, not
a measurement) is fixed here, in advance, as the significance threshold.

**Precondition (must hold on every seed or the read is UNDEFINED, not banded):** `scramble_null_pass ==
1.0` on all 6 seeds of BOTH arms (the anti-cheat that has held 6/6 across every arm in this lane so far).

**Secondary (capability) readout, reported alongside, not substituted for the primary:** per-seed
`capability_go` (the lane's standing `>= 5/6` threshold definition) for the fbgain arm, and
`LEARNED_linscore_held` / `RATE_lin_ceiling_held` per seed (unaffected by `--readout`, confirming the front
end/bank stay untouched, exactly as every prior lever in this lane reports).

**Bands (NEGATIVE evaluated first, per this lane's standing precedence-fix; collectively EXHAUSTIVE over
`(mean(d), |t|, capability_go)` -- a RESIDUAL band is added so no outcome falls through, the specific gap
the prior review round flagged):**

1. **NEGATIVE** if `mean(d) < 0` AND `|t| >= 2.571` <!--derived--> (the feedback loop significantly HURTS
   relative to gain-only) -- i.e. the spiking feedback loop is itself a worse method than doing nothing
   downstream of the gain template.
2. **GO (task-relevant SURPASS)** -- only if NEGATIVE does not fire -- if `mean(d) > 0` AND `|t| >= 2.571` <!--derived-->
   AND per-seed `capability_go >= 5/6` for the fbgain arm (i.e. this mechanism both
   significantly improves on the gain-only collapse AND clears this lane's own capability floor -- the
   collapse is FIXED, not merely reduced).
3. **PARTIAL** -- `mean(d) > 0` and EITHER `|t| < 2.571` <!--derived--> (not distinguishable from noise at
   n=6) OR `capability_go` improves without reaching 5/6 (the collapse is measurably reduced but not
   closed). Progress, not closure.
4. **NEUTRAL** -- `|t| < 2.571` <!--derived--> AND `capability_go` is UNCHANGED from the gain-only arm's own
   0/6 (per the banked NO-GO finding, both `--attn-gain-exponent 1.0 --fb-strength 0` and the retracted
   mechanism's own satdiv-normalized arm read `capability_go=false` on every seed measured so far) --
   "cannot reject no-effect at n=6," reported honestly, not rounded up or down.
5. **RESIDUAL** -- any outcome not covered by bands 1-4 (e.g. `mean(d) > 0` with `|t| >= 2.571` <!--derived--> but
   `capability_go` BELOW 5/6 and also below the gain-only baseline's own rate; or `mean(d) < 0` with
   `|t| < 2.571` <!--derived-->). Reported as exactly what it is (the specific numbers), not forced into the nearest named
   band -- the gap the prior review round's "some outcomes fit no band" finding named directly.

**Read honestly, not by the letter of the band** (carried forward from every prior lever in this lane): a
PARTIAL that does not clear `capability_go >= 5/6` is not a task GO regardless of how large `mean(d)` looks.

## The decisive command

```bash
# gain-only control arm (fb_strength=0, the comparison baseline for the paired-t gate)
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --s2-norm satdiv --s2-satdiv-sigma 8 --s2-satdiv-scale 760 --s2-satdiv-n 2.0 \
    --ridge 1.0 --n-glimpses 6 --n-s2 96 \
    --conj-bind fixed --conj-select competitive --conj-select-overcomplete 4 --conj-select-kwta-frac 0.1 \
    --conj-n 1152 --conj-offset-max 4 --readout attention-gated-soft-fbgain --attn-gain-exponent 1.0 \
    --fb-strength 0.0 \
    --heldout-position --scramble-null --seeds <SEED[S]> \
    --out research/findings/raw/lanes/perception/conjbind_fbgain_gainonly_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s<SEED>.json

# full mechanism arm (fb_strength=1.0, the treatment)
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --s2-norm satdiv --s2-satdiv-sigma 8 --s2-satdiv-scale 760 --s2-satdiv-n 2.0 \
    --ridge 1.0 --n-glimpses 6 --n-s2 96 \
    --conj-bind fixed --conj-select competitive --conj-select-overcomplete 4 --conj-select-kwta-frac 0.1 \
    --conj-n 1152 --conj-offset-max 4 --readout attention-gated-soft-fbgain --attn-gain-exponent 1.0 \
    --fb-strength 1.0 \
    --heldout-position --scramble-null --seeds <SEED[S]> \
    --out research/findings/raw/lanes/perception/conjbind_fbgain_full_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s<SEED>.json
```

No `cfg.seed` trap applies (standalone numpy runner, no `CoreSimConfig`/`SimulationBridge`); `--seeds`
alone controls determinism (unchanged from the retracted prereg's own verified derivation of this fact,
lines 175-180, not itself part of what was retracted).

## Seed 42, run locally below (sanity-only -- NOT a verdict; 5 of 6 seeds staged, not awaited)

See the run log appended after this pre-registration commit lands (a separate commit, per
`gates/prereg_before_run`). Both arms (gain-only and full) are run at seed 42 for the paired comparison to
have its first data point; per this lane's own standing 6-seed rule, no generalization or verdict claim is
made on n=1.
