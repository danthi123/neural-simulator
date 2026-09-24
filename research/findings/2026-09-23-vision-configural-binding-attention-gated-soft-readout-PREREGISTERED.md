---
type: preregistration
status: preregistered
date: 2026-09-23
mechanism: soft/graded attention-gain readout (`--readout attention-gated-soft --attn-gain-exponent 1.0`,
  `_attention_gated_soft_class_read` in `_vision_lindiscrim_readout_derisk.py`) stacked on the lane-best
  `--conj-select competitive` conjunction bank, AT the satdiv-GO front-end operating point (`--s2-norm satdiv
  --s2-satdiv-sigma 8 --s2-satdiv-scale 760 --s2-satdiv-n 2.0 --ridge 1.0 --n-glimpses 6`). Grounded in
  research/biology/attention-gated-readout.md's own named "next rung" and externally verified there by
  Reynolds & Heeger (2009), Neuron 61:168 (graded gain + divisive normalization, no hard elimination).
lane: vision (D-perception configural binding / position-invariant readout)
seeds: [42, 43, 44, 100, 101, 102]
verdict: PREREGISTERED -- no decisive result in this finding. Seed 42 run locally below (sanity/first data
  point only, NOT a verdict); seeds 43/44/100/101/102 staged on the pool. A separate finding reads the
  merged 6-seed result against the bands fixed here.
artifacts:
  - research/findings/raw/lanes/perception/vlin_attngatedsoft_frac0_smoke.json (byte-identical-off proof,
    tiny scale, `--attn-gain-exponent 0`)
  - research/findings/raw/lanes/perception/vlin_attngatedsoft_smoke.json (non-degenerate smoke, tiny scale,
    `--attn-gain-exponent 1.0`)
  - research/findings/raw/lanes/perception/conjbind_attngatedsoft_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s42.json
    (decisive-scale, seed 42 only, run locally below)
  - both smoke `.prov.json` sidecars, committed alongside
external: Reynolds & Heeger (2009), "The Normalization Model of Attention," Neuron 61:168, PMID 19186161,
  https://pubmed.ncbi.nlm.nih.gov/19186161/ -- already the basis for this mechanism's design (recorded again
  for this lane, this session, in research/queue/.external_searches.jsonl per the DR gate). No NEW external
  claim is made here; this finding executes an already-externally-grounded, already-built, never-yet-run
  mechanism, so no fresh literature search was owed beyond re-affirming the existing one.
builds_on:
  - research/findings/2026-09-09-vision-configural-binding-attention-gated-readout-NEXT-MECHANISM-PREREGISTERED.md
    (hard k-WTA attention-gated readout REGRESSED -- beat2/6-lb3/6 vs the competitive-selection baseline's
    beat4/6-lb6/6; named the graded/soft gain as the next rung, externally confirmed there)
  - research/findings/2026-09-17-vision-satdiv-divisive-norm-is-the-decisive-lever-for-position-invariant-readout-binding-not-load-bearing-GO.md
    (established the satdiv front-end as the current best operating point)
  - research/findings/2026-09-23-cpu-lane-harvest-perception-binding-negative-fixed-unconfirmed-competitive.md
    (competitive-selection AT the satdiv-GO operating point: BINDING-POSITIVE-UNCONFIRMED --
    wins decisively on the continuous `mean_d_width` metric, +0.2257 per that finding's own table <!--derived-->, but its
    per-seed `capability_go` regresses to 4/6 against the control's 5/6, seeds 42 and 101 specifically
    failing despite above-average `LEARNED_spkwta_held` -- this is the EXACT residual this finding's
    mechanism targets)
  - research/biology/attention-gated-readout.md (mechanism registry entry; this finding's landing updates
    its `current_finding`/`current_status`)
review_corrections_applied:
  - "the width-matched arm must be read with paired-difference statistics, not band-overlap" (this lane's
    2026-09-23 harvest re-review): the gate below reads the readout's causal contribution as PAIRED
    per-seed differences against the identical bank's plain-linear read (same seed, same trained
    discriminant, same C2 code -- only the readout changes), not as two independent-sample distributions.
    A width-matched control is not needed here in the first place, because this lever does not change
    population size or bank membership at all (see "Why no width-matched null is needed" below).
  - NEGATIVE evaluated before NEUTRAL/POSITIVE (the harvest's issue-4 precedence fix), applied identically
    below so the bands stay mutually exclusive.
---

# ⛔ RETRACTED (2026-09-23) — see [`docs/RETRACTED.md`](../../docs/RETRACTED.md)

**The "never decisively run" premise below is FALSE**, and the bands were written non-blind (same commit as the
seed-42 result they discuss, after it was observed). An untracked 6-seed decisive NO-GO of this exact readout
already existed (`conjbind_attngatedsoft_n1152_heldoutpos_scramblenull_6seed.json`, 2026-09-17) and was missed by
this document's own filesystem/git searches because it had never been committed. **Superseded by
[`research/findings/2026-09-23-vision-attention-gated-soft-readout-spiking-port-collapse-NOGO-banked.md`](2026-09-23-vision-attention-gated-soft-readout-spiking-port-collapse-NOGO-banked.md)**,
which banks BOTH 6-seed runs (12 of 12 seeds: `LEARNED_spkwta_held` at exact chance, `LEARNED_linscore_held`
real) as a clean NO-GO of this readout METHOD, and which names the next mechanism
(`research/findings/2026-09-23-vision-configural-binding-spiking-feedback-divisive-gain-control-readout-PREREGISTERED.md`).
**The measured seed-42 data point itself (`LEARNED_spkwta_held=0.25`, `LEARNED_linscore_held=0.5208`,
`spkport_cost=0.2708`) is NOT retracted** — it is correct and survives, folded into the superseding finding's
table. Everything below this banner is kept for the audit trail; do not cite its "PREREGISTERED"/"no verdict"
framing or its "largest gap" claim (both superseded).

---

# Attention-gated-soft readout: pre-registration (before any decisive-scale run)

**This is a pre-registration only.** It fixes the mechanism, the exact command, the GO/NEGATIVE/NEUTRAL
bands, and the anti-cheats BEFORE any decisive-scale seed is read for verdict purposes. Per the harness
task's own step ordering: seed 42 is run locally below (sanity-only, reported but NOT a verdict), then seeds
43/44/100/101/102 are staged on the pool and this finding does NOT wait for them.

## Why this lever, not a new one (before_you_build.sh run first)

`bash tools/before_you_build.sh` (this session, query: "configural binding readout: graded/divisive-norm
attention gain vs hard k-WTA at satdiv-GO operating point") surfaced exactly the two findings this
pre-registration builds on and nothing that already answers this question. `git log --all --oneline --grep`
for `attention-gated-soft` / `conjbind_attngatedsoft` and a filesystem search for any artifact matching that
name both come back with ONLY the 2026-09-16 BUILD commits (`b7108ca04`/`deb54b150`) -- the mechanism exists
on `main`, fully de-risked (byte-identical-off proven in the module docstring, a tiny non-degenerate smoke
run), but **no decisive-scale run and no finding have ever been committed for it.** This is not a re-derived
lever: it is the one named, built, and externally-grounded next rung this lane has been carrying unrun.

## The mechanism (already built, `_attention_gated_soft_class_read`, unchanged by this finding)

Identical top-down template to the hard-gated mode (`A_c = |w_c| / mean(|w_c|)`, read off the already-fitted
discriminant, zero new learning). The combination rule is the ONLY thing that differs from the hard mode:
continuous multiplicative gain `bd = r * A_c**attn_gain_exponent` (no k-WTA, no zeroing), then Reynolds &
Heeger's "gain then normalize" step via this file's own `_apply_s2_norm` satdiv primitive (reused, not
re-derived), with a per-trial data-driven semi-saturation constant. `--attn-gain-exponent <= 0`
short-circuits to `gated = r` before either step runs, reproducing `--readout linear` bit-for-bit. Full
derivation and the byte-identical-off argument are in the function's own docstring
(`research/runners/_vision_lindiscrim_readout_derisk.py:1505-1563`) and in
`research/biology/attention-gated-readout.md`.

## Why no width-matched null is needed here (unlike the 4-arm bank gate)

The harvest's 4-arm gate compared different conjunction BANKS (a flat pool, a fixed-random bank, a
competitive-selected bank) of different underlying widths, so a width-matched null was needed to isolate
"binding structure" from "more units." This lever changes NEITHER the bank NOR its width: `--conj-select
competitive` with the identical `--conj-select-overcomplete 4 --conj-select-kwta-frac 0.1 --conj-n 1152` is
held fixed in both arms of this comparison, and only `--readout` (linear vs attention-gated-soft) differs.
The two arms therefore share the identical trained discriminant, C2 code, and bank membership for a given
seed -- a genuine PAIRED comparison with the confound already controlled for, not something a width-matched
null would add information to.

## Newly-added automated selftest (fills the gap the review flagged: mechanism existed, no CI pin existed)

`tests/test_vision_attention_gated_soft_selftest.py` (new, this commit) -- 3 sub-second pure-numpy tests on
`_attention_gated_soft_class_read` directly (no SimulationBridge/subprocess):
1. `test_disabled_exponent_reproduces_plain_linear_read_exactly` -- `attn_gain_exponent <= 0` (0.0 and -3.0
   both checked) reproduces `_spiking_class_read` bit-for-bit (predictions AND class-spike counts). Fails
   if the short-circuit ever stops being an exact identity.
2. `test_enabled_exponent_is_not_a_relabelled_linear_read` -- at the real default (1.0), the gated read
   must differ from the plain linear read and must not collapse every trial to one class. Fails if the
   gain/normalize path silently became a no-op.
3. `test_class_read_dispatcher_routes_attention_gated_soft` -- pins the `_class_read` dispatch table so a
   refactor cannot silently point `"attention-gated-soft"` at the wrong function.

Verified locally: all 3 pass (`.venv/bin/python -m pytest tests/test_vision_attention_gated_soft_selftest.py
-v` -> 3 passed). The middle test was iterated once during authoring -- an initial, overly strict assertion
("every class gets nonzero spikes somewhere") failed on the synthetic random-weight fixture for reasons
unrelated to the mechanism (a property of 4 random classes over 24 trials, not a defect) and was replaced
with a check for non-collapse (`len(np.unique(pred)) > 1`), which is the property actually being tested.

Runner-level byte-identical-off was ALSO re-verified fresh this session (not just trusted from the
2026-09-16 docstring, in case of drift): the `--attn-gain-exponent 0` tiny smoke
(`vlin_attngatedsoft_frac0_smoke.json`) is identical to a matching `--readout linear` run except
`elapsed_seconds`/`config` (direct JSON diff, both fields excluded, confirmed empty diff). The
`--attn-gain-exponent 1.0` smoke (`vlin_attngatedsoft_smoke.json`) is non-degenerate: `LEARNED_spkwta` moves
from 0.25 (off) to 0.3125 (on) on this tiny fixture, `scramble_null_pass` still holds, no NaNs/crashes.

## Pre-registered GO gate (fixed BEFORE the decisive 6-seed run; the paired-difference fix applied)

**Primary (mechanism-attribution) statistic -- PAIRED per-seed differences**, isolating the readout's own
causal contribution with the bank held fixed:

```
d_s = LEARNED_spkwta_held(attention-gated-soft, seed s) - LEARNED_spkwta_held(competitive-linear, seed s)
```

against the ALREADY-BANKED competitive-selection-at-satdiv-GO baseline
(`research/findings/raw/lanes/perception/conjbind_competitive_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_6seed.json`,
per-seed `LEARNED_spkwta_held`: seed42=0.5208, seed43=0.6354, seed44=0.5208, seed100=0.4792, seed101=0.6667,
seed102=0.6146, mean=0.5729). Report `mean(d)`, `SD(d)`, `SE(d) = SD(d)/sqrt(6)`, and the
paired t-statistic `t = mean(d)/SE(d)` with `df=5`. **`t_crit = 2.571`** <!--derived--> (scipy.stats.t.ppf(0.975, 5),
a standard statistical constant, not a measurement; two-tailed alpha=0.05, df=5) is fixed here, in
advance, as the significance threshold -- not tuned post-hoc, not derived from an unpaired control SD (the
exact defect the harvest's re-review flagged in the prior gate's tau).

**Precondition (must hold on every seed or the read is UNDEFINED, not banded):** `scramble_null_pass == 1.0`
on all 6 seeds of the new arm (the anti-cheat that already held 6/6 across all four harvest arms).

**Precedence (NEGATIVE evaluated first, per the harvest's issue-4 fix, so bands stay mutually exclusive).**
Every `2.571` below is the SAME pre-registered `t_crit` fixed above, not a fresh number: <!--derived-->

1. **NEGATIVE** if `mean(d) < 0` AND `|t| >= 2.571` (readout significantly hurts), OR if per-seed <!--derived-->
   `capability_go` (>= 5/6 threshold, the SAME definition the harvest table uses) falls BELOW the
   competitive baseline's own 4/6 while `mean(d) <= 0`.
2. **POSITIVE (task-relevant SURPASS)** — only if NEGATIVE does not fire — if `mean(d) > 0` AND
   `|t| >= 2.571` AND per-seed `capability_go >= 5/6` (i.e., this readout fixes the EXACT residual that <!--derived-->
   kept competitive-selection from being a confirmed win: seeds 42 and 101's `capability_go` failures
   despite above-average `LEARNED_spkwta_held`).
3. **PARTIAL** — `mean(d) > 0` but either `|t| < 2.571` (not distinguishable from noise at this n) or <!--derived-->
   `capability_go` improves without reaching 5/6. Progress, not closure.
4. **NEUTRAL** — `|t| < 2.571` and `capability_go` unchanged from competitive's 4/6. Honestly reported as <!--derived-->
   "cannot reject no-effect at n=6," not rounded up or down.

**Read honestly, not by the letter of the band** (carried forward from every prior lever in this lane): a
PARTIAL that does not clear `capability_go >= 5/6` is not a task GO regardless of how large `mean(d)` looks,
and `RATE_lin_ceiling_held` (the host-ridge diagnostic, unaffected by `--readout`) is reported per seed to
confirm the front-end/bank are untouched, exactly as the harvest table does.

## The decisive command (identical to the banked competitive-selection-at-satdiv-GO run, `--readout
attention-gated-soft --attn-gain-exponent 1.0` the ONLY change)

```bash
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --s2-norm satdiv --s2-satdiv-sigma 8 --s2-satdiv-scale 760 --s2-satdiv-n 2.0 \
    --ridge 1.0 --n-glimpses 6 --n-s2 96 \
    --conj-bind fixed --conj-select competitive --conj-select-overcomplete 4 --conj-select-kwta-frac 0.1 \
    --conj-n 1152 --conj-offset-max 4 --readout attention-gated-soft --attn-gain-exponent 1.0 \
    --heldout-position --scramble-null --seeds <SEED[S]> \
    --out <per-seed output path>
```

No `cfg.seed` trap applies: this is a standalone numpy derisk runner (no `CoreSimConfig`/`SimulationBridge`),
and every source of randomness (`_make_conjunction_bank`, `_select_conjunctions_competitive`, label-shuffle
nulls, the LIF spike-read draws) is threaded through `np.random.default_rng(seed * k + offset)` calls keyed
off the CLI `--seeds` argument directly (verified by direct code read, `research/runners/
_vision_lindiscrim_readout_derisk.py` lines 466/585/643/805/962/1112/1769/1793) -- so `--seeds` alone fully
controls determinism for this runner.

## Seed 42, run locally (sanity-only -- NOT a verdict; 5 of 6 seeds still pending on the pool)

`tools/mem_ok.sh 2` passed (`avail=12G need=2G -> 10G left`). Ran the command above with `--seeds 42` only,
locally, single-tenant (measured peak RSS via `resource.getrusage(RUSAGE_CHILDREN)`, a local measurement not
carried in the runner's own JSON output: **0.644 GB** <!--derived-->, 97.1s elapsed per the artifact's own
`elapsed_seconds` -- both well inside the lane's usual light-numpy-job envelope):

| quantity | seed 42 (this run) |
|---|---|
| `overall_verdict` | `LINDISCRIM-READOUT-NOGO` |
| `LEARNED_spkwta_held` | **0.25 (= chance)** |
| `LEARNED_linscore_held` | **0.5208** (the underlying signed score, unaffected by the LIF spiking port) |
| `RANDOM_spkwta_held` | 0.25 |
| `RATE_lin_ceiling_held` | 0.4896 (comparable to the competitive baseline's own seed-42 value at this op-point, confirming the bank/front-end are untouched) |
| `spkport_cost_linscore_minus_spkwta` (this runner's own diagnostic) | **0.2708** |
| `capability_go` / `learning_load_bearing` / `scramble_null_pass` | false / false / **1.0 (holds)** |

**Honest read of this one seed, not a verdict.** `scramble_null_pass=1.0` holds, so the anti-cheat
precondition for trusting the read is intact. But `LEARNED_spkwta_held` collapsed to exact chance (0.25)
while `LEARNED_linscore_held` (the identical trained discriminant's raw signed score, before the LIF spiking
port) still carries real signal (0.5208) -- the runner's own `spkport_cost` diagnostic quantifies this gap at
0.2708, the largest such gap seen anywhere in this lane's history for this seed/bank. The front end and bank
are confirmed untouched (`RATE_lin_ceiling_held` matches the competitive baseline), so the candidate
explanation is that `--read-gain 2.5 --read-bias 1.0` (tuned against the OTHER readout modes' drive
magnitudes) is miscalibrated for the net drive this readout produces AFTER stacking a SECOND satdiv
normalization (the attention gate's own, on top of the S2 front end's) -- a genuinely NEW combined operating
point this mechanism's own 2026-09-16 de-risk smoke never tested (that smoke used the default `--s2-norm z`
front end, not `--s2-norm satdiv`). This is flagged here explicitly, not hidden: it is a real, measured
result on one seed, not a bug found by code inspection (the override-precedence in `_apply_s2_norm` was
checked directly and is correct -- the attention gate's own per-trial data-driven sigma is genuinely used,
not the front end's `--s2-satdiv-sigma 8`). Per this lane's own standing 6-seed rule, no generalization or
NO-GO claim is made on n=1 -- the remaining 5 seeds, staged below, will show whether this is a per-seed LIF
calibration artifact (recoverable by re-tuning `--read-gain`/`--read-bias` for this stacked operating point,
a NEXT RUNG if so) or a general property of stacking two divisive normalizations in series. The bands above
are read once all 6 seeds land, in a separate finding -- this one stays PREREGISTERED, not NO-GO.

## Remaining seeds (43, 44, 100, 101, 102) staged on the pool, not awaited

Per the harness task's instruction, this finding does not wait for the pool results. See the commit's
`research/queue/pool.queue` additions (one line per seed) for the exact staged commands; each carries its
own `--out` path (`conjbind_attngatedsoft_AT_satdiv_GO_sig8_sc760_r1p0_nglim6_s<SEED>.json`) so the 5
pool-dispatched jobs cannot clobber each other or the seed-42 local file.
