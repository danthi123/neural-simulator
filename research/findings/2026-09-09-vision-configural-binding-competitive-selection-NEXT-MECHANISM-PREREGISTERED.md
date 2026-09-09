---
type: finding
status: partial
claim_check: synthesis
date: 2026-09-09
mechanism: RECURRENT/COMPETITIVE S2.5 configural-binding UNIT SELECTION (--conj-select competitive in _vision_lindiscrim_readout_derisk.py) -- an overcomplete candidate bank of fixed-random (a,b,Delta) conjunction units competes via lateral-inhibition/k-WTA on TRAINING data only, and the final bank keeps the candidates with the highest cumulative post-inhibition (surviving) drive; grounded in research/biology/conjunction-competitive-selection.md
lane: vision (identity readout, D-perception configural binding)
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTERED (mechanism built + smoke-verified this session; decisive 6-seed run QUEUED, not yet landed -- this doc records the mechanism, the gate, and the byte-identical-off proof; the decisive verdict is appended when the queued run returns, per the no-defer law)
artifacts:
  - research/findings/raw/lanes/perception/vlin_competitive_smoke.json
  - research/findings/raw/lanes/perception/conjbind_triple_n1152_heldoutpos_scramblenull_6seed.json
  - research/findings/raw/lanes/perception/conjbind_triple_n4608_heldoutpos_scramblenull_6seed.json
  - research/findings/raw/lanes/perception/conjbind_bindarm_n1152_heldoutpos_scramblenull_6seed.json
---

# The conjunction bank now competes for its own membership, instead of freezing a random draw

**Status: mechanism BUILT, GO gate PRE-REGISTERED, byte-identical-off PROVEN, tiny smoke runs end-to-end. The
decisive 6-seed run is QUEUED (0-token pool/GPU lane) and its result is not yet in — reporting the mechanism and
the gate now, per the no-defer law (closure cannot wait on the run to be convenient); the verdict lands as an
addendum to this same file the moment the queued run returns, exactly as the triple-order finding's own
width-compensation follow-up did.**

## Why this lever, not another width/order sweep (the diagnosis carried forward)

<!--derived-->
The pairwise fixed-random conjunction bank (`--conj-bind fixed`, default `--conj-order pair`) is the lane's best
load-bearing result: `PARTIAL-beat0/6-lb4/6` (`conjbind_bindarm_n1152_heldoutpos_scramblenull_6seed.json`). The
natural escalation — third-order (triple) conjunctions, on the theory that a unit specifying all 3 of this task's
`n_slots=3` is a strictly higher-SNR feature — was built, pre-registered, and run TWICE this session
(`research/findings/2026-09-09-vision-configural-binding-triple-order-conjunction-NEXT-MECHANISM-
PREREGISTERED.md`): matched-budget (`conj_n=1152`) landed `PARTIAL-beat0/6-lb2/6`, WORSE than the pairwise arm,
and a 4x-width follow-up (`conj_n=4608`) landed **IDENTICAL** `beat0/6-lb2/6` — width does not correct it, so
fixed-random third-order sampling is a NO-GO **regardless of budget**. The diagnosed cause: at any fixed unit
budget, sampling `(a,b,Delta)` (or `(a,b,c;Delta1,Delta2)`) uniformly at random draws a near-constant *fraction*
of informative units from a combinatorial space that keeps growing — widening the draw does not raise that
fraction, it just samples more randomness.

<!--derived-->
That diagnosis names the actual missing ingredient: **nothing in the fixed-random scheme lets an informative
unit be preferred over an uninformative one.** The wall-reframe question (CLAUDE.md: "what does the real system
run alongside this that we replaced with a constant?") points at **competition** — a real population of
candidate synapses is not drawn once and frozen; it is driven by real input and the unresponsive members are
suppressed relative to the responsive ones. This is the pre-registered NEXT MECHANISM named at the end of the
triple-order finding: "a recurrent/competitive binding stage among conjunction units (lateral inhibition / k-WTA
so the bank self-selects the informative conjunctions instead of sampling them fixed-random)."

## The mechanism (built, `_select_conjunctions_competitive`, `research/biology/conjunction-competitive-selection.md`)

<!--derived-->
Additive, gated by a new `--conj-select {fixed,competitive}` flag, default `fixed` (byte-identical to every
prior run — proven below):

1. **Expand.** Sample a candidate bank `--conj-select-overcomplete`x larger than the target `--conj-n` (default
   4x, matching the triple-order lever's own width-compensation scale) with the SAME established fixed-random
   sampler (`_make_conjunction_bank`/`_make_conjunction_bank_triple`, unchanged) — a DG-style expand-first step
   (`research/biology/dg-ca3-sparse-index.md`: "pattern separation results from the divergence of entorhinal
   inputs onto a LARGER number of granule cells").
2. **Drive on training data only.** Every candidate unit is driven on `tr_c1` (the training images) through the
   SAME S2 drive pipeline (`_extract_patches` -> L2-norm -> cosine match -> `_apply_s2_norm` -> `_kwta_over_
   templates`) and the SAME coincidence-AND primitive (`_bind_conjunctions`/`_bind_conjunctions_triple`,
   `research/biology/coincidence-binding.md`, unchanged). No labels are read (images only, never `tr_cls`) —
   held and scrambled splits are never touched at selection time (the same anti-leakage discipline
   `_bcm_learn_s2_templates` already follows for S2 template learning).
3. **Compete.** Per `(image, location)` presentation, a lateral-inhibition/k-WTA competition keeps only the top
   `--conj-select-kwta-frac` fraction of CANDIDATE units active, zeroing the rest — the IDENTICAL top-k-by-
   current-drive rule already established one function up in this same file (`_bcm_learn_s2_templates`'s
   `competitive_frac`, a Foldiak 1991 / Kohonen 1982-style winner-relative competitive-learning gate), reused
   here to SELECT structure rather than gate a weight update.
4. **Select.** The final `--conj-n` bank = the candidates with the highest CUMULATIVE surviving (post-inhibition)
   drive summed over every training presentation. Everything downstream (`_c2_spike_code`, `_c2_rate_code`, the
   learned signed-linear readout, every anti-cheat) is UNCHANGED — it consumes the selected `(pairs, offsets)`
   exactly as it consumed the fixed-random ones, so a capability change is attributable ONLY to which units got
   selected, not to any readout-side edit.

## Byte-identical-off, proven

<!--derived-->
Re-ran the triple-order tiny smoke (`vlin_triple_smoke.json`'s own recipe) after this build, with `--conj-select`
omitted (defaults to `fixed`): every decode/reframe/dissociation/verdict number matched the committed reference
exactly; the ONLY diff was the three new config fields (`conj_select`, `conj_select_overcomplete`,
`conj_select_kwta_frac`) now recorded in the output's `config` block, which is expected (new argparse defaults
are always echoed) and does not touch any computed quantity. `--conj-select competitive` itself was smoke-tested
separately (`vlin_competitive_smoke.json`, seed 42, tiny scale `n_s2=24, conj_n=96, candidate_n=384, k_sel=38`):
runs end-to-end, `frac_selected_never_won=0.0` (every selected unit won at least one competition — no degenerate
collapse), `scramble_null_pass=True`. This smoke is sanity-only (far below the decisive op-point) and makes no
capability claim.

## Pre-registered GO gate (fixed BEFORE the decisive run)

<!--derived-->
Identical criteria and anti-cheats to the pairwise and triple-order levers — only the conjunction-bank SELECTION
method changes, so any result is attributable to selection, not to a different gate:

- **task GO**: `beats_config_c_nogo` (per-seed `learn_spkwta_held >= 0.34 + 0.10`) **AND**
  `learning_load_bearing` (`learned - random >= 0.10`), each at **>=5/6 seeds**, under
  `--heldout-position --scramble-null`.
- **Verdict bands, fixed in advance:** `beat>=5/6 & lb>=5/6` = GO. Some (>0) beats/lb short of 5/6 = PARTIAL.
  `beat0 & lb0` = NO-GO for this lever — bank it and take the next mechanism (attention-gated readout, named by
  the triple-order finding as the remaining untried candidate); closure is not deferred either way.
- **Read honestly, not by the letter of the band** (the triple-order finding's own lesson): a PARTIAL that is
  *worse* than the pairwise arm's `lb4/6` is a regression, not progress, even though both are technically
  "PARTIAL". The comparison that matters is against `conjbind_bindarm_n1152_heldoutpos_scramblenull_6seed.json`
  (`lb4/6`, `RATE_lin_ceiling_held=0.3403`), the lane's best result to date, not just against the NO-GO floor.

## The decisive run (queued, not yet landed)

<!--derived-->
Same scale/op-point as the pairwise PARTIAL and both triple-order runs (`conj_n=1152`, `n_s2=96`,
`--heldout-position --scramble-null`), `--conj-select competitive` the only mechanism change (default
`--conj-select-overcomplete 4 --conj-select-kwta-frac 0.1`):

```bash
OUTDIR=research/findings/raw/lanes/perception
OUTFILE=conjbind_competitive_n1152_heldoutpos_scramblenull_6seed.json    # queued, does not exist yet as of this commit
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --ridge 0.5 --conj-bind fixed --conj-select competitive --conj-select-overcomplete 4 \
    --conj-select-kwta-frac 0.1 --conj-n 1152 --conj-offset-max 4 \
    --n-s2 96 --heldout-position --scramble-null --seeds 42 43 44 100 101 102 \
    --out "$OUTDIR/$OUTFILE"
```

This is a LIGHT numpy job (~1GB RSS, ~150s wall at this scale per the pairwise/triple-order runs' own measured
elapsed times) — routed via `tools/pool_queue.sh add` (0 Claude tokens, remote) so it does not compete with any
brain-loading GPU process. **Pre-registered read of that run:** if `beat>=5/6 & lb>=5/6`, this is a GO and the
lane's configural-binding wall is closed by this mechanism. If it PARTIALs at or above `lb4/6` with
`RATE_lin_ceiling_held` above 0.3403 (the pairwise ceiling), competitive selection is progress and the next rung
is tuning `--conj-select-overcomplete`/`--conj-select-kwta-frac`. If it lands at or below `beat0/6-lb2/6` (the
triple-order floor), competitive selection at this operating point is banked as a NO-GO and the next mechanism
is the attention-gated readout named by the triple-order finding — a verdict on this METHOD, never a license to
abandon the CAPABILITY.

## Reproduce

```bash
# tiny smoke (seconds, sanity only):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --seeds 42 --n-s2 24 --conj-bind fixed --conj-select competitive --conj-select-overcomplete 4 \
    --conj-select-kwta-frac 0.1 --conj-n 96 --conj-offset-max 2 \
    --n-pos-total 4 --n-ex 2 --n-glimpses 1 --heldout-position --scramble-null \
    --out research/findings/raw/lanes/perception/vlin_competitive_smoke.json

# the decisive 6-seed run (queued, see above; not yet run as of this commit)
```
