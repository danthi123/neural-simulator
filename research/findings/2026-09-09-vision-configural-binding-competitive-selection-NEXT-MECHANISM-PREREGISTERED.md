---
type: finding
status: partial
claim_check: synthesis
date: 2026-09-09
mechanism: RECURRENT/COMPETITIVE S2.5 configural-binding UNIT SELECTION (--conj-select competitive in _vision_lindiscrim_readout_derisk.py) -- an overcomplete candidate bank of fixed-random (a,b,Delta) conjunction units competes via lateral-inhibition/k-WTA on TRAINING data only, and the final bank keeps the candidates with the highest cumulative post-inhibition (surviving) drive; grounded in research/biology/conjunction-competitive-selection.md
lane: vision (identity readout, D-perception configural binding)
seeds: [42, 43, 44, 100, 101, 102]
verdict: DECISIVE 6-seed run LANDED -- LINDISCRIM-READOUT-PARTIAL-beat4/6-lb6/6, the LANE'S BEST RESULT TO DATE by a wide margin (RATE_lin_ceiling_held 0.4288, vs the pairwise arm's 0.3403 and the triple-order arm's 0.2917-0.3403; learning_load_bearing PERFECT 6/6, vs pairwise 4/6 and triple-order 2/6). Not yet a task GO (needs beat>=5/6, landed 4/6 -- one seed, 44, missed the beat-margin by 0.0025 raw fraction; one seed, 100, is a clear miss at 0.32). Per the pre-registered read, this is PROGRESS (lb>=4/6 and ceiling above the pairwise arm's), not a banked NO-GO -- the next rung is tuning --conj-select-overcomplete/--conj-select-kwta-frac, not a new mechanism.
artifacts:
  - research/findings/raw/lanes/perception/conjbind_competitive_n1152_heldoutpos_scramblenull_6seed.json
  - research/findings/raw/lanes/perception/vlin_competitive_smoke.json
  - research/findings/raw/lanes/perception/conjbind_triple_n1152_heldoutpos_scramblenull_6seed.json
  - research/findings/raw/lanes/perception/conjbind_triple_n4608_heldoutpos_scramblenull_6seed.json
  - research/findings/raw/lanes/perception/conjbind_bindarm_n1152_heldoutpos_scramblenull_6seed.json
---

# The conjunction bank now competes for its own membership, instead of freezing a random draw

**Status: mechanism BUILT, GO gate PRE-REGISTERED before the run, byte-identical-off PROVEN, and the decisive
6-seed run has LANDED (run as a single local numpy job, ~88s, immediately after the mechanism build in this same
session -- not queued after all, since the queue's own argparse-validity gate checks the CANONICAL (unmerged)
checkout and correctly refused a flag that does not exist there yet; see "Why local, not queued" below).**
**Result: `LINDISCRIM-READOUT-PARTIAL-beat4/6-lb6/6` — the lane's best result to date, a large step up from every
prior lever, but not yet a task GO.**

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

## Why local, not queued through `tools/pool_queue.sh`

<!--derived-->
`tools/pool_queue.sh add` validates a candidate command by running the CANONICAL (main-branch) checkout's copy
of the runner with `--help` and checking every flag in the command resolves against it — a real seam-closing
check (a 2026-07-31 finding: a runner queued from a branch the pool nodes cannot see dispatches and dies
silently). This build lives on a topic branch inside an isolated worktree; the canonical checkout's working
tree does not yet have `--conj-select`/`--conj-select-overcomplete`/`--conj-select-kwta-frac` (they land on
`main` when this branch merges), so the queue correctly REFUSED the job (`does not accept: --conj-select
--conj-select-kwta-frac --conj-select-overcomplete`) rather than silently staging a job that would die on the
pool nodes. Per the task's own explicit fallback ("run it as a single local numpy job"), the decisive run was
executed directly instead — a single LIGHT numpy job (88.2s measured, well under the ~1GB / ~150s envelope
named for this lane), no GPU, no brain-loading process, nothing else running concurrently.

## The decisive result — the lane's best PARTIAL by a wide margin, not yet a GO

<!--derived-->
Same scale/op-point as the pairwise PARTIAL and both triple-order runs (`conj_n=1152`, `n_s2=96`,
`--heldout-position --scramble-null`, `--ridge 0.5`), `--conj-select competitive` (default
`--conj-select-overcomplete 4 --conj-select-kwta-frac 0.1`) the only mechanism change:

<!--derived-->
| quantity | pairwise PARTIAL (prior best) | triple n1152 (NO-GO) | triple n4608 (NO-GO) | **competitive (this run)** |
|---|---|---|---|---|
| `overall_verdict` | `PARTIAL-beat0/6-lb4/6` | `PARTIAL-beat0/6-lb2/6` | `PARTIAL-beat0/6-lb2/6` | **`PARTIAL-beat4/6-lb6/6`** |
| `LEARNED_spkwta_held` | 0.3281 | 0.3212 | (identical to n1152) | **0.4549** |
| `RANDOM_spkwta_held` | (n/a, not tabulated here) | -- | -- | 0.2535 |
| `RATE_lin_ceiling_held` | 0.3403 | 0.2917 | 0.2917 | **0.4288** |
| `learning_load_bearing` (>=5/6) | 4/6 | 2/6 | 2/6 | **6/6 (perfect)** |
| `beats_config_c_nogo` (>=5/6) | 0/6 | 0/6 | 0/6 | **4/6** |
| `beats_config_c_nogo_raw` (>0.34, no margin) | -- | -- | -- | 5/6 |
| `scramble_null_pass` | 6/6 | 6/6 | 6/6 | **6/6** |

<!--derived-->
Per-seed (`LEARNED_spkwta_held` / `RATE_lin_ceiling_held` / beats-margin / load-bearing): seed 42: 0.4479 /
0.4062 / **beat** / lb; seed 43: 0.4688 / 0.4688 / **beat** / lb (also full `capability_go`); seed 44: 0.4375 /
0.4167 / miss (needs >=0.44, landed 0.4375 — short by 0.0025) / lb; seed 100: 0.3229 / 0.2917 / miss (the clear
outlier — below even the raw 0.34 floor) / lb; seed 101: 0.5729 / 0.5312 / **beat** / lb; seed 102: 0.4792 /
0.4583 / **beat** / lb (also full `capability_go`). Every seed's `conj_select` diagnostic shows
`candidate_n=4608`, `k_sel_per_presentation=461`, `frac_selected_never_won=0.0` (every one of the 1152 selected
units won at least one competition somewhere in training — no degenerate collapse to a handful of units).

<!--derived-->
**Determinism verified**: seed 42 alone was re-run independently and its full per-seed row byte-compared equal
to the 6-seed run's seed-42 row (every decode/reframe/dissociation/verdict/conj_select field identical).

## Reading this honestly — progress, not yet closure

<!--derived-->
Per the pre-registered read fixed in the "decisive run" section originally: `learning_load_bearing` landed at
6/6 (>= the 4/6 recovery bar) and `RATE_lin_ceiling_held` (0.4288) landed well above the pairwise arm's 0.3403 —
both trigger the "competitive selection is progress" branch, not the "banked NO-GO, take the attention-gated
readout" branch. This is NOT a task GO: `beats_config_c_nogo` needs `>=5/6` and landed at `4/6` (seed 44 missed
the `+0.10` margin by a hair — `0.4375` vs `0.44` needed; seed 100 is a genuine miss at `0.32`, well below even
the raw `0.34` floor). The next rung, named by the pre-registered read, is tuning
`--conj-select-overcomplete`/`--conj-select-kwta-frac` (this run's untuned defaults, 4x/0.1, were never swept) —
seed 100's shortfall in particular is worth checking against a wider candidate pool or a softer competition
before concluding the operating point is exhausted. This is a verdict on the DEFAULT operating point of this
mechanism, not on the mechanism itself, which is already the best lever this lane has produced.

## An honest risk this result carries (named, not hidden)

<!--derived-->
The competitive selection step reads `tr_c1` — the SAME small training-image set (6 examples/class) the
downstream readout is then fit on — so the informativeness criterion used to choose which conjunctions survive
and the criterion used to fit the linear readout share one thin data source. This is the identical "honest
thin-data risk" `_bcm_learn_s2_templates`'s own docstring already names for S2 template learning, not a new
concern this mechanism introduces. It is NOT label leakage (selection reads images only, never `tr_cls`), and
the held-out-position (contiguous-block spatial extrapolation, never bracketed by training neighbours) and
scramble-null (the LEARNED readout itself must collapse to chance on pixel-scrambled held images) anti-cheats
are both intact at 6/6 — a readout that had merely memorized train-specific statistics through this shared data
path would not be expected to survive spatial extrapolation or collapse cleanly under scrambling, so this is
evidence against pure overfitting, though not a formal proof of its absence.

## Reproduce

```bash
# tiny smoke (seconds, sanity only):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --seeds 42 --n-s2 24 --conj-bind fixed --conj-select competitive --conj-select-overcomplete 4 \
    --conj-select-kwta-frac 0.1 --conj-n 96 --conj-offset-max 2 \
    --n-pos-total 4 --n-ex 2 --n-glimpses 1 --heldout-position --scramble-null \
    --out research/findings/raw/lanes/perception/vlin_competitive_smoke.json

# the decisive 6-seed run reported above:
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --ridge 0.5 --conj-bind fixed --conj-select competitive --conj-select-overcomplete 4 \
    --conj-select-kwta-frac 0.1 --conj-n 1152 --conj-offset-max 4 \
    --n-s2 96 --heldout-position --scramble-null --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/lanes/perception/conjbind_competitive_n1152_heldoutpos_scramblenull_6seed.json
```
