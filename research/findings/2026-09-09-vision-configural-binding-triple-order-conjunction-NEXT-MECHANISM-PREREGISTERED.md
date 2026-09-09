---
type: finding
status: partial
claim_check: synthesis
date: 2026-09-09
mechanism: THIRD-ORDER (triple) S2.5 configural-binding conjunction units (a,b,c;Delta1,Delta2), --conj-order triple in _vision_lindiscrim_readout_derisk.py -- built as a cascade of two applications of the established pairwise coincidence-binding primitive, targeting the diagnosed representational ceiling of the pairwise PARTIAL landing
lane: vision (identity readout, D-perception configural binding)
seeds: [42, 43, 44, 100, 101, 102]
verdict: PRE-REGISTERED GATE + decisive 6-seed run landed LINDISCRIM-READOUT-PARTIAL-beat0/6-lb2/6 -- WORSE than the pairwise arm it was meant to surpass (lb dropped 4/6->2/6, RATE_lin_ceiling_held dropped 0.3403->0.2917). Diagnosed cause: at matched unit budget (conj_n=1152), third-order sampling dilutes combinatorial coverage far more than it raises per-unit specificity. This lever is a NO-GO at matched budget; the width-compensated follow-up (conj_n=4608, 4x) LANDED IDENTICAL (beat0/6-lb2/6) — width does NOT correct it, so the third-order fixed-random bank is a NO-GO regardless of budget. Next mechanism (not deferred): recurrent/competitive binding stage, then attention-gated readout.
artifacts:
  - research/findings/raw/lanes/perception/conjbind_bindarm_n1152_heldoutpos_scramblenull_6seed.json
  - research/findings/raw/lanes/perception/conjbind_triple_n1152_heldoutpos_scramblenull_6seed.json
  - research/findings/raw/lanes/perception/vlin_triple_smoke.json
---

# Third-order conjunction units — a pre-registered lever that landed WORSE than the pairwise PARTIAL it targeted

**Status:** mechanism built, GO gate pre-registered BEFORE the decisive run, decisive 6-seed run landed (fast —
148s on CPU) DURING this same build session. Reporting the actual result, not the anticipated one: this specific
lever (fixed-random third-order conjunctions at the SAME 1152-unit budget as the pairwise arm) is a NO-GO — it
underperforms the pairwise PARTIAL it was built to surpass, not just fails to clear the floor. A follow-up lever
(width compensation) is diagnosed and queued below; closure is not deferred.

## Why this lever was tried (the diagnosis from the pairwise PARTIAL)

<!--derived-->
From the pairwise PARTIAL artifact's own `reframe_means`/`headroom`, averaged over its 6 seeds:

<!--derived-->
| quantity | value | reading |
|---|---|---|
| `learned_spkwta_held` (the fully-spiking signed-discriminant readout) | 0.3281 | at, not above, the floor |
| `config_c_nogo_floor` | 0.34 | the #72/#75 fully-spiking NO-GO |
| `rate_lin_ceiling_held` (idealized non-spiking linear readout, SAME features) | 0.3403 | the best possible LINEAR read of the PAIRWISE front end sits AT the floor, not above it |
| `spkport_cost` (rate-vs-spike gap attributable to the spike port itself) | 0.0017 | negligible — the spike port is NOT the bottleneck |

<!--derived-->
The load-bearing readout class (FF-inhibition + temporal integration) was confirmed NOT the residual — its own
ceiling on the pairwise features was already at the floor, and width/normalization sweeps of that SAME pairwise
layer were separately exhausted (2026-09-01/2026-09-03 findings). The wall-reframe question ("what companion
process did we replace with a constant?") pointed at the FRONT-END REPRESENTATION: this task's objects have
`n_slots=3` (`_vision_hmax_hierarchy_derisk.py:131-174`, each class a distinct permutation of 3 oriented strokes
across 3 slots), and a pairwise unit `AND(a@p, b@p+Delta)` can specify at most 2 of the 3 slots per unit — so a
unit ANDing all THREE slot-relevant templates in one place looked like a strictly higher-SNR feature, motivating
`--conj-order triple` (built as a cascade of two applications of the SAME established pairwise coincidence-
binding primitive, `research/biology/coincidence-binding.md` — no new biology claim; the two-layer branch-then-
soma dendritic correlate is Poirazi, Brannon & Mel 2003, Neuron 37:989, and Brincat & Connor 2004, Nat. Neurosci.
7:880 independently report IT neurons conjunctively encoding multi-part shape arrangements).

## Pre-registered GO gate (fixed BEFORE the decisive run — unchanged, quoted verbatim for the record)

- **task GO**: `beats_config_c_nogo` (per-seed `learn_spkwta_held >= 0.34 + 0.10`) **AND**
  `learning_load_bearing` (`learned - random >= 0.10`), each at **>=5/6 seeds**, under
  `--heldout-position --scramble-null`.
- **Verdict bands, fixed in advance:** `beat>=5/6 & lb>=5/6` = GO. Some (>0) beats/lb short of 5/6 = PARTIAL.
  `beat0 & lb0` = NO-GO for this lever — bank it and take the next mechanism; closure is not deferred either way.
- This band scheme did not anticipate a THIRD outcome that in fact occurred — a PARTIAL that is *worse* than the
  PARTIAL it was meant to improve on. Read honestly below rather than reported under the letter of the band.

## The decisive result — WORSE than the pairwise arm, not better

<!--derived-->
6 seeds (42/43/44/100/101/102), `conjbind_triple_n1152_heldoutpos_scramblenull_6seed.json`, same op-point as the
pairwise PARTIAL (`conj_n=1152`, `n_s2=96`, `--heldout-position --scramble-null`), only `--conj-order triple`
added:

<!--derived-->
| quantity | pairwise PARTIAL (prior) | triple (this run) | direction |
|---|---|---|---|
| `overall_verdict` | `PARTIAL-beat0/6-lb4/6` | `PARTIAL-beat0/6-lb2/6` | **worse** |
| `learned_spkwta_held` | 0.3281 | 0.3212 | flat/worse |
| `RATE_lin_ceiling_held` (idealized linear ceiling, no spike port) | 0.3403 | 0.2917 | **worse — the predicted lift did not happen; the ceiling FELL** |
| `learning_load_bearing` (>=5/6 needed) | 4/6 | 2/6 | **worse** |
| `beats_config_c_nogo` (>=5/6 needed) | 0/6 | 0/6 | unchanged (still 0) |
| `scramble_null_pass` | 6/6 | 6/6 | unchanged (anti-cheat still intact — this is not an instrument failure) |

<!--derived-->
The prediction was that a genuinely higher-order feature would raise the front end's own linear ceiling. Instead
the ceiling FELL (0.3403 -> 0.2917) and the readout's already-thin load-bearing margin shrank (4/6 -> 2/6). The
anti-cheats stayed intact (scramble-null still 6/6, so this is a real measurement, not a broken instrument) — the
lever itself is what failed.

## Diagnosed cause: combinatorial dilution dominates the specificity gain, at matched unit budget

The pairwise bank samples `(a,b,Delta)` from a space of `n_s2^2 * |offsets| = 96^2 * 8 ~= 74k` possible triples
into 1152 units (~1.6% coverage). The triple bank samples `(a,b,c,Delta1,Delta2)` from
`n_s2^3 * |offsets|^2 = 96^3 * 64 ~= 56.6M` possible quadruples into the SAME 1152 units (~2e-5 coverage) — **~800x
sparser coverage of a combinatorial space that grew ~760x**, for the same unit budget. Two compounding effects
follow: (1) the chance any unit lands near a behaviourally-relevant triple is far lower than for a relevant pair;
(2) `min`/`prod` of THREE independent-ish sub-1 nonnegative quantities is smaller in expectation than of two,
shrinking the per-unit drive relative to the LIF firing threshold and the ridge readout's noise floor — visible
directly in the RATE (non-spiking) ceiling falling, which isolates this from anything about the spike port. The
theoretical argument for higher specificity per unit was correct in isolation; it was outweighed by the
combinatorial cost of reaching that unit at all, at a budget that was never re-scaled for the larger space.

## No-defer: the width-compensated follow-up — ✅ LANDED, width does NOT recover it (NO-GO confirmed regardless of budget)

If combinatorial dilution is the dominant effect, width should partially compensate it — the pairwise arm's own
history validated exactly this style of lever (`conjbind_prod_n1024..n2304_6seed.json` width sweeps). A
**4x-wider triple bank (`conj_n=4608`)** is the direct, cheap test of that hypothesis, queued as a follow-up (same
0-token GPU-queue lane, ~4x the compute of this run, well under 10 minutes):

```bash
OUTDIR=research/findings/raw/lanes/perception
OUTFILE=conjbind_triple_n4608_heldoutpos_scramblenull_6seed.json    # queued, does not exist yet as of this commit
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --ridge 0.5 --conj-bind fixed --conj-order triple --conj-n 4608 --conj-offset-max 4 \
    --n-s2 96 --heldout-position --scramble-null --seeds 42 43 44 100 101 102 \
    --out "$OUTDIR/$OUTFILE"
```

**Pre-registered read of THAT run** (fixed now, before it lands): if `RATE_lin_ceiling_held` recovers to at or
above the pairwise arm's 0.3403 and `learning_load_bearing` recovers to >=4/6, combinatorial dilution is the full
explanation and further width sweeps (matching the pairwise arm's own history) are the next rung. If it does NOT
recover even at 4x width, the higher-order lever is banked as a NO-GO regardless of width, and the next mechanism
is one of the two NOT-yet-attempted candidates named by the original task scoping: a recurrent/competitive binding
stage among conjunction units (rather than a purely feedforward fixed-random bank), or an attention-gated readout
that reweights which units the class populations listen to per-trial. Either way, per the project's standing law,
this is a verdict on a METHOD (fixed-random third-order sampling at a given budget), not a license to abandon the
CAPABILITY — the pairwise PARTIAL (`lb4/6`) remains the best load-bearing result in this lane and is not
retracted by this lever's failure to beat it.

**LANDED RESULT (`conjbind_triple_n4608_heldoutpos_scramblenull_6seed.json`, harvested to main): `LINDISCRIM-READOUT-PARTIAL-beat0/6-lb2/6` — IDENTICAL to the matched-budget (n1152) verdict.** 4x width did NOT
recover `RATE_lin_ceiling_held` to the pairwise 0.3403, and `learning_load_bearing` stayed at 2/6 (not >=4/6). Per
the pre-registered read above, the higher-order (third-order fixed-random conjunction) lever is therefore a
**NO-GO regardless of budget** — combinatorial dilution is not the correctable bottleneck; the fixed-random
third-order bank itself is the wrong mechanism at this op-point. **NEXT MECHANISM (no-defer, not yet attempted):**
a **recurrent/competitive binding stage** among conjunction units (lateral inhibition / k-WTA so the bank
self-selects the informative conjunctions instead of sampling them fixed-random), then — if that also stalls — an
**attention-gated readout** that reweights which units each class population listens to per-trial. The pairwise
PARTIAL (`lb4/6`) remains the best load-bearing result in this lane and is not retracted. This is a verdict on the
METHOD (fixed-random higher-order sampling), never the CAPABILITY.

## What the smoke showed (sanity only, run before the decisive eval — unaffected by the above)

<!--derived-->
`vlin_triple_smoke.json`, seed 42 only, tiny scale (`n_s2=24, conj_n=96, n_pos_total=4, n_ex=2, n_glimpses=1`, far
below the decisive op-point): `LEARNED_spkwta_held=0.3125`, `RANDOM_spkwta_held=0.25`,
`RATE_lin_ceiling_held=0.1875`, `scramble_null_pass=True`, elapsed 1.7s. Re-run byte-for-byte identical (checked
by diffing two independent invocations; only `--out` and `elapsed_seconds` differed), confirming the new code
path is deterministic and the pairwise (`--conj-order pair`, default) path is provably unchanged (the dispatch
`bind_fn = _bind_conjunctions_triple if conj_order=="triple" else _bind_conjunctions` is an identical call to the
pre-existing one when `pair`). This smoke only ever confirmed the mechanism runs correctly end-to-end; it made no
capability claim, so it is not contradicted by the decisive result above.

## Reproduce

```bash
# tiny smoke (seconds, sanity only):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --seeds 42 --n-s2 24 --conj-bind fixed --conj-order triple --conj-n 96 --conj-offset-max 2 \
    --n-pos-total 4 --n-ex 2 --n-glimpses 1 --heldout-position --scramble-null \
    --out research/findings/raw/lanes/perception/vlin_triple_smoke.json

# the decisive 6-seed run reported above (matched budget, conj_n=1152):
SIM_BACKEND=numpy .venv/bin/python -u -m research.runners._vision_lindiscrim_readout_derisk \
    --ridge 0.5 --conj-bind fixed --conj-order triple --conj-n 1152 --conj-offset-max 4 \
    --n-s2 96 --heldout-position --scramble-null --seeds 42 43 44 100 101 102 \
    --out research/findings/raw/lanes/perception/conjbind_triple_n1152_heldoutpos_scramblenull_6seed.json
```
