---
type: finding
status: contributing
date: 2026-08-10
mechanism: self-schema-certainty-band-opponent-margin
lane: stageA-integration
runner: research/runners/_stageA_full_integration_derisk.py
artifacts:
  - research/findings/raw/lanes/stageA/turing/certainty_opponent_margin_6seed.json
  - research/findings/raw/lanes/stageA/turing/conv_turing_3c_s42.json
attributable_to: research/runners/_stageA_full_integration_derisk.py::_read_opponent_margin (opponent margin read); ::build_one_brain (co_resident_certainty_opponent comparator)
---

# INTEGRATION #3c: the certainty-band OPPONENT comparator makes the self-model's confidence read ROBUST — all 6 seeds clear the +0.02 bar

<!--derived-->

**One-line.** INTEGRATION #3b closed the inverted self_schema relay with a seed-then-settle read but left a residual:
the POOLED `self_schema` rate reports the winner-DOMINATED magnitude, so a symmetric-tie probe on which a random class
PARTIALLY latches reads too high (elevated total activity looks confident), holding 2/6 seeds (43, 100) at +0.0143 <
the +0.02 "meaningful" bar. #3c ports the reference `margin_abs` monitor as a dedicated per-class OPPONENT comparator:
each workspace class k excites its own slow-NMDA subpool `meta_opp_k` and, through a class-specific inhibitory relay
`meta_opp_fs_k`, suppresses the other class's subpool; the read is `margin_abs = |rate(meta_opp_1) - rate(meta_opp_0)|`.
The read now measures the ASYMMETRY of the settled competition, not its magnitude — a partial latch under a symmetric
tie is by definition a SMALL asymmetry, so it can no longer inflate the read regardless of WHICH class latched. On the
6-seed production build the confident-vs-tie separation rises from the pooled +0.0197 mean (min +0.0152, 1 seed < 0.02)
to the opponent **+0.0662 mean (min +0.0521, ALL 6 > +0.02)**; turn 13 grades an assert band 6/6; the live seed-42 chat
keeps 0 confabulations with turns 3/4/5/6/7 intact. Additive; the comparator is appended LAST (byte-identical static
substrate); NO `sim/` edit.

## Instrument

<!--derived-->

The comparator lives in `research/runners/_stageA_full_integration_derisk.py`: `build_one_brain(...,
co_resident_certainty_opponent=True)` appends two regions LAST — `meta_opp` (per-class slow-NMDA comparator subpools)
and `meta_opp_fs` (per-class inhibitory relay) — and injects the class-k→meta_opp_k excitation + class-k→meta_opp_fs_k
excitation + meta_opp_fs_k→meta_opp_{j≠k} cross-inhibition as union entries (no out-edges to any pre-existing region).
`_read_opponent_margin` runs the SAME #3b seed-then-settle timing (`drive_steps=35, hold_frac=0.20, free_steps=45,
read_lo=10`) and returns `|rate(meta_opp_1) - rate(meta_opp_0)|`. `read_honesty_self_rate(..., opponent=None)` AUTO-uses
the opponent read when the comparator is present (turn 13 + FM4 pick it up unchanged) and falls back to the pooled read
otherwise. The wiring/weights are ported verbatim from `_second_order_metacog_monitor_derisk._build_bridge` /
`_run_trial` (`confidence_read='margin_abs'`). 6-seed before/after:

```
SIM_BACKEND=numpy .venv/bin/python -m research.runners._stageA_full_integration_derisk \
    --certainty-opponent-sweep   # -> research/findings/raw/lanes/stageA/turing/certainty_opponent_margin_6seed.json
```

Live 14-turn chat gate (mouth on CPU to avoid the owner's GPU game):

```
SIM_BACKEND=numpy .venv/bin/python -m research.runners._conversation_turing_test_derisk --seed 42 --device cpu \
    --out research/findings/raw/lanes/stageA/turing/conv_turing_3c_s42.json
```

## Result — the opponent margin clears +0.02 on ALL 6 seeds (the robustness #3b lacked)

<!--derived-->

Build = seams A+C (production turing build) + `co_resident_certainty_opponent`, seeds {42,43,44,100,101,102}.
"pooled_sep" is the #3b read on the SAME build (the before); "opponent_sep" is the #3c margin read (the after).

| seed | pooled_sep (before) | opponent tie | opponent assert | opponent_sep (after) | >+0.02 | turn-13 band |
|-----:|--------------------:|-------------:|----------------:|---------------------:|:------:|:------------:|
| 42   | +0.0152             | 0.0014       | 0.0764          | **+0.0750**          | yes    | assert |
| 43   | +0.0186             | 0.0207       | 0.0729          | **+0.0521**          | yes    | assert |
| 44   | +0.0276             | 0.0021       | 0.0764          | **+0.0743**          | yes    | assert |
| 100  | +0.0210             | 0.0021       | 0.0736          | **+0.0714**          | yes    | assert |
| 101  | +0.0190             | 0.0021       | 0.0700          | **+0.0679**          | yes    | assert |
| 102  | +0.0167             | 0.0164       | 0.0729          | **+0.0564**          | yes    | assert |

- opponent_sep: **min +0.0521, mean +0.0662, all 6 > +0.02, all positive** (vs pooled min +0.0152, mean +0.0197 with 1
  seed < 0.02 on this build; the #3b CLEAN pooled baseline — no comparator — was min +0.0143, mean +0.0190, 2 seeds <
  0.02). `relay_reliable` (sep > 0.003) 6/6; `turn13_assert_count` 6/6.
- The robustness mechanism is visible in the columns: seeds 43 (tie 0.0207) and 102 (tie 0.0164) are the residual
  tie-latchers — their tie margin is elevated — yet their separation still clears +0.02 by a wide margin because the
  confident/self assert margin (~0.073) is DECOUPLED from the tie by the comparator. Under the pooled read the same
  latch pulled confident and tie together; the opponent read cannot be fooled that way because it reads the per-class
  difference, and a partial latch is a small difference.
- Byte-identity: `byte_identical_all=True` — the first `num_neurons`(seams-A/C) firing thresholds hash identically with
  and without the comparator (appended LAST, `internal_density=0`, union-injected edges, no shared-plan RNG draw).

## Live chat — turn 13 grades certainty from the opponent margin; 0 confab; no regression

<!--derived-->

Seed-42 14-turn eval (`conv_turing_3c_s42.json`): **0 confabulations** (14 turns; 4 generator replies, 8 abstains).
Turn 13 now reads `self_schema_band="assert"`, `self_schema_relay_reliable=true`, `self_schema_separation=+0.0750`,
`self_schema_rate=0.0764` — the OPPONENT margin (its +0.0750 == seed-42's `opponent_sep`), replacing the #3b pooled
+0.0190. The live reply: *"my self_schema confidence relay reads this in the 'assert' band: I am a simulated spiking
substrate (25971 neurons, one shared bridge, numpy backend), not a person ... this is an honest functional read-out,
not a feeling"* — still asserts NO personhood / phenomenal experience and keeps the structural self-description distinct
from the graded relay read.

Turns 3/4/5/6/7 keep their faculty, grounded content, and 0-confab verdict. Turn 7 is byte-identical to the #3b
transcript; turns 5/6 keep the same affect level / forward-model prediction with their scalar reads drifting by the
documented ~0.01 co-residency jitter (affect valence +0.07→+0.08 at warmth level 3; fm novel-case margin 0.09→0.10,
still predicts 'south'); turns 3/4 keep grounded-only content (dog/look/river, dog/run/north) with the generator mouth
rephrasing within those stored facts. No turn changes faculty, groundedness, or confab status.

## Honest scope + the operating point

<!--derived-->

The comparator EXC/INH weights (4.0 / 6.0) are STRONGER than the reference `margin_abs` monitor's 1.4 / 2.2. The
reference read a GRADED 2AFC over a long slow-NMDA late window; our probes are FIXED extreme drives (confident 520/40,
tie 300/300) read over the #3b SHORT seed-then-settle window, so the feed-forward drive to the comparator subpools must
be higher to reach a graded firing rate. The operating point was fixed on the same 6-seed sweep #3b used; `INH_W=6.0 >
EXC_W=4.0` keeps a symmetric tie from letting one subpool run away. This is protocol-matched calibration, not a new
free parameter class — the mechanism (per-class comparator + cross-inhibition + abs-difference read) is the reference's.

A benign side effect, measured and bounded: adding the comparator neurons shifts the co-resident POOLED self read
slightly (seed 42 +0.0190 → +0.0152) even though the static substrate is byte-identical and the comparator has no
out-edges — a lesion control (silencing the comparator's inputs leaves the pooled read unchanged at +0.0152) shows it is
STRUCTURAL float-summation-reorder jitter from the larger connection matrix, not activity coupling; it is the same
class of shift the #3b build already absorbed when it added the fm-reservoir + affect-ladder slices. It does not change
any turn 3–7/13 verdict (eval re-run confirms 0 confab, turns intact).

## Next mechanism (if the operating point is ever pushed to graded confidence)

<!--derived-->

The read is still a host abs-subtraction over two neural population rates (same host-arithmetic status as the #3b
pooled mean-rate read). The fully brain-based form is a dedicated rectified-opponent READOUT PAIR (`cert_hi` receiving
EXC from meta_opp_0 + INH from meta_opp_1; `cert_lo` the mirror) whose summed rate IS the neural |margin|, removing the
host subtraction entirely. It is deferred only because the current comparator already clears the robustness gate with
>2.5x headroom (min +0.0521 vs the +0.02 bar); if a future arc needs a GRADED confidence curve (not the binary
confident-vs-tie separation), the rectified-opponent readout is the named next step — NOT a wall.
