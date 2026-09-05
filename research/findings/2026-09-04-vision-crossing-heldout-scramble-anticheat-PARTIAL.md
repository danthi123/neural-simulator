---
type: finding
status: partial
claim_check: measured
date: 2026-09-04
mechanism: vision configural-binding crossing — held-out-position + scramble-null anti-cheats (open Q1)
lane: D·Perception
seeds: [42, 43, 44, 100, 101, 102]
artifacts:
  - research/findings/raw/lanes/perception/conjbind_widthctrl_n1152_heldoutpos_scramblenull_6seed.json
verdict: >
  The flat width-matched ELM crossing (n_s2=1152, no binding) is REAL but WEAKER than the earlier interleaved
  split showed — weaker, in fact, than an earlier version of this verdict itself reported, which paired two
  DIFFERENT metrics (corrected 2026-09-05, adversarial-verify w3qhweujd; see below). Under the two anti-cheats
  open-Q1 named — a contiguous held-out-position block split (genuine spatial extrapolation, never bracketed by
  trained neighbours) AND a scramble-null on the LEARNED spiking-WTA readout itself — the run returns
  `LINDISCRIM-READOUT-PARTIAL-beat3/6-lb6/6`. Apples-to-apples against the interleaved-split run
  (conjbind_widthctrl_n1152_6seed.json): the position-invariance-gated capability metric (capability_go, which
  ANDs held-out object decoding with position_pooled_out) falls from 5/6 to 2/6, and the plainer
  beats-the-NO-GO-floor metric falls from 6/6 to 3/6 — two separate, correctly-paired comparisons, not one
  metric compared against the other. LEADING with the stricter, gated metric: only 2/6 seeds are a genuine
  position-invariant capability crossing under both anti-cheats. Learning is load-bearing on ALL 6 seeds (learned
  >> random spiking-WTA) — the crossing is NOT a pure ELM-overfit — but its held-out-POSITION generalization is
  the real residual, and one of the three floor-clearing seeds (102) clears the floor while LEAKING position
  information (decodes held-out position at 0.4167, above the 0.40 invariance margin) rather than genuinely
  generalizing. A verdict on the METHOD's robustness, not the capability.
---

# Vision crossing under the harder anti-cheats: real but not position-robust

## What ran
`_vision_lindiscrim_readout_derisk.py` (the build-ahead-added `--heldout-position` + `--scramble-null` flags), the
flat width-matched control arm (`--conj-bind none --n-s2 1152 --ridge 0.5`), 6 seeds, on the mini-PC pool (numpy).
Result: `research/findings/raw/lanes/perception/conjbind_widthctrl_n1152_heldoutpos_scramblenull_6seed.json`,
overall verdict `LINDISCRIM-READOUT-PARTIAL-beat3/6-lb6/6` (the artifact's own verdict-string names the
floor-clearing count specifically; the stricter position-invariance-gated `capability_go` count discussed below
is a separate field, not part of this string).

## The numbers (apples-to-apples — leading with the gated capability metric)

> **⚠️ CORRECTION (2026-09-05, adversarial-verify `w3qhweujd`):** an earlier version of this section compared
> THIS run's `beats_config_c_nogo` (3/6) against the PRIOR run's `capability_go` (5/6) — two different fields,
> a wrong-quantity comparison. Corrected below with both metrics paired against their own prior-run counterpart
> (source: `research/findings/raw/lanes/perception/conjbind_widthctrl_n1152_6seed.json`, the interleaved-split
> run cited by `2026-09-03-vision-configural-binding-crossing-is-mostly-capacity-anticheat-caught-it.md`).

- **capability_go (position-invariance-gated): 2/6** — the strict metric that ANDs held-out object decoding with
  `position_pooled_out` (held-out position must decode at or below chance+margin) falls from **5/6**
  (`verdict_fracs.capability_go = 0.8333`) on the original interleaved split to **2/6**
  (`verdict_fracs.capability_go = 0.3333`, seeds 42 and 101 only) here. This is the metric that actually
  certifies "learned a position-invariant crossing," and it is the one that degrades most.
- **beats the NO-GO floor: 3/6** — under the contiguous held-out-position block split (train on the first half of
  position indices, test on the unseen second half) the readout clears the floor on only 3 of 6 seeds
  (`verdict_fracs.beats_config_c_nogo = 0.5`), versus **6/6** (`verdict_fracs.beats_config_c_nogo = 1.0`) on the
  original interleaved split — the correct like-for-like pairing for this metric.
- **learning load-bearing: 6/6** — the learned signed-discriminant spiking-WTA readout beats its random-weight
  twin on every seed, on BOTH splits. The crossing is a genuine learned effect, not a fixed-projection artifact.
- **seed 102's leak (surfaced here for the first time):** seed 102 clears the floor (`LEARNED_spkwta_held` 0.4792
  > `config_c_nogo_floor` 0.34) but FAILS `position_pooled_out` — its held-out position decodes at **0.4167**,
  above the **0.40** <!--derived: chance_position (0.25) + pos_decode_margin (0.15), both direct
  config/output fields from the artifact--> invariance margin (per-seed `dissociation.position_decode_heldsplit`
  and `.position_pooled_out: false` in the cited 6-seed artifact). This is a genuine POSITION LEAK, not a floor
  miss: the readout clears the object-decoding bar partly by exploiting position information it should not have
  under genuine spatial extrapolation.

## Reading it honestly
Open Q1 asked whether the 5/6 crossing (`capability_go`, the position-invariance-gated metric) was real or an
ELM-overfit at high width. The answer is BETWEEN: not an overfit (learning is load-bearing 6/6 on both splits),
but not robust to genuine spatial extrapolation either — `capability_go` falls to **2/6** on held-out positions
(the plainer beats-the-floor metric falls to 3/6; see the corrected numbers above). The interleaved split
flattered the result — trained neighbours bracketed every test position, so it measured interpolation, not
extrapolation. The scramble-null passing on every seed (the learned readout falls to chance on pixel-scrambled
held images) confirms the readout uses real structure, not a shortcut — so the failure under held-out position is
a genuine generalization gap, not a scramble-check artifact. **One of the three floor-clearing seeds (102) gets
there via a position leak, not genuine invariance** — it decodes held-out position at 0.4167, above the 0.40
margin `position_pooled_out` requires — so of the six seeds, only 2 (not 3) actually demonstrate a
position-invariant crossing; the third floor-clearing seed passes the weaker bar while still leaking position
information. Read this way, **the crossing is WEAKER under spatial extrapolation than a floor-only 3/6 headline
suggests** — the qualitative conclusion of this finding (real but not position-robust) is unchanged and, once the
gated metric and the seed-102 leak are surfaced, STRENGTHENS.

## Next (no-defer: the residual, quantified)
The residual is held-out-POSITION generalization on **4/6 seeds** (43/44/100 miss the floor outright; 102 clears
the floor but LEAKS position, decoding held-out position at 0.4167 against a 0.40 margin) — only 2/6 (42/101) show
a genuine position-invariant crossing under both anti-cheats. Two concrete levers: characterize WHICH seeds fail
and whether it tracks a covariate (S2 bank overlap with the held positions, and specifically why 102 leaks
position while 42/101 do not), and test whether the configural-binding arm (`--conj-bind fixed`, which cleared
6/6 under the old anti-cheats) holds up better than the flat ELM under held-out-position — i.e. whether binding
buys genuine position-invariance the flat pool lacks. That companion arm was named in the runner's GO-gate but
not run here.
