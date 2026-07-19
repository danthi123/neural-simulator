# Coupling scale probe (V=300, tr=3000, 2-seed): the ROBUST claims strengthen with scale, but the deep-TAIL stays bigram-limited at tractable scale — the honest scope of the mission-central coupling

**Date:** 2026-07-13
**Runner:** `research/runners/_reslm_couple_selssm_into_eprop_generator_derisk.py --tr-cap 3000 --ev-cap 600 --vocab 300 --n-sent 8000` · raw `research/findings/raw/_couplessm_scale/`. numpy; NO `sim/` edit.
**Status:** Confirms the by-depth nuance the adversarial-verify flagged, at a larger scale. The core GO is robust; the "clears the bigram floor at the deep tail" reading does NOT hold at tractable scale.

## Why

The 6-seed coupling GO (`-COUPLE-selssm-into-eprop-generator-GO-5of6.md`) beat the bigram as a d≥4 AGGREGATE but, per-depth, only at d=4-5 (the deep tail d≥6 was bigram-level). The Rung-3 scale trajectory showed the selective SSM's advantage GROWS with vocabulary — so does scaling data (tr 1200→3000) + vocab (V 200→300) push the deep-TAIL advantage past the bigram, or does the deep tail stay n-gram-limited (the CEILING / reservoir-scale regime)?

## Result — deep-aggregate GO holds; deep-TAIL stays bigram-limited (2 seeds, V=300, tr=3000)

Deep aggregate (d≥4): seed 42 sel_gain +0.672 (rand +0.524, fix +0.346) GO; seed 43 sel_gain +0.739 (rand +0.456, fix +0.293) GO — the ROBUST claims (sel>eprop, sel>fix ~2×) hold and if anything strengthen vs V=200.

By-depth (sel vs bigram; sel vs rand):

| depth | s42 sel−bigram | s42 sel<rand | s43 sel−bigram | s43 sel<rand |
|---|---|---|---|---|
| 4-5 | **−0.102** | +0.289 | **−0.225** | +0.355 |
| 6-9 | +0.137 | +0.099 | +0.009 | +0.256 |
| 10-99 | +0.144 | −0.017 | +0.106 | +0.202 |

- **eprop (reservoir-only) is huge at all depths** (4.05–4.51) and the coupled `sel` (3.5–3.8) beats it everywhere → the coupling ROBUSTLY improves the emergent generator at every depth (the mission-central, load-bearing result).
- **vs the bigram**: `sel` clears the floor at d=4-5 (−0.10/−0.23), ties at d=6-9, and is ABOVE (worse than) the bigram at the deepest tail d≥10 (+0.14/+0.11) — scaling V 200→300 + data 1200→3000 did NOT flip the deep tail below the n-gram floor.
- **`sel>rand` (selective-specific)** is strong at d=4-5 (+0.29/+0.36), present at d=6-9 (+0.10/+0.26), marginal at the deepest tail (−0.02/+0.20).

## ⇒ honest scope of the coupling (settled across two scales)

1. **ROBUST + mission-central:** the learned input-driven selective channel carries deep context the (e-prop-trained) fading reservoir loses — `sel>eprop` at every depth with large margins, `sel>fix` ~2× (the input-DRIVEN selectivity, not extra slow memory). This is the coupling's value and it strengthens with scale.
2. **Selective-specific (`sel>rand`)** is genuine at d=4-9 and scale-dependent at the deepest tail (the Sub-claim-B-analogue).
3. **NOT true at tractable scale:** the coupled generator does not clear the n-gram (bigram) floor at the deep tail (d≥10) — that is the CEILING / reservoir-scale regime (`2026-07-12-reslm-batched-scale-CONFOUND-FREE-...`, `2026-07-11-CEILING-...`: the long-range signal is thin until 50–200× more scale, where even a transformer/LSTM only reaches the +1.5 growing-with-depth). The coupling's honest claim is "carries deep context BETTER than the fading reservoir," NOT "beats the n-gram floor at the deep tail."

## ⚠️ a-1 REFRAME (2026-07-13) — the deep-tail-bigram-limit is the DOCUMENTED NULL-DISCRIMINATOR regime, NOT a selective-SSM deficiency

An a-1 knowledge-base check (before a planned tractable-scale joint sweep) surfaced `2026-07-13-SSM-language-escalation-toy-scale-NULL-DISCRIMINATOR-...` + `2026-07-11-CEILING-...`: at toy scale (V≤1000, ≤5M words) **even a well-trained full TRANSFORMER is WORSE than a bigram at deep context** (−0.06→−0.38, monotonically) — the long-range predictive signal is genuinely THIN at this scale for ANY model. ⇒ my "deep tail stays bigram-limited at V=200–300" is EXACTLY this documented null-discriminator regime: a tractable-scale test **cannot** discriminate whether the selective coupling reaches fluent long-range, so a negative/marginal there is scale-confounded, not a mechanism verdict. This CORRECTLY scopes the deep-tail caveat: the coupling is NOT deficient at the deep tail — *nothing* clears the n-gram floor there at this scale. The decisive test is the named **validated-signal regime (TinyStories 23.7M words / V=2000)**, where a transformer + full-backprop LSTM capture +0.5→+1.9 nats growing-with-depth. The coupling is already the mechanistically-right combination the null-discriminator finding points to (a MIXING reservoir for local n-gram conjunctions + a SELECTIVE DIAGONAL channel for non-fading distal holding); the fluency demonstration is a validated-scale GPU run (a scale lever, low-med VRAM allowed), NOT a tractable-CPU sweep. The a-1 check prevented running a null-discriminator sweep.

## Next
- The decisive fluent-long-range test is the VALIDATED-SCALE regime (23.7M/V=2000, GPU), per the null-discriminator finding — NOT more tractable-CPU sweeps.
- raw `research/findings/raw/_couplessm_scale/seed{42,43}_V300.json`.
