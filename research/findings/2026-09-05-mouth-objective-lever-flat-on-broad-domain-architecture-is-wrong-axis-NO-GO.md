---
type: finding
status: negative
claim_check: measured
date: 2026-09-05
mechanism: predictive-coding auxiliary OBJECTIVE (`--pred-aux-weight 1.0 --pred-aux-offsets 2`) on the deployable linattn mouth — wt103 broad-domain s43 A/B vs the pred-aux-OFF baseline
lane: language (own-voice mouth / retire the Qwen scaffold)
seeds: [43]
runner: research/runners/_emerge_wkv_lm_derisk.py
artifacts:
  - research/findings/raw/_emerge_wkv_lm_linattn_wt103_predaux_s43.json
  - research/findings/raw/_emerge_wkv_lm_linattn_wt103_scale_s43.json
builds_on:
  - research/findings/2026-09-05-mouth-broad-domain-fluency-deep-research-ladder.md
  - research/findings/2026-09-01-generative-cortex-token-supply-lever-broad-domain-plateau-is-starvation-not-capacity-wall.md
verdict: >
  NO-GO / FLAT (single-seed A/B). The deep-research ladder's #1-ranked lever — the predictive-coding auxiliary
  objective, "the strongest same-budget external datapoint below ~20M tokens" — gives NO meaningful lift on the
  broad-domain wt103 at the deployable linattn mouth (s43): the deep-context (10-99) margin_vs_trigram moves within
  noise of the pred-aux-OFF baseline, and every depth bucket is within a few thousandths, mixed sign (exact figures
  in the marked body). This is the SECOND architecture lever to come back small on this exact broad-domain test
  (delta-rule = a small sub-bar lift; content-addressing = exhausted). Read against our OWN 6-seed GO
  (2026-09-01-...-plateau-is-STARVATION-not-capacity-wall: at capacity-matched scale, deep-context NLL drops
  MONOTONICALLY as token-supply rises and the margin GROWS with tokens), the pattern is now decisive: the
  broad-domain mouth wall is TOKEN/DATA-bound, and ARCHITECTURE levers (objective, delta-rule, capacity per that
  same finding, content-addressing) are the WRONG AXIS — they each move the margin by ~0.00-0.04 while crossing the
  trigram needs ~+0.3-0.57. Per THE LAW this is a method verdict, not a capability wall: the capability's real
  lever is Chinchilla-scale token supply (push tok/param from the current low regime toward ~20-10000), which is a
  DATA + COMPUTE scaling endeavor (a bigger/mixed corpus beyond wt103's ~fully-used ~2.1M sentences, and/or more
  compute) — a strategic fork surfaced to the owner. Nothing wired; the mouth default remains linattn.
---

# Mouth objective lever: FLAT on broad-domain — the wall is token/data-bound, architecture is the wrong axis

## What ran
`_emerge_wkv_lm_derisk.py --recurrence linattn --pred-aux-weight 1.0 --pred-aux-offsets 2` on the byte-identical
wt103 config as the pred-aux-OFF baseline (s43), harvesting `research/findings/raw/_emerge_wkv_lm_linattn_wt103_predaux_s43.json`.
The DR ladder's rung-1 decisive test (k=2, weight 1.0, per the strongest external small-scale evidence).

## Derived — margins vs trigram (s43; direct reads of the two cited artifacts)
<!--derived: objective from _emerge_wkv_lm_linattn_wt103_predaux_s43.json; baseline from _emerge_wkv_lm_linattn_wt103_scale_s43.json; lift is their difference -->
| depth | 1 | 2 | 3 | 4-5 | 6-9 | 10-99 |
|---|---|---|---|---|---|---|
| objective (+pred-aux 1.0, k=2) | 0.984 | -0.582 | -0.454 | -0.412 | -0.363 | -0.281 |
| baseline (linattn, pred-aux off) | 0.989 | -0.570 | -0.454 | -0.402 | -0.356 | -0.286 |
| lift | -0.005 | -0.012 | 0.000 | -0.010 | -0.007 | +0.005 |

Deep-bucket lift +0.005 (noise); no depth improves beyond a few thousandths, mixed sign. FLAT.

## Reading it (no-defer) — the decisive pattern
Three architecture levers now measured on this exact broad-domain test: the objective is FLAT, the delta-rule a
sub-bar lift, content-addressing exhausted (all negative; the figures are in the marked table above and their own findings). None approaches the ~+0.3-0.57 needed to cross the trigram
at depth. Meanwhile our OWN token-supply finding is a 6-seed GO that the plateau is DATA-STARVATION (more tokens →
monotonic deep-NLL drop, margin grows, beats the trigram — the opposite of the architecture levers). ⇒ the
broad-domain mouth wall is TOKEN/DATA-bound; architecture is the wrong axis. The capability's real lever is
Chinchilla-scale token supply (tok/param → ~20-10000), which needs a bigger/mixed corpus (wt103's ~2.1M sentences
appear ~fully used) and/or more compute — a data+compute scaling fork (surfaced to the owner; connects to the
hardware/2nd-GPU consideration). Architecture levers are banked as small/flat, not the axis.

## Honest scope
Single-seed (s43) A/B — labeled direction-test, not a 6-seed claim. Additive; no production change. The delta-rule's
small sub-bar lift may still stack as a minor component once the DATA axis is moved, but it is not the lever.
