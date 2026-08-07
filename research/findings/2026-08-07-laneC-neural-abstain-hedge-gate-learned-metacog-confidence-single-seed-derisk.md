---
type: finding
status: contributing
date: 2026-08-07
mechanism: laneC-neural-abstain-hedge-gate-from-learned-metacog-confidence
runner: research/runners/_laneC_self_schema_neural_abstain_gate_derisk.py
seed-waiver: single-seed (42) FULL-SIZE de-risk only; the 6-seed 42/43/44/100/101/102 generalization is PENDING (command below). No generalization is claimed here.
artifacts:
  - research/findings/raw/lanes/metacog/laneC_neural_abstain_gate_s42_full.json
  - research/findings/raw/lanes/metacog/laneC_neural_abstain_gate_s42_full.json.prov.json
  - research/findings/raw/lanes/metacog/metacog_learned_acc_dynamic_6seed.json
---

# lane C honesty-as-behavior: a NEURAL abstain/hedge gate from the learned metacognitive-confidence pool — single-seed de-risk GO, 6-seed pending

<!--derived-->
**One-line result (single seed 42, full-size; NOT a 6-seed generalization).** The GO learned ACC/aPFC
metacognitive-confidence pool's SPIKES now gate an assert (output/motor) pool through a tonic inhibitory abstain
veto, so the brain withholds an answer its own monitor flags as unreliable. On a familiar-but-wrong battery the
neural gate raised selective accuracy from 0.800 (no gate) to 0.952 at 69% coverage (+0.152), beat a first-order
winner-magnitude gate at matched coverage (0.952 vs 0.778, +0.174; risk-coverage AUC 0.924 vs 0.831), hedged
confident errors far more than confident-correct items (0.714 vs 0.317, gap 0.397, base rate 0.308) where the
first-order gate hedged confident errors 0.000, collapsed to a 0.000 contingency gap (100% attributable) when the
self_schema->assert synapse was cut, and kept the moat (novel/unknown items hedged 0.650 vs 0.317 for
confident-correct; novel-assert reduced to 0.35). This is a functional honesty BEHAVIOR, not a claim of subjective
experience.

## Re-anchor: what was already built vs the genuine un-built step

<!--derived-->
The GO 6/6 dynamic ACC/aPFC learned monitor (`_second_order_metacog_monitor_derisk --confidence-read learned_acc
--learned-feature-mode dynamic`; type2_auc 0.831, meta_d 2.431;
`research/findings/raw/lanes/metacog/metacog_learned_acc_dynamic_6seed.json`) predicts first-order correctness, and
a fixed spiking relay (`_laneC_self_schema_metacog_integration_derisk`) already reads it into a `self_schema`
confidence pool. Both STOP at "the confidence rate separates correct from error" — the relay runner states it
"does NOT change production abstain/hedge behavior yet". The prior production wire-ins
(`_laneC_self_schema_honesty_wirein_derisk`, PARTIAL then the `source_consistency_floor` / `neural_source_consistency`
GOs) routed a SOURCE-MONITORING confidence (RF source echo / metadata), a different signal, into the hedge. The
genuine un-built step was to route the GO LEARNED-monitor confidence into an actual abstain/hedge BEHAVIOR. That
is what this runner does.

## Mechanism (brain-based)

One `SimulationBridge`, six regions: the GNW workspace 2AFC competition (+ shared inhibition) → slow-NMDA
`meta_schema` (driven by the learned confidence current, as in the GO relay) → `self_schema` confidence pool →
the NEW gate: `self_schema` EXCITES an `assert` (output/motor) pool, and a tonically-driven inhibitory
`abstain_fs` pool VETOES it. The assert pool fires only when the confidence excitation overcomes the caution veto.
The abstain DECISION is whether the assert pool fires — a motor read-out (body-legit, like reading which motor
pool fired), not a host threshold on the confidence scalar. Coverage is traced by the motor-readout threshold on
the assert pool's graded rate (standard risk-coverage), with the veto held at the confidence-graded operating
point (120 pA, from a response-surface probe). The inherited, documented scaffold is the learned monitor's host
logistic that renders confidence as a `meta_schema` drive current — the SAME step the GO relay already uses; the
abstain/hedge gate itself is on-substrate.

## Result (single seed 42, full-size — command below)

<!--derived-->

| metric | learned gate | baseline | anti-cheat |
|---|---:|---:|---|
| type1 accuracy (in-window) | 0.800 | — | task has genuine errors |
| self_schema confidence type2 AUC | 0.860 | — | routed monitor at validated strength |
| selective accuracy @ coverage 0.692 | 0.952 | 0.800 (no gate) | +0.152 |
| selective accuracy @ matched coverage | 0.952 | 0.778 (first-order) | +0.174 (d: beats first-order) |
| risk-coverage AUC | 0.924 | 0.831 (first-order) | +0.093 (d) |
| hedge: confident-correct / base rate | 0.317 / 0.308 | — | a: not preferentially hedged |
| hedge: familiar-but-wrong (confident errors) | 0.714 | 0.000 (first-order) | a/d: errors hedged, first-order can't |
| contingency gap (fw − hcc) | 0.397 | 0.000 (self→assert lesion) | b: 100% attributable, neural |
| moat: novel/unknown hedge vs confident-correct | 0.650 vs 0.317 | — | c: unknowns hedged more |
| moat: novel-assert rate (learned) | 0.35 | — | c: reduced from all-assert |

All four anti-cheats hold on this seed: (a) contingent — confident-correct hedged at the base rate (0.317 vs
0.308) while confident errors are hedged at 0.714; (b) neural — cutting the `self_schema->assert` synapse
collapses the contingency gap to 0.000 (100% attributable via `tools.lab.attributable_to`); (c) moat-safe —
novel/unknown items are hedged more than confident-correct (0.650 vs 0.317) and novel-assert drops to 0.35; (d)
beats first-order — the winner-magnitude gate hedges confident errors 0.000 and its risk-coverage AUC is lower.

## Honest boundaries

<!--derived-->
1. **Single seed only.** This is a full-size single-seed (42) de-risk, not a generalization. Seed 42 at the
   REDUCED smoke size (n_main 40) was small-sample-lucky (n_familiar_wrong 2); at n_main 72 with an undersized
   calibration block (few errors) the confidence signal weakened (self type2 AUC 0.573) and the contingency
   inverted. The result only stabilizes at the validated full calibration (96 trials) + full battery. The 6-seed
   run is required before any generalization.
2. **Pure novelty is not where the learned monitor wins.** On zero-signal novel items the raw winner-magnitude
   is itself informative (low signal → low magnitude → abstain), so the learned gate's novel-assert (0.35) is
   marginally ABOVE the first-order gate's (0.30). The learned monitor's advantage is specifically the
   familiar-but-wrong confident errors, not pure novelty detection. Recorded as a non-gating metric
   (`all_learned_novel_le_first_order`).
3. **Isolated 2AFC instrument.** "Novel/unknown" here is a zero-signal trial where the workspace is still forced
   to pick a winner; there is no retrieval-None "I see nothing" state. The conversation-pipeline wire-in (where
   the hard confabulation moat guards a real unknown) is the next step — feed this learned-monitor confidence
   through the existing default-off `meta_schema -> self_schema` honesty hook in `BrainConversationalAgent`.

## Reproduce (single seed) and the 6-seed validation (PENDING — for the parent)

Single seed (this artifact):

```bash
SIM_BACKEND=numpy PYTHONPATH=$PWD .venv/bin/python -u -m research.runners._laneC_self_schema_neural_abstain_gate_derisk \
  --seeds 42 --json research/findings/raw/lanes/metacog/laneC_neural_abstain_gate_s42_full.json
```

6-seed generalization (NOT run here; ~12 min CPU) — same runner and flags as above but with
`--seeds 42 43 44 100 101 102`, writing `--json` to a NEW output basename `laneC_neural_abstain_gate_6seed`
(dot-json) under the same `lanes/metacog/` raw directory:

```bash
OUT6="research/findings/raw/lanes/metacog/laneC_neural_abstain_gate_6seed"
SIM_BACKEND=numpy PYTHONPATH=$PWD .venv/bin/python -u -m research.runners._laneC_self_schema_neural_abstain_gate_derisk \
  --seeds 42 43 44 100 101 102 --json "$OUT6.json"
```

Watch on the 6-seed: (i) type1 accuracy must stay in [0.60, 0.90] on every seed (seed 42 sits at 0.80 — a seed
that drifts above 0.90 would fall out of window); (ii) the confident-error sample (n_familiar_wrong) per seed is
small (~7 at full size), so the contingency gap will be noisier than the risk-coverage AUC; the AUC and the
selective-accuracy gain vs first-order are the more stable headline metrics.

## Next mechanism

1. Run the 6-seed; promote to a headline honesty-behavior GO only if it holds.
2. Wire the learned-monitor confidence through the production default-off honesty hook (replacing/augmenting the
   source-consistency confidence with the learned correctness estimate), then re-run the hard-moat stressed
   battery to confirm 475/475 hard-moat abstains are preserved.
3. Burn down the inherited scaffold: the learned monitor's host logistic → a plastic ACC/aPFC error-monitor
   learned on-substrate.
