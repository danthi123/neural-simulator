---
type: finding
status: contributing
date: 2026-08-21
mechanism: recovered-pool-lane-de-risks
lane: emergence
seeds: [42, 43, 44, 100, 101, 102]
seed-waiver: these are RECOVERED 6-seed pool runs (banked from the nodes; not re-run this session) — the evidence is
  each runner's own 6-seed aggregate verdict + its input-destruction controls, quoted from the artifacts.
instrument: three headless 6-seed pool de-risks (mini-PC pool) whose results were stranded on the nodes by the
  result sync-back gap (board #56) and recovered here.
runner: research/runners/_emerge62_discover_function_words_derisk.py · research/runners/_b1_v1_selforg_rf_derisk.py ·
  research/runners/_laneB_curiosity_learning_progress_slope_derisk.py
artifacts:
  - research/findings/raw/_emerge62_discover_function_words.json
  - research/findings/raw/lanes/v1_selforg_rf_6seed.json
  - research/findings/raw/lanes/curiosity_lp_slope_derisk.json
---
# Recovered overnight pool de-risks — the function-word inventory + V1 receptive fields SELF-ORGANIZE from experience (2× 6-seed GO), + a curiosity learning-progress signal (banked)

Three 6-seed de-risks ran overnight on the mini-PC pool and their results were STRANDED on the nodes (the #56
sync-back gap); recovered + banked here so the compute is not lost. Two are clean emergence GOs (structure from
experience, no host design — the standing emergence bar), one is a strong per-seed curiosity signal.

## (1) Function-word inventory self-organizes — GO (6 seeds)
<!--derived-->
Artifact: research/findings/raw/_emerge62_discover_function_words.json (go: True). Over a controlled SVO+function-word
stream (content AND function words, NO label as input), a FIXED/pre-registered discovery rule (running freq-pct >= 0.9
AND context-coverage-pct >= 0.6 — the Goldilocks distributional signature, Yang-Getz / Redington / Dominey-Hinaut)
recovers the hand ground-truth CLOSED class at **F1 0.863** (P 0.760 R 1.000): all frame function words
['can','does','not','the'] recovered (frame-recall 1.00), content words correctly excluded. The DISCOVERED set feeds
the EMERGE-59 spiking-Broca frames (held-out render-ok 1.00, no-confab moat intact). Every input-destruction control
COLLAPSES: FREQUENCY-SHUFFLE F1 0.079 (margin >= 0.3), NO-STREAM → empty. Held-out generalizes (a withheld function
word 'does' still CLOSED, a withheld content word 'trout' OPEN, by their own stats vs frozen thresholds — not
memorized). **⇒ the open/closed distinction + the function-word inventory EMERGE from distributional experience — the
last host-designed closed-class lexical residual of the spiking-Broca producer is removed.** Honest scope: the BOUNDED
EMERGE frame domain (not open-ended generation); per-frame slot-order (EMERGE-63) + slot-inventory (EMERGE-64) are the
ranked follow-ons. NO sim/ edit; moat untouched.

## (2) V1 oriented receptive fields self-organize — GO (6 seeds)
<!--derived-->
Artifact: research/findings/raw/lanes/v1_selforg_rf_6seed.json (overall_verdict GO, 6 seeds). Both mechanisms (A, B)
pass on all seeds with GEOMETRY PRESERVED (the downstream load-bearing output) 6/6, and the discriminating control —
RF orientation tuning (OSI) — collapses to UNORIENTED on all seeds under the controls (chance orient-decode 0.125).
Learned orientation margin_mean 0.696 (mechanism A) vs the host reference 0.763 (orient-decode 0.977). ⇒ oriented
(Gabor-like) receptive fields self-organize from the input statistics rather than being hand-set — the perceptual
front end develops rather than being designed.

## (3) Curiosity learning-progress slope — strong per-seed signal (banked, no formal aggregate)
<!--derived-->
Artifact: research/findings/raw/lanes/curiosity_lp_slope_derisk.json. Per-seed (e.g. seed 42): the learning-progress
slope tracks the knowledge gap (corr_gap_mod 0.9955), the brain ASKS more on the UNKNOWN than the known
(rate_unknown 0.0366 vs rate_known 0.0; total_asks 117), and confidence RISES with mastery (conf_rise 0.539) while a
noisy/unlearnable topic is vetoed (noisy_late_rate 0.0). The runner banked per-seed results without a top-level
aggregate verdict, so this is recorded as a SIGNAL (not a formal GO); a follow-on should compute the 6-seed aggregate.

## Provenance + scope
Recovered from pool40/41/42 (`~/derisk-pool/sim`) after the #56 sync-back gap left them un-synced; the .prov.json
sidecars travel with each artifact. Not re-run this session — the verdicts are the runners' own. These are emergence
de-risks (structure-from-experience), default-off research results — not production-wired.
