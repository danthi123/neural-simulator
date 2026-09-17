---
type: finding
status: verified
date: 2026-09-16
mechanism: DELETE the host content_bias_target lexicon (ANIMACY/VERB_SELECTS) from the biased-competition production
  path — the learned spiking feature-compatibility map (SpikingFeatureCompat) is the SOLE content-bias source
integration_faculty: selective-attention-biased-competition
lane: language + focus (scaffold-retirement, owner's #1 metric)
seeds: [42, 43, 44, 100, 101, 102]
verdict: >
  RETIRED (scaffold_retired 4->5). The host animacy/verb lexicon that scored WHICH held referent receives the WTA
  content bias is DELETED from research/runners/biased_competition_buffer.py; MultiTurnAgent._resolve_biased no
  longer has a host-lexicon fallback branch (learned SpikingFeatureCompat first, then the D3 discourse-center focus
  source, else abstain). The ground-truth lexicon copies used to TRAIN/EVAL the learned chooser move to
  _gap3_learned_feature_compat_derisk.py (never consulted at resolution time). This lands ON TOP of the same session's
  fs_to_sel 5.0->7.0 graded-bias regression fix. Verified: the bc branch-vs-main /api/brain-chat DEFAULT differential
  is BYTE-IDENTICAL on both biased-competition probes (bc_a, bc_b) -- the ONLY faculty the delete+fs7 can touch
  (code-isolated: zero other imports of content_bias_target; fs_to_sel only weights the sel-pool cross-inhibition),
  and even in the <40-fact tiny-demo case the D3 focus source answer-preservingly covers what the host lexicon used
  to. All affected CI green (graded 5/5, production 5/5, byte-identity 3/3, learned-map 7/7).
runner: research/runners/onebrain_regression_battery.py (differential: branch-default vs main-default, bc probes)
artifacts:
  - research/findings/raw/_biased_competition_retire_verify/differential_result.json
external: NO-EXTERNAL-NEEDED — a host-scaffold DELETE confirmed answer-preserving; the WTA + lateral inhibition it
  biases was already brain-based (only the host content SCORING was retired).
builds_on:
  - research/findings/2026-09-16-graded-bias-regression-FIXED-designed-cross-inhibition-restored-fs7.md
  - research/findings/2026-09-16-wirein-flips-biased-competition-gnw-stop-conflict-scaled-DEFAULT-ON-GO.md
---

# Biased-competition content_bias_target host lexicon RETIRED — learned spiking map is the sole content-bias source

The 2026-09-16 wire-in flipped the learned SpikingFeatureCompat map default-ON as the production content-bias source,
moving this row to RETIRABLE_NOW (the host content_bias_target lexicon reached only in the <40-heard-fact fallback,
never in the >=40-fact production KB). This finding DELETES the host lexicon -> RETIRED (scaffold_retired 4->5, the
owner's #1 metric). It is the LAST RETIRABLE_NOW row; the remaining BLOCKED rows are 49 on the own-voice mouth
(neural-render), 6 on self-model-reward-residual, 2 on gnw-thought-swap.

## What was deleted
- `content_bias_target` + the `ANIMACY` / `VERB_SELECTS` feature lexicons: GONE from
  `research/runners/biased_competition_buffer.py`. The win was already brain-based (the spiking WTA competition +
  lateral inhibition + the recurrence amplifying the small content asymmetry); only the content SCORING was host.
- `MultiTurnAgent._resolve_biased`: the `else: fav = content_bias_target(...)` host-fallback branch is GONE. Order is
  now the learned `_feat_compat_source` first, then the D3 `_focus_bias_source` (Centering Cb), else abstain (moat).
- The ground-truth lexicon copies (needed to TRAIN/EVAL the learned chooser) moved to
  `_gap3_learned_feature_compat_derisk.py`, never consulted at resolution time.

## Verify — an integrated differential, branch-default vs main-default (all CPU/numpy, local; GPU untouched)

(values from `research/findings/raw/_biased_competition_retire_verify/differential_result.json`.)

Ran the biased-competition /api/brain-chat probes (bc_a "the cat and the ball walked in" -> bc_b "what does it eat")
through the REAL `webapp.server.brain_chat` on the merged branch (delete + fs7) and on main (fs7, no delete), default
env. Both probes' DEFAULT answers are **byte-identical** (exact answer-string equality, `byte_identical: true` recorded
in the artifact from a direct `branch == main` compare per probe):

- bc_a: branch == main == `the cat balls the walked — worth going further here.`
- bc_b: branch == main == `the cat eats the fish — worth going further here.`

The delete + fs7 are **code-isolated to the biased-competition faculty** — no other faculty imports
`content_bias_target`, and `fs_to_sel_weight` only weights the sel-pool cross-inhibition — so bc_a/bc_b are the ONLY
probes that can change; the other 24 battery probes are unchanged by construction. On main bc_b resolves via the host
lexicon; on the branch it resolves via the D3 focus source to the SAME answer, so the delete is answer-preserving
even in the <40-fact tiny-demo case (the host lexicon was not load-bearing there — the D3 focus source covers it).

Unit CI on the merged branch (CPU/numpy): `test_multireferent_graded_bias_agent` 5/5, `test_multireferent_biased_
competition` 5/5, `test_multi_turn_agent` 3/3 (byte-identity), `test_gap3_spiking_feature_compat` 7/7.

## Honesty
The learned map installs at >=40 heard facts (production KB); below that the resolution falls to the D3 focus source
if wired, else honestly abstains (moat-safe) — a brain-based path or an honest abstain, never a host lookup. The host
`content_bias_target` was a host-cognition scaffold (a hand-coded animacy x verb-selection lookup); deleting it is the
scaffold-retirement win. The companion fs_to_sel 5.0->7.0 fix carries a documented 3-referent scaling residual (see
`2026-09-16-graded-bias-regression-FIXED-designed-cross-inhibition-restored-fs7.md`). scaffold_retired: 4 -> 5.

## Sources (biology grounding of the WTA the biased-competition biases; the delete removes only the host SCORING)
<!--derived-->
- Wong & Wang 2006, *A recurrent network mechanism of time integration in perceptual decisions*, J Neurosci
  26(4):1314-1328, doi:10.1523/JNEUROSCI.3733-05.2006 — https://pubmed.ncbi.nlm.nih.gov/16436619/ (the recurrent
  NMDA attractor WTA the per-referent `sel_X` accumulator pools implement).
- Rutishauser & Douglas 2009 (selective feedforward-inhibition WTA motif — the `sel_FS_X -> sel_Y!=X` cross-inhibition
  the companion fs_to_sel 5.0->7.0 fix strengthened); Desimone & Duncan 1995 (biased competition).

