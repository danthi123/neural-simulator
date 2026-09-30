# WORKLOG — Stage-1 exit gate (branch `research/stage1-exit-gate`)

Append-only. One line per step, newest last. The local model appends here; Claude reads this at inspection.

- 2026-09-29 20:50 Claude: worktree created from main (fac3f699c); record checked (`before_you_build.sh` +
  four RAG queries, one per criterion); runner, tests, prereg and HANDOFF drafted.
- 2026-09-29 21:15 Claude: dev smoke launched on dev seed 7 (word intact_a + lesion_recruit, val_heldout 2 words).
  Finding: a git worktree has no `data/corpus/` (git-excluded), so the one-brain cross-edge pool and the learned
  animacy lexicon degraded silently in the first word arms -> `corpus_present` recorded per arm and required by
  the aggregate's preconditions; the worktree's `data` linked to the primary checkout; word arms re-run.
- 2026-09-29 21:20 Claude: val_heldout 2-word smoke PASS list clean (build 153 s, first content turn 124 s,
  exposure ~8 s, feel ~2 s; sources tiny-demo +LTM; hash ok).
- 2026-09-29 21:45 Claude: word/lesion_recruit (dev seed 7, corpus present) landed: 811 s wall (build+first turn 183 s,
  teach ~23 s, probe ~12 s, two lazy-organ turns 365 s and 168 s); patch called 21x; every new-word probe abstains
  or mis-resolves (`what does it eat` -> the built-in cat/fish), `what does the dog chase` -> cat survives; source
  tiny-demo +LTM; hash ok. The lesion behaves as the prereg predicts. cat_disc/intact_a and val_novel/intact_a launched.
- 2026-09-29 21:55 Claude: word/intact_a (dev seed 7) landed: 886 s wall; bear->otter, otter->clam, both moats,
  pre-exposure abstain and dog->cat all as required; `what does it eat` resolved `it` to the built-in cat (the D6
  referent store is a build-time vocabulary). DESIGN DECISION before any gate seed: that sub-probe is REPORTED,
  never counted (`anaphora_it_eat`); prereg 4.2 says why. Lesion contrast on the two counted probes: differs on
  both, unchanged on the built-in fact and the moats.
- 2026-09-29 22:20 Claude: cat_disc/intact_a (dev seed 7) landed: 1045 s wall; eleven teach turns stored; both
  inheritance probes abstain (direct: recall path exercised, no block; generative: the open-ended branch fired and
  proposed nothing); identity controls and moat correct. The predicted C1 FAIL shape, instrument intact.
- 2026-09-29 22:30 Claude: lived/intact_a (dev seed 7) landed: 2813 s wall (the forced numpy BTSP write costs
  1075 s on the teach turn and 500-700 s on each later verified recall); every C3 check as required (recall after
  three distractors, the dAP disclosure with in_memory true, both pre-exposure abstains, the moat, dog->cat).
  Wart: "tell me about the sky" was ACQUIRED as [tell, about, sky]; distractors changed to questions before the
  prereg commit; FAILURE_LOG row added. ETA revised in the prereg (section 10) and the HANDOFF: ~95 job-hours.
- 2026-09-29 22:33 Claude: commit 1 = the prereg alone: 2a7097ccc.
- 2026-09-29 22:45 Claude: commit 2 was blocked by gates/attribution_required (the runner measured lesion vs
  intact arms and never subtracted the null). Fixed the cause: the aggregate now calls tools.lab.attributable_to
  per seed and lesion (treatment = primary-field differences intact_a-vs-lesion on the scored turns; control = the
  same count intact_a-vs-intact_b; C4 uses |r_intact - r_lesion| vs |r_a - r_b|), recorded as `attributable` in
  each lesion record; the aggregate's prints go to stderr so `--aggregate > report.md` stays clean.
