# WORKLOG — Stage-1 C1 category inheritance (branch `research/stage1-c1-category`)

Append-only. One line per step, newest last. The local model appends here; Claude reads this at inspection.

- 2026-09-30 09:45 Claude: worktree from main (f5e3732fe); `research/stage1-exit-gate` merged in (its C1 group and
  scorer are imported, unchanged). Record read: `before_you_build.sh` (top hits rung-3c, EMERGE-26, EMERGE-30) + RAG;
  the three findings and the rung-3c / EMERGE-26 runners read in full. Choice: rung-3c's learned member->context
  pathway with its hand-placed context->property projection replaced by a learned frame->frame association
  (prereg section 2). EMERGE-26's superordinate is a host dict literal (`SUPER = {...}`), unusable for `cat_disc`.
- 2026-09-30 10:00 Claude: organ-level dev sweeps (dev seeds 1-8, no gate seed). Bipartite member<->frame (IAC) variant
  with identical neurons: 3-hop read rates non-monotone in the learned-weight bound (mode-locking of identical
  assemblies; e.g. worm 0.000 at bound 40, 0.050 at 60, 0.000 at 90); OU 300 pA: every link grew at encoding
  (runaway). Chosen: member->frame + learned frame->frame, seeded per-neuron heterogeneity, bound 40: dev seeds
  1-8 canary `eat worm` 0.10-0.12 / `eat bug` 0.000; permuted `bug` 0.09-0.13 / `worm` 0.000; read lesion 0.000;
  lateral_learn 0.000; lateral read lesion keeps `worm` (~0.086, the link is written at encoding).
- 2026-09-30 10:25 Claude: code committed (f394633b3, 437a914f3, c081ff8a9): organ, wiring (default off), runner,
  tests (14 passed). Dev seed 7 real-handler arms queued locally behind `tools/mem_ok.sh 9` (the workstation was
  memory-contended by sibling dev runs).
- 2026-09-30 11:26 Claude: dev seed 7 `intact_a` (real handler) landed, 1142.8 s: every teach turn observed; `what does
  the canary eat` -> "perhaps the canary eats the worm [a guess from what I've learned -- not something I was
  taught]" (hypothesis_svo, recalled_svo null); organ worm 0.1192 / bug 0.0; identity controls and moat hold.
- 2026-09-30 11:48 Claude: dev seed 7 `lesion_inherit` landed, 1199.1 s: the direct probe abstains (lesion `all`,
  `lesion_held` true); the primary reply fields differ from intact on `p_canary_eat` only.
- 2026-09-30 12:08 Claude: organ dev artifact (dev seeds 1-8) + the two real-handler shards committed (a98a398d3);
  prereg committed alone (e3ef82263).
- 2026-09-30 12:09 Claude: dev seed 7 `permuted` landed, 1327.4 s: `what does the canary eat` -> "perhaps the canary
  eats the bug" (organ bug 0.1042 / worm 0.0); `what does the canary swim` recalls lake (taught in this arm).
  The remaining dev arms (`off`, `intact_b`, the three informational ablations) run behind `mem_ok.sh 9`.
