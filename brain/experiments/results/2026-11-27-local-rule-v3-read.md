# Local rule v3 (two-phase layerwise targets, no backward path) read (11-27): NEGATIVE-DEV at 0.1B and 0.4B

Registered prereg `brain/research/2026-11-27-local-rule-v3-prereg.md` (design: Fable, 2026-11-27-local-rule-v3-DESIGN.md);
run unattended by tools/engine on pool40/41/42 (smoke 0.1B 2.9 GB, 0.4B 6.7 GB), commit 6ff41baa. G0 passed 12/12 (the
cortex reproduces every answer with the telling in context). No dev cell met the pick rule; six seeds not run.

Seed-7 dev (9 cells, lr x keep): the rule is engaged and fits its target: summed per-layer value error falls 459 -> 115-213
(0.1B) and 5037 -> 964-2062 (0.4B). Recall stays 0-2 of 12 (0.1B best lr 0.3 keep 1: recall 2, generic KL 0.039 under the
0.040 cap but held-out drift 0.20 over its cap; 0.4B 0-1). Gradient reference: 3/12 (0.1B), 5/12 (0.4B). Clip binds on
9-31% of updates.
Reading: matching the clamped "telling in context" activity on the telling text stores the sentence's activity but does not
transfer to answering the question (the design's named top risk, copy-mode targets). With v1 (learns, drifts), SHY, v2 (no
drift, no learning) and v3 (fits its target, no recall), four local-rule methods have now failed at 0.1B/0.4B: the 12-31
milestone (sleep by a local rule) is at risk. Next methods named by the record: question-form targets (the design's
follow-up: clamp the cortex answering the question with the telling in context, target the answer-position activity),
or a declared research round on biological credit assignment before any fifth method. Verdict on these methods only.
