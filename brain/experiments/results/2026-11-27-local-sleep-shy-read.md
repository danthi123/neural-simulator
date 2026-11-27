# Local sleep + synaptic downscaling (SHY), 0.4B six seeds (11-27): NEGATIVE by the registered rule

Prereg `brain/research/2026-11-27-local-sleep-shy-prereg-DRAFT.md` (registered after pre-data review); dev pick
`results/2026-11-27-local-sleep/shy_dev_pick.md` (transport lr 0.3, g* 0.6). Six seeds on pool41 (42-44) and pool42
(100-102), CPU fp32, commit 52b846adb; verdict by `tools/engine/specs/local_sleep_shy.verdict` (`shy_six_04b/verdict.json`).

Means over six seeds: gradient@1 recall 2.33 (dev seed 7 had 5), gradient@0.6 1.67, local@0.6 1.17, shuffled@0.6 0.50,
no_sleep 0. Clauses failed: (a) recall bar 2, (b) local beats shuffled on >= 5/6 (got fewer), (c) and (e) drift. (d) and (f)
held. NEGATIVE.

Report only: the local rule diverged on 2 of 12 local-rule nights (s101 local KL 14-18; s44 shuffled KL 4-11; gradient
never did), which dominates the mean drift. On the other seeds local@0.6 drift was 0.026-0.037 against gradient@1
0.022-0.034, so the cut does bring drift near the gradient arm, but recall is below the bar on 5/6 seeds regardless
(per seed local@0.6: 1, 1, 3, 0, 0, 2). Instability at lr 0.3 is a property of this rule (no per-step norm control on the
factor product, see the 0.1B read). Verdict on this method (local transport rule + post-night downscale), not on local
learning. The next method (local rule v2: layerwise anchoring) is built and in pre-data review.
